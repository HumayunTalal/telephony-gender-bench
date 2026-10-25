"""Combining several models, per clip and per call.

Per clip: each model gives (label, p_female). We vote. A model that
abstained (label == 'noise') does not vote. If nobody votes the clip is
'noise'. Ties go to 'unknown' rather than to a coin flip; downstream a call
with an unknown first-two-turns label is treated as not classifiable.

Per call: the customer side of a call is several short turn clips. Two labels
are produced.

    label_1  from the first two turns merged
    label_2  from every turn merged

For a two-turn call they are the same thing. When they differ, the call goes
on the QC list. The usual reasons are the agent's voice leaking into the
customer channel on a later turn, or a second person picking up the phone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter

import numpy as np
import pandas as pd

VOICED = ("male", "female")


@dataclass
class Vote:
    model: str
    label: str            # male | female | noise
    p_female: float       # 0..1 ; nan if abstained
    weight: float = 1.0


def vote(votes: list[Vote], min_margin: float = 0.0) -> tuple[str, float]:
    """Return (label, confidence). Confidence is the weighted share of the
    winning side among voiced votes. `min_margin` lets you require e.g. 0.2
    between male and female share before committing."""
    voiced = [v for v in votes if v.label in VOICED]
    if not voiced:
        return "noise", 0.0
    share = Counter()
    for v in voiced:
        share[v.label] += v.weight
    total = sum(share.values())
    m, f = share["male"] / total, share["female"] / total
    if abs(m - f) <= min_margin:
        return "unknown", max(m, f)
    return ("male", m) if m > f else ("female", f)


def mean_p_female(votes: list[Vote]) -> float:
    ps = [v.p_female for v in votes if v.label in VOICED and not np.isnan(v.p_female)]
    return float(np.mean(ps)) if ps else float("nan")


@dataclass
class CallResult:
    call_id: str
    label_1: str
    conf_1: float
    label_2: str
    conf_2: float
    n_turns: int
    needs_qc: bool
    reason: str = ""
    turn_labels: list[str] = field(default_factory=list)


def merge_turns(turn_votes: list[list[Vote]], min_margin: float = 0.0) -> tuple[str, float]:
    """Merge the votes of several turns into one call-level label. Each
    model's votes across turns are pooled, so a model that answered on turn 1
    and abstained on turn 2 still counts once."""
    pooled = [v for tv in turn_votes for v in tv]
    return vote(pooled, min_margin=min_margin)


def classify_call(
    call_id: str,
    turn_votes: list[list[Vote]],
    first_n: int = 2,
    min_margin: float = 0.0,
    min_conf: float = 0.6,
) -> CallResult:
    turn_votes = [tv for tv in turn_votes]  # list of per-turn vote lists, in turn order
    n = len(turn_votes)
    l1, c1 = merge_turns(turn_votes[:first_n], min_margin)
    l2, c2 = merge_turns(turn_votes, min_margin)
    per_turn = [vote(tv, min_margin)[0] for tv in turn_votes]

    needs_qc, reason = False, ""
    if l1 != l2:
        needs_qc, reason = True, "label_1 != label_2"
    elif l1 == "unknown":
        needs_qc, reason = True, "tie"
    elif l1 in VOICED and c1 < min_conf:
        needs_qc, reason = True, f"low confidence {c1:.2f}"
    elif len({t for t in per_turn if t in VOICED}) > 1:
        needs_qc, reason = True, "turns disagree"

    return CallResult(call_id, l1, c1, l2, c2, n, needs_qc, reason, per_turn)


def results_frame(results: list[CallResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "call_id": r.call_id,
                "label_1": r.label_1,
                "conf_1": round(r.conf_1, 3),
                "label_2": r.label_2,
                "conf_2": round(r.conf_2, 3),
                "n_turns": r.n_turns,
                "needs_qc": r.needs_qc,
                "reason": r.reason,
                "turn_labels": "|".join(r.turn_labels),
            }
            for r in results
        ]
    )
