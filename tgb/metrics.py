"""Scoring with abstention accounted for.

A model that is allowed to say "noise" (or to abstain because the VAD gate
found no speech) can look very accurate on the clips it chose to answer. So
every summary carries three numbers side by side:

    accuracy_all       correct / all clips, where an abstention on a voiced
                       clip counts as wrong
    accuracy_answered  correct / clips the model actually gave male/female on
    coverage           answered / voiced clips

Production cares about the first. The model comparison is fairer on the
second, as long as coverage is printed next to it.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

CLASSES = ("male", "female", "noise")
VOICED = ("male", "female")


def confusion(gold: list[str], pred: list[str]) -> pd.DataFrame:
    c = Counter(zip(gold, pred))
    table = pd.DataFrame(0, index=list(CLASSES), columns=list(CLASSES), dtype=int)
    for (g, p), n in c.items():
        table.loc[g, p] = n
    table.index.name = "gold \\ pred"
    return table


def summary(gold: list[str], pred: list[str]) -> dict:
    assert len(gold) == len(pred)
    n = len(gold)
    correct = sum(g == p for g, p in zip(gold, pred))

    voiced_idx = [i for i, g in enumerate(gold) if g in VOICED]
    answered_idx = [i for i in voiced_idx if pred[i] in VOICED]
    correct_answered = sum(gold[i] == pred[i] for i in answered_idx)

    per_class = {}
    for c in CLASSES:
        idx = [i for i, g in enumerate(gold) if g == c]
        per_class[f"recall_{c}"] = (sum(pred[i] == c for i in idx) / len(idx)) if idx else float("nan")

    noise_as_voice = sum(1 for i, g in enumerate(gold) if g == "noise" and pred[i] in VOICED)
    noise_total = sum(1 for g in gold if g == "noise")

    return {
        "n": n,
        "accuracy_all": correct / n if n else float("nan"),
        "accuracy_answered": correct_answered / len(answered_idx) if answered_idx else float("nan"),
        "coverage": len(answered_idx) / len(voiced_idx) if voiced_idx else float("nan"),
        "noise_passed_as_voice": noise_as_voice / noise_total if noise_total else float("nan"),
        **per_class,
    }


def summary_by(df: pd.DataFrame, by: str, gold_col: str = "gold", pred_col: str = "pred") -> pd.DataFrame:
    rows = []
    for key, grp in df.groupby(by):
        s = summary(grp[gold_col].tolist(), grp[pred_col].tolist())
        s[by] = key
        rows.append(s)
    return pd.DataFrame(rows).set_index(by)
