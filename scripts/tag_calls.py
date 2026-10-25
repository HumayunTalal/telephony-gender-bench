#!/usr/bin/env python
"""Tag a call database with the combiner.

Input CSV: one row per customer turn audio.
    call_id, turn, clip
`turn` is the 1-based index of the customer turn within the call; `clip` is
the audio file relative to --audio-dir.

Output:
    tagged/calls.csv      label_1, label_2, confidences, needs_qc, reason
    tagged/qc_list.csv    only the rows that need a human look
    tagged/turns.csv      every model's answer on every turn, for debugging
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tgb import audio, models  # noqa: E402
from tgb.combine import Vote, classify_call, results_frame  # noqa: E402
from tgb.pipeline import FrontEnd, run_clip  # noqa: E402

log = logging.getLogger("tag_calls")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calls", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--models", nargs="+", default=models.names())
    ap.add_argument("--frontend", default="energy", choices=["none", "energy", "silero"])
    ap.add_argument("--first-n", type=int, default=2, help="turns that make label_1")
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--min-margin", type=float, default=0.0)
    ap.add_argument("--out", default="tagged")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    calls = pd.read_csv(args.calls, dtype={"call_id": str})
    calls = calls.sort_values(["call_id", "turn"])
    fe = FrontEnd(detector=args.frontend)
    loaded = {m: models.get(m) for m in args.models}
    for m in loaded.values():
        m.load()

    turn_rows, results = [], []
    audio_dir = Path(args.audio_dir)
    for call_id, grp in tqdm(calls.groupby("call_id", sort=False), desc="calls"):
        turn_votes = []
        for _, r in grp.iterrows():
            wave = audio.load(audio_dir / r["clip"]).wave
            votes = []
            for mname, model in loaded.items():
                o = run_clip(wave, fe, model)
                votes.append(Vote(mname, o.label, o.p_female))
                turn_rows.append({"call_id": call_id, "turn": r["turn"], "model": mname,
                                  "label": o.label, "p_female": o.p_female,
                                  "speech_s": round(o.speech_seconds, 2)})
            turn_votes.append(votes)
        results.append(classify_call(call_id, turn_votes, first_n=args.first_n,
                                     min_margin=args.min_margin, min_conf=args.min_conf))

    df = results_frame(results)
    df.to_csv(out / "calls.csv", index=False)
    df[df["needs_qc"]].to_csv(out / "qc_list.csv", index=False)
    pd.DataFrame(turn_rows).to_csv(out / "turns.csv", index=False)

    n = len(df)
    log.info("%d calls: %d male, %d female, %d noise/unknown, %d need QC (%.1f%%)",
             n, (df.label_1 == "male").sum(), (df.label_1 == "female").sum(),
             (~df.label_1.isin(["male", "female"])).sum(), df.needs_qc.sum(),
             100 * df.needs_qc.mean() if n else 0)


if __name__ == "__main__":
    main()
