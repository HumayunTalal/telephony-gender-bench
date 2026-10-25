#!/usr/bin/env python
"""Run every (model, frontend) pair over a labelled clip list.

Writes:
    results/<model>__<frontend>.csv      one row per clip
    results/summary.csv                  one row per pair
    results/confusion__<model>__<frontend>.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tgb import audio, labels, metrics, models  # noqa: E402
from tgb.pipeline import FrontEnd, run_clip  # noqa: E402

log = logging.getLogger("benchmark")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="label CSV, see data/README.md")
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--models", nargs="+", default=models.names())
    ap.add_argument("--frontend", nargs="+", default=["none", "energy"], choices=["none", "energy", "silero"])
    ap.add_argument("--min-speech", type=float, default=0.6, help="seconds of speech needed to answer")
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (smoke test)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = labels.read_labels(args.labels)
    if args.limit:
        df = df.head(args.limit)
    log.info("labels: %s", labels.summarise(df))

    # load audio once, reuse across pairs
    audio_dir = Path(args.audio_dir)
    waves = {}
    for clip in tqdm(df["clip"], desc="loading"):
        waves[clip] = audio.load(audio_dir / clip).wave

    summary_rows = []
    for mname in args.models:
        model = models.get(mname)
        t0 = time.time()
        model.load()
        log.info("loaded %s in %.1fs", mname, time.time() - t0)

        for det in args.frontend:
            fe = FrontEnd(detector=det, min_speech_seconds=args.min_speech)
            rows = []
            t0 = time.time()
            for _, r in tqdm(df.iterrows(), total=len(df), desc=f"{mname}/{det}"):
                o = run_clip(waves[r["clip"]], fe, model)
                rows.append(
                    {
                        "clip": r["clip"],
                        "gold": r["gold"],
                        "noisy": r["noisy"],
                        "pred": o.label,
                        "p_female": o.p_female,
                        "abstained": o.abstained,
                        "speech_s": round(o.speech_seconds, 2),
                        "speech_ratio": round(o.speech_ratio, 3),
                        "detector": o.detector,
                    }
                )
            elapsed = time.time() - t0
            res = pd.DataFrame(rows)
            res.to_csv(out / f"{mname}__{det}.csv", index=False)

            s = metrics.summary(res["gold"].tolist(), res["pred"].tolist())
            s.update(model=mname, frontend=det, ms_per_clip=round(1000 * elapsed / max(len(res), 1), 1))
            # split by noisy / clean voiced clips, the production question
            for noisy, grp in res[res["gold"] != "noise"].groupby("noisy"):
                k = "noisy" if noisy else "clean"
                s[f"acc_all_{k}"] = metrics.summary(grp["gold"].tolist(), grp["pred"].tolist())["accuracy_all"]
            summary_rows.append(s)
            metrics.confusion(res["gold"].tolist(), res["pred"].tolist()).to_csv(
                out / f"confusion__{mname}__{det}.csv"
            )
            log.info("%s/%s acc_all=%.3f acc_answered=%.3f coverage=%.3f",
                     mname, det, s["accuracy_all"], s["accuracy_answered"], s["coverage"])

    summ = pd.DataFrame(summary_rows)
    cols = ["model", "frontend", "n", "accuracy_all", "accuracy_answered", "coverage",
            "noise_passed_as_voice", "recall_male", "recall_female", "recall_noise",
            "acc_all_clean", "acc_all_noisy", "ms_per_clip"]
    summ = summ[[c for c in cols if c in summ.columns]]
    summ.to_csv(out / "summary.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
