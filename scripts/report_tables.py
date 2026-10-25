#!/usr/bin/env python
"""Turn results/summary.csv into the markdown tables pasted into the report."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def pct(x):
    return "" if pd.isna(x) else f"{100 * x:.1f}"


def main(results_dir="results"):
    summ = pd.read_csv(Path(results_dir) / "summary.csv")
    print("| model | front end | n | acc (all) | acc (answered) | coverage | noise→voice | acc clean | acc noisy | ms/clip |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in summ.iterrows():
        print(f"| {r.model} | {r.frontend} | {int(r.n)} | {pct(r.accuracy_all)} | {pct(r.accuracy_answered)} | "
              f"{pct(r.coverage)} | {pct(r.noise_passed_as_voice)} | {pct(r.get('acc_all_clean'))} | "
              f"{pct(r.get('acc_all_noisy'))} | {r.ms_per_clip} |")

    for f in sorted(Path(results_dir).glob("confusion__*.csv")):
        c = pd.read_csv(f, index_col=0)
        print(f"\n**{f.stem.replace('confusion__', '').replace('__', ' / ')}**\n")
        print(c.to_markdown())


if __name__ == "__main__":
    main(*sys.argv[1:])
