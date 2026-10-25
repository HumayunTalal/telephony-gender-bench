"""The two-column label schema and a couple of helpers for reading it.

tag        : male | female | noise
noise_flag : noise | ''   (is there background noise on top of the voice)

clean male voice        -> male,
male voice over a TV    -> male,noise
nothing but a dog       -> noise,noise

`gold(tag, noise_flag)` collapses the pair to the single target a classifier
is scored against. Noise on top of a voice is still that voice.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TAGS = ("male", "female", "noise")
NOISE_FLAGS = ("noise", "")


def gold(tag: str, noise_flag: str = "") -> str:
    tag = (tag or "").strip().lower()
    if tag not in TAGS:
        raise ValueError(f"bad tag {tag!r}")
    return tag


def is_noisy(tag: str, noise_flag: str = "") -> bool:
    return (noise_flag or "").strip().lower() == "noise"


def read_labels(path: str | Path) -> pd.DataFrame:
    """Read a label CSV, validate it, return a DataFrame with
    columns [clip, tag, noise_flag, gold, noisy] plus whatever else was there."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"clip", "tag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"label file missing columns: {sorted(missing)}")
    if "noise_flag" not in df.columns:
        df["noise_flag"] = ""

    df["tag"] = df["tag"].str.strip().str.lower()
    df["noise_flag"] = df["noise_flag"].str.strip().str.lower()

    bad = df[~df["tag"].isin(TAGS)]
    if len(bad):
        raise ValueError(f"{len(bad)} rows with an unknown tag, first: {bad.iloc[0].to_dict()}")
    bad = df[~df["noise_flag"].isin(NOISE_FLAGS)]
    if len(bad):
        raise ValueError(f"{len(bad)} rows with an unknown noise_flag, first: {bad.iloc[0].to_dict()}")

    df["gold"] = [gold(t, n) for t, n in zip(df["tag"], df["noise_flag"])]
    df["noisy"] = [is_noisy(t, n) for t, n in zip(df["tag"], df["noise_flag"])]
    return df


def summarise(df: pd.DataFrame) -> dict:
    out = {"n": int(len(df))}
    for t in TAGS:
        out[t] = int((df["gold"] == t).sum())
    out["noisy_voice"] = int(((df["gold"] != "noise") & df["noisy"]).sum())
    return out
