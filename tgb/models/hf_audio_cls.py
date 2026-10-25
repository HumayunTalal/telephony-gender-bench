"""Wrapper for any Hugging Face `audio-classification` model with male/female
labels. The first one we used was prithivMLmods/Common-Voice-Gender-Detection
(a wav2vec2 head trained on Common Voice). Others of the same shape can be
registered with a different repo id; the label-name mapping is the only thing
that tends to differ.
"""
from __future__ import annotations

import numpy as np

from .base import Prediction, label_from_p

FEMALE_NAMES = {"female", "f", "woman", "label_1"}
MALE_NAMES = {"male", "m", "man", "label_0"}


class HFAudioClassifier:
    sample_rate = 16_000

    def __init__(self, repo_id: str, name: str | None = None, device: str | None = None):
        self.repo_id = repo_id
        self.name = name or repo_id.split("/")[-1]
        self.device = device
        self._pipe = None

    def load(self) -> None:
        if self._pipe is not None:
            return
        import torch
        from transformers import pipeline

        dev = self.device
        if dev is None:
            dev = 0 if torch.cuda.is_available() else -1
        self._pipe = pipeline("audio-classification", model=self.repo_id, device=dev)

    def predict(self, wave: np.ndarray) -> Prediction:
        self.load()
        out = self._pipe({"raw": wave.astype(np.float32), "sampling_rate": self.sample_rate}, top_k=None)
        scores = {o["label"].lower(): float(o["score"]) for o in out}
        p_f = sum(s for k, s in scores.items() if k in FEMALE_NAMES)
        p_m = sum(s for k, s in scores.items() if k in MALE_NAMES)
        if p_f + p_m == 0:
            raise RuntimeError(f"{self.repo_id}: no male/female labels in {list(scores)}")
        p_female = p_f / (p_f + p_m)
        return Prediction(label_from_p(p_female), p_female, raw=scores)
