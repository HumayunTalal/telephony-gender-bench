"""JaesungHuh/voice-gender-classifier: an ECAPA-TDNN speaker encoder with a
gender head, trained on VoxCeleb. It is not a transformers model, the repo
ships its own `model.py`, so we pull the snapshot and import it from there.

This one is interesting for our audio because VoxCeleb is in-the-wild
YouTube speech, so it has seen noise, but it is still 16 kHz wideband.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from .base import Prediction, label_from_p

REPO_ID = "JaesungHuh/voice-gender-classifier"


class ECAPAGender:
    sample_rate = 16_000
    name = "ecapa_gender"

    def __init__(self, repo_id: str = REPO_ID, device: str | None = None):
        self.repo_id = repo_id
        self.device = device
        self._model = None
        self._torch = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from huggingface_hub import snapshot_download

        local = Path(snapshot_download(self.repo_id))
        spec = importlib.util.spec_from_file_location("ecapa_gender_model", local / "model.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["ecapa_gender_model"] = mod
        spec.loader.exec_module(mod)

        model = mod.ECAPA_gender.from_pretrained(self.repo_id)
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(dev).eval()
        self._model, self._torch, self.device = model, torch, dev

    def predict(self, wave: np.ndarray) -> Prediction:
        self.load()
        torch = self._torch
        x = torch.from_numpy(wave.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._model(x)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        # the repo's label order is pred_labels = ['male', 'female']
        p_female = float(probs[1]) if len(probs) == 2 else float(probs[-1])
        return Prediction(label_from_p(p_female), p_female, raw={"probs": probs.tolist()})
