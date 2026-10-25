"""One clip through one front end and one model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import audio, vad
from .models.base import GenderModel, Prediction


@dataclass
class FrontEnd:
    detector: str = "energy"          # none | energy | silero
    min_speech_seconds: float = 0.6
    min_speech_ratio: float = 0.15
    max_seconds: float = 8.0          # clips longer than this are trimmed; turn audios rarely are
    min_seconds: float = 1.0          # pad shorter clips so the models do not choke
    normalise: bool = True

    @property
    def name(self) -> str:
        return self.detector


@dataclass
class ClipOutcome:
    label: str                 # male | female | noise
    p_female: float
    abstained: bool
    speech_seconds: float
    speech_ratio: float
    detector: str


def run_clip(wave16k: np.ndarray, fe: FrontEnd, model: GenderModel) -> ClipOutcome:
    sr = audio.TARGET_SR
    w = audio.trim_to(wave16k, sr, fe.max_seconds)

    # VAD runs on the clip at its recorded level. Normalising first would lift a
    # clip that is nothing but -50 dBFS hiss up to full scale and the absolute
    # floor in the energy VAD would never trigger. That was an early mistake.
    speech, res, abstain = vad.gate(
        w, sr, detector=fe.detector,
        min_speech_seconds=fe.min_speech_seconds,
        min_speech_ratio=fe.min_speech_ratio,
    )
    if abstain:
        return ClipOutcome("noise", float("nan"), True, res.speech_seconds, res.speech_ratio, res.detector)

    if fe.normalise:
        speech = audio.peak_normalise(speech)
    speech = audio.pad_to(speech, sr, fe.min_seconds)
    pred: Prediction = model.predict(speech)
    return ClipOutcome(pred.label, pred.p_female, False, res.speech_seconds, res.speech_ratio, res.detector)
