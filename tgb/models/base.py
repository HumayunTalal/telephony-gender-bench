"""Every model wrapper exposes the same thing:

    predict(wave16k: np.ndarray) -> Prediction(label, p_female, raw)

`label` is 'male' or 'female'. Abstention is not the model's decision; the
VAD gate upstream decides that and the caller writes 'noise'. Keeping the two
separate makes the metrics honest: the model never gets credit for an
abstention it did not make.

`p_female` is whatever the model's own head produces mapped to a female
probability, so that models with different label names can be averaged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Prediction:
    label: str
    p_female: float
    raw: dict | None = None


class GenderModel(Protocol):
    name: str
    sample_rate: int

    def load(self) -> None: ...

    def predict(self, wave: np.ndarray) -> Prediction: ...


def label_from_p(p_female: float, threshold: float = 0.5) -> str:
    return "female" if p_female >= threshold else "male"
