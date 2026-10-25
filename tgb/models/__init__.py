"""Model registry. Add a line here to put a new model in the benchmark."""
from __future__ import annotations

from .base import GenderModel, Prediction  # noqa: F401


def _cv_gender():
    from .hf_audio_cls import HFAudioClassifier

    return HFAudioClassifier("prithivMLmods/Common-Voice-Gender-Detection", name="cv_gender")


def _ecapa_gender():
    from .ecapa_gender import ECAPAGender

    return ECAPAGender()


REGISTRY = {
    "cv_gender": _cv_gender,
    "ecapa_gender": _ecapa_gender,
    # add more HF audio-classification models like this:
    # "some_other": lambda: HFAudioClassifier("org/repo", name="some_other"),
}


def get(name: str) -> GenderModel:
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]()


def names() -> list[str]:
    return sorted(REGISTRY)
