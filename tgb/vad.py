"""Voice activity detection and the speech gate.

Two detectors. The energy VAD has no dependencies and is what the harness
falls back to. Silero is better on real calls but needs torch, so it is
optional and loaded lazily.

The important output is not the mask itself but `speech_ratio` and the
decision to abstain. The production problem was never "the VAD missed some
speech". It was the opposite: a line hum, a TV, or a dog bark got through as
speech, the gender model was handed a clip with no voice in it, and it
answered anyway. So the gate below is deliberately strict. If less than
`min_speech_seconds` of speech survives, the clip is marked `noise` and the
gender model is not asked.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

from .audio import rms_dbfs

log = logging.getLogger(__name__)


@dataclass
class VadResult:
    mask: np.ndarray          # bool per frame
    frame_len: int            # samples per frame
    speech_ratio: float       # fraction of frames marked speech
    speech_seconds: float
    detector: str


# ---------------------------------------------------------------------------
# Energy VAD


def energy_vad(
    wave: np.ndarray,
    sr: int,
    frame_ms: float = 30.0,
    threshold_db: float = -35.0,
    adaptive: bool = True,
    hangover_frames: int = 3,
) -> VadResult:
    """Frame energy against a threshold.

    With `adaptive=True` the threshold is the noise floor (10th percentile of
    frame energy) plus 12 dB, which copes with clips that are quiet overall.
    The absolute floor `threshold_db` still applies, so a clip that is nothing
    but a -50 dBFS hum does not get promoted to speech by the adaptive rule.
    """
    frame_len = int(sr * frame_ms / 1000)
    if frame_len <= 0 or len(wave) < frame_len:
        return VadResult(np.zeros(0, bool), max(frame_len, 1), 0.0, 0.0, "energy")

    n_frames = len(wave) // frame_len
    frames = wave[: n_frames * frame_len].reshape(n_frames, frame_len)
    energies = np.array([rms_dbfs(f) for f in frames])

    thr = threshold_db
    if adaptive:
        finite = energies[np.isfinite(energies)]
        floor = np.percentile(finite, 10)
        # floor + 12 dB, but never above the louder part of the clip, otherwise a
        # clip that is speech from start to finish gets no speech frames at all
        ceiling = np.percentile(finite, 70) - 3.0
        thr = max(threshold_db, min(floor + 12.0, ceiling))

    mask = energies > thr

    # hangover: keep a few frames after speech so we do not chop word tails
    if hangover_frames > 0 and mask.any():
        out = mask.copy()
        run = 0
        for i, m in enumerate(mask):
            if m:
                run = hangover_frames
            elif run > 0:
                out[i] = True
                run -= 1
        mask = out

    ratio = float(mask.mean())
    return VadResult(mask, frame_len, ratio, ratio * n_frames * frame_len / sr, "energy")


# ---------------------------------------------------------------------------
# Silero VAD (optional)

_silero = None


def _load_silero():
    global _silero
    if _silero is None:
        import torch  # noqa: F401  (import error surfaces to the caller)

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
        )
        get_speech_timestamps = utils[0]
        _silero = (model, get_speech_timestamps)
    return _silero


def silero_vad(wave: np.ndarray, sr: int, threshold: float = 0.5) -> VadResult:
    import torch

    model, get_ts = _load_silero()
    if sr not in (8000, 16000):
        raise ValueError("silero expects 8k or 16k audio")
    ts = get_ts(torch.from_numpy(wave), model, sampling_rate=sr, threshold=threshold)
    frame_len = int(sr * 0.03)
    n_frames = max(len(wave) // frame_len, 1)
    mask = np.zeros(n_frames, dtype=bool)
    speech = 0
    for seg in ts:
        a, b = seg["start"], seg["end"]
        speech += b - a
        mask[a // frame_len : max(b // frame_len, a // frame_len + 1)] = True
    ratio = float(mask.mean())
    return VadResult(mask, frame_len, ratio, speech / sr, "silero")


# ---------------------------------------------------------------------------
# Gate


def apply_mask(wave: np.ndarray, res: VadResult) -> np.ndarray:
    """Keep only the speech frames, concatenated. Good enough for a gender
    model, which does not care about the gaps."""
    if res.mask.size == 0:
        return wave[:0]
    keep = []
    for i, m in enumerate(res.mask):
        if m:
            keep.append(wave[i * res.frame_len : (i + 1) * res.frame_len])
    return np.concatenate(keep) if keep else wave[:0]


def gate(
    wave: np.ndarray,
    sr: int,
    detector: str = "energy",
    min_speech_seconds: float = 0.6,
    min_speech_ratio: float = 0.15,
) -> tuple[np.ndarray, VadResult, bool]:
    """Return (speech_only_wave, vad_result, should_abstain)."""
    if detector == "silero":
        try:
            res = silero_vad(wave, sr)
        except Exception as e:  # torch missing, hub offline, etc.
            log.warning("silero unavailable (%s), falling back to energy VAD", e)
            res = energy_vad(wave, sr)
    elif detector == "energy":
        res = energy_vad(wave, sr)
    elif detector == "none":
        res = VadResult(np.ones(1, bool), len(wave) or 1, 1.0, len(wave) / sr, "none")
        return wave, res, False
    else:
        raise ValueError(f"unknown detector {detector!r}")

    abstain = res.speech_seconds < min_speech_seconds or res.speech_ratio < min_speech_ratio
    return apply_mask(wave, res), res, abstain
