"""Audio loading and the small amount of conditioning every model gets.

Call recordings arrive as 8 kHz mono PCM (sometimes mu-law WAV, sometimes
plain 16-bit). Every model we test expects 16 kHz, so the first thing we do
is resample. Resampling does not add any information above 4 kHz; it just
puts the signal in the sample rate the model was trained on. That is worth
keeping in mind when reading the results: the models are seeing a band-limited
version of what they were trained on, not a degraded one.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16_000


@dataclass
class Clip:
    wave: np.ndarray  # float32, mono, in [-1, 1]
    sr: int
    path: str | None = None

    @property
    def duration(self) -> float:
        return len(self.wave) / self.sr


def load(path: str | Path, target_sr: int = TARGET_SR) -> Clip:
    """Load any soundfile-readable file as mono float32 at target_sr."""
    wave, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)  # mono; stereo calls are agent/customer split, caller handles that
    if sr != target_sr:
        wave = resample(wave, sr, target_sr)
    return Clip(wave=wave, sr=target_sr, path=str(path))


def resample(wave: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return wave
    g = np.gcd(sr_in, sr_out)
    return resample_poly(wave, sr_out // g, sr_in // g).astype(np.float32)


def peak_normalise(wave: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = np.max(np.abs(wave)) if len(wave) else 0.0
    if m < 1e-6:
        return wave
    return (wave * (peak / m)).astype(np.float32)


def rms_dbfs(wave: np.ndarray) -> float:
    if len(wave) == 0:
        return -np.inf
    r = np.sqrt(np.mean(wave.astype(np.float64) ** 2))
    return 20 * np.log10(r + 1e-12)


def trim_to(wave: np.ndarray, sr: int, max_seconds: float) -> np.ndarray:
    n = int(max_seconds * sr)
    return wave[:n] if len(wave) > n else wave


def pad_to(wave: np.ndarray, sr: int, min_seconds: float) -> np.ndarray:
    """Some models fall over on very short input. Zero-pad rather than loop,
    so we are not inventing speech."""
    n = int(min_seconds * sr)
    if len(wave) >= n:
        return wave
    return np.concatenate([wave, np.zeros(n - len(wave), dtype=np.float32)])
