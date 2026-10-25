import numpy as np
import soundfile as sf

from tgb import audio
from tgb.models.base import Prediction
from tgb.pipeline import FrontEnd, run_clip


class FakeModel:
    """Says female when the mean frequency is high. Enough to test plumbing."""
    name = "fake"
    sample_rate = 16_000

    def load(self):
        pass

    def predict(self, wave):
        zc = np.mean(np.abs(np.diff(np.sign(wave)))) / 2  # zero-crossing rate
        p = float(min(max(zc * 20, 0.0), 1.0))
        return Prediction("female" if p >= 0.5 else "male", p)


def test_load_resamples_8k_to_16k(tmp_path):
    sr = 8000
    t = np.arange(sr) / sr
    w = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    f = tmp_path / "a.wav"
    sf.write(f, w, sr, subtype="PCM_16")
    c = audio.load(f)
    assert c.sr == 16_000
    assert abs(c.duration - 1.0) < 0.01


def test_pipeline_abstains_then_answers():
    sr = 16_000
    quiet = np.zeros(sr * 2, dtype=np.float32)
    o = run_clip(quiet, FrontEnd(detector="energy"), FakeModel())
    assert o.label == "noise" and o.abstained

    t = np.arange(sr * 2) / sr
    voiced = (0.5 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    o = run_clip(voiced, FrontEnd(detector="energy"), FakeModel())
    assert o.label in ("male", "female") and not o.abstained
    assert o.speech_seconds > 1.5


def test_short_clip_is_padded_not_dropped():
    sr = 16_000
    t = np.arange(int(sr * 0.7)) / sr
    w = (0.5 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    o = run_clip(w, FrontEnd(detector="energy", min_speech_seconds=0.5, min_seconds=1.0), FakeModel())
    assert not o.abstained
