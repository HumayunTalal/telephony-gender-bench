import numpy as np

from tgb import vad

SR = 16_000


def tone(seconds, freq=220.0, amp=0.5):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


def test_energy_vad_finds_the_loud_part():
    w = np.concatenate([silence(1.0), tone(1.0), silence(1.0)])
    r = vad.energy_vad(w, SR, adaptive=False)
    assert 0.25 < r.speech_ratio < 0.45  # one third plus hangover
    assert 0.9 < r.speech_seconds < 1.2


def test_quiet_hum_does_not_become_speech():
    # a -50 dBFS hum is below the absolute floor even though it is the loudest thing in the clip
    w = tone(3.0, freq=60.0, amp=0.003)
    r = vad.energy_vad(w, SR)
    assert r.speech_ratio == 0.0


def test_gate_abstains_on_near_silence():
    w = np.concatenate([silence(2.0), tone(0.2), silence(1.0)])
    speech, res, abstain = vad.gate(w, SR, detector="energy", min_speech_seconds=0.6)
    assert abstain
    assert len(speech) < len(w)


def test_gate_none_passes_everything_through():
    w = tone(1.0)
    speech, res, abstain = vad.gate(w, SR, detector="none")
    assert not abstain
    assert len(speech) == len(w)


def test_silero_falls_back_without_torch(monkeypatch):
    def boom(*a, **k):
        raise ImportError("no torch here")

    monkeypatch.setattr(vad, "silero_vad", boom)
    w = tone(2.0)
    speech, res, abstain = vad.gate(w, SR, detector="silero")
    assert res.detector == "energy"
    assert not abstain
