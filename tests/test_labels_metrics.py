import math

import pytest

from tgb import labels, metrics


def test_read_labels(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text("clip,tag,noise_flag\n1.wav,male,\n2.wav,Female,noise\n3.wav,noise,noise\n")
    df = labels.read_labels(p)
    assert df["gold"].tolist() == ["male", "female", "noise"]
    assert df["noisy"].tolist() == [False, True, True]
    assert labels.summarise(df) == {"n": 3, "male": 1, "female": 1, "noise": 1, "noisy_voice": 1}


def test_bad_tag_is_rejected(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text("clip,tag\n1.wav,skip\n")
    with pytest.raises(ValueError):
        labels.read_labels(p)


def test_summary_counts_abstention_on_voiced_as_wrong():
    gold = ["male", "male", "female", "noise"]
    pred = ["male", "noise", "female", "noise"]
    s = metrics.summary(gold, pred)
    assert s["accuracy_all"] == 0.75
    assert s["accuracy_answered"] == 1.0
    assert s["coverage"] == pytest.approx(2 / 3)
    assert s["recall_noise"] == 1.0
    assert s["noise_passed_as_voice"] == 0.0


def test_noise_passed_as_voice():
    s = metrics.summary(["noise", "noise"], ["male", "noise"])
    assert s["noise_passed_as_voice"] == 0.5


def test_confusion_shape():
    c = metrics.confusion(["male", "female"], ["female", "female"])
    assert c.loc["male", "female"] == 1
    assert c.loc["female", "female"] == 1
    assert c.values.sum() == 2


def test_empty_is_nan_not_crash():
    s = metrics.summary([], [])
    assert math.isnan(s["accuracy_all"])
