from tgb.combine import Vote, vote, classify_call


def v(model, label, p=None):
    if p is None:
        p = 0.9 if label == "female" else 0.1
    return Vote(model, label, float("nan") if label == "noise" else p)


def test_majority_wins():
    label, conf = vote([v("a", "male"), v("b", "male"), v("c", "female")])
    assert label == "male"
    assert abs(conf - 2 / 3) < 1e-9


def test_abstentions_do_not_vote():
    label, conf = vote([v("a", "noise"), v("b", "female")])
    assert label == "female"
    assert conf == 1.0


def test_all_abstain_is_noise():
    assert vote([v("a", "noise"), v("b", "noise")]) == ("noise", 0.0)


def test_tie_is_unknown_not_a_coin_flip():
    label, _ = vote([v("a", "male"), v("b", "female")])
    assert label == "unknown"


def test_margin_requirement():
    votes = [v("a", "male"), v("b", "male"), v("c", "female")]
    assert vote(votes, min_margin=0.5)[0] == "unknown"
    assert vote(votes, min_margin=0.2)[0] == "male"


def test_two_turn_call_has_equal_labels():
    turns = [[v("a", "male"), v("b", "male")], [v("a", "male"), v("b", "male")]]
    r = classify_call("c1", turns)
    assert r.label_1 == r.label_2 == "male"
    assert not r.needs_qc


def test_agent_leak_on_turn_three_goes_to_qc():
    # customer is female on turns 1-2, turn 3 is dominated by the (male) agent voice
    turns = [
        [v("a", "female"), v("b", "female")],
        [v("a", "female"), v("b", "female")],
        [v("a", "male"), v("b", "male")],
        [v("a", "male"), v("b", "male")],
        [v("a", "male"), v("b", "male")],
    ]
    r = classify_call("c2", turns)
    assert r.label_1 == "female"
    assert r.label_2 == "male"
    assert r.needs_qc and r.reason == "label_1 != label_2"


def test_low_confidence_goes_to_qc():
    turns = [[v("a", "male"), v("b", "male"), v("c", "female"), v("d", "female"), v("e", "male")]]
    r = classify_call("c3", turns, min_conf=0.7)
    assert r.label_1 == "male"
    assert r.needs_qc and r.reason.startswith("low confidence")
