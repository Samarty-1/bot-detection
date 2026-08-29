"""Smoke tests -- mostly guarding against the leaks that made earlier versions
of this project score a meaningless 1.000.

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.detect import IsolationForestDetector, RuleBaseline, SupervisedUpperBound
from src.evaluate import baseline_cost, cost_sweep, precision_at_k, report
from src.features import build
from src.generate import generate


@pytest.fixture(scope="module")
def data():
    events = generate(n_humans=600, n_naive=12, n_advanced=18, seed=3)
    X, y, meta = build(events)
    return events, X, y, meta


def test_generator_produces_all_three_classes(data):
    events, _, _, _ = data
    assert set(events["actor_class"]) == {"human", "naive_bot", "advanced_bot"}


def test_labels_never_reach_the_feature_matrix(data):
    _, X, _, _ = data
    leaked = {"actor_class", "_label", "session_id", "_ja3", "_asn_type"}
    assert not (leaked & set(X.columns))
    assert X.select_dtypes(exclude="number").empty, "non-numeric column would break the models"


def test_no_nulls_or_infinities(data):
    _, X, _, _ = data
    assert not X.isna().any().any()
    assert np.isfinite(X.to_numpy()).all()


def test_advanced_bots_blend_into_the_mainstream_fingerprint(data):
    """The advanced tier must not own a private JA3 bucket.

    An earlier version gave each pool its own spoofed hash, which made bucket
    size a perfect label and drove every metric to 1.000.
    """
    events, _, _, _ = data
    adv = set(events.loc[events["actor_class"] == "advanced_bot", "ja3_hash"])
    human = set(events.loc[events["actor_class"] == "human", "ja3_hash"])
    assert adv, "no advanced bot traffic generated"
    assert adv <= human, "advanced bots must share a fingerprint with real Chrome users"


def test_no_single_feature_separates_the_advanced_tier(data):
    """Guards against a feature that trivially solves the hard tier.

    If any one feature perfectly splits advanced bots from humans, the problem
    is an artifact of the generator rather than a detection result.
    """
    _, X, _, meta = data
    adv = (meta["actor_class"] == "advanced_bot").to_numpy()
    human = (meta["actor_class"] == "human").to_numpy()

    perfect = []
    for col in X.columns:
        a, h = X.loc[adv, col].to_numpy(), X.loc[human, col].to_numpy()
        if a.size == 0 or h.size == 0:
            continue
        # A clean split in either direction means zero overlap in range.
        if a.min() > h.max() or a.max() < h.min():
            perfect.append(col)
    assert not perfect, f"these features perfectly separate advanced bots: {perfect}"


def test_rule_baseline_catches_naive_and_misses_advanced(data):
    _, X, _, meta = data
    scores = RuleBaseline().fit(X).score(X)
    naive = (meta["actor_class"] == "naive_bot").to_numpy()
    adv = (meta["actor_class"] == "advanced_bot").to_numpy()
    assert scores[naive].mean() > 0.9, "rules should catch the naive tier"
    assert scores[adv].mean() < 0.1, "rules should be blind to the advanced tier"


def test_detectors_score_in_unit_interval(data):
    _, X, y, _ = data
    for det in (RuleBaseline().fit(X), IsolationForestDetector(seed=3).fit(X)):
        s = det.score(X)
        assert s.shape == (len(X),)
        assert s.min() >= 0.0 and s.max() <= 1.0


def test_supervised_beats_unsupervised_on_the_hard_tier(data):
    """The headline finding of the project, asserted.

    Labels buy you the advanced tier; unsupervised outlier ranking does not.
    """
    _, X, y, meta = data
    iso = IsolationForestDetector(seed=3).fit(X)
    gb = SupervisedUpperBound(seed=3).fit(X, y)
    X_te, y_te = gb.holdout()
    assert report("gb", y_te.reset_index(drop=True), gb.score(X_te),
                  meta.loc[X_te.index].reset_index(drop=True))["pr_auc"] > \
           report("iso", y, iso.score(X), meta)["pr_auc"]


def test_cost_sweep_always_offers_flagging_nobody():
    """Regression test: a constant-score detector must not flag the whole site."""
    y = np.array([0] * 95 + [1] * 5)
    flat = np.zeros(100)
    sweep = cost_sweep(y, flat)
    best = sweep.loc[sweep["cost"].idxmin()]
    assert best["flagged"] == 0
    assert best["cost"] == baseline_cost(__import__("pandas").Series(y))


def test_precision_at_k_ranks_correctly():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    assert precision_at_k(y, scores, 2) == 1.0
    assert precision_at_k(y, scores, 4) == 0.5
