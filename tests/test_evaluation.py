"""
WEEK 5 — evaluation and the operating point.

Mostly invariants on the threshold sweep: raising a threshold can only flag
fewer people, every positive is either caught or missed, a rate stays in
[0, 1]. These hold for *any* correct sweep on *any* model, which is what makes
them worth writing — they will still be true next year on a different model.

Whether the operating point you chose is the right one for your use case is not
something a test can judge. That argument belongs in your PR.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from helpers import LABEL_COLUMN, load_module, student

TEST_SIZE = 0.2
RANDOM_STATE = 42


@pytest.fixture(scope="module")
def starter():
    return load_module("evaluate", "threshold_sweep", "expected_value")


@pytest.fixture(scope="module")
def holdout(starter):
    return student("load_holdout()", starter.load_holdout)


@pytest.fixture(scope="module")
def sweep(starter, holdout):
    _, y_test, y_proba = holdout
    return student("threshold_sweep()", starter.threshold_sweep, y_test, y_proba)


# --------------------------------------------------------------------------
# Evaluating on the right rows
# --------------------------------------------------------------------------

def test_holdout_is_the_same_split_train_used(holdout, summary, model_features):
    """
    Evaluating on rows the model trained on flatters it, for the same reason
    last week's leaky column did.
    """
    X_test, y_test, y_proba = holdout
    y = summary[LABEL_COLUMN].astype(int)
    _, expected_X, _, expected_y = train_test_split(
        summary[model_features], y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    assert len(y_test) == len(expected_y), (
        f"your held-out set has {len(y_test)} rows, train.py's has {len(expected_y)}"
    )
    assert set(np.asarray(X_test.index)) == set(np.asarray(expected_X.index)), (
        "your test set contains different members than the one train.py held out"
    )
    assert np.all((np.asarray(y_proba) >= 0) & (np.asarray(y_proba) <= 1))


# --------------------------------------------------------------------------
# The sweep
# --------------------------------------------------------------------------

def test_sweep_has_the_columns_a_decision_needs(sweep):
    required = {"threshold", "n_flagged", "tp", "fp", "fn", "precision", "recall"}
    missing = required - set(sweep.columns)
    assert not missing, f"missing columns: {sorted(missing)}"
    assert len(sweep) > 1


def test_raising_the_threshold_flags_fewer_people(sweep):
    """The monotonicity that makes a threshold a dial worth turning."""
    ordered = sweep.sort_values("threshold")
    flagged = ordered["n_flagged"].to_numpy()
    assert np.all(np.diff(flagged) <= 0), (
        "n_flagged goes UP somewhere as the threshold rises. Raising the bar "
        f"can only shrink the flagged set:\n{ordered.to_string(index=False)}"
    )
    recall = ordered["recall"].to_numpy()
    assert np.all(np.diff(recall) <= 1e-9), "recall should never rise with the threshold"


def test_confusion_counts_are_consistent(sweep, holdout):
    _, y_test, _ = holdout
    positives = int(np.asarray(y_test).sum())
    for row in sweep.itertuples():
        assert row.tp + row.fn == positives, (
            f"at threshold {row.threshold}: tp + fn = {row.tp + row.fn}, but there "
            f"are {positives} actual positives. Every positive is either caught "
            "or missed."
        )
        assert row.tp + row.fp == row.n_flagged
        assert 0.0 <= row.precision <= 1.0 and 0.0 <= row.recall <= 1.0, (
            "precision/recall outside [0, 1] — usually a 0/0 at a threshold "
            "where nothing is flagged. Decide what to report there."
        )


def test_flagging_nobody_and_everybody_are_both_representable(sweep, holdout):
    """
    The two ends of the dial are where off-by-one and divide-by-zero bugs live,
    and where a stakeholder will point first ("what if we called everyone?").
    """
    _, y_test, _ = holdout
    n = len(np.asarray(y_test))
    assert sweep["n_flagged"].max() <= n, "flagged more members than exist in the holdout"
    assert sweep["n_flagged"].min() >= 0
    assert sweep["tp"].max() <= int(np.asarray(y_test).sum())


# --------------------------------------------------------------------------
# expected_value
# --------------------------------------------------------------------------

def test_expected_value_of_doing_nothing_is_nothing(starter):
    got = student("expected_value()", starter.expected_value, 0, 0, 250, 4000)
    assert got == pytest.approx(0.0), "flag nobody, spend nothing, save nothing"


def test_catching_one_more_member_is_worth_more_than_it_costs(starter):
    """Monotone in true positives — otherwise the program is pointless."""
    base = student("expected_value()", starter.expected_value, 10, 10, 250, 4000)
    better = student("expected_value()", starter.expected_value, 11, 10, 250, 4000)
    assert better > base, (
        f"catching an extra high-cost member moved expected value from {base} "
        f"to {better}. Outreach that works has to pay."
    )


def test_every_false_positive_costs_money(starter):
    """Monotone the other way: you pay to reach out to people you were wrong about."""
    base = student("expected_value()", starter.expected_value, 10, 10, 250, 4000)
    worse = student("expected_value()", starter.expected_value, 10, 11, 250, 4000)
    assert worse < base, (
        f"an extra false positive left expected value at {worse} against {base}. "
        "Every flagged member costs the outreach spend, including the ones you "
        "were wrong about."
    )


def test_expected_value_scales_with_the_size_of_the_programme(starter):
    """
    Run the same programme at twice the size and the value doubles. Pins the
    shape of the arithmetic without pinning the arithmetic.
    """
    single = student("expected_value()", starter.expected_value, 10, 90, 250, 4000)
    double = student("expected_value()", starter.expected_value, 20, 180, 250, 4000)
    assert double == pytest.approx(2 * single)


def test_flagging_only_the_wrong_people_loses_money(starter):
    got = student("expected_value()", starter.expected_value, 0, 50, 250, 4000)
    assert got < 0, f"50 members contacted, none of them high-cost, and the result is {got}"


# --------------------------------------------------------------------------
# The curves
# --------------------------------------------------------------------------

def test_curves_get_written_to_disk(starter, holdout):
    """A smoke test: the plotting path runs end to end and produces a real file."""
    _, y_test, y_proba = holdout
    path = student("plot_curves()", starter.plot_curves, y_test, y_proba)
    assert path is not None and path.exists(), "plot_curves() should return the path it wrote"
    assert path.stat().st_size > 1000, "the saved image looks empty"
