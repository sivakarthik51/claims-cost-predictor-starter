"""
WEEK 6 — the fairness metrics.

The per-group rates are easy to compute and easy to get subtly wrong: a swapped
numerator turns "the model misses 40% of high-cost women" into a number that
looks fine.

The checks here are the ones that catch that without handing you the
arithmetic. Two kinds: **extreme cases**, where the right answer is forced by
the definition (a group the model gets entirely right has TPR 1 and FPR 0, and
there is nothing to compute), and **invariances**, which say what must *not*
change (duplicating a group's rows cannot move its rates; renaming a group
cannot either). Between them they pin the metric down. Reach for both when you
are testing any statistic you cannot easily recompute.
"""

import numpy as np
import pandas as pd
import pytest

from helpers import load_module, student


@pytest.fixture(scope="module")
def starter():
    return load_module("fairness", "confusion_counts", "group_metrics")


def frame(group, y_true, y_pred) -> pd.DataFrame:
    return pd.DataFrame({"group": group, "y_true": y_true, "y_pred": y_pred})


def metrics(starter, df) -> pd.DataFrame:
    return student("group_metrics()", starter.group_metrics, df, "group")


# --------------------------------------------------------------------------
# confusion_counts
# --------------------------------------------------------------------------

def test_confusion_counts_account_for_every_member(starter):
    counts = student(
        "confusion_counts()", starter.confusion_counts, [1, 1, 0, 0, 1, 0], [1, 0, 0, 1, 1, 0]
    )
    assert set(counts) == {"tp", "fp", "fn", "tn"}, f"got keys {sorted(counts)}"
    assert sum(counts.values()) == 6, (
        f"the four counts sum to {sum(counts.values())}, not 6. Every member is "
        "in exactly one of the four boxes."
    )


def test_confusion_counts_on_perfect_predictions(starter):
    """Nothing to compute here: if the model is never wrong, fp and fn are 0."""
    truth = [1, 1, 0, 0, 0]
    counts = student("confusion_counts()", starter.confusion_counts, truth, truth)
    assert counts == {"tp": 2, "fp": 0, "fn": 0, "tn": 3}, (
        f"got {counts} on predictions that are exactly right. tp: flagged and "
        "high-cost. fn: high-cost and not flagged — the members the model fails."
    )


def test_confusion_counts_when_everyone_is_flagged(starter):
    """Flag everybody and nobody can be a false negative or a true negative."""
    counts = student(
        "confusion_counts()", starter.confusion_counts, [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]
    )
    assert counts["fn"] == 0 and counts["tn"] == 0, f"got {counts}"
    assert counts["tp"] == 2 and counts["fp"] == 3, f"got {counts}"


# --------------------------------------------------------------------------
# group_metrics — shape
# --------------------------------------------------------------------------

def test_group_metrics_shape(starter):
    df = frame(["A"] * 4 + ["B"] * 4, [1, 1, 0, 0] * 2, [1, 0, 1, 0, 1, 0, 0, 0])
    table = metrics(starter, df)

    required = {"group", "n", "base_rate", "positive_rate", "tpr", "fpr", "fnr", "precision"}
    assert required <= set(table.columns), f"missing: {sorted(required - set(table.columns))}"
    assert len(table) == 2, "one row per group"
    assert list(table["group"]) == ["A", "B"], "sort by group so runs are comparable"
    assert table["n"].sum() == len(df), "every member belongs to exactly one group"


def test_every_rate_is_a_rate(starter):
    df = frame(["A"] * 4 + ["B"] * 4, [1, 1, 0, 0] * 2, [1, 0, 1, 0, 1, 0, 0, 0])
    table = metrics(starter, df)
    for col in ("base_rate", "positive_rate", "tpr", "fpr", "fnr", "precision"):
        values = table[col].dropna()
        assert ((values >= 0) & (values <= 1)).all(), (
            f"{col} has values outside [0, 1]:\n{table[['group', col]]}"
        )


def test_fnr_and_tpr_are_complements(starter):
    df = frame(["A"] * 4 + ["B"] * 4, [1, 1, 0, 0] * 2, [1, 0, 1, 0, 1, 0, 0, 0])
    table = metrics(starter, df)
    assert np.allclose(table["tpr"] + table["fnr"], 1.0), "fnr is 1 - tpr by definition"


# --------------------------------------------------------------------------
# group_metrics — extreme cases, where the definition forces the answer
# --------------------------------------------------------------------------

def test_a_group_the_model_gets_entirely_right(starter):
    df = frame(["A"] * 4, [1, 1, 0, 0], [1, 1, 0, 0])
    row = metrics(starter, df).set_index("group").loc["A"]
    assert row["tpr"] == pytest.approx(1.0), "every high-cost member was caught"
    assert row["fpr"] == pytest.approx(0.0), "nobody was flagged who should not be"
    assert row["fnr"] == pytest.approx(0.0)
    assert row["precision"] == pytest.approx(1.0)


def test_a_group_where_everyone_is_flagged(starter):
    """Catches a swapped numerator: here tpr and fpr must both be 1."""
    df = frame(["A"] * 4, [1, 1, 0, 0], [1, 1, 1, 1])
    row = metrics(starter, df).set_index("group").loc["A"]
    assert row["positive_rate"] == pytest.approx(1.0)
    assert row["tpr"] == pytest.approx(1.0)
    assert row["fpr"] == pytest.approx(1.0)


def test_a_group_where_nobody_is_flagged(starter):
    df = frame(["A"] * 4, [1, 1, 0, 0], [0, 0, 0, 0])
    row = metrics(starter, df).set_index("group").loc["A"]
    assert row["positive_rate"] == pytest.approx(0.0)
    assert row["tpr"] == pytest.approx(0.0)
    assert row["fnr"] == pytest.approx(1.0), "every high-cost member was missed"


def test_group_with_no_positives_reports_nan_not_zero(starter):
    """
    A group with nobody high-cost has no TPR. Reporting 0.0 claims the model
    misses everyone in that group, which is a different — and false — statement.
    """
    df = frame(["C"] * 3, [0, 0, 0], [0, 1, 0])
    table = metrics(starter, df)
    assert np.isnan(table.loc[0, "tpr"]), f"tpr should be nan, got {table.loc[0, 'tpr']}"


# --------------------------------------------------------------------------
# group_metrics — invariances, which say what must NOT change
# --------------------------------------------------------------------------

def test_rates_do_not_depend_on_group_size(starter):
    """
    Duplicate every member of a group and its rates are unchanged — they are
    rates, not counts. A sum where there should be a mean fails here.
    """
    df = frame(["A"] * 4, [1, 1, 0, 0], [1, 0, 1, 0])
    once = metrics(starter, df).set_index("group")
    twice = metrics(starter, pd.concat([df, df], ignore_index=True)).set_index("group")

    assert twice.loc["A", "n"] == 8, "n counts members, so it should double"
    for col in ("base_rate", "positive_rate", "tpr", "fpr", "fnr", "precision"):
        assert twice.loc["A", col] == pytest.approx(once.loc["A", col]), (
            f"{col} changed when the group was duplicated: "
            f"{once.loc['A', col]} -> {twice.loc['A', col]}"
        )


def test_groups_do_not_contaminate_each_other(starter):
    """
    A group's rates must be computed from its own rows only. Adding a second,
    very different group cannot move the first one's numbers.
    """
    a_only = frame(["A"] * 4, [1, 1, 0, 0], [1, 0, 1, 0])
    both = pd.concat(
        [a_only, frame(["B"] * 4, [1, 1, 1, 1], [1, 1, 1, 1])], ignore_index=True
    )

    alone = metrics(starter, a_only).set_index("group")
    together = metrics(starter, both).set_index("group")
    for col in ("base_rate", "positive_rate", "tpr", "fpr", "fnr", "precision"):
        assert together.loc["A", col] == pytest.approx(alone.loc["A", col]), (
            f"group A's {col} changed when group B was added — the groupby is "
            "leaking rows between groups"
        )


def test_base_rate_ignores_the_predictions(starter):
    """
    base_rate describes the members, not the model. Change the predictions and
    it must not move — that is what separates it from positive_rate.

    The two prediction vectors deliberately have different means, so an
    implementation that reads y_pred cannot slip through unnoticed.
    """
    df = frame(["A"] * 4, [1, 1, 0, 0], [1, 1, 1, 0])
    flipped = df.assign(y_pred=1 - df["y_pred"])

    base = metrics(starter, df).loc[0, "base_rate"]
    assert base == pytest.approx(metrics(starter, flipped).loc[0, "base_rate"]), (
        "base_rate changed when the predictions changed. It is the share of "
        "members who are actually high-cost, which the model has no say in."
    )
    assert metrics(starter, df).loc[0, "positive_rate"] != pytest.approx(
        metrics(starter, flipped).loc[0, "positive_rate"]
    ), "positive_rate did NOT change when every prediction flipped — check which column it reads"


def test_base_rate_of_a_group_that_is_entirely_high_cost(starter):
    """Forced by the definition: everyone is high-cost, so the base rate is 1."""
    df = frame(["A"] * 4, [1, 1, 1, 1], [1, 0, 1, 0])
    assert metrics(starter, df).loc[0, "base_rate"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# disparate_impact_ratio
# --------------------------------------------------------------------------

def test_equal_rates_are_perfectly_fair(starter):
    ratio = student(
        "disparate_impact_ratio()", starter.disparate_impact_ratio, pd.Series({"A": 0.3, "B": 0.3})
    )
    assert ratio == pytest.approx(1.0)


def test_ratio_never_exceeds_one(starter):
    for rates in ({"A": 0.5, "B": 0.4}, {"A": 0.1, "B": 0.9}, {"A": 0.2, "B": 0.2, "C": 0.7}):
        ratio = student(
            "disparate_impact_ratio()", starter.disparate_impact_ratio, pd.Series(rates)
        )
        assert 0.0 <= ratio <= 1.0, f"{rates} gave {ratio}; a disparity ratio is at most 1"


def test_ratio_is_unchanged_when_every_rate_is_scaled(starter):
    """
    A ratio compares groups to each other, so doubling everybody's rate cannot
    change it. An implementation that subtracts instead of dividing fails here.
    """
    rates = pd.Series({"A": 0.4, "B": 0.2, "C": 0.3})
    once = student("disparate_impact_ratio()", starter.disparate_impact_ratio, rates)
    scaled = student("disparate_impact_ratio()", starter.disparate_impact_ratio, rates * 2)
    assert scaled == pytest.approx(once)


def test_ratio_does_not_depend_on_group_order(starter):
    rates = pd.Series({"A": 0.4, "B": 0.2, "C": 0.3})
    forwards = student("disparate_impact_ratio()", starter.disparate_impact_ratio, rates)
    backwards = student(
        "disparate_impact_ratio()", starter.disparate_impact_ratio, rates.iloc[::-1]
    )
    assert backwards == pytest.approx(forwards)


def test_adding_a_worse_off_group_can_only_widen_the_disparity(starter):
    rates = pd.Series({"A": 0.4, "B": 0.3})
    before = student("disparate_impact_ratio()", starter.disparate_impact_ratio, rates)
    after = student(
        "disparate_impact_ratio()",
        starter.disparate_impact_ratio,
        pd.concat([rates, pd.Series({"C": 0.1})]),
    )
    assert after <= before + 1e-12, (
        f"adding a group with a much lower rate moved the ratio from {before} "
        f"to {after}. A new worst-off group cannot make things look fairer."
    )


def test_unknown_rates_are_skipped_not_counted_as_zero(starter):
    ratio = student(
        "disparate_impact_ratio()",
        starter.disparate_impact_ratio,
        pd.Series({"A": 0.5, "B": 0.25, "C": np.nan}),
    )
    assert ratio == pytest.approx(0.5), "nan rates are unknown, not zero — skip them"


def test_80_percent_rule_boundary(starter):
    assert student("passes_80_percent_rule()", starter.passes_80_percent_rule, 0.95) is True
    assert student("passes_80_percent_rule()", starter.passes_80_percent_rule, 0.79) is False
    assert student("passes_80_percent_rule()", starter.passes_80_percent_rule, 0.80) is True, (
        "the four-fifths rule is a floor: exactly 0.80 passes"
    )


# --------------------------------------------------------------------------
# The blind-model comparison
# --------------------------------------------------------------------------

def test_blind_comparison_reports_before_and_after(starter, summary, model, model_features):
    proba = model.predict_proba(summary[model_features])[:, 1]
    scored = summary.assign(
        y_true=summary["is_high_cost"].astype(int),
        y_proba=proba,
        y_pred=(proba >= starter.DECISION_THRESHOLD).astype(int),
    )
    comparison = student("blind_model_comparison()", starter.blind_model_comparison, scored)
    assert isinstance(comparison, pd.DataFrame) and len(comparison) >= 1
    assert comparison.select_dtypes("number").shape[1] >= 2, (
        "the comparison needs the disparity before dropping the demographic "
        "columns and after, side by side — one number alone says nothing"
    )
    assert comparison.select_dtypes("number").notna().any().any(), (
        "every number in the comparison is nan — check that you re-scored the "
        "data with each model rather than reusing one set of predictions"
    )
