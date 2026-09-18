"""
WEEK 2 — is the dataset you generated actually sane?

These tests check the generator, not your code. They should pass from day one.
If they don't, something is wrong with your environment or your data files, and
that is worth fixing before you build anything on top of them.
"""

import subprocess
import sys

import pandas as pd
import pytest

from helpers import ROOT

from config import HIGH_COST_THRESHOLD_CENTS, STUDENT_SEED


def test_three_csvs_exist_and_are_not_empty(data_dir):
    for name in ("members.csv", "claims.csv", "member_summary.csv"):
        path = data_dir / name
        assert path.exists(), f"{name} is missing — run `python generate_claims_data.py`"
        assert len(pd.read_csv(path)) > 0, f"{name} has no rows"


def test_one_summary_row_per_member(members, summary):
    assert len(summary) == len(members)
    assert summary["member_id"].is_unique
    assert set(summary["member_id"]) == set(members["member_id"])


def test_summary_has_no_missing_values(summary):
    missing = summary.isna().sum()
    assert missing.sum() == 0, f"unexpected nulls:\n{missing[missing > 0]}"


def test_the_label_separates_cleanly_at_the_threshold(summary):
    """
    The label is not magic: the threshold is a line, and every member is on one
    side of it. Stated as a separation property rather than by recomputing the
    comparison — if the two groups overlap at all, something is stale.
    """
    assert set(summary["is_high_cost"].unique()) <= {0, 1}

    paid = summary["total_plan_paid_cents"]
    positives = paid[summary["is_high_cost"] == 1]
    negatives = paid[summary["is_high_cost"] == 0]

    assert int(positives.min()) > HIGH_COST_THRESHOLD_CENTS, (
        "a member is flagged high-cost whose plan-paid total is at or below "
        f"{HIGH_COST_THRESHOLD_CENTS} cents. Did you regenerate the data after "
        "changing HIGH_COST_THRESHOLD_USD?"
    )
    assert int(negatives.max()) <= HIGH_COST_THRESHOLD_CENTS, (
        "a member is not flagged whose plan-paid total is above "
        f"{HIGH_COST_THRESHOLD_CENTS} cents. Did you regenerate the data after "
        "changing HIGH_COST_THRESHOLD_USD?"
    )


def test_positives_are_rare_but_present(summary):
    prevalence = summary["is_high_cost"].mean()
    assert 0.01 < prevalence < 0.30, (
        f"prevalence is {prevalence:.1%}. Anything outside 1–30% means the "
        "threshold and the cost distribution are badly mismatched — a model "
        "cannot learn from 3 positives, and a 'rare event' at 50% is not rare."
    )


def test_claims_reference_real_members(members, claims):
    orphans = set(claims["member_id"]) - set(members["member_id"])
    assert not orphans, f"{len(orphans)} claims belong to members who don't exist"


def test_some_claims_are_denied(claims):
    """Denied claims exist on purpose — they are the Week 3 filter."""
    statuses = set(claims["adjudication_status"])
    assert statuses == {"approved", "denied"}, statuses
    denied_rate = (claims["adjudication_status"] == "denied").mean()
    assert 0.02 < denied_rate < 0.20, f"denied rate is {denied_rate:.1%}"


@pytest.mark.slow
def test_same_seed_gives_identical_data(tmp_path, summary):
    """
    Reproducibility is a claim you can test, so test it.

    Re-running the generator at your STUDENT_SEED in a clean directory has to
    reproduce your dataset byte for byte. If it doesn't, some part of the
    pipeline is drawing randomness that the seed doesn't control, and none of
    your results are reproducible — including the ones in your presentation.
    """
    subprocess.run(
        [sys.executable, str(ROOT / "generate_claims_data.py")],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin", "STUDENT_SEED": str(STUDENT_SEED)},
    )
    regenerated = pd.read_csv(tmp_path / "data" / "member_summary.csv")
    pd.testing.assert_frame_equal(summary, regenerated)


@pytest.mark.slow
def test_a_different_seed_gives_a_different_cohort(tmp_path, summary):
    """Your cohort is yours: a different seed is a genuinely different dataset."""
    other_seed = str(STUDENT_SEED + 1)
    subprocess.run(
        [sys.executable, str(ROOT / "generate_claims_data.py")],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin", "STUDENT_SEED": other_seed},
    )
    other = pd.read_csv(tmp_path / "data" / "member_summary.csv")
    assert not summary["member_id"].equals(other["member_id"])
    assert summary["total_plan_paid_cents"].sum() != other["total_plan_paid_cents"].sum()
