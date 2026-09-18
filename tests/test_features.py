"""
WEEK 3 — does your feature table hold up?

These are the checks you would write for an aggregation you had to trust in
production: one row per member, the parts summing to the whole, every count
bounded by the thing it counts, and the column totals tying back to the source.

Note what is *not* here. Nothing recomputes your aggregation and diffs it. A
test that reimplements the code under test only proves you can write the same
bug twice — and in a real job there is no reference table to diff against, so
constraining the result is the only tool you have. It is also the harder skill,
which is why it is the one being taught.
"""

import numpy as np
import pandas as pd
import pytest

from helpers import load_module, student

from config import HIGH_COST_THRESHOLD_CENTS

EXPECTED_FEATURE_COLUMNS = [
    "member_id",
    "total_claims",
    "medical_claims",
    "pharmacy_claims",
    "specialty_drug_fills",
    "total_billed_cents",
    "total_allowed_cents",
    "total_plan_paid_cents",
    "total_oop_cents",
    "unique_diagnosis_codes",
    "inpatient_claims",
    "has_specialty_drug",
]

COUNT_COLUMNS = [
    "total_claims",
    "medical_claims",
    "pharmacy_claims",
    "specialty_drug_fills",
    "unique_diagnosis_codes",
    "inpatient_claims",
]

MONEY_COLUMNS = [
    "total_billed_cents",
    "total_allowed_cents",
    "total_plan_paid_cents",
    "total_oop_cents",
]


@pytest.fixture(scope="module")
def starter():
    return load_module("features", "approved_claims", "build_member_features")


@pytest.fixture(scope="module")
def approved(starter, claims):
    return student("approved_claims()", starter.approved_claims, claims)


@pytest.fixture(scope="module")
def features(starter, members, approved):
    return student("build_member_features()", starter.build_member_features, members, approved)


@pytest.fixture(scope="module")
def labelled(starter, features):
    return student("add_label()", starter.add_label, features.copy())


# --------------------------------------------------------------------------
# The filter
# --------------------------------------------------------------------------

def test_approved_claims_keeps_exactly_the_approved_rows(approved, claims):
    assert set(approved["adjudication_status"]) == {"approved"}

    expected = set(claims.loc[claims["adjudication_status"] == "approved", "claim_id"])
    got = set(approved["claim_id"])
    assert got == expected, (
        f"{len(expected - got)} approved claim(s) missing, "
        f"{len(got - expected)} non-approved claim(s) kept"
    )


# --------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------

def test_one_row_per_member(features, members):
    assert len(features) == len(members), (
        "your table should have exactly one row per member. More rows usually "
        "means the merge fanned out; fewer means members got dropped."
    )
    assert features["member_id"].is_unique
    assert set(features["member_id"]) == set(members["member_id"])


def test_all_expected_columns_present(features):
    missing = [c for c in EXPECTED_FEATURE_COLUMNS if c not in features.columns]
    assert not missing, f"missing columns: {missing}"


def test_counts_are_numeric_and_never_null(features):
    for col in COUNT_COLUMNS + MONEY_COLUMNS + ["has_specialty_drug"]:
        series = features[col]
        assert np.issubdtype(series.dtype, np.number) or series.dtype == bool, (
            f"{col} has dtype {series.dtype}. A left merge leaves NaN for members "
            "with no approved claims, which silently turns counts into floats "
            "(or objects). Decide what zero claims should look like and fill it."
        )
        assert series.notna().all(), f"{col} contains nulls"


def test_counts_and_money_are_never_negative(features):
    for col in COUNT_COLUMNS + MONEY_COLUMNS:
        bad = features[features[col] < 0]
        assert bad.empty, f"{col} is negative for {len(bad)} member(s)"


# --------------------------------------------------------------------------
# Internal consistency — the parts have to agree with each other
# --------------------------------------------------------------------------

def test_medical_and_pharmacy_partition_the_claim_count(features):
    total = features["medical_claims"] + features["pharmacy_claims"]
    bad = features[total != features["total_claims"]]
    assert bad.empty, (
        f"medical_claims + pharmacy_claims != total_claims for {len(bad)} member(s). "
        "Every claim is one or the other, so the two parts have to add up."
    )


@pytest.mark.parametrize(
    "child,parent,why",
    [
        ("inpatient_claims", "medical_claims", "inpatient claims are medical claims"),
        ("specialty_drug_fills", "pharmacy_claims", "a specialty fill is a pharmacy claim"),
        (
            "unique_diagnosis_codes",
            "medical_claims",
            "only medical claims carry a diagnosis code, and one claim carries one code",
        ),
    ],
)
def test_subcounts_cannot_exceed_their_parent(features, child, parent, why):
    """
    Each of these is a subset of the other, so it can never be larger.

    The diagnosis-code bound is the sharp one: it fires if blanks are being
    counted as a code, or if denied claims crept back in.
    """
    bad = features[features[child] > features[parent]]
    assert bad.empty, (
        f"{child} > {parent} for {len(bad)} member(s), which is impossible — {why}.\n"
        f"First few:\n{bad[['member_id', child, parent]].head()}"
    )


def test_unique_diagnosis_codes_cannot_exceed_the_codebook(features, claims):
    """
    Pigeonhole: nobody can have more distinct codes than exist in the data.

    Sharper than the per-member bound for members with many claims, and it is
    what separates "count the distinct codes" from "count the claims that have
    one" — the second passes every other check in this file.
    """
    codebook = int(claims["diagnosis_code"].nunique())
    bad = features[features["unique_diagnosis_codes"] > codebook]
    assert bad.empty, (
        f"{len(bad)} member(s) have more distinct diagnosis codes than the "
        f"{codebook} that exist in claims.csv. Highest is "
        f"{int(features['unique_diagnosis_codes'].max())}."
    )


def test_has_specialty_drug_agrees_with_the_fill_count(features):
    flag = features["has_specialty_drug"].astype(bool)
    expected = features["specialty_drug_fills"] > 0
    bad = features[flag != expected]
    assert bad.empty, (
        f"has_specialty_drug disagrees with specialty_drug_fills for {len(bad)} "
        "member(s). Two columns describing the same fact must never disagree."
    )


# --------------------------------------------------------------------------
# Tie-out — the member table has to reconcile with the claims it came from
# --------------------------------------------------------------------------

def test_column_totals_tie_back_to_the_claims_table(features, approved):
    """
    Add the member table back up and it has to equal the source table.

    This is a reconciliation, not a reimplementation: it says every claim and
    every cent landed on somebody, without saying anything about how you got
    them onto the right member. An aggregation that drops members, double
    counts a merge, or loses a column fails here.
    """
    expected = {
        "total_claims": len(approved),
        "total_billed_cents": int(approved["billed_cents"].sum()),
        "total_allowed_cents": int(approved["allowed_cents"].sum()),
        "total_plan_paid_cents": int(approved["plan_paid_cents"].sum()),
        "total_oop_cents": int(approved["out_of_pocket_cents"].sum()),
        "inpatient_claims": int((approved["claim_type"] == "Institutional").sum()),
        "pharmacy_claims": int(approved["is_pharmacy"].astype(bool).sum()),
    }
    wrong = {
        col: (int(features[col].sum()), want)
        for col, want in expected.items()
        if int(features[col].sum()) != want
    }
    assert not wrong, "column totals do not reconcile with the claims table:\n" + "\n".join(
        f"    {col:24s} yours {got:>14,}   claims.csv {want:>14,}   off by {got - want:+,}"
        for col, (got, want) in wrong.items()
    )


def test_denied_claims_were_excluded(features, claims):
    """
    Tied to the raw table on purpose.

    The reconciliation above compares against *your* approved_claims() output,
    so it still passes if that filter never ran. This one does not.
    """
    denied = claims[claims["adjudication_status"] == "denied"]
    if denied.empty:
        pytest.skip("no denied claims in this dataset")

    assert int(features["total_claims"].sum()) < len(claims), (
        f"your member table accounts for every one of the {len(claims)} claims, "
        f"including the {len(denied)} denied ones. Denied claims were never paid."
    )
    assert int(features["total_billed_cents"].sum()) < int(claims["billed_cents"].sum())


# --------------------------------------------------------------------------
# The edge case that this dataset does not happen to contain
# --------------------------------------------------------------------------

def test_members_with_no_approved_claims_still_get_a_row(starter, members, approved):
    """
    A member whose every claim was denied is still a member, and still needs a
    prediction. An inner join drops them; a left join leaves NaN behind.

    Your cohort may not contain such a member, so this builds one. Bugs that
    only appear on empty groups are exactly the ones real data finds later.
    """
    sample = members.head(3).copy()
    with_claims = set(sample["member_id"].iloc[:2])
    subset = approved[approved["member_id"].isin(with_claims)]

    table = student("build_member_features()", starter.build_member_features, sample, subset)

    assert len(table) == 3, (
        f"three members in, {len(table)} row(s) out. The member with no approved "
        "claims was dropped — an inner join will do that."
    )

    orphan = sample["member_id"].iloc[2]
    row = table[table["member_id"] == orphan]
    assert len(row) == 1, f"member {orphan} is missing from the table"

    for col in COUNT_COLUMNS + MONEY_COLUMNS:
        value = row.iloc[0][col]
        assert pd.notna(value) and value == 0, (
            f"{col} is {value!r} for a member with no approved claims; it should "
            "be 0. Zero claims is a fact about that member, not missing data."
        )
    assert not bool(row.iloc[0]["has_specialty_drug"])


# --------------------------------------------------------------------------
# The label
# --------------------------------------------------------------------------

def test_label_is_binary_and_separates_at_the_threshold(labelled):
    """
    Stated as a separation property rather than as the comparison itself: every
    flagged member is above the line, every unflagged member is not.
    """
    assert set(labelled["is_high_cost"].unique()) <= {0, 1}

    paid = labelled["total_plan_paid_cents"]
    positives = paid[labelled["is_high_cost"] == 1]
    negatives = paid[labelled["is_high_cost"] == 0]

    if positives.empty or negatives.empty:
        pytest.fail(
            "every member got the same label. The threshold is "
            f"{HIGH_COST_THRESHOLD_CENTS} cents and plan-paid ranges from "
            f"{int(paid.min())} to {int(paid.max())}."
        )

    assert int(positives.min()) > HIGH_COST_THRESHOLD_CENTS, (
        "a member is flagged high-cost whose plan-paid total is at or below the "
        f"{HIGH_COST_THRESHOLD_CENTS}-cent threshold"
    )
    assert int(negatives.max()) <= HIGH_COST_THRESHOLD_CENTS, (
        "a member is not flagged whose plan-paid total is above the "
        f"{HIGH_COST_THRESHOLD_CENTS}-cent threshold"
    )
