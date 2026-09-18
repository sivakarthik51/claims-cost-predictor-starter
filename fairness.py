"""
WEEK 6 STARTER — audit the model across demographic groups.

The model is one number per member. A fairness audit asks whether that number
is equally wrong for everybody, and "equally wrong" turns out to have several
mutually incompatible definitions.

Fill in every section marked TODO. When you run this file it should:
  1. Score every member with the trained model
  2. Report per-group rates for sex and age_bucket
  3. Apply the 80% rule
  4. Compare against a model retrained without the demographic columns
  5. Write reports/fairness_report.md

Run:
    python fairness.py

Check your work:
    pytest tests/test_fairness.py -v
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from config import HIGH_COST_THRESHOLD_USD

DATA_PATH = Path("data/member_summary.csv")
MODEL_PATH = Path("model.joblib")
REPORT_PATH = Path("reports/fairness_report.md")

PROTECTED_COLUMNS = ["sex", "age_bucket"]
DECISION_THRESHOLD = 0.5  # swap in the threshold you defended in Week 5

# ---------------------------------------------------------------------------
# TODO 6 — your written assessment.
#
# After you have the numbers, fill this in. It is graded on whether the claim
# follows from your own cohort's numbers, not on whether it is reassuring.
# Reference specific rates. Name the group that is worse off. If a subgroup is
# too small to conclude anything, say that instead of reporting a ratio to two
# decimal places as if it meant something.
# ---------------------------------------------------------------------------
ASSESSMENT = ""


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    """
    TODO 1 — return {"tp": .., "fp": .., "fn": .., "tn": ..}.

    Write it with vectorized comparisons rather than a loop. Both inputs are
    0/1 arrays or Series.
    """
    raise NotImplementedError("Implement confusion_counts()")


def group_metrics(df: pd.DataFrame, group_col: str,
                  y_true_col: str = "y_true", y_pred_col: str = "y_pred") -> pd.DataFrame:
    """
    TODO 2 — one row per group value, with these columns:

        group, n, base_rate, positive_rate, tpr, fpr, fnr, precision

        n              members in the group
        base_rate      share of the group that is actually high-cost
        positive_rate  share of the group the model flags   (demographic parity)
        tpr            of the actually-high-cost, share flagged
        fpr            of the actually-normal, share flagged (equalized odds)
        fnr            1 - tpr; the members the model misses
        precision      of the flagged, share actually high-cost

    Build it on top of confusion_counts(). Sort by group so the output is stable
    across runs. Guard the divisions: a group with zero actual positives has no
    TPR, and reporting 0.0 there is a lie — use np.nan.
    """
    raise NotImplementedError("Implement group_metrics()")


def disparate_impact_ratio(rates: pd.Series) -> float:
    """
    TODO 3 — the four-fifths ratio: the lowest rate divided by the highest.

    Ignore nan rates. Returns a number in [0, 1], where 1.0 is exact parity.
    """
    raise NotImplementedError("Implement disparate_impact_ratio()")


def passes_80_percent_rule(ratio: float) -> bool:
    """
    TODO 4 — True when the ratio clears the EEOC four-fifths guideline.

    Careful with the boundary: exactly 0.80 passes.
    """
    raise NotImplementedError("Implement passes_80_percent_rule()")


def blind_model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO 5 — retrain without the demographic columns and re-audit.

    `df` arrives with y_true / y_pred already on it from the full model. Score
    the blind model yourself rather than reusing those columns, so the two
    numbers you compare come from two actual models.

    Steps:
      - take the fitted pipeline's feature list, drop PROTECTED_COLUMNS
      - build and fit a pipeline of the same shape on the same training split
      - score, threshold, and run group_metrics() again per protected column
      - return a DataFrame with one row per protected column and the disparate
        impact ratio with the demographic features and without, side by side

    Predict the result before you run it, in one sentence: ____________________

    Most students expect the disparity to disappear. Whatever you find, the
    interesting question is *why*. Which remaining columns carry the same
    information the demographic columns did? Check with:
        df.groupby("sex")[numeric_cols].mean()
    """
    raise NotImplementedError("Implement blind_model_comparison()")


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    pipeline = joblib.load(MODEL_PATH)

    features = list(pipeline["preprocessor"].feature_names_in_)
    scored = df.assign(
        y_true=df["is_high_cost"].astype(int),
        y_proba=pipeline.predict_proba(df[features])[:, 1],
    )
    scored["y_pred"] = (scored["y_proba"] >= DECISION_THRESHOLD).astype(int)

    print(f"Auditing at threshold {DECISION_THRESHOLD} "
          f"(label: >${HIGH_COST_THRESHOLD_USD:,} plan-paid)")

    lines = [f"# Fairness audit\n", f"Decision threshold: {DECISION_THRESHOLD}\n"]
    for col in PROTECTED_COLUMNS:
        table = group_metrics(scored, col)
        ratio = disparate_impact_ratio(table.set_index("group")["positive_rate"])
        verdict = "PASS" if passes_80_percent_rule(ratio) else "FAIL"

        print(f"\n=== {col} ===")
        print(table.to_string(index=False))
        print(f"disparate impact ratio (positive rate): {ratio:.3f} → {verdict}")

        lines += [f"\n## {col}\n", table.to_markdown(index=False),
                  f"\n\nDisparate impact ratio: {ratio:.3f} ({verdict})\n"]

    print("\n=== Blind model comparison ===")
    comparison = blind_model_comparison(scored)
    print(comparison.to_string(index=False))
    lines += ["\n## Blind model comparison\n", comparison.to_markdown(index=False),
              f"\n\n## Assessment\n\n{ASSESSMENT}\n"]

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
