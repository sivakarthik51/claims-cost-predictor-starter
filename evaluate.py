"""
WEEK 5 STARTER — evaluation, calibration, and picking an operating point.

A ROC-AUC is not a decision. Somebody has to say "flag members above 0.31" and
defend it. That is the work in this file.

Fill in every section marked TODO. When you run this file it should:
  1. Rebuild the same held-out test set train.py used
  2. Plot ROC, precision-recall, and calibration curves to reports/
  3. Print a threshold sweep table
  4. Print the expected dollar value of your chosen operating point

Run:
    python evaluate.py

Check your work:
    pytest tests/test_evaluation.py -v
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from config import HIGH_COST_THRESHOLD_USD

DATA_PATH = Path("data/member_summary.csv")
MODEL_PATH = Path("model.joblib")
REPORT_DIR = Path("reports")

# Must match train.py so you are scoring on data the model never saw.
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# TODO 5 — your operating point.
#
# Pick ONE use case and the threshold that fits it. This choice is yours, it is
# the spine of your final presentation, and there is no single right answer
#
#   "care_management"  a nurse team calls flagged members early. Outreach costs
#                      real money per member and there is a limited number of
#                      nurses. False positives waste calls; false negatives mean
#                      somebody's cancer gets found late.
#   "stop_loss"        the plan sets aside reserve dollars for members likely to
#                      breach the threshold. Under-reserving is a solvency
#                      problem; over-reserving is idle capital.
#   "pricing"          next year's premium is set using predicted risk. Errors
#                      here move money between all members.
#
# Set CHOSEN_USE_CASE to one of those strings, CHOSEN_THRESHOLD to a probability
# between 0 and 1, and write a justification that mentions the trade-off you are
# accepting. "0.5 is the default" is not a justification.
# ---------------------------------------------------------------------------
VALID_USE_CASES = ("care_management", "stop_loss", "pricing")

CHOSEN_USE_CASE = "TODO"
CHOSEN_THRESHOLD = None
THRESHOLD_JUSTIFICATION = ""

# Rough economics for the care-management case, in dollars. Change them if your
# use case implies different numbers, and say so in your write-up.
OUTREACH_COST_PER_FLAGGED = 250
SAVINGS_PER_TRUE_POSITIVE = 4_000


def load_holdout() -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    TODO 1 — rebuild the held-out test set and score it.

    Return (X_test, y_test, y_proba) where y_proba is the model's predicted
    probability of the positive class.

    Load the model with joblib.load(MODEL_PATH) and split with the same
    TEST_SIZE / RANDOM_STATE / stratify=y that train.py used. If your split does
    not match, you are quietly evaluating on rows the model trained on — which
    is the same category of mistake as last week's.

    Hint: the model is a Pipeline, so it takes the raw feature DataFrame.
    Take the feature column names from the pipeline itself rather than
    hard-coding them:
        pipeline["preprocessor"].feature_names_in_
    """
    raise NotImplementedError("Implement load_holdout()")


def threshold_sweep(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    """
    TODO 2 — build the table that turns probabilities into decisions.

    Return a DataFrame with one row per threshold and these columns:
        threshold, n_flagged, tp, fp, fn, precision, recall

    Use thresholds = np.arange(0.05, 1.0, 0.05) when none are passed.
    A member is flagged when y_proba >= threshold.

    Watch out for the threshold where nothing gets flagged: precision is 0/0
    there. Decide what you want to report (0.0 is fine) rather than letting
    numpy hand you a nan and a warning.
    """
    raise NotImplementedError("Implement threshold_sweep()")


def expected_value(tp: int, fp: int, outreach_cost: float, savings_per_tp: float) -> float:
    """
    TODO 3 — the dollar value of an operating point.

    Every flagged member (tp + fp) costs `outreach_cost`. Every true positive
    returns `savings_per_tp`. Return the net dollars.

    This is a deliberately crude model of the world. Its job is to make the
    trade-off explicit, not to be right.
    """
    raise NotImplementedError("Implement expected_value()")


def plot_curves(y_true, y_proba, out_path: Path = REPORT_DIR / "evaluation_curves.png") -> Path:
    """
    TODO 4 — three panels, saved as one PNG. Return the path you wrote.

    Panel 1: ROC curve            sklearn.metrics.roc_curve + roc_auc_score
    Panel 2: precision-recall     sklearn.metrics.precision_recall_curve
             curve                and average_precision_score
    Panel 3: calibration curve    sklearn.calibration.calibration_curve
                                  (n_bins=10, strategy="quantile")

    Mark CHOSEN_THRESHOLD on the ROC and PR panels so a reader can see which
    point on the curve you are actually proposing to operate at.

    On panel 3, plot the diagonal y=x for reference. A point above the diagonal
    means the model under-predicts risk in that bin; below means it over-
    predicts. Note which way yours leans — that matters for the stop-loss and
    pricing use cases much more than for ranking members for outreach.

    Use matplotlib (pip install -r requirements-dev.txt) and remember to
    out_path.parent.mkdir(exist_ok=True) before saving.
    """
    raise NotImplementedError("Implement plot_curves()")


def main() -> None:
    X_test, y_test, y_proba = load_holdout()
    print(f"Scored {len(y_test)} held-out members "
          f"({int(y_test.sum())} of them high-cost at ${HIGH_COST_THRESHOLD_USD:,})")

    sweep = threshold_sweep(y_test, y_proba)
    sweep = sweep.assign(
        net_dollars=[
            expected_value(row.tp, row.fp, OUTREACH_COST_PER_FLAGGED, SAVINGS_PER_TRUE_POSITIVE)
            for row in sweep.itertuples()
        ]
    )
    print("\n=== Threshold sweep ===")
    print(sweep.to_string(index=False))

    path = plot_curves(y_test, y_proba)
    print(f"\nCurves written to {path}")

    print(f"\n=== My operating point ===")
    print(f"Use case:      {CHOSEN_USE_CASE}")
    print(f"Threshold:     {CHOSEN_THRESHOLD}")
    print(f"Justification: {THRESHOLD_JUSTIFICATION}")


if __name__ == "__main__":
    main()
