"""
WEEK 4 — the leakage guard.

This is a real test, and the one most ML projects are missing. It encodes a
rule you want enforced on every model you ship: no input may rank the outcome
so well on its own that it is plainly a restatement of it, and no model may
score so well on held-out data that the only explanation is a leak.

Nothing here checks that your model is good. These check that it is honest.

Read this file only after you have run train.py once and written down the
ROC-AUC it gave you.
"""

import pytest

from helpers import (
    AUC_FLOOR,
    ID_COLUMNS,
    LABEL_COLUMN,
    TOO_GOOD_MODEL_AUC,
    leak_message,
    leak_report,
    load_module,
    skip_if_placeholder,
    student,
)


@pytest.fixture(scope="module")
def chosen_features():
    """The feature list you settled on, from features.model_features()."""
    module = load_module("features", "model_features")
    return skip_if_placeholder(
        student("model_features()", module.model_features), "model_features()"
    )


# --------------------------------------------------------------------------
# The feature list you chose
# --------------------------------------------------------------------------

def test_feature_list_excludes_the_label_and_the_identifiers(chosen_features):
    assert LABEL_COLUMN not in chosen_features, (
        f"{LABEL_COLUMN} is in your feature list. It is the thing you are "
        "predicting — it cannot also be an input."
    )
    identifiers = sorted(set(chosen_features) & set(ID_COLUMNS))
    assert not identifiers, (
        f"identifiers {identifiers} are in your feature list. A member_id "
        "predicts nothing; it just lets the model memorize rows."
    )


def test_feature_list_is_leak_free(chosen_features, summary):
    y = summary[LABEL_COLUMN].astype(int)
    hits = leak_report(summary, chosen_features, y)
    assert not hits, leak_message(hits, "model_features()")


# --------------------------------------------------------------------------
# The feature list train.py actually trains on
# --------------------------------------------------------------------------

def test_training_features_are_leak_free(summary):
    """
    The one that catches the trap.

    train.py ships with every numeric column in member_summary.csv in its
    feature list, which is why your first ROC-AUC looked the way it did.
    """
    train = load_module("train", "NUMERIC_FEATURES", "CATEGORICAL_FEATURES")
    features = list(train.CATEGORICAL_FEATURES) + list(train.NUMERIC_FEATURES)

    assert LABEL_COLUMN not in features, "you are training on the label itself"

    y = summary[LABEL_COLUMN].astype(int)
    hits = leak_report(summary, features, y)
    assert not hits, leak_message(hits, "train.py's NUMERIC_FEATURES")


# --------------------------------------------------------------------------
# The artifact that actually gets deployed
# --------------------------------------------------------------------------

def test_saved_model_is_leak_free(summary, model_features):
    """model.joblib is what Render serves. Check that one too."""
    y = summary[LABEL_COLUMN].astype(int)
    hits = leak_report(summary, model_features, y)
    assert not hits, leak_message(hits, "the feature list inside model.joblib")


def test_saved_model_is_not_too_good_to_be_true(summary, model, model_features):
    """
    A held-out ROC-AUC this high on a real problem means the label got in.

    Nobody predicts who will cost a health plan $25,000 next year with 99.8%
    ranking accuracy from claim counts and demographics. If you see that number
    in industry, the first thing to look for is leakage, not a promotion.

    Note this is measured on rows held out of training. In-sample it is normal
    for a random forest to look near-perfect on any feature set, which is its
    own lesson: the number you quote has to come from data the model never saw.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    X = summary[model_features]
    y = summary[LABEL_COLUMN].astype(int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    assert auc < TOO_GOOD_MODEL_AUC, (
        f"the saved model scores held-out AUC {auc:.4f}. That is not skill.\n"
        "Find the column carrying the answer, drop it, retrain, and re-run.\n"
        "Expect to land somewhere around 0.88-0.99 once it is honest — and note "
        f"the deliverable floor is {AUC_FLOOR}, so honest is still good enough."
    )
