"""
WEEK 4/5 — is the saved model a deployable artifact?

These run against whatever model.joblib you last saved, the same file the API
will load in Week 7.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from helpers import AUC_FLOOR, LABEL_COLUMN

TEST_SIZE = 0.2
RANDOM_STATE = 42


def test_model_is_a_pipeline_with_preprocessing_inside(model):
    """
    Preprocessing belongs inside the saved artifact.

    If you scale features in train.py and save only the classifier, the API has
    to remember to scale them the same way — and the day it doesn't, predictions
    are quietly wrong with no error anywhere. That is train/serve skew.
    """
    assert isinstance(model, Pipeline), f"model.joblib holds a {type(model).__name__}"
    steps = list(model.named_steps)
    assert "preprocessor" in steps and "classifier" in steps, (
        f"expected steps named 'preprocessor' and 'classifier', got {steps}"
    )


def test_probabilities_are_probabilities(model, summary, model_features):
    proba = model.predict_proba(summary[model_features])[:, 1]
    assert proba.min() >= 0.0 and proba.max() <= 1.0
    assert proba.std() > 0.01, (
        "every member gets essentially the same score — the model is predicting "
        "the base rate and nothing else"
    )


def test_holdout_auc_clears_the_quality_gate(model, summary, model_features):
    """
    A minimum-quality gate, the kind a team puts in CI so a bad retrain cannot
    reach production unnoticed.

    Uses the same split constants as train.py, so this only means what it says
    if you did not change them.
    """
    X = summary[model_features]
    y = summary[LABEL_COLUMN].astype(int)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    assert auc >= AUC_FLOOR, f"held-out ROC-AUC {auc:.4f} < {AUC_FLOOR}"


def test_unseen_category_does_not_crash_the_model(model, summary, model_features):
    """
    Production sends categories you never trained on. Region lists change, and
    a member arrives from one you have never seen. That has to come back as a
    prediction, not a 500.
    """
    row = summary[model_features].head(1).copy()
    categorical = [c for c in row.columns if row[c].dtype == object]
    if not categorical:
        pytest.skip("no categorical features in this model")
    row.loc[:, categorical[0]] = "Antarctica"
    try:
        proba = float(model.predict_proba(row)[0, 1])
    except Exception as exc:
        pytest.fail(
            f"an unseen value in '{categorical[0]}' crashed the model: {exc}\n"
            "Your encoder has to have a defined behaviour for categories that "
            "were not in the training data."
        )
    assert 0.0 <= proba <= 1.0


def test_class_imbalance_was_handled(model, summary, model_features):
    """
    A model that never predicts the positive class is useless, however accurate.

    At the 0.5 cut, it should flag *someone*. There is more than one way to
    fix it if it does not — know which one you chose and why.
    """
    pred = model.predict(summary[model_features])
    flagged = int(np.sum(pred == 1))
    assert flagged > 0, (
        "the model flags zero members as high-cost at the default 0.5 cut. With "
        f"{int(summary[LABEL_COLUMN].sum())} actual positives in the data, "
        "predicting 'normal' for everyone scores well on accuracy and finds "
        "nobody to help."
    )
