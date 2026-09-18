"""Fixtures shared by the test suite."""

import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

from helpers import ROOT

SUMMARY_PATH = ROOT / "data" / "member_summary.csv"
MODEL_PATH = ROOT / "model.joblib"


def _generate_data() -> None:
    subprocess.run(
        [sys.executable, "generate_claims_data.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )


@pytest.fixture(scope="session")
def data_dir() -> Path:
    if not SUMMARY_PATH.exists():
        _generate_data()
    return ROOT / "data"


@pytest.fixture(scope="session")
def members(data_dir) -> pd.DataFrame:
    return pd.read_csv(data_dir / "members.csv")


@pytest.fixture(scope="session")
def claims(data_dir) -> pd.DataFrame:
    return pd.read_csv(data_dir / "claims.csv")


@pytest.fixture(scope="session")
def summary(data_dir) -> pd.DataFrame:
    """The generator's member-level table — features plus the label."""
    return pd.read_csv(data_dir / "member_summary.csv")


@pytest.fixture(scope="session")
def model(data_dir):
    """The trained pipeline you saved. Skips if you haven't trained one yet."""
    if not MODEL_PATH.exists():
        pytest.skip("model.joblib not found — run `python train.py` first")
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        pytest.skip(
            f"model.joblib exists but will not load ({type(exc).__name__}: {exc}).\n"
            "A pickled model is only loadable by a compatible scikit-learn "
            "version, which is why requirements.txt pins one and why Render "
            "retrains on every deploy. Retrain yours: python train.py"
        )


@pytest.fixture(scope="session")
def model_features(model) -> list[str]:
    """The feature columns the saved model actually expects."""
    try:
        return list(model["preprocessor"].feature_names_in_)
    except (TypeError, KeyError, AttributeError):
        pytest.skip(
            "could not read feature names off model.joblib — is the first "
            "pipeline step named 'preprocessor'?"
        )
