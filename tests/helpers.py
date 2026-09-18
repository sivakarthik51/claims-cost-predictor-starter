"""
Shared test helpers.

The tests in this directory are your feedback loop: they are meant to tell you
whether the thing you just wrote is right, without waiting for a session. A
test that fails is a to-do item, and the failure message is supposed to tell you
what to fix. If one ever doesn't, say so — an unhelpful failure message is a bug
in the test, not in you.

Tests for code you haven't written yet SKIP rather than fail, so a green-ish run
early in the project is expected.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# A single column whose value alone ranks members this well is not a feature,
# it is the answer. See tests/test_leakage.py.
LEAK_SINGLE_FEATURE_AUC = 0.99

# A whole model scoring above this on HELD-OUT data is not a good model, it is a
# model that has been handed the label. The margin is real, not arbitrary: on
# this dataset an honest model tops out around 0.99, while any model given a
# post-adjudication dollar column lands at 0.9995+. The per-column guard above
# is the primary detector; this is the backstop.
TOO_GOOD_MODEL_AUC = 0.998

# The deliverable floor from the project scope.
AUC_FLOOR = 0.85

ID_COLUMNS = ("member_id", "plan_year_start", "plan_year_end")
LABEL_COLUMN = "is_high_cost"


def load_module(module_name: str, *attrs: str):
    """Import one of the project modules, skipping the test if it isn't ready yet."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - environment issue
        pytest.skip(f"could not import {module_name}: {exc}")
    for attr in attrs:
        if not hasattr(module, attr):
            pytest.skip(f"{module_name}.{attr} does not exist yet")
    return module


def student(label: str, fn, *args, **kwargs):
    """Call student code, turning 'not written yet' into a skip."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{label} is still a TODO — implement it, then re-run this test")


def skip_if_placeholder(value, label: str):
    """Skip when a constant is still sitting at its starter placeholder."""
    placeholder = (
        value is None
        or value == "TODO"
        or (isinstance(value, str) and not value.strip())
        or (isinstance(value, (list, tuple, dict, set)) and len(value) == 0)
    )
    if placeholder:
        pytest.skip(f"{label} has not been filled in yet")
    return value


def single_feature_auc(values, y_true) -> float:
    """
    How well does this one column rank members on its own?

    Direction-free: a column that ranks perfectly *backwards* is just as much of
    a leak as one that ranks perfectly forwards, so take the better of the two.
    """
    x = np.asarray(values, dtype=float)
    if np.allclose(x, x[0]):
        return 0.5
    auc = roc_auc_score(np.asarray(y_true, dtype=int), x)
    return max(auc, 1.0 - auc)


class LeakHits(list):
    """
    The (column, auc) pairs from leak_report().

    Which columns leak is the exercise, so no failure names them. A plain list
    would give them away anyway: the tests assert on these hits, and pytest's
    assertion introspection prints the operand's repr, so `assert not hits`
    would render every column name whatever the failure message said.
    """

    def __repr__(self) -> str:
        return f"<{len(self)} column(s) that restate the label — finding them is the exercise>"


def leak_report(df, columns, y_true) -> "LeakHits":
    """Return (column, auc) for every column that ranks members too well."""
    hits = []
    for col in columns:
        if col not in df.columns or not np.issubdtype(df[col].dtype, np.number):
            continue
        auc = single_feature_auc(df[col], y_true)
        if auc >= LEAK_SINGLE_FEATURE_AUC:
            hits.append((col, auc))
    return LeakHits(sorted(hits, key=lambda pair: -pair[1]))


def leak_message(hits: list[tuple[str, float]], where: str) -> str:
    return "\n".join(
        [
            f"{where} contains {len(hits)} column(s) that all but restate the label.",
            "",
            "Which ones is the exercise, so this will not name them.",
            "",
            "A column that separates high-cost members this cleanly by itself is not",
            "predicting the outcome — it IS the outcome, recorded after the fact.",
            "",
            "Ask this of every column in the list, one at a time:",
            "",
            '    "On the day I need this prediction, do I already know this number?"',
            "",
            "To measure it instead of reasoning about it, single_feature_auc() in",
            "tests/helpers.py scores how well one column ranks members on its own.",
            "",
            "Drop the ones that fail and re-run. Your ROC-AUC will go down. That drop",
            "is the real number; the one you had before was fiction.",
        ]
    )
