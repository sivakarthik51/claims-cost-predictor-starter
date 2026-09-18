"""
WEEK 7 — the API contract.

Two kinds of test here. The first kind checks that bad input is rejected before
it reaches the model — a validation layer that lets nonsense through is worse
than none, because the nonsense comes back as a confident probability. The
second kind checks that the API and the model agree about the same member,
which is the test that catches train/serve skew.

Every test runs against the app in app/main.py, using a member drawn from your
own dataset as the request body.
"""

import numpy as np
import pandas as pd
import pytest

from helpers import load_module

from config import HIGH_COST_THRESHOLD_USD

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    TestClient = None

@pytest.fixture
def client(model_features):
    if TestClient is None:
        pytest.skip("pip install -r requirements-dev.txt to get the FastAPI test client")

    module = load_module("app.main", "app")
    schema = getattr(module, "MemberFeatures", None)
    if schema is None or not schema.model_fields:
        pytest.skip("app/schemas.py: MemberFeatures has no fields yet")

    missing = sorted(set(model_features) - set(schema.model_fields))
    extra = sorted(set(schema.model_fields) - set(model_features))
    if missing or extra:
        pytest.fail(
            "your request schema and your model disagree about the features.\n"
            f"  in model.joblib but not in MemberFeatures: {missing or 'none'}\n"
            f"  in MemberFeatures but not in model.joblib: {extra or 'none'}\n\n"
            "The schema has to describe exactly what the model was trained on — "
            "one field per feature, same names. If you changed your feature list "
            "in train.py, retrain and update the schema to match.\n"
            "To see the list your model expects:\n"
            "  python -c \"import joblib; "
            "print(list(joblib.load('model.joblib')['preprocessor'].feature_names_in_))\""
        )

    try:
        with TestClient(module.app) as test_client:
            yield test_client
    except NotImplementedError:
        pytest.skip("app/main.py: the startup handler is still a TODO")


def _as_json(row) -> dict:
    """One row of the feature table as a JSON-ready request body."""
    body = {}
    for key, value in row.items():
        if isinstance(value, (np.bool_, bool)):
            body[key] = bool(value)
        elif isinstance(value, (np.integer, int)):
            body[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            body[key] = float(value)
        else:
            body[key] = str(value)
    return body


@pytest.fixture
def payload(summary, model_features):
    """A real member from your dataset, as a JSON-ready request body."""
    return _as_json(summary[model_features].iloc[0])


@pytest.fixture
def payloads(summary, model, model_features):
    """
    Members spanning the risk range: the highest-scoring, the lowest, and some
    from the middle. One member is not enough to catch a bug that only shows up
    at certain feature values.
    """
    scored = summary.assign(_p=model.predict_proba(summary[model_features])[:, 1])
    ordered = scored.sort_values("_p")
    picks = pd.concat([ordered.head(3), ordered.iloc[len(ordered) // 2 : len(ordered) // 2 + 3], ordered.tail(3)])
    return [_as_json(row) for _, row in picks[model_features].iterrows()]


def test_health_reports_whether_the_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body.get("status") == "ok"
    assert body.get("model_loaded") is True, (
        "/health says the model is not loaded. Render uses this endpoint to "
        "decide whether your service is healthy, so it has to be truthful."
    )


def test_predict_returns_the_documented_shape(client, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"is_high_cost", "probability", "threshold_dollars"}
    assert isinstance(body["is_high_cost"], bool)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["threshold_dollars"] == HIGH_COST_THRESHOLD_USD, (
        "echo the active threshold. A probability with no threshold attached "
        "cannot be interpreted by the caller."
    )


def test_api_and_model_agree_on_every_member(client, payloads, model, model_features):
    """
    The train/serve skew test.

    For each member, the API and a direct model call are looking at the same
    features, so they must return the same probability. When they don't, the row
    your endpoint hands the model is not the row you think it is — and nothing
    raises an error anywhere. You simply get a different answer in production
    than the one in your notebook.

    Things this catches: reading the wrong column out of predict_proba,
    transforming units or scaling inside the endpoint (the pipeline already does
    it), reordering or renaming columns, rounding before thresholding.
    """
    for payload in payloads:
        api_probability = client.post("/predict", json=payload).json()["probability"]
        direct = float(model.predict_proba(pd.DataFrame([payload])[model_features])[0, 1])
        assert api_probability == pytest.approx(direct, abs=5e-4), (
            f"the API says {api_probability:.4f}, the model says {direct:.4f} for "
            f"this member:\n{payload}\n\n"
            "The endpoint should hand the pipeline the features exactly as they "
            "came in — no scaling, no unit conversion, no reordering. The "
            "pipeline was fitted with its preprocessing inside it."
        )


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ({"region": "Atlantis"}, "a region that does not exist in training"),
        ({"total_claims": -3}, "a negative claim count"),
    ],
)
def test_invalid_values_are_rejected_with_422(client, payload, mutation, reason):
    key = next(iter(mutation))
    if key not in payload:
        pytest.skip(f"{key} is not a feature of your model")
    response = client.post("/predict", json={**payload, **mutation})
    assert response.status_code == 422, (
        f"sent {reason} and got {response.status_code}. The schema should refuse "
        "this before the model ever sees it — a nonsense input that reaches the "
        "model comes back as a confident probability."
    )


def test_missing_field_is_rejected_with_422(client, payload):
    incomplete = dict(payload)
    incomplete.pop(next(iter(incomplete)))
    assert client.post("/predict", json=incomplete).status_code == 422


def test_openapi_docs_describe_the_endpoint(client):
    """The auto-generated contract is how a caller learns to use this service."""
    spec = client.get("/openapi.json").json()
    assert "/predict" in spec["paths"], "POST /predict is missing from the OpenAPI spec"
    assert "/health" in spec["paths"]
