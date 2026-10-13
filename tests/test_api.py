"""Unit tests for the FastAPI endpoints (/health, /predict, error handling)."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

import app as app_module
from train import METRICS_PATH

VALID_PAYLOAD = {
    "age": 0.02,
    "sex": -0.044,
    "bmi": 0.06,
    "bp": -0.03,
    "s1": -0.02,
    "s2": 0.03,
    "s3": -0.02,
    "s4": 0.02,
    "s5": 0.02,
    "s6": -0.001,
}


def make_client() -> TestClient:
    return TestClient(app_module.app)


def test_health_returns_ok_and_version():
    with make_client() as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "model_version" in body


def test_predict_returns_numeric_prediction():
    with make_client() as client:
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"prediction"}
        assert isinstance(body["prediction"], float)


def test_predict_is_deterministic():
    with make_client() as client:
        first = client.post("/predict", json=VALID_PAYLOAD).json()
        second = client.post("/predict", json=VALID_PAYLOAD).json()
        assert first == second


def test_missing_feature_returns_422_with_json_error():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "s6"}
    with make_client() as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        body = response.json()
        assert "detail" in body  # documented error structure (see README)
        assert "s6" in json.dumps(body)  # error names the missing feature


def test_invalid_type_returns_422():
    payload = dict(VALID_PAYLOAD, bmi="not-a-number")
    with make_client() as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        assert "detail" in response.json()


def test_training_writes_metrics():
    assert METRICS_PATH.exists(), "train.py must produce metrics/metrics.json"
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert metrics["rmse"] > 0
    assert metrics["random_seed"] == 42
    assert metrics["model_version"] in {"dev", "test", "v0.1", "v0.2"}
