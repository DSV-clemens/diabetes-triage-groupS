"""Training script for the Virtual Diabetes Clinic Triage service.

Trains the diabetes progression model, evaluates it on a held-out split
and stores the fitted pipeline plus evaluation metrics on disk.

Usage:
    python train.py            # trains with MODEL_VERSION env (default: dev)
    MODEL_VERSION=v0.1 python train.py
"""

import json
import os
from pathlib import Path

import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
TEST_SIZE = 0.2

MODEL_DIR = Path("model")
METRICS_DIR = Path("metrics")
MODEL_PATH = MODEL_DIR / "pipeline.joblib"
METRICS_PATH = METRICS_DIR / "metrics.json"

# v0.1 baseline: StandardScaler + LinearRegression.
MODEL_DESCRIPTION = "StandardScaler + LinearRegression"


def train() -> dict:
    """Load the diabetes dataset, fit the v0.1 baseline pipeline and save it."""
    dataset = load_diabetes(as_frame=True)
    X = dataset.frame.drop(columns=["target"])
    y = dataset.frame["target"]  # progression index (higher = worse)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5

    metrics = {
        "model_version": os.environ.get("MODEL_VERSION", "dev"),
        "model": MODEL_DESCRIPTION,
        "rmse": round(float(rmse), 4),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "test_size": TEST_SIZE,
        "random_seed": RANDOM_SEED,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()
