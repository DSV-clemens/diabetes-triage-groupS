"""Training script for the Virtual Diabetes Clinic Triage service.

Trains the diabetes progression model, evaluates it on a held-out split
and stores the fitted pipeline plus evaluation metrics on disk.

v0.2 model: StandardScaler + Ridge(alpha=1) - selected via repeated
cross-validation on the training split only (see CHANGELOG.md). Also
calibrates a high-risk flag threshold on the training targets.

Usage:
    python train.py            # trains with MODEL_VERSION env (default: dev)
    MODEL_VERSION=v0.2 python train.py
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
TEST_SIZE = 0.2

# High-risk flag: threshold calibrated on TRAIN targets only (60th percentile,
# F1-optimal in train cross-validation - no test leakage). See CHANGELOG.md v0.2.
FLAG_THRESHOLD_PERCENTILE = 60

# v0.1 baseline RMSE on the identical split & seed, for the improvement delta.
RMSE_V01_REFERENCE = 53.8534

MODEL_DIR = Path("model")
METRICS_DIR = Path("metrics")
MODEL_PATH = MODEL_DIR / "pipeline.joblib"
METRICS_PATH = METRICS_DIR / "metrics.json"

MODEL_DESCRIPTION = "StandardScaler + Ridge(alpha=1)"


def train() -> dict:
    """Load the diabetes dataset, fit the v0.2 pipeline and save it + metrics."""
    dataset = load_diabetes(as_frame=True)
    X = dataset.frame.drop(columns=["target"])
    y = dataset.frame["target"]  # progression index (higher = worse)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_test_np = y_test.to_numpy()
    predictions = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test_np, predictions) ** 0.5

    # High-risk flag: threshold from TRAIN targets, metrics on the test split.
    flag_threshold = float(np.percentile(y_train, FLAG_THRESHOLD_PERCENTILE))
    y_flag = y_test_np >= flag_threshold
    p_flag = predictions >= flag_threshold
    tp = int((y_flag & p_flag).sum())
    fp = int((~y_flag & p_flag).sum())
    fn = int((y_flag & ~p_flag).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    metrics = {
        "model_version": os.environ.get("MODEL_VERSION", "dev"),
        "model": MODEL_DESCRIPTION,
        "rmse": round(float(rmse), 4),
        "rmse_v0_1_reference": RMSE_V01_REFERENCE,
        "rmse_delta_vs_v0_1": round(float(rmse) - RMSE_V01_REFERENCE, 4),
        "flag_threshold": round(flag_threshold, 1),
        "flag_threshold_percentile": FLAG_THRESHOLD_PERCENTILE,
        "flag_precision": round(precision, 3),
        "flag_recall": round(recall, 3),
        "flag_f1": round(f1, 3),
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
