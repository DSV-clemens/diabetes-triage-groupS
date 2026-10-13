"""FastAPI service for the Virtual Diabetes Clinic Triage.

Endpoints:
    GET  /health   -> {"status": "ok", "model_version": "<version>"}
    POST /predict  -> {"prediction": <float>}   (422 + JSON error on bad input)
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "model/pipeline.joblib"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "dev")

FEATURE_ORDER = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

pipeline = None


def load_model() -> None:
    """Load the trained pipeline from disk (baked into the Docker image)."""
    global pipeline
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"No trained model found at {MODEL_PATH}. Run `python train.py` first "
            "or use the released Docker image, which bakes the model in."
        )
    pipeline = joblib.load(MODEL_PATH)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()  # fail fast at startup instead of on the first request
    yield


app = FastAPI(
    title="Virtual Diabetes Clinic Triage",
    description="Predicts a short-term diabetes progression index for triage.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


class PatientFeatures(BaseModel):
    """Patient check-in features (already standardized, as in the sklearn dataset)."""

    age: float = Field(..., description="Standardized age")
    sex: float = Field(..., description="Standardized sex")
    bmi: float = Field(..., description="Standardized body mass index")
    bp: float = Field(..., description="Standardized average blood pressure")
    s1: float = Field(..., description="Standardized serum measurement s1 (tc)")
    s2: float = Field(..., description="Standardized serum measurement s2 (ldl)")
    s3: float = Field(..., description="Standardized serum measurement s3 (hdl)")
    s4: float = Field(..., description="Standardized serum measurement s4 (tch)")
    s5: float = Field(..., description="Standardized serum measurement s5 (ltg)")
    s6: float = Field(..., description="Standardized serum measurement s6 (glu)")


@app.get("/health")
def health() -> dict:
    """Liveness/readiness probe for internal monitoring."""
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
def predict(features: PatientFeatures) -> dict:
    """Return the predicted progression index (higher = worse)."""
    # DataFrame with feature names: matches how the pipeline was fitted
    # (avoids sklearn feature-name warnings in the server logs).
    row = pd.DataFrame(
        [[getattr(features, name) for name in FEATURE_ORDER]],
        columns=FEATURE_ORDER,
    )
    prediction = float(pipeline.predict(row)[0])
    return {"prediction": round(prediction, 3)}
