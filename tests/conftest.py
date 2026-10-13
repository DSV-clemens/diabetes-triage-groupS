"""Pytest fixtures: ensure a trained model exists before API tests run."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("MODEL_VERSION", "test")

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "pipeline.joblib"


def _ensure_model() -> None:
    """Train the (tiny) baseline model if it is not present yet."""
    if MODEL_PATH.exists():
        return
    import train

    train.MODEL_DIR = MODEL_PATH.parent
    train.MODEL_PATH = MODEL_PATH
    train.train()


_ensure_model()
