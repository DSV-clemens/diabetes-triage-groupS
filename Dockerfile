# Virtual Diabetes Clinic Triage - service image
# Self-contained: the trained model is baked in at build time
# (deterministic training with fixed seed, no artifacts needed in git).

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install pinned dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the service and training code.
COPY app.py train.py ./

# Bake in the model version (set by the release workflow, e.g. v0.1).
ARG MODEL_VERSION=dev
ENV MODEL_VERSION=${MODEL_VERSION}

# Train at build time: same code + fixed seed -> reproducible image.
RUN python train.py

EXPOSE 80

# Lightweight container healthcheck via Python (no curl needed in-slim image).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:80/health').read()" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
