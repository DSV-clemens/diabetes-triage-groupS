# Virtual Diabetes Clinic Triage (ML Service)

Small ML service that predicts a short-term **diabetes progression index**
(higher = worse) from standardized patient features, so a triage nurse can
prioritize follow-up calls. Built with FastAPI + scikit-learn, shipped as a
self-contained Docker image, automated with GitHub Actions.

**v0.1 baseline model:** `StandardScaler + LinearRegression`, trained on the
open scikit-learn Diabetes dataset (hold-out split, seed fixed for
reproducibility).

---

## Endpoints

### `GET /health`

```bash
curl http://localhost:80/health
```

```json
{"status": "ok", "model_version": "v0.1"}
```

### `POST /predict`

```bash
curl -X POST http://localhost:80/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001}'
```

```json
{"prediction": 152.3}
```

The 10 features are the standardized columns of the scikit-learn Diabetes
dataset (`age`, `sex`, `bmi`, `bp`, `s1`–`s6`), exactly as returned by
`sklearn.datasets.load_diabetes(as_frame=True)`.

### Error responses (invalid input)

Invalid requests — a missing or wrongly typed feature, malformed JSON —
return **HTTP 422** with a JSON body in FastAPI's standard validation format:

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "s6"],
      "msg": "Field required",
      "input": {"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02},
      "url": "https://errors.pydantic.dev/2.10/v/missing"
    }
  ]
}
```

`detail[].loc` names the offending field (here: missing `s6`).

---

## Run with Docker (recommended)

Released images are publicly available on GHCR:

```bash
docker pull ghcr.io/<namespace>/<repo>:v0.1
docker run -it -p 80:80 ghcr.io/<namespace>/<repo>:v0.1
```

The image is self-contained: the trained model is **baked in at build time**
(training runs inside the build with a fixed seed, so it is reproducible).
The service listens on port **80** inside the container.

## Run locally (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python train.py                       # trains the model + writes metrics/metrics.json
MODEL_VERSION=v0.1 uvicorn app:app --port 80
```

## Development & tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check .        # lint
pytest -q           # unit tests (auto-trains the tiny model if missing)
```

## CI/CD (GitHub Actions)

- **CI** (`.github/workflows/ci.yml`): on every push to `main` and every PR —
  lint (ruff), unit tests (pytest), training smoke run, metrics artifact upload.
- **Release** (`.github/workflows/release.yml`): on pushing a tag `v*` —
  trains the model, builds the Docker image with the model baked in,
  runs container smoke tests (`/health`, `/predict`, 422 on invalid input),
  pushes the image to GHCR and publishes a GitHub Release with metrics.

```bash
git tag v0.1
git push origin v0.1   # triggers the release workflow
```

## Reproducing the training metrics

```bash
python train.py   # deterministic: seed 42, fixed versions in requirements.txt
cat metrics/metrics.json
```

## Project layout

```
app.py                       # FastAPI service (/health, /predict)
train.py                     # training + evaluation (model/, metrics/)
tests/                       # unit tests (pytest + TestClient)
Dockerfile                   # self-contained image (python:3.12-slim, port 80)
.github/workflows/ci.yml     # lint/test/train on PR & push
.github/workflows/release.yml# tag v*: build, smoke test, GHCR, GitHub Release
```
