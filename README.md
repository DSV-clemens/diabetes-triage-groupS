# Virtual Diabetes Clinic Triage (ML Service)

**Assignment 3 — Group S (DSV, Stockholm University)**

Small ML service that predicts a short-term **diabetes progression index**
(higher = worse) from standardized patient features, so a triage nurse can
prioritize follow-up calls. Built with FastAPI + scikit-learn, shipped as a
self-contained Docker image, automated with GitHub Actions.

> **AI usage:** This project was completed with AI support as permitted by
> [DSV's AI policy](https://www.su.se/english/divisions/department-of-computer-and-systems-sciences/education/during-your-studies/dsvs-ai-policy).
> The use of AI tools is declared in [`AI_USAGE.md`](AI_USAGE.md).

## Versions

The service has two released versions. **Both are used the same way** —
identical endpoints, identical request/response formats; they only differ in
the trained model. Each running container reports its version via `GET /health`.

|                            | v0.1 (baseline)                    | v0.2 (improvement)                                  |
| -------------------------- | ---------------------------------- | --------------------------------------------------- |
| Model                      | StandardScaler + LinearRegression  | StandardScaler + Ridge(alpha=1)                     |
| High-risk flag             | —                                  | threshold 170.0 (Precision 0.70 / Recall 0.64)       |
| Test RMSE (hold-out, seed 42) | 53.8534                          | **53.7775**                                         |
| Prediction for the sample payload below | 235.95                | 226.913                                             |

See `CHANGELOG.md` for the full v0.1 → v0.2 comparison, the metric deltas and
the model-selection evidence (repeated cross-validation on the training split).

## Run with Docker (released images)

Both released images are publicly available on GHCR. **GHCR paths are all
lowercase**, so the image name differs from the repository name:

```bash
# v0.1 (baseline)
docker pull ghcr.io/dsv-clemens/diabetes-triage-groups:v0.1
docker run -it -p 80:80 ghcr.io/dsv-clemens/diabetes-triage-groups:v0.1

# v0.2 (improvement) — stop the v0.1 container first (same port)
docker pull ghcr.io/dsv-clemens/diabetes-triage-groups:v0.2
docker run -it -p 80:80 ghcr.io/dsv-clemens/diabetes-triage-groups:v0.2
```

The images are self-contained: the trained model is **baked in at build
time** (training runs inside the build with a fixed seed, so it is
reproducible). The service listens on port **80** inside the container.

## Endpoints (identical in both versions)

### `GET /health`

```bash
curl http://localhost:80/health
```

```json
{"status": "ok", "model_version": "v0.1"}
```

The `model_version` value reflects the running image: `"v0.1"` for the v0.1
image, `"v0.2"` for the v0.2 image.

### `POST /predict`

```bash
curl -X POST http://localhost:80/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001}'
```

The response format is identical for both versions; the score differs because
the models differ:

```json
{"prediction": 235.95}
```

(v0.1 returns `235.95` for the payload above, v0.2 returns `226.913`. Your own
payloads will of course produce other scores.)

The 10 features are the standardized columns of the scikit-learn Diabetes
dataset (`age`, `sex`, `bmi`, `bp`, `s1`–`s6`), exactly as returned by
`sklearn.datasets.load_diabetes(as_frame=True)`.

**v0.2 only — high-risk flag:** v0.2 ships a calibrated triage threshold of
**170.0**. The `/predict` format is unchanged on purpose (it must stay
`{"prediction": <float>}`); the dashboard can flag patients client-side with
`prediction >= 170.0` (Precision 0.70 / Recall 0.64 on the held-out split, see
`CHANGELOG.md`).

### Error responses (invalid input, identical in both versions)

Invalid requests — a missing or wrongly typed feature, malformed JSON —
return **HTTP 422** with a JSON body in FastAPI's standard validation format:

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "s6"],
      "msg": "Field required",
      "input": {"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02}
    }
  ]
}
```

`detail[].loc` names the offending field (here: missing `s6`).

## Run locally (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python train.py                       # trains the model + writes metrics/metrics.json
MODEL_VERSION=v0.2 uvicorn app:app --port 80
```

`train.py` on `main` trains the **v0.2** model (Ridge). To reproduce the
metrics of a specific version locally, check out its tag first:

```bash
git checkout v0.1 && python train.py   # metrics.json: RMSE 53.8534
git checkout v0.2 && python train.py   # metrics.json: RMSE 53.7775 (+ flag metrics)
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
git tag v0.2
git push origin v0.2   # triggers the release workflow
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
