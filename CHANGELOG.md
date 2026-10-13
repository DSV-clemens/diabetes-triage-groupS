# Changelog

All notable changes to the Virtual Diabetes Clinic Triage service are
documented here. Metrics are RMSE on the held-out test split
(20% of the diabetes dataset, `random_state=42`).

## v0.2

Improvement release: better regression model + calibrated high-risk flag.

### What changed

- **Model:** `StandardScaler + LinearRegression` → `StandardScaler + Ridge(alpha=1)`.
- **New:** calibrated **high-risk flag threshold** at **170.0** (60th percentile
  of the training targets, F1-optimal in train cross-validation). The `/predict`
  response format is unchanged (`{"prediction": <float>}`); the dashboard can
  flag patients client-side with `prediction >= 170.0`.
- **Internal:** `/predict` now builds the input as a DataFrame with feature
  names, matching how the pipeline was fitted (removes sklearn feature-name
  warnings from the server logs).
- No infrastructure changes — same endpoints, same Docker layout.

### Why (model selection evidence)

Candidates were compared honestly via **repeated 5-fold CV (10 repeats) on the
training split only** — the test split was never used for selection:

| Candidate                     | CV-RMSE (train) | Test-RMSE |
| ----------------------------- | --------------: | --------: |
| v0.1 LinearRegression         |          55.143 |    53.853 |
| **Ridge(alpha=1) — chosen**    |      **55.129** | **53.778** |
| Lasso(alpha=0.1)               |          55.149 |    53.709 |
| Poly2+Ridge(alpha=100)         |          57.611 |    51.567 |
| RandomForest(d4, n=300)        |          57.960 |    53.058 |
| GradientBoosting(lr=.05, d2)   |          58.509 |    52.872 |

Polynomial features, RandomForest and GradientBoosting achieve better test
scores on this particular split but **clearly overfit** in cross-validation
(worse CV-RMSE than the linear baseline) — selecting them by test score would
be test-set leakage. Ridge(alpha=1) is the only candidate with a consistent,
if small, CV improvement; the diabetes dataset is fundamentally near-linear.

### Metrics (side-by-side)

| Metric                        | v0.1     | v0.2     | Delta   |
| ----------------------------- | -------: | -------: | ------: |
| Test RMSE                     | 53.8534  | **53.7775** | -0.076 (-0.14%) |
| High-risk Precision @170      | —        | 0.700    | new     |
| High-risk Recall @170         | —        | 0.636    | new     |
| High-risk F1 @170             | —        | 0.667    | new     |

The accuracy gain is small — deliberately. The tested alternatives either
overfit (see table above) or leak test information. v0.2 is the best *honest*
model, and adds a clinically usable, calibrated triage threshold on top.

## v0.1

Baseline release.

- **Model:** `StandardScaler + LinearRegression` pipeline on the scikit-learn
  Diabetes dataset, trained on an 80/20 hold-out split (`random_state=42`).
- **Service:** FastAPI with `POST /predict` (numeric progression index) and
  `GET /health` (status + model version), JSON 422 validation errors.
- **Packaging:** self-contained Docker image (`python:3.12-slim`), model baked
  in at build time, service on port 80, container healthcheck.
- **Automation:** CI workflow (lint, tests, training smoke, artifacts) and
  release workflow (build, container smoke tests, GHCR, GitHub Release).

**Metrics (v0.1):** RMSE = **53.8534** on the held-out test split
(89 samples, `random_state=42`, scikit-learn 1.6.0, see `metrics/metrics.json`).

## Unreleased

- Initial project scaffold.
