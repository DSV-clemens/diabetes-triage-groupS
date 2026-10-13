# Changelog

All notable changes to the Virtual Diabetes Clinic Triage service are
documented here. Metrics are RMSE on the held-out test split
(20% of the diabetes dataset, `random_state=42`).

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
