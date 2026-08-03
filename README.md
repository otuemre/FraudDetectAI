# FraudDetectAI

[![Run Tests](https://github.com/otuemre/FraudDetectAI/actions/workflows/tests.yml/badge.svg)](https://github.com/otuemre/FraudDetectAI/actions/workflows/tests.yml)

A production-oriented credit card fraud detection system — from raw transaction data to a served, monitored model. Built to reflect how a real ML system is engineered, not just how a model is trained: SQL-backed data pipelines, tuned XGBoost with a class-imbalance strategy comparison, unsupervised baselines, SHAP explainability, a FastAPI serving layer, structured logging, drift/latency monitoring, and CI-tested components.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Explainability](#explainability)
- [Monitoring](#monitoring)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Usage](#api-usage)
- [Testing & CI/CD](#testing--cicd)
- [Design Decisions Worth Noting](#design-decisions-worth-noting)
- [Future Work](#future-work)

## Overview

FraudDetectAI classifies credit card transactions as fraudulent or legitimate using the well-known [ULB Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 0.17% fraud rate, PCA-anonymized features `v1`–`v28`).

The project's goal wasn't just "train a good classifier" — it's a full pipeline: data lives in Postgres, gets ingested and split reproducibly, feeds a tuned XGBoost model compared against three alternative strategies, gets served via a REST API, and is monitored in production-realistic ways (without peeking at ground truth it wouldn't have in real life).

## Architecture

```
Postgres (raw transactions)
        │
        ▼
 data_ingestion  →  dedup + stratified split  →  training_data / testing_data (SQL)
        │
        ▼
 preprocessing   →  hour_of_day engineering + RobustScaler (fit on train only)
        │
        ▼
 hyperparameter_tuning (Optuna, PR-AUC objective)
        │
        ▼
 XGBoost (final fit)  →  evaluation  →  model + scaler persisted to disk
        │
        ▼
   FastAPI (/predict, /health)
        │
        ▼
 prediction_logs (Postgres)  →  monitoring (confidence bands, PSI drift, latency)
```

## Modeling Approach

**Why RobustScaler, not StandardScaler:** `v1`–`v28` arrive already PCA-transformed and roughly standardized. Only `amount` and the engineered `hour_of_day` needed scaling — and `amount` is heavily right-skewed with real outliers in both classes, so a scaler built on median/IQR (RobustScaler) is far less distorted by extreme values than one built on mean/standard deviation.

**Feature engineering:** raw `time` (seconds since first transaction) is only meaningful as a cyclical day/night signal, not as a raw offset. It's converted to `hour_of_day` (0–24), which EDA showed fraud disproportionately concentrated in overnight hours relative to legitimate traffic.

**Class imbalance — four strategies compared** (5-fold stratified CV on the training set):

| Model | Accuracy | Precision | Recall | PR-AUC |
|---|---|---|---|---|
| Logistic Regression (class_weight) | 0.9743 | 0.0566 | 0.9126 | 0.7520 |
| Logistic Regression + SMOTE | 0.9902 | 0.1408 | 0.8889 | 0.7563 |
| XGBoost (class_weight) | 0.9996 | **0.9058** | 0.8280 | 0.8489 |
| XGBoost + SMOTE | 0.9995 | 0.8404 | 0.8359 | 0.8575 |

> Logistic regression hit sklearn's `lbfgs` convergence limit (`max_iter` reached without converging) in both variants — its precision numbers above should be read as directional, not a fully-converged baseline. Even accounting for that, the qualitative conclusion holds: at this precision level (~1 correct fraud flag per 7–18 alerts), logistic regression isn't viable for a real alerting system regardless of convergence.

XGBoost with class-weighting was selected over SMOTE — comparable PR-AUC, but meaningfully better precision and a tighter cross-validation variance, which matters more for an alerting system than a marginal PR-AUC gain.

**Hyperparameter tuning:** Optuna (TPE sampler, 50 trials, 5-fold stratified CV, optimizing average precision) tuned `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`, and `scale_pos_weight` (searched around the theoretical class-ratio value rather than fixing it), reaching a best CV PR-AUC of 0.8526.

## Results

Final tuned XGBoost, evaluated once on the held-out test set (never touched during CV or tuning):

| Metric | Value |
|---|---|
| Precision (Fraud) | 0.9157 |
| Recall (Fraud) | 0.8000 |
| F1 (Fraud) | 0.8539 |
| PR-AUC | 0.8143 |
| ROC-AUC | 0.9799 |

Test-set PR-AUC (0.8143) sits a bit below the best CV PR-AUC (0.8526) — expected, since CV reports an average across folds while the test set is a single held-out sample; the gap isn't a sign of a problem, just normal variance between a multi-fold estimate and one final held-out check.

**Unsupervised comparison** (phase 2, trained without fraud labels):

| Model | Precision | Recall | F1 (Fraud) |
|---|---|---|---|
| XGBoost (supervised) | 0.893 | 0.789 | 0.838 |
| AutoEncoder (reconstruction error) | 0.47 | 0.47 | 0.47 |
| Isolation Forest | 0.224 | 0.232 | 0.228 |

The AutoEncoder (trained only on legitimate transactions, flagged by reconstruction error) roughly doubles Isolation Forest's F1 by learning a joint representation of "normal" across all 30 features, rather than isolating on individual splits — but both unsupervised approaches trail the supervised model significantly, since neither ever sees an actual fraud example during training.

## Explainability

SHAP (`TreeExplainer`) confirms the model's top drivers largely match EDA's correlation analysis (`v14`, `v12`, `v10`, `v17`) — but also surfaces `v4` as a top-2 global driver despite weak linear correlation with the target, indicating XGBoost is capturing nonlinear/interaction effects that simple correlation can't see. Individual force plots trace exactly which features pushed a specific transaction toward a fraud classification, useful for case-level review.

## Monitoring

Monitoring is built under a real production constraint: **no ground-truth labels are available at prediction time** (real fraud labels arrive weeks later via chargebacks/investigation, if at all). Every monitoring signal here is derived without peeking at true labels:

- **Confidence-band flagging** — predictions with `fraud_probability` between 0.4–0.6 are surfaced as low-confidence, a candidate signal for routing to human review rather than full automation.
- **PSI (Population Stability Index) drift detection** — compares each feature's distribution in recent live predictions against the original training distribution (quantile-binned, standard 0.1/0.25 watch/alert thresholds), catching silent data drift before it manifests as degraded performance.
- **Latency monitoring** — p50/p95/p99/max prediction latency, aggregated hourly, tracked from the same `prediction_logs` table.

## Project Structure

```
src/
  exception.py           # CustomException — structured error context (type, file, function, line)
  logger.py               # JSON logging: per-run training logs, daily-rotated error logs
  utils.py                 # model/scaler save-load, generic SQL read/write
  config.py               # Centralized config loader (config/config.yaml)
  db/
    get_engine.py          # Single SQLAlchemy engine, shared across the codebase
    create_log_table.py    # prediction_logs schema
  components/
    data_ingestion.py       # dedup + stratified split, persisted to SQL
    preprocessing.py        # hour_of_day engineering, RobustScaler fit/apply
    hyperparameter_tuning.py # Optuna study (PR-AUC objective)
    evaluation.py            # metrics computation, JSON-serializable
  pipelines/
    train_pipeline.py       # orchestrates ingestion -> preprocessing -> tuning -> fit -> eval -> save
    predict_pipeline.py      # single-transaction inference, logs to prediction_logs
api/
  main.py                   # FastAPI: /predict, /health
scripts/
  simulate_stream.py        # replays test set through the live API, one transaction at a time
  drift_monitoring.py        # PSI report
  latency_monitoring.py      # latency percentile report
  monitoring.py               # confidence-band + daily volume report
tests/
  test_preprocessing.py
  test_exception.py
  test_evaluation.py
  test_api.py
config/
  config.yaml               # all tunable thresholds, filenames, hyperparameter search ranges
.github/workflows/
  tests.yml                 # CI: pytest on every push/PR to master
```

## Getting Started

```bash
git clone https://github.com/otuemre/FraudDetectAI.git
cd FraudDetectAI

cp .env.example .env  # fill in Postgres credentials

docker compose up -d  # starts Postgres

pip install -r requirements.txt
```

Run the training pipeline:
```bash
python -m src.pipelines.train_pipeline
```

Start the API:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Usage

**Health check:**
```bash
curl http://localhost:8000/health
```

**Predict:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"time": 40000, "v1": -1.35, "v2": -0.07, ..., "v28": -0.02, "amount": 149.62}'
```
Interactive docs available at `http://localhost:8000/docs` (FastAPI auto-generated Swagger UI).

## Testing & CI/CD

```bash
pytest tests/ -v
```

Tests cover pure logic (`preprocessing`, `exception`) and the API contract (`test_api.py`, via mocked prediction calls — no live DB or model file required in CI) and metric computation (`evaluation`, via a fake model with known outputs). Deliberately **not** unit-tested: SQL-dependent components (`data_ingestion`, `monitoring`, `drift_monitoring`) and Optuna's search itself — these require either a live database or aren't meaningfully verifiable via unit tests; a larger deployment would add integration tests against a test database.

GitHub Actions runs the full suite on every push/PR to `master` (`.github/workflows/tests.yml`).

## Design Decisions Worth Noting

- **Config vs. environment variables:** `config.yaml` holds *behavioral* settings (thresholds, hyperparameter ranges, table/file names) that might reasonably change between runs. `.env` holds *secrets/environment-specific* values (DB credentials). Structural paths (`models/`, `data/`) are neither — they're fixed by the codebase's own layout and live directly in code.
- **`prediction_logs` stores the full feature vector**, including raw `amount`/`time`-derived fields — deliberately not privacy-safe for a real deployment with genuine customer data, but appropriate here since all "live" traffic is a replay of a held-out test set with no real new data involved. A production system would encrypt or minimize this.
- **CustomException + structured JSON logging**, wrapping every component's I/O boundary, was built specifically to make debugging traceable (exact file/function/line/error type) in JSON logs rather than parsing raw tracebacks — and it caught a real bug during test-writing (an error-type field that was silently logging the wrong value).
- **Isolation Forest and the AutoEncoder are evaluated with knowledge of the true fraud rate** (via `contamination`/threshold quantile), which is a simplification a truly blind unsupervised deployment wouldn't have. Noted here rather than hidden.

## Future Work

- Delayed ground-truth reconciliation (simulating chargebacks arriving weeks later, to compute real retrospective accuracy)
- Containerize the API alongside Postgres in `docker-compose.yml`
- Deploy via AWS App Runner
- Threshold tuning based on business cost trade-offs (precision/recall operating point), rather than a fixed 0.5 default

## License

See [LICENSE.md](LICENSE.md).