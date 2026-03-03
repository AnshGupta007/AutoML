# 🤖 AutoML Pipeline — Production-Ready End-to-End ML System

[![CI](https://github.com/your-org/automl-project/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/automl-project/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **production-ready, end-to-end automated ML pipeline** that takes a raw CSV file as input and produces a deployed, monitored REST API as output — with zero manual feature engineering.

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)
- [Monitoring](#monitoring)
- [CI/CD](#cicd)

---

## 🏗 Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ CSV/S3/GCS  │───▶│  Ingest &   │───▶│  Feature    │───▶│  Training   │
│   Input     │    │  Validate   │    │ Engineering │    │ (HPO + CV)  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                      Pandera           SmartImputer         Optuna
                      EDA Report        SmartEncoder         MLflow
                      DriftDetect       FeatureSelector      Stacking

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Monitoring │◀───│  FastAPI    │◀───│  Deploy     │◀───│  Evaluate   │
│ + Alerting  │    │   REST API  │    │ ONNX+joblib │    │ SHAP+Fair   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
  Evidently AI       /predict           Docker             Calibration
  Prometheus         /batch             K8s ready          Error Analysis
  Grafana            /metrics           CI/CD
```

---

## 📁 Project Structure

```
automl-project/
├── configs/                # Hydra YAML configs
│   ├── default.yaml        # Main config with stage composition
│   ├── data/default.yaml   # Data source + validation thresholds
│   ├── model/              # XGBoost, LightGBM, CatBoost, RF, MLP configs
│   ├── training/           # HPO, CV, ensemble settings
│   ├── evaluation/         # SHAP, fairness, calibration
│   ├── deployment/         # API host/port/workers
│   └── monitoring/         # Drift thresholds, alerting
├── data/
│   ├── raw/                # Raw input CSVs
│   ├── processed/          # Cleaned reference dataset (DVC tracked)
│   └── features/           # Versioned feature store (Parquet)
├── src/
│   ├── ingestion/          # DataLoader, DataValidator, EDA, DriftDetector
│   ├── features/           # TypeDetector, Imputer, Encoder, Selector, Pipeline, Store
│   ├── training/           # TaskDetector, ModelZoo, OptunaHPO, CrossValidator,
│   │                       # StackingEnsemble, BlendingEnsemble, Trainer, Serializer
│   ├── evaluation/         # MetricsCalculator, SHAPExplainer, FairnessAnalyzer,
│   │                       # ErrorAnalyzer, ConfidenceCalibrator, EvaluationReporter
│   ├── deployment/         # FastAPI app, Pydantic schemas, BatchPredictor
│   ├── monitoring/         # ProductionDriftDetector, ModelPerformanceMonitor,
│   │                       # RetrainingTrigger, AlertManager
│   └── orchestration/      # Prefect DAG, pipeline stages
├── tests/
│   ├── unit/               # 48+ unit tests
│   └── integration/        # API integration tests
├── docker/
│   ├── Dockerfile          # Multi-stage build
│   ├── docker-compose.yml  # API + MLflow + Prometheus + Grafana
│   └── prometheus.yml      # Scrape config
├── .github/workflows/
│   ├── ci.yml              # Lint → Test → Docker Build
│   └── cd.yml              # Staging → Smoke Test → Production
├── run_pipeline.py         # Hydra CLI entrypoint
├── dvc.yaml                # DVC reproducibility pipeline
├── requirements.txt        # All Python dependencies
└── environment.yml         # Conda environment
```

---

## ⚡ Quick Start

### 1. Setup Environment

```bash
# Using conda (recommended)
conda env create -f environment.yml
conda activate automl-pipeline

# Or using pip
pip install -r requirements.txt
```

### 2. Configure Your Data

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Place your CSV at `data/raw/sample.csv`, or override via CLI:

```bash
python run_pipeline.py data.path="data/raw/my_dataset.csv" data.target_column="label"
```

### 3. Run the Full Pipeline

```bash
# Full end-to-end run (default: XGBoost + 30 HPO trials)
python run_pipeline.py

# Override model and HPO settings
python run_pipeline.py model=lightgbm training.n_trials=50

# Run with multi-model ensemble
python run_pipeline.py training.ensemble.enabled=true training.ensemble.method=stacking
```

### 4. Start the API

```bash
# Development mode (auto-reload)
python run_pipeline.py +stage=serve deployment.reload=true

# Or directly with uvicorn
uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Make Predictions

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature_0": 1.5, "feature_1": -0.3}, "return_proba": true}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -F "file=@new_data.csv"

# Health check
curl http://localhost:8000/health
```

---

## 🔧 Pipeline Stages

| Stage | Key Classes | What it does |
|---|---|---|
| **Ingestion** | `DataLoader`, `DataValidator` | Load CSV/S3/GCS, validate schema, run EDA |
| **Features** | `FeaturePipeline`, `FeatureStore` | Auto-detect types, impute, encode, select |
| **Training** | `OptunaHPO`, `Trainer` | HPO (30 trials), 5-fold CV, ensemble |
| **Evaluation** | `EvaluationReporter` | SHAP, fairness, calibration, error analysis |
| **Deployment** | `FastAPI`, `BatchPredictor` | REST API + ONNX inference + batch jobs |
| **Monitoring** | `ProductionDriftDetector` | Evidently AI drift reports, Prometheus |

### Individual Stage Runs

```bash
python run_pipeline.py +stage=ingest    # Data ingestion only
python run_pipeline.py +stage=train     # Training only (requires features)
python run_pipeline.py +stage=monitor   # Drift + performance check
python run_pipeline.py +stage=serve     # Start API server
```

---

## ⚙ Configuration

All config is managed by Hydra. Override any parameter from the CLI:

```bash
# Change model
python run_pipeline.py model=catboost

# Increase HPO budget
python run_pipeline.py training.n_trials=100 training.timeout=7200

# Use time-series cross-validation
python run_pipeline.py training.use_time_series_split=true

# Enable fairness analysis
python run_pipeline.py evaluation.fairness.enabled=true \
  "evaluation.fairness.protected_columns=[gender,age_group]"

# Change monitoring thresholds
python run_pipeline.py monitoring.drift.drift_threshold=0.15
```

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/model/info` | Version, metrics, feature names |
| `POST` | `/predict` | Single prediction + SHAP (optional) |
| `POST` | `/predict/batch` | CSV file upload → batch predictions |
| `POST` | `/explain` | SHAP feature contributions |
| `GET` | `/metrics` | Prometheus metrics endpoint |
| `GET` | `/docs` | Swagger UI interactive docs |

---

## 🧪 Testing

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ --cov=src --cov-report=html

# Integration tests (API)
pytest tests/integration/ -v -m integration

# Skip slow tests
pytest tests/ -m "not slow"
```

---

## 🐳 Docker Deployment

```bash
# Build and start all services (API + MLflow + Prometheus + Grafana)
cd docker/
docker-compose up --build

# Services:
# - API:        http://localhost:8000
# - MLflow:     http://localhost:5000
# - Prometheus: http://localhost:9090
# - Grafana:    http://localhost:3000 (admin/admin)
```

---

## 📊 Monitoring

After training, monitoring runs automatically on each API call:

- **Feature Drift**: Evidently AI + PSI fallback compares live requests against the training reference dataset
- **Performance Degradation**: Rolling accuracy check vs. training baseline
- **Automated Retraining**: Configurable trigger when drift or performance thresholds are exceeded

```bash
# Run standalone monitoring check
python run_pipeline.py +stage=monitor
```

---

## 🔄 CI/CD

### GitHub Actions Workflow

1. **CI** (on every push/PR): Lint (Black + isort + Flake8) → Unit tests + coverage → Docker build
2. **CD** (on `main` push): Build image → Deploy to **staging** → Smoke tests → Promote to **production** → Slack notification

### Required GitHub Secrets

```
STAGING_HOST, STAGING_USER, STAGING_SSH_KEY
PROD_HOST, PROD_USER, PROD_SSH_KEY
SLACK_WEBHOOK_URL
```

---

## 🔒 Security

- No hardcoded credentials — use `.env` (excluded from git) and GitHub Secrets
- Non-root Docker user (`appuser`)
- CORS middleware with configurable origins
- Input validation via Pydantic schemas

---

## 📦 Supported Models

| Model | Classification | Regression |
|---|---|---|
| XGBoost | ✅ | ✅ |
| LightGBM | ✅ | ✅ |
| CatBoost | ✅ | ✅ |
| Random Forest | ✅ | ✅ |
| Logistic/Linear Regression | ✅ | ✅ |
| MLP (scikit-learn) | ✅ | ✅ |
| Stacking Ensemble | ✅ | ✅ |
| Blending Ensemble | ✅ | ✅ |

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| Config Management | Hydra |
| Data Versioning | DVC |
| Experiment Tracking | MLflow |
| Hyperparameter Optimization | Optuna (TPE) |
| Drift Detection | Evidently AI |
| Explainability | SHAP |
| Fairness | Fairlearn |
| Orchestration | Prefect |
| API Framework | FastAPI + Uvicorn |
| Inference Format | ONNX + joblib |
| Monitoring | Prometheus + Grafana |
| Alerting | Slack + SMTP |
| CI/CD | GitHub Actions |
| Containerization | Docker |

---

## 📄 License

MIT © 2026
