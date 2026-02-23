# AutoMLPro 🤖
### Intelligent Automated Machine Learning Pipeline

[![CI](https://github.com/yourusername/automl-pro/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/automl-pro/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](Dockerfile)

> **Production-ready AutoML pipeline for tabular data.**  
> Drop in a CSV → get a trained, evaluated, explained, deployed, and monitored model — all automatically.

---

## ✨ Features

| Capability | Details |
|---|---|
| **Data** | CSV/Parquet/Excel/JSON ingestion, validation, EDA profiling, DVC versioning |
| **Features** | Type detection, KNN/iterative imputation, target/WoE/binary encoding, datetime & text features, polynomial generation, MI/LASSO selection |
| **Models** | XGBoost, LightGBM, CatBoost, Ridge/Logistic, PyTorch MLP |
| **HPO** | Optuna TPE sampler + Hyperband pruning |
| **Ensemble** | Stacking (OOF) + blending |
| **Evaluation** | 10+ metrics, SHAP + LIME explanations, fairness (Fairlearn), calibration, HTML report |
| **Deployment** | FastAPI REST API with single/batch predict, model hot-reload |
| **Monitoring** | Evidently data drift, rolling performance tracking, Slack/email/PagerDuty alerts, auto-retrain trigger |
| **Orchestration** | Prefect flows (with plain-Python fallback) |
| **MLflow** | Full experiment tracking, artifact logging |
| **CI/CD** | GitHub Actions: lint + unit + integration tests + Docker build + publish |

---

## 🚀 Quick Start

### 1. Clone & install
```bash
git clone https://github.com/yourusername/automl-pro.git
cd automl-pro
pip install -e ".[dev]"
```

### 2. Train on your CSV
```bash
python -m src.main train \
  --data data/raw/your_data.csv \
  --target label_column \
  --task-type classification \
  --n-trials 50
```

### 3. Serve the prediction API
```bash
python -m src.main serve --host 0.0.0.0 --port 8000
# API docs: http://localhost:8000/docs
```

### 4. One-shot monitoring check
```bash
python -m src.main monitor \
  --reference data/raw/train.csv \
  --current data/raw/new_batch.csv
```

---

## 🐳 Docker (recommended)

```bash
# Start everything (API + Dashboard + MLflow + PostgreSQL + Redis)
docker-compose up -d

# API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"age": 35, "income": 65000}}'
```

| Service | URL |
|---|---|
| Prediction API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Streamlit Dashboard | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |

---

## 📁 Project Structure

```
automl-pro/
├── src/
│   ├── data/          # Ingestion, validation, profiling, splitting
│   ├── features/      # Type detection, imputation, encoding, scaling, pipelines
│   ├── models/        # XGBoost, LightGBM, CatBoost, Linear, Neural Net, Ensemble
│   ├── evaluation/    # Metrics, cross-validation, SHAP, fairness, calibration
│   ├── deployment/    # FastAPI, schemas, middleware, batch predictor
│   ├── monitoring/    # Data drift, performance, alerting, retrain trigger
│   ├── orchestration/ # Prefect steps + pipeline
│   └── main.py        # CLI entry point
├── configs/           # Hydra YAML configs
├── tests/             # Unit + integration tests
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## 🧪 Tests

```bash
# Unit tests with coverage
pytest tests/unit/ --cov=src

# Integration tests
pytest tests/integration/ -m integration

# All tests
pytest tests/
```

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and adjust settings:
```bash
cp .env.example .env
```

Key config overrides (via Hydra or env vars):
- `MODEL_CANDIDATES` — comma-separated list: `xgboost,lightgbm,catboost`
- `ENABLE_HPO` — `true/false`
- `N_TRIALS` — number of Optuna HPO trials
- `MLFLOW_TRACKING_URI` — MLflow server URL

---

## 📄 License

MIT — see [LICENSE](LICENSE).
