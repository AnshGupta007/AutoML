# PROJECT: AutoMLPro

## Project Overview
**AutoMLPro** is a comprehensive, production-ready Automated Machine Learning (AutoML) framework designed to handle the end-to-end ML lifecycle—from data ingestion and feature engineering to automated model selection, HPO, and deployment as a REST API.

---

## Core Objectives
- **End-to-End Automation**: Automate the entire pipeline from raw CSV to a deployed FastAPI endpoint.
- **Robust Feature Engineering**: Implement intelligent imputation, encoding, and semantic type detection.
- **Model Optimization**: Use fast benchmarking followed by deep Optuna-based Hyperparameter Optimization (HPO).
- **Explainable AI**: Integrate SHAP/LIME for model transparency.
- **Production Grade**: Ensure modularity, high test coverage, and clear separation between training, orchestration, and serving.

---

## Technological Stack
- **Languages**: Python 3.10+
- **Core ML**: Scikit-learn, XGBoost, LightGBM, CatBoost
- **Data Handling**: Pandas, NumPy
- **Orchestration**: Prefect
- **HPO**: Optuna
- **Serving**: FastAPI, Uvicorn
- **Tracking**: MLflow
- **Aesthetics/UI**: Streamlit (Dashboard)
- **Deployment**: Docker, Docker Compose

---

## Technical Constraints & Considerations
- **Environment**: Must run on Windows (using Conda/Pip) and Linux (Docker).
- **Dependencies**: Handle version conflicts (e.g., numpy 2.x, sklearn experimental guards).
- **Performance**: Support chunked loading for large datasets and efficient HPO trials.

---

## Known Context & Requirements
- Target: Tabular data (Classification/Regression).
- Output: HTML reports and persistent model artifacts.
- API: RESTful interface with Pydantic validation.

---

## Source of Truth Gaps
> [!NOTE]
> To finalize this PROJECT.md, the following details are requested from the user:
> 1. Are there specific performance SLAs for the API (e.g., latency under 50ms)?
> 2. Should we prioritize "Time to Model" or "Best Possible Performance" in default settings?
> 3. Are there specific fairness or bias detection requirements beyond the current SHAP integration?
