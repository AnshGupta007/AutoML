"""
FastAPI application for the AutoMLPro prediction API.
Provides endpoints for single/batch prediction, model info, training triggers, and health.
"""
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.deployment.middleware import RequestLoggingMiddleware
from src.deployment.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    TrainingRequest,
    TrainingResponse,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

_model = None
_feature_pipeline = None
_model_metadata: dict = {}
_models_dir = Path(os.getenv("MODELS_DIR", "models"))

APP_VERSION = "1.0.0"


def _load_model_artifacts() -> None:
    """Load model and pipeline from disk at startup."""
    global _model, _feature_pipeline, _model_metadata

    model_path = _models_dir / "best_model.joblib"
    pipeline_path = _models_dir / "feature_pipeline.joblib"
    meta_path = _models_dir / "metadata.json"

    if model_path.exists():
        _model = joblib.load(model_path)
        log.info(f"Model loaded from {model_path}")
    else:
        log.warning(f"No model found at {model_path}. Train first.")

    if pipeline_path.exists():
        _feature_pipeline = joblib.load(pipeline_path)
        log.info("Feature pipeline loaded.")

    if meta_path.exists():
        import json
        with open(meta_path) as f:
            _model_metadata = json.load(f)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model_artifacts()
    yield
    log.info("API shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AutoMLPro API",
    description="Production-ready AutoML REST API — train, predict, explain.",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """API health check."""
    return HealthResponse(
        status="healthy",
        version=APP_VERSION,
        model_loaded=_model is not None,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Return metadata about the currently loaded model."""
    if _model is None:
        raise HTTPException(status_code=404, detail="No model loaded")

    feature_names = getattr(_model, "_feature_names", [])
    if not feature_names and _feature_pipeline is not None:
        feature_names = getattr(_feature_pipeline, "_feature_names_out", [])

    return ModelInfoResponse(
        model_name=getattr(_model, "name", "unknown"),
        task_type=getattr(_model, "task_type", "unknown"),
        feature_names=feature_names,
        metrics=_model_metadata.get("metrics", {}),
        trained_at=_model_metadata.get("trained_at"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """Single-record prediction."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    try:
        features_df = pd.DataFrame([request.features])
        if _feature_pipeline is not None:
            features_df = _feature_pipeline.transform(features_df)

        prediction = _model.predict(features_df)[0]
        proba = None
        if request.return_proba:
            try:
                proba = _model.predict_proba(features_df)[0].tolist()
            except Exception:
                pass

        return PredictionResponse(
            prediction=_to_json(prediction),
            probability=proba,
            model_id=_model_metadata.get("run_id"),
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
    except Exception as e:
        log.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction for multiple records."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    try:
        df = pd.DataFrame(request.records)
        if _feature_pipeline is not None:
            df = _feature_pipeline.transform(df)

        predictions = _model.predict(df).tolist()
        probas = None
        if request.return_proba:
            try:
                probas = _model.predict_proba(df).tolist()
            except Exception:
                pass

        return BatchPredictionResponse(
            predictions=[_to_json(p) for p in predictions],
            probabilities=probas,
            count=len(predictions),
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
    except Exception as e:
        log.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", response_model=TrainingResponse, tags=["Training"])
async def trigger_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Trigger a new training run in the background."""
    import uuid
    run_id = str(uuid.uuid4())[:8]

    async def _run_training():
        try:
            from src.data.ingestion import DataIngestion
            from src.data.splitter import DataSplitter
            from src.models.trainer import Trainer

            log.info(f"[{run_id}] Starting training: {request.dataset_path}")
            ingestion = DataIngestion()
            result = ingestion.load(request.dataset_path)
            df = result["data"]

            splitter = DataSplitter(strategy="stratified" if request.task_type != "regression" else "random")
            splits = splitter.split(df, target_column=request.target_column)

            trainer = Trainer(
                task_type=request.task_type,
                experiment_name=request.experiment_name,
                enable_hpo=request.enable_hpo,
                enable_ensemble=request.enable_ensemble,
                n_trials=request.n_trials,
            )
            trainer.train(splits["X_train"], splits["y_train"], splits["X_val"], splits["y_val"])

            # Reload model
            _load_model_artifacts()
            log.info(f"[{run_id}] Training complete.")
        except Exception as e:
            log.error(f"[{run_id}] Training failed: {e}")

    background_tasks.add_task(_run_training)
    return TrainingResponse(
        run_id=run_id,
        status="started",
        message=f"Training job {run_id} started in background.",
    )


@app.post("/model/reload", tags=["Model"])
async def reload_model():
    """Reload model artifacts from disk."""
    _load_model_artifacts()
    return {"status": "reloaded", "model_loaded": _model is not None}


def _to_json(value):
    """Convert numpy/pandas scalar to JSON-serializable Python type."""
    try:
        return value.item()
    except AttributeError:
        return value
