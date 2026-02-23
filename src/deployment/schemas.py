"""
FastAPI request/response schemas using Pydantic.
"""
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Prediction schemas
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """Single-record prediction request."""
    features: dict[str, Any] = Field(..., description="Feature name → value mapping")
    model_id: Optional[str] = Field(None, description="Model version ID to use")
    return_proba: bool = Field(False, description="Return class probabilities")

    class Config:
        json_schema_extra = {
            "example": {
                "features": {"age": 35, "income": 65000, "credit_score": 720},
                "return_proba": True,
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response."""
    prediction: Any
    probability: Optional[list[float]] = None
    model_id: Optional[str] = None
    latency_ms: Optional[float] = None


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    records: list[dict[str, Any]] = Field(..., description="List of feature dicts")
    model_id: Optional[str] = None
    return_proba: bool = False


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: list[Any]
    probabilities: Optional[list[list[float]]] = None
    count: int
    latency_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# Training schemas
# ---------------------------------------------------------------------------

class TrainingRequest(BaseModel):
    """Trigger a new training run."""
    dataset_path: str = Field(..., description="Path to CSV dataset")
    target_column: str = Field(..., description="Name of target column")
    task_type: str = Field("auto", description="classification | regression | auto")
    experiment_name: str = Field("automl_experiment")
    enable_hpo: bool = True
    enable_ensemble: bool = True
    n_trials: int = Field(30, ge=1, le=500)


class TrainingResponse(BaseModel):
    """Training trigger response."""
    run_id: str
    status: str
    message: str


# ---------------------------------------------------------------------------
# Health / status schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    task_type: str
    feature_names: list[str]
    metrics: dict[str, float]
    trained_at: Optional[str] = None
