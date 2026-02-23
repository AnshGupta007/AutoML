"""
Custom exception hierarchy for AutoMLPro.
All domain-specific exceptions inherit from AutoMLProError.
"""


class AutoMLProError(Exception):
    """Base exception for all AutoMLPro errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# ── Data Exceptions ────────────────────────────────────────────────────────────
class DataError(AutoMLProError):
    """Base exception for data-related errors."""


class DataIngestionError(DataError):
    """Raised when data cannot be loaded from the source."""


class DataValidationError(DataError):
    """Raised when data fails schema or quality validation.

    Attributes:
        failed_checks: List of specific validation checks that failed.
    """

    def __init__(self, message: str, failed_checks: list | None = None) -> None:
        self.failed_checks = failed_checks or []
        super().__init__(message, {"failed_checks": self.failed_checks})


class InsufficientDataError(DataError):
    """Raised when dataset has too few samples for training."""

    def __init__(self, n_samples: int, min_required: int) -> None:
        super().__init__(
            f"Insufficient data: got {n_samples} samples, need at least {min_required}",
            {"n_samples": n_samples, "min_required": min_required},
        )


class UnsupportedFormatError(DataError):
    """Raised when an unsupported file format is provided."""

    def __init__(self, fmt: str, supported: list[str]) -> None:
        super().__init__(
            f"Unsupported format '{fmt}'. Supported: {supported}",
            {"format": fmt, "supported_formats": supported},
        )


# ── Feature Exceptions ─────────────────────────────────────────────────────────
class FeatureError(AutoMLProError):
    """Base exception for feature engineering errors."""


class FeatureEngineeringError(FeatureError):
    """Raised when feature generation fails."""


class PipelineFitError(FeatureError):
    """Raised when feature pipeline fails to fit."""


class ColumnNotFoundError(FeatureError):
    """Raised when an expected column is missing from the DataFrame."""

    def __init__(self, column: str, available: list[str]) -> None:
        super().__init__(
            f"Column '{column}' not found. Available: {available[:10]}...",
            {"column": column},
        )


# ── Model Exceptions ───────────────────────────────────────────────────────────
class ModelError(AutoMLProError):
    """Base exception for model-related errors."""


class ModelTrainingError(ModelError):
    """Raised when model training fails."""


class ModelNotFittedError(ModelError):
    """Raised when attempting to predict with an unfitted model."""

    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"Model '{model_name}' has not been fitted yet. Call .fit() first.",
            {"model_name": model_name},
        )


class UnsupportedTaskError(ModelError):
    """Raised when an unsupported ML task type is requested."""

    def __init__(self, task: str, supported: list[str]) -> None:
        super().__init__(
            f"Task '{task}' is not supported. Choose from: {supported}",
            {"task": task, "supported_tasks": supported},
        )


class HyperparameterTuningError(ModelError):
    """Raised when HPO fails."""


# ── Deployment Exceptions ──────────────────────────────────────────────────────
class DeploymentError(AutoMLProError):
    """Base exception for deployment-related errors."""


class ModelSerializationError(DeploymentError):
    """Raised when model serialization or deserialization fails."""


class ModelNotFoundError(DeploymentError):
    """Raised when a requested model artifact doesn't exist."""

    def __init__(self, model_name: str, search_path: str) -> None:
        super().__init__(
            f"Model '{model_name}' not found at '{search_path}'",
            {"model_name": model_name, "search_path": search_path},
        )


class PredictionError(DeploymentError):
    """Raised when inference fails."""


# ── Monitoring Exceptions ──────────────────────────────────────────────────────
class MonitoringError(AutoMLProError):
    """Base exception for monitoring-related errors."""


class DriftDetectionError(MonitoringError):
    """Raised when drift detection computation fails."""


# ── Configuration Exceptions ───────────────────────────────────────────────────
class ConfigurationError(AutoMLProError):
    """Raised when configuration is invalid or missing required values."""


class EnvironmentError(AutoMLProError):  # noqa: A001
    """Raised when required environment variables or system dependencies are missing."""
