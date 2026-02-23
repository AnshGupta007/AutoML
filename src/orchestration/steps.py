"""
Orchestration pipeline steps.
Implements Prefect (or fallback plain-Python) task steps for the AutoML pipeline.
"""
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

# Try to import Prefect; fall back to plain decorators if not installed
try:
    from prefect import task
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False

    def task(fn=None, **kwargs):
        """No-op task decorator fallback."""
        if fn is not None:
            return fn
        def decorator(f):
            return f
        return decorator


@task(name="ingest_data", retries=3, retry_delay_seconds=5)
def ingest_data(data_path: str) -> dict:
    """Load data from file. Returns dict with 'data' (DataFrame) and 'metadata' keys."""
    from src.data.ingestion import DataIngestion
    ingestion = DataIngestion()
    # DataIngestion.load() returns a (DataFrame, metadata_dict) tuple
    df, metadata = ingestion.load(data_path)
    log.info(f"Data ingested: {df.shape}, columns={list(df.columns[:5])}...")
    return {"data": df, "metadata": metadata}


@task(name="validate_data")
def validate_data(ingestion_result: dict, target_column: str) -> dict:
    """Validate data quality and auto-fix issues."""
    from src.data.validation import DataValidator
    df = ingestion_result["data"]
    validator = DataValidator()
    validation_result = validator.validate(df, target_column=target_column)
    log.info(f"Data validation issues: {len(validation_result.get('issues', []))}")
    return {"data": df, "validation_report": validation_result}


@task(name="profile_data")
def profile_data(data_result: dict, output_dir: str = "reports") -> Optional[str]:
    """Generate EDA profile report. Non-critical — errors are logged but don't fail the pipeline."""
    try:
        from src.data.profiling import DataProfiler
        df = data_result["data"]
        profiler = DataProfiler(output_dir=output_dir, enable_full_report=False)
        json_stats = profiler.profile(df, name="dataset")
        log.info("EDA profiling complete")
        return str(json_stats)
    except Exception as exc:
        log.warning(f"EDA profiling failed (non-critical): {exc}")
        return None


@task(name="split_data")
def split_data(data_result: dict, target_column: str, strategy: str = "stratified") -> dict:
    """Split data into train/val/test."""
    from src.data.splitter import DataSplitter
    df = data_result["data"]
    splitter = DataSplitter(strategy=strategy)
    splits = splitter.split(df, target_column=target_column)
    log.info(f"Train size: {len(splits['X_train'])}, Val: {len(splits['X_val'])}, Test: {len(splits['X_test'])}")
    return splits


@task(name="engineer_features")
def engineer_features(splits: dict, task_type: str = "auto") -> dict:
    """Apply feature engineering pipeline."""
    from src.features.pipeline import FeatureEngineeringPipeline

    pipeline = FeatureEngineeringPipeline(task_type=task_type)
    X_train_eng = pipeline.fit_transform(splits["X_train"], splits["y_train"])
    X_val_eng = pipeline.transform(splits["X_val"])
    X_test_eng = pipeline.transform(splits["X_test"])

    log.info(f"Feature engineering: {X_train_eng.shape[1]} features")
    return {
        "X_train": X_train_eng,
        "X_val": X_val_eng,
        "X_test": X_test_eng,
        "y_train": splits["y_train"],
        "y_val": splits["y_val"],
        "y_test": splits["y_test"],
        "feature_pipeline": pipeline,
    }


@task(name="train_model")
def train_model(
    feature_splits: dict,
    task_type: str = "auto",
    experiment_name: str = "automl",
    enable_hpo: bool = True,
    enable_ensemble: bool = True,
    n_trials: int = 30,
) -> dict:
    """Train the AutoML model stack."""
    from src.models.trainer import Trainer

    trainer = Trainer(
        task_type=task_type,
        experiment_name=experiment_name,
        enable_hpo=enable_hpo,
        enable_ensemble=enable_ensemble,
        n_trials=n_trials,
    )
    result = trainer.train(
        feature_splits["X_train"],
        feature_splits["y_train"],
        feature_splits["X_val"],
        feature_splits["y_val"],
        feature_pipeline=feature_splits["feature_pipeline"],
    )
    return result


@task(name="evaluate_model")
def evaluate_model(training_result: dict, feature_splits: dict) -> dict:
    """Evaluate model on held-out test set."""
    from src.evaluation.metrics import compute_metrics
    from src.evaluation.explainability import ModelExplainer

    model = training_result["best_model"]
    X_test = feature_splits["X_test"]
    y_test = feature_splits["y_test"]

    y_pred = model.predict(X_test)
    y_proba = None
    try:
        y_proba = model.predict_proba(X_test)
    except Exception:
        pass

    task_type = getattr(model, "task_type", "classification")
    metrics = compute_metrics(y_test, y_pred, task_type, y_proba)
    log.info(f"Test metrics: {metrics}")

    # SHAP explanations
    explainer = ModelExplainer(model, task_type=task_type)
    explainer.fit(feature_splits["X_train"])
    fi = explainer.get_global_importance(X_test.head(200))

    return {
        "metrics": metrics,
        "feature_importances": fi.to_dict() if fi is not None else {},
        "run_id": training_result.get("run_id"),
    }


@task(name="generate_report")
def generate_report(
    eval_result: dict,
    training_result: dict,
    output_dir: str = "reports",
) -> str:
    """Generate HTML evaluation report."""
    from src.evaluation.report_generator import ReportGenerator
    import pandas as pd

    reporter = ReportGenerator(output_dir=output_dir)
    fi_series = None
    fi_dict = eval_result.get("feature_importances", {})
    if fi_dict:
        fi_series = pd.Series(fi_dict).sort_values(ascending=False)

    path = reporter.generate(
        metrics=eval_result["metrics"],
        model_name=training_result.get("model_name", "unknown"),
        feature_importances=fi_series,
    )
    return str(path)
