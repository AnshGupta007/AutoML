"""
Main orchestration pipeline.
Composes all steps into a runnable end-to-end flow.
"""
from typing import Optional

from src.utils.logger import get_logger

log = get_logger(__name__)

try:
    from prefect import flow
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    def flow(fn=None, **kwargs):
        if fn is not None:
            return fn
        def decorator(f):
            return f
        return decorator


@flow(name="automl_pipeline", log_prints=True)
def automl_pipeline(
    data_path: str,
    target_column: str,
    task_type: str = "auto",
    experiment_name: str = "automl",
    enable_hpo: bool = True,
    enable_ensemble: bool = True,
    n_trials: int = 30,
    output_dir: str = "reports",
) -> dict:
    """End-to-end AutoML pipeline flow.

    Steps:
        1. Ingest → 2. Validate → 3. Profile → 4. Split →
        5. Feature Engineering → 6. Train → 7. Evaluate → 8. Report

    Args:
        data_path: Path to input CSV file.
        target_column: Name of the target column.
        task_type: 'classification' | 'regression' | 'auto'.
        experiment_name: MLflow experiment name.
        enable_hpo: Whether to run Optuna HPO.
        enable_ensemble: Whether to build stacking ensemble.
        n_trials: Number of HPO trials.
        output_dir: Directory for reports and artifacts.

    Returns:
        Dict with training results, metrics, and report path.
    """
    from src.orchestration.steps import (
        engineer_features,
        evaluate_model,
        generate_report,
        ingest_data,
        profile_data,
        split_data,
        train_model,
        validate_data,
    )

    log.info(f"[START] AutoML Pipeline starting: {data_path} | target={target_column}")

    # Step 1: Ingest
    ingestion_result = ingest_data(data_path)

    # Step 2: Validate
    data_result = validate_data(ingestion_result, target_column)

    # Step 3: Profile
    profile_data(data_result, output_dir=output_dir)

    # Step 4: Split
    strategy = "stratified" if task_type != "regression" else "random"
    splits = split_data(data_result, target_column, strategy=strategy)

    # Step 5: Feature Engineering
    feature_splits = engineer_features(splits, task_type=task_type)

    # Step 6: Train
    training_result = train_model(
        feature_splits,
        task_type=task_type,
        experiment_name=experiment_name,
        enable_hpo=enable_hpo,
        enable_ensemble=enable_ensemble,
        n_trials=n_trials,
    )

    # Step 7: Evaluate
    eval_result = evaluate_model(training_result, feature_splits)

    # Step 8: Report
    report_path = generate_report(eval_result, training_result, output_dir=output_dir)

    log.info(f"[DONE] Pipeline complete! Report: {report_path}")
    log.info(f"Test metrics: {eval_result['metrics']}")

    return {
        "training_result": training_result,
        "eval_result": eval_result,
        "report_path": report_path,
    }
