"""
AutoML Pipeline — Hydra CLI Entrypoint
=========================================
Entry point for the end-to-end ML pipeline.
Uses Hydra for config management — override any parameter via CLI:

    python run_pipeline.py model=xgboost data.test_size=0.2 training.n_trials=50

Run individual stages:
    python run_pipeline.py +stage=ingest     # Data ingestion only
    python run_pipeline.py +stage=train      # Training only (requires features)
    python run_pipeline.py +stage=serve      # Start FastAPI server
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import hydra  # type: ignore
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def _setup_logging(cfg: DictConfig) -> None:
    """Configure loguru with level and file rotation."""
    log_cfg = cfg.get("logging", {})
    log_level = log_cfg.get("level", "INFO")
    log_file = log_cfg.get("file", "logs/pipeline.log")
    rotation = log_cfg.get("rotation", "100 MB")
    retention = log_cfg.get("retention", "30 days")

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=log_level, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    logger.add(log_file, level="DEBUG", rotation=rotation, retention=retention,
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}")


def _load_env() -> None:
    """Load .env file if present."""
    env_path = Path(".env")
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)
        logger.debug(".env file loaded")


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> Optional[dict]:
    """
    Main pipeline entrypoint.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config (merged from configs/default.yaml + CLI overrides).

    Returns
    -------
    dict or None
    """
    _load_env()
    _setup_logging(cfg)

    # Print resolved config for reproducibility
    logger.info("Pipeline config:\n" + OmegaConf.to_yaml(cfg))

    # Check for single-stage run
    stage = cfg.get("stage", None)

    if stage == "serve":
        return _serve(cfg)
    elif stage == "ingest":
        return _run_ingest_only(cfg)
    elif stage == "train":
        return _run_train_only(cfg)
    elif stage == "monitor":
        return _run_monitor_only(cfg)
    else:
        # Full pipeline run
        return _run_full_pipeline(cfg)


def _run_full_pipeline(cfg: DictConfig) -> dict:
    """Execute the complete pipeline via Prefect or direct execution."""
    logger.info("Starting full AutoML pipeline")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    try:
        from src.orchestration.dag import run_pipeline_flow
        result = run_pipeline_flow(config=cfg_dict)
        logger.success("Pipeline completed successfully!")
        return result
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


def _serve(cfg: DictConfig) -> None:
    """Start the FastAPI server."""
    import uvicorn

    deploy_cfg = cfg.get("deployment", {})
    host = deploy_cfg.get("host", "0.0.0.0")
    port = int(deploy_cfg.get("port", 8000))
    reload = deploy_cfg.get("reload", False)

    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run("src.deployment.api:app", host=host, port=port, reload=reload)


def _run_ingest_only(cfg: DictConfig) -> dict:
    """Run only the data ingestion stage."""
    from src.ingestion.loader import DataLoader
    from src.ingestion.validator import DataValidator

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    loader = DataLoader()
    df, meta = loader.load(data_cfg["path"])
    validator = DataValidator()
    result = validator.validate(df, raise_on_error=False)
    logger.info(f"Ingestion complete: {df.shape}, passed={result.passed}")
    return {"shape": df.shape, "validation": result.as_dict()}


def _run_train_only(cfg: DictConfig) -> dict:
    """Run only the training stage using stored features."""
    from src.features.store import FeatureStore

    store = FeatureStore()
    X_train, X_test, feat_meta = store.load("latest")
    # Load targets from parquet sidecar
    import pandas as pd
    version = store.list_versions()[-1]
    y_train = pd.read_parquet(f"data/features/{version}/y_train.parquet")["target"]
    y_test = pd.read_parquet(f"data/features/{version}/y_test.parquet")["target"]

    from src.orchestration.stages import train_stage
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    result = train_stage(X_train, y_train, X_test, y_test, config=cfg_dict)
    logger.info(f"Training complete: run_id={result['run_id']}")
    return result


def _run_monitor_only(cfg: DictConfig) -> dict:
    """Run drift detection and performance monitoring."""
    from src.monitoring.drift_detector import ProductionDriftDetector
    from src.monitoring.performance_monitor import ModelPerformanceMonitor

    drift_detector = ProductionDriftDetector()
    drift_report = drift_detector.run_report()

    perf_monitor = ModelPerformanceMonitor()
    perf_report = perf_monitor.check_performance()

    from src.monitoring.retraining_trigger import RetrainingTrigger
    trigger = RetrainingTrigger(
        auto_trigger=cfg.monitoring.retraining.get("auto_trigger", False)
    )
    trigger_result = trigger.evaluate(drift_report=drift_report, perf_report=perf_report)

    logger.info(f"Monitor run complete. Should retrain: {trigger_result['should_retrain']}")
    return {"drift": drift_report, "performance": perf_report, "trigger": trigger_result}


if __name__ == "__main__":
    main()
