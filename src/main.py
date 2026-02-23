"""
Main entry point for AutoMLPro.
Supports three modes: train, serve, and monitor.
"""
import argparse
import sys
from pathlib import Path

from src.utils.logger import get_logger

log = get_logger("automl_pro")


def run_train(args: argparse.Namespace) -> None:
    """Run the end-to-end AutoML training pipeline."""
    from src.orchestration.pipeline import automl_pipeline

    log.info("Mode: TRAIN")
    result = automl_pipeline(
        data_path=args.data,
        target_column=args.target,
        task_type=args.task_type,
        experiment_name=args.experiment,
        enable_hpo=not args.no_hpo,
        enable_ensemble=not args.no_ensemble,
        n_trials=args.n_trials,
        output_dir=args.output_dir,
    )
    print("\n" + "=" * 60)
    print("[DONE] Training complete!")
    print(f"   Metrics: {result['eval_result']['metrics']}")
    print(f"   Report:  {result['report_path']}")
    print("=" * 60)


def run_serve(args: argparse.Namespace) -> None:
    """Launch the FastAPI prediction server."""
    import uvicorn

    log.info(f"Mode: SERVE on {args.host}:{args.port}")
    uvicorn.run(
        "src.deployment.api:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


def run_monitor(args: argparse.Namespace) -> None:
    """Run a one-shot monitoring check."""
    import joblib
    import pandas as pd

    from src.monitoring.data_drift import DataDriftDetector
    from src.monitoring.model_performance import ModelPerformanceMonitor
    from src.monitoring.retraining_trigger import RetrainingTrigger

    log.info("Mode: MONITOR")
    models_dir = Path(args.models_dir)

    # Load reference + current
    reference_df = pd.read_csv(args.reference)
    current_df = pd.read_csv(args.current)

    # Drift detection
    detector = DataDriftDetector(output_dir=args.output_dir)
    detector.set_reference(reference_df)
    drift_report = detector.detect(current_df)

    # Performance check (if labels provided)
    perf_report = None
    if args.labels:
        labels = pd.read_csv(args.labels)
        model = joblib.load(models_dir / "best_model.joblib")
        pipeline = joblib.load(models_dir / "feature_pipeline.joblib")
        current_features = pipeline.transform(current_df)
        y_pred = model.predict(current_features)
        y_true = labels.iloc[:, 0]
        monitor = ModelPerformanceMonitor(task_type=args.task_type)
        monitor.log_predictions(y_true, y_pred)
        perf_report = monitor.check_degradation()

    # Trigger decision
    trigger = RetrainingTrigger()
    decision = trigger.evaluate(drift_report, perf_report)

    print("\n" + "=" * 60)
    print(f"Drift detected: {drift_report['drift_detected']} (share={drift_report['drift_share']:.1%})")
    if perf_report:
        print(f"Performance degraded: {perf_report.get('degraded')}")
    print(f"Retrain needed: {decision['should_retrain']}")
    for r in decision["reasons"]:
        print(f"  Reason: {r}")
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="automl-pro",
        description="AutoMLPro — Intelligent Automated Machine Learning Pipeline",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # Train subcommand
    train_p = sub.add_parser("train", help="Run the full AutoML training pipeline")
    train_p.add_argument("--data", required=True, help="Path to input CSV file")
    train_p.add_argument("--target", required=True, help="Target column name")
    train_p.add_argument("--task-type", default="auto",
                         choices=["auto", "classification", "regression"])
    train_p.add_argument("--experiment", default="automl_experiment")
    train_p.add_argument("--n-trials", type=int, default=30)
    train_p.add_argument("--no-hpo", action="store_true")
    train_p.add_argument("--no-ensemble", action="store_true")
    train_p.add_argument("--output-dir", default="reports")

    # Serve subcommand
    serve_p = sub.add_parser("serve", help="Launch prediction API server")
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--workers", type=int, default=2)

    # Monitor subcommand
    monitor_p = sub.add_parser("monitor", help="Run monitoring checks")
    monitor_p.add_argument("--reference", required=True, help="Path to reference CSV")
    monitor_p.add_argument("--current", required=True, help="Path to current batch CSV")
    monitor_p.add_argument("--labels", default=None, help="Path to true labels CSV")
    monitor_p.add_argument("--task-type", default="classification")
    monitor_p.add_argument("--models-dir", default="models")
    monitor_p.add_argument("--output-dir", default="reports/monitoring")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "train": run_train,
        "serve": run_serve,
        "monitor": run_monitor,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
