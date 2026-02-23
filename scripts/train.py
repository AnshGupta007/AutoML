"""
train.py — convenience wrapper script to launch a quick training run from CLI.

Usage:
    python scripts/train.py --data data/raw/my_data.csv --target label
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestration.pipeline import automl_pipeline


def main():
    parser = argparse.ArgumentParser(description="Train AutoMLPro model")
    parser.add_argument("--data", required=True, help="Path to CSV file")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--task-type", default="auto",
                        choices=["auto", "classification", "regression"])
    parser.add_argument("--experiment", default="automl")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--no-hpo", action="store_true")
    parser.add_argument("--no-ensemble", action="store_true")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

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

    print(f"\n✅ Training complete!")
    print(f"   Metrics:  {result['eval_result']['metrics']}")
    print(f"   Report:   {result['report_path']}")


if __name__ == "__main__":
    main()
