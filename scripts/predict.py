"""
predict.py — run batch predictions from command line.

Usage:
    python scripts/predict.py --input data/new.csv --output predictions.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
from src.deployment.batch_predictor import BatchPredictor


def main():
    parser = argparse.ArgumentParser(description="Run batch predictions")
    parser.add_argument("--input", required=True, help="Path to input CSV or Parquet")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--proba", action="store_true", help="Include probability columns")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    model = joblib.load(models_dir / "best_model.joblib")
    pipeline = None
    pipeline_path = models_dir / "feature_pipeline.joblib"
    if pipeline_path.exists():
        pipeline = joblib.load(pipeline_path)

    predictor = BatchPredictor(
        model, pipeline,
        chunk_size=args.chunk_size,
        return_proba=args.proba,
    )
    result = predictor.predict_from_file(args.input, args.output)
    print(f"✅ Predictions saved: {len(result):,} rows → {args.output}")


if __name__ == "__main__":
    main()
