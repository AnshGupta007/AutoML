"""
Batch predictor — efficient offline batch inference on large datasets.
"""
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.utils.logger import get_logger
from src.utils.timer import Timer

log = get_logger(__name__)


class BatchPredictor:
    """Runs batch inference on a CSV/Parquet file and writes predictions.

    Processes data in chunks to handle large files without OOM.

    Example:
        >>> predictor = BatchPredictor(model, feature_pipeline, chunk_size=10000)
        >>> predictor.predict_from_file("data/new_data.csv", "output/predictions.csv")
    """

    def __init__(
        self,
        model,
        feature_pipeline=None,
        chunk_size: int = 10_000,
        return_proba: bool = False,
    ) -> None:
        self.model = model
        self.feature_pipeline = feature_pipeline
        self.chunk_size = chunk_size
        self.return_proba = return_proba

    def predict_from_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> pd.DataFrame:
        """Run batch predictions on an input file.

        Args:
            input_path: Path to CSV or Parquet input.
            output_path: Path to write predictions CSV.

        Returns:
            Full predictions DataFrame.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        log.info(f"Batch prediction: {input_path} → {output_path}")

        if input_path.suffix == ".parquet":
            df = pd.read_parquet(input_path)
            chunks = [df]
        else:
            chunks = pd.read_csv(input_path, chunksize=self.chunk_size)

        all_preds = []
        with Timer("batch prediction"):
            for i, chunk in enumerate(chunks):
                log.debug(f"Processing chunk {i + 1}...")
                chunk_preds = self._predict_chunk(chunk)
                all_preds.append(chunk_preds)

        result = pd.concat(all_preds, ignore_index=True)
        result.to_csv(output_path, index=False)
        log.info(f"Predictions saved: {len(result)} rows → {output_path}")
        return result

    def _predict_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply pipeline + model to a single chunk."""
        original_index = df.index.copy()

        if self.feature_pipeline is not None:
            df_transformed = self.feature_pipeline.transform(df)
        else:
            df_transformed = df

        predictions = self.model.predict(df_transformed)
        result = pd.DataFrame({"prediction": predictions}, index=original_index)

        if self.return_proba:
            try:
                probas = self.model.predict_proba(df_transformed)
                for cls_idx in range(probas.shape[1]):
                    result[f"proba_class_{cls_idx}"] = probas[:, cls_idx]
            except NotImplementedError:
                pass

        return result
