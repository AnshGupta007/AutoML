"""
Feature store management.
Provides save/load/list operations for feature pipeline artifacts
and transformed datasets with metadata tracking.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import joblib
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


class FeatureStore:
    """Lightweight feature store for managing feature artifacts.

    Stores fitted feature pipelines, feature statistics, and
    transformed datasets indexed by experiment name and version.

    Example:
        >>> store = FeatureStore("artifacts/feature_pipelines")
        >>> store.save_pipeline(pipeline, name="titanic_v1")
        >>> pipeline = store.load_pipeline("titanic_v1")
    """

    def __init__(self, base_dir: Union[str, Path] = "artifacts/feature_pipelines") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_pipeline(
        self,
        pipeline,
        name: str,
        metadata: Optional[dict] = None,
    ) -> Path:
        """Save a fitted feature pipeline.

        Args:
            pipeline: Fitted FeatureEngineeringPipeline.
            name: Unique identifier for this pipeline.
            metadata: Optional metadata dict to store alongside.

        Returns:
            Path to saved pipeline file.
        """
        artifact_dir = self.base_dir / name
        artifact_dir.mkdir(parents=True, exist_ok=True)

        pipeline_path = artifact_dir / "pipeline.joblib"
        joblib.dump(pipeline, pipeline_path, compress=3)

        meta = {
            "name": name,
            "saved_at": datetime.utcnow().isoformat(),
            "feature_names": getattr(pipeline, "_feature_names_out", []),
            **(metadata or {}),
        }
        with open(artifact_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        log.info(f"Feature pipeline '{name}' saved at {pipeline_path}")
        return pipeline_path

    def load_pipeline(self, name: str):
        """Load a fitted feature pipeline.

        Args:
            name: Pipeline identifier.

        Returns:
            Loaded pipeline object.

        Raises:
            FileNotFoundError: If pipeline not found.
        """
        pipeline_path = self.base_dir / name / "pipeline.joblib"
        if not pipeline_path.exists():
            raise FileNotFoundError(
                f"Feature pipeline '{name}' not found at {pipeline_path}"
            )
        pipeline = joblib.load(pipeline_path)
        log.info(f"Feature pipeline '{name}' loaded from {pipeline_path}")
        return pipeline

    def save_features(
        self,
        X: pd.DataFrame,
        split: str,
        name: str,
    ) -> Path:
        """Save a transformed feature DataFrame as Parquet.

        Args:
            X: Transformed feature DataFrame.
            split: Split name (train, val, test).
            name: Experiment/pipeline name.

        Returns:
            Path to saved Parquet file.
        """
        artifact_dir = self.base_dir / name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / f"{split}_features.parquet"
        X.to_parquet(path, index=False)
        log.info(f"Features saved: {split} / {name} → {path}")
        return path

    def load_features(self, name: str, split: str) -> pd.DataFrame:
        """Load saved feature DataFrame.

        Args:
            name: Experiment name.
            split: Split name (train, val, test).

        Returns:
            Feature DataFrame.
        """
        path = self.base_dir / name / f"{split}_features.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Features for '{name}/{split}' not found at {path}")
        return pd.read_parquet(path)

    def list_pipelines(self) -> list[dict]:
        """List all stored pipelines and their metadata.

        Returns:
            List of metadata dicts.
        """
        pipelines = []
        for d in sorted(self.base_dir.iterdir()):
            meta_path = d / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    pipelines.append(json.load(f))
        return pipelines

    def delete_pipeline(self, name: str) -> None:
        """Delete a stored pipeline and its artifacts."""
        import shutil
        artifact_dir = self.base_dir / name
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
            log.info(f"Feature pipeline '{name}' deleted")
        else:
            log.warning(f"Pipeline '{name}' not found; nothing deleted")
