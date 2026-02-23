"""
Feature engineering pipeline.
Orchestrates type detection → imputation → datetime/text extraction →
encoding → scaling → feature generation → feature selection.
"""
import joblib
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from sklearn.pipeline import Pipeline

from src.features.datetime_features import DatetimeFeatureExtractor
from src.features.encoder import SmartEncoder
from src.features.feature_generator import FeatureGenerator
from src.features.feature_selector import SmartFeatureSelector
from src.features.imputer import SmartImputer
from src.features.scaler import SmartScaler
from src.features.text_features import TextFeatureExtractor
from src.features.type_detector import TypeDetector
from src.utils.logger import get_logger
from src.utils.timer import Timer

log = get_logger(__name__)


class FeatureEngineeringPipeline:
    """End-to-end feature engineering pipeline.

    Composes all feature transformers in the correct order and
    exposes a simple fit/transform/fit_transform interface.

    Example:
        >>> pipe = FeatureEngineeringPipeline.from_config(cfg)
        >>> X_train_eng = pipe.fit_transform(X_train, y_train)
        >>> X_test_eng = pipe.transform(X_test)
    """

    def __init__(
        self,
        numeric_impute_strategy: str = "median",
        categorical_impute_strategy: str = "most_frequent",
        encoding_strategy: str = "target",
        scaling_strategy: str = "robust",
        enable_datetime_features: bool = True,
        enable_text_features: bool = True,
        enable_feature_generation: bool = True,
        polynomial_degree: int = 2,
        enable_feature_selection: bool = True,
        selection_method: str = "mutual_info",
        max_features: Optional[int] = 50,
        task_type: str = "auto",
        categorical_threshold: int = 50,
        random_state: int = 42,
    ) -> None:
        self.task_type = task_type
        self.random_state = random_state

        self.type_detector = TypeDetector(
            categorical_threshold=categorical_threshold
        )
        self.imputer = SmartImputer(
            numeric_strategy=numeric_impute_strategy,
            categorical_strategy=categorical_impute_strategy,
        )
        self.datetime_extractor = DatetimeFeatureExtractor(auto_detect=enable_datetime_features)
        self.text_extractor = TextFeatureExtractor(auto_detect=enable_text_features)
        self.encoder = SmartEncoder(default_strategy=encoding_strategy)
        self.scaler = SmartScaler(strategy=scaling_strategy)
        self.feature_generator = FeatureGenerator(
            polynomial_degree=polynomial_degree,
            interaction_features=enable_feature_generation,
            ratio_features=enable_feature_generation,
        ) if enable_feature_generation else None
        self.feature_selector = SmartFeatureSelector(
            method=selection_method,
            max_features=max_features,
            task_type=task_type,
        ) if enable_feature_selection else None

        self._is_fitted = False
        self._column_types: dict = {}
        self._feature_names_in: list[str] = []
        self._feature_names_out: list[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineeringPipeline":
        """Fit all transformers on the training data.

        Args:
            X: Feature DataFrame (no target column).
            y: Target series (needed for target encoding, feature selection).

        Returns:
            self
        """
        log.info(f"Fitting feature pipeline on shape: {X.shape}")
        with Timer("feature pipeline fit"):
            self._feature_names_in = list(X.columns)
            self._column_types = self.type_detector.detect(X)

            # Impute → Datetime → Text → Encode → Scale → Generate → Select
            X = self.imputer.fit_transform(X)
            X = self.datetime_extractor.fit_transform(X)
            X = self.text_extractor.fit_transform(X)
            X = self.encoder.fit_transform(X, y)
            X = self.scaler.fit_transform(X)

            if self.feature_generator:
                X = self.feature_generator.fit_transform(X)

            if self.feature_selector and y is not None:
                X = self.feature_selector.fit_transform(X, y)

            self._feature_names_out = list(X.columns)
            self._is_fitted = True

        log.info(f"Feature pipeline fitted. Output shape: {X.shape}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted transformers.

        Args:
            X: Feature DataFrame.

        Returns:
            Transformed DataFrame.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline is not fitted. Call .fit() first.")

        X = self.imputer.transform(X)
        X = self.datetime_extractor.transform(X)
        X = self.text_extractor.transform(X)
        X = self.encoder.transform(X)
        X = self.scaler.transform(X)

        if self.feature_generator:
            X = self.feature_generator.transform(X)

        if self.feature_selector:
            X = self.feature_selector.transform(X)

        return X

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y)

        # Re-run transform on training data (avoids double-fitting)
        X = self.imputer.transform(X)
        X = self.datetime_extractor.transform(X)
        X = self.text_extractor.transform(X)
        X = self.encoder.transform(X)
        X = self.scaler.transform(X)
        if self.feature_generator:
            X = self.feature_generator.transform(X)
        if self.feature_selector and y is not None:
            X = self.feature_selector.transform(X)

        return X

    @property
    def feature_names_out(self) -> list[str]:
        return self._feature_names_out

    @property
    def feature_names_in(self) -> list[str]:
        return self._feature_names_in

    def save(self, path: Union[str, Path]) -> None:
        """Save fitted pipeline to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        log.info(f"Feature pipeline saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FeatureEngineeringPipeline":
        """Load saved pipeline from disk."""
        pipeline = joblib.load(path)
        log.info(f"Feature pipeline loaded from {path}")
        return pipeline

    @classmethod
    def from_config(cls, cfg: dict) -> "FeatureEngineeringPipeline":
        """Construct pipeline from a configuration dictionary."""
        features_cfg = cfg.get("features", {})
        imp = features_cfg.get("imputation", {})
        enc = features_cfg.get("encoding", {})
        sel = features_cfg.get("feature_selection", {})
        gen = features_cfg.get("feature_generation", {})
        scl = features_cfg.get("scaling", {})

        return cls(
            numeric_impute_strategy=imp.get("numeric_strategy", "median"),
            categorical_impute_strategy=imp.get("categorical_strategy", "most_frequent"),
            encoding_strategy=enc.get("default_strategy", "target"),
            scaling_strategy=scl.get("strategy", "robust"),
            enable_feature_generation=gen.get("enable", True),
            polynomial_degree=gen.get("polynomial_degree", 2),
            enable_feature_selection=sel.get("enable", True),
            selection_method=sel.get("method", "mutual_info"),
            max_features=sel.get("max_features", 50),
        )
