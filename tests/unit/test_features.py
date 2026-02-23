"""
Unit tests for feature engineering components.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.datetime_features import DatetimeFeatureExtractor
from src.features.encoder import SmartEncoder
from src.features.imputer import SmartImputer
from src.features.scaler import SmartScaler
from src.features.type_detector import TypeDetector


def test_type_detector_basic(mixed_df):
    """TypeDetector should classify all columns without error."""
    detector = TypeDetector()
    types = detector.detect(mixed_df, target_column="target")
    assert "num_1" in types
    assert types["num_1"] in ("numeric_continuous", "numeric_discrete")
    assert types["cat_low"].startswith("categorical")
    assert types["target"] == "target"


def test_imputer_basic(classification_dataset):
    """SmartImputer should remove all NaN values."""
    X, y = classification_dataset
    X_with_nan = X.copy()
    X_with_nan.iloc[:10, 0] = np.nan
    imputer = SmartImputer(numeric_strategy="median")
    X_imputed = imputer.fit_transform(X_with_nan)
    assert X_imputed.isnull().sum().sum() == 0


def test_imputer_knn(classification_dataset):
    """SmartImputer with KNN strategy should work correctly."""
    X, y = classification_dataset
    X_with_nan = X.copy()
    X_with_nan.iloc[:5, :3] = np.nan
    imputer = SmartImputer(numeric_strategy="knn")
    X_imputed = imputer.fit_transform(X_with_nan)
    assert X_imputed.isnull().sum().sum() == 0


def test_smart_encoder_target(mixed_df):
    """SmartEncoder with target strategy should encode categorical columns."""
    y = mixed_df["target"]
    X = mixed_df.drop(columns=["target", "date_col", "text_col"])
    cat_cols = ["cat_low", "cat_high"]
    encoder = SmartEncoder(default_strategy="target", cols=cat_cols)
    X_enc = encoder.fit_transform(X, y)
    # Encoded columns should be numeric
    for col in cat_cols:
        assert pd.api.types.is_numeric_dtype(X_enc[col]), f"{col} not encoded properly"


def test_smart_scaler_robust(classification_dataset):
    """SmartScaler with robust strategy should scale numeric columns to ~zero-centered."""
    X, _ = classification_dataset
    scaler = SmartScaler(strategy="robust")
    X_scaled = scaler.fit_transform(X)
    # Median should be near 0 for robust scaler
    assert abs(X_scaled.median().mean()) < 1.0


def test_datetime_extractor(mixed_df):
    """DatetimeFeatureExtractor should produce additional columns from date column."""
    extractor = DatetimeFeatureExtractor(datetime_cols=["date_col"], drop_original=True)
    X_out = extractor.fit_transform(mixed_df.drop(columns=["target"]))
    assert "date_col" not in X_out.columns
    assert "date_col_year" in X_out.columns
    assert "date_col_month" in X_out.columns


def test_scaler_inverse(classification_dataset):
    """SmartScaler inverse_transform should approximately recover original values."""
    X, _ = classification_dataset
    scaler = SmartScaler(strategy="standard")
    X_scaled = scaler.fit_transform(X)
    X_recovered = scaler.inverse_transform(X_scaled)
    assert np.allclose(X.values, X_recovered.values, atol=1e-4)
