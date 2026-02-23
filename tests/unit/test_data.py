"""
Unit tests for the data module.
"""
import pandas as pd
import pytest

from src.data.ingestion import DataIngestion
from src.data.splitter import DataSplitter
from src.data.validation import DataValidator


def test_data_ingestion_csv(small_csv):
    """DataIngestion should load a CSV file successfully."""
    ingestion = DataIngestion()
    result = ingestion.load(small_csv)
    assert "data" in result
    assert isinstance(result["data"], pd.DataFrame)
    assert len(result["data"]) == 5


def test_data_validation_basic(mixed_df):
    """DataValidator should run without error on mixed data."""
    validator = DataValidator()
    report = validator.validate(mixed_df, target_column="target")
    assert "warnings" in report
    assert isinstance(report["warnings"], list)


def test_data_splitter_stratified(classification_dataset):
    """DataSplitter should create train/val/test splits with expected sizes."""
    X, y = classification_dataset
    df = pd.concat([X, y.rename("target")], axis=1)
    splitter = DataSplitter(test_size=0.2, val_size=0.1, strategy="stratified")
    splits = splitter.split(df, target_column="target")
    assert "X_train" in splits
    assert len(splits["X_train"]) > 100
    assert len(splits["X_test"]) > 0
    # Check total rows add up
    total = len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_test"])
    assert total == len(df)


def test_data_splitter_random(regression_dataset):
    """DataSplitter should work with random strategy."""
    X, y = regression_dataset
    df = pd.concat([X, y.rename("target")], axis=1)
    splitter = DataSplitter(strategy="random")
    splits = splitter.split(df, target_column="target")
    assert len(splits["X_train"]) > 0
