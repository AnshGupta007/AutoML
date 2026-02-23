"""
Pytest configuration and fixtures for AutoMLPro tests.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression


@pytest.fixture(scope="session")
def classification_dataset():
    """Generate a small binary classification dataset."""
    X_arr, y_arr = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )
    X = pd.DataFrame(X_arr, columns=[f"feat_{i}" for i in range(10)])
    y = pd.Series(y_arr, name="target")
    return X, y


@pytest.fixture(scope="session")
def regression_dataset():
    """Generate a small regression dataset."""
    X_arr, y_arr = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=6,
        noise=0.1,
        random_state=42,
    )
    X = pd.DataFrame(X_arr, columns=[f"feat_{i}" for i in range(10)])
    y = pd.Series(y_arr, name="target")
    return X, y


@pytest.fixture(scope="session")
def multiclass_dataset():
    """Generate a 3-class classification dataset."""
    X_arr, y_arr = make_classification(
        n_samples=300,
        n_features=10,
        n_classes=3,
        n_informative=6,
        random_state=42,
    )
    X = pd.DataFrame(X_arr, columns=[f"feat_{i}" for i in range(10)])
    y = pd.Series(y_arr, name="target")
    return X, y


@pytest.fixture(scope="session")
def mixed_df():
    """DataFrame with numeric, categorical, datetime, and text columns."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "num_1": rng.normal(0, 1, n),
        "num_2": rng.integers(0, 100, n).astype(float),
        "cat_low": rng.choice(["A", "B", "C"], n),
        "cat_high": [f"item_{i}" for i in rng.integers(0, 60, n)],
        "date_col": pd.date_range("2021-01-01", periods=n, freq="D"),
        "text_col": ["this is a sample text string " * 3] * n,
        "bool_col": rng.choice([True, False], n),
        "target": rng.integers(0, 2, n),
    })


@pytest.fixture
def small_csv(tmp_path):
    """Write a tiny CSV file to a temp directory and return its path."""
    df = pd.DataFrame({
        "age": [25, 35, 45, 55, 65],
        "income": [40000, 60000, 80000, 100000, 120000],
        "label": [0, 0, 1, 1, 1],
    })
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)
