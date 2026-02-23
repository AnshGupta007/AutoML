"""
Unit tests for model wrappers.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.linear_model import LinearModel
from src.models.xgboost_model import XGBoostModel


def test_xgboost_classification(classification_dataset):
    """XGBoostModel should train and predict on binary classification."""
    X, y = classification_dataset
    model = XGBoostModel(task_type="classification", n_estimators=50)
    model.fit(X.iloc[:160], y.iloc[:160], X.iloc[160:], y.iloc[160:])
    assert model.is_fitted
    preds = model.predict(X.iloc[160:])
    assert len(preds) == 40
    assert set(np.unique(preds)).issubset({0, 1})


def test_xgboost_predict_proba(classification_dataset):
    """XGBoostModel.predict_proba should return probabilities."""
    X, y = classification_dataset
    model = XGBoostModel(task_type="classification", n_estimators=50)
    model.fit(X, y)
    probas = model.predict_proba(X)
    assert probas.shape == (len(X), 2)
    assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-5)


def test_xgboost_regression(regression_dataset):
    """XGBoostModel should train and predict on regression."""
    X, y = regression_dataset
    model = XGBoostModel(task_type="regression", n_estimators=50)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    # R² should be positive for a fitted model on training data
    from sklearn.metrics import r2_score
    assert r2_score(y, preds) > 0


def test_linear_model_classification(classification_dataset):
    """LinearModel should train and predict correctly."""
    X, y = classification_dataset
    model = LinearModel(task_type="classification")
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_linear_model_regression(regression_dataset):
    """LinearModel ridge regression should converge."""
    X, y = regression_dataset
    model = LinearModel(task_type="regression", model_type="ridge")
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_model_not_fitted_raises():
    """Predicting before fitting should raise an error."""
    from src.utils.exceptions import ModelNotFittedError
    model = XGBoostModel()
    with pytest.raises((ModelNotFittedError, RuntimeError, Exception)):
        model.predict(pd.DataFrame([[1, 2, 3]]))


def test_xgboost_feature_importance(classification_dataset):
    """get_feature_importance should return a non-empty Series."""
    X, y = classification_dataset
    model = XGBoostModel(task_type="classification", n_estimators=50)
    model.fit(X, y)
    fi = model.get_feature_importance()
    assert fi is not None
    assert len(fi) == X.shape[1]
