"""
Integration test — runs the full AutoML pipeline end-to-end on a small dataset.
"""
import pandas as pd
import pytest

from sklearn.datasets import make_classification


@pytest.mark.integration
def test_full_pipeline_classification(tmp_path):
    """End-to-end pipeline should train and predict correctly on a small dataset."""
    from src.orchestration.pipeline import automl_pipeline

    # Create a small CSV
    X_arr, y_arr = make_classification(
        n_samples=200, n_features=10, n_informative=6, random_state=42
    )
    df = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(10)])
    df["target"] = y_arr
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    result = automl_pipeline(
        data_path=str(csv_path),
        target_column="target",
        task_type="classification",
        experiment_name="test_run",
        enable_hpo=False,     # Disable HPO for speed
        enable_ensemble=False,
        n_trials=0,
        output_dir=str(tmp_path / "reports"),
    )

    assert "training_result" in result
    assert "eval_result" in result
    assert "report_path" in result
    assert "accuracy" in result["eval_result"]["metrics"]
    assert result["eval_result"]["metrics"]["accuracy"] > 0.5


@pytest.mark.integration
def test_feature_pipeline_end_to_end(mixed_df):
    """Feature engineering pipeline should transform mixed data without error."""
    from src.features.pipeline import FeatureEngineeringPipeline

    y = mixed_df["target"]
    X = mixed_df.drop(columns=["target"])

    pipeline = FeatureEngineeringPipeline(
        enable_feature_generation=False,
        enable_feature_selection=False,
    )
    X_train = X.iloc[:80]
    y_train = y.iloc[:80]
    X_test = X.iloc[80:]

    X_train_eng = pipeline.fit_transform(X_train, y_train)
    X_test_eng = pipeline.transform(X_test)

    assert X_train_eng.shape[1] == X_test_eng.shape[1]
    assert X_train_eng.isnull().sum().sum() == 0
