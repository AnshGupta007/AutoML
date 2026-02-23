"""Final smoke test — ASCII only output for Windows compat."""
import sys
import pandas as pd
from sklearn.datasets import make_classification
from src.features.pipeline import FeatureEngineeringPipeline
from src.models.xgboost_model import XGBoostModel
from src.evaluation.metrics import compute_metrics

X, y = make_classification(n_samples=100, n_features=8, random_state=42)
X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
y_s  = pd.Series(y, name="target")

fp = FeatureEngineeringPipeline(enable_feature_generation=False, enable_feature_selection=False)
X_tr = fp.fit_transform(X_df[:80], y_s[:80])
X_te = fp.transform(X_df[80:])
print(f"Feature pipeline OK: train={X_tr.shape}, test={X_te.shape}")

m = XGBoostModel(task_type="classification")
m.fit(X_tr, y_s[:80])
preds = m.predict(X_te)
print(f"XGBoost OK: {len(preds)} predictions")

metrics = compute_metrics(y_s[80:], preds, "classification")
print(f"Metrics OK: accuracy={metrics['accuracy']:.3f}, f1={metrics.get('f1_weighted',0):.3f}")

print("STATUS: ALL PIPELINE COMPONENTS FUNCTIONAL")
