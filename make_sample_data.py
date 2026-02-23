"""Generate a sample CSV for pipeline testing."""
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=500, n_features=10, n_informative=6,
    n_redundant=2, random_state=42
)
cols = [f"feature_{i}" for i in range(10)]
df = pd.DataFrame(X, columns=cols)
df["target"] = y

# Add a few missing values to test imputation
rng = np.random.default_rng(42)
for col in cols[:3]:
    mask = rng.random(len(df)) < 0.05
    df.loc[mask, col] = np.nan

df.to_csv("data/sample_train.csv", index=False)
print(f"Saved data/sample_train.csv: {df.shape}, target balance={df.target.mean():.2f}")
