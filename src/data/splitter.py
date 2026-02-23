"""
Data splitting strategies.
Provides train/validation/test splits for: random, stratified, time series,
and group-based strategies. Supports multi-fold CV preparation.
"""
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
    train_test_split,
)

from src.utils.exceptions import DataError
from src.utils.logger import get_logger

log = get_logger(__name__)


class DataSplitter:
    """Split data into train / validation / test sets.

    Supports multiple strategies:
    - ``random``: Simple random split.
    - ``stratified``: Stratified by target class distribution.
    - ``time_series``: Chronological split (no shuffle).
    - ``group``: Group-aware split (no data leakage across groups).

    Example:
        >>> splitter = DataSplitter(test_size=0.2, val_size=0.1, strategy="stratified")
        >>> splits = splitter.split(df, target_column="label")
        >>> X_train, y_train = splits["train"]
        >>> X_val, y_val = splits["val"]
        >>> X_test, y_test = splits["test"]
    """

    SUPPORTED_STRATEGIES = ["random", "stratified", "time_series", "group"]

    def __init__(
        self,
        test_size: float = 0.20,
        val_size: float = 0.10,
        strategy: str = "stratified",
        shuffle: bool = True,
        random_state: int = 42,
        date_column: Optional[str] = None,
        group_column: Optional[str] = None,
    ) -> None:
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from: {self.SUPPORTED_STRATEGIES}"
            )
        self.test_size = test_size
        self.val_size = val_size
        self.strategy = strategy
        self.shuffle = shuffle
        self.random_state = random_state
        self.date_column = date_column
        self.group_column = group_column

    def split(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> dict[str, Union[pd.DataFrame, pd.Series]]:
        """Split DataFrame into train/val/test.

        Args:
            df: Full dataset.
            target_column: Name of the target column.

        Returns:
            Dict with keys 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'.
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]

        if self.strategy == "time_series":
            return self._time_split(X, y, df)
        elif self.strategy == "stratified":
            return self._stratified_split(X, y)
        elif self.strategy == "group":
            return self._group_split(X, y, df)
        else:
            return self._random_split(X, y)

    def split_array(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict[str, Union[pd.DataFrame, pd.Series]]:
        """Split pre-separated X and y arrays."""
        return self.split(
            pd.concat([X, y.rename("__target__")], axis=1),
            target_column="__target__",
        )

    # ── Split strategies ──────────────────────────────────────────────────────

    def _random_split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> dict:
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )
        if self.val_size > 0:
            # val_size is relative to original; adjust for trainval set
            adjusted_val = self.val_size / (1 - self.test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval,
                test_size=adjusted_val,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
        else:
            X_train, y_train = X_trainval, y_trainval
            X_val, y_val = X_trainval.iloc[:0], y_trainval.iloc[:0]

        return self._build_output(X_train, y_train, X_val, y_val, X_test, y_test)

    def _stratified_split(self, X: pd.DataFrame, y: pd.Series) -> dict:
        sss_test = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        train_val_idx, test_idx = next(sss_test.split(X, y))

        X_trainval, y_trainval = X.iloc[train_val_idx], y.iloc[train_val_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        if self.val_size > 0:
            adjusted_val = self.val_size / (1 - self.test_size)
            sss_val = StratifiedShuffleSplit(
                n_splits=1,
                test_size=adjusted_val,
                random_state=self.random_state,
            )
            train_idx, val_idx = next(sss_val.split(X_trainval, y_trainval))
            X_train = X_trainval.iloc[train_idx]
            y_train = y_trainval.iloc[train_idx]
            X_val = X_trainval.iloc[val_idx]
            y_val = y_trainval.iloc[val_idx]
        else:
            X_train, y_train = X_trainval, y_trainval
            X_val, y_val = X_trainval.iloc[:0], y_trainval.iloc[:0]

        return self._build_output(X_train, y_train, X_val, y_val, X_test, y_test)

    def _time_split(
        self, X: pd.DataFrame, y: pd.Series, df_orig: pd.DataFrame
    ) -> dict:
        """Chronological split — sorts by date if available."""
        if self.date_column and self.date_column in df_orig.columns:
            sort_order = df_orig[self.date_column].argsort()
            X = X.iloc[sort_order]
            y = y.iloc[sort_order]
            log.info(f"Sorted by date column: '{self.date_column}'")

        n = len(X)
        test_n = int(n * self.test_size)
        val_n = int(n * self.val_size)
        train_n = n - test_n - val_n

        X_train = X.iloc[:train_n]
        y_train = y.iloc[:train_n]
        X_val = X.iloc[train_n: train_n + val_n]
        y_val = y.iloc[train_n: train_n + val_n]
        X_test = X.iloc[train_n + val_n:]
        y_test = y.iloc[train_n + val_n:]

        return self._build_output(X_train, y_train, X_val, y_val, X_test, y_test)

    def _group_split(
        self, X: pd.DataFrame, y: pd.Series, df_orig: pd.DataFrame
    ) -> dict:
        if not self.group_column or self.group_column not in df_orig.columns:
            log.warning("group_column not found; falling back to random split.")
            return self._random_split(X, y)

        groups = df_orig[self.group_column]
        gss = GroupShuffleSplit(
            n_splits=1, test_size=self.test_size, random_state=self.random_state
        )
        train_val_idx, test_idx = next(gss.split(X, y, groups))
        X_trainval, y_trainval = X.iloc[train_val_idx], y.iloc[train_val_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        return self._build_output(
            X_trainval, y_trainval,
            X_trainval.iloc[:0], y_trainval.iloc[:0],
            X_test, y_test,
        )

    @staticmethod
    def _build_output(X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
        result = {
            "X_train": X_train.reset_index(drop=True),
            "y_train": y_train.reset_index(drop=True),
            "X_val": X_val.reset_index(drop=True),
            "y_val": y_val.reset_index(drop=True),
            "X_test": X_test.reset_index(drop=True),
            "y_test": y_test.reset_index(drop=True),
        }
        log.info(
            f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )
        return result
