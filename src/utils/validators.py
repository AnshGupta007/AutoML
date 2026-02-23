"""
Input validation utilities.
Provides reusable validators for paths, DataFrames, configs, and API inputs.
"""
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from src.utils.exceptions import (
    ColumnNotFoundError,
    ConfigurationError,
    DataValidationError,
    UnsupportedFormatError,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

SUPPORTED_FILE_FORMATS = ["csv", "parquet", "xlsx", "xls", "json", "feather", "orc"]
SUPPORTED_TASK_TYPES = ["classification", "regression", "auto"]


def validate_file_path(path: Union[str, Path]) -> Path:
    """Validate that a file path exists and has a supported format.

    Args:
        path: File path to validate.

    Returns:
        Resolved Path object.

    Raises:
        FileNotFoundError: If the path does not exist.
        UnsupportedFormatError: If the file extension is not supported.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: '{p}'")
    if not p.is_file():
        raise FileNotFoundError(f"Path is not a file: '{p}'")

    ext = p.suffix.lstrip(".")
    if ext not in SUPPORTED_FILE_FORMATS:
        raise UnsupportedFormatError(ext, SUPPORTED_FILE_FORMATS)

    log.debug(f"Validated file path: {p}")
    return p


def validate_dataframe(
    df: pd.DataFrame,
    min_rows: int = 10,
    min_cols: int = 2,
    allow_empty: bool = False,
) -> None:
    """Validate a DataFrame meets minimum requirements.

    Args:
        df: DataFrame to validate.
        min_rows: Minimum number of rows required.
        min_cols: Minimum number of columns required.
        allow_empty: Whether to allow empty DataFrames.

    Raises:
        DataValidationError: If the DataFrame fails validation.
        TypeError: If input is not a DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

    if not allow_empty and df.empty:
        raise DataValidationError("DataFrame is empty.")

    errors = []
    if len(df) < min_rows:
        errors.append(f"Too few rows: {len(df)} < {min_rows}")
    if len(df.columns) < min_cols:
        errors.append(f"Too few columns: {len(df.columns)} < {min_cols}")

    if errors:
        raise DataValidationError(
            "DataFrame validation failed", failed_checks=errors
        )

    log.debug(f"DataFrame validated: shape={df.shape}")


def validate_target_column(df: pd.DataFrame, target_column: str) -> None:
    """Validate that the target column exists in the DataFrame.

    Args:
        df: Input DataFrame.
        target_column: Name of the target column.

    Raises:
        ColumnNotFoundError: If target column is absent.
    """
    if target_column not in df.columns:
        raise ColumnNotFoundError(target_column, list(df.columns))
    log.debug(f"Target column '{target_column}' validated.")


def validate_task_type(task_type: str) -> str:
    """Validate and normalize task type.

    Args:
        task_type: Task type string.

    Returns:
        Lowercase normalized task type.

    Raises:
        UnsupportedTaskError: If task type is not supported.
    """
    from src.utils.exceptions import UnsupportedTaskError

    normalized = task_type.strip().lower()
    if normalized not in SUPPORTED_TASK_TYPES:
        raise UnsupportedTaskError(task_type, SUPPORTED_TASK_TYPES)
    return normalized


def validate_config(config: dict, required_keys: list[str]) -> None:
    """Validate that a configuration dictionary contains required keys.

    Args:
        config: Configuration dictionary.
        required_keys: List of keys that must be present.

    Raises:
        ConfigurationError: If required keys are missing.
    """
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ConfigurationError(
            f"Missing required config keys: {missing}",
            {"missing_keys": missing},
        )


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize DataFrame column names for safe usage.

    Replaces spaces and special characters with underscores,
    converts to lowercase.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with sanitized column names.
    """
    import re

    original_cols = list(df.columns)
    new_cols = [
        re.sub(r"[^\w]", "_", str(c)).strip("_").lower()
        for c in df.columns
    ]

    # Handle duplicate names after sanitization
    seen: dict[str, int] = {}
    deduped_cols = []
    for col in new_cols:
        if col in seen:
            seen[col] += 1
            deduped_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            deduped_cols.append(col)

    df.columns = deduped_cols
    renamed = {o: n for o, n in zip(original_cols, deduped_cols) if o != n}
    if renamed:
        log.debug(f"Sanitized column names: {renamed}")

    return df


def validate_positive_int(value: Any, name: str) -> int:
    """Validate a value is a positive integer.

    Args:
        value: Value to validate.
        name: Parameter name for error messages.

    Returns:
        Validated integer.

    Raises:
        ValueError: If value is not a positive integer.
    """
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be a positive integer, got: {value!r}")
    if v <= 0:
        raise ValueError(f"'{name}' must be > 0, got: {v}")
    return v
