"""
Memory optimization utilities.
Provides DataFrame memory reduction, profiling, and garbage collection helpers.
"""
import gc
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Reduce DataFrame memory usage by downcasting numeric dtypes.

    Iterates over all numeric columns and selects the smallest dtype
    that can represent the data without loss of information.

    Args:
        df: Input DataFrame to optimize.
        verbose: Whether to log memory savings.

    Returns:
        DataFrame with reduced memory footprint.

    Example:
        >>> df = reduce_memory_usage(pd.read_csv("large_file.csv"))
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and str(col_type) != "category":
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)  # float16 has precision issues
                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    if verbose:
        savings = 100 * (start_mem - end_mem) / start_mem
        log.info(
            f"Memory reduced: {start_mem:.2f} MB → {end_mem:.2f} MB "
            f"({savings:.1f}% reduction)"
        )

    return df


def get_memory_usage_mb(df: pd.DataFrame) -> float:
    """Get DataFrame memory usage in megabytes.

    Args:
        df: Input DataFrame.

    Returns:
        Memory usage in MB.
    """
    return df.memory_usage(deep=True).sum() / 1024 ** 2


def convert_to_categorical(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    threshold: int = 50,
) -> pd.DataFrame:
    """Convert low-cardinality object columns to categorical dtype.

    Args:
        df: Input DataFrame.
        columns: Specific columns to convert. If None, auto-detect.
        threshold: Max unique values to consider a column categorical.

    Returns:
        DataFrame with categorical columns.
    """
    if columns is None:
        columns = [
            c for c in df.select_dtypes(include="object").columns
            if df[c].nunique() <= threshold
        ]

    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype("category")

    log.debug(f"Converted {len(columns)} columns to categorical: {columns}")
    return df


def free_memory(*objects) -> None:
    """Explicitly delete objects and run garbage collection.

    Useful for freeing large DataFrames or model objects after use.

    Args:
        *objects: Objects to delete.

    Example:
        >>> free_memory(large_df, temp_model)
    """
    for obj in objects:
        del obj
    gc.collect()
    log.debug("Garbage collection performed")


def check_available_memory() -> dict:
    """Check system available memory.

    Returns:
        Dictionary with total, available, and used memory in MB.
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total_mb": mem.total / 1024 ** 2,
            "available_mb": mem.available / 1024 ** 2,
            "used_mb": mem.used / 1024 ** 2,
            "percent_used": mem.percent,
        }
    except ImportError:
        log.warning("psutil not installed. Cannot check system memory.")
        return {}
