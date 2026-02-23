"""
Data ingestion module.
Loads tabular data from CSV, Parquet, Excel, JSON, and other formats
with automatic type detection, metadata extraction, and chunked loading.
"""
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.utils.exceptions import DataIngestionError, UnsupportedFormatError
from src.utils.logger import get_logger
from src.utils.memory import reduce_memory_usage
from src.utils.timer import Timer
from src.utils.validators import validate_file_path

log = get_logger(__name__)

LOADERS = {
    "csv": "_load_csv",
    "parquet": "_load_parquet",
    "xlsx": "_load_excel",
    "xls": "_load_excel",
    "json": "_load_json",
    "feather": "_load_feather",
    "orc": "_load_orc",
}


class DataIngestion:
    """Load tabular data from multiple file formats.

    Supports CSV, Parquet, Excel, JSON, Feather, and ORC.
    Provides automatic memory optimization and metadata extraction.

    Example:
        >>> ingestion = DataIngestion()
        >>> df, meta = ingestion.load("data/train.csv")
        >>> print(meta)
    """

    def __init__(
        self,
        encoding: str = "utf-8",
        sep: str = ",",
        low_memory: bool = False,
        optimize_memory: bool = True,
        max_file_size_mb: int = 2000,
    ) -> None:
        self.encoding = encoding
        self.sep = sep
        self.low_memory = low_memory
        self.optimize_memory = optimize_memory
        self.max_file_size_mb = max_file_size_mb

    def load(self, path: Union[str, Path], **kwargs) -> tuple[pd.DataFrame, dict]:
        """Load data from a file, returning (DataFrame, metadata).

        Args:
            path: Path to the data file.
            **kwargs: Format-specific loading kwargs passed to the loader.

        Returns:
            Tuple of (DataFrame, metadata dict).

        Raises:
            DataIngestionError: On any loading failure.
            UnsupportedFormatError: If file format is not supported.
        """
        file_path = validate_file_path(path)
        self._check_file_size(file_path)

        ext = file_path.suffix.lstrip(".").lower()
        if ext not in LOADERS:
            raise UnsupportedFormatError(ext, list(LOADERS.keys()))

        loader_name = LOADERS[ext]
        loader = getattr(self, loader_name)

        log.info(f"Loading data from: {file_path} [{ext.upper()}]")
        try:
            with Timer(f"load_{ext}"):
                df = loader(file_path, **kwargs)
        except Exception as exc:
            raise DataIngestionError(f"Failed to load '{file_path}': {exc}") from exc

        if self.optimize_memory:
            df = reduce_memory_usage(df, verbose=True)

        metadata = self._extract_metadata(df, file_path)
        log.info(
            f"Data loaded successfully | shape={df.shape} "
            f"| memory={metadata['memory_mb']:.2f}MB"
        )
        return df, metadata

    def load_chunks(self, path: Union[str, Path], chunksize: int = 100_000):
        """Generator that yields DataFrame chunks from a CSV or Parquet file.

        Args:
            path: Path to the data file.
            chunksize: Number of rows per chunk.

        Yields:
            DataFrame chunks.
        """
        file_path = validate_file_path(path)
        ext = file_path.suffix.lstrip(".").lower()

        if ext == "csv":
            log.info(f"Loading CSV in chunks of {chunksize} from {file_path}")
            for chunk in pd.read_csv(file_path, chunksize=chunksize, encoding=self.encoding):
                yield chunk
        elif ext == "parquet":
            import pyarrow.parquet as pq
            table = pq.read_table(file_path)
            n_rows = table.num_rows
            for start in range(0, n_rows, chunksize):
                yield table.slice(start, chunksize).to_pandas()
        else:
            raise DataIngestionError(
                f"Chunked loading is only supported for CSV/Parquet, not '{ext}'"
            )

    # ── Private loaders ───────────────────────────────────────────────────────

    def _load_csv(self, path: Path, **kwargs) -> pd.DataFrame:
        defaults = {
            "encoding": self.encoding,
            "sep": self.sep,
            "low_memory": self.low_memory,
        }
        defaults.update(kwargs)
        return pd.read_csv(path, **defaults)

    def _load_parquet(self, path: Path, **kwargs) -> pd.DataFrame:
        return pd.read_parquet(path, **kwargs)

    def _load_excel(self, path: Path, **kwargs) -> pd.DataFrame:
        sheet = kwargs.pop("sheet_name", 0)
        return pd.read_excel(path, sheet_name=sheet, **kwargs)

    def _load_json(self, path: Path, **kwargs) -> pd.DataFrame:
        orient = kwargs.pop("orient", "records")
        return pd.read_json(path, orient=orient, **kwargs)

    def _load_feather(self, path: Path, **kwargs) -> pd.DataFrame:
        return pd.read_feather(path, **kwargs)

    def _load_orc(self, path: Path, **kwargs) -> pd.DataFrame:
        return pd.read_orc(path, **kwargs)

    def _check_file_size(self, path: Path) -> None:
        size_mb = path.stat().st_size / 1024 ** 2
        if size_mb > self.max_file_size_mb:
            log.warning(
                f"File size {size_mb:.1f}MB exceeds configured max "
                f"{self.max_file_size_mb}MB. Consider using load_chunks()."
            )

    @staticmethod
    def _extract_metadata(df: pd.DataFrame, path: Path) -> dict:
        """Extract metadata from a loaded DataFrame."""
        return {
            "path": str(path),
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_mb": df.memory_usage(deep=True).sum() / 1024 ** 2,
            "null_counts": df.isnull().sum().to_dict(),
            "null_pct": (df.isnull().mean() * 100).round(2).to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
        }


def load_data(
    path: Union[str, Path],
    optimize_memory: bool = True,
    **kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Convenience function to load data without instantiating DataIngestion.

    Args:
        path: File path.
        optimize_memory: Reduce memory usage after loading.
        **kwargs: Passed to the underlying loader.

    Returns:
        Tuple of (DataFrame, metadata).
    """
    ingestion = DataIngestion(optimize_memory=optimize_memory)
    return ingestion.load(path, **kwargs)
