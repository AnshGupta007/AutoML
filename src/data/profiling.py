"""
Automated EDA and data profiling module.
Generates statistical summaries, distribution plots,
and HTML/JSON profiling reports.
"""
import json
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.utils.logger import get_logger
from src.utils.timer import Timer

log = get_logger(__name__)


class DataProfiler:
    """Generates automated EDA reports for tabular datasets.

    Produces a lightweight quick-stats report always, and optionally
    a full ydata-profiling HTML report for richer exploration.

    Example:
        >>> profiler = DataProfiler(output_dir="reports/eda")
        >>> stats = profiler.profile(df, name="titanic")
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "reports/eda",
        enable_full_report: bool = True,
        dark_mode: bool = False,
        minimal: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_full_report = enable_full_report
        self.dark_mode = dark_mode
        self.minimal = minimal

    def profile(
        self,
        df: pd.DataFrame,
        name: str = "dataset",
        target_column: Optional[str] = None,
    ) -> dict:
        """Run profiling and return statistics dict.

        Args:
            df: DataFrame to profile.
            name: Dataset name used in report filenames.
            target_column: If provided, include target distribution stats.

        Returns:
            Dictionary with quick stats. HTML report saved to output_dir.
        """
        log.info(f"Profiling dataset '{name}' with shape {df.shape}")

        stats = self._compute_quick_stats(df, target_column)
        stats_path = self.output_dir / f"{name}_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        log.info(f"Quick stats saved to {stats_path}")

        if self.enable_full_report:
            self._generate_ydata_report(df, name)

        return stats

    def _compute_quick_stats(
        self, df: pd.DataFrame, target_column: Optional[str]
    ) -> dict:
        """Compute lightweight statistical summary."""
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        stats: dict = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 3),
            "null_counts": df.isnull().sum().to_dict(),
            "null_percentages": (df.isnull().mean() * 100).round(2).to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric_columns": numeric_cols,
            "categorical_columns": cat_cols,
        }

        if numeric_cols:
            desc = df[numeric_cols].describe().to_dict()
            stats["numeric_summary"] = {
                col: {k: round(v, 4) for k, v in vals.items()}
                for col, vals in desc.items()
            }

        if cat_cols:
            stats["categorical_summary"] = {
                col: {
                    "n_unique": int(df[col].nunique()),
                    "top_values": df[col].value_counts().head(10).to_dict(),
                }
                for col in cat_cols
            }

        if target_column and target_column in df.columns:
            target = df[target_column]
            stats["target"] = {
                "column": target_column,
                "dtype": str(target.dtype),
                "n_unique": int(target.nunique()),
                "null_count": int(target.isnull().sum()),
                "value_counts": target.value_counts().head(20).to_dict(),
            }
            if pd.api.types.is_numeric_dtype(target):
                stats["target"]["mean"] = float(target.mean())
                stats["target"]["std"] = float(target.std())
                stats["target"]["min"] = float(target.min())
                stats["target"]["max"] = float(target.max())

        return stats

    def _generate_ydata_report(self, df: pd.DataFrame, name: str) -> Optional[Path]:
        """Generate full HTML profiling report using ydata-profiling."""
        try:
            from ydata_profiling import ProfileReport

            with Timer("ydata-profiling report"):
                profile = ProfileReport(
                    df,
                    title=f"AutoMLPro EDA — {name}",
                    dark_mode=self.dark_mode,
                    minimal=self.minimal,
                    correlations={
                        "pearson": {"calculate": True},
                        "spearman": {"calculate": False},
                    },
                )
                report_path = self.output_dir / f"{name}_profile.html"
                profile.to_file(report_path)
                log.info(f"Full profiling report saved to {report_path}")
                return report_path

        except ImportError:
            log.warning("ydata-profiling not installed. Skipping full HTML report.")
            return None
        except Exception as exc:
            log.warning(f"Failed to generate ydata report: {exc}")
            return None
