"""
Data validation module.
Validates DataFrame schema, data quality, and statistical properties
using Pandera and custom rule-based checks.
"""
from typing import Optional

import pandas as pd

from src.utils.exceptions import DataValidationError, InsufficientDataError
from src.utils.logger import get_logger

log = get_logger(__name__)


class DataValidator:
    """Validates data quality and schema before training.

    Runs a battery of checks including null thresholds, duplicate detection,
    constant column detection, and cardinality analysis.

    Example:
        >>> validator = DataValidator(min_samples=100)
        >>> report = validator.validate(df, target_column="target")
    """

    def __init__(
        self,
        min_samples: int = 100,
        null_threshold: float = 0.95,
        duplicate_threshold: float = 0.50,
        min_features: int = 2,
        raise_on_error: bool = True,
    ) -> None:
        self.min_samples = min_samples
        self.null_threshold = null_threshold
        self.duplicate_threshold = duplicate_threshold
        self.min_features = min_features
        self.raise_on_error = raise_on_error

    def validate(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> dict:
        """Run all validation checks on the DataFrame.

        Args:
            df: DataFrame to validate.
            target_column: Optional target column name.

        Returns:
            Validation report dict with 'passed', 'warnings', 'errors'.

        Raises:
            DataValidationError: If any critical check fails and raise_on_error is True.
            InsufficientDataError: If dataset has too few samples.
        """
        report: dict = {"passed": True, "warnings": [], "errors": [], "stats": {}}

        self._check_size(df, report)
        self._check_null_columns(df, report)
        self._check_duplicates(df, report)
        self._check_constant_columns(df, report)
        self._check_high_cardinality(df, report)

        if target_column:
            self._check_target(df, target_column, report)

        self._log_report(report)

        if report["errors"] and self.raise_on_error:
            raise DataValidationError(
                "Data validation failed",
                failed_checks=report["errors"],
            )

        return report

    def auto_fix(self, df: pd.DataFrame, report: dict) -> pd.DataFrame:
        """Apply automatic fixes based on validation report.

        Drops high-null columns and duplicate rows.

        Args:
            df: DataFrame to fix.
            report: Report from .validate().

        Returns:
            Cleaned DataFrame.
        """
        # Drop columns flagged as high-null
        cols_to_drop = [
            w["column"]
            for w in report.get("warnings", [])
            if w.get("rule") == "high_null_column"
        ]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            log.info(f"Dropped {len(cols_to_drop)} high-null columns: {cols_to_drop}")

        # Drop duplicate rows
        n_dupes = df.duplicated().sum()
        if n_dupes > 0:
            df = df.drop_duplicates()
            log.info(f"Dropped {n_dupes} duplicate rows.")

        return df

    # ── Private checks ────────────────────────────────────────────────────────

    def _check_size(self, df: pd.DataFrame, report: dict) -> None:
        n = len(df)
        report["stats"]["n_rows"] = n
        report["stats"]["n_cols"] = len(df.columns)

        if n < self.min_samples:
            report["errors"].append({
                "rule": "insufficient_samples",
                "message": f"Only {n} samples; need at least {self.min_samples}",
                "value": n,
            })

        if len(df.columns) < self.min_features:
            report["errors"].append({
                "rule": "insufficient_features",
                "message": f"Only {len(df.columns)} columns; need at least {self.min_features}",
            })

    def _check_null_columns(self, df: pd.DataFrame, report: dict) -> None:
        null_pct = df.isnull().mean()
        report["stats"]["null_pcts"] = null_pct.to_dict()

        high_null = null_pct[null_pct > self.null_threshold]
        for col, pct in high_null.items():
            report["warnings"].append({
                "rule": "high_null_column",
                "column": col,
                "message": f"Column '{col}' has {pct:.0%} null values",
                "value": pct,
            })

    def _check_duplicates(self, df: pd.DataFrame, report: dict) -> None:
        n_dupes = df.duplicated().sum()
        dup_pct = n_dupes / len(df)
        report["stats"]["duplicate_rows"] = int(n_dupes)
        report["stats"]["duplicate_pct"] = float(dup_pct)

        if dup_pct > self.duplicate_threshold:
            report["warnings"].append({
                "rule": "high_duplicates",
                "message": f"{n_dupes} duplicate rows ({dup_pct:.0%} of data)",
                "value": float(dup_pct),
            })

    def _check_constant_columns(self, df: pd.DataFrame, report: dict) -> None:
        constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
        report["stats"]["constant_columns"] = constant_cols
        for col in constant_cols:
            report["warnings"].append({
                "rule": "constant_column",
                "column": col,
                "message": f"Column '{col}' has only one unique value (constant)",
            })

    def _check_high_cardinality(
        self, df: pd.DataFrame, report: dict, threshold: int = 10000
    ) -> None:
        obj_cols = df.select_dtypes(include="object")
        high_card = [c for c in obj_cols if df[c].nunique() > threshold]
        report["stats"]["high_cardinality_columns"] = high_card
        for col in high_card:
            report["warnings"].append({
                "rule": "high_cardinality",
                "column": col,
                "message": (
                    f"Column '{col}' has {df[col].nunique()} unique values. "
                    "Consider text feature extraction."
                ),
            })

    def _check_target(
        self, df: pd.DataFrame, target_column: str, report: dict
    ) -> None:
        if target_column not in df.columns:
            report["errors"].append({
                "rule": "target_not_found",
                "message": f"Target column '{target_column}' not found in DataFrame",
            })
            return

        target = df[target_column]
        null_pct = target.isnull().mean()
        report["stats"]["target_null_pct"] = float(null_pct)
        report["stats"]["target_nunique"] = int(target.nunique())

        if null_pct > 0:
            report["errors"].append({
                "rule": "target_has_nulls",
                "message": f"Target column has {null_pct:.1%} null values",
                "value": float(null_pct),
            })

        if target.nunique() < 2:
            report["errors"].append({
                "rule": "target_single_class",
                "message": "Target column has only one unique class",
            })

    def _log_report(self, report: dict) -> None:
        n_errors = len(report["errors"])
        n_warnings = len(report["warnings"])

        if n_errors == 0 and n_warnings == 0:
            log.info("✅ Data validation passed with no issues.")
        elif n_errors == 0:
            log.warning(f"⚠️  Data validation: {n_warnings} warning(s), 0 error(s)")
        else:
            report["passed"] = False
            log.error(f"❌ Data validation: {n_warnings} warning(s), {n_errors} error(s)")
