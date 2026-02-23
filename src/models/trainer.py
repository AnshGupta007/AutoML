"""
Master training orchestrator.
Coordinates data splitting, feature engineering, model selection,
HPO, ensemble building, MLflow tracking, and model persistence.
"""
import json
from pathlib import Path
from typing import Optional, Union

import mlflow
import pandas as pd

from src.features.pipeline import FeatureEngineeringPipeline
from src.models.ensemble import StackingEnsemble
from src.models.hyperparameter_tuner import HyperparameterTuner
from src.models.model_selector import MODEL_REGISTRY, ModelSelector
from src.utils.logger import get_logger
from src.utils.timer import Timer

log = get_logger(__name__)

DEFAULT_SEARCH_SPACES = {
    "xgboost": {
        "n_estimators": (100, 1000),
        "learning_rate": (0.01, 0.3),
        "max_depth": (3, 10),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "reg_alpha": (0.0, 5.0),
        "reg_lambda": (0.5, 5.0),
    },
    "lightgbm": {
        "n_estimators": (100, 2000),
        "learning_rate": (0.01, 0.3),
        "num_leaves": (20, 300),
        "min_child_samples": (5, 100),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "reg_alpha": (0.0, 5.0),
        "reg_lambda": (0.0, 5.0),
    },
    "catboost": {
        "iterations": (100, 1000),
        "learning_rate": (0.01, 0.3),
        "depth": (4, 10),
        "l2_leaf_reg": (1.0, 10.0),
    },
    "linear": {
        "C": (0.01, 100.0),
        "alpha": (0.001, 100.0),
    },
}


class Trainer:
    """Master training orchestrator for the AutoMLPro pipeline.

    Workflow:
        1. Model selection (quick CV comparison)
        2. Hyperparameter optimization (Optuna)
        3. Final model training on full train set
        4. Optional ensemble construction
        5. MLflow tracking
        6. Model persistence

    Example:
        >>> trainer = Trainer(task_type="classification", experiment_name="titanic")
        >>> result = trainer.train(X_train, y_train, X_val, y_val)
    """

    def __init__(
        self,
        task_type: str = "auto",
        experiment_name: str = "automl_experiment",
        mlflow_tracking_uri: str = "mlruns",
        n_trials: int = 30,
        cv_folds: int = 5,
        enable_hpo: bool = True,
        enable_ensemble: bool = True,
        ensemble_strategy: str = "stacking",
        model_candidates: Optional[list[str]] = None,
        models_dir: Union[str, Path] = "models",
        random_state: int = 42,
        metric: str = "auto",
    ) -> None:
        self.task_type = task_type
        self.experiment_name = experiment_name
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.enable_hpo = enable_hpo
        self.enable_ensemble = enable_ensemble
        self.ensemble_strategy = ensemble_strategy
        self.model_candidates = model_candidates or ["xgboost", "lightgbm", "catboost", "linear"]
        self.models_dir = Path(models_dir)
        self.random_state = random_state
        self.metric = metric

        self.best_model = None
        self.feature_pipeline: Optional[FeatureEngineeringPipeline] = None
        self.run_id: Optional[str] = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        feature_pipeline: Optional[FeatureEngineeringPipeline] = None,
    ) -> dict:
        """Execute the full training pipeline.

        Args:
            X_train: Training features (already engineered or raw).
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.
            feature_pipeline: Pre-fitted feature pipeline (optional).

        Returns:
            Dict with keys: best_model, feature_pipeline, metrics, run_id.
        """
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        task = self._resolve_task(y_train)

        with mlflow.start_run(run_name=f"automl_{task}") as run:
            self.run_id = run.info.run_id
            log.info(f"MLflow run started: {self.run_id}")

            mlflow.log_params({
                "task_type": task,
                "n_train": len(X_train),
                "n_features": X_train.shape[1],
                "enable_hpo": self.enable_hpo,
                "enable_ensemble": self.enable_ensemble,
                "n_trials": self.n_trials,
            })

            # Phase 1: Feature Engineering
            if feature_pipeline is not None:
                self.feature_pipeline = feature_pipeline
            else:
                self.feature_pipeline = FeatureEngineeringPipeline(task_type=task)
                with Timer("feature engineering"):
                    X_train = self.feature_pipeline.fit_transform(X_train, y_train)
                    if X_val is not None:
                        X_val = self.feature_pipeline.transform(X_val)

            # Phase 2: Model Selection
            log.info("Phase 2: Model selection")
            selector = ModelSelector(
                candidates=self.model_candidates,
                metric=self.metric,
                cv_folds=self.cv_folds,
                task_type=task,
                random_state=self.random_state,
            )
            with Timer("model selection"):
                top_models = selector.select(X_train, y_train, X_val, y_val)

            mlflow.log_param("top_models", [r["model_name"] for r in top_models])

            # Phase 3: HPO on best model
            best_model_name = top_models[0]["model_name"]
            log.info(f"Phase 3: HPO on {best_model_name}")

            best_params = {}
            if self.enable_hpo and best_model_name in DEFAULT_SEARCH_SPACES:
                tuner = HyperparameterTuner(
                    n_trials=self.n_trials,
                    cv_folds=self.cv_folds,
                    metric=self.metric,
                    random_state=self.random_state,
                )
                search_space = DEFAULT_SEARCH_SPACES[best_model_name]
                best_params = tuner.tune(
                    MODEL_REGISTRY[best_model_name],
                    X_train, y_train, search_space, task,
                )
                mlflow.log_params({f"hpo_{k}": v for k, v in best_params.items()})

            # Phase 4: Train final model
            log.info("Phase 4: Training final model")
            final_model = MODEL_REGISTRY[best_model_name](
                task_type=task,
                random_state=self.random_state,
                **best_params,
            )
            with Timer(f"final {best_model_name} fit"):
                final_model.fit(X_train, y_train, X_val, y_val)

            # Phase 5: Ensemble
            if self.enable_ensemble and len(top_models) >= 2:
                log.info("Phase 5: Building stacking ensemble")
                base_model_objs = [r["model"] for r in top_models[:3]]
                ensemble = StackingEnsemble(
                    base_models=base_model_objs,
                    task_type=task,
                    cv_folds=min(3, self.cv_folds),
                    random_state=self.random_state,
                )
                try:
                    ensemble.fit(X_train, y_train)
                    self.best_model = ensemble
                    log.info("Using stacking ensemble as final model.")
                except Exception as e:
                    log.warning(f"Ensemble failed, using single best model: {e}")
                    self.best_model = final_model
            else:
                self.best_model = final_model

            # Phase 6: Evaluate
            metrics = {}
            if X_val is not None and y_val is not None:
                metrics = self._evaluate(self.best_model, X_val, y_val, task)
                mlflow.log_metrics(metrics)
                log.info(f"Validation metrics: {metrics}")

            # Phase 7: Persist
            self.models_dir.mkdir(parents=True, exist_ok=True)
            model_path = self.models_dir / "best_model.joblib"
            pipeline_path = self.models_dir / "feature_pipeline.joblib"

            import joblib
            joblib.dump(self.best_model, model_path, compress=3)
            joblib.dump(self.feature_pipeline, pipeline_path, compress=3)
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(pipeline_path))

            # Log feature importances
            fi = self.best_model.get_feature_importance() if hasattr(self.best_model, "get_feature_importance") else None
            if fi is not None:
                fi_path = self.models_dir / "feature_importances.json"
                fi.to_json(fi_path)
                mlflow.log_artifact(str(fi_path))

            log.info(f"Training complete. Model saved to {model_path}")

        return {
            "best_model": self.best_model,
            "feature_pipeline": self.feature_pipeline,
            "metrics": metrics,
            "run_id": self.run_id,
            "model_name": best_model_name,
            "best_params": best_params,
        }

    def _evaluate(self, model, X_val, y_val, task: str) -> dict:
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
            mean_squared_error, mean_absolute_error, r2_score,
        )
        import numpy as np

        y_pred = model.predict(X_val)
        metrics: dict = {}

        if task == "classification":
            metrics["accuracy"] = float(accuracy_score(y_val, y_pred))
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_val, y_pred))
            metrics["f1_weighted"] = float(
                f1_score(y_val, y_pred, average="weighted", zero_division=0)
            )
            try:
                proba = model.predict_proba(X_val)
                if proba.shape[1] == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_val, proba[:, 1]))
                else:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_val, proba, multi_class="ovr", average="weighted")
                    )
            except Exception:
                pass
        else:
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_val, y_pred)))
            metrics["mae"] = float(mean_absolute_error(y_val, y_pred))
            metrics["r2"] = float(r2_score(y_val, y_pred))

        return metrics

    def _resolve_task(self, y: pd.Series) -> str:
        if self.task_type != "auto":
            return self.task_type
        return "classification" if y.nunique() <= 20 else "regression"
