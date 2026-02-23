"""
PyTorch neural network model for tabular data.
Implements a configurable MLP with batch normalization, dropout, and early stopping.
"""
from typing import Optional

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

log = get_logger(__name__)


class NeuralNetModel(BaseModel):
    """PyTorch-based MLP for tabular classification and regression.

    Features: configurable hidden dims, dropout, batch norm, early stopping.

    Example:
        >>> model = NeuralNetModel(hidden_dims=[256, 128, 64], max_epochs=100)
        >>> model.fit(X_train, y_train, X_val, y_val)
    """

    def __init__(
        self,
        task_type: str = "auto",
        hidden_dims: list[int] = None,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 512,
        max_epochs: int = 200,
        patience: int = 20,
        weight_decay: float = 1e-4,
        batch_norm: bool = True,
        device: str = "auto",
        random_state: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(name="neural_net", task_type=task_type, random_state=random_state)
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.weight_decay = weight_decay
        self.batch_norm = batch_norm
        self.device = device
        self.params = kwargs

        self._encoder = None
        self._n_classes: int = 0

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> "NeuralNetModel":
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            log.error("PyTorch not installed. Cannot train NeuralNetModel.")
            raise

        task = self._resolve_task(y_train)
        self.task_type = task
        device = self._get_device()

        X_tr = torch.tensor(X_train.values.astype(np.float32), device=device)
        n_features = X_tr.shape[1]

        if task == "classification":
            self._n_classes = int(y_train.nunique())
            y_tr = torch.tensor(y_train.values.astype(np.int64), device=device)
        else:
            self._n_classes = 1
            y_tr = torch.tensor(y_train.values.astype(np.float32), device=device)

        train_ds = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        self._model = self._build_network(n_features, nn).to(device)
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        if task == "classification":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        best_val_loss = float("inf")
        no_improve = 0
        best_state = None

        for epoch in range(self.max_epochs):
            self._model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                out = self._model(X_batch)
                if task == "regression":
                    out = out.squeeze(1)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()

            if X_val is not None and y_val is not None:
                val_loss = self._eval_loss(X_val, y_val, criterion, device, task)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= self.patience:
                    log.info(f"Early stopping at epoch {epoch + 1}")
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._model.eval()
        self._feature_names = list(X_train.columns)
        self._is_fitted = True
        self._device = device
        log.info(f"NeuralNet ({task}) fitted for {epoch + 1} epochs.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        self._check_fitted()
        X_t = torch.tensor(X.values.astype(np.float32), device=self._device)
        self._model.eval()
        with torch.no_grad():
            out = self._model(X_t)
        if self.task_type == "classification":
            return torch.argmax(out, dim=1).cpu().numpy()
        return out.squeeze(1).cpu().numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        self._check_fitted()
        if self.task_type != "classification":
            raise NotImplementedError
        X_t = torch.tensor(X.values.astype(np.float32), device=self._device)
        self._model.eval()
        with torch.no_grad():
            out = self._model(X_t)
        probs = torch.softmax(out, dim=1).cpu().numpy()
        return probs

    def _eval_loss(self, X_val, y_val, criterion, device, task):
        import torch
        X_t = torch.tensor(X_val.values.astype(np.float32), device=device)
        if task == "classification":
            y_t = torch.tensor(y_val.values.astype(np.int64), device=device)
        else:
            y_t = torch.tensor(y_val.values.astype(np.float32), device=device)
        self._model.eval()
        with torch.no_grad():
            out = self._model(X_t)
            if task == "regression":
                out = out.squeeze(1)
            return criterion(out, y_t).item()

    def _build_network(self, n_features: int, nn):
        import torch.nn as nn

        layers = []
        in_dim = n_features
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            in_dim = hidden_dim

        out_dim = self._n_classes if self.task_type == "classification" else 1
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def _get_device(self):
        try:
            import torch
            if self.device == "auto":
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return torch.device(self.device)
        except ImportError:
            return "cpu"

    def _resolve_task(self, y: pd.Series) -> str:
        if self.task_type != "auto":
            return self.task_type
        return "classification" if y.nunique() <= 20 else "regression"
