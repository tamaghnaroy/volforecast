"""
Machine Learning volatility forecaster wrappers.

Provides BaseForecaster-compatible wrappers for:
- Random Forest (scikit-learn, always available)
- LSTM (PyTorch, optional dependency)
- Transformer (PyTorch, optional dependency)

All wrappers use a common feature generation contract:
lagged squared returns and optionally lagged realized measures as inputs.

Feature matrix X_t = [r_{t-1}^2, ..., r_{t-p}^2, (RV_{t-1}, ..., RV_{t-p})]
Target y_t = r_t^2 (or RV_t if available)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from volforecast.core.base import BaseForecaster, ForecastResult, ModelSpec
from volforecast.core.targets import VolatilityTarget, TargetSpec


def _build_features(
    returns: NDArray[np.float64],
    n_lags: int,
    rv: Optional[NDArray[np.float64]] = None,
) -> tuple:
    """Build lagged feature matrix and target vector.

    Returns (X, y, valid_start_index).
    """
    T = len(returns)
    r2 = returns ** 2

    n_feat = n_lags if rv is None else 2 * n_lags
    X = np.empty((T - n_lags, n_feat), dtype=np.float64)
    y = np.empty(T - n_lags, dtype=np.float64)

    for t in range(n_lags, T):
        for lag in range(n_lags):
            X[t - n_lags, lag] = r2[t - 1 - lag]
        if rv is not None:
            for lag in range(n_lags):
                X[t - n_lags, n_lags + lag] = rv[t - 1 - lag]
        # Target: RV if available, else squared return
        y[t - n_lags] = rv[t] if rv is not None else r2[t]

    return X, y, n_lags


# ═══════════════════════════════════════════════════
# Random Forest Volatility Forecaster
# ═══════════════════════════════════════════════════

class RFVolForecaster(BaseForecaster):
    """Random Forest volatility forecaster.

    Uses scikit-learn RandomForestRegressor with lagged features.

    Parameters
    ----------
    n_lags : int
        Number of lagged squared returns / RV features (default 22).
    n_estimators : int
        Number of trees (default 100).
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self, n_lags: int = 22, n_estimators: int = 100, random_state: int = 42,
    ) -> None:
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model = None
        self._params: dict[str, Any] = {}
        self._returns: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._rv: Optional[NDArray[np.float64]] = None
        self._last_X: Optional[NDArray[np.float64]] = None
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name="Random Forest Vol",
            abbreviation="RF",
            family="ML",
            target=VolatilityTarget.CONDITIONAL_VARIANCE,
            assumptions=("non-parametric ensemble", "lagged feature inputs"),
            complexity="O(T * n_estimators * log T)",
            reference="Breiman (2001), Machine Learning",
            extends=(),
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "RFVolForecaster":
        from sklearn.ensemble import RandomForestRegressor

        self._returns = np.asarray(returns, dtype=np.float64)
        rv = None
        if realized_measures is not None and "RV" in realized_measures:
            rv = np.asarray(realized_measures["RV"], dtype=np.float64)
            self._rv = rv

        X, y, _ = _build_features(self._returns, self.n_lags, rv)

        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X, y)
        self._last_X = X[-1:]
        self._params = {
            "n_lags": self.n_lags,
            "n_estimators": self.n_estimators,
            "n_features": X.shape[1],
        }
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        r2 = self._returns ** 2
        rv = self._rv
        n_lags = self.n_lags
        T = len(self._returns)

        forecasts = np.empty(horizon, dtype=np.float64)
        # Build current feature vector
        recent_r2 = list(r2[-(n_lags):])
        recent_rv = list(rv[-(n_lags):]) if rv is not None else None

        for h in range(horizon):
            feat = np.array(recent_r2[::-1], dtype=np.float64)  # most recent first
            if recent_rv is not None:
                feat_rv = np.array(recent_rv[::-1], dtype=np.float64)
                feat = np.concatenate([feat, feat_rv])
            pred = self._model.predict(feat.reshape(1, -1))[0]
            pred = max(pred, 1e-20)
            forecasts[h] = pred
            # Shift features for next step
            recent_r2.append(pred)
            recent_r2.pop(0)
            if recent_rv is not None:
                recent_rv.append(pred)
                recent_rv.pop(0)

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.CONDITIONAL_VARIANCE,
                horizon=horizon,
            ),
            model_name="Random Forest Vol",
            metadata={"params": self._params.copy()},
        )

    def update(
        self,
        new_returns: NDArray[np.float64],
        new_realized: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> None:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        new_r = np.asarray(new_returns, dtype=np.float64)
        self._returns = np.concatenate([self._returns, new_r])
        if new_realized is not None and "RV" in new_realized:
            new_rv = np.asarray(new_realized["RV"], dtype=np.float64)
            if self._rv is not None:
                self._rv = np.concatenate([self._rv, new_rv])
            else:
                self._rv = new_rv

    def get_params(self) -> dict[str, Any]:
        return self._params.copy()


# ═══════════════════════════════════════════════════
# LSTM Volatility Forecaster
# ═══════════════════════════════════════════════════

class LSTMVolForecaster(BaseForecaster):
    """LSTM volatility forecaster (requires PyTorch).

    Parameters
    ----------
    n_lags : int
        Sequence length for LSTM input (default 22).
    hidden_size : int
        LSTM hidden dimension (default 32).
    n_epochs : int
        Training epochs (default 50).
    lr : float
        Learning rate (default 1e-3).
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        n_lags: int = 22,
        hidden_size: int = 32,
        n_epochs: int = 50,
        lr: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        self.n_lags = n_lags
        self.hidden_size = hidden_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.random_state = random_state
        self._model = None
        self._params: dict[str, Any] = {}
        self._returns: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._rv: Optional[NDArray[np.float64]] = None
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name="LSTM Vol",
            abbreviation="LSTM",
            family="ML",
            target=VolatilityTarget.CONDITIONAL_VARIANCE,
            assumptions=("recurrent neural network", "sequential feature inputs"),
            complexity="O(T * hidden_size^2 * n_epochs)",
            reference="Hochreiter & Schmidhuber (1997), Neural Computation",
            extends=(),
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "LSTMVolForecaster":
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError(
                "PyTorch is required for LSTMVolForecaster. "
                "Install with: pip install torch  (or use volforecast[ml])"
            )

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._returns = np.asarray(returns, dtype=np.float64)
        rv = None
        if realized_measures is not None and "RV" in realized_measures:
            rv = np.asarray(realized_measures["RV"], dtype=np.float64)
            self._rv = rv

        X, y, _ = _build_features(self._returns, self.n_lags, rv)
        n_features_per_step = 1 if rv is None else 2

        # Reshape X to (N, seq_len, n_features_per_step)
        N = X.shape[0]
        X_seq = X[:, :self.n_lags].reshape(N, self.n_lags, 1)
        if rv is not None:
            X_rv = X[:, self.n_lags:].reshape(N, self.n_lags, 1)
            X_seq = np.concatenate([X_seq, X_rv], axis=2)

        X_t = torch.tensor(X_seq, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        # Simple LSTM model
        class _LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)
                self.softplus = nn.Softplus()

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                return self.softplus(out)

        model = _LSTMModel(n_features_per_step, self.hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        model.train()
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            pred = model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()

        model.eval()
        self._model = model
        self._n_features_per_step = n_features_per_step
        self._params = {
            "n_lags": self.n_lags,
            "hidden_size": self.hidden_size,
            "n_epochs": self.n_epochs,
            "n_features": n_features_per_step,
        }
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        import torch

        r2 = self._returns ** 2
        rv = self._rv
        n_lags = self.n_lags

        forecasts = np.empty(horizon, dtype=np.float64)
        recent_r2 = list(r2[-n_lags:])
        recent_rv = list(rv[-n_lags:]) if rv is not None else None

        self._model.eval()
        with torch.no_grad():
            for h in range(horizon):
                seq = np.array(recent_r2, dtype=np.float32).reshape(1, n_lags, 1)
                if recent_rv is not None:
                    seq_rv = np.array(recent_rv, dtype=np.float32).reshape(1, n_lags, 1)
                    seq = np.concatenate([seq, seq_rv], axis=2)
                x_t = torch.tensor(seq, dtype=torch.float32)
                pred = self._model(x_t).item()
                pred = max(pred, 1e-20)
                forecasts[h] = pred
                recent_r2.append(pred)
                recent_r2.pop(0)
                if recent_rv is not None:
                    recent_rv.append(pred)
                    recent_rv.pop(0)

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.CONDITIONAL_VARIANCE,
                horizon=horizon,
            ),
            model_name="LSTM Vol",
            metadata={"params": self._params.copy()},
        )

    def update(
        self,
        new_returns: NDArray[np.float64],
        new_realized: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> None:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        new_r = np.asarray(new_returns, dtype=np.float64)
        self._returns = np.concatenate([self._returns, new_r])
        if new_realized is not None and "RV" in new_realized:
            new_rv = np.asarray(new_realized["RV"], dtype=np.float64)
            if self._rv is not None:
                self._rv = np.concatenate([self._rv, new_rv])

    def get_params(self) -> dict[str, Any]:
        return self._params.copy()


# ═══════════════════════════════════════════════════
# Transformer Volatility Forecaster
# ═══════════════════════════════════════════════════

class TransformerVolForecaster(BaseForecaster):
    """Transformer-based volatility forecaster (requires PyTorch).

    Parameters
    ----------
    n_lags : int
        Sequence length (default 22).
    d_model : int
        Transformer embedding dimension (default 32).
    n_heads : int
        Number of attention heads (default 2).
    n_epochs : int
        Training epochs (default 50).
    lr : float
        Learning rate (default 1e-3).
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        n_lags: int = 22,
        d_model: int = 32,
        n_heads: int = 2,
        n_epochs: int = 50,
        lr: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        self.n_lags = n_lags
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_epochs = n_epochs
        self.lr = lr
        self.random_state = random_state
        self._model = None
        self._params: dict[str, Any] = {}
        self._returns: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._rv: Optional[NDArray[np.float64]] = None
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name="Transformer Vol",
            abbreviation="Transformer",
            family="ML",
            target=VolatilityTarget.CONDITIONAL_VARIANCE,
            assumptions=("self-attention mechanism", "sequential feature inputs"),
            complexity="O(T * n_lags^2 * d_model * n_epochs)",
            reference="Vaswani et al. (2017), NeurIPS",
            extends=(),
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "TransformerVolForecaster":
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError(
                "PyTorch is required for TransformerVolForecaster. "
                "Install with: pip install torch  (or use volforecast[ml])"
            )

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._returns = np.asarray(returns, dtype=np.float64)
        rv = None
        if realized_measures is not None and "RV" in realized_measures:
            rv = np.asarray(realized_measures["RV"], dtype=np.float64)
            self._rv = rv

        X, y, _ = _build_features(self._returns, self.n_lags, rv)
        n_features_per_step = 1 if rv is None else 2

        N = X.shape[0]
        X_seq = X[:, :self.n_lags].reshape(N, self.n_lags, 1)
        if rv is not None:
            X_rv = X[:, self.n_lags:].reshape(N, self.n_lags, 1)
            X_seq = np.concatenate([X_seq, X_rv], axis=2)

        X_t = torch.tensor(X_seq, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        class _TransformerModel(nn.Module):
            def __init__(self, input_size, d_model, n_heads, seq_len):
                super().__init__()
                self.input_proj = nn.Linear(input_size, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
                    batch_first=True, dropout=0.0,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
                self.fc = nn.Linear(d_model, 1)
                self.softplus = nn.Softplus()

            def forward(self, x):
                x = self.input_proj(x)
                x = self.encoder(x)
                x = self.fc(x[:, -1, :])
                return self.softplus(x)

        model = _TransformerModel(n_features_per_step, self.d_model, self.n_heads, self.n_lags)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        model.train()
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            pred = model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()

        model.eval()
        self._model = model
        self._n_features_per_step = n_features_per_step
        self._params = {
            "n_lags": self.n_lags,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_epochs": self.n_epochs,
            "n_features": n_features_per_step,
        }
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        import torch

        r2 = self._returns ** 2
        rv = self._rv
        n_lags = self.n_lags

        forecasts = np.empty(horizon, dtype=np.float64)
        recent_r2 = list(r2[-n_lags:])
        recent_rv = list(rv[-n_lags:]) if rv is not None else None

        self._model.eval()
        with torch.no_grad():
            for h in range(horizon):
                seq = np.array(recent_r2, dtype=np.float32).reshape(1, n_lags, 1)
                if recent_rv is not None:
                    seq_rv = np.array(recent_rv, dtype=np.float32).reshape(1, n_lags, 1)
                    seq = np.concatenate([seq, seq_rv], axis=2)
                x_t = torch.tensor(seq, dtype=torch.float32)
                pred = self._model(x_t).item()
                pred = max(pred, 1e-20)
                forecasts[h] = pred
                recent_r2.append(pred)
                recent_r2.pop(0)
                if recent_rv is not None:
                    recent_rv.append(pred)
                    recent_rv.pop(0)

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.CONDITIONAL_VARIANCE,
                horizon=horizon,
            ),
            model_name="Transformer Vol",
            metadata={"params": self._params.copy()},
        )

    def update(
        self,
        new_returns: NDArray[np.float64],
        new_realized: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> None:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        new_r = np.asarray(new_returns, dtype=np.float64)
        self._returns = np.concatenate([self._returns, new_r])
        if new_realized is not None and "RV" in new_realized:
            new_rv = np.asarray(new_realized["RV"], dtype=np.float64)
            if self._rv is not None:
                self._rv = np.concatenate([self._rv, new_rv])

    def get_params(self) -> dict[str, Any]:
        return self._params.copy()
