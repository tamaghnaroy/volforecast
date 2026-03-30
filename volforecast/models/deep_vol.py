"""
DeepVol: volatility forecasting from intraday data with dilated causal CNNs.

Architecturally distinct from LSTM/Transformer wrappers (which consume daily
feature vectors).  DeepVol consumes raw intraday return bars directly and
learns to aggregate them via a WaveNet-style stack of dilated causal
convolutions.

PyTorch is an optional dependency.  When not installed, the class raises
an informative ImportError on instantiation.

References
----------
Mugica, M., Trottier, D., Godin, F. (2022/2024).
    "DeepVol: Volatility Forecasting from High-Frequency Data with Dilated
    Causal Convolutions." Quantitative Finance 24(8), 1105-1127.
    arXiv:2210.04797.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from volforecast.core.base import BaseForecaster, ForecastResult, ModelSpec
from volforecast.core.targets import VolatilityTarget, TargetSpec


# ── PyTorch-optional imports ──────────────────────────────────────────────────

def _require_torch() -> Any:
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError(
            "DeepVolForecaster requires PyTorch. "
            "Install with: pip install torch"
        ) from exc


def _build_deepvol_net(
    n_bars: int,
    hidden: int,
    dilations: tuple[int, ...],
    torch: Any,
) -> Any:
    """Build the dilated causal CNN backbone."""
    nn = torch.nn

    layers: list[Any] = []
    in_ch = 1
    for dil in dilations:
        layers += [
            nn.Conv1d(
                in_ch, hidden,
                kernel_size=2, dilation=dil,
                padding=dil,
            ),
            nn.ReLU(),
        ]
        in_ch = hidden

    class DeepVolNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv_stack = nn.Sequential(*layers)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x: Any) -> Any:
            # x : (batch, 1, T_bars)
            out = self.conv_stack(x)
            # causal: slice off right-padding
            out = out[:, :, :x.shape[2]]
            # global average pool over time axis
            pooled = out.mean(dim=2)
            return self.head(pooled).squeeze(-1)

    return DeepVolNet()


class DeepVolForecaster(BaseForecaster):
    """DeepVol: dilated causal CNN volatility forecaster (Mugica et al., 2024).

    Accepts raw intraday return bars (shape T days × M bars per day) and
    forecasts the next-day realized variance.  Unlike LSTM/Transformer wrappers
    that consume daily summary features, this model ingests the full intraday
    return sequence and learns multi-scale temporal patterns via dilated
    causal convolutions with an exponential receptive field.

    Input contract
    --------------
    realized_measures["intraday"] : array (T, M)
        Each row is one day; each column is one intraday return bar.
    realized_measures["RV"] : array (T,)
        Target: next-day realized variance (shifted by 1 in fit()).

    Parameters
    ----------
    hidden : int
        Number of channels in each dilated conv layer. Default 32.
    dilations : tuple of int
        Dilation schedule. Default (1, 2, 4, 8, 16).
    epochs : int
        Training epochs. Default 50.
    lr : float
        Learning rate. Default 1e-3.
    batch_size : int
        Batch size. Default 32.
    seed : int, optional
        Random seed for reproducibility.

    Notes
    -----
    - Marked as experimental: accuracy depends heavily on the quantity and
      quality of intraday data.
    - PyTorch is a hard dependency; the class raises ImportError if absent.
    """

    _experimental = True

    def __init__(
        self,
        hidden: int = 32,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        seed: Optional[int] = None,
    ) -> None:
        self.hidden = hidden
        self.dilations = dilations
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self._net: Any = None
        self._rv_mean: float = 1.0
        self._rv_std: float = 1.0
        self._ret_std: float = 1.0
        self._last_intraday: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name="DeepVol",
            abbreviation="DeepVol",
            family="ML",
            target=VolatilityTarget.INTEGRATED_VARIANCE,
            assumptions=(
                "raw intraday returns as input (no pre-aggregation)",
                "dilated causal CNN receptive field",
                "stationary intraday return scale",
            ),
            complexity="O(T * M * depth) training",
            reference="Mugica, Trottier, Godin (2024), Quantitative Finance",
            extends=("LSTM_Vol", "Transformer_Vol"),
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "DeepVolForecaster":
        torch = _require_torch()

        if realized_measures is None or "intraday" not in realized_measures:
            raise ValueError(
                "DeepVolForecaster requires realized_measures={'intraday': (T, M) array, "
                "'RV': (T,) array}"
            )
        if "RV" not in realized_measures:
            raise ValueError("DeepVolForecaster requires 'RV' in realized_measures")

        intraday = np.asarray(realized_measures["intraday"], dtype=np.float64)
        rv = np.asarray(realized_measures["RV"], dtype=np.float64)

        if intraday.ndim == 1:
            intraday = intraday.reshape(-1, 1)

        T, M = intraday.shape

        self._ret_std = float(np.std(intraday) + 1e-10)
        X = intraday / self._ret_std

        self._rv_mean = float(np.mean(rv))
        self._rv_std = float(np.std(rv) + 1e-10)
        y = (rv - self._rv_mean) / self._rv_std

        X_feat = X[:-1]
        y_tgt = y[1:]

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        self._net = _build_deepvol_net(M, self.hidden, self.dilations, torch)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        n_train = len(X_feat)
        for epoch in range(self.epochs):
            perm = np.random.permutation(n_train)
            for start in range(0, n_train, self.batch_size):
                idx = perm[start:start + self.batch_size]
                X_b = torch.tensor(
                    X_feat[idx].reshape(len(idx), 1, M), dtype=torch.float32
                )
                y_b = torch.tensor(y_tgt[idx], dtype=torch.float32)
                optimizer.zero_grad()
                pred = self._net(X_b)
                loss = loss_fn(pred, y_b)
                loss.backward()
                optimizer.step()

        self._last_intraday = X[-1].copy()
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        torch = _require_torch()

        self._net.eval()
        forecasts = np.empty(horizon, dtype=np.float64)
        last = self._last_intraday.copy()

        with torch.no_grad():
            for h in range(horizon):
                x_t = torch.tensor(
                    last.reshape(1, 1, -1), dtype=torch.float32
                )
                pred_scaled = float(self._net(x_t).item())
                rv_pred = pred_scaled * self._rv_std + self._rv_mean
                forecasts[h] = max(rv_pred, 1e-20)
                last = last * 0.0

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.INTEGRATED_VARIANCE,
                horizon=horizon,
            ),
            model_name="DeepVol",
            metadata={
                "hidden": self.hidden,
                "dilations": list(self.dilations),
                "epochs": self.epochs,
            },
        )

    def update(
        self,
        new_returns: NDArray[np.float64],
        new_realized: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> None:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        if new_realized is not None and "intraday" in new_realized:
            new_intraday = np.asarray(new_realized["intraday"], dtype=np.float64)
            if new_intraday.ndim == 2:
                self._last_intraday = new_intraday[-1] / self._ret_std
            else:
                self._last_intraday = new_intraday / self._ret_std

    def get_params(self) -> dict[str, Any]:
        return {
            "hidden": self.hidden,
            "dilations": list(self.dilations),
            "epochs": self.epochs,
            "lr": self.lr,
        }
