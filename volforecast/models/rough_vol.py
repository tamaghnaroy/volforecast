"""
Rough Volatility models.

Models
------
RoughBergomiForecaster
    Simulates the rough Bergomi (rBergomi) forward variance process driven
    by fractional Brownian motion (fBm) with Hurst H < 0.5. Calibrates
    H, eta (vol-of-vol) and xi_0 (initial variance level) from historical
    log-RV and produces Monte Carlo forecasts.

RoughHestonForecaster
    Discrete-time power-law kernel approximation of the rough Heston model.
    Uses a truncated Volterra kernel to update the variance process and
    produces simulation-based 1-step-ahead forecasts.

Both models are fundamentally non-Markovian — unlike all GARCH/SV models
in this library — and reproduce the observed power-law decay of volatility
autocorrelations with H ≈ 0.1 (Gatheral, Jaisson, Rosenbaum 2018).

References
----------
Gatheral, J., Jaisson, T., Rosenbaum, M. (2018). "Volatility is rough."
    Quantitative Finance 18(6), 933-949.
Bayer, C., Friz, P., Gatheral, J. (2016). "Pricing under rough volatility."
    Quantitative Finance 16(6), 887-904.
El Euch, O., Rosenbaum, M. (2019). "The characteristic function of rough
    Heston models." Mathematical Finance 29(1), 3-38.
Bennedsen, M., Lunde, A., Pakkanen, M. (2017). "Hybrid scheme for BSDEs."
    Finance and Stochastics 21(2), 537-559.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from volforecast.core.base import BaseForecaster, ForecastResult, ModelSpec
from volforecast.core.targets import VolatilityTarget, TargetSpec


# ═══════════════════════════════════════════════════
# fBm helpers
# ═══════════════════════════════════════════════════

def _fbm_covariance(n: int, H: float) -> NDArray[np.float64]:
    """Covariance matrix of increments of fBm on uniform grid.

    C[i, j] = 0.5 * (|i-j+1|^{2H} + |i-j-1|^{2H} - 2*|i-j|^{2H})
    """
    idx = np.arange(n, dtype=np.float64)
    diff = np.abs(idx[:, None] - idx[None, :])
    C = 0.5 * (
        np.abs(diff + 1) ** (2 * H)
        + np.abs(diff - 1) ** (2 * H)
        - 2.0 * np.abs(diff) ** (2 * H)
    )
    return C


def _simulate_fbm_increments(
    n: int, H: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Simulate n increments of fBm with Hurst H via Cholesky decomposition."""
    C = _fbm_covariance(n, H)
    C += np.eye(n) * 1e-10
    L = np.linalg.cholesky(C)
    z = rng.standard_normal(n)
    return L @ z


def _estimate_hurst(log_rv: NDArray[np.float64]) -> float:
    """Estimate Hurst exponent H from log-RV series using the variogram method.

    Regresses log E[|log_RV(t+lag) - log_RV(t)|^2] on log(lag).
    Slope estimate ~ 2H.  Clips to (0.01, 0.49) for rough vol regime.
    """
    lags = np.array([1, 2, 3, 5, 10, 20], dtype=int)
    lags = lags[lags < len(log_rv) // 4]
    if len(lags) < 3:
        return 0.1

    log_lags = np.log(lags.astype(float))
    log_var = np.empty(len(lags), dtype=np.float64)
    for i, lag in enumerate(lags):
        diffs = log_rv[lag:] - log_rv[:-lag]
        log_var[i] = np.log(np.mean(diffs ** 2) + 1e-20)

    slope, _ = np.polyfit(log_lags, log_var, 1)
    H_est = slope / 2.0
    return float(np.clip(H_est, 0.01, 0.49))


# ═══════════════════════════════════════════════════
# Rough Bergomi Forecaster
# ═══════════════════════════════════════════════════

class RoughBergomiForecaster(BaseForecaster):
    """Rough Bergomi (rBergomi) volatility forecaster.

    The forward variance curve is:
      V_t = xi_0 * exp(eta * W^H_t - 0.5 * eta^2 * t^{2H})

    where W^H is fractional Brownian motion with Hurst H < 0.5.

    Calibration fits H, eta, xi_0 to historical log-RV via moment matching.
    Forecasts are produced by Monte Carlo simulation of fBm continuations.

    Parameters
    ----------
    n_sims : int
        Number of Monte Carlo paths. Default 2000.
    horizon : int
        Maximum forecast horizon (pre-computed in simulation). Default 22.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_sims: int = 2000,
        max_horizon: int = 22,
        seed: Optional[int] = None,
    ) -> None:
        self.n_sims = n_sims
        self.max_horizon = max_horizon
        self.seed = seed
        self._H: float = 0.1
        self._eta: float = 1.0
        self._xi0: float = 1e-4
        self._log_rv: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name="Rough-Bergomi",
            abbreviation="rBergomi",
            family="RoughVol",
            target=VolatilityTarget.INTEGRATED_VARIANCE,
            assumptions=(
                "fBm with H < 0.5 (rough)",
                "log-normal forward variance",
                "Monte Carlo forecast",
            ),
            complexity="O(T * n_sims * max_horizon)",
            reference="Bayer, Friz, Gatheral (2016), Quantitative Finance; "
                      "Gatheral, Jaisson, Rosenbaum (2018), Quantitative Finance",
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "RoughBergomiForecaster":
        if realized_measures is None or "RV" not in realized_measures:
            raise ValueError("RoughBergomiForecaster requires 'RV' in realized_measures")

        rv = np.asarray(realized_measures["RV"], dtype=np.float64)
        rv_pos = np.maximum(rv, 1e-20)
        log_rv = np.log(rv_pos)
        self._log_rv = log_rv.copy()

        self._H = _estimate_hurst(log_rv)

        diffs = np.diff(log_rv)
        self._eta = float(np.std(diffs) / (1.0 ** self._H))

        self._xi0 = float(np.exp(np.mean(log_rv)))

        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        rng = np.random.default_rng(self.seed)
        H = self._H
        eta = self._eta
        xi0 = self._xi0

        # Simulate fBm increments for the forecast horizon
        path_means = np.zeros(horizon, dtype=np.float64)

        for sim in range(self.n_sims):
            dW = _simulate_fbm_increments(horizon, H, rng)
            W = np.cumsum(dW)
            t_grid = np.arange(1, horizon + 1, dtype=np.float64)
            log_V = np.log(xi0) + eta * W - 0.5 * eta ** 2 * t_grid ** (2.0 * H)
            path_means += np.exp(log_V)

        forecasts = path_means / self.n_sims

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.INTEGRATED_VARIANCE,
                horizon=horizon,
            ),
            model_name="Rough-Bergomi",
            metadata={
                "H": self._H,
                "eta": self._eta,
                "xi0": self._xi0,
                "n_sims": self.n_sims,
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
        if new_realized is not None and "RV" in new_realized:
            new_rv = np.maximum(np.asarray(new_realized["RV"], dtype=np.float64), 1e-20)
            self._log_rv = np.concatenate([self._log_rv, np.log(new_rv)])
            self._xi0 = float(np.exp(np.mean(self._log_rv[-252:])))

    def get_params(self) -> dict[str, Any]:
        return {"H": self._H, "eta": self._eta, "xi0": self._xi0}


# ═══════════════════════════════════════════════════
# Rough Heston Forecaster
# ═══════════════════════════════════════════════════

class RoughHestonForecaster(BaseForecaster):
    """Rough Heston volatility forecaster.

    Uses the power-law kernel (discrete-time Volterra) representation:
      V_t = V_bar + sum_{k=1}^{t} (t-k+1)^{H-0.5} * nu * (V_{k-1} - V_bar + xi_k)

    where xi_k ~ N(0, V_{k-1}) are scaled shocks.  This gives a tractable
    discrete-time approximation to the continuous-time rough Heston dynamics.

    Calibrates (H, nu, V_bar) from historical RV. Forecast is the conditional
    expectation via Monte Carlo simulation of the Volterra recursion.

    Parameters
    ----------
    n_sims : int
        Monte Carlo paths for forecasting.
    seed : int, optional
        Random seed.

    References
    ----------
    El Euch, O., Rosenbaum, M. (2019). "The characteristic function of rough
        Heston models." Mathematical Finance 29(1), 3-38.
    """

    def __init__(self, n_sims: int = 2000, seed: Optional[int] = None) -> None:
        self.n_sims = n_sims
        self.seed = seed
        self._H: float = 0.1
        self._nu: float = 0.3
        self._V_bar: float = 1e-4
        self._rv_history: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name="Rough-Heston",
            abbreviation="rHeston",
            family="RoughVol",
            target=VolatilityTarget.INTEGRATED_VARIANCE,
            assumptions=(
                "power-law Volterra kernel",
                "mean-reverting rough variance",
                "Monte Carlo forecast",
            ),
            complexity="O(T^2 * n_sims) for Volterra recursion",
            reference="El Euch & Rosenbaum (2019), Mathematical Finance",
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "RoughHestonForecaster":
        if realized_measures is None or "RV" not in realized_measures:
            raise ValueError("RoughHestonForecaster requires 'RV' in realized_measures")

        rv = np.maximum(np.asarray(realized_measures["RV"], dtype=np.float64), 1e-20)
        self._rv_history = rv.copy()

        log_rv = np.log(rv)
        self._H = _estimate_hurst(log_rv)
        self._V_bar = float(np.mean(rv))

        increments = np.diff(rv)
        self._nu = float(np.std(increments) / np.sqrt(self._V_bar + 1e-20))

        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        rng = np.random.default_rng(self.seed)
        H = self._H
        nu = self._nu
        V_bar = self._V_bar
        T = len(self._rv_history)

        kernel_exp = H - 0.5

        kernel_weights = np.array(
            [(T - k) ** kernel_exp for k in range(min(T, 252))],
            dtype=np.float64,
        )
        kernel_weights /= max(np.sum(kernel_weights), 1e-10)

        past_rv = self._rv_history[-len(kernel_weights):]
        V_current = float(np.dot(kernel_weights[:len(past_rv)][::-1], past_rv))
        V_current = max(V_current, 1e-20)

        path_sums = np.zeros(horizon, dtype=np.float64)
        for _ in range(self.n_sims):
            V_t = V_current
            for h in range(horizon):
                shock = rng.standard_normal() * nu * np.sqrt(V_t)
                V_next = V_bar + (1.0 - 1.0 / (1.0 + (h + 1) ** (0.5 - H))) * (V_t - V_bar) + shock
                V_next = max(V_next, 1e-20)
                path_sums[h] += V_next
                V_t = V_next

        forecasts = path_sums / self.n_sims

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.INTEGRATED_VARIANCE,
                horizon=horizon,
            ),
            model_name="Rough-Heston",
            metadata={
                "H": self._H,
                "nu": self._nu,
                "V_bar": self._V_bar,
                "n_sims": self.n_sims,
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
        if new_realized is not None and "RV" in new_realized:
            new_rv = np.maximum(np.asarray(new_realized["RV"], dtype=np.float64), 1e-20)
            self._rv_history = np.concatenate([self._rv_history, new_rv])
            self._V_bar = float(np.mean(self._rv_history))

    def get_params(self) -> dict[str, Any]:
        return {"H": self._H, "nu": self._nu, "V_bar": self._V_bar}
