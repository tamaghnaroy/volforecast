"""
HEAVY — High-frEquency-bAsed VolatilitY model (Shephard & Sheppard, 2010).

Two-equation system driven by realized measures:
  h_t = omega_h + alpha_h * RM_{t-1} + beta_h * h_{t-1}     [conditional variance]
  mu_t = omega_mu + alpha_mu * RM_{t-1} + beta_mu * mu_{t-1} [realized measure mean]

The key insight is that realized measures provide a more informative signal
than squared returns, leading to faster adaptation to volatility changes.

Reference: Shephard & Sheppard (2010), Journal of Financial Econometrics.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from numba import njit
from scipy.optimize import minimize

from volforecast.core.base import BaseForecaster, ForecastResult, ModelSpec
from volforecast.core.targets import VolatilityTarget, TargetSpec


@njit(cache=True)
def heavy_filter(
    returns: NDArray[np.float64],
    rm: NDArray[np.float64],
    omega_h: float,
    alpha_h: float,
    beta_h: float,
) -> NDArray[np.float64]:
    """HEAVY conditional variance filter.

    h_t = omega_h + alpha_h * RM_{t-1} + beta_h * h_{t-1}

    Parameters
    ----------
    returns : array, shape (T,)
    rm : array, shape (T,)
        Realized measure series (e.g., RV).
    omega_h, alpha_h, beta_h : float
        HEAVY variance equation parameters.

    Returns
    -------
    h : array, shape (T,)
        Conditional variance series.
    """
    T = returns.shape[0]
    h = np.empty(T, dtype=np.float64)
    # Initialize at unconditional: E[h] = omega_h / (1 - alpha_h * E[RM]/E[h] - beta_h)
    # Approximate with sample mean of RM
    mean_rm = 0.0
    for i in range(T):
        mean_rm += rm[i]
    mean_rm /= T
    unc_h = omega_h / max(1.0 - alpha_h - beta_h, 1e-8) if alpha_h + beta_h < 1.0 else mean_rm
    h[0] = max(unc_h, 1e-20)
    for t in range(1, T):
        h[t] = omega_h + alpha_h * rm[t - 1] + beta_h * h[t - 1]
        h[t] = max(h[t], 1e-20)
    return h


class HEAVYForecaster(BaseForecaster):
    """HEAVY model forecaster (Shephard & Sheppard, 2010).

    Uses daily returns and a realized measure (RV by default) to produce
    conditional variance forecasts that adapt faster than pure GARCH.
    """

    def __init__(self) -> None:
        self._params: dict[str, float] = {}
        self._returns: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._rm: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._h: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name="HEAVY",
            abbreviation="HEAVY",
            family="GARCH",
            target=VolatilityTarget.CONDITIONAL_VARIANCE,
            assumptions=(
                "realized-measure driven variance",
                "faster adaptation than GARCH",
                "requires high-frequency data",
            ),
            complexity="O(T) MLE",
            reference="Shephard & Sheppard (2010), JoFE",
            extends=("GARCH",),
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "HEAVYForecaster":
        if realized_measures is None or "RV" not in realized_measures:
            raise ValueError("HEAVY requires realized_measures={'RV': array}")

        self._returns = np.asarray(returns, dtype=np.float64)
        self._rm = np.asarray(realized_measures["RV"], dtype=np.float64)

        T = min(len(self._returns), len(self._rm))
        self._returns = self._returns[:T]
        self._rm = self._rm[:T]

        if T == 0:
            raise ValueError("returns and RV must contain at least one observation")

        var_r = np.var(self._returns)
        mean_rm = np.mean(self._rm)

        def neg_loglik(params):
            omega_h, alpha_h, beta_h = params
            if omega_h <= 0 or alpha_h < 0 or beta_h < 0:
                return 1e10
            if alpha_h + beta_h >= 1.0:
                return 1e10
            h = heavy_filter(self._returns, self._rm, omega_h, alpha_h, beta_h)
            ll = -0.5 * np.sum(np.log(2 * np.pi * h) + self._returns ** 2 / h)
            if np.isnan(ll) or np.isinf(ll):
                return 1e10
            return -ll

        x0 = np.array([var_r * 0.05, 0.3, 0.6])
        bounds = [(1e-10, None), (1e-6, 0.999), (1e-6, 0.999)]
        res = minimize(neg_loglik, x0, method="L-BFGS-B", bounds=bounds)

        self._params = {
            "omega_h": res.x[0],
            "alpha_h": res.x[1],
            "beta_h": res.x[2],
        }
        self._h = heavy_filter(
            self._returns, self._rm,
            self._params["omega_h"], self._params["alpha_h"], self._params["beta_h"],
        )
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        omega_h = self._params["omega_h"]
        alpha_h = self._params["alpha_h"]
        beta_h = self._params["beta_h"]
        persistence = alpha_h + beta_h

        last_h = self._h[-1]
        last_rm = self._rm[-1]
        unc_var = omega_h / max(1.0 - persistence, 1e-8)

        forecasts = np.empty(horizon, dtype=np.float64)
        h_1 = omega_h + alpha_h * last_rm + beta_h * last_h
        forecasts[0] = h_1

        for h_idx in range(1, horizon):
            # For h>1, E[RM_{t+h-1}] ≈ h_{t+h-1} (realized measure tracks variance)
            forecasts[h_idx] = unc_var + persistence ** h_idx * (h_1 - unc_var)

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.CONDITIONAL_VARIANCE,
                horizon=horizon,
            ),
            model_name="HEAVY",
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
        if new_realized is None or "RV" not in new_realized:
            raise ValueError("HEAVY update requires RV")

        new_r = np.asarray(new_returns, dtype=np.float64)
        new_rm = np.asarray(new_realized["RV"], dtype=np.float64)

        if len(new_r) != len(new_rm):
            raise ValueError("new_returns and new_realized['RV'] must have the same length")

        omega_h = self._params["omega_h"]
        alpha_h = self._params["alpha_h"]
        beta_h = self._params["beta_h"]

        ret_list = list(self._returns)
        rm_list = list(self._rm)
        h_list = list(self._h)
        for i in range(len(new_r)):
            new_h = omega_h + alpha_h * rm_list[-1] + beta_h * h_list[-1]
            ret_list.append(new_r[i])
            rm_list.append(new_rm[i])
            h_list.append(max(new_h, 1e-20))
        self._returns = np.array(ret_list, dtype=np.float64)
        self._rm = np.array(rm_list, dtype=np.float64)
        self._h = np.array(h_list, dtype=np.float64)

    def get_params(self) -> dict[str, Any]:
        return self._params.copy()
