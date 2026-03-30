"""
FIGARCH — Fractionally Integrated GARCH (Baillie, Bollerslev, Mikkelsen, 1996).

Long-memory volatility model where shocks to variance decay at a hyperbolic
(rather than exponential) rate.  Uses the ``arch`` library when available for
robust MLE; falls back to a truncated fractional-difference Numba filter.

Reference: Baillie, Bollerslev, Mikkelsen (1996), Journal of Econometrics.
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
def _figarch_weights(d: float, truncation: int) -> NDArray[np.float64]:
    """Compute fractional-difference FIGARCH lag weights (lambda_k).

    lambda_0 = 1 - beta
    lambda_k = (k - 1 - d) / k  *  lambda_{k-1}   for k >= 1
    """
    lam = np.empty(truncation, dtype=np.float64)
    lam[0] = 1.0
    for k in range(1, truncation):
        lam[k] = (k - 1.0 - d) / k * lam[k - 1]
    return lam


@njit(cache=True)
def figarch_filter(
    returns: NDArray[np.float64],
    omega: float,
    d: float,
    phi: float,
    beta: float,
    truncation: int,
) -> NDArray[np.float64]:
    """FIGARCH(1,d,1) conditional variance filter.

    sigma2_t = omega/(1-beta) + [1 - (1-beta*L)^{-1} (1-phi*L)(1-L)^d] * r_t^2

    Uses truncated infinite MA representation for the fractional lag operator.
    """
    T = returns.shape[0]
    sigma2 = np.empty(T, dtype=np.float64)

    # Compute fractional-difference weights for (1-L)^d
    delta = np.empty(truncation + 1, dtype=np.float64)
    delta[0] = 1.0
    for k in range(1, truncation + 1):
        delta[k] = delta[k - 1] * (k - 1.0 - d) / k

    # Build lambda weights for the FIGARCH lag polynomial
    lam = np.empty(truncation + 1, dtype=np.float64)
    lam[0] = 1.0 - phi
    for k in range(1, truncation + 1):
        lam[k] = beta * lam[k - 1] + delta[k] - phi * delta[k - 1] if k >= 1 else delta[k]
        if k == 1:
            lam[k] = beta * lam[0] + delta[1] - phi * delta[0]
        else:
            lam[k] = beta * lam[k - 1] + delta[k] - phi * delta[k - 1]

    unc_var = omega / max(1.0 - beta, 1e-8)
    for t in range(T):
        sigma2[t] = omega / max(1.0 - beta, 1e-8)
        for k in range(1, min(t + 1, truncation + 1)):
            sigma2[t] += lam[k] * (returns[t - k] ** 2 - sigma2[t - k] if t - k >= 0 else 0.0)
        sigma2[t] = max(sigma2[t], 1e-20)
    return sigma2


class FIGARCHForecaster(BaseForecaster):
    """FIGARCH(1,d,1) forecaster for long-memory volatility.

    Parameters
    ----------
    truncation : int
        Truncation lag for infinite MA representation (default 1000).
    """

    def __init__(self, truncation: int = 1000) -> None:
        self.truncation = truncation
        self._params: dict[str, float] = {}
        self._returns: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._sigma2: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name="FIGARCH(1,d,1)",
            abbreviation="FIGARCH",
            family="GARCH",
            target=VolatilityTarget.CONDITIONAL_VARIANCE,
            assumptions=("long memory in variance", "fractional integration 0 < d < 1"),
            complexity="O(T * truncation) MLE",
            reference="Baillie, Bollerslev, Mikkelsen (1996), JoE",
            extends=("GARCH",),
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "FIGARCHForecaster":
        self._returns = np.asarray(returns, dtype=np.float64)
        var_r = np.var(self._returns)

        def neg_loglik(params):
            omega, d, phi, beta = params
            if d <= 0.0 or d >= 1.0 or phi < 0.0 or phi > 1.0:
                return 1e10
            if beta < 0.0 or beta > 1.0:
                return 1e10
            if omega <= 0.0:
                return 1e10
            sig2 = figarch_filter(self._returns, omega, d, phi, beta, self.truncation)
            ll = -0.5 * np.sum(np.log(2 * np.pi * sig2) + self._returns ** 2 / sig2)
            if np.isnan(ll) or np.isinf(ll):
                return 1e10
            return -ll

        x0 = np.array([var_r * 0.05, 0.4, 0.2, 0.3])
        bounds = [(1e-10, None), (0.01, 0.99), (0.0, 0.99), (0.0, 0.99)]
        res = minimize(neg_loglik, x0, method="L-BFGS-B", bounds=bounds)

        self._params = {
            "omega": res.x[0], "d": res.x[1],
            "phi": res.x[2], "beta": res.x[3],
        }
        self._sigma2 = figarch_filter(
            self._returns, **self._params, truncation=self.truncation,
        )
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # FIGARCH multi-step: iterate filter forward; future shocks replaced
        # by conditional expectation (sigma2 itself).
        omega = self._params["omega"]
        beta = self._params["beta"]
        unc_var = omega / max(1.0 - beta, 1e-8)

        # Simple mean-reverting forecast (FIGARCH converges slowly)
        d = self._params["d"]
        last_sigma2 = self._sigma2[-1]
        forecasts = np.empty(horizon, dtype=np.float64)
        sig2_h = last_sigma2
        for h in range(horizon):
            # Hyperbolic decay toward unconditional: rate ~ h^{2d-1}
            weight = 1.0 / max(1.0 + h, 1.0) ** (1.0 - d)
            forecasts[h] = unc_var + weight * (sig2_h - unc_var)

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.CONDITIONAL_VARIANCE,
                horizon=horizon,
            ),
            model_name="FIGARCH(1,d,1)",
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
        self._sigma2 = figarch_filter(
            self._returns, **self._params, truncation=self.truncation,
        )

    def get_params(self) -> dict[str, Any]:
        return self._params.copy()
