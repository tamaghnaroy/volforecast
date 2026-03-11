"""
Realized GARCH model (Hansen, Huang, Shek, 2012).

Joint model for returns and realized measures:
  r_t = sqrt(h_t) * z_t                          [return equation]
  log(h_t) = omega + beta*log(h_{t-1}) + gamma*log(x_{t-1})  [GARCH equation]  
  log(x_t) = xi + phi*log(h_t) + tau(z_t) + u_t  [measurement equation]

where x_t is the realized measure (e.g., RV), and tau(z) = tau_1*z + tau_2*(z^2 - 1)
captures the leverage effect.

Reference: Hansen, Huang, Shek (2012). "Realized GARCH: A Joint Model for
Returns and Realized Measures of Volatility." JAE 27(6), 877-906.
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
def _rgarch_filter(
    returns: NDArray[np.float64],
    log_rv: NDArray[np.float64],
    omega: float,
    beta: float,
    gamma: float,
    xi: float,
    phi: float,
    tau1: float,
    tau2: float,
    sigma_u2: float,
) -> tuple:
    """Realized GARCH log-likelihood filter.

    Returns (log_h, log_likelihood).
    """
    T = returns.shape[0]
    log_h = np.empty(T, dtype=np.float64)

    # Initialize with unconditional value
    if abs(1.0 - beta - phi * gamma) > 1e-8:
        log_h0 = (omega + gamma * xi) / (1.0 - beta - phi * gamma)
    else:
        log_h0 = np.log(np.var(returns) + 1e-10)
    log_h[0] = log_h0

    total_ll = 0.0

    for t in range(T):
        h_t = np.exp(log_h[t])

        # Return contribution: -0.5*(log(2*pi) + log(h_t) + r_t^2/h_t)
        ll_r = -0.5 * (np.log(2.0 * np.pi) + log_h[t] + returns[t] ** 2 / max(h_t, 1e-20))

        # Measurement equation residual
        z_t = returns[t] / np.sqrt(max(h_t, 1e-20))
        tau_z = tau1 * z_t + tau2 * (z_t ** 2 - 1.0)
        mu_x = xi + phi * log_h[t] + tau_z
        resid_u = log_rv[t] - mu_x

        # Measurement contribution: -0.5*(log(2*pi*sigma_u2) + resid_u^2/sigma_u2)
        ll_x = -0.5 * (np.log(2.0 * np.pi * sigma_u2) + resid_u ** 2 / sigma_u2)

        total_ll += ll_r + ll_x

        # Update for next period
        if t < T - 1:
            log_h[t + 1] = omega + beta * log_h[t] + gamma * log_rv[t]

    return log_h, total_ll


class RealizedGARCHForecaster(BaseForecaster):
    """Realized GARCH(1,1) forecaster.

    Jointly models returns and a realized measure (typically log-RV),
    exploiting high-frequency information for improved volatility estimates.
    """

    def __init__(self) -> None:
        self._params: dict[str, float] = {}
        self._returns: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._log_rv: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._log_h: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name="Realized GARCH(1,1)",
            abbreviation="RGARCH",
            family="GARCH",
            target=VolatilityTarget.CONDITIONAL_VARIANCE,
            assumptions=(
                "joint model returns + realized measure",
                "log-variance dynamics",
                "measurement equation with leverage",
            ),
            complexity="O(T) MLE",
            reference="Hansen, Huang, Shek (2012), JAE",
            extends=("GARCH", "HEAVY"),
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "RealizedGARCHForecaster":
        if realized_measures is None or "RV" not in realized_measures:
            raise ValueError("Realized GARCH requires realized_measures={'RV': array}")

        self._returns = np.asarray(returns, dtype=np.float64)
        rv = np.asarray(realized_measures["RV"], dtype=np.float64)
        self._log_rv = np.log(np.maximum(rv, 1e-20))

        T = len(self._returns)
        min_len = min(T, len(self._log_rv))
        self._returns = self._returns[:min_len]
        self._log_rv = self._log_rv[:min_len]

        if min_len == 0:
            raise ValueError("returns and realized RV must contain at least one observation")

        # MLE estimation
        def neg_loglik(params):
            omega, beta, gamma, xi, phi, tau1, tau2, log_sig_u2 = params
            sigma_u2 = np.exp(log_sig_u2)
            if beta < -0.999 or beta > 0.999:
                return 1e10
            _, ll = _rgarch_filter(
                self._returns, self._log_rv,
                omega, beta, gamma, xi, phi, tau1, tau2, sigma_u2,
            )
            if np.isnan(ll) or np.isinf(ll):
                return 1e10
            return -ll

        # Starting values
        var_r = np.var(self._returns)
        mean_lrv = np.mean(self._log_rv)
        x0 = np.array([
            mean_lrv * 0.05,  # omega
            0.6,               # beta
            0.3,               # gamma
            mean_lrv * 0.1,   # xi
            0.9,               # phi
            -0.1,              # tau1
            0.05,              # tau2
            np.log(0.3),       # log(sigma_u^2)
        ])

        res = minimize(neg_loglik, x0, method="Nelder-Mead",
                       options={"maxiter": 10000, "xatol": 1e-8})

        p = res.x
        self._params = {
            "omega": p[0], "beta": p[1], "gamma": p[2],
            "xi": p[3], "phi": p[4], "tau1": p[5], "tau2": p[6],
            "sigma_u2": np.exp(p[7]),
        }

        self._log_h, _ = _rgarch_filter(
            self._returns, self._log_rv,
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], np.exp(p[7]),
        )
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        omega = self._params["omega"]
        beta = self._params["beta"]
        gamma = self._params["gamma"]
        xi = self._params["xi"]
        phi = self._params["phi"]

        last_log_h = self._log_h[-1]
        last_log_rv = self._log_rv[-1]

        forecasts = np.empty(horizon, dtype=np.float64)

        # h-step ahead: E[log h_{t+h}] using iterated expectations
        # E[log x_t] = xi + phi * log h_t  (since E[tau(z)] = 0, E[u] = 0)
        log_h_h = omega + beta * last_log_h + gamma * last_log_rv
        forecasts[0] = np.exp(log_h_h)

        for h in range(1, horizon):
            # E[log x_{t+h-1}] = xi + phi * log_h_h
            E_log_x = xi + phi * log_h_h
            log_h_h = omega + beta * log_h_h + gamma * E_log_x
            forecasts[h] = np.exp(log_h_h)

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.CONDITIONAL_VARIANCE,
                horizon=horizon,
            ),
            model_name="Realized GARCH(1,1)",
            metadata={"params": self._params.copy()},
        )

    def update(
        self,
        new_returns: NDArray[np.float64],
        new_realized: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> None:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        if new_realized is None or "RV" not in new_realized:
            raise ValueError("Realized GARCH update requires RV")

        new_r = np.asarray(new_returns, dtype=np.float64)
        new_rv = np.asarray(new_realized["RV"], dtype=np.float64)

        if len(new_r) != len(new_rv):
            raise ValueError("new_returns and new_realized['RV'] must have the same length")

        new_log_rv = np.log(np.maximum(new_rv, 1e-20))

        omega = self._params["omega"]
        beta = self._params["beta"]
        gamma = self._params["gamma"]

        for i in range(len(new_r)):
            new_log_h = omega + beta * self._log_h[-1] + gamma * self._log_rv[-1]
            self._returns = np.append(self._returns, new_r[i])
            self._log_rv = np.append(self._log_rv, new_log_rv[i])
            self._log_h = np.append(self._log_h, new_log_h)

    def get_params(self) -> dict[str, Any]:
        return self._params.copy()
