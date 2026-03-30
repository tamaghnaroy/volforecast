"""
GARCH-MIDAS model (Engle, Ghysels, Sohn, 2013).

Multiplicative component model separating short-run and long-run volatility:
  r_t = sqrt(tau_t * g_t) * z_t,   z_t ~ N(0, 1)

Short-run component (daily GARCH):
  g_t = (1 - alpha - beta) + alpha * r_{t-1}^2 / tau_{t-1} + beta * g_{t-1}

Long-run component (MIDAS with Beta-weighting):
  log(tau_t) = m + theta * sum_{k=1}^{K} w_k(omega1, omega2) * X_{t-k}

where X is a low-frequency covariate (e.g., monthly RV rolling window)
and w_k are Beta polynomial weights.

Reference: Engle, Ghysels, Sohn (2013), Review of Economics and Statistics.
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
def _beta_weights(K: int, omega1: float, omega2: float) -> NDArray[np.float64]:
    """Compute normalized Beta polynomial weights for MIDAS.

    w_k = k^{omega1-1} * (K-k)^{omega2-1} / sum(...)
    """
    w = np.empty(K, dtype=np.float64)
    for k in range(K):
        x = (k + 1.0) / (K + 1.0)
        w[k] = x ** (omega1 - 1.0) * (1.0 - x) ** (omega2 - 1.0)
    w_sum = 0.0
    for k in range(K):
        w_sum += w[k]
    if w_sum > 1e-20:
        for k in range(K):
            w[k] /= w_sum
    else:
        for k in range(K):
            w[k] = 1.0 / K
    return w


@njit(cache=True)
def _garch_midas_filter(
    returns: NDArray[np.float64],
    tau: NDArray[np.float64],
    alpha: float,
    beta: float,
) -> NDArray[np.float64]:
    """Short-run GARCH(1,1) component filter given long-run tau.

    g_t = (1-alpha-beta) + alpha * r_{t-1}^2 / tau_{t-1} + beta * g_{t-1}
    sigma2_t = tau_t * g_t
    """
    T = returns.shape[0]
    g = np.empty(T, dtype=np.float64)
    sigma2 = np.empty(T, dtype=np.float64)

    g[0] = 1.0  # unconditional g = 1
    sigma2[0] = max(tau[0] * g[0], 1e-20)

    for t in range(1, T):
        r2_scaled = returns[t - 1] ** 2 / max(tau[t - 1], 1e-20)
        g[t] = (1.0 - alpha - beta) + alpha * r2_scaled + beta * g[t - 1]
        g[t] = max(g[t], 1e-10)
        sigma2[t] = max(tau[t] * g[t], 1e-20)

    return sigma2


class GARCHMIDASForecaster(BaseForecaster):
    """GARCH-MIDAS forecaster (Engle, Ghysels, Sohn, 2013).

    Decomposes volatility into a short-run GARCH component and a long-run
    MIDAS component driven by rolling realized variance windows.

    Parameters
    ----------
    K : int
        Number of MIDAS lags for long-run component (default 22, ~1 month).
    """

    def __init__(self, K: int = 22) -> None:
        self.K = K
        self._params: dict[str, float] = {}
        self._returns: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._sigma2: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._tau: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._rv_lf: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name="GARCH-MIDAS",
            abbreviation="GMIDAS",
            family="GARCH",
            target=VolatilityTarget.CONDITIONAL_VARIANCE,
            assumptions=(
                "multiplicative short/long-run components",
                "Beta polynomial MIDAS weighting",
                "long-run driven by rolling realized variance",
            ),
            complexity="O(T*K) MLE",
            reference="Engle, Ghysels, Sohn (2013), REStat",
            extends=("GARCH",),
        )

    def _compute_tau(self, rv_lf, m, theta, omega1, omega2, K):
        """Compute long-run component tau from low-frequency RV."""
        T = len(rv_lf)
        tau = np.empty(T, dtype=np.float64)
        w = _beta_weights(K, omega1, omega2)
        for t in range(T):
            weighted_rv = 0.0
            for k in range(K):
                idx = t - 1 - k
                if idx >= 0:
                    weighted_rv += w[k] * rv_lf[idx]
                else:
                    weighted_rv += w[k] * rv_lf[0]
            tau[t] = np.exp(m + theta * weighted_rv)
        return tau

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "GARCHMIDASForecaster":
        self._returns = np.asarray(returns, dtype=np.float64)
        T = len(self._returns)
        K = self.K

        # Construct low-frequency covariate: rolling K-day realized variance
        r2 = self._returns ** 2
        rv_lf = np.empty(T, dtype=np.float64)
        for t in range(T):
            start = max(0, t - K)
            rv_lf[t] = np.mean(r2[start:t + 1])
        self._rv_lf = rv_lf

        var_r = np.var(self._returns)

        def neg_loglik(params):
            m, theta, log_om1, log_om2, alpha, beta = params
            omega1 = np.exp(log_om1) + 1.0
            omega2 = np.exp(log_om2) + 1.0
            if alpha < 0 or beta < 0 or alpha + beta >= 1.0:
                return 1e10

            tau = self._compute_tau(rv_lf, m, theta, omega1, omega2, K)
            sigma2 = _garch_midas_filter(self._returns, tau, alpha, beta)
            ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + self._returns ** 2 / sigma2)
            if np.isnan(ll) or np.isinf(ll):
                return 1e10
            return -ll

        x0 = np.array([np.log(var_r), 1.0, 0.0, 0.0, 0.05, 0.90])
        bounds = [(-20, 5), (-10, 50), (-3, 3), (-3, 3), (1e-6, 0.4), (0.5, 0.999)]
        res = minimize(neg_loglik, x0, method="L-BFGS-B", bounds=bounds)

        p = res.x
        omega1 = np.exp(p[2]) + 1.0
        omega2 = np.exp(p[3]) + 1.0

        self._params = {
            "m": p[0], "theta": p[1],
            "omega1": omega1, "omega2": omega2,
            "alpha": p[4], "beta": p[5],
        }

        self._tau = self._compute_tau(rv_lf, p[0], p[1], omega1, omega2, K)
        self._sigma2 = _garch_midas_filter(self._returns, self._tau, p[4], p[5])
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        alpha = self._params["alpha"]
        beta = self._params["beta"]
        persistence = alpha + beta
        last_sigma2 = self._sigma2[-1]
        last_tau = self._tau[-1]

        # Short-run g converges to 1; long-run tau assumed constant for forecast
        unc_var = last_tau  # g -> 1
        forecasts = np.empty(horizon, dtype=np.float64)
        forecasts[0] = last_sigma2
        for h in range(1, horizon):
            forecasts[h] = unc_var + persistence ** h * (last_sigma2 - unc_var)

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.CONDITIONAL_VARIANCE,
                horizon=horizon,
            ),
            model_name="GARCH-MIDAS",
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

        # Recompute rolling RV and tau
        T = len(self._returns)
        K = self.K
        r2 = self._returns ** 2
        rv_lf = np.empty(T, dtype=np.float64)
        for t in range(T):
            start = max(0, t - K)
            rv_lf[t] = np.mean(r2[start:t + 1])
        self._rv_lf = rv_lf

        self._tau = self._compute_tau(
            rv_lf, self._params["m"], self._params["theta"],
            self._params["omega1"], self._params["omega2"], K,
        )
        self._sigma2 = _garch_midas_filter(
            self._returns, self._tau, self._params["alpha"], self._params["beta"],
        )

    def get_params(self) -> dict[str, Any]:
        return self._params.copy()
