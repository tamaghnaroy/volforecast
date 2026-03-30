"""
Score-driven (GAS / DCS) volatility model (Creal, Koopman, Lucas, 2013).

Generalized Autoregressive Score model for volatility:
  f_{t+1} = omega + A * s_t + B * f_t

where f_t is the time-varying parameter (log-variance), and s_t is the
scaled score of the conditional density at time t.

For Gaussian innovations:
  s_t = (r_t^2 / sigma_t^2 - 1)  (scaled score of log-variance)

For Student-t(nu) innovations:
  s_t = ((nu + 1) * r_t^2 / (nu - 2 + r_t^2 / sigma_t^2) / sigma_t^2 - 1)

The exponential link maps f_t to variance: sigma_t^2 = exp(f_t).

References:
- Creal, Koopman, Lucas (2013), Journal of Applied Econometrics.
- Harvey (2013), "Dynamic Models for Volatility and Heavy Tails."
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
def gas_normal_filter(
    returns: NDArray[np.float64],
    omega: float,
    A: float,
    B: float,
) -> NDArray[np.float64]:
    """GAS(1,1) filter with Gaussian innovations and exponential link.

    f_{t+1} = omega + A * s_t + B * f_t
    sigma2_t = exp(f_t)
    s_t = r_t^2 / sigma2_t - 1  (scaled score for log-variance)
    """
    T = returns.shape[0]
    sigma2 = np.empty(T, dtype=np.float64)

    # Initialize f at unconditional: f = omega / (1 - B)
    if abs(1.0 - B) > 1e-8:
        f = omega / (1.0 - B)
    else:
        var_sum = 0.0
        for i in range(T):
            var_sum += returns[i] ** 2
        f = np.log(var_sum / T)

    for t in range(T):
        sigma2[t] = np.exp(f)
        s_t = returns[t] ** 2 / max(sigma2[t], 1e-20) - 1.0
        f = omega + A * s_t + B * f
    return sigma2


@njit(cache=True)
def gas_student_filter(
    returns: NDArray[np.float64],
    omega: float,
    A: float,
    B: float,
    nu: float,
) -> NDArray[np.float64]:
    """GAS(1,1) filter with Student-t innovations and exponential link.

    s_t = ((nu+1) * r_t^2 / ((nu-2)*sigma2_t + r_t^2) - 1)
    """
    T = returns.shape[0]
    sigma2 = np.empty(T, dtype=np.float64)

    if abs(1.0 - B) > 1e-8:
        f = omega / (1.0 - B)
    else:
        var_sum = 0.0
        for i in range(T):
            var_sum += returns[i] ** 2
        f = np.log(var_sum / T)

    for t in range(T):
        sigma2[t] = np.exp(f)
        r2 = returns[t] ** 2
        sig2 = max(sigma2[t], 1e-20)
        # Student-t scaled score for log-variance
        s_t = (nu + 1.0) * r2 / ((nu - 2.0) * sig2 + r2) - 1.0
        f = omega + A * s_t + B * f
    return sigma2


class GASVolForecaster(BaseForecaster):
    """Score-driven (GAS) volatility model.

    Parameters
    ----------
    dist : str
        Innovation distribution: "normal" or "t".
    """

    def __init__(self, dist: str = "normal") -> None:
        if dist not in ("normal", "t"):
            raise ValueError("dist must be 'normal' or 't'")
        self.dist = dist
        self._params: dict[str, float] = {}
        self._returns: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._sigma2: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name=f"GAS(1,1)-{self.dist}",
            abbreviation="GAS",
            family="GAS",
            target=VolatilityTarget.CONDITIONAL_VARIANCE,
            assumptions=(
                "score-driven dynamics",
                f"{self.dist} innovations",
                "exponential link for variance",
            ),
            complexity="O(T) MLE",
            reference="Creal, Koopman, Lucas (2013), JAE",
            extends=(),
        )

    def _run_filter(self, returns, params):
        if self.dist == "normal":
            return gas_normal_filter(returns, params["omega"], params["A"], params["B"])
        else:
            return gas_student_filter(
                returns, params["omega"], params["A"], params["B"], params["nu"]
            )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "GASVolForecaster":
        self._returns = np.asarray(returns, dtype=np.float64)
        T = len(self._returns)

        if self.dist == "normal":
            def neg_loglik(params):
                omega, A, B = params
                if abs(B) >= 0.9999 or A < -2.0 or A > 2.0:
                    return 1e10
                sig2 = gas_normal_filter(self._returns, omega, A, B)
                ll = -0.5 * np.sum(np.log(2 * np.pi * sig2) + self._returns ** 2 / sig2)
                if np.isnan(ll) or np.isinf(ll):
                    return 1e10
                return -ll

            x0 = np.array([0.01, 0.1, 0.98])
            bounds = [(-5.0, 5.0), (0.001, 1.5), (0.001, 0.9999)]
            res = minimize(neg_loglik, x0, method="L-BFGS-B", bounds=bounds)
            self._params = {"omega": res.x[0], "A": res.x[1], "B": res.x[2]}
        else:
            def neg_loglik_t(params):
                omega, A, B, log_nu_m2 = params
                nu = np.exp(log_nu_m2) + 2.01
                if abs(B) >= 0.9999:
                    return 1e10
                sig2 = gas_student_filter(self._returns, omega, A, B, nu)
                # Student-t log-likelihood
                from scipy.special import gammaln
                ll = np.sum(
                    gammaln((nu + 1) / 2) - gammaln(nu / 2)
                    - 0.5 * np.log(np.pi * (nu - 2) * sig2)
                    - (nu + 1) / 2 * np.log(1 + self._returns ** 2 / ((nu - 2) * sig2))
                )
                if np.isnan(ll) or np.isinf(ll):
                    return 1e10
                return -ll

            x0 = np.array([0.01, 0.1, 0.98, np.log(5.0)])
            bounds = [(-5.0, 5.0), (0.001, 1.5), (0.001, 0.9999), (np.log(0.1), np.log(50.0))]
            res = minimize(neg_loglik_t, x0, method="L-BFGS-B", bounds=bounds)
            nu = np.exp(res.x[3]) + 2.01
            self._params = {
                "omega": res.x[0], "A": res.x[1], "B": res.x[2], "nu": nu,
            }

        self._sigma2 = self._run_filter(self._returns, self._params)
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        omega = self._params["omega"]
        B = self._params["B"]

        # Unconditional log-variance
        unc_f = omega / max(1.0 - B, 1e-8)
        unc_var = np.exp(unc_f)

        last_sigma2 = self._sigma2[-1]
        last_f = np.log(max(last_sigma2, 1e-20))
        last_r = self._returns[-1]

        # One-step score
        if self.dist == "normal":
            s = last_r ** 2 / max(last_sigma2, 1e-20) - 1.0
        else:
            nu = self._params["nu"]
            r2 = last_r ** 2
            s = (nu + 1.0) * r2 / ((nu - 2.0) * max(last_sigma2, 1e-20) + r2) - 1.0

        A = self._params["A"]
        f_next = omega + A * s + B * last_f

        forecasts = np.empty(horizon, dtype=np.float64)
        forecasts[0] = np.exp(f_next)
        for h in range(1, horizon):
            # E[s_{t+h}] = 0 under correct model => f_{t+h+1} = omega + B * f_{t+h}
            f_next = omega + B * f_next
            forecasts[h] = np.exp(f_next)

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.CONDITIONAL_VARIANCE,
                horizon=horizon,
            ),
            model_name=f"GAS(1,1)-{self.dist}",
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
        self._sigma2 = self._run_filter(self._returns, self._params)

    def get_params(self) -> dict[str, Any]:
        return self._params.copy()
