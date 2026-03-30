"""
GARCH-family volatility forecasters.

Wraps the ``arch`` library for estimation while providing the unified
BaseForecaster interface, explicit target declaration, and online update.

Models implemented:
- GARCH(1,1): Bollerslev (1986)
- EGARCH(1,1): Nelson (1991)
- GJR-GARCH(1,1): Glosten, Jagannathan, Runkle (1993)
- APARCH(1,1): Ding, Granger, Engle (1993)
- CGARCH(1,1): Engle & Lee (1999) — component GARCH
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from numba import njit

from volforecast.core.base import BaseForecaster, ForecastResult, ModelSpec
from volforecast.core.targets import VolatilityTarget, TargetSpec, COND_VAR_1STEP


# ═══════════════════════════════════════════════════
# Numba-optimized GARCH(1,1) filter (for speed)
# ═══════════════════════════════════════════════════

@njit(cache=True)
def garch11_filter(
    returns: NDArray[np.float64],
    omega: float,
    alpha: float,
    beta: float,
) -> NDArray[np.float64]:
    """GARCH(1,1) conditional variance filter.

    sigma2_t = omega + alpha * r_{t-1}^2 + beta * sigma2_{t-1}

    Parameters
    ----------
    returns : array, shape (T,)
    omega, alpha, beta : float
        GARCH parameters.

    Returns
    -------
    sigma2 : array, shape (T,)
        Conditional variance series.
    """
    T = returns.shape[0]
    sigma2 = np.empty(T, dtype=np.float64)
    # Initialize at unconditional variance
    unc_var = omega / max(1.0 - alpha - beta, 1e-8)
    sigma2[0] = unc_var
    for t in range(1, T):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
    return sigma2


@njit(cache=True)
def gjr_garch11_filter(
    returns: NDArray[np.float64],
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
) -> NDArray[np.float64]:
    """GJR-GARCH(1,1) filter with leverage effect.

    sigma2_t = omega + (alpha + gamma * I(r_{t-1}<0)) * r_{t-1}^2 + beta * sigma2_{t-1}
    """
    T = returns.shape[0]
    sigma2 = np.empty(T, dtype=np.float64)
    unc_var = omega / max(1.0 - alpha - 0.5 * gamma - beta, 1e-8)
    sigma2[0] = unc_var
    for t in range(1, T):
        leverage = gamma if returns[t - 1] < 0.0 else 0.0
        sigma2[t] = omega + (alpha + leverage) * returns[t - 1] ** 2 + beta * sigma2[t - 1]
    return sigma2


@njit(cache=True)
def egarch11_filter(
    returns: NDArray[np.float64],
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
) -> NDArray[np.float64]:
    """EGARCH(1,1) filter (Nelson, 1991).

    log(sigma2_t) = omega + alpha * (|z_{t-1}| - E|z|) + gamma * z_{t-1} + beta * log(sigma2_{t-1})
    where z_t = r_t / sigma_t, E|z| = sqrt(2/pi) for Gaussian.
    """
    T = returns.shape[0]
    log_sigma2 = np.empty(T, dtype=np.float64)
    sigma2 = np.empty(T, dtype=np.float64)
    e_abs_z = np.sqrt(2.0 / np.pi)

    # Initialize
    log_sigma2[0] = omega / max(1.0 - beta, 1e-8)
    sigma2[0] = np.exp(log_sigma2[0])

    for t in range(1, T):
        sig_prev = np.sqrt(max(sigma2[t - 1], 1e-20))
        z = returns[t - 1] / sig_prev
        log_sigma2[t] = (
            omega
            + alpha * (np.abs(z) - e_abs_z)
            + gamma * z
            + beta * log_sigma2[t - 1]
        )
        sigma2[t] = np.exp(log_sigma2[t])
    return sigma2


@njit(cache=True)
def cgarch11_filter(
    returns: NDArray[np.float64],
    omega: float,
    alpha: float,
    beta: float,
    phi: float,
    rho: float,
) -> NDArray[np.float64]:
    """Component GARCH(1,1) filter (Engle & Lee, 1999).

    q_t = omega + rho * q_{t-1} + phi * (r_{t-1}^2 - sigma2_{t-1})   [permanent]
    sigma2_t = q_t + alpha * (r_{t-1}^2 - q_{t-1}) + beta * (sigma2_{t-1} - q_{t-1})  [transitory]
    """
    T = returns.shape[0]
    sigma2 = np.empty(T, dtype=np.float64)
    q = np.empty(T, dtype=np.float64)

    unc_var = omega / max(1.0 - rho, 1e-8)
    q[0] = unc_var
    sigma2[0] = unc_var

    for t in range(1, T):
        q[t] = omega + rho * q[t - 1] + phi * (returns[t - 1] ** 2 - sigma2[t - 1])
        sigma2[t] = (
            q[t]
            + alpha * (returns[t - 1] ** 2 - q[t - 1])
            + beta * (sigma2[t - 1] - q[t - 1])
        )
        sigma2[t] = max(sigma2[t], 1e-10)
    return sigma2


@njit(cache=True)
def arch_q_filter(
    returns: NDArray[np.float64],
    omega: float,
    alphas: NDArray[np.float64],
) -> NDArray[np.float64]:
    """ARCH(q) conditional variance filter.

    sigma2_t = omega + sum_{i=1}^{q} alpha_i * r_{t-i}^2

    Parameters
    ----------
    returns : array, shape (T,)
    omega : float
        Intercept.
    alphas : array, shape (q,)
        ARCH lag coefficients.

    Returns
    -------
    sigma2 : array, shape (T,)
        Conditional variance series.
    """
    T = returns.shape[0]
    q = alphas.shape[0]
    sigma2 = np.empty(T, dtype=np.float64)
    # Initialize at unconditional variance
    alpha_sum = 0.0
    for i in range(q):
        alpha_sum += alphas[i]
    unc_var = omega / max(1.0 - alpha_sum, 1e-8)
    for t in range(T):
        s = omega
        for i in range(q):
            if t - 1 - i >= 0:
                s += alphas[i] * returns[t - 1 - i] ** 2
            else:
                s += alphas[i] * unc_var
        sigma2[t] = max(s, 1e-20)
    return sigma2


@njit(cache=True)
def ewma_filter(
    returns: NDArray[np.float64],
    lambda_: float,
) -> NDArray[np.float64]:
    """EWMA / RiskMetrics conditional variance filter.

    sigma2_t = lambda * sigma2_{t-1} + (1 - lambda) * r_{t-1}^2

    Parameters
    ----------
    returns : array, shape (T,)
    lambda_ : float
        Decay factor in (0, 1).

    Returns
    -------
    sigma2 : array, shape (T,)
        Conditional variance series.
    """
    T = returns.shape[0]
    sigma2 = np.empty(T, dtype=np.float64)
    # Initialize at sample variance
    var_sum = 0.0
    for i in range(T):
        var_sum += returns[i] ** 2
    sigma2[0] = var_sum / T
    for t in range(1, T):
        sigma2[t] = lambda_ * sigma2[t - 1] + (1.0 - lambda_) * returns[t - 1] ** 2
    return sigma2


# ═══════════════════════════════════════════════════
# GARCH Forecaster (wraps arch library for estimation)
# ═══════════════════════════════════════════════════

class GARCHForecaster(BaseForecaster):
    """GARCH(1,1) forecaster.

    Uses arch library for MLE estimation, Numba for fast filtering.

    Parameters
    ----------
    dist : str
        Error distribution: "normal", "t", "skewt", "ged".
    """

    def __init__(self, dist: str = "normal") -> None:
        self.dist = dist
        self._params: dict[str, float] = {}
        self._returns: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._sigma2: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name="GARCH(1,1)",
            abbreviation="GARCH",
            family="GARCH",
            target=VolatilityTarget.CONDITIONAL_VARIANCE,
            assumptions=("stationary returns", "symmetric response", "finite 4th moment"),
            complexity="O(T) MLE",
            reference="Bollerslev (1986), JoE",
            extends=("ARCH",),
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "GARCHForecaster":
        from arch import arch_model

        self._returns = np.asarray(returns, dtype=np.float64)
        r_scaled = self._returns * 100.0  # arch convention: percentage returns

        am = arch_model(r_scaled, vol="Garch", p=1, q=1, dist=self.dist, mean="Zero")
        res = am.fit(disp="off", **kwargs)

        # Extract parameters and rescale back
        self._params = {
            "omega": res.params["omega"] / 1e4,
            "alpha": res.params["alpha[1]"],
            "beta": res.params["beta[1]"],
        }
        self._arch_result = res

        # Run Numba filter for internal state
        self._sigma2 = garch11_filter(
            self._returns,
            self._params["omega"],
            self._params["alpha"],
            self._params["beta"],
        )
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        omega = self._params["omega"]
        alpha = self._params["alpha"]
        beta = self._params["beta"]
        persistence = alpha + beta

        # h-step forecast: sigma2_{T+h|T}
        last_sigma2 = self._sigma2[-1]
        unc_var = omega / max(1.0 - persistence, 1e-8)

        forecasts = np.empty(horizon, dtype=np.float64)
        sig2_1 = omega + alpha * self._returns[-1] ** 2 + beta * last_sigma2
        forecasts[0] = sig2_1
        for h in range(1, horizon):
            forecasts[h] = unc_var + persistence ** h * (sig2_1 - unc_var)

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.CONDITIONAL_VARIANCE,
                horizon=horizon,
            ),
            model_name="GARCH(1,1)",
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
        omega = self._params["omega"]
        alpha = self._params["alpha"]
        beta = self._params["beta"]

        new_sig2_list = []
        ret_list = list(self._returns)
        sig2_list = list(self._sigma2)
        for r in new_r:
            new_sig2 = omega + alpha * ret_list[-1] ** 2 + beta * sig2_list[-1]
            ret_list.append(r)
            sig2_list.append(new_sig2)
            new_sig2_list.append(new_sig2)
        self._returns = np.array(ret_list, dtype=np.float64)
        self._sigma2 = np.array(sig2_list, dtype=np.float64)

    def get_params(self) -> dict[str, Any]:
        return self._params.copy()


class EGARCHForecaster(BaseForecaster):
    """EGARCH(1,1) forecaster (Nelson, 1991)."""

    def __init__(self, dist: str = "normal") -> None:
        self.dist = dist
        self._params: dict[str, float] = {}
        self._returns: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._sigma2: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name="EGARCH(1,1)",
            abbreviation="EGARCH",
            family="GARCH",
            target=VolatilityTarget.CONDITIONAL_VARIANCE,
            assumptions=("log-variance dynamics", "asymmetric news impact"),
            complexity="O(T) MLE",
            reference="Nelson (1991), Econometrica",
            extends=("GARCH",),
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "EGARCHForecaster":
        from arch import arch_model

        self._returns = np.asarray(returns, dtype=np.float64)
        r_scaled = self._returns * 100.0

        am = arch_model(r_scaled, vol="EGARCH", p=1, o=1, q=1, dist=self.dist, mean="Zero")
        res = am.fit(disp="off", **kwargs)

        self._params = {
            "omega": res.params["omega"],
            "alpha": res.params["alpha[1]"],
            "gamma": res.params["gamma[1]"],
            "beta": res.params["beta[1]"],
        }
        self._arch_result = res

        # Numba filter (rescale omega for decimal returns)
        # EGARCH works in log space so we adjust omega
        self._sigma2 = egarch11_filter(
            self._returns,
            self._params["omega"] - 2.0 * np.log(100.0) * (1.0 - self._params["beta"]),
            self._params["alpha"],
            self._params["gamma"],
            self._params["beta"],
        )
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Use Numba filter for iterated h-step forecasts
        # For h>1, assume future z_t = 0 (conditional expectation)
        omega_adj = self._params["omega"] - 2.0 * np.log(100.0) * (1.0 - self._params["beta"])
        alpha = self._params["alpha"]
        gamma = self._params["gamma"]
        beta = self._params["beta"]
        e_abs_z = np.sqrt(2.0 / np.pi)

        last_sig2 = self._sigma2[-1]
        last_log_sig2 = np.log(max(last_sig2, 1e-20))
        last_r = self._returns[-1]

        forecasts = np.empty(horizon, dtype=np.float64)
        log_sig2_h = last_log_sig2

        for h in range(horizon):
            if h == 0:
                sig_prev = np.sqrt(max(last_sig2, 1e-20))
                z = last_r / sig_prev
                log_sig2_h = (
                    omega_adj
                    + alpha * (abs(z) - e_abs_z)
                    + gamma * z
                    + beta * last_log_sig2
                )
            else:
                # E[|z|] = sqrt(2/pi), E[z] = 0 under conditional expectation
                log_sig2_h = (
                    omega_adj
                    + alpha * (e_abs_z - e_abs_z)  # = 0
                    + beta * log_sig2_h
                )
            forecasts[h] = np.exp(log_sig2_h)

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.CONDITIONAL_VARIANCE,
                horizon=horizon,
            ),
            model_name="EGARCH(1,1)",
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
        new_r = np.asarray(new_returns, dtype=np.float64)
        self._returns = np.concatenate([self._returns, new_r])
        # Re-filter with new data
        omega_adj = self._params["omega"] - 2.0 * np.log(100.0) * (1.0 - self._params["beta"])
        self._sigma2 = egarch11_filter(
            self._returns, omega_adj,
            self._params["alpha"], self._params["gamma"], self._params["beta"],
        )

    def get_params(self) -> dict[str, Any]:
        return self._params.copy()


class GJRGARCHForecaster(BaseForecaster):
    """GJR-GARCH(1,1) forecaster (Glosten, Jagannathan, Runkle, 1993)."""

    def __init__(self, dist: str = "normal") -> None:
        self.dist = dist
        self._params: dict[str, float] = {}
        self._returns: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._sigma2: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name="GJR-GARCH(1,1)",
            abbreviation="GJR",
            family="GARCH",
            target=VolatilityTarget.CONDITIONAL_VARIANCE,
            assumptions=("leverage effect via indicator", "stationary"),
            complexity="O(T) MLE",
            reference="Glosten, Jagannathan, Runkle (1993), JoF",
            extends=("GARCH",),
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "GJRGARCHForecaster":
        from arch import arch_model

        self._returns = np.asarray(returns, dtype=np.float64)
        r_scaled = self._returns * 100.0

        am = arch_model(r_scaled, vol="Garch", p=1, o=1, q=1, dist=self.dist, mean="Zero")
        res = am.fit(disp="off", **kwargs)

        self._params = {
            "omega": res.params["omega"] / 1e4,
            "alpha": res.params["alpha[1]"],
            "gamma": res.params["gamma[1]"],
            "beta": res.params["beta[1]"],
        }
        self._arch_result = res

        self._sigma2 = gjr_garch11_filter(
            self._returns,
            self._params["omega"],
            self._params["alpha"],
            self._params["gamma"],
            self._params["beta"],
        )
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        omega = self._params["omega"]
        alpha = self._params["alpha"]
        gamma = self._params["gamma"]
        beta = self._params["beta"]
        persistence = alpha + 0.5 * gamma + beta

        last_sigma2 = self._sigma2[-1]
        last_r = self._returns[-1]
        unc_var = omega / max(1.0 - persistence, 1e-8)

        forecasts = np.empty(horizon, dtype=np.float64)
        leverage = gamma if last_r < 0.0 else 0.0
        sig2_h = omega + (alpha + leverage) * last_r ** 2 + beta * last_sigma2
        forecasts[0] = sig2_h

        for h in range(1, horizon):
            sig2_h = unc_var + persistence ** h * (forecasts[0] - unc_var)
            forecasts[h] = sig2_h

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.CONDITIONAL_VARIANCE,
                horizon=horizon,
            ),
            model_name="GJR-GARCH(1,1)",
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
        new_r = np.asarray(new_returns, dtype=np.float64)
        omega = self._params["omega"]
        alpha = self._params["alpha"]
        gamma = self._params["gamma"]
        beta = self._params["beta"]

        ret_list = list(self._returns)
        sig2_list = list(self._sigma2)
        for r in new_r:
            leverage = gamma if ret_list[-1] < 0.0 else 0.0
            new_sig2 = omega + (alpha + leverage) * ret_list[-1] ** 2 + beta * sig2_list[-1]
            ret_list.append(r)
            sig2_list.append(new_sig2)
        self._returns = np.array(ret_list, dtype=np.float64)
        self._sigma2 = np.array(sig2_list, dtype=np.float64)

    def get_params(self) -> dict[str, Any]:
        return self._params.copy()


class APARCHForecaster(BaseForecaster):
    """Asymmetric Power ARCH (Ding, Granger, Engle, 1993)."""

    def __init__(self, dist: str = "normal") -> None:
        self.dist = dist
        self._params: dict[str, float] = {}
        self._returns: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._sigma2: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name="APARCH(1,1)",
            abbreviation="APARCH",
            family="GARCH",
            target=VolatilityTarget.CONDITIONAL_VARIANCE,
            assumptions=("power transformation delta", "asymmetry gamma"),
            complexity="O(T) MLE, extra param delta",
            reference="Ding, Granger, Engle (1993), JIMF",
            extends=("GARCH", "GJR"),
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "APARCHForecaster":
        from arch import arch_model

        self._returns = np.asarray(returns, dtype=np.float64)
        r_scaled = self._returns * 100.0

        am = arch_model(r_scaled, vol="APARCH", p=1, o=1, q=1, dist=self.dist, mean="Zero")
        res = am.fit(disp="off", **kwargs)

        self._params = dict(res.params)
        self._arch_result = res
        cv = res.conditional_volatility
        self._sigma2 = np.asarray(cv if isinstance(cv, np.ndarray) else cv.values, dtype=np.float64) ** 2 / 1e4
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        if horizon == 1:
            fcast = self._arch_result.forecast(horizon=1)
            forecasts = fcast.variance.iloc[-1].values / 1e4
        else:
            # arch doesn't support analytic multi-step APARCH forecasts.
            # Use mean-reverting iterated forecast toward unconditional variance.
            params = self._params
            # APARCH persistence: alpha + gamma*kappa + beta (approximate)
            alpha_key = [k for k in params if 'alpha' in k.lower()]
            beta_key = [k for k in params if 'beta' in k.lower()]
            alpha_val = params[alpha_key[0]] if alpha_key else 0.05
            beta_val = params[beta_key[0]] if beta_key else 0.90
            persistence = min(alpha_val + beta_val, 0.9999)

            last_sigma2 = self._sigma2[-1]
            unc_var = np.mean(self._sigma2)  # empirical unconditional variance

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
            model_name="APARCH(1,1)",
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
        self._returns = np.concatenate([
            self._returns, np.asarray(new_returns, dtype=np.float64)
        ])
        # Re-filter conditional variance with fixed params on extended series
        from arch import arch_model
        r_scaled = self._returns * 100.0
        am = arch_model(r_scaled, vol="APARCH", p=1, o=1, q=1, dist=self.dist, mean="Zero")
        params_array = np.array(list(self._params.values()))
        res = am.fix(params_array)
        self._arch_result = res
        cv = res.conditional_volatility
        self._sigma2 = np.asarray(cv if isinstance(cv, np.ndarray) else cv.values, dtype=np.float64) ** 2 / 1e4

    def get_params(self) -> dict[str, Any]:
        return self._params.copy()


class CGARCHForecaster(BaseForecaster):
    """Component GARCH(1,1) (Engle & Lee, 1999).

    Decomposes variance into permanent (long-run) and transitory components.
    Uses custom Numba filter since arch doesn't support CGARCH directly.
    """

    def __init__(self) -> None:
        self._params: dict[str, float] = {}
        self._returns: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._sigma2: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name="Component GARCH(1,1)",
            abbreviation="CGARCH",
            family="GARCH",
            target=VolatilityTarget.CONDITIONAL_VARIANCE,
            assumptions=("permanent + transitory components", "mean-reverting"),
            complexity="O(T) MLE",
            reference="Engle & Lee (1999), in Engle (ed.)",
            extends=("GARCH",),
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "CGARCHForecaster":
        from scipy.optimize import minimize

        self._returns = np.asarray(returns, dtype=np.float64)
        T = len(self._returns)
        var_r = np.var(self._returns)

        def neg_loglik(params):
            omega, alpha, beta, phi, rho = params
            if alpha < 0 or beta < 0 or phi < 0 or rho < 0 or rho > 1:
                return 1e10
            if alpha + beta >= 1:
                return 1e10
            sig2 = cgarch11_filter(self._returns, omega, alpha, beta, phi, rho)
            ll = -0.5 * np.sum(np.log(2 * np.pi * sig2) + self._returns ** 2 / sig2)
            return -ll

        x0 = np.array([var_r * 0.01, 0.05, 0.90, 0.02, 0.99])
        bounds = [(1e-8, None), (1e-6, 0.5), (0.5, 0.999), (1e-6, 0.5), (0.5, 0.9999)]
        res = minimize(neg_loglik, x0, method="L-BFGS-B", bounds=bounds)

        self._params = {
            "omega": res.x[0], "alpha": res.x[1], "beta": res.x[2],
            "phi": res.x[3], "rho": res.x[4],
        }
        self._sigma2 = cgarch11_filter(
            self._returns, **{k: v for k, v in self._params.items()}
        )
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        # Multi-step CGARCH forecast
        omega = self._params["omega"]
        alpha = self._params["alpha"]
        beta = self._params["beta"]
        rho = self._params["rho"]

        last_sigma2 = self._sigma2[-1]
        unc_var = omega / max(1.0 - rho, 1e-8)

        forecasts = np.empty(horizon, dtype=np.float64)
        forecasts[0] = last_sigma2
        for h in range(1, horizon):
            forecasts[h] = unc_var + (alpha + beta) ** h * (last_sigma2 - unc_var)

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.CONDITIONAL_VARIANCE,
                horizon=horizon,
            ),
            model_name="CGARCH(1,1)",
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
        new_r = np.asarray(new_returns, dtype=np.float64)
        self._returns = np.concatenate([self._returns, new_r])
        self._sigma2 = cgarch11_filter(
            self._returns, **{k: v for k, v in self._params.items()}
        )

    def get_params(self) -> dict[str, Any]:
        return self._params.copy()


# ═══════════════════════════════════════════════════
# ARCH(q) Forecaster — Engle (1982)
# ═══════════════════════════════════════════════════

class ARCHForecaster(BaseForecaster):
    """ARCH(q) forecaster (Engle, 1982).

    Pure autoregressive conditional heteroskedasticity model with *q* lags.
    No GARCH persistence term — serves as a foundational baseline.

    Parameters
    ----------
    q : int
        Number of ARCH lags (default 1).
    dist : str
        Error distribution for MLE: "normal", "t", "skewt", "ged".
    """

    def __init__(self, q: int = 1, dist: str = "normal") -> None:
        if q < 1:
            raise ValueError("q must be >= 1")
        self.q = q
        self.dist = dist
        self._params: dict[str, float] = {}
        self._returns: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._sigma2: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._alphas: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name=f"ARCH({self.q})",
            abbreviation="ARCH",
            family="GARCH",
            target=VolatilityTarget.CONDITIONAL_VARIANCE,
            assumptions=("stationary returns", "no persistence term", "finite 4th moment"),
            complexity="O(T) MLE",
            reference="Engle (1982), Econometrica",
            extends=(),
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "ARCHForecaster":
        from arch import arch_model

        self._returns = np.asarray(returns, dtype=np.float64)
        r_scaled = self._returns * 100.0  # arch convention: percentage returns

        am = arch_model(r_scaled, vol="ARCH", p=self.q, dist=self.dist, mean="Zero")
        res = am.fit(disp="off", **kwargs)

        # Extract and rescale parameters
        self._params = {"omega": res.params["omega"] / 1e4}
        alphas = []
        for i in range(1, self.q + 1):
            key = f"alpha[{i}]"
            self._params[key] = res.params[key]
            alphas.append(res.params[key])
        self._alphas = np.array(alphas, dtype=np.float64)
        self._arch_result = res

        # Run Numba filter for internal state
        self._sigma2 = arch_q_filter(
            self._returns, self._params["omega"], self._alphas,
        )
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        omega = self._params["omega"]
        alphas = self._alphas
        q = self.q
        alpha_sum = float(np.sum(alphas))
        unc_var = omega / max(1.0 - alpha_sum, 1e-8)

        # Build extended squared-return / forecast buffer
        # For h-step-ahead, past squared returns are known; future ones
        # are replaced by their conditional expectation (the forecast itself).
        past_r2 = self._returns ** 2
        buf_len = len(past_r2) + horizon
        buf = np.empty(buf_len, dtype=np.float64)
        buf[:len(past_r2)] = past_r2

        forecasts = np.empty(horizon, dtype=np.float64)
        T = len(past_r2)
        for h in range(horizon):
            s = omega
            for i in range(q):
                idx = T + h - 1 - i
                if idx >= 0:
                    s += alphas[i] * buf[idx]
                else:
                    s += alphas[i] * unc_var
            forecasts[h] = s
            buf[T + h] = s  # E[r_{T+h+1}^2 | F_T] = sigma2_{T+h+1}

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.CONDITIONAL_VARIANCE,
                horizon=horizon,
            ),
            model_name=f"ARCH({self.q})",
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
        self._sigma2 = arch_q_filter(
            self._returns, self._params["omega"], self._alphas,
        )

    def get_params(self) -> dict[str, Any]:
        return self._params.copy()


# ═══════════════════════════════════════════════════
# EWMA / RiskMetrics Forecaster
# ═══════════════════════════════════════════════════

class EWMAForecaster(BaseForecaster):
    """Exponentially Weighted Moving Average (EWMA) / RiskMetrics forecaster.

    sigma2_t = lambda * sigma2_{t-1} + (1 - lambda) * r_{t-1}^2

    When ``estimate_lambda=False`` (default), uses the fixed RiskMetrics
    decay factor (0.94 for daily data).  When ``estimate_lambda=True``,
    estimates lambda via profile MLE over the Gaussian log-likelihood.

    Parameters
    ----------
    lambda_ : float
        Decay factor in (0, 1). Default 0.94 (RiskMetrics daily).
    estimate_lambda : bool
        If True, estimate lambda via MLE instead of using the fixed value.
    """

    def __init__(self, lambda_: float = 0.94, estimate_lambda: bool = False) -> None:
        if not (0.0 < lambda_ < 1.0):
            raise ValueError("lambda_ must be in (0, 1)")
        self.lambda_ = lambda_
        self.estimate_lambda = estimate_lambda
        self._params: dict[str, float] = {}
        self._returns: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._sigma2: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name="EWMA (RiskMetrics)",
            abbreviation="EWMA",
            family="GARCH",
            target=VolatilityTarget.CONDITIONAL_VARIANCE,
            assumptions=("unit persistence (IGARCH(1,1) with omega=0)", "no intercept"),
            complexity="O(T)",
            reference="RiskMetrics Technical Document (1996), J.P. Morgan",
            extends=("ARCH",),
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "EWMAForecaster":
        self._returns = np.asarray(returns, dtype=np.float64)

        if self.estimate_lambda:
            from scipy.optimize import minimize_scalar

            def neg_loglik(lam):
                sig2 = ewma_filter(self._returns, lam)
                # Gaussian log-likelihood (constant terms dropped)
                ll = -0.5 * np.sum(np.log(sig2) + self._returns ** 2 / sig2)
                return -ll

            res = minimize_scalar(neg_loglik, bounds=(0.8, 0.9999), method="bounded")
            self.lambda_ = float(res.x)

        self._params = {"lambda": self.lambda_}
        self._sigma2 = ewma_filter(self._returns, self.lambda_)
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # EWMA is IGARCH(1,1) with omega=0 => multi-step forecast is flat
        # sigma2_{T+h|T} = lambda * sigma2_T + (1-lambda) * r_T^2  for h=1
        # For h>1: sigma2_{T+h|T} = sigma2_{T+1|T} (flat, since persistence=1)
        lam = self.lambda_
        last_sigma2 = self._sigma2[-1]
        last_r = self._returns[-1]

        sig2_1 = lam * last_sigma2 + (1.0 - lam) * last_r ** 2
        forecasts = np.full(horizon, sig2_1, dtype=np.float64)

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.CONDITIONAL_VARIANCE,
                horizon=horizon,
            ),
            model_name="EWMA (RiskMetrics)",
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
        lam = self.lambda_
        ret_list = list(self._returns)
        sig2_list = list(self._sigma2)
        for r in new_r:
            new_sig2 = lam * sig2_list[-1] + (1.0 - lam) * ret_list[-1] ** 2
            ret_list.append(r)
            sig2_list.append(new_sig2)
        self._returns = np.array(ret_list, dtype=np.float64)
        self._sigma2 = np.array(sig2_list, dtype=np.float64)

    def get_params(self) -> dict[str, Any]:
        return self._params.copy()
