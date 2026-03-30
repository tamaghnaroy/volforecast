"""
Markov-Switching Volatility model (Hamilton & Susmel, 1994; Gray, 1996).

Two-regime model where the latent state S_t follows a first-order Markov chain:
  r_t | S_t=k ~ N(0, sigma_k^2)

Transition matrix:
  P(S_t=j | S_{t-1}=i) = p_{ij}

Inference via the Hamilton (1989) filter.  Multi-step forecasts use the
Chapman-Kolmogorov equation for regime probabilities.

References:
- Hamilton & Susmel (1994), Journal of Econometrics.
- Gray (1996), Journal of Financial Economics.
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
def _hamilton_filter(
    returns: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    P: NDArray[np.float64],
) -> tuple:
    """Hamilton filter for K-regime Gaussian mixture.

    Parameters
    ----------
    returns : (T,)
    sigma2 : (K,)  regime-conditional variances
    P : (K, K)  transition matrix, P[i,j] = P(S_t=j | S_{t-1}=i)

    Returns
    -------
    filtered_probs : (T, K)  P(S_t=k | r_1,...,r_t)
    log_likelihood : float
    """
    T = returns.shape[0]
    K = sigma2.shape[0]
    filtered = np.empty((T, K), dtype=np.float64)
    total_ll = 0.0

    # Ergodic probabilities as initial state
    # For 2-state: pi_1 = (1-p22)/(2-p11-p22), pi_2 = (1-p11)/(2-p11-p22)
    xi = np.ones(K, dtype=np.float64) / K
    if K == 2:
        denom = 2.0 - P[0, 0] - P[1, 1]
        if abs(denom) > 1e-8:
            xi[0] = (1.0 - P[1, 1]) / denom
            xi[1] = (1.0 - P[0, 0]) / denom

    for t in range(T):
        # Prediction: P(S_t=j | r_1,...,r_{t-1})
        pred = np.zeros(K, dtype=np.float64)
        for j in range(K):
            for i in range(K):
                pred[j] += P[i, j] * xi[i]

        # Likelihood contribution: f(r_t | S_t=k)
        lik = np.empty(K, dtype=np.float64)
        for k in range(K):
            s2 = max(sigma2[k], 1e-20)
            lik[k] = np.exp(-0.5 * (np.log(2.0 * np.pi * s2) + returns[t] ** 2 / s2))

        # Joint: f(r_t, S_t=k | past)
        joint = np.empty(K, dtype=np.float64)
        marginal = 0.0
        for k in range(K):
            joint[k] = lik[k] * pred[k]
            marginal += joint[k]

        marginal = max(marginal, 1e-300)
        total_ll += np.log(marginal)

        # Update: P(S_t=k | r_1,...,r_t)
        for k in range(K):
            xi[k] = joint[k] / marginal
            filtered[t, k] = xi[k]

    return filtered, total_ll


class MSVolForecaster(BaseForecaster):
    """Markov-Switching Volatility forecaster.

    Parameters
    ----------
    n_regimes : int
        Number of latent regimes (default 2).
    """

    def __init__(self, n_regimes: int = 2) -> None:
        if n_regimes < 2:
            raise ValueError("n_regimes must be >= 2")
        self.n_regimes = n_regimes
        self._params: dict[str, float] = {}
        self._returns: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._sigma2_regimes: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._P: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._filtered_probs: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name=f"MS({self.n_regimes})-Vol",
            abbreviation="MSVol",
            family="MS",
            target=VolatilityTarget.CONDITIONAL_VARIANCE,
            assumptions=(
                f"{self.n_regimes}-regime Markov chain",
                "regime-conditional Gaussian variances",
                "Hamilton filter inference",
            ),
            complexity="O(T * K^2)",
            reference="Hamilton & Susmel (1994), JoE; Gray (1996), JFE",
            extends=(),
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "MSVolForecaster":
        self._returns = np.asarray(returns, dtype=np.float64)
        T = len(self._returns)
        K = self.n_regimes

        if T < 20:
            raise ValueError("MS model requires at least 20 observations")

        var_r = np.var(self._returns)

        def _raw_to_P(raw_flat, K):
            """Convert K*K unconstrained params to transition matrix via row-softmax."""
            P = np.empty((K, K), dtype=np.float64)
            for i in range(K):
                row = raw_flat[i * K:(i + 1) * K]
                max_r = np.max(row)
                exp_r = np.exp(row - max_r)
                row_sum = np.sum(exp_r)
                P[i, :] = exp_r / row_sum
            return P

        def neg_loglik(x):
            # Unpack: K log-variances + K*K unconstrained transition params
            log_sigma2 = x[:K]
            sigma2 = np.exp(log_sigma2)
            P = _raw_to_P(x[K:], K)

            _, ll = _hamilton_filter(self._returns, sigma2, P)
            if np.isnan(ll) or np.isinf(ll):
                return 1e10
            return -ll

        # Starting values
        sorted_r2 = np.sort(self._returns ** 2)
        x0_log_sigma2 = []
        for k in range(K):
            start = int(T * k / K)
            end = int(T * (k + 1) / K)
            x0_log_sigma2.append(np.log(max(np.mean(sorted_r2[start:end]), 1e-10)))
        x0_log_sigma2 = np.sort(x0_log_sigma2)

        # K*K unconstrained transition params: high diagonal = persistence
        x0_trans = []
        for i in range(K):
            for j in range(K):
                if j == i:
                    x0_trans.append(3.0)
                else:
                    x0_trans.append(0.0)

        x0 = np.concatenate([np.asarray(x0_log_sigma2), np.asarray(x0_trans)])

        res = minimize(neg_loglik, x0, method="Nelder-Mead",
                       options={"maxiter": 10000, "xatol": 1e-8})

        # Extract results
        log_sigma2 = res.x[:K]
        sigma2 = np.exp(log_sigma2)
        P = _raw_to_P(res.x[K:], K)

        # Sort regimes by variance (low to high)
        order = np.argsort(sigma2)
        sigma2 = sigma2[order]
        P = P[order][:, order]

        self._sigma2_regimes = sigma2
        self._P = P

        self._params = {}
        for k in range(K):
            self._params[f"sigma2_{k}"] = float(sigma2[k])
        for i in range(K):
            for j in range(K):
                self._params[f"p_{i}{j}"] = float(P[i, j])

        self._filtered_probs, _ = _hamilton_filter(self._returns, sigma2, P)
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        K = self.n_regimes
        sigma2 = self._sigma2_regimes
        P = self._P

        # Current filtered probabilities
        xi = self._filtered_probs[-1].copy()

        forecasts = np.empty(horizon, dtype=np.float64)
        for h in range(horizon):
            # Predict regime probabilities h steps ahead
            xi = xi @ P
            # E[sigma2_{t+h}] = sum_k P(S_{t+h}=k) * sigma2_k
            forecasts[h] = np.sum(xi * sigma2)

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.CONDITIONAL_VARIANCE,
                horizon=horizon,
            ),
            model_name=f"MS({K})-Vol",
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
        self._filtered_probs, _ = _hamilton_filter(
            self._returns, self._sigma2_regimes, self._P,
        )

    def get_params(self) -> dict[str, Any]:
        return self._params.copy()


# ═══════════════════════════════════════════════════
# MSGARCH: Markov-Switching GARCH
# ═══════════════════════════════════════════════════

@njit(cache=True)
def _msgarch_filter(
    returns: NDArray[np.float64],
    omega: NDArray[np.float64],
    alpha: NDArray[np.float64],
    beta_g: NDArray[np.float64],
    P: NDArray[np.float64],
) -> tuple:
    """MSGARCH Hamilton filter with Gray (1996) collapsing.

    Each regime k has GARCH(1,1): sigma2_t,k = omega_k + alpha_k*r_{t-1}^2
    + beta_k * sigma2_{t-1,collapsed}, where the collapsed variance is the
    filtered-probability-weighted average across regimes.

    Parameters
    ----------
    returns : (T,)
    omega, alpha, beta_g : (K,) per-regime GARCH parameters
    P : (K, K) transition matrix

    Returns
    -------
    filtered : (T, K) filtered regime probabilities
    sigma2 : (T, K) per-regime conditional variances
    ll : float total log-likelihood
    """
    T = returns.shape[0]
    K = omega.shape[0]
    filtered = np.zeros((T, K), dtype=np.float64)
    sigma2 = np.zeros((T, K), dtype=np.float64)
    total_ll = 0.0

    xi = np.ones(K, dtype=np.float64) / K
    if K == 2:
        denom = 2.0 - P[0, 0] - P[1, 1]
        if abs(denom) > 1e-8:
            xi[0] = (1.0 - P[1, 1]) / denom
            xi[1] = (1.0 - P[0, 0]) / denom

    for k in range(K):
        denom_k = max(1.0 - alpha[k] - beta_g[k], 1e-8)
        sigma2[0, k] = omega[k] / denom_k

    s2_c = 0.0
    for k in range(K):
        s2_c += xi[k] * sigma2[0, k]

    for t in range(T):
        if t > 0:
            r_prev2 = returns[t - 1] ** 2
            for k in range(K):
                sigma2[t, k] = max(omega[k] + alpha[k] * r_prev2 + beta_g[k] * s2_c, 1e-20)

        pred = np.zeros(K, dtype=np.float64)
        for j in range(K):
            for i in range(K):
                pred[j] += P[i, j] * xi[i]

        joint = np.zeros(K, dtype=np.float64)
        marginal = 0.0
        for k in range(K):
            s2k = max(sigma2[t, k], 1e-20)
            lik_k = (1.0 / np.sqrt(2.0 * np.pi * s2k)) * np.exp(
                -0.5 * returns[t] ** 2 / s2k
            )
            joint[k] = lik_k * pred[k]
            marginal += joint[k]

        marginal = max(marginal, 1e-300)
        total_ll += np.log(marginal)

        s2_c = 0.0
        for k in range(K):
            xi[k] = joint[k] / marginal
            filtered[t, k] = xi[k]
            s2_c += xi[k] * sigma2[t, k]

    return filtered, sigma2, total_ll


class MSGARCHForecaster(BaseForecaster):
    """Markov-Switching GARCH model (Haas, Mittnik, Paolella, 2004).

    K-regime model where each regime has its own GARCH(1,1):
      sigma2_{t,k} = omega_k + alpha_k * r_{t-1}^2
                   + beta_k * sigma2_{t-1, collapsed}

    sigma2_collapsed = sum_k P(S_{t-1}=k|F_{t-1}) * sigma2_{t-1,k}
    (Gray 1996 collapsing avoids exponential path proliferation).

    Unlike MSVolForecaster, which has fixed per-regime variance constants,
    this model has full GARCH dynamics inside each regime so volatility
    persistence and shock response differ across regimes.

    Parameters
    ----------
    n_regimes : int
        Number of latent regimes (default 2).

    References
    ----------
    Haas, M., Mittnik, S., Paolella, M. (2004).
        "A New Approach to Markov-Switching GARCH Models."
        Journal of Financial Econometrics 2(4), 493-530.
    Gray, S. (1996). "Modeling the conditional distribution of interest
        rates as a regime-switching process." JFE 42(1), 27-62.
    Augustyniak, M. (2014). "Maximum likelihood estimation of the
        Markov-switching GARCH model." CSDA 76, 61-75.
    """

    def __init__(self, n_regimes: int = 2) -> None:
        if n_regimes < 2:
            raise ValueError("n_regimes must be >= 2")
        self.n_regimes = n_regimes
        self._params: dict[str, float] = {}
        self._returns: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._omega: NDArray[np.float64] = np.zeros(n_regimes, dtype=np.float64)
        self._alpha: NDArray[np.float64] = np.zeros(n_regimes, dtype=np.float64)
        self._beta_g: NDArray[np.float64] = np.zeros(n_regimes, dtype=np.float64)
        self._P: NDArray[np.float64] = np.zeros((n_regimes, n_regimes), dtype=np.float64)
        self._filtered_probs: NDArray[np.float64] = np.zeros((1, n_regimes), dtype=np.float64)
        self._sigma2_last: NDArray[np.float64] = np.zeros(n_regimes, dtype=np.float64)
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name=f"MSGARCH({self.n_regimes})",
            abbreviation="MSGARCH",
            family="MS",
            target=VolatilityTarget.CONDITIONAL_VARIANCE,
            assumptions=(
                f"{self.n_regimes}-regime Markov chain",
                "GARCH(1,1) dynamics per regime",
                "Gray collapsing for tractable inference",
            ),
            complexity="O(T * K^2)",
            reference="Haas, Mittnik, Paolella (2004), J. Financial Econometrics",
            extends=("MSVol",),
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "MSGARCHForecaster":
        self._returns = np.asarray(returns, dtype=np.float64)
        T = len(self._returns)
        K = self.n_regimes

        if T < 50:
            raise ValueError("MSGARCH requires at least 50 observations")

        var_r = float(np.var(self._returns))

        def _raw_to_P(raw_flat: NDArray[np.float64]) -> NDArray[np.float64]:
            P = np.empty((K, K), dtype=np.float64)
            for i in range(K):
                row = raw_flat[i * K:(i + 1) * K]
                exp_r = np.exp(row - np.max(row))
                P[i, :] = exp_r / np.sum(exp_r)
            return P

        def neg_loglik(x: NDArray[np.float64]) -> float:
            n_garch = 3 * K
            raw_g = x[:n_garch]
            raw_P = x[n_garch:]

            omega_k = np.exp(raw_g[0::3]) * var_r
            alpha_k = 1.0 / (1.0 + np.exp(-raw_g[1::3])) * 0.3
            beta_k  = 1.0 / (1.0 + np.exp(-raw_g[2::3])) * 0.9
            P_mat = _raw_to_P(raw_P)

            if np.any(alpha_k + beta_k >= 1.0):
                return 1e10

            _, _, ll = _msgarch_filter(self._returns, omega_k, alpha_k, beta_k, P_mat)
            if np.isnan(ll) or np.isinf(ll):
                return 1e10
            return -ll

        sorted_r2 = np.sort(self._returns ** 2)
        x0_g = []
        for k in range(K):
            start = int(T * k / K)
            end = int(T * (k + 1) / K)
            seg_var = max(float(np.mean(sorted_r2[start:end])), 1e-10)
            omega_init = seg_var * 0.05
            x0_g += [np.log(max(omega_init / var_r, 1e-10)), 0.0, 1.0]

        x0_P = []
        for i in range(K):
            for j in range(K):
                x0_P.append(3.0 if j == i else 0.0)

        x0 = np.array(x0_g + x0_P, dtype=np.float64)
        res = minimize(
            neg_loglik, x0, method="Nelder-Mead",
            options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-7},
        )

        n_garch = 3 * K
        raw_g = res.x[:n_garch]
        omega_f = np.exp(raw_g[0::3]) * var_r
        alpha_f = 1.0 / (1.0 + np.exp(-raw_g[1::3])) * 0.3
        beta_f  = 1.0 / (1.0 + np.exp(-raw_g[2::3])) * 0.9
        P_f = _raw_to_P(res.x[n_garch:])

        order = np.argsort(omega_f / np.maximum(1.0 - alpha_f - beta_f, 1e-8))
        self._omega  = omega_f[order]
        self._alpha  = alpha_f[order]
        self._beta_g = beta_f[order]
        self._P      = P_f[order][:, order]

        filtered, sigma2, _ = _msgarch_filter(
            self._returns, self._omega, self._alpha, self._beta_g, self._P
        )
        self._filtered_probs = filtered
        self._sigma2_last    = sigma2[-1].copy()

        self._params = {}
        for k in range(K):
            self._params[f"omega_{k}"]  = float(self._omega[k])
            self._params[f"alpha_{k}"]  = float(self._alpha[k])
            self._params[f"beta_{k}"]   = float(self._beta_g[k])
        for i in range(K):
            for j in range(K):
                self._params[f"p_{i}{j}"] = float(self._P[i, j])

        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        K = self.n_regimes
        xi = self._filtered_probs[-1].copy()
        s2_k = self._sigma2_last.copy()

        r_last2 = float(self._returns[-1] ** 2)
        forecasts = np.empty(horizon, dtype=np.float64)

        for h in range(horizon):
            xi_pred = xi @ self._P

            new_s2 = np.empty(K, dtype=np.float64)
            s2_c = float(np.dot(xi, s2_k))
            r2_input = r_last2 if h == 0 else s2_c
            for k in range(K):
                new_s2[k] = self._omega[k] + self._alpha[k] * r2_input + self._beta_g[k] * s2_c

            forecasts[h] = float(np.dot(xi_pred, new_s2))
            xi = xi_pred
            s2_k = new_s2
            r_last2 = forecasts[h]

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.CONDITIONAL_VARIANCE,
                horizon=horizon,
            ),
            model_name=f"MSGARCH({K})",
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
        filtered, sigma2, _ = _msgarch_filter(
            self._returns, self._omega, self._alpha, self._beta_g, self._P
        )
        self._filtered_probs = filtered
        self._sigma2_last    = sigma2[-1].copy()

    def get_params(self) -> dict[str, Any]:
        return self._params.copy()
