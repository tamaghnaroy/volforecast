"""
Stochastic Volatility models (Taylor, 1986; Harvey, Ruiz, Shephard, 1994).

Latent log-volatility AR(1) state-space model:
  r_t = exp(h_t / 2) * eps_t,   eps_t ~ N(0, 1)
  h_t = mu + phi * (h_{t-1} - mu) + sigma_eta * eta_t,  eta_t ~ N(0, 1)

SV-J adds a jump component:
  r_t = exp(h_t / 2) * eps_t + J_t * q_t
  where q_t ~ Bernoulli(lambda), J_t ~ N(mu_j, sigma_j^2)

Inference via quasi-maximum likelihood (QML) using the Kim, Shephard,
Chib (1998) log-chi-squared mixture approximation.

References:
- Taylor (1986), "Modelling Financial Time Series."
- Harvey, Ruiz, Shephard (1994), Review of Economic Studies.
- Kim, Shephard, Chib (1998), Review of Economic Studies.
- Bates (1996) for jump extension.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from volforecast.core.base import BaseForecaster, ForecastResult, ModelSpec
from volforecast.core.targets import VolatilityTarget, TargetSpec


# KSC (1998) 7-component mixture approximation for log(chi^2_1)
# Each row: (probability, mean, variance)
_KSC_PARAMS = np.array([
    [0.00730, -10.12999, 5.79596],
    [0.10556,  -3.97281, 2.61369],
    [0.00002, -11.40040, 5.17950],
    [0.04395,  -5.56241, 2.81930],
    [0.34001,  -0.65098, 0.16735],
    [0.24566,  -2.35859, 1.10960],
    [0.25750,   0.52478, 0.64009],
], dtype=np.float64)


def _sv_qml_loglik(
    params: NDArray[np.float64],
    log_r2: NDArray[np.float64],
) -> float:
    """Quasi-ML log-likelihood for basic SV via KSC mixture approximation.

    The observation equation is:
      log(r_t^2) = h_t + log(eps_t^2)
    where log(eps_t^2) ~ log(chi^2_1), approximated by a 7-component Gaussian
    mixture, enabling Kalman filter inference.
    """
    mu, phi, log_sig_eta = params
    sigma_eta = np.exp(log_sig_eta)

    if abs(phi) >= 0.9999 or sigma_eta < 1e-8:
        return 1e10

    T = len(log_r2)
    probs = _KSC_PARAMS[:, 0]
    means = _KSC_PARAMS[:, 1]
    variances = _KSC_PARAMS[:, 2]
    n_mix = len(probs)

    # Use a simplified approach: weighted Kalman filter over mixture components
    # For each observation, compute the marginal likelihood as a mixture
    sigma_eta2 = sigma_eta ** 2

    # State: h_t with prior h_0 ~ N(mu, sigma_eta^2 / (1 - phi^2))
    state_mean = mu
    state_var = sigma_eta2 / max(1.0 - phi ** 2, 1e-8)

    total_ll = 0.0

    for t in range(T):
        # Prediction step
        pred_mean = mu + phi * (state_mean - mu)
        pred_var = phi ** 2 * state_var + sigma_eta2

        # Observation likelihood as mixture
        obs = log_r2[t]
        mix_ll = 0.0
        post_mean_num = 0.0
        post_var_inv_sum = 0.0
        weights = np.empty(n_mix)

        for j in range(n_mix):
            obs_mean = pred_mean + means[j]
            obs_var = pred_var + variances[j]
            residual = obs - obs_mean
            log_comp = -0.5 * (np.log(2 * np.pi * obs_var) + residual ** 2 / obs_var)
            weights[j] = np.log(probs[j]) + log_comp

        # Log-sum-exp for numerical stability
        max_w = np.max(weights)
        log_marginal = max_w + np.log(np.sum(np.exp(weights - max_w)))
        total_ll += log_marginal

        # Collapsed update (approximate posterior as single Gaussian)
        norm_weights = np.exp(weights - max_w)
        norm_weights /= np.sum(norm_weights)

        new_mean = 0.0
        new_var = 0.0
        for j in range(n_mix):
            obs_var_j = pred_var + variances[j]
            K_j = pred_var / obs_var_j
            post_mean_j = pred_mean + K_j * (obs - pred_mean - means[j])
            post_var_j = pred_var * (1.0 - K_j)
            new_mean += norm_weights[j] * post_mean_j
            new_var += norm_weights[j] * (post_var_j + post_mean_j ** 2)
        new_var -= new_mean ** 2
        new_var = max(new_var, 1e-10)

        state_mean = new_mean
        state_var = new_var

    if np.isnan(total_ll) or np.isinf(total_ll):
        return 1e10
    return -total_ll


class SVForecaster(BaseForecaster):
    """Stochastic Volatility forecaster (Taylor, 1986).

    Latent log-volatility AR(1) model estimated via quasi-maximum likelihood
    using the KSC (1998) mixture approximation.
    """

    def __init__(self) -> None:
        self._params: dict[str, float] = {}
        self._returns: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._h: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._state_mean: float = 0.0
        self._state_var: float = 1.0
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name="Stochastic Volatility",
            abbreviation="SV",
            family="SV",
            target=VolatilityTarget.CONDITIONAL_VARIANCE,
            assumptions=(
                "latent log-vol AR(1)",
                "Gaussian innovations",
                "QML estimation via KSC mixture",
            ),
            complexity="O(T) QML",
            reference="Taylor (1986); Harvey, Ruiz, Shephard (1994)",
            extends=(),
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "SVForecaster":
        self._returns = np.asarray(returns, dtype=np.float64)
        T = len(self._returns)
        if T < 10:
            raise ValueError("SV model requires at least 10 observations")

        # Transform: log(r_t^2), with offset for zero returns
        r2 = self._returns ** 2
        r2 = np.maximum(r2, 1e-20)
        log_r2 = np.log(r2)

        # Starting values
        sample_var = np.var(log_r2)
        x0 = np.array([np.mean(log_r2) + 1.27, 0.95, np.log(0.2)])

        res = minimize(
            _sv_qml_loglik, x0, args=(log_r2,),
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-6},
        )

        mu, phi, log_sig_eta = res.x
        sigma_eta = np.exp(log_sig_eta)
        phi = np.clip(phi, -0.999, 0.999)
        sigma_eta = max(sigma_eta, 1e-6)

        self._params = {
            "mu": mu, "phi": phi, "sigma_eta": sigma_eta,
        }

        # Run a Kalman smoother pass to extract filtered h_t
        sigma_eta2 = sigma_eta ** 2
        self._h = np.empty(T, dtype=np.float64)
        state_mean = mu
        state_var = sigma_eta2 / max(1.0 - phi ** 2, 1e-8)

        probs = _KSC_PARAMS[:, 0]
        means = _KSC_PARAMS[:, 1]
        variances = _KSC_PARAMS[:, 2]

        for t in range(T):
            pred_mean = mu + phi * (state_mean - mu)
            pred_var = phi ** 2 * state_var + sigma_eta2

            obs = log_r2[t]
            weights = np.empty(len(probs))
            for j in range(len(probs)):
                obs_var = pred_var + variances[j]
                residual = obs - pred_mean - means[j]
                weights[j] = np.log(probs[j]) - 0.5 * (
                    np.log(2 * np.pi * obs_var) + residual ** 2 / obs_var
                )
            max_w = np.max(weights)
            norm_weights = np.exp(weights - max_w)
            norm_weights /= np.sum(norm_weights)

            new_mean = 0.0
            new_var = 0.0
            for j in range(len(probs)):
                obs_var_j = pred_var + variances[j]
                K_j = pred_var / obs_var_j
                pm_j = pred_mean + K_j * (obs - pred_mean - means[j])
                pv_j = pred_var * (1.0 - K_j)
                new_mean += norm_weights[j] * pm_j
                new_var += norm_weights[j] * (pv_j + pm_j ** 2)
            new_var -= new_mean ** 2
            new_var = max(new_var, 1e-10)

            self._h[t] = new_mean
            state_mean = new_mean
            state_var = new_var

        self._state_mean = state_mean
        self._state_var = state_var
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        mu = self._params["mu"]
        phi = self._params["phi"]
        sigma_eta = self._params["sigma_eta"]

        forecasts = np.empty(horizon, dtype=np.float64)
        h_pred = self._state_mean
        for h_idx in range(horizon):
            h_pred = mu + phi * (h_pred - mu)
            # E[exp(h_t)] when h_t ~ N(h_pred, v_pred): exp(h_pred + v_pred/2)
            # For simplicity, use point forecast
            forecasts[h_idx] = np.exp(h_pred)

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.CONDITIONAL_VARIANCE,
                horizon=horizon,
            ),
            model_name="Stochastic Volatility",
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
        mu = self._params["mu"]
        phi = self._params["phi"]
        sigma_eta = self._params["sigma_eta"]
        sigma_eta2 = sigma_eta ** 2

        probs = _KSC_PARAMS[:, 0]
        means = _KSC_PARAMS[:, 1]
        variances = _KSC_PARAMS[:, 2]

        h_list = list(self._h)
        state_mean = self._state_mean
        state_var = self._state_var

        for r in new_r:
            r2 = max(r ** 2, 1e-20)
            obs = np.log(r2)

            pred_mean = mu + phi * (state_mean - mu)
            pred_var = phi ** 2 * state_var + sigma_eta2

            weights = np.empty(len(probs))
            for j in range(len(probs)):
                obs_var = pred_var + variances[j]
                residual = obs - pred_mean - means[j]
                weights[j] = np.log(probs[j]) - 0.5 * (
                    np.log(2 * np.pi * obs_var) + residual ** 2 / obs_var
                )
            max_w = np.max(weights)
            norm_weights = np.exp(weights - max_w)
            norm_weights /= np.sum(norm_weights)

            new_mean = 0.0
            new_var = 0.0
            for j in range(len(probs)):
                obs_var_j = pred_var + variances[j]
                K_j = pred_var / obs_var_j
                pm_j = pred_mean + K_j * (obs - pred_mean - means[j])
                pv_j = pred_var * (1.0 - K_j)
                new_mean += norm_weights[j] * pm_j
                new_var += norm_weights[j] * (pv_j + pm_j ** 2)
            new_var -= new_mean ** 2
            new_var = max(new_var, 1e-10)

            h_list.append(new_mean)
            state_mean = new_mean
            state_var = new_var

        self._returns = np.concatenate([self._returns, new_r])
        self._h = np.array(h_list, dtype=np.float64)
        self._state_mean = state_mean
        self._state_var = state_var

    def get_params(self) -> dict[str, Any]:
        return self._params.copy()


class SVJForecaster(BaseForecaster):
    """Stochastic Volatility with Jumps (Bates, 1996).

    Extends the basic SV model with a compound Poisson jump component.
    Uses a two-stage approach: first estimates the SV parameters via QML,
    then estimates jump parameters from the residuals.
    """

    def __init__(self) -> None:
        self._params: dict[str, float] = {}
        self._sv: SVForecaster = SVForecaster()
        self._returns: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name="SV with Jumps",
            abbreviation="SVJ",
            family="SV",
            target=VolatilityTarget.CONDITIONAL_VARIANCE,
            assumptions=(
                "latent log-vol AR(1)",
                "compound Poisson jumps",
                "two-stage QML + jump detection",
            ),
            complexity="O(T) QML + jump estimation",
            reference="Bates (1996), Review of Financial Studies",
            extends=("SV",),
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "SVJForecaster":
        self._returns = np.asarray(returns, dtype=np.float64)

        # Stage 1: Fit base SV model
        self._sv.fit(self._returns, realized_measures, **kwargs)
        sv_params = self._sv.get_params()

        # Stage 2: Detect jumps from standardized residuals
        vol = np.exp(self._sv._h / 2.0)
        vol = np.maximum(vol, 1e-10)
        z = self._returns / vol

        # Jumps are detected as |z| > threshold (BNS-style)
        threshold = 3.0
        jump_indicator = np.abs(z) > threshold
        n_jumps = np.sum(jump_indicator)
        T = len(self._returns)
        lambda_j = max(n_jumps / T, 1e-4)

        if n_jumps > 0:
            jump_returns = self._returns[jump_indicator]
            mu_j = float(np.mean(jump_returns))
            sigma_j = float(np.std(jump_returns)) if n_jumps > 1 else float(np.std(self._returns))
        else:
            mu_j = 0.0
            sigma_j = float(np.std(self._returns))

        self._params = {
            **sv_params,
            "lambda_j": lambda_j,
            "mu_j": mu_j,
            "sigma_j": sigma_j,
        }
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # SV component forecast
        sv_forecast = self._sv.predict(horizon, **kwargs)
        sv_var = sv_forecast.point

        # Add jump contribution: E[J_t^2 * q_t] = lambda * (mu_j^2 + sigma_j^2)
        lambda_j = self._params["lambda_j"]
        mu_j = self._params["mu_j"]
        sigma_j = self._params["sigma_j"]
        jump_var = lambda_j * (mu_j ** 2 + sigma_j ** 2)

        forecasts = sv_var + jump_var

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.CONDITIONAL_VARIANCE,
                horizon=horizon,
            ),
            model_name="SV with Jumps",
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
        self._sv.update(new_r, new_realized, **kwargs)

    def get_params(self) -> dict[str, Any]:
        return self._params.copy()
