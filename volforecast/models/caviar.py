"""
CAViaR — Conditional Autoregressive Value-at-Risk (Engle & Manganelli, 2004).

Quantile-native volatility / risk model that directly models the conditional
quantile dynamics without specifying the full distribution.

Symmetric Absolute Value (SAV) specification:
  q_t = omega + alpha * |r_{t-1}| + beta * q_{t-1}

Asymmetric Slope (AS) specification:
  q_t = omega + alpha_pos * max(r_{t-1}, 0) + alpha_neg * min(r_{t-1}, 0) + beta * q_{t-1}

Indirect GARCH specification:
  q_t = sqrt(omega + alpha * r_{t-1}^2 + beta * q_{t-1}^2)

Estimation via quantile regression loss (check function / pinball loss).

Reference: Engle & Manganelli (2004), Journal of Business & Economic Statistics.
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
def _caviar_sav_filter(
    returns: NDArray[np.float64],
    omega: float,
    alpha: float,
    beta: float,
    q0: float,
) -> NDArray[np.float64]:
    """SAV CAViaR filter: q_t = omega + alpha * |r_{t-1}| + beta * q_{t-1}."""
    T = returns.shape[0]
    q = np.empty(T, dtype=np.float64)
    q[0] = q0
    for t in range(1, T):
        q[t] = omega + alpha * abs(returns[t - 1]) + beta * q[t - 1]
    return q


@njit(cache=True)
def _caviar_as_filter(
    returns: NDArray[np.float64],
    omega: float,
    alpha_pos: float,
    alpha_neg: float,
    beta: float,
    q0: float,
) -> NDArray[np.float64]:
    """Asymmetric Slope CAViaR filter."""
    T = returns.shape[0]
    q = np.empty(T, dtype=np.float64)
    q[0] = q0
    for t in range(1, T):
        r = returns[t - 1]
        pos_part = r if r > 0 else 0.0
        neg_part = r if r < 0 else 0.0
        q[t] = omega + alpha_pos * pos_part - alpha_neg * neg_part + beta * q[t - 1]
    return q


@njit(cache=True)
def _caviar_igarch_filter(
    returns: NDArray[np.float64],
    omega: float,
    alpha: float,
    beta: float,
    q0: float,
) -> NDArray[np.float64]:
    """Indirect GARCH CAViaR filter: q_t = sqrt(omega + alpha*r_{t-1}^2 + beta*q_{t-1}^2)."""
    T = returns.shape[0]
    q = np.empty(T, dtype=np.float64)
    q[0] = q0
    for t in range(1, T):
        inside = omega + alpha * returns[t - 1] ** 2 + beta * q[t - 1] ** 2
        q[t] = np.sqrt(max(inside, 1e-20))
    return q


@njit(cache=True)
def _quantile_loss(
    returns: NDArray[np.float64],
    q: NDArray[np.float64],
    tau: float,
) -> float:
    """Quantile regression loss (check/pinball function)."""
    T = returns.shape[0]
    loss = 0.0
    for t in range(T):
        resid = returns[t] - q[t]
        if resid >= 0:
            loss += tau * resid
        else:
            loss += (tau - 1.0) * resid
    return loss


class CAViaRForecaster(BaseForecaster):
    """CAViaR conditional quantile forecaster (Engle & Manganelli, 2004).

    Parameters
    ----------
    tau : float
        Quantile level (default 0.05 for 5% VaR).
    spec : str
        CAViaR specification: "SAV" (symmetric), "AS" (asymmetric), "IGARCH".
    """

    def __init__(self, tau: float = 0.05, spec: str = "SAV") -> None:
        if not (0.0 < tau < 1.0):
            raise ValueError("tau must be in (0, 1)")
        if spec not in ("SAV", "AS", "IGARCH"):
            raise ValueError("spec must be 'SAV', 'AS', or 'IGARCH'")
        self.tau = tau
        self.spec = spec
        self._params: dict[str, float] = {}
        self._returns: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._q: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name=f"CAViaR-{self.spec}(tau={self.tau})",
            abbreviation="CAViaR",
            family="Quantile",
            target=VolatilityTarget.CONDITIONAL_VARIANCE,
            assumptions=(
                f"conditional quantile dynamics ({self.spec})",
                f"quantile level tau={self.tau}",
                "no distributional assumption",
            ),
            complexity="O(T) quantile regression",
            reference="Engle & Manganelli (2004), JBES",
            extends=(),
        )

    def _run_filter(self, returns, params, q0):
        if self.spec == "SAV":
            return _caviar_sav_filter(
                returns, params["omega"], params["alpha"], params["beta"], q0,
            )
        elif self.spec == "AS":
            return _caviar_as_filter(
                returns, params["omega"], params["alpha_pos"],
                params["alpha_neg"], params["beta"], q0,
            )
        else:  # IGARCH
            return _caviar_igarch_filter(
                returns, params["omega"], params["alpha"], params["beta"], q0,
            )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "CAViaRForecaster":
        self._returns = np.asarray(returns, dtype=np.float64)
        T = len(self._returns)

        # Initial quantile from empirical quantile
        q0 = float(np.quantile(self._returns, self.tau))
        self._q0 = q0

        if self.spec == "SAV":
            def objective(x):
                omega, alpha, beta = x
                if beta < 0 or beta > 0.9999 or alpha < 0:
                    return 1e10
                q = _caviar_sav_filter(self._returns, omega, alpha, beta, q0)
                return _quantile_loss(self._returns, q, self.tau)

            x0 = np.array([q0 * 0.1, 0.1, 0.85])
            bounds = [(-0.1, 0.0), (0.001, 1.0), (0.5, 0.9999)]
            res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
            self._params = {"omega": res.x[0], "alpha": res.x[1], "beta": res.x[2]}

        elif self.spec == "AS":
            def objective(x):
                omega, alpha_pos, alpha_neg, beta = x
                if beta < 0 or beta > 0.9999:
                    return 1e10
                q = _caviar_as_filter(self._returns, omega, alpha_pos, alpha_neg, beta, q0)
                return _quantile_loss(self._returns, q, self.tau)

            x0 = np.array([q0 * 0.1, 0.05, 0.15, 0.85])
            bounds = [(-0.1, 0.0), (0.001, 1.0), (0.001, 1.0), (0.5, 0.9999)]
            res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
            self._params = {
                "omega": res.x[0], "alpha_pos": res.x[1],
                "alpha_neg": res.x[2], "beta": res.x[3],
            }

        else:  # IGARCH
            def objective(x):
                omega, alpha, beta = x
                if beta < 0 or beta > 0.9999 or alpha < 0 or omega < 0:
                    return 1e10
                q = _caviar_igarch_filter(self._returns, omega, alpha, beta, abs(q0))
                return _quantile_loss(self._returns, q, self.tau)

            x0 = np.array([q0 ** 2 * 0.05, 0.1, 0.85])
            bounds = [(1e-10, None), (0.001, 0.5), (0.5, 0.9999)]
            res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
            self._params = {"omega": res.x[0], "alpha": res.x[1], "beta": res.x[2]}

        self._q = self._run_filter(self._returns, self._params, q0)
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        last_q = self._q[-1]
        last_r = self._returns[-1]

        forecasts = np.empty(horizon, dtype=np.float64)

        if self.spec == "SAV":
            omega = self._params["omega"]
            alpha = self._params["alpha"]
            beta = self._params["beta"]
            q_h = omega + alpha * abs(last_r) + beta * last_q
            forecasts[0] = q_h
            for h in range(1, horizon):
                # E[|r|] ≈ sqrt(2/pi) * |q| for Gaussian
                e_abs_r = np.sqrt(2.0 / np.pi) * abs(q_h)
                q_h = omega + alpha * e_abs_r + beta * q_h
                forecasts[h] = q_h

        elif self.spec == "AS":
            omega = self._params["omega"]
            alpha_pos = self._params["alpha_pos"]
            alpha_neg = self._params["alpha_neg"]
            beta = self._params["beta"]
            pos_part = last_r if last_r > 0 else 0.0
            neg_part = last_r if last_r < 0 else 0.0
            q_h = omega + alpha_pos * pos_part - alpha_neg * neg_part + beta * last_q
            forecasts[0] = q_h
            for h in range(1, horizon):
                # Symmetric expectation: E[r+] = E[-r-] ≈ 0.5*E[|r|]
                e_abs_r = np.sqrt(2.0 / np.pi) * abs(q_h) * 0.5
                q_h = omega + (alpha_pos + alpha_neg) * e_abs_r + beta * q_h
                forecasts[h] = q_h

        else:  # IGARCH
            omega = self._params["omega"]
            alpha = self._params["alpha"]
            beta = self._params["beta"]
            q_h = np.sqrt(max(omega + alpha * last_r ** 2 + beta * last_q ** 2, 1e-20))
            forecasts[0] = q_h
            for h in range(1, horizon):
                # E[r^2] ≈ q_h^2
                q_h = np.sqrt(max(omega + alpha * q_h ** 2 + beta * q_h ** 2, 1e-20))
                forecasts[h] = q_h

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.CONDITIONAL_VARIANCE,
                horizon=horizon,
                notes=f"Conditional quantile at tau={self.tau}, not variance",
            ),
            model_name=f"CAViaR-{self.spec}",
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
        self._q = self._run_filter(self._returns, self._params, self._q0)

    def get_params(self) -> dict[str, Any]:
        return self._params.copy()

    def hit_rate(self) -> float:
        """Compute empirical hit rate: fraction of r_t < q_t."""
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        hits = np.sum(self._returns < self._q)
        return float(hits / len(self._returns))
