"""
Copula-GARCH multivariate volatility model.

Separates univariate GARCH marginal volatility from the joint dependence
structure via a copula.  This allows asymmetric and heavy-tailed joint
distributions that DCC-GARCH (with Gaussian/Student-t assumptions) cannot
capture.

Protocol
--------
1. Fit GARCH(1,1) marginals → get standardized residuals.
2. Apply PIT (Probability Integral Transform) to uniform [0,1] margins.
3. Fit chosen copula (Gaussian or Student-t) to the uniform margins.
4. Optionally fit GPD tails above/below a threshold (EVT step).
5. Simulate joint scenarios for portfolio VaR / ES.

References
----------
Joe, H. (2014). Dependence Modeling with Copulas. CRC Press.
McNeil, A., Frey, R., Embrechts, P. (2005). Quantitative Risk Management.
    Princeton UP.
Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs marges.
    Publications de l'Institut Statistique de l'Université de Paris 8, 229-231.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import minimize
from numba import njit

from volforecast.core.base import BaseForecaster, ForecastResult, ModelSpec
from volforecast.core.targets import VolatilityTarget, TargetSpec


# ═══════════════════════════════════════════════════
# GARCH(1,1) helpers (univariate, for marginals)
# ═══════════════════════════════════════════════════

@njit(cache=True)
def _garch11_cg(
    r: NDArray[np.float64],
    omega: float,
    alpha: float,
    beta: float,
) -> NDArray[np.float64]:
    T = r.shape[0]
    h = np.empty(T, dtype=np.float64)
    h[0] = omega / max(1.0 - alpha - beta, 1e-8)
    for t in range(1, T):
        h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
    return h


# ═══════════════════════════════════════════════════
# EVT / GPD tail fitting
# ═══════════════════════════════════════════════════

def _fit_gpd_tail(u: NDArray[np.float64], threshold: float) -> tuple[float, float]:
    """Fit Generalized Pareto Distribution above threshold via MLE.

    Returns (xi, sigma) for GPD(xi, sigma).
    """
    exceedances = u[u > threshold] - threshold
    if len(exceedances) < 5:
        return 0.0, float(np.std(u))

    def neg_ll(params: NDArray[np.float64]) -> float:
        xi, log_sigma = params
        sigma = np.exp(log_sigma)
        if xi == 0.0:
            return float(len(exceedances) * log_sigma + np.sum(exceedances) / sigma)
        arg = 1.0 + xi * exceedances / sigma
        if np.any(arg <= 0.0):
            return 1e10
        return float(len(exceedances) * log_sigma + (1.0 + 1.0 / xi) * np.sum(np.log(arg)))

    res = minimize(neg_ll, [0.1, np.log(np.mean(exceedances) + 1e-10)],
                   method="Nelder-Mead", options={"maxiter": 2000})
    xi_hat = float(res.x[0])
    sigma_hat = float(np.exp(res.x[1]))
    return xi_hat, sigma_hat


def _gpd_pit(u: NDArray[np.float64], threshold: float, xi: float, sigma: float) -> NDArray[np.float64]:
    """PIT for the GPD tail above threshold."""
    p_thresh = float(np.mean(u <= threshold))
    out = np.where(
        u <= threshold,
        stats.norm.cdf(u),
        p_thresh + (1.0 - p_thresh) * (
            1.0 - (1.0 + xi * np.maximum(u - threshold, 0.0) / sigma) ** (-1.0 / xi)
            if xi != 0.0
            else 1.0 - np.exp(-np.maximum(u - threshold, 0.0) / sigma)
        ),
    )
    return np.clip(out, 1e-6, 1.0 - 1e-6)


class CopulaGARCHForecaster(BaseForecaster):
    """Copula-GARCH multivariate volatility model.

    Fits GARCH(1,1) marginals per asset and a Gaussian or Student-t copula
    on the PIT-transformed residuals.  Joint simulation produces per-asset
    conditional variance forecasts and portfolio-level VaR / ES estimates.

    Parameters
    ----------
    copula : str
        "gaussian" or "student_t". Default "student_t".
    use_evt : bool
        If True, fit GPD tails above/below the 95th/5th percentile of
        standardized residuals for better tail modelling.
    evt_threshold_pct : float
        Quantile threshold for EVT (default 0.95 for upper tail).
    n_sims : int
        Monte Carlo paths for portfolio VaR/ES. Default 10000.

    Notes
    -----
    Accepts a 2-D returns array (T, N). Pass via the ``returns`` argument.
    """

    def __init__(
        self,
        copula: str = "student_t",
        use_evt: bool = False,
        evt_threshold_pct: float = 0.95,
        n_sims: int = 10000,
    ) -> None:
        if copula not in ("gaussian", "student_t"):
            raise ValueError("copula must be 'gaussian' or 'student_t'")
        self.copula = copula
        self.use_evt = use_evt
        self.evt_threshold_pct = evt_threshold_pct
        self.n_sims = n_sims

        self._omega: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._alpha_g: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._beta_g: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._copula_corr: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._copula_df: float = 10.0
        self._h_last: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._r_last: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._gpd_params: list[tuple[float, float, float]] = []
        self._N: int = 0
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name=f"Copula-GARCH({self.copula})",
            abbreviation="CopGARCH",
            family="Multivariate",
            target=VolatilityTarget.CONDITIONAL_VARIANCE,
            assumptions=(
                "GARCH(1,1) marginals",
                f"{self.copula} copula for dependence",
                "EVT tails" if self.use_evt else "parametric tails",
            ),
            complexity="O(T * N^2)",
            reference="McNeil, Frey, Embrechts (2005), Quantitative Risk Management",
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "CopulaGARCHForecaster":
        R = np.atleast_2d(np.asarray(returns, dtype=np.float64))
        if R.shape[0] < R.shape[1]:
            R = R.T
        T, N = R.shape
        self._N = N

        if T < 50:
            raise ValueError("CopulaGARCHForecaster requires at least 50 observations")

        var_r = np.var(R, axis=0)
        omega = np.empty(N, dtype=np.float64)
        alpha_g = np.empty(N, dtype=np.float64)
        beta_g = np.empty(N, dtype=np.float64)
        Z_pit = np.empty((T, N), dtype=np.float64)

        for n in range(N):
            r_n = R[:, n]

            def neg_ll(x: NDArray[np.float64], r: NDArray[np.float64] = r_n) -> float:
                om = np.exp(x[0]) * float(var_r[n])
                al = 1.0 / (1.0 + np.exp(-x[1])) * 0.5
                be = 1.0 / (1.0 + np.exp(-x[2])) * 0.97
                if al + be >= 1.0:
                    return 1e10
                h_t = _garch11_cg(r, om, al, be)
                ll = 0.0
                for t in range(len(r)):
                    ht = max(h_t[t], 1e-20)
                    ll += -0.5 * (np.log(2.0 * np.pi) + np.log(ht) + r[t] ** 2 / ht)
                return -ll

            res = minimize(
                neg_ll, [0.0, 0.0, 2.0], method="Nelder-Mead",
                options={"maxiter": 5000, "xatol": 1e-7},
            )
            om = np.exp(res.x[0]) * float(var_r[n])
            al = 1.0 / (1.0 + np.exp(-res.x[1])) * 0.5
            be = 1.0 / (1.0 + np.exp(-res.x[2])) * 0.97
            omega[n], alpha_g[n], beta_g[n] = om, al, be

            h_n = _garch11_cg(R[:, n], om, al, be)
            z_n = R[:, n] / np.sqrt(np.maximum(h_n, 1e-20))

            if self.use_evt:
                thr = float(np.quantile(np.abs(z_n), self.evt_threshold_pct))
                xi_hat, sigma_hat = _fit_gpd_tail(np.abs(z_n), thr)
                self._gpd_params.append((thr, xi_hat, sigma_hat))
                pit_n = _gpd_pit(z_n, thr, xi_hat, sigma_hat)
            else:
                pit_n = np.clip(stats.norm.cdf(z_n), 1e-6, 1.0 - 1e-6)

            Z_pit[:, n] = stats.norm.ppf(pit_n)

        self._omega, self._alpha_g, self._beta_g = omega, alpha_g, beta_g

        if self.copula == "gaussian":
            self._copula_corr = np.corrcoef(Z_pit.T)
            if N == 1:
                self._copula_corr = np.array([[1.0]])
        else:
            corr_init = np.corrcoef(Z_pit.T)
            if N == 1:
                corr_init = np.array([[1.0]])

            def neg_ll_t(x: NDArray[np.float64]) -> float:
                df = np.exp(x[0]) + 2.0
                ll = float(np.sum(
                    stats.multivariate_t.logpdf(Z_pit, loc=np.zeros(N), shape=corr_init, df=df)
                ))
                return -ll if not np.isnan(ll) else 1e10

            res_t = minimize(neg_ll_t, [np.log(8.0)], method="Nelder-Mead",
                             options={"maxiter": 1000})
            self._copula_df = float(np.exp(res_t.x[0]) + 2.0)
            self._copula_corr = corr_init

        self._h_last = np.array([
            _garch11_cg(R[:, n], omega[n], alpha_g[n], beta_g[n])[-1]
            for n in range(N)
        ], dtype=np.float64)
        self._r_last = R[-1].copy()
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        N = self._N
        h = self._h_last.copy()
        r2 = self._r_last ** 2
        sigma_forecasts = np.empty((horizon, N), dtype=np.float64)

        for step in range(horizon):
            h_next = np.empty(N, dtype=np.float64)
            for n in range(N):
                h_next[n] = max(
                    self._omega[n] + self._alpha_g[n] * r2[n] + self._beta_g[n] * h[n],
                    1e-20,
                )
            sigma_forecasts[step] = np.sqrt(h_next)
            r2 = h_next.copy()
            h = h_next.copy()

        rng = np.random.default_rng()
        if self.copula == "gaussian":
            U = stats.multivariate_normal.rvs(
                mean=np.zeros(N), cov=self._copula_corr, size=self.n_sims
            )
        else:
            U = stats.multivariate_t.rvs(
                loc=np.zeros(N), shape=self._copula_corr,
                df=self._copula_df, size=self.n_sims,
            )

        Z_sim = stats.norm.ppf(np.clip(stats.norm.cdf(U), 1e-6, 1.0 - 1e-6))
        portfolio_returns = Z_sim @ (sigma_forecasts[0] / np.sqrt(N))

        var_95 = float(np.percentile(portfolio_returns, 5))
        es_95 = float(np.mean(portfolio_returns[portfolio_returns <= var_95]))

        return ForecastResult(
            point=sigma_forecasts[0],
            target_spec=TargetSpec(
                target=VolatilityTarget.CONDITIONAL_VARIANCE,
                horizon=horizon,
                proxy_description="Per-asset realized variance",
            ),
            model_name=f"Copula-GARCH({self.copula})",
            metadata={
                "copula": self.copula,
                "copula_df": self._copula_df,
                "VaR_95_equal_weight": var_95,
                "ES_95_equal_weight": es_95,
                "sigma_forecasts": sigma_forecasts.tolist(),
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
        R_new = np.atleast_2d(np.asarray(new_returns, dtype=np.float64))
        if R_new.shape[0] < R_new.shape[1]:
            R_new = R_new.T
        for obs in range(R_new.shape[0]):
            r_t = R_new[obs]
            h_new = np.empty(self._N, dtype=np.float64)
            for n in range(self._N):
                h_new[n] = max(
                    self._omega[n]
                    + self._alpha_g[n] * self._r_last[n] ** 2
                    + self._beta_g[n] * self._h_last[n],
                    1e-20,
                )
            self._h_last = h_new
            self._r_last = r_t

    def get_params(self) -> dict[str, Any]:
        if not self._fitted:
            return {}
        return {
            "copula": self.copula,
            "copula_corr": self._copula_corr.tolist(),
            "copula_df": self._copula_df,
            "omega": self._omega.tolist(),
            "alpha": self._alpha_g.tolist(),
            "beta": self._beta_g.tolist(),
        }
