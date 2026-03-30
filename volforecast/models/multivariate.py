"""
Multivariate volatility forecasters.

Models
------
DCCGARCHForecaster : Dynamic Conditional Correlation GARCH (Engle, 2002).
    Two-stage estimator: GARCH marginals then DCC correlation dynamics.
    Outputs per-asset conditional variances and the full conditional
    covariance matrix H_t.

References
----------
Engle, R. (2002). "Dynamic Conditional Correlation: A Simple Class of
    Multivariate Generalized Autoregressive Conditional Heteroscedasticity
    Models." Journal of Business & Economic Statistics 20(3), 339-350.
Engle, R., Sheppard, K. (2001). "Theoretical and Empirical Properties of
    Dynamic Conditional Correlation Multivariate GARCH." NBER WP 8554.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from numba import njit
from scipy.optimize import minimize

from volforecast.core.base import BaseForecaster, ForecastResult, ModelSpec
from volforecast.core.targets import VolatilityTarget, TargetSpec


# ═══════════════════════════════════════════════════
# Numba-optimized GARCH(1,1) univariate filter
# ═══════════════════════════════════════════════════

@njit(cache=True)
def _garch11_filter_mv(
    r: NDArray[np.float64],
    omega: float,
    alpha: float,
    beta: float,
) -> NDArray[np.float64]:
    """GARCH(1,1) conditional variance series."""
    T = r.shape[0]
    h = np.empty(T, dtype=np.float64)
    unc = omega / max(1.0 - alpha - beta, 1e-8)
    h[0] = unc
    for t in range(1, T):
        h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
    return h


@njit(cache=True)
def _dcc_filter(
    Z: NDArray[np.float64],
    a: float,
    b: float,
    Q_bar: NDArray[np.float64],
) -> tuple:
    """DCC filter: Q_t = (1-a-b)*Q_bar + a*z_{t-1}z_{t-1}' + b*Q_{t-1}.

    Parameters
    ----------
    Z : (T, N) standardized residuals (z_t = r_t / sigma_t)
    a, b : DCC parameters (a + b < 1)
    Q_bar : (N, N) unconditional covariance of Z

    Returns
    -------
    R_series : (T, N, N) correlation matrices
    Q_last : (N, N) last Q matrix (for forecasting)
    ll : float DCC log-likelihood contribution
    """
    T, N = Z.shape[0], Z.shape[1]
    R_series = np.empty((T, N, N), dtype=np.float64)
    ll = 0.0

    Q = Q_bar.copy()

    for t in range(T):
        if t > 0:
            z_prev = Z[t - 1]
            Q_new = np.empty((N, N), dtype=np.float64)
            for i in range(N):
                for j in range(N):
                    Q_new[i, j] = (
                        (1.0 - a - b) * Q_bar[i, j]
                        + a * z_prev[i] * z_prev[j]
                        + b * Q[i, j]
                    )
            Q = Q_new

        Q_diag_inv_sqrt = np.empty(N, dtype=np.float64)
        for i in range(N):
            Q_diag_inv_sqrt[i] = 1.0 / max(np.sqrt(Q[i, i]), 1e-10)

        R = np.empty((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(N):
                R[i, j] = Q[i, j] * Q_diag_inv_sqrt[i] * Q_diag_inv_sqrt[j]
        R_series[t] = R

        det_R = 0.0
        if N == 2:
            det_R = R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0]
        else:
            det_R = max(_det_nxn(R, N), 1e-300)

        if det_R > 0.0:
            z_t = Z[t]
            R_inv = _inv_nxn(R, N)
            zt_Rinv_z = 0.0
            for i in range(N):
                for j in range(N):
                    zt_Rinv_z += z_t[i] * R_inv[i, j] * z_t[j]
            zt_zt = 0.0
            for i in range(N):
                zt_zt += z_t[i] ** 2
            ll += -0.5 * (np.log(max(det_R, 1e-300)) + zt_Rinv_z - zt_zt)

    return R_series, Q, ll


@njit(cache=True)
def _det_nxn(A: NDArray[np.float64], N: int) -> float:
    """Determinant via LU (works for small N)."""
    if N == 1:
        return A[0, 0]
    if N == 2:
        return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    M = A.copy()
    det = 1.0
    for col in range(N):
        pivot = M[col, col]
        if abs(pivot) < 1e-14:
            return 0.0
        det *= pivot
        for row in range(col + 1, N):
            factor = M[row, col] / pivot
            for k in range(col, N):
                M[row, k] -= factor * M[col, k]
    return det


@njit(cache=True)
def _inv_nxn(A: NDArray[np.float64], N: int) -> NDArray[np.float64]:
    """Matrix inverse via Gauss-Jordan for small N."""
    M = np.empty((N, 2 * N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            M[i, j] = A[i, j]
            M[i, j + N] = 1.0 if i == j else 0.0

    for col in range(N):
        pivot = M[col, col]
        if abs(pivot) < 1e-14:
            pivot = 1e-14
        for k in range(2 * N):
            M[col, k] /= pivot
        for row in range(N):
            if row != col:
                factor = M[row, col]
                for k in range(2 * N):
                    M[row, k] -= factor * M[col, k]

    inv = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            inv[i, j] = M[i, j + N]
    return inv


class DCCGARCHForecaster(BaseForecaster):
    """Dynamic Conditional Correlation GARCH (Engle, 2002).

    Two-stage estimation:
    1. Fit GARCH(1,1) for each asset independently (marginal stage).
    2. Fit DCC parameters (a, b) on standardized residuals (DCC stage).

    Forecast output:
    - ``point`` : per-asset conditional standard deviations, shape (N, horizon).
    - ``metadata["H"]`` : conditional covariance matrix at horizon 1, (N, N).
    - ``metadata["R"]`` : conditional correlation matrix at horizon 1, (N, N).

    Parameters
    ----------
    n_assets : int
        Number of assets. Inferred from data if not set.

    Notes
    -----
    The ``fit`` method accepts a 2-D returns array (T, N). The ``returns``
    argument of ``BaseForecaster.fit`` is broadcast to shape (T, N); pass
    a 2-D array directly.
    """

    def __init__(self) -> None:
        self._omega: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._alpha_g: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._beta_g: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._dcc_a: float = 0.05
        self._dcc_b: float = 0.90
        self._Q_bar: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._Q_last: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._h_last: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._r_last: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._returns2d: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._N: int = 0
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name=f"DCC-GARCH({self._N})",
            abbreviation="DCC",
            family="Multivariate",
            target=VolatilityTarget.CONDITIONAL_VARIANCE,
            assumptions=(
                "GARCH(1,1) marginals",
                "DCC correlation dynamics",
                "Gaussian innovations",
            ),
            complexity="O(T * N^2)",
            reference="Engle (2002), J. Business & Economic Statistics",
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "DCCGARCHForecaster":
        """Fit DCC-GARCH.

        Parameters
        ----------
        returns : array (T, N) or (T,)
            Multi-asset return matrix. If 1-D, treated as a single asset.
        """
        R = np.atleast_2d(np.asarray(returns, dtype=np.float64))
        if R.shape[0] < R.shape[1]:
            R = R.T
        T, N = R.shape
        self._N = N
        self._returns2d = R

        if T < 50:
            raise ValueError("DCC-GARCH requires at least 50 observations")

        # ── Stage 1: GARCH marginals ──────────────────────────────────────────
        omega = np.empty(N, dtype=np.float64)
        alpha_g = np.empty(N, dtype=np.float64)
        beta_g = np.empty(N, dtype=np.float64)
        H = np.empty((T, N), dtype=np.float64)
        Z = np.empty((T, N), dtype=np.float64)

        var_r = np.var(R, axis=0)

        for n in range(N):
            r_n = R[:, n]

            def neg_ll_garch(x: NDArray[np.float64], r: NDArray[np.float64] = r_n) -> float:
                om = np.exp(x[0]) * float(var_r[n])
                al = 1.0 / (1.0 + np.exp(-x[1])) * 0.5
                be = 1.0 / (1.0 + np.exp(-x[2])) * 0.97
                if al + be >= 1.0:
                    return 1e10
                h_t = _garch11_filter_mv(r, om, al, be)
                ll = 0.0
                for t in range(len(r)):
                    ht = max(h_t[t], 1e-20)
                    ll += -0.5 * (np.log(2.0 * np.pi) + np.log(ht) + r[t] ** 2 / ht)
                return -ll

            res = minimize(
                neg_ll_garch, [0.0, 0.0, 2.0], method="Nelder-Mead",
                options={"maxiter": 5000, "xatol": 1e-7},
            )
            om = np.exp(res.x[0]) * float(var_r[n])
            al = 1.0 / (1.0 + np.exp(-res.x[1])) * 0.5
            be = 1.0 / (1.0 + np.exp(-res.x[2])) * 0.97
            omega[n], alpha_g[n], beta_g[n] = om, al, be

            h_n = _garch11_filter_mv(R[:, n], om, al, be)
            H[:, n] = h_n
            Z[:, n] = R[:, n] / np.sqrt(np.maximum(h_n, 1e-20))

        self._omega, self._alpha_g, self._beta_g = omega, alpha_g, beta_g

        # ── Stage 2: DCC ──────────────────────────────────────────────────────
        Q_bar = np.cov(Z.T)
        if N == 1:
            Q_bar = Q_bar.reshape(1, 1)
        self._Q_bar = Q_bar

        def neg_ll_dcc(x: NDArray[np.float64]) -> float:
            a = 1.0 / (1.0 + np.exp(-x[0])) * 0.3
            b = 1.0 / (1.0 + np.exp(-x[1])) * 0.97
            if a + b >= 1.0:
                return 1e10
            _, _, ll = _dcc_filter(Z, a, b, Q_bar)
            return -ll

        res2 = minimize(
            neg_ll_dcc, [0.0, 2.0], method="Nelder-Mead",
            options={"maxiter": 3000, "xatol": 1e-7},
        )
        self._dcc_a = float(1.0 / (1.0 + np.exp(-res2.x[0])) * 0.3)
        self._dcc_b = float(1.0 / (1.0 + np.exp(-res2.x[1])) * 0.97)

        _, self._Q_last, _ = _dcc_filter(Z, self._dcc_a, self._dcc_b, Q_bar)
        self._h_last = H[-1].copy()
        self._r_last = R[-1].copy()
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        N = self._N
        h = self._h_last.copy()
        r2 = self._r_last ** 2
        Q = self._Q_last.copy()

        sigma_forecasts = np.empty((horizon, N), dtype=np.float64)

        for step in range(horizon):
            h_next = np.empty(N, dtype=np.float64)
            for n in range(N):
                h_next[n] = self._omega[n] + self._alpha_g[n] * r2[n] + self._beta_g[n] * h[n]
                h_next[n] = max(h_next[n], 1e-20)
            sigma_forecasts[step] = np.sqrt(h_next)

            Q_diag_inv_sqrt = np.array([1.0 / max(np.sqrt(Q[i, i]), 1e-10) for i in range(N)])
            R_mat = Q * np.outer(Q_diag_inv_sqrt, Q_diag_inv_sqrt)

            h_prev_fcast = h if step == 0 else h_next
            z2 = r2 / np.maximum(h_prev_fcast, 1e-20)
            Q = (1.0 - self._dcc_a - self._dcc_b) * self._Q_bar
            Q += self._dcc_a * np.outer(np.sqrt(z2), np.sqrt(z2)) * R_mat
            Q += self._dcc_b * self._Q_last

            r2 = h_next.copy()
            h = h_next.copy()

        Q_diag_inv_sqrt = np.array([1.0 / max(np.sqrt(self._Q_last[i, i]), 1e-10) for i in range(N)])
        R_now = self._Q_last * np.outer(Q_diag_inv_sqrt, Q_diag_inv_sqrt)
        D_now = np.diag(np.sqrt(self._h_last))
        H_now = D_now @ R_now @ D_now

        return ForecastResult(
            point=sigma_forecasts[0],
            target_spec=TargetSpec(
                target=VolatilityTarget.CONDITIONAL_VARIANCE,
                horizon=horizon,
                proxy_description="Per-asset realized variance",
            ),
            model_name=f"DCC-GARCH({N})",
            metadata={
                "H": H_now.tolist(),
                "R": R_now.tolist(),
                "dcc_a": self._dcc_a,
                "dcc_b": self._dcc_b,
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

        self._returns2d = np.vstack([self._returns2d, R_new])
        N = self._N

        for obs in range(R_new.shape[0]):
            r_t = R_new[obs]
            h_new = np.empty(N, dtype=np.float64)
            for n in range(N):
                h_new[n] = max(
                    self._omega[n]
                    + self._alpha_g[n] * self._r_last[n] ** 2
                    + self._beta_g[n] * self._h_last[n],
                    1e-20,
                )
            z_prev = self._r_last / np.sqrt(np.maximum(self._h_last, 1e-20))
            Q_new = (
                (1.0 - self._dcc_a - self._dcc_b) * self._Q_bar
                + self._dcc_a * np.outer(z_prev, z_prev)
                + self._dcc_b * self._Q_last
            )
            self._Q_last = Q_new
            self._h_last = h_new
            self._r_last = r_t

    def get_params(self) -> dict[str, Any]:
        if not self._fitted:
            return {}
        return {
            "omega": self._omega.tolist(),
            "alpha": self._alpha_g.tolist(),
            "beta": self._beta_g.tolist(),
            "dcc_a": self._dcc_a,
            "dcc_b": self._dcc_b,
        }
