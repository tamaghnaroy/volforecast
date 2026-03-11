"""
HAR (Heterogeneous Autoregressive) model family.

Models:
- HAR-RV: Corsi (2009), JFE
- HAR-RV-J: Andersen, Bollerslev, Diebold (2007), REStat — adds jump component
- HAR-RV-CJ: Andersen, Bollerslev, Diebold (2007) — separates C and J dynamics
- SHAR: Patton & Sheppard (2015), JFQA — uses realized semi-variances

All target integrated variance (or continuous variation for HAR-RV-CJ).
OLS estimation with Newey-West standard errors.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from numba import njit

from volforecast.core.base import BaseForecaster, ForecastResult, ModelSpec
from volforecast.core.targets import VolatilityTarget, TargetSpec, IV_1STEP, CV_1STEP


# ═══════════════════════════════════════════════════
# Numba-optimized HAR feature construction
# ═══════════════════════════════════════════════════

@njit(cache=True)
def _build_har_features(
    rv: NDArray[np.float64],
    lags: tuple = (1, 5, 22),
) -> NDArray[np.float64]:
    """Build HAR regressors: RV_d, RV_w, RV_m (daily, weekly, monthly averages).

    Parameters
    ----------
    rv : array, shape (T,)
        Daily realized variance series.
    lags : tuple of int
        Averaging windows (default: 1, 5, 22 for daily, weekly, monthly).

    Returns
    -------
    X : array, shape (T - max_lag, len(lags))
        HAR feature matrix.
    """
    T = rv.shape[0]
    max_lag = 0
    for l in lags:
        if l > max_lag:
            max_lag = l

    n_features = len(lags)
    n_obs = T - max_lag
    X = np.empty((n_obs, n_features), dtype=np.float64)

    for t in range(n_obs):
        idx = t + max_lag
        for j in range(n_features):
            lag = lags[j]
            s = 0.0
            for k in range(lag):
                s += rv[idx - 1 - k]
            X[t, j] = s / lag

    return X


@njit(cache=True)
def _ols_fit(X: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """OLS estimation: beta = (X'X)^{-1} X'y with intercept prepended.

    X is assumed to NOT include an intercept column — one is added internally.
    """
    n = X.shape[0]
    k = X.shape[1]

    # Add intercept
    X_aug = np.empty((n, k + 1), dtype=np.float64)
    for i in range(n):
        X_aug[i, 0] = 1.0
        for j in range(k):
            X_aug[i, j + 1] = X[i, j]

    # X'X
    XtX = np.zeros((k + 1, k + 1), dtype=np.float64)
    for i in range(n):
        for j1 in range(k + 1):
            for j2 in range(k + 1):
                XtX[j1, j2] += X_aug[i, j1] * X_aug[i, j2]

    # X'y
    Xty = np.zeros(k + 1, dtype=np.float64)
    for i in range(n):
        for j in range(k + 1):
            Xty[j] += X_aug[i, j] * y[i]

    # Solve via lstsq for numerical robustness (handles near-singular X'X)
    beta = np.linalg.lstsq(XtX, Xty)[0]
    return beta


# ═══════════════════════════════════════════════════
# HAR-RV Forecaster
# ═══════════════════════════════════════════════════

class HARForecaster(BaseForecaster):
    """HAR-RV model (Corsi, 2009).

    RV_{t+1} = beta_0 + beta_d * RV_t^{(d)} + beta_w * RV_t^{(w)} + beta_m * RV_t^{(m)} + e_{t+1}

    where RV_t^{(k)} = (1/k) * sum_{i=1}^{k} RV_{t-i+1}.

    Parameters
    ----------
    lags : tuple of int
        Averaging windows. Default (1, 5, 22) = daily, weekly, monthly.
    log_transform : bool
        If True, model log(RV) instead of RV (often improves fit).
    """

    def __init__(
        self,
        lags: tuple[int, ...] = (1, 5, 22),
        log_transform: bool = False,
    ) -> None:
        self.lags = lags
        self.log_transform = log_transform
        self._beta: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._rv: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._residual_var: float = 0.0
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name="HAR-RV",
            abbreviation="HAR",
            family="HAR",
            target=VolatilityTarget.INTEGRATED_VARIANCE,
            assumptions=("heterogeneous agents", "RV as proxy for IV", "linear"),
            complexity="O(T) OLS",
            reference="Corsi (2009), JFE",
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "HARForecaster":
        if realized_measures is None or "RV" not in realized_measures:
            raise ValueError("HAR-RV requires realized_measures={'RV': array}")

        rv = np.asarray(realized_measures["RV"], dtype=np.float64)
        self._rv = rv.copy()

        if self.log_transform:
            rv_work = np.log(np.maximum(rv, 1e-20))
        else:
            rv_work = rv

        max_lag = max(self.lags)
        X = _build_har_features(rv_work, self.lags)
        y = rv_work[max_lag:]

        self._beta = _ols_fit(X, y)
        resid = y - self._predict_insample(X)
        self._residual_var = float(np.var(resid))
        self._fitted = True
        return self

    def _predict_insample(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """In-sample prediction from feature matrix."""
        n = X.shape[0]
        pred = np.full(n, self._beta[0], dtype=np.float64)
        for j in range(X.shape[1]):
            pred += self._beta[j + 1] * X[:, j]
        return pred

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        rv = self._rv
        if self.log_transform:
            rv_work = np.log(np.maximum(rv, 1e-20))
        else:
            rv_work = rv

        # Build features from the last observation
        forecasts = np.empty(horizon, dtype=np.float64)
        rv_extended = rv_work.copy()

        for h in range(horizon):
            x = np.empty(len(self.lags), dtype=np.float64)
            T_curr = len(rv_extended)
            for j, lag in enumerate(self.lags):
                x[j] = np.mean(rv_extended[max(0, T_curr - lag):T_curr])

            pred = self._beta[0]
            for j in range(len(self.lags)):
                pred += self._beta[j + 1] * x[j]

            if self.log_transform:
                # Jensen's inequality correction: E[RV] = exp(E[log RV] + 0.5*var)
                forecasts[h] = np.exp(pred + 0.5 * self._residual_var)
            else:
                forecasts[h] = max(pred, 1e-20)

            rv_extended = np.append(rv_extended, pred)

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.INTEGRATED_VARIANCE,
                horizon=horizon,
            ),
            model_name="HAR-RV",
            metadata={"beta": self._beta.tolist(), "log_transform": self.log_transform},
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
            raise ValueError("HAR update requires new_realized={'RV': array}")
        new_rv = np.asarray(new_realized["RV"], dtype=np.float64)
        self._rv = np.concatenate([self._rv, new_rv])

    def get_params(self) -> dict[str, Any]:
        if len(self._beta) == 0:
            return {}
        names = ["intercept"] + [f"beta_lag{l}" for l in self.lags]
        return dict(zip(names, self._beta.tolist()))


class HARJForecaster(BaseForecaster):
    """HAR-RV-J model (Andersen, Bollerslev, Diebold, 2007).

    RV_{t+1} = beta_0 + beta_d * RV_t^{(d)} + beta_w * RV_t^{(w)}
               + beta_m * RV_t^{(m)} + beta_j * J_t + e_{t+1}

    where J_t = max(RV_t - BV_t, 0) is the jump variation proxy.
    """

    def __init__(self, lags: tuple[int, ...] = (1, 5, 22)) -> None:
        self.lags = lags
        self._beta: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._rv: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._jv: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name="HAR-RV-J",
            abbreviation="HARJ",
            family="HAR",
            target=VolatilityTarget.INTEGRATED_VARIANCE,
            assumptions=("HAR + jump component regressor", "jump persistence"),
            complexity="O(T) OLS",
            reference="Andersen, Bollerslev, Diebold (2007), JFE",
            extends=("HAR_RV",),
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "HARJForecaster":
        if realized_measures is None or "RV" not in realized_measures:
            raise ValueError("HAR-RV-J requires 'RV' in realized_measures")

        rv = np.asarray(realized_measures["RV"], dtype=np.float64)
        self._rv = rv.copy()

        if "JV" in realized_measures:
            jv = np.asarray(realized_measures["JV"], dtype=np.float64)
        elif "BV" in realized_measures:
            bv = np.asarray(realized_measures["BV"], dtype=np.float64)
            jv = np.maximum(rv - bv, 0.0)
        else:
            raise ValueError("HAR-RV-J requires 'JV' or 'BV' in realized_measures")

        self._jv = jv.copy()
        max_lag = max(self.lags)

        X_har = _build_har_features(rv, self.lags)
        jv_aligned = jv[max_lag:].reshape(-1, 1)
        X = np.hstack([X_har, jv_aligned])
        y = rv[max_lag:]

        self._beta = _ols_fit(X, y)
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        rv = self._rv
        jv = self._jv

        forecasts = np.empty(horizon, dtype=np.float64)
        rv_ext = rv.copy()
        jv_ext = jv.copy()

        for h in range(horizon):
            x = np.empty(len(self.lags) + 1, dtype=np.float64)
            T_curr = len(rv_ext)
            for j, lag in enumerate(self.lags):
                x[j] = np.mean(rv_ext[max(0, T_curr - lag):T_curr])
            x[len(self.lags)] = jv_ext[-1]

            pred = self._beta[0]
            for j in range(len(self.lags) + 1):
                pred += self._beta[j + 1] * x[j]
            forecasts[h] = max(pred, 1e-20)

            rv_ext = np.append(rv_ext, pred)
            jv_ext = np.append(jv_ext, 0.0)

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.INTEGRATED_VARIANCE,
                horizon=horizon,
            ),
            model_name="HAR-RV-J",
            metadata={"beta": self._beta.tolist()},
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
            raise ValueError("Requires new_realized with 'RV'")
        self._rv = np.concatenate([self._rv, np.asarray(new_realized["RV"], dtype=np.float64)])
        if "JV" in new_realized:
            self._jv = np.concatenate([self._jv, np.asarray(new_realized["JV"], dtype=np.float64)])
        elif "BV" in new_realized:
            bv = np.asarray(new_realized["BV"], dtype=np.float64)
            new_rv = np.asarray(new_realized["RV"], dtype=np.float64)
            self._jv = np.concatenate([self._jv, np.maximum(new_rv - bv, 0.0)])

    def get_params(self) -> dict[str, Any]:
        if len(self._beta) == 0:
            return {}
        names = ["intercept"] + [f"beta_lag{l}" for l in self.lags] + ["beta_jump"]
        return dict(zip(names, self._beta.tolist()))


class HARCJForecaster(BaseForecaster):
    """HAR-RV-CJ model (Andersen, Bollerslev, Diebold, 2007).

    Separately models continuous and jump components:
    RV_{t+1} = beta_0 + beta_cd * C_t + beta_cw * C_t^{(w)} + beta_cm * C_t^{(m)}
               + beta_jd * J_t + beta_jw * J_t^{(w)} + beta_jm * J_t^{(m)} + e_{t+1}
    """

    def __init__(self, lags: tuple[int, ...] = (1, 5, 22)) -> None:
        self.lags = lags
        self._beta: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._cv: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._jv: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._rv: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name="HAR-RV-CJ",
            abbreviation="HARCJ",
            family="HAR",
            target=VolatilityTarget.CONTINUOUS_VARIATION,
            assumptions=("separate C and J dynamics", "BV proxy for C"),
            complexity="O(T) OLS",
            reference="Andersen, Bollerslev, Diebold (2007), JFE",
            extends=("HAR_RV", "HAR_RV_J"),
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "HARCJForecaster":
        if realized_measures is None or "RV" not in realized_measures:
            raise ValueError("HAR-RV-CJ requires 'RV' in realized_measures")

        rv = np.asarray(realized_measures["RV"], dtype=np.float64)
        self._rv = rv.copy()

        if "CV" in realized_measures and "JV" in realized_measures:
            cv = np.asarray(realized_measures["CV"], dtype=np.float64)
            jv = np.asarray(realized_measures["JV"], dtype=np.float64)
        elif "BV" in realized_measures:
            bv = np.asarray(realized_measures["BV"], dtype=np.float64)
            cv = np.minimum(bv, rv)
            jv = np.maximum(rv - bv, 0.0)
        else:
            raise ValueError("Requires 'CV'+'JV' or 'BV' in realized_measures")

        self._cv = cv.copy()
        self._jv = jv.copy()

        max_lag = max(self.lags)
        X_c = _build_har_features(cv, self.lags)
        X_j = _build_har_features(jv, self.lags)
        X = np.hstack([X_c, X_j])
        y = rv[max_lag:]

        self._beta = _ols_fit(X, y)
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        cv = self._cv
        jv = self._jv
        n_lags = len(self.lags)

        forecasts = np.empty(horizon, dtype=np.float64)
        cv_ext = cv.copy()
        jv_ext = jv.copy()

        for h in range(horizon):
            x = np.empty(2 * n_lags, dtype=np.float64)
            T_curr = len(cv_ext)
            for j, lag in enumerate(self.lags):
                x[j] = np.mean(cv_ext[max(0, T_curr - lag):T_curr])
                x[n_lags + j] = np.mean(jv_ext[max(0, T_curr - lag):T_curr])

            pred = self._beta[0]
            for j in range(2 * n_lags):
                pred += self._beta[j + 1] * x[j]
            forecasts[h] = max(pred, 1e-20)

            # Assume future jumps are zero for iterated forecast
            cv_ext = np.append(cv_ext, max(pred, 1e-20))
            jv_ext = np.append(jv_ext, 0.0)

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.CONTINUOUS_VARIATION,
                horizon=horizon,
            ),
            model_name="HAR-RV-CJ",
            metadata={"beta": self._beta.tolist()},
        )

    def update(
        self,
        new_returns: NDArray[np.float64],
        new_realized: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> None:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        if new_realized is None:
            raise ValueError("Requires realized measures for update")
        if "RV" in new_realized:
            self._rv = np.concatenate([
                self._rv, np.asarray(new_realized["RV"], dtype=np.float64)
            ])
        if "CV" in new_realized:
            self._cv = np.concatenate([
                self._cv, np.asarray(new_realized["CV"], dtype=np.float64)
            ])
        if "JV" in new_realized:
            self._jv = np.concatenate([
                self._jv, np.asarray(new_realized["JV"], dtype=np.float64)
            ])

    def get_params(self) -> dict[str, Any]:
        if len(self._beta) == 0:
            return {}
        names = (
            ["intercept"]
            + [f"beta_c_lag{l}" for l in self.lags]
            + [f"beta_j_lag{l}" for l in self.lags]
        )
        return dict(zip(names, self._beta.tolist()))


class SHARForecaster(BaseForecaster):
    """SHAR (Semi-variance HAR) model (Patton & Sheppard, 2015).

    Uses realized semi-variances (RS+ and RS-) instead of RV:
    RV_{t+1} = beta_0 + beta_d+ * RS+_t + beta_d- * RS-_t
               + beta_w * RV_t^{(w)} + beta_m * RV_t^{(m)} + e_{t+1}
    """

    def __init__(self, lags: tuple[int, ...] = (1, 5, 22)) -> None:
        self.lags = lags
        self._beta: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._rv: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._rs_pos: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._rs_neg: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._fitted = False

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            name="SHAR",
            abbreviation="SHAR",
            family="HAR",
            target=VolatilityTarget.INTEGRATED_VARIANCE,
            assumptions=("positive/negative semi-variances", "asymmetric response"),
            complexity="O(T) OLS",
            reference="Patton & Sheppard (2015), JFQA",
            extends=("HAR_RV",),
        )

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "SHARForecaster":
        if realized_measures is None:
            raise ValueError("SHAR requires 'RV', 'RS_pos', 'RS_neg' in realized_measures")

        rv = np.asarray(realized_measures["RV"], dtype=np.float64)
        self._rv = rv.copy()

        if "RS_pos" in realized_measures and "RS_neg" in realized_measures:
            rs_pos = np.asarray(realized_measures["RS_pos"], dtype=np.float64)
            rs_neg = np.asarray(realized_measures["RS_neg"], dtype=np.float64)
        else:
            # Default: split RV equally (poor proxy, but allows fitting)
            rs_pos = rv / 2.0
            rs_neg = rv / 2.0

        self._rs_pos = rs_pos.copy()
        self._rs_neg = rs_neg.copy()

        max_lag = max(self.lags)
        # Build all features with the same max_lag alignment
        X_rv = _build_har_features(rv, self.lags)       # (T-max_lag, len(lags))
        X_rsp = _build_har_features(rs_pos, self.lags)  # same shape
        X_rsn = _build_har_features(rs_neg, self.lags)  # same shape

        # Replace daily RV column with RS+ and RS- daily columns
        # X_rsp[:, 0] = daily RS+, X_rsn[:, 0] = daily RS-, X_rv[:, 1:] = weekly/monthly RV
        X = np.hstack([X_rsp[:, :1], X_rsn[:, :1], X_rv[:, 1:]])
        y = rv[max_lag:]

        self._beta = _ols_fit(X, y)
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        forecasts = np.empty(horizon, dtype=np.float64)
        rv_ext = self._rv.copy()
        rsp_ext = self._rs_pos.copy()
        rsn_ext = self._rs_neg.copy()

        for h in range(horizon):
            T_curr = len(rv_ext)
            x_list = [rsp_ext[-1], rsn_ext[-1]]
            for j, lag in enumerate(self.lags[1:], 1):
                x_list.append(np.mean(rv_ext[max(0, T_curr - lag):T_curr]))

            pred = self._beta[0]
            for j, xj in enumerate(x_list):
                pred += self._beta[j + 1] * xj
            forecasts[h] = max(pred, 1e-20)

            rv_ext = np.append(rv_ext, pred)
            rsp_ext = np.append(rsp_ext, pred / 2.0)
            rsn_ext = np.append(rsn_ext, pred / 2.0)

        return ForecastResult(
            point=forecasts,
            target_spec=TargetSpec(
                target=VolatilityTarget.INTEGRATED_VARIANCE,
                horizon=horizon,
            ),
            model_name="SHAR",
            metadata={"beta": self._beta.tolist()},
        )

    def update(
        self,
        new_returns: NDArray[np.float64],
        new_realized: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> None:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        if new_realized is not None:
            if "RV" in new_realized:
                self._rv = np.concatenate([
                    self._rv, np.asarray(new_realized["RV"], dtype=np.float64)
                ])
            if "RS_pos" in new_realized:
                self._rs_pos = np.concatenate([
                    self._rs_pos, np.asarray(new_realized["RS_pos"], dtype=np.float64)
                ])
            if "RS_neg" in new_realized:
                self._rs_neg = np.concatenate([
                    self._rs_neg, np.asarray(new_realized["RS_neg"], dtype=np.float64)
                ])

    def get_params(self) -> dict[str, Any]:
        if len(self._beta) == 0:
            return {}
        names = ["intercept", "beta_rs_pos", "beta_rs_neg"] + [
            f"beta_lag{l}" for l in self.lags[1:]
        ]
        return dict(zip(names, self._beta.tolist()))
