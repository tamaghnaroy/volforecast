"""
Distribution-free conformal prediction intervals for volatility forecasts.

Wraps any BaseForecaster to provide finite-sample valid coverage guarantees
without distributional assumptions. Two variants:

- SplitConformalVol: split-conformal (Vovk et al. 2005) — requires a held-out
  calibration set; valid in iid / exchangeable settings.
- OnlineConformalVol: adaptive online conformal (Gibbs & Candes 2021) — valid
  for arbitrary non-stationary sequences; updates alpha level after each step.

References
----------
Vovk, V., Gammerman, A., Shafer, G. (2005). Algorithmic Learning in a Random
    World. Springer.
Gibbs, I., Candes, E. (2021). "Adaptive Conformal Inference Under Distribution
    Shift." NeurIPS.
Zaffran, M., Feron, O., Goude, Y., Josse, J., Dieuleveut, A. (2022). "Adaptive
    Conformal Predictions for Time Series." ICML.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from volforecast.core.base import BaseForecaster, ForecastResult


class SplitConformalVol:
    """Split-conformal prediction interval wrapper for any BaseForecaster.

    Protocol
    --------
    1. Fit the base forecaster on a training set.
    2. Call calibrate() with a held-out calibration set; this computes
       nonconformity scores and stores the empirical quantile.
    3. Call predict_interval() to get a symmetric interval around the base
       point forecast with coverage guarantee 1 - alpha.

    Coverage guarantee (exchangeable data):
      P(y_{n+1} in [f_{n+1} - q, f_{n+1} + q]) >= 1 - alpha

    where q = quantile(|r_i|, ceil((n+1)(1-alpha)) / n) and r_i are
    calibration residuals.

    Parameters
    ----------
    forecaster : BaseForecaster
        Any fitted or unfitted BaseForecaster instance.
    alpha : float
        Miscoverage rate. Default 0.10 gives 90 % coverage intervals.
    """

    def __init__(self, forecaster: BaseForecaster, alpha: float = 0.10) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.forecaster = forecaster
        self.alpha = alpha
        self._q_hat: float = float("inf")
        self._calibrated = False

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "SplitConformalVol":
        """Fit the underlying forecaster."""
        self.forecaster.fit(returns, realized_measures, **kwargs)
        return self

    def calibrate(
        self,
        cal_returns: NDArray[np.float64],
        cal_actuals: NDArray[np.float64],
        cal_realized: Optional[dict[str, NDArray[np.float64]]] = None,
    ) -> "SplitConformalVol":
        """Compute the conformal quantile from a calibration set.

        Parameters
        ----------
        cal_returns : array (n_cal,)
            Calibration return series (used for rolling predict).
        cal_actuals : array (n_cal,)
            Realized volatility proxy for calibration period.
        cal_realized : dict, optional
            Realized measures for calibration period.
        """
        n = len(cal_actuals)
        scores = np.empty(n, dtype=np.float64)

        rv_store = None
        if cal_realized is not None and "RV" in cal_realized:
            rv_store = np.asarray(cal_realized["RV"], dtype=np.float64)

        for i in range(n):
            result = self.forecaster.predict(horizon=1)
            f_i = float(result.point[0])
            scores[i] = abs(f_i - float(cal_actuals[i]))

            new_r = cal_returns[i:i+1]
            new_rm: Optional[dict[str, NDArray[np.float64]]] = None
            if rv_store is not None:
                new_rm = {"RV": rv_store[i:i+1]}
            self.forecaster.update(new_r, new_rm)

        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(level, 1.0)
        self._q_hat = float(np.quantile(scores, level))
        self._calibrated = True
        return self

    def predict_interval(
        self, horizon: int = 1, **kwargs: Any
    ) -> tuple[ForecastResult, NDArray[np.float64], NDArray[np.float64]]:
        """Return point forecast and symmetric conformal interval.

        Returns
        -------
        result : ForecastResult
            Base point forecast.
        lower : array (horizon,)
            Lower bound: point - q_hat.
        upper : array (horizon,)
            Upper bound: point + q_hat.
        """
        if not self._calibrated:
            raise RuntimeError("Call calibrate() before predict_interval().")
        result = self.forecaster.predict(horizon=horizon, **kwargs)
        pts = result.point
        lower = np.maximum(pts - self._q_hat, 1e-20)
        upper = pts + self._q_hat
        return result, lower, upper

    @property
    def q_hat(self) -> float:
        """Empirical conformal quantile (half-width of the interval)."""
        return self._q_hat


class OnlineConformalVol:
    """Online adaptive conformal prediction for volatility (Gibbs & Candes 2021).

    Maintains a time-varying miscoverage rate alpha_t that is updated after
    each step. Provides asymptotic average coverage 1 - alpha regardless of
    the data-generating process (no exchangeability required).

    Update rule:
      alpha_{t+1} = alpha_t + gamma * (alpha - 1_{y_t not in interval_t})

    where gamma is the step size controlling responsiveness.

    Parameters
    ----------
    forecaster : BaseForecaster
        Any fitted BaseForecaster instance.
    alpha : float
        Target miscoverage rate (default 0.10 -> 90 % coverage).
    gamma : float
        Step size for alpha update. Smaller = smoother, larger = faster adapt.
    init_q : float, optional
        Initial half-width. If None, uses the unconditional volatility estimate.
    """

    def __init__(
        self,
        forecaster: BaseForecaster,
        alpha: float = 0.10,
        gamma: float = 0.005,
        init_q: Optional[float] = None,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.forecaster = forecaster
        self.alpha = alpha
        self.gamma = gamma
        self._alpha_t: float = alpha
        self._scores: list[float] = []
        self._q_t: float = init_q if init_q is not None else float("inf")

    def _update_q(self) -> None:
        if len(self._scores) < 5:
            return
        n = len(self._scores)
        level = np.ceil((n + 1) * (1.0 - self._alpha_t)) / n
        level = float(np.clip(level, 0.0, 1.0))
        self._q_t = float(np.quantile(self._scores, level))

    def step(
        self,
        new_return: float,
        actual: float,
        new_realized: Optional[dict[str, NDArray[np.float64]]] = None,
    ) -> tuple[float, float, float, bool]:
        """Observe one new step and update the adaptive alpha.

        Parameters
        ----------
        new_return : float
            New return observation.
        actual : float
            Realised volatility proxy (e.g. RV).
        new_realized : dict, optional
            Realized measures for the new observation.

        Returns
        -------
        forecast : float
            1-step-ahead point forecast (computed before update).
        lower, upper : float
            Prediction interval bounds.
        covered : bool
            Whether actual fell inside the interval.
        """
        result = self.forecaster.predict(horizon=1)
        f = float(result.point[0])

        lower = max(f - self._q_t, 1e-20)
        upper = f + self._q_t
        covered = bool(lower <= actual <= upper)

        score = abs(f - actual)
        self._scores.append(score)

        miss = 0.0 if covered else 1.0
        self._alpha_t = float(np.clip(
            self._alpha_t + self.gamma * (self.alpha - miss), 0.001, 0.999
        ))
        self._update_q()

        new_r = np.array([new_return], dtype=np.float64)
        self.forecaster.update(new_r, new_realized)

        return f, lower, upper, covered

    def predict_interval(
        self, horizon: int = 1
    ) -> tuple[ForecastResult, NDArray[np.float64], NDArray[np.float64]]:
        """Current interval based on stored q_t."""
        result = self.forecaster.predict(horizon=horizon)
        pts = result.point
        lower = np.maximum(pts - self._q_t, 1e-20)
        upper = pts + self._q_t
        return result, lower, upper

    @property
    def alpha_t(self) -> float:
        """Current adaptive miscoverage level."""
        return self._alpha_t

    @property
    def q_t(self) -> float:
        """Current half-width of prediction interval."""
        return self._q_t


def coverage_diagnostic(
    lowers: NDArray[np.float64],
    uppers: NDArray[np.float64],
    actuals: NDArray[np.float64],
) -> dict[str, float]:
    """Compute empirical coverage and interval width statistics.

    Parameters
    ----------
    lowers, uppers : array (T,)
        Lower and upper prediction interval bounds.
    actuals : array (T,)
        Realized volatility proxies.

    Returns
    -------
    dict with keys: coverage, mean_width, median_width, winkler_score
    """
    lowers = np.asarray(lowers, dtype=np.float64)
    uppers = np.asarray(uppers, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)

    covered = (actuals >= lowers) & (actuals <= uppers)
    widths = uppers - lowers
    winkler = np.where(
        covered, widths, widths + 2.0 * np.minimum(lowers - actuals, actuals - uppers)
    )

    return {
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(widths)),
        "median_width": float(np.median(widths)),
        "winkler_score": float(np.mean(winkler)),
    }
