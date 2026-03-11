"""
Volatility forecast loss functions.

Implements loss functions that are:
1. Robust to proxy noise (Patton, 2011, Theorem 1)
2. Consistent for ranking forecasters even when the true volatility is latent

Key result (Patton, 2011):
  A loss function L(f, sigma^2) yields the same ranking of forecasters
  when evaluated at a *proxy* hat{sigma}^2 = sigma^2 + eta (eta noise)
  if and only if L belongs to the "robust" class:
    L(f, y) = C(y) + b(f) * (y - f) + a(f)
  where b(f) = a'(f). MSE and QLIKE are the two canonical members.

References
----------
- Patton (2011). "Volatility forecast comparison using imperfect volatility
  proxies." Journal of Econometrics 160(1), 246-256.
- Hansen & Lunde (2006). "Consistent ranking of volatility models."
  Journal of Econometrics 131(1-2), 97-121.
- Laurent, Rombouts, Violante (2013). "On loss functions and ranking
  forecasting models with and without jumps." JIMF.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from numba import njit


# ═══════════════════════════════════════════════════
# Robust loss functions (Patton, 2011)
# ═══════════════════════════════════════════════════

@njit(cache=True)
def _mse_core(forecasts: NDArray[np.float64], proxies: NDArray[np.float64]) -> float:
    """MSE loss: L = (f - y)^2. Robust to proxy noise."""
    n = forecasts.shape[0]
    s = 0.0
    for i in range(n):
        d = forecasts[i] - proxies[i]
        s += d * d
    return s / n


def mse_loss(
    forecasts: NDArray[np.float64],
    proxies: NDArray[np.float64],
) -> float:
    """Mean Squared Error loss (robust to proxy noise).

    L(f, y) = (f - y)^2

    Parameters
    ----------
    forecasts : array, shape (T,)
        Volatility forecasts.
    proxies : array, shape (T,)
        Volatility proxy (e.g., RV).

    Returns
    -------
    float
        Average MSE loss.
    """
    f = np.ascontiguousarray(forecasts, dtype=np.float64)
    y = np.ascontiguousarray(proxies, dtype=np.float64)
    return _mse_core(f, y)


@njit(cache=True)
def _qlike_core(forecasts: NDArray[np.float64], proxies: NDArray[np.float64]) -> float:
    """QLIKE loss: L = y/f + log(f). Robust to proxy noise."""
    n = forecasts.shape[0]
    s = 0.0
    for i in range(n):
        f = max(forecasts[i], 1e-20)
        s += proxies[i] / f + np.log(f)
    return s / n


def qlike_loss(
    forecasts: NDArray[np.float64],
    proxies: NDArray[np.float64],
) -> float:
    """QLIKE loss (robust to proxy noise).

    L(f, y) = y/f + log(f)

    The QLIKE loss is the negative of the Gaussian quasi-log-likelihood
    (up to constants). It penalizes under-prediction more heavily than
    over-prediction and is robust to noise in the proxy.

    Parameters
    ----------
    forecasts : array, shape (T,)
    proxies : array, shape (T,)

    Returns
    -------
    float
    """
    f = np.ascontiguousarray(forecasts, dtype=np.float64)
    y = np.ascontiguousarray(proxies, dtype=np.float64)
    return _qlike_core(f, y)


@njit(cache=True)
def _mae_core(forecasts: NDArray[np.float64], proxies: NDArray[np.float64]) -> float:
    """MAE loss (NOT robust to proxy noise)."""
    n = forecasts.shape[0]
    s = 0.0
    for i in range(n):
        s += np.abs(forecasts[i] - proxies[i])
    return s / n


def mae_loss(
    forecasts: NDArray[np.float64],
    proxies: NDArray[np.float64],
) -> float:
    """Mean Absolute Error loss.

    WARNING: MAE is NOT robust to proxy noise (Patton, 2011). Use MSE
    or QLIKE for reliable forecast comparison. Included for completeness.

    Parameters
    ----------
    forecasts, proxies : array, shape (T,)

    Returns
    -------
    float
    """
    f = np.ascontiguousarray(forecasts, dtype=np.float64)
    y = np.ascontiguousarray(proxies, dtype=np.float64)
    return _mae_core(f, y)


@njit(cache=True)
def _mse_log_core(forecasts: NDArray[np.float64], proxies: NDArray[np.float64]) -> float:
    """MSE of logs: L = (log f - log y)^2."""
    n = forecasts.shape[0]
    s = 0.0
    for i in range(n):
        lf = np.log(max(forecasts[i], 1e-20))
        ly = np.log(max(proxies[i], 1e-20))
        d = lf - ly
        s += d * d
    return s / n


def mse_log_loss(
    forecasts: NDArray[np.float64],
    proxies: NDArray[np.float64],
) -> float:
    """MSE of log-transformed values.

    L(f, y) = (log(f) - log(y))^2

    NOT in the Patton (2011) robust class, but commonly used.
    Useful when volatility is modeled in log space (EGARCH, log-HAR).

    Parameters
    ----------
    forecasts, proxies : array, shape (T,)

    Returns
    -------
    float
    """
    f = np.ascontiguousarray(forecasts, dtype=np.float64)
    y = np.ascontiguousarray(proxies, dtype=np.float64)
    return _mse_log_core(f, y)


def patton_robust_loss(
    forecasts: NDArray[np.float64],
    proxies: NDArray[np.float64],
    b: int = -2,
) -> float:
    """General Patton (2011) robust loss family.

    For b != 0, -1:
      L(f, y) = (1/(b*(b+1))) * (y^{b+1} - f^{b+1}) - (1/b)*f^b*(y - f)

    Special cases:
    - b = -2: QLIKE (up to affine transformation)
    - b = 1: MSE (up to affine transformation)

    Parameters
    ----------
    forecasts, proxies : array, shape (T,)
    b : int
        Loss function parameter. b=1 gives MSE, b=-2 gives QLIKE.

    Returns
    -------
    float
    """
    f = np.maximum(np.asarray(forecasts, dtype=np.float64), 1e-20)
    y = np.maximum(np.asarray(proxies, dtype=np.float64), 1e-20)

    if b == 1:
        return mse_loss(f, y)
    elif b == -2:
        return qlike_loss(f, y)
    elif b == 0:
        # L = y*log(y/f) - (y - f)  (exponential family)
        return float(np.mean(y * np.log(y / f) - (y - f)))
    elif b == -1:
        # L = y/f - log(y/f) - 1
        return float(np.mean(y / f - np.log(y / f) - 1.0))
    else:
        bp1 = b + 1
        term1 = (1.0 / (b * bp1)) * (y ** bp1 - f ** bp1)
        term2 = (1.0 / b) * f ** b * (y - f)
        return float(np.mean(term1 - term2))


def heterogeneous_loss(
    forecasts: NDArray[np.float64],
    proxies: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> float:
    """Heterogeneous loss combining MSE and QLIKE.

    L = w * MSE + (1-w) * QLIKE, averaged across time with optional
    time-varying weights (e.g., inverse of proxy level for robustness).

    Parameters
    ----------
    forecasts, proxies : array, shape (T,)
    weights : array, shape (T,), optional
        Weight on MSE component. Default: 0.5 (equal).

    Returns
    -------
    float
    """
    f = np.asarray(forecasts, dtype=np.float64)
    y = np.asarray(proxies, dtype=np.float64)
    T = len(f)

    if weights is None:
        w = np.full(T, 0.5, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)

    f_safe = np.maximum(f, 1e-20)
    mse_t = (f - y) ** 2
    qlike_t = y / f_safe + np.log(f_safe)

    return float(np.mean(w * mse_t + (1.0 - w) * qlike_t))
