"""
Numba-optimized realized volatility measures.

References
----------
- Andersen, Bollerslev, Diebold, Labys (2003). "Modeling and Forecasting
  Realized Volatility." Econometrica 71(2), 579-625.
- Barndorff-Nielsen & Shephard (2004). "Power and bipower variation..."
  Journal of Financial Econometrics 2(1), 1-37.
- Andersen, Dobrev, Schaumburg (2012). "Jump-robust volatility estimation..."
  Journal of Econometrics 169(1), 36-46.
- Barndorff-Nielsen, Hansen, Lunde, Shephard (2008). "Designing realized
  kernels..." Econometrica 76(6), 1481-1536.
- Zhang, Mykland, Ait-Sahalia (2005). "A tale of two time scales..."
  JASA 100(472), 1394-1411.
- Jacod, Li, Mykland, Podolskij, Vetter (2009). "Microstructure noise..."
  Stochastic Processes and their Applications 119(7), 2249-2276.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange


# ═══════════════════════════════════════════════════
# Realized Variance
# ═══════════════════════════════════════════════════

@njit(cache=True)
def _rv_core(intraday_returns: NDArray[np.float64]) -> float:
    """Sum of squared intraday returns."""
    s = 0.0
    for i in range(intraday_returns.shape[0]):
        s += intraday_returns[i] ** 2
    return s


def realized_variance(
    intraday_returns: NDArray[np.float64],
    annualize: bool = False,
    trading_days: int = 252,
) -> float:
    """Realized Variance: RV_t = sum_{i=1}^{n} r_{t,i}^2.

    Parameters
    ----------
    intraday_returns : array, shape (n,)
        Intraday log returns within one day.
    annualize : bool
        If True, multiply by trading_days.
    trading_days : int
        Number of trading days per year.

    Returns
    -------
    float
        Realized variance estimate.
    """
    rv = _rv_core(np.ascontiguousarray(intraday_returns, dtype=np.float64))
    if annualize:
        rv *= trading_days
    return rv


def realized_variance_series(
    intraday_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute RV for each row (day) of an intraday return matrix.

    Parameters
    ----------
    intraday_matrix : array, shape (T, n)
        Each row is one day of intraday returns.

    Returns
    -------
    array, shape (T,)
    """
    return _rv_series_core(np.ascontiguousarray(intraday_matrix, dtype=np.float64))


@njit(cache=True, parallel=True)
def _rv_series_core(mat: NDArray[np.float64]) -> NDArray[np.float64]:
    T = mat.shape[0]
    out = np.empty(T, dtype=np.float64)
    for t in prange(T):
        s = 0.0
        for i in range(mat.shape[1]):
            s += mat[t, i] ** 2
        out[t] = s
    return out


# ═══════════════════════════════════════════════════
# Bipower Variation
# ═══════════════════════════════════════════════════

# mu_1 = E[|Z|] = sqrt(2/pi) for Z ~ N(0,1)
_MU1 = np.sqrt(2.0 / np.pi)


@njit(cache=True)
def _bv_core(intraday_returns: NDArray[np.float64]) -> float:
    """Bipower Variation: BV_t = mu_1^{-2} * (n/(n-1)) * sum |r_i| * |r_{i-1}|."""
    n = intraday_returns.shape[0]
    if n < 2:
        return 0.0
    s = 0.0
    for i in range(1, n):
        s += np.abs(intraday_returns[i]) * np.abs(intraday_returns[i - 1])
    # Finite-sample correction: n/(n-1)
    return s * (n / (n - 1)) / (_MU1 ** 2)


def bipower_variation(intraday_returns: NDArray[np.float64]) -> float:
    """Bipower Variation (Barndorff-Nielsen & Shephard, 2004).

    Consistent estimator of integrated variance, robust to finite-activity jumps.

    BV_t = mu_1^{-2} * (n/(n-1)) * sum_{i=2}^{n} |r_{t,i}| |r_{t,i-1}|

    Parameters
    ----------
    intraday_returns : array, shape (n,)

    Returns
    -------
    float
    """
    return _bv_core(np.ascontiguousarray(intraday_returns, dtype=np.float64))


# ═══════════════════════════════════════════════════
# Median Realized Variance (MedRV)
# ═══════════════════════════════════════════════════

# Scaling constant for MedRV: pi / (6 - 4*sqrt(3) + pi)
_MEDRV_SCALE = np.pi / (6.0 - 4.0 * np.sqrt(3.0) + np.pi)


@njit(cache=True)
def _medrv_core(intraday_returns: NDArray[np.float64]) -> float:
    """Median RV: based on median of three consecutive |returns|."""
    n = intraday_returns.shape[0]
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(2, n):
        a = np.abs(intraday_returns[i - 2])
        b = np.abs(intraday_returns[i - 1])
        c = np.abs(intraday_returns[i])
        # median of (a, b, c) = a + b + c - max(a,b,c) - min(a,b,c)
        med = a + b + c - max(a, max(b, c)) - min(a, min(b, c))
        s += med ** 2
    return s * (n / (n - 2)) * _MEDRV_SCALE


def median_rv(intraday_returns: NDArray[np.float64]) -> float:
    """Median Realized Variance (Andersen, Dobrev, Schaumburg, 2012).

    More robust to jumps than BV, uses median of three consecutive |returns|.

    Parameters
    ----------
    intraday_returns : array, shape (n,)

    Returns
    -------
    float
    """
    return _medrv_core(np.ascontiguousarray(intraday_returns, dtype=np.float64))


# ═══════════════════════════════════════════════════
# MinRV
# ═══════════════════════════════════════════════════

# Scaling constant for MinRV: pi / (pi - 2)
_MINRV_SCALE = np.pi / (np.pi - 2.0)


@njit(cache=True)
def _minrv_core(intraday_returns: NDArray[np.float64]) -> float:
    """MinRV: based on minimum of two consecutive |returns|^2."""
    n = intraday_returns.shape[0]
    if n < 2:
        return 0.0
    s = 0.0
    for i in range(1, n):
        a = intraday_returns[i - 1] ** 2
        b = intraday_returns[i] ** 2
        s += min(a, b)
    return s * (n / (n - 1)) * _MINRV_SCALE


def min_rv(intraday_returns: NDArray[np.float64]) -> float:
    """MinRV (Andersen, Dobrev, Schaumburg, 2012).

    Uses minimum of two consecutive squared returns; highly jump-robust.

    Parameters
    ----------
    intraday_returns : array, shape (n,)

    Returns
    -------
    float
    """
    return _minrv_core(np.ascontiguousarray(intraday_returns, dtype=np.float64))


# ═══════════════════════════════════════════════════
# Realized Kernel (Parzen kernel)
# ═══════════════════════════════════════════════════

@njit(cache=True)
def _parzen_kernel(x: float) -> float:
    """Parzen kernel function."""
    ax = np.abs(x)
    if ax <= 0.5:
        return 1.0 - 6.0 * ax * ax + 6.0 * ax * ax * ax
    elif ax <= 1.0:
        return 2.0 * (1.0 - ax) ** 3
    return 0.0


@njit(cache=True)
def _bartlett_kernel(x: float) -> float:
    """Bartlett (Newey-West) kernel function."""
    ax = np.abs(x)
    if ax <= 1.0:
        return 1.0 - ax
    return 0.0


@njit(cache=True)
def _cubic_kernel(x: float) -> float:
    """Cubic (Priestley-Epanechnikov) kernel function."""
    ax = np.abs(x)
    if ax <= 1.0:
        return 1.0 - 3.0 * ax * ax + 2.0 * ax * ax * ax
    return 0.0


@njit(cache=True)
def _rk_core(intraday_returns: NDArray[np.float64], bandwidth: int) -> float:
    """Realized Kernel with Parzen kernel."""
    n = intraday_returns.shape[0]
    gamma0 = 0.0
    for i in range(n):
        gamma0 += intraday_returns[i] ** 2

    rk = gamma0
    for h in range(1, bandwidth + 1):
        gamma_h = 0.0
        for i in range(h, n):
            gamma_h += intraday_returns[i] * intraday_returns[i - h]
        weight = _parzen_kernel(h / (bandwidth + 1.0))
        rk += 2.0 * weight * gamma_h
    return rk


@njit(cache=True)
def _rk_bartlett_core(intraday_returns: NDArray[np.float64], bandwidth: int) -> float:
    """Realized Kernel with Bartlett kernel."""
    n = intraday_returns.shape[0]
    gamma0 = 0.0
    for i in range(n):
        gamma0 += intraday_returns[i] ** 2

    rk = gamma0
    for h in range(1, bandwidth + 1):
        gamma_h = 0.0
        for i in range(h, n):
            gamma_h += intraday_returns[i] * intraday_returns[i - h]
        weight = _bartlett_kernel(h / (bandwidth + 1.0))
        rk += 2.0 * weight * gamma_h
    return rk


@njit(cache=True)
def _rk_cubic_core(intraday_returns: NDArray[np.float64], bandwidth: int) -> float:
    """Realized Kernel with cubic kernel."""
    n = intraday_returns.shape[0]
    gamma0 = 0.0
    for i in range(n):
        gamma0 += intraday_returns[i] ** 2

    rk = gamma0
    for h in range(1, bandwidth + 1):
        gamma_h = 0.0
        for i in range(h, n):
            gamma_h += intraday_returns[i] * intraday_returns[i - h]
        weight = _cubic_kernel(h / (bandwidth + 1.0))
        rk += 2.0 * weight * gamma_h
    return rk


def _auto_bandwidth_rk(
    intraday_returns: NDArray[np.float64],
    kernel: str = "parzen",
) -> int:
    """Rule-of-thumb bandwidth (Barndorff-Nielsen et al. 2009, eq. 26).

    H* = c_star * xi^{4/5} * n^{3/5}  where xi^2 = IQ / RV^2.
    """
    r = intraday_returns
    n = len(r)
    rk_naive = float(np.sum(r ** 2))
    if rk_naive < 1e-20:
        return max(1, int(np.ceil(n ** 0.6)))
    iq_est = float(n * np.sum(r ** 4) / 3.0)
    xi2 = iq_est / max(rk_naive ** 2, 1e-30)
    c_star = {"parzen": 3.5134, "bartlett": 2.8284, "cubic": 3.1484}.get(kernel, 3.5134)
    H_star = c_star * (xi2 ** 0.4) * (n ** 0.6)
    return max(1, int(np.round(H_star)))


def realized_kernel(
    intraday_returns: NDArray[np.float64],
    bandwidth: int | None = None,
    kernel: str = "parzen",
) -> float:
    """Realized Kernel (Barndorff-Nielsen, Hansen, Lunde, Shephard, 2008/2009).

    Noise-robust estimator of integrated variance.

    Parameters
    ----------
    intraday_returns : array, shape (n,)
    bandwidth : int, optional
        Kernel bandwidth H. If None, uses the Barndorff-Nielsen et al. (2009)
        rule-of-thumb: H* = c_star * xi^{4/5} * n^{3/5}.
    kernel : str
        Kernel function. One of "parzen" (default), "bartlett", "cubic".

    Returns
    -------
    float
    """
    r = np.ascontiguousarray(intraday_returns, dtype=np.float64)
    n = r.shape[0]
    if bandwidth is None:
        bandwidth = _auto_bandwidth_rk(r, kernel)
    if kernel == "bartlett":
        return _rk_bartlett_core(r, bandwidth)
    elif kernel == "cubic":
        return _rk_cubic_core(r, bandwidth)
    return _rk_core(r, bandwidth)


# ═══════════════════════════════════════════════════
# Two-Scale Realized Variance (TSRV)
# ═══════════════════════════════════════════════════

@njit(cache=True)
def _tsrv_core(prices: NDArray[np.float64], K: int) -> float:
    """Two-Scale RV: RV^{(all)} - (n_bar/n) * RV^{(1)}.

    Uses K subgrids.
    """
    n = prices.shape[0] - 1  # number of returns
    if n < 2 or K < 2:
        return 0.0

    # RV on full grid (all ticks)
    rv_all = 0.0
    for i in range(1, n + 1):
        rv_all += (prices[i] - prices[i - 1]) ** 2

    # Average RV on K sparse subgrids
    rv_sparse = 0.0
    for k in range(K):
        # Subgrid: indices k, k+K, k+2K, ...
        rv_k = 0.0
        count = 0
        prev = k
        idx = k + K
        while idx <= n:
            rv_k += (prices[idx] - prices[prev]) ** 2
            prev = idx
            idx += K
            count += 1
        if count > 0:
            rv_sparse += rv_k
    rv_sparse /= K

    # Bias correction
    n_bar = float(n) / K
    tsrv = rv_sparse - (n_bar / n) * rv_all
    return max(tsrv, 0.0)


def tsrv(
    prices: NDArray[np.float64],
    K: int | None = None,
) -> float:
    """Two-Scale Realized Variance (Zhang, Mykland, Ait-Sahalia, 2005).

    Parameters
    ----------
    prices : array, shape (n+1,)
        Log prices (not returns).
    K : int, optional
        Number of subgrids. If None, uses K ~ n^{2/3}.

    Returns
    -------
    float
    """
    p = np.ascontiguousarray(prices, dtype=np.float64)
    n = p.shape[0] - 1
    if K is None:
        K = max(2, int(np.ceil(n ** (2.0 / 3.0))))
    return _tsrv_core(p, K)


# ═══════════════════════════════════════════════════
# Pre-Averaging Estimator
# ═══════════════════════════════════════════════════

@njit(cache=True)
def _preavg_core(prices: NDArray[np.float64], kn: int) -> float:
    """Pre-averaging estimator of integrated variance."""
    n = prices.shape[0] - 1
    if n < 2 * kn:
        return 0.0

    # Weight function g(x) = min(x, 1-x) (piecewise linear)
    # psi_1 = integral of g(x)^2 dx = 1/12 for this choice
    # psi_2 = integral of g'(x)^2 dx = 1 for this choice
    psi1_kn = 1.0 / 12.0
    psi2_kn = 1.0

    # Compute pre-averaged returns: Y_bar_i = sum_{j=1}^{kn-1} g(j/kn) * delta_price_{i+j}
    m = n - kn + 1
    ybar = np.zeros(m, dtype=np.float64)
    for i in range(m):
        s = 0.0
        for j in range(1, kn):
            gval = min(float(j) / kn, 1.0 - float(j) / kn)
            s += gval * (prices[i + j] - prices[i + j - 1])
        ybar[i] = s

    # Main term: sum of Y_bar^2
    main_sum = 0.0
    for i in range(m):
        main_sum += ybar[i] ** 2
    main_sum /= (m * psi1_kn)

    # Bias correction: subtract noise term
    noise_sum = 0.0
    for i in range(1, n + 1):
        noise_sum += (prices[i] - prices[i - 1]) ** 2
    bias = psi2_kn / (2.0 * kn * psi1_kn) * noise_sum / n

    return max(main_sum - bias, 0.0)


def pre_averaging(
    prices: NDArray[np.float64],
    kn: int | None = None,
) -> float:
    """Pre-Averaging Estimator (Jacod et al., 2009).

    Parameters
    ----------
    prices : array, shape (n+1,)
        Log prices.
    kn : int, optional
        Window size. If None, uses kn ~ n^{1/2}.

    Returns
    -------
    float
    """
    p = np.ascontiguousarray(prices, dtype=np.float64)
    n = p.shape[0] - 1
    if kn is None:
        kn = max(2, int(np.ceil(np.sqrt(n))))
    return _preavg_core(p, kn)


# ═══════════════════════════════════════════════════
# Realized Semi-Variance
# ═══════════════════════════════════════════════════

@njit(cache=True)
def _rsv_core(intraday_returns: NDArray[np.float64]) -> tuple:
    """Realized semi-variances: positive and negative."""
    n = intraday_returns.shape[0]
    rs_pos = 0.0
    rs_neg = 0.0
    for i in range(n):
        r = intraday_returns[i]
        if r > 0.0:
            rs_pos += r * r
        else:
            rs_neg += r * r
    return rs_pos, rs_neg


def realized_semivariance(
    intraday_returns: NDArray[np.float64],
) -> tuple[float, float]:
    """Realized Semi-Variances (Barndorff-Nielsen, Kinnebrock, Shephard, 2010).

    RS^+ = sum r_i^2 * I(r_i > 0)
    RS^- = sum r_i^2 * I(r_i < 0)

    Parameters
    ----------
    intraday_returns : array, shape (n,)

    Returns
    -------
    tuple (RS_positive, RS_negative)
    """
    r = np.ascontiguousarray(intraday_returns, dtype=np.float64)
    return _rsv_core(r)
