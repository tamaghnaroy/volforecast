"""
Jump detection and decomposition.

Implements:
- BNS jump test (Barndorff-Nielsen & Shephard, 2006)
- Jump variation estimation
- Continuous variation estimation
- Full jump decomposition (C/J separation)

References
----------
- Barndorff-Nielsen & Shephard (2006). "Econometrics of testing for jumps
  in financial economics using bipower variation." JFE 4(1), 1-30.
- Huang & Tauchen (2005). "The relative contribution of jumps to total
  price variance." Journal of Financial Econometrics 3(4), 456-499.
- Andersen, Bollerslev, Diebold (2007). "Roughing it up: including jump
  components in the measurement, modeling, and forecasting of return
  volatility." Review of Economics and Statistics 89(4), 701-720.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from numba import njit
from scipy import stats

from volforecast.realized.measures import (
    realized_variance,
    bipower_variation,
    median_rv,
)


# ═══════════════════════════════════════════════════
# Tri-power and Quad-power quarticity for variance of BV
# ═══════════════════════════════════════════════════

_MU1 = np.sqrt(2.0 / np.pi)
# mu_{4/3} = E[|Z|^{4/3}] = 2^{2/3} * Gamma(7/6) / Gamma(1/2)
_MU43 = 2.0 ** (2.0 / 3.0) * np.exp(
    float(np.real(np.log(complex(math.gamma(7.0 / 6.0)))))
    - 0.5 * np.log(np.pi)
)


@njit(cache=True)
def _tripower_quarticity(intraday_returns: NDArray[np.float64]) -> float:
    """Tri-power quarticity for BNS test variance estimation.

    TPQ = n * mu_{4/3}^{-3} * (n/(n-2)) * sum |r_i|^{4/3} |r_{i-1}|^{4/3} |r_{i-2}|^{4/3}
    """
    n = intraday_returns.shape[0]
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(2, n):
        s += (
            np.abs(intraday_returns[i]) ** (4.0 / 3.0)
            * np.abs(intraday_returns[i - 1]) ** (4.0 / 3.0)
            * np.abs(intraday_returns[i - 2]) ** (4.0 / 3.0)
        )
    return s * n * (n / (n - 2))


@njit(cache=True)
def _quadpower_quarticity(intraday_returns: NDArray[np.float64]) -> float:
    """Quad-power quarticity: alternative variance estimator.

    QQQ = n * mu_1^{-4} * (n/(n-3)) * sum prod_{k=0}^{3} |r_{i-k}|
    """
    n = intraday_returns.shape[0]
    if n < 4:
        return 0.0
    s = 0.0
    for i in range(3, n):
        s += (
            np.abs(intraday_returns[i])
            * np.abs(intraday_returns[i - 1])
            * np.abs(intraday_returns[i - 2])
            * np.abs(intraday_returns[i - 3])
        )
    return s * n * (n / (n - 3))


# ═══════════════════════════════════════════════════
# BNS Jump Test
# ═══════════════════════════════════════════════════

@dataclass
class JumpTestResult:
    """Result of a jump significance test.

    Attributes
    ----------
    statistic : float
        Test statistic (asymptotically N(0,1) under H0: no jumps).
    p_value : float
        One-sided p-value (right tail = jump detected).
    jump_detected : bool
        Whether jump is detected at given significance level.
    rv : float
        Realized variance.
    bv : float
        Bipower variation.
    significance_level : float
        Significance level used.
    """
    statistic: float
    p_value: float
    jump_detected: bool
    rv: float
    bv: float
    significance_level: float


def bns_jump_test(
    intraday_returns: NDArray[np.float64],
    significance_level: float = 0.05,
    use_log_version: bool = True,
) -> JumpTestResult:
    """Barndorff-Nielsen & Shephard (2006) jump test.

    Tests H0: no jumps (QV = IV, so RV ≈ BV)
    vs    H1: jumps present (RV > BV significantly)

    The ratio-type statistic:
        z = (RV - BV) / sqrt(v * max(TPQ, QQQ))

    where v = (mu_1^{-4} + 2*mu_1^{-2} - 3) / n is the asymptotic variance
    factor under H0.

    Parameters
    ----------
    intraday_returns : array, shape (n,)
    significance_level : float
    use_log_version : bool
        If True, use the log version (better finite-sample properties).

    Returns
    -------
    JumpTestResult
    """
    r = np.ascontiguousarray(intraday_returns, dtype=np.float64)
    n = r.shape[0]

    rv = realized_variance(r)
    bv = bipower_variation(r)

    if rv < 1e-20:
        return JumpTestResult(
            statistic=0.0, p_value=1.0, jump_detected=False,
            rv=rv, bv=bv, significance_level=significance_level,
        )

    # Variance factor
    vn = (_MU1 ** (-4) + 2.0 * _MU1 ** (-2) - 3.0) / n

    # Use max of TPQ and QQQ for robustness
    tpq = _tripower_quarticity(r) / (_MU43 ** 3)
    qqq = _quadpower_quarticity(r) / (_MU1 ** 4)
    iq_est = max(tpq, qqq) / (n * n)  # Normalize to get IQ estimate

    if use_log_version:
        # Log version: z = sqrt(n) * (log(RV) - log(BV)) / sqrt(v * IQ / BV^2)
        if bv < 1e-20:
            z = 0.0
        else:
            z_var = vn * max(iq_est, 1e-20) / (bv * bv)
            z = (np.log(rv) - np.log(bv)) / np.sqrt(max(z_var, 1e-20))
    else:
        # Linear version
        z_var = vn * max(iq_est, 1e-20)
        z = (rv - bv) / np.sqrt(max(z_var, 1e-20))

    p_value = 1.0 - stats.norm.cdf(z)

    return JumpTestResult(
        statistic=z,
        p_value=p_value,
        jump_detected=p_value < significance_level,
        rv=rv,
        bv=bv,
        significance_level=significance_level,
    )


# ═══════════════════════════════════════════════════
# Jump / Continuous Variation Decomposition
# ═══════════════════════════════════════════════════

@dataclass
class JumpDecomposition:
    """Full decomposition of quadratic variation into continuous and jump parts.

    QV = C + J where:
    - C = continuous variation (integrated variance without jumps)
    - J = jump variation (sum of squared jumps)
    """
    rv: float
    continuous: float
    jump: float
    jump_detected: bool
    jump_test: Optional[JumpTestResult] = None


def jump_variation(
    intraday_returns: NDArray[np.float64],
    significance_level: float = 0.05,
    robust_measure: str = "BV",
) -> float:
    """Estimate jump variation: J_t = max(RV_t - C_t, 0).

    Parameters
    ----------
    intraday_returns : array, shape (n,)
    significance_level : float
        For the BNS test to threshold jump detection.
    robust_measure : str
        "BV" for bipower variation, "MedRV" for median RV.

    Returns
    -------
    float
        Jump variation estimate (non-negative by construction).
    """
    r = np.ascontiguousarray(intraday_returns, dtype=np.float64)
    rv = realized_variance(r)
    c = continuous_variation(r, robust_measure=robust_measure)
    # Threshold: only attribute to jumps if BNS test rejects
    test = bns_jump_test(r, significance_level=significance_level)
    if test.jump_detected:
        return max(rv - c, 0.0)
    return 0.0


def continuous_variation(
    intraday_returns: NDArray[np.float64],
    robust_measure: str = "BV",
) -> float:
    """Estimate continuous variation using a jump-robust measure.

    Parameters
    ----------
    intraday_returns : array, shape (n,)
    robust_measure : str
        "BV", "MedRV", or "MinRV".

    Returns
    -------
    float
    """
    r = np.ascontiguousarray(intraday_returns, dtype=np.float64)
    rv = realized_variance(r)
    if robust_measure == "BV":
        c = bipower_variation(r)
    elif robust_measure == "MedRV":
        c = median_rv(r)
    else:
        from volforecast.realized.measures import min_rv
        c = min_rv(r)
    # Truncate at RV to ensure C <= RV
    return min(c, rv)


def jump_decomposition(
    intraday_returns: NDArray[np.float64],
    significance_level: float = 0.05,
    robust_measure: str = "BV",
) -> JumpDecomposition:
    """Full C/J decomposition of quadratic variation.

    Parameters
    ----------
    intraday_returns : array, shape (n,)
    significance_level : float
    robust_measure : str

    Returns
    -------
    JumpDecomposition
    """
    r = np.ascontiguousarray(intraday_returns, dtype=np.float64)
    rv = realized_variance(r)
    test = bns_jump_test(r, significance_level=significance_level)

    c = continuous_variation(r, robust_measure=robust_measure)

    if test.jump_detected:
        j = max(rv - c, 0.0)
    else:
        j = 0.0
        c = rv  # No jumps detected: all variation is continuous

    return JumpDecomposition(
        rv=rv, continuous=c, jump=j,
        jump_detected=test.jump_detected, jump_test=test,
    )
