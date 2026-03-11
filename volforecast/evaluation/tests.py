"""
Statistical tests for volatility forecast comparison.

Implements:
- Diebold-Mariano test with HAC standard errors (Diebold & Mariano, 1995)
- Mincer-Zarnowitz efficiency regression (Mincer & Zarnowitz, 1969)
- Model Confidence Set (Hansen, Lunde, Nason, 2011)

References
----------
- Diebold & Mariano (1995). "Comparing predictive accuracy."
  Journal of Business & Economic Statistics 13(3), 253-263.
- Harvey, Leybourne, Newbold (1997). "Testing the equality of prediction
  mean squared errors." IJOF 13(2), 281-291.
- Mincer & Zarnowitz (1969). "The Evaluation of Economic Forecasts."
  In Economic Forecasts and Expectations.
- Hansen, Lunde, Nason (2011). "The Model Confidence Set."
  Econometrica 79(2), 453-497.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass
class DMTestResult:
    """Diebold-Mariano test result.

    Attributes
    ----------
    statistic : float
        DM test statistic (asymptotically N(0,1)).
    p_value : float
        Two-sided p-value.
    mean_loss_diff : float
        Mean loss differential (negative = model 1 better).
    hac_variance : float
        HAC-estimated variance of the loss differential.
    preferred : str
        "model1", "model2", or "neither" at given significance.
    """
    statistic: float
    p_value: float
    mean_loss_diff: float
    hac_variance: float
    preferred: str


def _newey_west_variance(x: NDArray[np.float64], max_lag: int) -> float:
    """Newey-West HAC variance estimator.

    Parameters
    ----------
    x : array, shape (T,)
        Time series (e.g., loss differentials).
    max_lag : int
        Maximum lag for autocovariance truncation.

    Returns
    -------
    float
        HAC variance estimate.
    """
    T = len(x)
    x_demeaned = x - np.mean(x)

    # Gamma_0
    gamma0 = float(np.dot(x_demeaned, x_demeaned)) / T

    # Sum weighted autocovariances
    hac_var = gamma0
    for h in range(1, max_lag + 1):
        gamma_h = float(np.dot(x_demeaned[h:], x_demeaned[:-h])) / T
        weight = 1.0 - h / (max_lag + 1.0)  # Bartlett kernel
        hac_var += 2.0 * weight * gamma_h

    return max(hac_var, 1e-20)


def diebold_mariano_test(
    losses1: NDArray[np.float64],
    losses2: NDArray[np.float64],
    horizon: int = 1,
    significance: float = 0.05,
    harvey_correction: bool = True,
) -> DMTestResult:
    """Diebold-Mariano test for equal predictive accuracy.

    Tests H0: E[L1_t - L2_t] = 0  (equal forecast accuracy)
    vs    H1: E[L1_t - L2_t] != 0

    Uses Newey-West HAC standard errors for h-step-ahead forecasts
    and the Harvey, Leybourne, Newbold (1997) small-sample correction.

    Parameters
    ----------
    losses1, losses2 : array, shape (T,)
        Per-period loss series for models 1 and 2.
    horizon : int
        Forecast horizon (determines HAC lag truncation).
    significance : float
        Significance level.
    harvey_correction : bool
        Apply HLN small-sample correction (recommended).

    Returns
    -------
    DMTestResult
    """
    d = np.asarray(losses1, dtype=np.float64) - np.asarray(losses2, dtype=np.float64)
    T = len(d)
    d_bar = float(np.mean(d))

    # HAC variance with lag = h-1 (for h-step-ahead)
    max_lag = max(horizon - 1, 0)
    hac_var = _newey_west_variance(d, max_lag)

    # DM statistic
    dm_stat = d_bar / np.sqrt(hac_var / T)

    # Harvey, Leybourne, Newbold correction
    if harvey_correction and T > 1:
        hlnc = np.sqrt(
            (T + 1.0 - 2.0 * horizon + horizon * (horizon - 1.0) / T) / T
        )
        dm_stat_adj = dm_stat * hlnc
        # Use t-distribution with T-1 df
        p_value = 2.0 * stats.t.sf(abs(dm_stat_adj), df=T - 1)
        stat_final = dm_stat_adj
    else:
        p_value = 2.0 * stats.norm.sf(abs(dm_stat))
        stat_final = dm_stat

    if p_value < significance:
        preferred = "model1" if d_bar < 0 else "model2"
    else:
        preferred = "neither"

    return DMTestResult(
        statistic=stat_final,
        p_value=p_value,
        mean_loss_diff=d_bar,
        hac_variance=hac_var,
        preferred=preferred,
    )


@dataclass
class MZTestResult:
    """Mincer-Zarnowitz regression result.

    Tests forecast efficiency: y_t = alpha + beta * f_t + e_t
    H0: alpha = 0, beta = 1 (efficient forecast).
    """
    alpha: float
    beta: float
    alpha_se: float
    beta_se: float
    r_squared: float
    f_stat: float
    f_pvalue: float
    efficient: bool


def mincer_zarnowitz_test(
    forecasts: NDArray[np.float64],
    proxies: NDArray[np.float64],
    significance: float = 0.05,
) -> MZTestResult:
    """Mincer-Zarnowitz efficiency regression.

    Regresses the proxy on the forecast:
      y_t = alpha + beta * f_t + e_t

    An efficient forecast has alpha=0, beta=1 (unbiased and unit slope).
    Joint F-test for H0: (alpha, beta) = (0, 1).

    Parameters
    ----------
    forecasts : array, shape (T,)
    proxies : array, shape (T,)
    significance : float

    Returns
    -------
    MZTestResult
    """
    f = np.asarray(forecasts, dtype=np.float64)
    y = np.asarray(proxies, dtype=np.float64)
    T = len(f)

    # OLS: y = alpha + beta * f
    X = np.column_stack([np.ones(T), f])
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha, beta = beta_hat[0], beta_hat[1]

    # Residuals and R^2
    y_hat = X @ beta_hat
    resid = y - y_hat
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(y - np.mean(y), y - np.mean(y)))
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-20)

    # Standard errors (OLS)
    sigma2 = ss_res / max(T - 2, 1)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    alpha_se, beta_se = se[0], se[1]

    # Joint F-test: H0: (alpha, beta) = (0, 1)
    R = np.eye(2)  # Restriction matrix
    r = np.array([0.0, 1.0])  # Restriction values
    diff = beta_hat - r
    Rb = R @ beta_hat - r
    V = sigma2 * XtX_inv
    try:
        f_stat = float(Rb.T @ np.linalg.inv(R @ V @ R.T) @ Rb) / 2.0
    except np.linalg.LinAlgError:
        f_stat = 0.0
    f_pvalue = 1.0 - stats.f.cdf(f_stat, dfn=2, dfd=max(T - 2, 1))

    efficient = f_pvalue > significance

    return MZTestResult(
        alpha=alpha, beta=beta,
        alpha_se=alpha_se, beta_se=beta_se,
        r_squared=r_squared,
        f_stat=f_stat, f_pvalue=f_pvalue,
        efficient=efficient,
    )


@dataclass
class MCSResult:
    """Model Confidence Set result.

    Attributes
    ----------
    included : list[int]
        Indices of models in the confidence set.
    p_values : NDArray[np.float64]
        p-values for each model (models with p > alpha are included).
    eliminated_order : list[int]
        Order in which models were eliminated.
    """
    included: list[int]
    p_values: NDArray[np.float64]
    eliminated_order: list[int]


def _block_bootstrap_indices(
    rng: np.random.Generator,
    T: int,
    block_length: int,
) -> NDArray[np.int64]:
    """Generate moving-block bootstrap indices."""
    n_blocks = int(np.ceil(T / block_length))
    block_starts = rng.integers(0, T - block_length + 1, size=n_blocks)
    indices = []
    for start in block_starts:
        indices.extend(range(start, min(start + block_length, T)))
    return np.array(indices[:T], dtype=np.int64)


def _stationary_bootstrap_indices(
    rng: np.random.Generator,
    T: int,
    expected_block_length: float,
) -> NDArray[np.int64]:
    """Generate stationary bootstrap indices (Politis & Romano, 1994).

    Block lengths are geometric(1/expected_block_length), giving random
    block lengths that better preserve dependence structure.
    """
    p = 1.0 / max(expected_block_length, 1.0)
    indices = np.empty(T, dtype=np.int64)
    indices[0] = rng.integers(0, T)
    for t in range(1, T):
        if rng.random() < p:
            # Start a new block
            indices[t] = rng.integers(0, T)
        else:
            # Continue current block (with wrap-around)
            indices[t] = (indices[t - 1] + 1) % T
    return indices


def model_confidence_set(
    loss_matrix: NDArray[np.float64],
    alpha: float = 0.10,
    n_bootstrap: int = 5000,
    block_length: Optional[int] = None,
    bootstrap_type: str = "moving_block",
    seed: int = 42,
) -> MCSResult:
    """Model Confidence Set (Hansen, Lunde, Nason, 2011).

    Iteratively eliminates the worst model until no model is significantly
    worse than others, yielding a set of models with equal predictive ability.

    Uses the T_max statistic with bootstrap resampling.

    Parameters
    ----------
    loss_matrix : array, shape (T, M)
        Loss values for T periods and M models.
    alpha : float
        Significance level.
    n_bootstrap : int
        Number of bootstrap replications.
    block_length : int, optional
        Block length for bootstrap. Default: T^{1/3} (Hall, Horowitz, Jing 1995).
    bootstrap_type : str
        "moving_block" (default) or "stationary" (Politis & Romano, 1994).
        Stationary bootstrap uses geometric random block lengths and may
        better preserve dependence structure in volatility loss series.
    seed : int
        Random seed.

    Returns
    -------
    MCSResult
    """
    rng = np.random.default_rng(seed)
    T, M = loss_matrix.shape

    if block_length is None:
        block_length = max(1, int(np.ceil(T ** (1.0 / 3.0))))

    alive = list(range(M))
    eliminated_order: list[int] = []
    p_values = np.zeros(M, dtype=np.float64)

    while len(alive) > 1:
        m = len(alive)
        L = loss_matrix[:, alive]  # (T, m)

        # Compute pairwise loss differentials
        d_bar = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            for j in range(m):
                d_bar[i, j] = np.mean(L[:, i] - L[:, j])

        # T_max statistic: max over pairs of |d_bar_ij| / se(d_bar_ij)
        t_stats = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            for j in range(i + 1, m):
                d_ij = L[:, i] - L[:, j]
                var_d = _newey_west_variance(d_ij, max(1, block_length))
                se = np.sqrt(var_d / T)
                if se > 1e-20:
                    t_stats[i, j] = abs(d_bar[i, j]) / se
                    t_stats[j, i] = t_stats[i, j]

        t_max = np.max(t_stats)

        # Bootstrap for T_max distribution
        boot_t_max = np.zeros(n_bootstrap, dtype=np.float64)

        for b in range(n_bootstrap):
            if bootstrap_type == "stationary":
                indices = _stationary_bootstrap_indices(rng, T, float(block_length))
            else:
                indices = _block_bootstrap_indices(rng, T, block_length)

            L_boot = L[indices, :]
            L_boot_centered = L_boot - np.mean(L, axis=0, keepdims=True)

            boot_t = 0.0
            for i in range(m):
                for j in range(i + 1, m):
                    d_boot = L_boot_centered[:, i] - L_boot_centered[:, j]
                    d_boot_mean = np.mean(d_boot)
                    var_boot = _newey_west_variance(d_boot, max(1, block_length))
                    se_boot = np.sqrt(max(var_boot, 1e-20) / T)
                    t_boot = abs(d_boot_mean) / max(se_boot, 1e-20)
                    boot_t = max(boot_t, t_boot)

            boot_t_max[b] = boot_t

        # p-value for T_max
        p_val = float(np.mean(boot_t_max >= t_max))

        if p_val < alpha:
            # Eliminate the worst model (highest average loss)
            avg_losses = np.mean(L, axis=0)
            worst_idx = int(np.argmax(avg_losses))
            worst_model = alive[worst_idx]
            eliminated_order.append(worst_model)
            p_values[worst_model] = p_val
            alive.pop(worst_idx)
        else:
            # Cannot reject: remaining models form the MCS
            break

    # Assign p-values to surviving models
    for model_idx in alive:
        p_values[model_idx] = 1.0  # Not eliminated

    return MCSResult(
        included=alive,
        p_values=p_values,
        eliminated_order=eliminated_order,
    )
