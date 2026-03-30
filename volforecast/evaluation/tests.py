"""
Statistical tests for volatility forecast comparison.

Implements:
- Diebold-Mariano test with HAC standard errors (Diebold & Mariano, 1995)
- Mincer-Zarnowitz efficiency regression (Mincer & Zarnowitz, 1969)
- Model Confidence Set (Hansen, Lunde, Nason, 2011)
- Hit rate / coverage test (Christoffersen, 1998)
- Dynamic Quantile (DQ) test (Engle & Manganelli, 2004)

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
- Christoffersen (1998). "Evaluating interval forecasts."
  International Economic Review 39(4), 841-862.
- Engle & Manganelli (2004). "CAViaR: Conditional autoregressive value at
  risk by regression quantiles." JBES 22(4), 367-381.
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
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        # Singular X'X — constant forecasts or degenerate input
        return MZTestResult(
            alpha=alpha, beta=beta,
            alpha_se=np.inf, beta_se=np.inf,
            r_squared=r_squared,
            f_stat=0.0, f_pvalue=1.0,
            efficient=False,
        )
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


# ═══════════════════════════════════════════════════
# Quantile Forecast Diagnostic Tests
# ═══════════════════════════════════════════════════

@dataclass
class HitRateTestResult:
    """Christoffersen (1998) coverage test result.

    Attributes
    ----------
    tau : float
        Nominal quantile level.
    hit_rate : float
        Empirical exceedance rate = mean(r_t < q_t).
    uc_statistic : float
        LR statistic for unconditional coverage H0: hit_rate == tau.
    uc_pvalue : float
        p-value for unconditional coverage test (chi2, df=1).
    ind_statistic : float
        LR statistic for serial independence of hits (first-order Markov).
    ind_pvalue : float
        p-value for independence test (chi2, df=1).
    cc_statistic : float
        LR statistic for conditional coverage (uc + ind, chi2, df=2).
    cc_pvalue : float
        p-value for conditional coverage test.
    n_hits : int
        Number of exceedances observed.
    n_obs : int
        Total number of observations.
    """
    tau: float
    hit_rate: float
    uc_statistic: float
    uc_pvalue: float
    ind_statistic: float
    ind_pvalue: float
    cc_statistic: float
    cc_pvalue: float
    n_hits: int
    n_obs: int


def hit_rate_test(
    returns: NDArray[np.float64],
    quantile_forecasts: NDArray[np.float64],
    tau: float = 0.05,
) -> HitRateTestResult:
    """Christoffersen (1998) unconditional and conditional coverage tests.

    Tests whether a sequence of quantile (VaR) forecasts has correct empirical
    coverage and serially independent hit indicators.

    - **Unconditional Coverage (UC):** H0: P(r_t < q_t) = tau
    - **Independence (Ind):** H0: Hit_t are serially uncorrelated (i.i.d.)
    - **Conditional Coverage (CC):** H0: UC and Ind hold jointly (df=2)

    Parameters
    ----------
    returns : array, shape (T,)
        Realized return series.
    quantile_forecasts : array, shape (T,)
        Conditional quantile forecast series at level *tau*.
        For VaR forecasts these are typically negative values.
    tau : float
        Nominal quantile level (e.g., 0.05 for 5% VaR).

    Returns
    -------
    HitRateTestResult

    References
    ----------
    Christoffersen (1998). "Evaluating interval forecasts."
    International Economic Review 39(4), 841-862.
    """
    r = np.asarray(returns, dtype=np.float64)
    q = np.asarray(quantile_forecasts, dtype=np.float64)
    T = len(r)

    hits = (r < q).astype(np.float64)
    n1 = int(np.sum(hits))
    n0 = T - n1
    p_hat = n1 / T

    _eps = 1e-15

    # --- Unconditional Coverage (Kupiec, 1995) ---
    if p_hat > _eps and p_hat < 1.0 - _eps:
        log_l0 = n1 * np.log(tau + _eps) + n0 * np.log(1.0 - tau + _eps)
        log_l1 = n1 * np.log(p_hat) + n0 * np.log(1.0 - p_hat)
        uc_stat = float(-2.0 * (log_l0 - log_l1))
    else:
        uc_stat = 0.0
    uc_stat = max(uc_stat, 0.0)
    uc_pval = float(stats.chi2.sf(uc_stat, df=1))

    # --- Independence: first-order Markov transition test ---
    n00 = float(np.sum((hits[:-1] == 0) & (hits[1:] == 0)))
    n01 = float(np.sum((hits[:-1] == 0) & (hits[1:] == 1)))
    n10 = float(np.sum((hits[:-1] == 1) & (hits[1:] == 0)))
    n11 = float(np.sum((hits[:-1] == 1) & (hits[1:] == 1)))

    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p_joint = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    log_l_ind0 = (
        (n00 + n10) * np.log(max(1.0 - p_joint, _eps))
        + (n01 + n11) * np.log(max(p_joint, _eps))
    )
    log_l_ind1 = (
        n00 * np.log(max(1.0 - p01, _eps))
        + n01 * np.log(max(p01, _eps))
        + n10 * np.log(max(1.0 - p11, _eps))
        + n11 * np.log(max(p11, _eps))
    )
    ind_stat = float(-2.0 * (log_l_ind0 - log_l_ind1))
    ind_stat = max(ind_stat, 0.0)
    ind_pval = float(stats.chi2.sf(ind_stat, df=1))

    # --- Conditional Coverage: uc + ind ---
    cc_stat = uc_stat + ind_stat
    cc_pval = float(stats.chi2.sf(cc_stat, df=2))

    return HitRateTestResult(
        tau=tau,
        hit_rate=p_hat,
        uc_statistic=uc_stat,
        uc_pvalue=uc_pval,
        ind_statistic=ind_stat,
        ind_pvalue=ind_pval,
        cc_statistic=cc_stat,
        cc_pvalue=cc_pval,
        n_hits=n1,
        n_obs=T,
    )


@dataclass
class DQTestResult:
    """Dynamic Quantile (DQ) test result (Engle & Manganelli, 2004).

    Attributes
    ----------
    statistic : float
        DQ chi-squared test statistic.
    p_value : float
        p-value; small values reject H0 that hits are unpredictable.
    n_lags : int
        Number of lagged Hit terms used as instruments.
    df : int
        Degrees of freedom = n_lags + 2 (constant + lags + q_t^2).
    """
    statistic: float
    p_value: float
    n_lags: int
    df: int


def dq_test(
    returns: NDArray[np.float64],
    quantile_forecasts: NDArray[np.float64],
    tau: float = 0.05,
    n_lags: int = 4,
) -> DQTestResult:
    """Dynamic Quantile (DQ) test for quantile forecast calibration.

    Tests H0: the centered hit series Hit_t = I(r_t < q_t) - tau is
    orthogonal to {1, Hit_{t-1}, ..., Hit_{t-K}, q_t^2}.

    A well-calibrated quantile forecast should produce hits that are
    unpredictable from past hits and from the current forecast level.
    The test statistic is:

        DQ = (X'Hit)' (X'X)^{-1} (X'Hit) / [tau*(1-tau)]  ~  chi2(K+2)

    where X = [1, Hit_{t-1}, ..., Hit_{t-K}, q_t^2].

    Parameters
    ----------
    returns : array, shape (T,)
        Realized return series.
    quantile_forecasts : array, shape (T,)
        Conditional quantile forecast series at level *tau*.
    tau : float
        Nominal quantile level (default 0.05).
    n_lags : int
        Number of lagged Hit_t terms in the instrument matrix (default 4).

    Returns
    -------
    DQTestResult

    References
    ----------
    Engle & Manganelli (2004). "CAViaR: Conditional autoregressive value at
    risk by regression quantiles." JBES 22(4), 367-381.
    """
    r = np.asarray(returns, dtype=np.float64)
    q = np.asarray(quantile_forecasts, dtype=np.float64)
    T = len(r)

    hit = (r < q).astype(np.float64) - tau

    start = n_lags
    T_eff = T - start
    df = n_lags + 2

    # Build instrument matrix X: [constant, Hit_{t-1..K}, q_t^2]
    X = np.ones((T_eff, df), dtype=np.float64)
    for k in range(n_lags):
        X[:, k + 1] = hit[start - k - 1 : T - k - 1]
    X[:, n_lags + 1] = q[start:] ** 2

    y = hit[start:]

    try:
        XtX = X.T @ X
        Xty = X.T @ y
        beta = np.linalg.solve(XtX, Xty)
        dq_stat = float(Xty @ beta / (tau * (1.0 - tau)))
    except np.linalg.LinAlgError:
        dq_stat = 0.0

    dq_stat = max(dq_stat, 0.0)
    p_value = float(stats.chi2.sf(dq_stat, df=df))

    return DQTestResult(
        statistic=dq_stat,
        p_value=p_value,
        n_lags=n_lags,
        df=df,
    )
