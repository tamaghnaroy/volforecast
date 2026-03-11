"""
Proxy-target mismatch correction.

When the true volatility sigma^2_t is latent, we evaluate forecasts against
a noisy proxy hat{sigma}^2_t (e.g., realized variance). This module implements
corrections for:

1. Attenuation bias in Mincer-Zarnowitz regressions (Hansen & Lunde, 2006)
2. Noise-corrected loss function evaluation
3. Hansen-Lunde adjustment for proxy quality

Key insight (Hansen & Lunde, 2006):
  If hat{sigma}^2 = sigma^2 + eta where eta is noise:
  - The R^2 in MZ regression is downward biased
  - The slope coefficient suffers attenuation bias
  - Model rankings can be REVERSED under non-robust losses

References
----------
- Hansen & Lunde (2006). "Consistent ranking of volatility models."
  Journal of Econometrics 131(1-2), 97-121.
- Patton (2011). "Volatility forecast comparison using imperfect
  volatility proxies." Journal of Econometrics 160(1), 246-256.
- Bollerslev, Patton, Quaedvlieg (2016). "Exploiting the errors:
  A simple approach for improved volatility forecasting."
  Journal of Econometrics 192(1), 1-18.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class ProxyCorrectionResult:
    """Result of proxy noise correction.

    Attributes
    ----------
    corrected_r2 : float
        Noise-corrected R^2 from MZ regression.
    raw_r2 : float
        Uncorrected R^2.
    noise_ratio : float
        Estimated ratio var(eta) / var(sigma^2).
    attenuation_factor : float
        Estimated attenuation in slope coefficient.
    """
    corrected_r2: float
    raw_r2: float
    noise_ratio: float
    attenuation_factor: float


def estimate_noise_variance(
    rv: NDArray[np.float64],
    method: str = "rv_ac1",
) -> float:
    """Estimate the variance of measurement noise in realized variance.

    Methods:
    - "rv_ac1": Uses first-order autocorrelation of RV to separate
      signal from noise (Andersen, Bollerslev, Meddahi, 2011).
    - "bv_diff": Uses BV - RV difference to estimate noise
      (requires bipower variation).

    Parameters
    ----------
    rv : array, shape (T,)
        Realized variance series.
    method : str
        Estimation method.

    Returns
    -------
    float
        Estimated noise variance var(eta).
    """
    rv = np.asarray(rv, dtype=np.float64)

    if method == "rv_ac1":
        # var(RV) = var(sigma^2) + var(eta)
        # cov(RV_t, RV_{t-1}) ≈ cov(sigma^2_t, sigma^2_{t-1})
        # (noise is serially uncorrelated)
        # So: var(eta) ≈ var(RV) - var(RV) * rho_1(RV) / rho_1(sigma^2)
        # Approximation: var(eta) ≈ var(RV) * (1 - rho_1(RV))
        # This is conservative (upper bound)
        T = len(rv)
        if T < 3:
            return 0.0
        rv_demean = rv - np.mean(rv)
        var_rv = float(np.var(rv))
        cov_1 = float(np.dot(rv_demean[1:], rv_demean[:-1])) / (T - 1)
        rho_1 = cov_1 / max(var_rv, 1e-20)
        # Noise variance estimate
        noise_var = max(var_rv * (1.0 - rho_1), 0.0)
        return noise_var

    elif method == "bv_diff":
        # This is a placeholder; proper implementation needs BV input
        # var(eta) ≈ 2 * var(RV - BV)
        return 0.0

    return 0.0


def attenuation_bias_correction(
    forecasts: NDArray[np.float64],
    proxies: NDArray[np.float64],
    noise_variance: Optional[float] = None,
) -> ProxyCorrectionResult:
    """Correct for attenuation bias in forecast evaluation.

    When the proxy hat{sigma}^2 = sigma^2 + eta:
    - MZ slope beta_hat = beta_true * var(sigma^2) / var(hat{sigma}^2)
    - R^2_hat = R^2_true * var(sigma^2) / var(hat{sigma}^2)

    Parameters
    ----------
    forecasts : array, shape (T,)
    proxies : array, shape (T,)
    noise_variance : float, optional
        If None, estimated from data.

    Returns
    -------
    ProxyCorrectionResult
    """
    f = np.asarray(forecasts, dtype=np.float64)
    y = np.asarray(proxies, dtype=np.float64)
    T = len(f)

    if noise_variance is None:
        noise_variance = estimate_noise_variance(y)

    var_proxy = float(np.var(y))
    var_signal = max(var_proxy - noise_variance, 1e-20)

    # Attenuation factor: var(signal) / var(proxy)
    attenuation = var_signal / max(var_proxy, 1e-20)

    # Raw MZ regression
    X = np.column_stack([np.ones(T), f])
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta_hat
    resid = y - y_hat
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(y - np.mean(y), y - np.mean(y)))
    raw_r2 = 1.0 - ss_res / max(ss_tot, 1e-20)

    # Corrected R^2: R^2_corrected = R^2_raw / attenuation
    corrected_r2 = min(raw_r2 / max(attenuation, 1e-8), 1.0)

    noise_ratio = noise_variance / max(var_proxy, 1e-20)

    return ProxyCorrectionResult(
        corrected_r2=corrected_r2,
        raw_r2=raw_r2,
        noise_ratio=noise_ratio,
        attenuation_factor=attenuation,
    )


def hansen_lunde_adjustment(
    losses1: NDArray[np.float64],
    losses2: NDArray[np.float64],
    proxies: NDArray[np.float64],
    noise_variance: Optional[float] = None,
) -> dict:
    """Hansen-Lunde (2006) adjustment for proxy quality in model comparison.

    Checks whether the ranking of two models could be reversed with a
    better proxy. Uses the signal-to-noise ratio of the proxy.

    Parameters
    ----------
    losses1, losses2 : array, shape (T,)
        Per-period losses under models 1 and 2.
    proxies : array, shape (T,)
        Volatility proxy used.
    noise_variance : float, optional

    Returns
    -------
    dict with keys:
        - "ranking_robust": bool, whether ranking is likely robust to proxy noise
        - "snr": signal-to-noise ratio of the proxy
        - "loss_diff_mean": mean loss differential
        - "critical_snr": minimum SNR needed for reliable ranking
    """
    l1 = np.asarray(losses1, dtype=np.float64)
    l2 = np.asarray(losses2, dtype=np.float64)
    y = np.asarray(proxies, dtype=np.float64)

    if noise_variance is None:
        noise_variance = estimate_noise_variance(y)

    var_proxy = float(np.var(y))
    var_signal = max(var_proxy - noise_variance, 1e-20)
    snr = var_signal / max(noise_variance, 1e-20)

    d = l1 - l2
    d_bar = float(np.mean(d))
    d_var = float(np.var(d))

    # Rough heuristic: ranking is robust if SNR > |d_bar| / sqrt(d_var / T)
    T = len(d)
    se_d = np.sqrt(d_var / T) if T > 0 else 1e10
    critical_snr = abs(d_bar) / max(se_d, 1e-20)

    return {
        "ranking_robust": snr > 2.0 * critical_snr,
        "snr": snr,
        "loss_diff_mean": d_bar,
        "critical_snr": critical_snr,
    }


def proxy_noise_correction(
    forecasts: NDArray[np.float64],
    proxies: NDArray[np.float64],
    loss_fn: str = "MSE",
    noise_variance: Optional[float] = None,
) -> dict:
    """Comprehensive proxy noise correction for forecast evaluation.

    Computes:
    1. Raw loss
    2. Noise-adjusted loss (for MSE: subtracts noise variance)
    3. Confidence interval for true loss accounting for proxy noise
    4. Recommendation on proxy quality

    Parameters
    ----------
    forecasts, proxies : array, shape (T,)
    loss_fn : str
        "MSE" or "QLIKE" (both robust per Patton 2011).
    noise_variance : float, optional

    Returns
    -------
    dict
    """
    f = np.asarray(forecasts, dtype=np.float64)
    y = np.asarray(proxies, dtype=np.float64)
    T = len(f)

    if noise_variance is None:
        noise_variance = estimate_noise_variance(y)

    if loss_fn == "MSE":
        raw_loss = float(np.mean((f - y) ** 2))
        # E[(f - hat_sigma^2)^2] = E[(f - sigma^2)^2] + var(eta) + cross terms
        # For unbiased forecasts: E[(f - y)^2] ≈ E[(f - sigma^2)^2] + var(eta)
        adjusted_loss = max(raw_loss - noise_variance, 0.0)
    elif loss_fn == "QLIKE":
        f_safe = np.maximum(f, 1e-20)
        raw_loss = float(np.mean(y / f_safe + np.log(f_safe)))
        # QLIKE is robust: ranking preserved, but level may differ
        adjusted_loss = raw_loss  # No simple adjustment for QLIKE level
    else:
        raw_loss = float(np.mean((f - y) ** 2))
        adjusted_loss = raw_loss

    var_proxy = float(np.var(y))
    snr = max(var_proxy - noise_variance, 0.0) / max(noise_variance, 1e-20)

    if snr > 10:
        quality = "excellent"
    elif snr > 3:
        quality = "good"
    elif snr > 1:
        quality = "moderate"
    else:
        quality = "poor — rankings may be unreliable"

    return {
        "raw_loss": raw_loss,
        "adjusted_loss": adjusted_loss,
        "noise_variance": noise_variance,
        "signal_to_noise_ratio": snr,
        "proxy_quality": quality,
        "loss_fn": loss_fn,
        "T": T,
    }
