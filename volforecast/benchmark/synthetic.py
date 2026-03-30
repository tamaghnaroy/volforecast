"""
Synthetic data generation for benchmarking.

Data Generating Processes (DGPs):
1. GARCH(1,1) with known parameters — tests GARCH-family forecasters
2. Jump-diffusion (Merton, 1976) — tests jump detection and HAR-CJ
3. Stochastic volatility (Heston, 1993) — tests SV and realized-measure methods

All DGPs return both daily returns and intraday returns (for computing
realized measures), enabling end-to-end benchmarking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from numba import njit


@dataclass
class SyntheticData:
    """Container for synthetic benchmark data.

    Attributes
    ----------
    daily_returns : NDArray, shape (T,)
        Daily log returns.
    true_variance : NDArray, shape (T,)
        True conditional variance (known from DGP).
    intraday_returns : NDArray, shape (T, n_intraday)
        Intraday returns for computing realized measures.
    true_continuous : NDArray, shape (T,), optional
        True continuous variation (for jump DGPs).
    true_jumps : NDArray, shape (T,), optional
        True jump variation.
    dgp_name : str
        Name of the DGP.
    params : dict
        DGP parameters used.
    """
    daily_returns: NDArray[np.float64]
    true_variance: NDArray[np.float64]
    intraday_returns: NDArray[np.float64]
    true_continuous: Optional[NDArray[np.float64]] = None
    true_jumps: Optional[NDArray[np.float64]] = None
    dgp_name: str = ""
    params: Optional[dict] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


@njit(cache=True)
def _garch_dgp(
    T: int,
    n_intraday: int,
    omega: float,
    alpha: float,
    beta: float,
    seed: int,
) -> tuple:
    """Generate GARCH(1,1) data with intraday returns."""
    np.random.seed(seed)

    daily_returns = np.empty(T, dtype=np.float64)
    true_var = np.empty(T, dtype=np.float64)
    intraday = np.empty((T, n_intraday), dtype=np.float64)

    unc_var = omega / (1.0 - alpha - beta)
    true_var[0] = unc_var

    for t in range(T):
        if t > 0:
            true_var[t] = omega + alpha * daily_returns[t - 1] ** 2 + beta * true_var[t - 1]

        # Generate intraday returns: each has variance = true_var[t] / n_intraday
        intraday_var = true_var[t] / n_intraday
        daily_r = 0.0
        for i in range(n_intraday):
            r_i = np.sqrt(intraday_var) * np.random.randn()
            intraday[t, i] = r_i
            daily_r += r_i
        daily_returns[t] = daily_r

    return daily_returns, true_var, intraday


def generate_garch_data(
    T: int = 2000,
    n_intraday: int = 78,
    omega: float = 1e-6,
    alpha: float = 0.05,
    beta: float = 0.93,
    seed: int = 42,
) -> SyntheticData:
    """Generate synthetic data from GARCH(1,1) DGP.

    Parameters
    ----------
    T : int
        Number of daily observations.
    n_intraday : int
        Number of intraday intervals per day (78 = 5-min for 6.5hr day).
    omega, alpha, beta : float
        GARCH(1,1) parameters.
    seed : int
        Random seed.

    Returns
    -------
    SyntheticData
    """
    dr, tv, intra = _garch_dgp(T, n_intraday, omega, alpha, beta, seed)
    return SyntheticData(
        daily_returns=dr,
        true_variance=tv,
        intraday_returns=intra,
        dgp_name="GARCH(1,1)",
        params={"omega": omega, "alpha": alpha, "beta": beta,
                "T": T, "n_intraday": n_intraday, "seed": seed},
    )


@njit(cache=True)
def _jump_diffusion_dgp(
    T: int,
    n_intraday: int,
    mu: float,
    sigma_base: float,
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
    omega: float,
    alpha: float,
    beta: float,
    seed: int,
) -> tuple:
    """Generate jump-diffusion data with GARCH base volatility."""
    np.random.seed(seed)

    daily_returns = np.empty(T, dtype=np.float64)
    true_var = np.empty(T, dtype=np.float64)
    true_continuous = np.empty(T, dtype=np.float64)
    true_jumps = np.empty(T, dtype=np.float64)
    intraday = np.empty((T, n_intraday), dtype=np.float64)

    unc_var = omega / max(1.0 - alpha - beta, 1e-8)
    true_var[0] = unc_var

    dt = 1.0 / n_intraday

    for t in range(T):
        if t > 0:
            true_var[t] = omega + alpha * daily_returns[t - 1] ** 2 + beta * true_var[t - 1]

        cont_var = true_var[t]
        intraday_vol = np.sqrt(cont_var * dt)

        daily_r = 0.0
        jump_sum_sq = 0.0

        for i in range(n_intraday):
            # Continuous component
            r_cont = mu * dt + intraday_vol * np.random.randn()

            # Jump component (Poisson)
            jump = 0.0
            if np.random.random() < jump_intensity * dt:
                jump = jump_mean + jump_std * np.random.randn()
                jump_sum_sq += jump * jump

            intraday[t, i] = r_cont + jump
            daily_r += r_cont + jump

        daily_returns[t] = daily_r
        true_continuous[t] = cont_var
        true_jumps[t] = jump_sum_sq

    return daily_returns, true_var, intraday, true_continuous, true_jumps


def generate_jump_diffusion_data(
    T: int = 2000,
    n_intraday: int = 78,
    mu: float = 0.0,
    sigma_base: float = 0.01,
    jump_intensity: float = 0.1,
    jump_mean: float = 0.0,
    jump_std: float = 0.02,
    omega: float = 1e-6,
    alpha: float = 0.05,
    beta: float = 0.93,
    seed: int = 42,
) -> SyntheticData:
    """Generate synthetic data from jump-diffusion with GARCH base.

    Merton (1976)-style jumps with time-varying base volatility.

    Parameters
    ----------
    T : int
        Number of daily observations.
    n_intraday : int
        Intraday intervals per day.
    mu : float
        Drift.
    jump_intensity : float
        Expected number of jumps per day.
    jump_mean, jump_std : float
        Jump size distribution parameters.
    omega, alpha, beta : float
        GARCH parameters for base volatility.
    seed : int

    Returns
    -------
    SyntheticData
    """
    dr, tv, intra, tc, tj = _jump_diffusion_dgp(
        T, n_intraday, mu, sigma_base, jump_intensity,
        jump_mean, jump_std, omega, alpha, beta, seed,
    )
    return SyntheticData(
        daily_returns=dr,
        true_variance=tv,
        intraday_returns=intra,
        true_continuous=tc,
        true_jumps=tj,
        dgp_name="Jump-Diffusion",
        params={"T": T, "n_intraday": n_intraday, "mu": mu,
                "jump_intensity": jump_intensity, "jump_mean": jump_mean,
                "jump_std": jump_std, "omega": omega, "alpha": alpha,
                "beta": beta, "seed": seed},
    )


@njit(cache=True)
def _sv_dgp(
    T: int,
    n_intraday: int,
    mu: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
    seed: int,
) -> tuple:
    """Generate Heston (1993) stochastic volatility data.

    dS/S = mu dt + sqrt(V) dW_1
    dV   = kappa*(theta - V) dt + xi*sqrt(V) dW_2
    corr(dW_1, dW_2) = rho
    """
    np.random.seed(seed)

    daily_returns = np.empty(T, dtype=np.float64)
    true_var = np.empty(T, dtype=np.float64)
    intraday = np.empty((T, n_intraday), dtype=np.float64)

    dt = 1.0 / n_intraday
    sqrt_dt = np.sqrt(dt)

    V = v0

    for t in range(T):
        daily_r = 0.0
        iv_sum = 0.0  # Integrated variance for the day

        for i in range(n_intraday):
            # Correlated Brownian motions
            z1 = np.random.randn()
            z2 = rho * z1 + np.sqrt(1.0 - rho * rho) * np.random.randn()

            V_pos = max(V, 1e-10)
            sqrt_V = np.sqrt(V_pos)

            # Return
            r_i = mu * dt + sqrt_V * sqrt_dt * z1
            intraday[t, i] = r_i
            daily_r += r_i

            # Variance process (Euler-Maruyama with reflection)
            V = V + kappa * (theta - V_pos) * dt + xi * sqrt_V * sqrt_dt * z2
            V = max(V, 1e-10)

            iv_sum += V_pos * dt

        daily_returns[t] = daily_r
        true_var[t] = iv_sum  # Integrated variance over the day

    return daily_returns, true_var, intraday


def generate_sv_data(
    T: int = 2000,
    n_intraday: int = 78,
    mu: float = 0.0,
    kappa: float = 5.0,
    theta: float = 0.04,
    xi: float = 0.5,
    rho: float = -0.7,
    v0: float = 0.04,
    seed: int = 42,
) -> SyntheticData:
    """Generate synthetic data from Heston SV model.

    Parameters
    ----------
    T : int
    n_intraday : int
    mu : float
        Drift.
    kappa : float
        Mean-reversion speed.
    theta : float
        Long-run variance.
    xi : float
        Vol-of-vol.
    rho : float
        Leverage correlation (typically negative).
    v0 : float
        Initial variance.
    seed : int

    Returns
    -------
    SyntheticData
    """
    dr, tv, intra = _sv_dgp(T, n_intraday, mu, kappa, theta, xi, rho, v0, seed)
    return SyntheticData(
        daily_returns=dr,
        true_variance=tv,
        intraday_returns=intra,
        dgp_name="Heston SV",
        params={"T": T, "n_intraday": n_intraday, "mu": mu,
                "kappa": kappa, "theta": theta, "xi": xi,
                "rho": rho, "v0": v0, "seed": seed},
    )
