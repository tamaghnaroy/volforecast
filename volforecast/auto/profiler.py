"""
Data profiler for AutoVolForecaster (Phase 1).

Characterises the input series so that the candidate selector can make
data-driven model-family decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class DataProfile:
    """Result of profiling the input series."""
    T: int
    has_intraday: bool
    has_realized: bool
    hurst_exp: float
    has_long_memory: bool        # H > 0.6
    has_rough_vol: bool          # H < 0.5
    has_leverage: bool
    jump_fraction: float
    has_jumps: bool              # jump_fraction > 0.05
    excess_kurtosis: float
    heavy_tails: bool            # excess_kurtosis > 5
    has_regime_switching: bool
    rv: Optional[NDArray[np.float64]] = None
    bv: Optional[NDArray[np.float64]] = None
    jv: Optional[NDArray[np.float64]] = None
    cv: Optional[NDArray[np.float64]] = None


class DataProfiler:
    """Profile a return series for automated model selection."""

    @staticmethod
    def profile(
        returns: NDArray[np.float64],
        intraday_returns: Optional[NDArray[np.float64]] = None,
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
    ) -> DataProfile:
        """Characterise the input series.

        Parameters
        ----------
        returns : array, shape (T,)
            Daily log-returns.
        intraday_returns : array, shape (T, n_intraday), optional
        realized_measures : dict, optional
            Pre-computed realized measures (must contain at least "RV").

        Returns
        -------
        DataProfile
        """
        returns = np.asarray(returns, dtype=np.float64)
        T = len(returns)

        has_intraday = intraday_returns is not None
        has_realized = realized_measures is not None

        # --- Hurst exponent (R/S method) ---
        hurst_exp = DataProfiler._hurst_rs(returns)
        has_long_memory = hurst_exp > 0.6
        has_rough_vol = hurst_exp < 0.5

        # --- Leverage test: sign correlation of r_t and r_{t+1}^2 ---
        has_leverage = DataProfiler._test_leverage(returns)

        # --- Jump detection ---
        rv, bv, jv, cv = None, None, None, None
        jump_fraction = 0.0
        if has_intraday:
            rv, bv, jv, cv, jump_fraction = DataProfiler._compute_realized_and_jumps(
                intraday_returns
            )
        elif has_realized and "RV" in realized_measures:
            rv = realized_measures["RV"]
            bv = realized_measures.get("BV")
            jv = realized_measures.get("JV")
            cv = realized_measures.get("CV")
            if bv is not None and rv is not None:
                jump_fraction = float(np.mean(
                    np.maximum(rv - bv, 0.0) > 0.01 * rv
                ))
        has_jumps = jump_fraction > 0.05

        # --- Tail heaviness ---
        excess_kurtosis = float(DataProfiler._excess_kurtosis(returns))
        heavy_tails = excess_kurtosis > 5.0

        # --- Regime-switching indicator ---
        has_regime_switching = DataProfiler._test_regime_switching(returns)

        return DataProfile(
            T=T,
            has_intraday=has_intraday,
            has_realized=has_realized,
            hurst_exp=hurst_exp,
            has_long_memory=has_long_memory,
            has_rough_vol=has_rough_vol,
            has_leverage=has_leverage,
            jump_fraction=jump_fraction,
            has_jumps=has_jumps,
            excess_kurtosis=excess_kurtosis,
            heavy_tails=heavy_tails,
            has_regime_switching=has_regime_switching,
            rv=rv,
            bv=bv,
            jv=jv,
            cv=cv,
        )

    # ─── Private helpers ──────────────────────────────────────────────

    @staticmethod
    def _hurst_rs(x: NDArray[np.float64]) -> float:
        """Estimate Hurst exponent via rescaled range (R/S) on |returns|."""
        y = np.abs(x)
        T = len(y)
        if T < 20:
            return 0.5  # indeterminate

        max_k = min(T // 2, 512)
        ns = []
        rs = []
        for n in [16, 32, 64, 128, 256, 512]:
            if n > max_k:
                break
            n_blocks = T // n
            if n_blocks < 1:
                continue
            rs_vals = []
            for i in range(n_blocks):
                block = y[i * n:(i + 1) * n]
                m = np.mean(block)
                s = np.std(block, ddof=1)
                if s < 1e-15:
                    continue
                cumdev = np.cumsum(block - m)
                r = np.max(cumdev) - np.min(cumdev)
                rs_vals.append(r / s)
            if rs_vals:
                ns.append(n)
                rs.append(np.mean(rs_vals))

        if len(ns) < 2:
            return 0.5

        log_n = np.log(ns)
        log_rs = np.log(rs)
        # OLS: log(R/S) = H * log(n) + c
        A = np.column_stack([log_n, np.ones(len(log_n))])
        coeffs = np.linalg.lstsq(A, log_rs, rcond=None)[0]
        H = float(np.clip(coeffs[0], 0.01, 0.99))
        return H

    @staticmethod
    def _test_leverage(returns: NDArray[np.float64]) -> bool:
        """Test for leverage effect: negative correlation between r_t and r_{t+1}^2."""
        if len(returns) < 10:
            return False
        r = returns[:-1]
        r2_next = returns[1:] ** 2
        corr = np.corrcoef(r, r2_next)[0, 1]
        return bool(corr < -0.05)

    @staticmethod
    def _excess_kurtosis(returns: NDArray[np.float64]) -> float:
        """Compute excess kurtosis."""
        if len(returns) < 4:
            return 0.0
        m = np.mean(returns)
        s = np.std(returns, ddof=1)
        if s < 1e-15:
            return 0.0
        z = (returns - m) / s
        return float(np.mean(z ** 4) - 3.0)

    @staticmethod
    def _test_regime_switching(returns: NDArray[np.float64]) -> bool:
        """Simple regime-switching test based on ACF structure of squared returns."""
        r2 = returns ** 2
        T = len(r2)
        if T < 50:
            return False
        m = np.mean(r2)
        r2c = r2 - m
        var = float(np.dot(r2c, r2c)) / T

        if var < 1e-20:
            return False

        # Check if ACF at lag 10-20 is still substantial (slow decay = regime)
        acf_long = 0.0
        count = 0
        for lag in range(10, min(21, T)):
            c = float(np.dot(r2c[lag:], r2c[:-lag])) / T
            acf_long += c / var
            count += 1
        if count == 0:
            return False
        avg_acf_long = acf_long / count

        # ACF at lag 1
        acf1 = float(np.dot(r2c[1:], r2c[:-1])) / T / var

        # Regime switching = slow decay (long ACF still > 50% of lag-1 ACF)
        if acf1 < 0.05:
            return False
        return bool(avg_acf_long > 0.5 * acf1)

    @staticmethod
    def _compute_realized_and_jumps(
        intraday_returns: NDArray[np.float64],
    ) -> tuple:
        """Compute realized measures and jump fraction from intraday data."""
        from volforecast.realized.measures import (
            realized_variance,
            bipower_variation,
        )
        T = intraday_returns.shape[0]
        rv = np.array([realized_variance(intraday_returns[t]) for t in range(T)])
        bv = np.array([bipower_variation(intraday_returns[t]) for t in range(T)])
        jv = np.maximum(rv - bv, 0.0)
        cv = np.minimum(bv, rv)
        # Jump fraction: share of days where JV > 1% of RV
        jump_fraction = float(np.mean(jv > 0.01 * rv))
        return rv, bv, jv, cv, jump_fraction
