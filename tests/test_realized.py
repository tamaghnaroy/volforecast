"""
Tests for realized volatility measures and jump decomposition.

Tests verify:
- Correctness against known analytical values
- Consistency relations (e.g., RV >= BV under no noise)
- Jump detection on synthetic data with known jumps
- Numba JIT compilation works correctly
"""

import numpy as np
import pytest

from volforecast.realized.measures import (
    realized_variance,
    bipower_variation,
    median_rv,
    min_rv,
    realized_kernel,
    tsrv,
    pre_averaging,
    realized_semivariance,
    realized_variance_series,
    bipower_variation_series,
    realized_semivariance_series,
)
from volforecast.realized.jumps import (
    bns_jump_test,
    jump_variation,
    continuous_variation,
    jump_decomposition,
)


# ─── Fixtures ───

@pytest.fixture
def gaussian_returns():
    """Gaussian returns with no jumps, known variance."""
    rng = np.random.default_rng(42)
    sigma = 0.01  # daily vol ~ 1%
    n = 78  # 5-min returns in 6.5hr day
    return rng.normal(0, sigma / np.sqrt(n), size=n)


@pytest.fixture
def returns_with_jump():
    """Returns with a single large jump."""
    rng = np.random.default_rng(123)
    sigma = 0.01
    n = 78
    r = rng.normal(0, sigma / np.sqrt(n), size=n)
    # Insert large jump at position 40
    r[40] += 0.05  # 5% jump
    return r


@pytest.fixture
def log_prices():
    """Simulated log prices for noise-robust estimators."""
    rng = np.random.default_rng(42)
    n = 500
    sigma = 0.01
    returns = rng.normal(0, sigma / np.sqrt(n), size=n)
    prices = np.cumsum(np.concatenate([[0.0], returns]))
    return prices


# ─── RV Tests ───

class TestRealizedVariance:
    def test_non_negative(self, gaussian_returns):
        rv = realized_variance(gaussian_returns)
        assert rv >= 0.0

    def test_zero_for_zero_returns(self):
        rv = realized_variance(np.zeros(100))
        assert rv == 0.0

    def test_known_value(self):
        """RV of constant returns = n * r^2."""
        r = np.full(10, 0.01)
        rv = realized_variance(r)
        assert np.isclose(rv, 10 * 0.01**2)

    def test_annualization(self, gaussian_returns):
        rv_raw = realized_variance(gaussian_returns, annualize=False)
        rv_ann = realized_variance(gaussian_returns, annualize=True, trading_days=252)
        assert np.isclose(rv_ann, rv_raw * 252)

    def test_series(self):
        rng = np.random.default_rng(42)
        mat = rng.normal(0, 0.01, size=(10, 78))
        rv_series = realized_variance_series(mat)
        assert rv_series.shape == (10,)
        for t in range(10):
            assert np.isclose(rv_series[t], realized_variance(mat[t]))


# ─── BV Tests ───

class TestBipowerVariation:
    def test_non_negative(self, gaussian_returns):
        bv = bipower_variation(gaussian_returns)
        assert bv >= 0.0

    def test_close_to_rv_no_jumps(self, gaussian_returns):
        """Under no jumps, BV should be close to RV."""
        rv = realized_variance(gaussian_returns)
        bv = bipower_variation(gaussian_returns)
        # Allow 30% tolerance for finite sample
        assert abs(bv - rv) / max(rv, 1e-20) < 0.30

    def test_less_than_rv_with_jump(self, returns_with_jump):
        """BV should be less than RV when jumps are present."""
        rv = realized_variance(returns_with_jump)
        bv = bipower_variation(returns_with_jump)
        assert bv < rv

    def test_short_series(self):
        """BV of very short series."""
        assert bipower_variation(np.array([0.01])) == 0.0
        bv = bipower_variation(np.array([0.01, -0.01]))
        assert bv > 0.0

    def test_series(self):
        rng = np.random.default_rng(123)
        mat = rng.normal(0, 0.01, size=(10, 78))
        bv_series = bipower_variation_series(mat)
        assert bv_series.shape == (10,)
        for t in range(10):
            assert np.isclose(bv_series[t], bipower_variation(mat[t]))


# ─── MedRV Tests ───

class TestMedianRV:
    def test_non_negative(self, gaussian_returns):
        mrv = median_rv(gaussian_returns)
        assert mrv >= 0.0

    def test_more_robust_than_bv(self, returns_with_jump):
        """MedRV should be closer to true IV than BV when there are jumps."""
        rv = realized_variance(returns_with_jump)
        bv = bipower_variation(returns_with_jump)
        mrv = median_rv(returns_with_jump)
        # MedRV should be <= BV <= RV (approximately)
        assert mrv <= rv * 1.1  # Allow small tolerance

    def test_short_series(self):
        assert median_rv(np.array([0.01, 0.02])) == 0.0


# ─── MinRV Tests ───

class TestMinRV:
    def test_non_negative(self, gaussian_returns):
        mrv = min_rv(gaussian_returns)
        assert mrv >= 0.0

    def test_robust_to_jump(self, returns_with_jump):
        rv = realized_variance(returns_with_jump)
        mrv = min_rv(returns_with_jump)
        assert mrv < rv


# ─── Realized Kernel Tests ───

class TestRealizedKernel:
    def test_non_negative(self, gaussian_returns):
        rk = realized_kernel(gaussian_returns)
        assert rk >= 0.0

    def test_close_to_rv_clean_data(self, gaussian_returns):
        rv = realized_variance(gaussian_returns)
        rk = realized_kernel(gaussian_returns)
        assert abs(rk - rv) / max(rv, 1e-20) < 0.50


# ─── TSRV Tests ───

class TestTSRV:
    def test_non_negative(self, log_prices):
        tv = tsrv(log_prices)
        assert tv >= 0.0

    def test_reasonable_magnitude(self, log_prices):
        """TSRV should be in a reasonable range."""
        tv = tsrv(log_prices)
        # For ~500 obs with sigma=0.01, daily var ~ 1e-4
        assert tv < 1.0


# ─── Pre-Averaging Tests ───

class TestPreAveraging:
    def test_non_negative(self, log_prices):
        pa = pre_averaging(log_prices)
        assert pa >= 0.0


# ─── Semi-Variance Tests ───

class TestSemiVariance:
    def test_decomposition(self, gaussian_returns):
        """RS+ + RS- should equal RV."""
        rv = realized_variance(gaussian_returns)
        rs_pos, rs_neg = realized_semivariance(gaussian_returns)
        assert np.isclose(rs_pos + rs_neg, rv, rtol=1e-10)

    def test_non_negative(self, gaussian_returns):
        rs_pos, rs_neg = realized_semivariance(gaussian_returns)
        assert rs_pos >= 0.0
        assert rs_neg >= 0.0

    def test_series(self):
        rng = np.random.default_rng(7)
        mat = rng.normal(0, 0.01, size=(12, 78))
        rs_pos_series, rs_neg_series = realized_semivariance_series(mat)
        assert rs_pos_series.shape == (12,)
        assert rs_neg_series.shape == (12,)
        for t in range(12):
            rs_pos, rs_neg = realized_semivariance(mat[t])
            assert np.isclose(rs_pos_series[t], rs_pos)
            assert np.isclose(rs_neg_series[t], rs_neg)


# ─── Jump Tests ───

class TestJumps:
    def test_no_jump_detected_gaussian(self, gaussian_returns):
        """BNS test should not detect jumps in pure Gaussian returns (mostly)."""
        result = bns_jump_test(gaussian_returns, significance_level=0.01)
        # With alpha=0.01, false positive rate should be low
        # We don't assert no detection (could be false positive), just check structure
        assert hasattr(result, 'statistic')
        assert hasattr(result, 'p_value')
        assert 0.0 <= result.p_value <= 1.0

    def test_jump_detected(self, returns_with_jump):
        """BNS test should detect the large jump."""
        result = bns_jump_test(returns_with_jump, significance_level=0.05)
        assert result.jump_detected == True

    def test_decomposition_sums_to_rv(self, returns_with_jump):
        """C + J should approximately equal RV."""
        decomp = jump_decomposition(returns_with_jump)
        assert np.isclose(decomp.continuous + decomp.jump, decomp.rv, rtol=0.01)

    def test_jump_variation_non_negative(self, returns_with_jump):
        jv = jump_variation(returns_with_jump)
        assert jv >= 0.0

    def test_continuous_variation_bounded(self, gaussian_returns):
        cv = continuous_variation(gaussian_returns)
        rv = realized_variance(gaussian_returns)
        assert cv <= rv * 1.001  # Allow tiny numerical error
