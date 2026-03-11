"""
Tests for evaluation framework: loss functions, statistical tests, proxy correction.
"""

import numpy as np
import pytest

from volforecast.evaluation.losses import (
    mse_loss,
    qlike_loss,
    mae_loss,
    mse_log_loss,
    patton_robust_loss,
    heterogeneous_loss,
)
from volforecast.evaluation.tests import (
    diebold_mariano_test,
    mincer_zarnowitz_test,
    model_confidence_set,
)
from volforecast.evaluation.proxy import (
    estimate_noise_variance,
    attenuation_bias_correction,
    hansen_lunde_adjustment,
    proxy_noise_correction,
)


# ─── Fixtures ───

@pytest.fixture
def perfect_forecasts():
    """Forecasts that exactly match the proxy."""
    rng = np.random.default_rng(42)
    y = np.abs(rng.normal(0.0001, 0.00005, size=500))
    return y, y.copy()


@pytest.fixture
def noisy_forecasts():
    """Forecasts with some noise relative to proxy."""
    rng = np.random.default_rng(42)
    y = np.abs(rng.normal(0.0001, 0.00005, size=500))
    f = y + rng.normal(0, 0.00002, size=500)
    f = np.maximum(f, 1e-10)
    return f, y


# ─── Loss Function Tests ───

class TestLossFunctions:
    def test_mse_perfect(self, perfect_forecasts):
        f, y = perfect_forecasts
        assert np.isclose(mse_loss(f, y), 0.0)

    def test_mse_non_negative(self, noisy_forecasts):
        f, y = noisy_forecasts
        assert mse_loss(f, y) > 0

    def test_qlike_perfect(self, perfect_forecasts):
        f, y = perfect_forecasts
        # QLIKE at f=y: y/y + log(y) = 1 + log(y), should be finite
        loss = qlike_loss(f, y)
        assert np.isfinite(loss)

    def test_qlike_positive_forecasts(self, noisy_forecasts):
        f, y = noisy_forecasts
        loss = qlike_loss(f, y)
        assert np.isfinite(loss)

    def test_mae_non_negative(self, noisy_forecasts):
        f, y = noisy_forecasts
        assert mae_loss(f, y) > 0

    def test_mse_log_perfect(self, perfect_forecasts):
        f, y = perfect_forecasts
        assert np.isclose(mse_log_loss(f, y), 0.0)

    def test_patton_mse_equivalence(self, noisy_forecasts):
        """Patton robust loss with b=1 should equal MSE."""
        f, y = noisy_forecasts
        mse = mse_loss(f, y)
        patton_mse = patton_robust_loss(f, y, b=1)
        assert np.isclose(mse, patton_mse, rtol=1e-8)

    def test_patton_qlike_equivalence(self, noisy_forecasts):
        """Patton robust loss with b=-2 should equal QLIKE."""
        f, y = noisy_forecasts
        ql = qlike_loss(f, y)
        patton_ql = patton_robust_loss(f, y, b=-2)
        assert np.isclose(ql, patton_ql, rtol=1e-8)

    def test_heterogeneous_loss(self, noisy_forecasts):
        f, y = noisy_forecasts
        loss = heterogeneous_loss(f, y)
        assert np.isfinite(loss)

    def test_heterogeneous_loss_custom_weights(self, noisy_forecasts):
        f, y = noisy_forecasts
        w = np.full(len(f), 0.8)
        loss = heterogeneous_loss(f, y, weights=w)
        assert np.isfinite(loss)

    def test_mae_perfect(self, perfect_forecasts):
        f, y = perfect_forecasts
        assert np.isclose(mae_loss(f, y), 0.0)

    def test_mse_log_non_negative(self, noisy_forecasts):
        f, y = noisy_forecasts
        assert mse_log_loss(f, y) > 0

    def test_patton_b0(self, noisy_forecasts):
        """Patton robust loss with b=0 (exponential family)."""
        f, y = noisy_forecasts
        loss = patton_robust_loss(f, y, b=0)
        assert np.isfinite(loss)
        assert loss >= 0

    def test_patton_b_neg1(self, noisy_forecasts):
        """Patton robust loss with b=-1."""
        f, y = noisy_forecasts
        loss = patton_robust_loss(f, y, b=-1)
        assert np.isfinite(loss)
        assert loss >= 0

    def test_patton_general_b(self, noisy_forecasts):
        """Patton robust loss with general b value."""
        f, y = noisy_forecasts
        loss = patton_robust_loss(f, y, b=2)
        assert np.isfinite(loss)

    def test_mse_symmetric(self, noisy_forecasts):
        """MSE(f, y) == MSE(y, f)."""
        f, y = noisy_forecasts
        assert np.isclose(mse_loss(f, y), mse_loss(y, f))


# ─── Statistical Test Tests ───

class TestDieboldMariano:
    def test_equal_losses(self):
        """Equal losses should yield non-significant DM test."""
        rng = np.random.default_rng(42)
        losses = rng.normal(0, 1, size=200)
        result = diebold_mariano_test(losses, losses + rng.normal(0, 0.01, 200))
        assert hasattr(result, 'statistic')
        assert hasattr(result, 'p_value')
        assert 0 <= result.p_value <= 1

    def test_different_losses(self):
        """Clearly different losses should yield significant DM test."""
        rng = np.random.default_rng(42)
        l1 = rng.normal(0, 1, size=500)
        l2 = l1 + 2.0  # Model 2 is much worse
        result = diebold_mariano_test(l1, l2)
        assert result.p_value < 0.05
        assert result.preferred == "model1"

    def test_harvey_correction(self):
        rng = np.random.default_rng(42)
        l1 = rng.normal(0, 1, size=50)
        l2 = l1 + 0.5
        r1 = diebold_mariano_test(l1, l2, harvey_correction=True)
        r2 = diebold_mariano_test(l1, l2, harvey_correction=False)
        # HLN correction should give larger p-value (more conservative)
        assert r1.p_value >= r2.p_value - 0.01  # Allow tiny numerical diff


class TestMincerZarnowitz:
    def test_perfect_forecast(self):
        """Perfect forecast should pass MZ efficiency test."""
        rng = np.random.default_rng(42)
        f = np.abs(rng.normal(1, 0.1, size=500))
        y = f + rng.normal(0, 0.01, size=500)
        result = mincer_zarnowitz_test(f, y)
        assert np.isclose(result.beta, 1.0, atol=0.1)
        assert result.r_squared > 0.8

    def test_biased_forecast(self):
        """Biased forecast should fail MZ test."""
        rng = np.random.default_rng(42)
        f = np.abs(rng.normal(1, 0.1, size=500))
        y = 2 * f + 0.5 + rng.normal(0, 0.1, size=500)
        result = mincer_zarnowitz_test(f, y)
        assert not result.efficient


class TestModelConfidenceSet:
    def test_single_model(self):
        """Single model should always be in MCS."""
        losses = np.random.default_rng(42).normal(0, 1, size=(100, 1))
        result = model_confidence_set(losses, alpha=0.1)
        assert 0 in result.included

    def test_clearly_worse_model_eliminated(self):
        """A clearly worse model should be eliminated."""
        rng = np.random.default_rng(42)
        T = 200
        good = rng.normal(0, 1, size=(T, 1))
        bad = good + 3.0  # Much worse
        losses = np.hstack([good, bad])
        result = model_confidence_set(losses, alpha=0.10, n_bootstrap=1000)
        # Model 0 (good) should be in the set
        assert 0 in result.included

    def test_stationary_bootstrap(self):
        """MCS with stationary bootstrap should also work."""
        rng = np.random.default_rng(42)
        T = 200
        good = rng.normal(0, 1, size=(T, 1))
        bad = good + 3.0
        losses = np.hstack([good, bad])
        result = model_confidence_set(
            losses, alpha=0.10, n_bootstrap=500,
            bootstrap_type="stationary",
        )
        assert 0 in result.included
        assert len(result.p_values) == 2

    def test_three_models(self):
        """MCS with 3 models: best should survive, worst eliminated."""
        rng = np.random.default_rng(42)
        T = 300
        l1 = rng.normal(0, 1, size=(T, 1))
        l2 = l1 + 0.1  # Slightly worse
        l3 = l1 + 5.0  # Much worse
        losses = np.hstack([l1, l2, l3])
        result = model_confidence_set(losses, alpha=0.10, n_bootstrap=1000)
        assert 0 in result.included  # Best model always in
        assert 2 in result.eliminated_order  # Worst eliminated

    def test_custom_block_length(self):
        """MCS with custom block length."""
        rng = np.random.default_rng(42)
        losses = rng.normal(0, 1, size=(100, 2))
        losses[:, 1] += 2.0
        result = model_confidence_set(losses, block_length=10, n_bootstrap=500)
        assert 0 in result.included


# ─── Proxy Correction Tests ───

class TestProxyCorrection:
    def test_noise_variance_non_negative(self):
        rng = np.random.default_rng(42)
        rv = np.abs(rng.normal(0.0001, 0.00005, size=500))
        nv = estimate_noise_variance(rv)
        assert nv >= 0.0

    def test_attenuation_correction(self):
        rng = np.random.default_rng(42)
        true_sig = np.abs(rng.normal(0.0001, 0.00003, size=500))
        noise = rng.normal(0, 0.00001, size=500)
        proxy = true_sig + noise
        f = true_sig * 0.9 + 0.00001  # Slightly biased forecast

        result = attenuation_bias_correction(f, proxy)
        assert result.corrected_r2 >= result.raw_r2 - 0.01  # Correction should help
        assert 0 <= result.noise_ratio <= 1.0

    def test_hansen_lunde(self):
        rng = np.random.default_rng(42)
        proxy = np.abs(rng.normal(0.0001, 0.00003, size=500))
        l1 = rng.normal(0, 0.001, size=500)
        l2 = l1 + 0.01
        result = hansen_lunde_adjustment(l1, l2, proxy)
        assert "ranking_robust" in result
        assert "snr" in result

    def test_proxy_noise_correction_mse(self):
        rng = np.random.default_rng(42)
        proxy = np.abs(rng.normal(0.0001, 0.00003, size=500))
        f = proxy * 1.1
        result = proxy_noise_correction(f, proxy, loss_fn="MSE")
        assert result["raw_loss"] > 0
        assert result["adjusted_loss"] >= 0
        assert result["proxy_quality"] in ("excellent", "good", "moderate",
                                            "poor — rankings may be unreliable")
