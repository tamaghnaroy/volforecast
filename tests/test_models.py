"""
Tests for GARCH and HAR model families.

Tests verify:
- Models fit without error on synthetic data
- Forecasts are positive and finite
- Online update extends state correctly
- Model specs are well-formed
- Numba filters produce correct results on known inputs
"""

import numpy as np
import pytest

from volforecast.models.garch import (
    GARCHForecaster,
    GJRGARCHForecaster,
    EGARCHForecaster,
    CGARCHForecaster,
    APARCHForecaster,
    ARCHForecaster,
    EWMAForecaster,
    garch11_filter,
    gjr_garch11_filter,
    egarch11_filter,
    cgarch11_filter,
    arch_q_filter,
    ewma_filter,
)
from volforecast.models.har import (
    HARForecaster,
    HARJForecaster,
    HARCJForecaster,
    SHARForecaster,
)
from volforecast.models.realized_garch import RealizedGARCHForecaster
from volforecast.core.targets import VolatilityTarget
from volforecast.benchmark.synthetic import generate_garch_data


# ─── Fixtures ───

@pytest.fixture(scope="module")
def garch_data():
    """Synthetic GARCH(1,1) data for testing."""
    return generate_garch_data(T=800, n_intraday=78, seed=42)


@pytest.fixture(scope="module")
def rv_series(garch_data):
    """Pre-computed RV series from synthetic data."""
    from volforecast.realized.measures import realized_variance
    return np.array([realized_variance(garch_data.intraday_returns[t])
                     for t in range(len(garch_data.daily_returns))])


@pytest.fixture(scope="module")
def bv_series(garch_data):
    """Pre-computed BV series."""
    from volforecast.realized.measures import bipower_variation
    return np.array([bipower_variation(garch_data.intraday_returns[t])
                     for t in range(len(garch_data.daily_returns))])


# ─── Numba Filter Tests ───

class TestNumbaFilters:
    def test_garch11_filter_shape(self):
        r = np.random.default_rng(42).normal(0, 0.01, size=100)
        sig2 = garch11_filter(r, 1e-6, 0.05, 0.93)
        assert sig2.shape == (100,)
        assert np.all(sig2 > 0)

    def test_garch11_filter_known(self):
        """Verify filter against manual calculation."""
        r = np.array([0.01, -0.02, 0.005], dtype=np.float64)
        omega, alpha, beta = 1e-6, 0.1, 0.85
        sig2 = garch11_filter(r, omega, alpha, beta)
        unc = omega / (1 - alpha - beta)
        assert np.isclose(sig2[0], unc)
        expected_1 = omega + alpha * r[0]**2 + beta * unc
        assert np.isclose(sig2[1], expected_1)

    def test_gjr_filter_leverage(self):
        """Negative returns should produce higher variance than positive."""
        r_neg = np.array([0.01, -0.02, 0.01], dtype=np.float64)
        r_pos = np.array([0.01, 0.02, 0.01], dtype=np.float64)
        sig2_neg = gjr_garch11_filter(r_neg, 1e-6, 0.05, 0.05, 0.90)
        sig2_pos = gjr_garch11_filter(r_pos, 1e-6, 0.05, 0.05, 0.90)
        assert sig2_neg[2] > sig2_pos[2]

    def test_egarch_filter_positive(self):
        r = np.random.default_rng(42).normal(0, 0.01, size=100)
        sig2 = egarch11_filter(r, -0.1, 0.1, -0.05, 0.98)
        assert np.all(sig2 > 0)
        assert np.all(np.isfinite(sig2))

    def test_cgarch_filter_positive(self):
        r = np.random.default_rng(42).normal(0, 0.01, size=100)
        sig2 = cgarch11_filter(r, 1e-6, 0.05, 0.90, 0.02, 0.99)
        assert np.all(sig2 > 0)


# ─── GARCH Forecaster Tests ───

class TestGARCHForecaster:
    def test_fit_predict(self, garch_data):
        model = GARCHForecaster(dist="normal")
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=1)
        assert result.point.shape == (1,)
        assert result.point[0] > 0
        assert result.target_spec.target == VolatilityTarget.CONDITIONAL_VARIANCE

    def test_multi_step_forecast(self, garch_data):
        model = GARCHForecaster()
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=10)
        assert result.point.shape == (10,)
        assert np.all(result.point > 0)
        # Forecasts should converge toward unconditional variance
        params = model.get_params()
        unc_var = params["omega"] / (1 - params["alpha"] - params["beta"])
        assert abs(result.point[-1] - unc_var) < abs(result.point[0] - unc_var) + 1e-12

    def test_online_update(self, garch_data):
        model = GARCHForecaster()
        model.fit(garch_data.daily_returns[:500])
        n_before = len(model._returns)
        model.update(garch_data.daily_returns[500:502])
        assert len(model._returns) == n_before + 2

    def test_model_spec(self):
        model = GARCHForecaster()
        spec = model.model_spec
        assert spec.family == "GARCH"
        assert spec.abbreviation == "GARCH"

    def test_predict_before_fit_raises(self):
        model = GARCHForecaster()
        with pytest.raises(RuntimeError):
            model.predict()


class TestGJRGARCHForecaster:
    def test_fit_predict(self, garch_data):
        model = GJRGARCHForecaster()
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=5)
        assert result.point.shape == (5,)
        assert np.all(result.point > 0)


class TestEGARCHForecaster:
    def test_fit_predict(self, garch_data):
        model = EGARCHForecaster()
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=3)
        assert result.point.shape == (3,)
        assert np.all(result.point > 0)
        assert np.all(np.isfinite(result.point))


class TestCGARCHForecaster:
    def test_fit_predict(self, garch_data):
        model = CGARCHForecaster()
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=5)
        assert result.point.shape == (5,)
        assert np.all(result.point > 0)


# ─── HAR Forecaster Tests ───

class TestHARForecaster:
    def test_fit_predict(self, garch_data, rv_series):
        model = HARForecaster()
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500]})
        result = model.predict(horizon=1)
        assert result.point.shape == (1,)
        assert result.point[0] > 0
        assert result.target_spec.target == VolatilityTarget.INTEGRATED_VARIANCE

    def test_log_transform(self, garch_data, rv_series):
        model = HARForecaster(log_transform=True)
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500]})
        result = model.predict(horizon=5)
        assert result.point.shape == (5,)
        assert np.all(result.point > 0)

    def test_multi_step(self, garch_data, rv_series):
        model = HARForecaster()
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500]})
        result = model.predict(horizon=22)
        assert result.point.shape == (22,)

    def test_requires_rv(self, garch_data):
        model = HARForecaster()
        with pytest.raises(ValueError):
            model.fit(garch_data.daily_returns[:500])

    def test_online_update(self, garch_data, rv_series):
        model = HARForecaster()
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500]})
        n_before = len(model._rv)
        model.update(garch_data.daily_returns[500:505],
                     new_realized={"RV": rv_series[500:505]})
        assert len(model._rv) == n_before + 5


class TestHARJForecaster:
    def test_fit_predict(self, garch_data, rv_series, bv_series):
        model = HARJForecaster()
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500], "BV": bv_series[:500]})
        result = model.predict(horizon=1)
        assert result.point[0] > 0


class TestHARCJForecaster:
    def test_fit_predict(self, garch_data, rv_series, bv_series):
        model = HARCJForecaster()
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500], "BV": bv_series[:500]})
        result = model.predict(horizon=5)
        assert result.point.shape == (5,)
        assert np.all(result.point > 0)


class TestSHARForecaster:
    def test_fit_predict(self, garch_data, rv_series):
        from volforecast.realized.measures import realized_semivariance
        rs = [realized_semivariance(garch_data.intraday_returns[t])
              for t in range(500)]
        rs_pos = np.array([r[0] for r in rs])
        rs_neg = np.array([r[1] for r in rs])
        model = SHARForecaster()
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500],
                                     "RS_pos": rs_pos, "RS_neg": rs_neg})
        result = model.predict(horizon=1)
        assert result.point[0] > 0


# ─── Realized GARCH Tests ───

class TestRealizedGARCH:
    def test_fit_predict(self, garch_data, rv_series):
        model = RealizedGARCHForecaster()
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500]})
        result = model.predict(horizon=5)
        assert result.point.shape == (5,)
        assert np.all(result.point > 0)
        assert np.all(np.isfinite(result.point))

    def test_online_update(self, garch_data, rv_series):
        model = RealizedGARCHForecaster()
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500]})
        n_before = len(model._returns)
        model.update(garch_data.daily_returns[500:503],
                     new_realized={"RV": rv_series[500:503]})
        assert len(model._returns) == n_before + 3


# ─── APARCH Extended Tests ───

class TestAPARCHForecaster:
    def test_fit_predict(self, garch_data):
        model = APARCHForecaster()
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=1)
        assert result.point.shape == (1,)
        assert result.point[0] > 0

    def test_multi_step(self, garch_data):
        model = APARCHForecaster()
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=5)
        assert result.point.shape == (5,)
        assert np.all(result.point > 0)
        assert np.all(np.isfinite(result.point))

    def test_online_update(self, garch_data):
        model = APARCHForecaster()
        model.fit(garch_data.daily_returns[:500])
        n_before = len(model._returns)
        model.update(garch_data.daily_returns[500:505])
        assert len(model._returns) == n_before + 5

    def test_get_params(self, garch_data):
        model = APARCHForecaster()
        model.fit(garch_data.daily_returns[:500])
        params = model.get_params()
        assert len(params) > 0

    def test_predict_before_fit_raises(self):
        model = APARCHForecaster()
        with pytest.raises(RuntimeError):
            model.predict()

    def test_update_before_fit_raises(self):
        model = APARCHForecaster()
        with pytest.raises(RuntimeError):
            model.update(np.array([0.01]))

    def test_model_spec(self):
        model = APARCHForecaster()
        spec = model.model_spec
        assert spec.abbreviation == "APARCH"
        assert spec.family == "GARCH"


# ─── CGARCH Extended Tests ───

class TestCGARCHExtended:
    def test_multi_step(self, garch_data):
        model = CGARCHForecaster()
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=10)
        assert result.point.shape == (10,)
        assert np.all(result.point > 0)
        assert np.all(np.isfinite(result.point))

    def test_online_update(self, garch_data):
        model = CGARCHForecaster()
        model.fit(garch_data.daily_returns[:500])
        n_before = len(model._returns)
        model.update(garch_data.daily_returns[500:505])
        assert len(model._returns) == n_before + 5
        assert len(model._sigma2) == n_before + 5

    def test_get_params(self, garch_data):
        model = CGARCHForecaster()
        model.fit(garch_data.daily_returns[:500])
        params = model.get_params()
        assert "omega" in params
        assert "alpha" in params
        assert "beta" in params
        assert "rho" in params

    def test_predict_before_fit_raises(self):
        model = CGARCHForecaster()
        with pytest.raises(RuntimeError):
            model.predict()

    def test_update_before_fit_raises(self):
        model = CGARCHForecaster()
        with pytest.raises(RuntimeError):
            model.update(np.array([0.01]))


# ─── GJR-GARCH Extended Tests ───

class TestGJRExtended:
    def test_multi_step(self, garch_data):
        model = GJRGARCHForecaster()
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=10)
        assert result.point.shape == (10,)
        assert np.all(result.point > 0)

    def test_online_update(self, garch_data):
        model = GJRGARCHForecaster()
        model.fit(garch_data.daily_returns[:500])
        n_before = len(model._returns)
        model.update(garch_data.daily_returns[500:503])
        assert len(model._returns) == n_before + 3

    def test_predict_before_fit_raises(self):
        model = GJRGARCHForecaster()
        with pytest.raises(RuntimeError):
            model.predict()

    def test_update_before_fit_raises(self):
        model = GJRGARCHForecaster()
        with pytest.raises(RuntimeError):
            model.update(np.array([0.01]))

    def test_get_params(self, garch_data):
        model = GJRGARCHForecaster()
        model.fit(garch_data.daily_returns[:500])
        params = model.get_params()
        assert "gamma" in params


# ─── EGARCH Extended Tests ───

class TestEGARCHExtended:
    def test_online_update(self, garch_data):
        model = EGARCHForecaster()
        model.fit(garch_data.daily_returns[:500])
        n_before = len(model._returns)
        model.update(garch_data.daily_returns[500:505])
        assert len(model._returns) == n_before + 5

    def test_predict_before_fit_raises(self):
        model = EGARCHForecaster()
        with pytest.raises(RuntimeError):
            model.predict()

    def test_update_before_fit_raises(self):
        model = EGARCHForecaster()
        with pytest.raises(RuntimeError):
            model.update(np.array([0.01]))


# ─── HAR-CJ Extended Tests ───

class TestHARCJExtended:
    def test_multi_step(self, garch_data, rv_series, bv_series):
        model = HARCJForecaster()
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500], "BV": bv_series[:500]})
        result = model.predict(horizon=10)
        assert result.point.shape == (10,)
        assert np.all(result.point > 0)

    def test_online_update(self, garch_data, rv_series, bv_series):
        model = HARCJForecaster()
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500], "BV": bv_series[:500]})
        n_before = len(model._rv)
        cv = np.minimum(bv_series[500:505], rv_series[500:505])
        jv = np.maximum(rv_series[500:505] - bv_series[500:505], 0.0)
        model.update(garch_data.daily_returns[500:505],
                     new_realized={"RV": rv_series[500:505], "CV": cv, "JV": jv})
        assert len(model._rv) == n_before + 5

    def test_predict_before_fit_raises(self):
        model = HARCJForecaster()
        with pytest.raises(RuntimeError):
            model.predict()

    def test_requires_rv(self, garch_data):
        model = HARCJForecaster()
        with pytest.raises(ValueError):
            model.fit(garch_data.daily_returns[:500])

    def test_requires_bv_or_cv(self, garch_data, rv_series):
        model = HARCJForecaster()
        with pytest.raises(ValueError):
            model.fit(garch_data.daily_returns[:500],
                      realized_measures={"RV": rv_series[:500]})

    def test_get_params(self, garch_data, rv_series, bv_series):
        model = HARCJForecaster()
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500], "BV": bv_series[:500]})
        params = model.get_params()
        assert "intercept" in params
        assert "beta_c_lag1" in params
        assert "beta_j_lag1" in params


# ─── SHAR Extended Tests ───

class TestSHARExtended:
    def test_multi_step(self, garch_data, rv_series):
        from volforecast.realized.measures import realized_semivariance
        rs = [realized_semivariance(garch_data.intraday_returns[t])
              for t in range(500)]
        rs_pos = np.array([r[0] for r in rs])
        rs_neg = np.array([r[1] for r in rs])
        model = SHARForecaster()
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500],
                                     "RS_pos": rs_pos, "RS_neg": rs_neg})
        result = model.predict(horizon=5)
        assert result.point.shape == (5,)
        assert np.all(result.point > 0)

    def test_online_update(self, garch_data, rv_series):
        from volforecast.realized.measures import realized_semivariance
        rs = [realized_semivariance(garch_data.intraday_returns[t])
              for t in range(505)]
        rs_pos = np.array([r[0] for r in rs])
        rs_neg = np.array([r[1] for r in rs])
        model = SHARForecaster()
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500],
                                     "RS_pos": rs_pos[:500], "RS_neg": rs_neg[:500]})
        n_before = len(model._rv)
        model.update(garch_data.daily_returns[500:505],
                     new_realized={"RV": rv_series[500:505],
                                   "RS_pos": rs_pos[500:505],
                                   "RS_neg": rs_neg[500:505]})
        assert len(model._rv) == n_before + 5

    def test_predict_before_fit_raises(self):
        model = SHARForecaster()
        with pytest.raises(RuntimeError):
            model.predict()

    def test_requires_realized(self, garch_data):
        model = SHARForecaster()
        with pytest.raises(ValueError):
            model.fit(garch_data.daily_returns[:500])

    def test_get_params(self, garch_data, rv_series):
        from volforecast.realized.measures import realized_semivariance
        rs = [realized_semivariance(garch_data.intraday_returns[t])
              for t in range(500)]
        rs_pos = np.array([r[0] for r in rs])
        rs_neg = np.array([r[1] for r in rs])
        model = SHARForecaster()
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500],
                                     "RS_pos": rs_pos, "RS_neg": rs_neg})
        params = model.get_params()
        assert "intercept" in params
        assert "beta_rs_pos" in params
        assert "beta_rs_neg" in params


# ─── HAR-J Extended Tests ───

class TestHARJExtended:
    def test_multi_step(self, garch_data, rv_series, bv_series):
        model = HARJForecaster()
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500], "BV": bv_series[:500]})
        result = model.predict(horizon=5)
        assert result.point.shape == (5,)
        assert np.all(result.point > 0)

    def test_online_update(self, garch_data, rv_series, bv_series):
        model = HARJForecaster()
        jv = np.maximum(rv_series - bv_series, 0.0)
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500], "BV": bv_series[:500]})
        n_before = len(model._rv)
        model.update(garch_data.daily_returns[500:505],
                     new_realized={"RV": rv_series[500:505], "JV": jv[500:505]})
        assert len(model._rv) == n_before + 5

    def test_requires_rv(self, garch_data):
        model = HARJForecaster()
        with pytest.raises(ValueError):
            model.fit(garch_data.daily_returns[:500])

    def test_get_params(self, garch_data, rv_series, bv_series):
        model = HARJForecaster()
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500], "BV": bv_series[:500]})
        params = model.get_params()
        assert "intercept" in params
        assert "beta_jump" in params


# ─── Realized GARCH Extended Tests ───

class TestRealizedGARCHExtended:
    def test_predict_before_fit_raises(self):
        model = RealizedGARCHForecaster()
        with pytest.raises(RuntimeError):
            model.predict()

    def test_update_before_fit_raises(self):
        model = RealizedGARCHForecaster()
        with pytest.raises(RuntimeError):
            model.update(np.array([0.01]), new_realized={"RV": np.array([0.0001])})

    def test_requires_rv(self, garch_data):
        model = RealizedGARCHForecaster()
        with pytest.raises(ValueError):
            model.fit(garch_data.daily_returns[:500])

    def test_fit_with_empty_inputs_raises(self):
        model = RealizedGARCHForecaster()
        with pytest.raises(ValueError, match="at least one observation"):
            model.fit(np.array([]), realized_measures={"RV": np.array([])})

    def test_update_mismatched_lengths_raises(self, garch_data, rv_series):
        model = RealizedGARCHForecaster()
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500]})
        with pytest.raises(ValueError, match="same length"):
            model.update(garch_data.daily_returns[500:503],
                         new_realized={"RV": rv_series[500:502]})


    def test_get_params(self, garch_data, rv_series):
        model = RealizedGARCHForecaster()
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500]})
        params = model.get_params()
        assert "omega" in params
        assert "beta" in params


# ─── ARCH(q) / EWMA Numba Filter Tests ───

class TestARCHQFilter:
    def test_arch1_filter_shape(self):
        r = np.random.default_rng(42).normal(0, 0.01, size=100)
        alphas = np.array([0.3], dtype=np.float64)
        sig2 = arch_q_filter(r, 1e-6, alphas)
        assert sig2.shape == (100,)
        assert np.all(sig2 > 0)

    def test_arch1_filter_known(self):
        """ARCH(1) filter should match manual calculation."""
        r = np.array([0.01, -0.02, 0.005], dtype=np.float64)
        omega = 1e-6
        alphas = np.array([0.3], dtype=np.float64)
        sig2 = arch_q_filter(r, omega, alphas)
        unc = omega / (1.0 - 0.3)
        assert np.isclose(sig2[0], omega + 0.3 * unc)
        expected_1 = omega + 0.3 * r[0] ** 2
        assert np.isclose(sig2[1], expected_1)

    def test_arch5_filter_shape(self):
        r = np.random.default_rng(42).normal(0, 0.01, size=200)
        alphas = np.array([0.1, 0.08, 0.06, 0.04, 0.02], dtype=np.float64)
        sig2 = arch_q_filter(r, 1e-6, alphas)
        assert sig2.shape == (200,)
        assert np.all(sig2 > 0)
        assert np.all(np.isfinite(sig2))


class TestEWMAFilter:
    def test_ewma_filter_shape(self):
        r = np.random.default_rng(42).normal(0, 0.01, size=100)
        sig2 = ewma_filter(r, 0.94)
        assert sig2.shape == (100,)
        assert np.all(sig2 > 0)

    def test_ewma_filter_known(self):
        """Verify EWMA recursion against manual calculation."""
        r = np.array([0.01, -0.02, 0.005, 0.003], dtype=np.float64)
        lam = 0.94
        sig2 = ewma_filter(r, lam)
        sample_var = np.mean(r ** 2)
        assert np.isclose(sig2[0], sample_var)
        expected_1 = lam * sample_var + (1.0 - lam) * r[0] ** 2
        assert np.isclose(sig2[1], expected_1)
        expected_2 = lam * expected_1 + (1.0 - lam) * r[1] ** 2
        assert np.isclose(sig2[2], expected_2)

    def test_ewma_high_lambda_smooth(self):
        """Higher lambda should produce smoother variance series."""
        r = np.random.default_rng(42).normal(0, 0.01, size=200)
        sig2_low = ewma_filter(r, 0.80)
        sig2_high = ewma_filter(r, 0.99)
        # Higher lambda => lower variance of the variance series
        assert np.std(sig2_high[10:]) < np.std(sig2_low[10:])


# ─── ARCHForecaster Tests ───

class TestARCHForecaster:
    def test_fit_predict(self, garch_data):
        model = ARCHForecaster(q=1)
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=1)
        assert result.point.shape == (1,)
        assert result.point[0] > 0
        assert result.target_spec.target == VolatilityTarget.CONDITIONAL_VARIANCE

    def test_multi_step_forecast(self, garch_data):
        model = ARCHForecaster(q=1)
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=10)
        assert result.point.shape == (10,)
        assert np.all(result.point > 0)
        assert np.all(np.isfinite(result.point))
        # ARCH multi-step forecasts should converge toward unconditional variance
        params = model.get_params()
        alpha_sum = sum(v for k, v in params.items() if k.startswith("alpha"))
        unc_var = params["omega"] / max(1.0 - alpha_sum, 1e-8)
        assert abs(result.point[-1] - unc_var) < abs(result.point[0] - unc_var) + 1e-10

    def test_higher_order(self, garch_data):
        model = ARCHForecaster(q=3)
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=5)
        assert result.point.shape == (5,)
        assert np.all(result.point > 0)
        params = model.get_params()
        assert "alpha[1]" in params
        assert "alpha[2]" in params
        assert "alpha[3]" in params

    def test_online_update(self, garch_data):
        model = ARCHForecaster(q=1)
        model.fit(garch_data.daily_returns[:500])
        n_before = len(model._returns)
        model.update(garch_data.daily_returns[500:503])
        assert len(model._returns) == n_before + 3
        assert len(model._sigma2) == n_before + 3

    def test_model_spec(self):
        model = ARCHForecaster(q=5)
        spec = model.model_spec
        assert spec.abbreviation == "ARCH"
        assert spec.family == "GARCH"
        assert spec.name == "ARCH(5)"

    def test_predict_before_fit_raises(self):
        model = ARCHForecaster()
        with pytest.raises(RuntimeError):
            model.predict()

    def test_update_before_fit_raises(self):
        model = ARCHForecaster()
        with pytest.raises(RuntimeError):
            model.update(np.array([0.01]))

    def test_invalid_q_raises(self):
        with pytest.raises(ValueError):
            ARCHForecaster(q=0)


# ─── EWMAForecaster Tests ───

class TestEWMAForecaster:
    def test_fit_predict_fixed_lambda(self, garch_data):
        model = EWMAForecaster(lambda_=0.94)
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=1)
        assert result.point.shape == (1,)
        assert result.point[0] > 0
        assert result.target_spec.target == VolatilityTarget.CONDITIONAL_VARIANCE

    def test_multi_step_flat(self, garch_data):
        """EWMA multi-step forecasts should be flat (unit persistence)."""
        model = EWMAForecaster()
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=10)
        assert result.point.shape == (10,)
        assert np.all(result.point > 0)
        # All horizons should be equal (IGARCH flat forecast)
        assert np.allclose(result.point, result.point[0])

    def test_estimate_lambda(self, garch_data):
        model = EWMAForecaster(estimate_lambda=True)
        model.fit(garch_data.daily_returns[:500])
        params = model.get_params()
        assert 0.8 < params["lambda"] < 1.0
        result = model.predict(horizon=1)
        assert result.point[0] > 0

    def test_online_update(self, garch_data):
        model = EWMAForecaster()
        model.fit(garch_data.daily_returns[:500])
        n_before = len(model._returns)
        model.update(garch_data.daily_returns[500:505])
        assert len(model._returns) == n_before + 5
        assert len(model._sigma2) == n_before + 5

    def test_update_consistency(self, garch_data):
        """Update should produce same result as fitting on full data."""
        r = garch_data.daily_returns[:505]
        model_full = EWMAForecaster(lambda_=0.94)
        model_full.fit(r)

        model_update = EWMAForecaster(lambda_=0.94)
        model_update.fit(r[:500])
        model_update.update(r[500:505])

        # Last sigma2 should be identical
        assert np.isclose(model_update._sigma2[-1], model_full._sigma2[-1], rtol=1e-10)

    def test_model_spec(self):
        model = EWMAForecaster()
        spec = model.model_spec
        assert spec.abbreviation == "EWMA"
        assert spec.family == "GARCH"

    def test_predict_before_fit_raises(self):
        model = EWMAForecaster()
        with pytest.raises(RuntimeError):
            model.predict()

    def test_update_before_fit_raises(self):
        model = EWMAForecaster()
        with pytest.raises(RuntimeError):
            model.update(np.array([0.01]))

    def test_invalid_lambda_raises(self):
        with pytest.raises(ValueError):
            EWMAForecaster(lambda_=1.0)
        with pytest.raises(ValueError):
            EWMAForecaster(lambda_=0.0)
        with pytest.raises(ValueError):
            EWMAForecaster(lambda_=-0.5)
