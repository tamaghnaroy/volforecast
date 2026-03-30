"""
Tests for new model families: FIGARCH, HEAVY, SV, GAS, MS-Vol, MIDAS, CAViaR, ML wrappers.

Tests verify:
- Models fit without error on synthetic data
- Forecasts are positive and finite
- Online update extends state correctly
- Model specs are well-formed
- Error handling for invalid inputs
"""

import numpy as np
import pytest

from volforecast.models.figarch import FIGARCHForecaster, figarch_filter
from volforecast.models.heavy import HEAVYForecaster, heavy_filter
from volforecast.models.sv import SVForecaster, SVJForecaster
from volforecast.models.gas import GASVolForecaster, gas_normal_filter, gas_student_filter
from volforecast.models.markov_switching import MSVolForecaster
from volforecast.models.midas import GARCHMIDASForecaster
from volforecast.models.caviar import CAViaRForecaster
from volforecast.models.ml_wrappers import RFVolForecaster
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


# ─── FIGARCH Tests ───

class TestFIGARCHFilter:
    def test_filter_shape(self):
        r = np.random.default_rng(42).normal(0, 0.01, size=200)
        sig2 = figarch_filter(r, 1e-6, 0.4, 0.2, 0.3, 100)
        assert sig2.shape == (200,)
        assert np.all(sig2 > 0)
        assert np.all(np.isfinite(sig2))


class TestFIGARCHForecaster:
    def test_fit_predict(self, garch_data):
        model = FIGARCHForecaster(truncation=200)
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=1)
        assert result.point.shape == (1,)
        assert result.point[0] > 0
        assert result.target_spec.target == VolatilityTarget.CONDITIONAL_VARIANCE

    def test_multi_step(self, garch_data):
        model = FIGARCHForecaster(truncation=200)
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=10)
        assert result.point.shape == (10,)
        assert np.all(result.point > 0)
        assert np.all(np.isfinite(result.point))

    def test_fractional_d_in_range(self, garch_data):
        model = FIGARCHForecaster(truncation=200)
        model.fit(garch_data.daily_returns[:500])
        d = model.get_params()["d"]
        assert 0.0 < d < 1.0

    def test_online_update(self, garch_data):
        model = FIGARCHForecaster(truncation=200)
        model.fit(garch_data.daily_returns[:500])
        n_before = len(model._returns)
        model.update(garch_data.daily_returns[500:505])
        assert len(model._returns) == n_before + 5

    def test_model_spec(self):
        model = FIGARCHForecaster()
        spec = model.model_spec
        assert spec.abbreviation == "FIGARCH"
        assert spec.family == "GARCH"

    def test_predict_before_fit_raises(self):
        model = FIGARCHForecaster()
        with pytest.raises(RuntimeError):
            model.predict()

    def test_update_before_fit_raises(self):
        model = FIGARCHForecaster()
        with pytest.raises(RuntimeError):
            model.update(np.array([0.01]))


# ─── HEAVY Tests ───

class TestHEAVYFilter:
    def test_filter_shape(self):
        r = np.random.default_rng(42).normal(0, 0.01, size=200)
        rm = np.abs(r) ** 2 * 1.2 + 1e-6
        h = heavy_filter(r, rm, 1e-6, 0.3, 0.6)
        assert h.shape == (200,)
        assert np.all(h > 0)


class TestHEAVYForecaster:
    def test_fit_predict(self, garch_data, rv_series):
        model = HEAVYForecaster()
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500]})
        result = model.predict(horizon=1)
        assert result.point.shape == (1,)
        assert result.point[0] > 0

    def test_multi_step(self, garch_data, rv_series):
        model = HEAVYForecaster()
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500]})
        result = model.predict(horizon=10)
        assert result.point.shape == (10,)
        assert np.all(result.point > 0)
        assert np.all(np.isfinite(result.point))

    def test_online_update(self, garch_data, rv_series):
        model = HEAVYForecaster()
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500]})
        n_before = len(model._returns)
        model.update(garch_data.daily_returns[500:503],
                     new_realized={"RV": rv_series[500:503]})
        assert len(model._returns) == n_before + 3

    def test_requires_rv(self, garch_data):
        model = HEAVYForecaster()
        with pytest.raises(ValueError):
            model.fit(garch_data.daily_returns[:500])

    def test_update_requires_rv(self, garch_data, rv_series):
        model = HEAVYForecaster()
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500]})
        with pytest.raises(ValueError):
            model.update(garch_data.daily_returns[500:503])

    def test_update_mismatched_lengths(self, garch_data, rv_series):
        model = HEAVYForecaster()
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500]})
        with pytest.raises(ValueError, match="same length"):
            model.update(garch_data.daily_returns[500:503],
                         new_realized={"RV": rv_series[500:502]})

    def test_predict_before_fit_raises(self):
        model = HEAVYForecaster()
        with pytest.raises(RuntimeError):
            model.predict()

    def test_model_spec(self):
        model = HEAVYForecaster()
        spec = model.model_spec
        assert spec.abbreviation == "HEAVY"


# ─── SV Tests ───

class TestSVForecaster:
    def test_fit_predict(self, garch_data):
        model = SVForecaster()
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=1)
        assert result.point.shape == (1,)
        assert result.point[0] > 0

    def test_multi_step(self, garch_data):
        model = SVForecaster()
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=5)
        assert result.point.shape == (5,)
        assert np.all(result.point > 0)
        assert np.all(np.isfinite(result.point))

    def test_persistence_in_range(self, garch_data):
        model = SVForecaster()
        model.fit(garch_data.daily_returns[:500])
        phi = model.get_params()["phi"]
        assert -1.0 < phi < 1.0

    def test_online_update(self, garch_data):
        model = SVForecaster()
        model.fit(garch_data.daily_returns[:500])
        n_before = len(model._returns)
        model.update(garch_data.daily_returns[500:505])
        assert len(model._returns) == n_before + 5

    def test_predict_before_fit_raises(self):
        model = SVForecaster()
        with pytest.raises(RuntimeError):
            model.predict()

    def test_model_spec(self):
        model = SVForecaster()
        spec = model.model_spec
        assert spec.abbreviation == "SV"
        assert spec.family == "SV"


class TestSVJForecaster:
    def test_fit_predict(self, garch_data):
        model = SVJForecaster()
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=3)
        assert result.point.shape == (3,)
        assert np.all(result.point > 0)

    def test_jump_params(self, garch_data):
        model = SVJForecaster()
        model.fit(garch_data.daily_returns[:500])
        params = model.get_params()
        assert "lambda_j" in params
        assert "mu_j" in params
        assert "sigma_j" in params
        assert params["lambda_j"] >= 0

    def test_online_update(self, garch_data):
        model = SVJForecaster()
        model.fit(garch_data.daily_returns[:500])
        n_before = len(model._returns)
        model.update(garch_data.daily_returns[500:503])
        assert len(model._returns) == n_before + 3

    def test_predict_before_fit_raises(self):
        model = SVJForecaster()
        with pytest.raises(RuntimeError):
            model.predict()

    def test_model_spec(self):
        model = SVJForecaster()
        spec = model.model_spec
        assert spec.abbreviation == "SVJ"
        assert spec.family == "SV"


# ─── GAS Tests ───

class TestGASFilter:
    def test_normal_filter_shape(self):
        r = np.random.default_rng(42).normal(0, 0.01, size=200)
        sig2 = gas_normal_filter(r, 0.01, 0.1, 0.98)
        assert sig2.shape == (200,)
        assert np.all(sig2 > 0)

    def test_student_filter_shape(self):
        r = np.random.default_rng(42).normal(0, 0.01, size=200)
        sig2 = gas_student_filter(r, 0.01, 0.1, 0.98, 5.0)
        assert sig2.shape == (200,)
        assert np.all(sig2 > 0)


class TestGASVolForecaster:
    def test_fit_predict_normal(self, garch_data):
        model = GASVolForecaster(dist="normal")
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=1)
        assert result.point.shape == (1,)
        assert result.point[0] > 0

    def test_fit_predict_student(self, garch_data):
        model = GASVolForecaster(dist="t")
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=5)
        assert result.point.shape == (5,)
        assert np.all(result.point > 0)
        params = model.get_params()
        assert "nu" in params
        assert params["nu"] > 2.0

    def test_multi_step(self, garch_data):
        model = GASVolForecaster(dist="normal")
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=10)
        assert result.point.shape == (10,)
        assert np.all(result.point > 0)
        assert np.all(np.isfinite(result.point))

    def test_online_update(self, garch_data):
        model = GASVolForecaster()
        model.fit(garch_data.daily_returns[:500])
        n_before = len(model._returns)
        model.update(garch_data.daily_returns[500:505])
        assert len(model._returns) == n_before + 5

    def test_predict_before_fit_raises(self):
        model = GASVolForecaster()
        with pytest.raises(RuntimeError):
            model.predict()

    def test_invalid_dist_raises(self):
        with pytest.raises(ValueError):
            GASVolForecaster(dist="cauchy")

    def test_model_spec(self):
        model = GASVolForecaster()
        spec = model.model_spec
        assert spec.abbreviation == "GAS"
        assert spec.family == "GAS"


# ─── Markov-Switching Tests ───

class TestMSVolForecaster:
    def test_fit_predict(self, garch_data):
        model = MSVolForecaster(n_regimes=2)
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=1)
        assert result.point.shape == (1,)
        assert result.point[0] > 0

    def test_multi_step(self, garch_data):
        model = MSVolForecaster(n_regimes=2)
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=10)
        assert result.point.shape == (10,)
        assert np.all(result.point > 0)
        assert np.all(np.isfinite(result.point))

    def test_regime_variances_ordered(self, garch_data):
        model = MSVolForecaster(n_regimes=2)
        model.fit(garch_data.daily_returns[:500])
        params = model.get_params()
        assert params["sigma2_0"] <= params["sigma2_1"]

    def test_transition_probs_sum_to_one(self, garch_data):
        model = MSVolForecaster(n_regimes=2)
        model.fit(garch_data.daily_returns[:500])
        params = model.get_params()
        assert np.isclose(params["p_00"] + params["p_01"], 1.0, atol=1e-6)
        assert np.isclose(params["p_10"] + params["p_11"], 1.0, atol=1e-6)

    def test_online_update(self, garch_data):
        model = MSVolForecaster(n_regimes=2)
        model.fit(garch_data.daily_returns[:500])
        n_before = len(model._returns)
        model.update(garch_data.daily_returns[500:505])
        assert len(model._returns) == n_before + 5

    def test_predict_before_fit_raises(self):
        model = MSVolForecaster()
        with pytest.raises(RuntimeError):
            model.predict()

    def test_invalid_regimes_raises(self):
        with pytest.raises(ValueError):
            MSVolForecaster(n_regimes=1)

    def test_model_spec(self):
        model = MSVolForecaster()
        spec = model.model_spec
        assert spec.abbreviation == "MSVol"
        assert spec.family == "MS"


# ─── GARCH-MIDAS Tests ───

class TestGARCHMIDASForecaster:
    def test_fit_predict(self, garch_data):
        model = GARCHMIDASForecaster(K=22)
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=1)
        assert result.point.shape == (1,)
        assert result.point[0] > 0

    def test_multi_step(self, garch_data):
        model = GARCHMIDASForecaster(K=22)
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=10)
        assert result.point.shape == (10,)
        assert np.all(result.point > 0)
        assert np.all(np.isfinite(result.point))

    def test_params_reasonable(self, garch_data):
        model = GARCHMIDASForecaster(K=22)
        model.fit(garch_data.daily_returns[:500])
        params = model.get_params()
        assert 0 < params["alpha"] < 1
        assert 0 < params["beta"] < 1
        assert params["alpha"] + params["beta"] < 1.0

    def test_online_update(self, garch_data):
        model = GARCHMIDASForecaster(K=22)
        model.fit(garch_data.daily_returns[:500])
        n_before = len(model._returns)
        model.update(garch_data.daily_returns[500:505])
        assert len(model._returns) == n_before + 5

    def test_predict_before_fit_raises(self):
        model = GARCHMIDASForecaster()
        with pytest.raises(RuntimeError):
            model.predict()

    def test_model_spec(self):
        model = GARCHMIDASForecaster()
        spec = model.model_spec
        assert spec.abbreviation == "GMIDAS"
        assert spec.family == "GARCH"


# ─── CAViaR Tests ───

class TestCAViaRForecaster:
    def test_fit_predict_sav(self, garch_data):
        model = CAViaRForecaster(tau=0.05, spec="SAV")
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=1)
        assert result.point.shape == (1,)
        assert np.isfinite(result.point[0])

    def test_fit_predict_as(self, garch_data):
        model = CAViaRForecaster(tau=0.05, spec="AS")
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=5)
        assert result.point.shape == (5,)
        assert np.all(np.isfinite(result.point))

    def test_fit_predict_igarch(self, garch_data):
        model = CAViaRForecaster(tau=0.05, spec="IGARCH")
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=3)
        assert result.point.shape == (3,)
        assert np.all(np.isfinite(result.point))

    def test_hit_rate_near_tau(self, garch_data):
        model = CAViaRForecaster(tau=0.05, spec="SAV")
        model.fit(garch_data.daily_returns[:500])
        hr = model.hit_rate()
        # Hit rate should be roughly near tau (within reason)
        assert 0.0 < hr < 0.5

    def test_online_update(self, garch_data):
        model = CAViaRForecaster(tau=0.05, spec="SAV")
        model.fit(garch_data.daily_returns[:500])
        n_before = len(model._returns)
        model.update(garch_data.daily_returns[500:505])
        assert len(model._returns) == n_before + 5

    def test_predict_before_fit_raises(self):
        model = CAViaRForecaster()
        with pytest.raises(RuntimeError):
            model.predict()

    def test_invalid_tau_raises(self):
        with pytest.raises(ValueError):
            CAViaRForecaster(tau=0.0)
        with pytest.raises(ValueError):
            CAViaRForecaster(tau=1.0)

    def test_invalid_spec_raises(self):
        with pytest.raises(ValueError):
            CAViaRForecaster(spec="invalid")

    def test_model_spec(self):
        model = CAViaRForecaster()
        spec = model.model_spec
        assert spec.abbreviation == "CAViaR"
        assert spec.family == "Quantile"


# ─── RF Wrapper Tests ───

class TestRFVolForecaster:
    def test_fit_predict(self, garch_data):
        model = RFVolForecaster(n_lags=10, n_estimators=20, random_state=42)
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=1)
        assert result.point.shape == (1,)
        assert result.point[0] > 0

    def test_multi_step(self, garch_data):
        model = RFVolForecaster(n_lags=10, n_estimators=20, random_state=42)
        model.fit(garch_data.daily_returns[:500])
        result = model.predict(horizon=5)
        assert result.point.shape == (5,)
        assert np.all(result.point > 0)
        assert np.all(np.isfinite(result.point))

    def test_with_rv(self, garch_data, rv_series):
        model = RFVolForecaster(n_lags=10, n_estimators=20, random_state=42)
        model.fit(garch_data.daily_returns[:500],
                  realized_measures={"RV": rv_series[:500]})
        result = model.predict(horizon=3)
        assert result.point.shape == (3,)
        assert np.all(result.point > 0)

    def test_online_update(self, garch_data):
        model = RFVolForecaster(n_lags=10, n_estimators=20, random_state=42)
        model.fit(garch_data.daily_returns[:500])
        n_before = len(model._returns)
        model.update(garch_data.daily_returns[500:505])
        assert len(model._returns) == n_before + 5

    def test_predict_before_fit_raises(self):
        model = RFVolForecaster()
        with pytest.raises(RuntimeError):
            model.predict()

    def test_model_spec(self):
        model = RFVolForecaster()
        spec = model.model_spec
        assert spec.abbreviation == "RF"
        assert spec.family == "ML"


# ─── LSTM/Transformer Tests (only if torch available) ───

def _torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
class TestLSTMVolForecaster:
    def test_fit_predict(self, garch_data):
        from volforecast.models.ml_wrappers import LSTMVolForecaster
        model = LSTMVolForecaster(n_lags=10, hidden_size=8, n_epochs=5, random_state=42)
        model.fit(garch_data.daily_returns[:200])
        result = model.predict(horizon=1)
        assert result.point.shape == (1,)
        assert result.point[0] > 0

    def test_multi_step(self, garch_data):
        from volforecast.models.ml_wrappers import LSTMVolForecaster
        model = LSTMVolForecaster(n_lags=10, hidden_size=8, n_epochs=5, random_state=42)
        model.fit(garch_data.daily_returns[:200])
        result = model.predict(horizon=3)
        assert result.point.shape == (3,)
        assert np.all(result.point > 0)

    def test_model_spec(self):
        from volforecast.models.ml_wrappers import LSTMVolForecaster
        model = LSTMVolForecaster()
        spec = model.model_spec
        assert spec.abbreviation == "LSTM"
        assert spec.family == "ML"


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
class TestTransformerVolForecaster:
    def test_fit_predict(self, garch_data):
        from volforecast.models.ml_wrappers import TransformerVolForecaster
        model = TransformerVolForecaster(
            n_lags=10, d_model=8, n_heads=2, n_epochs=5, random_state=42,
        )
        model.fit(garch_data.daily_returns[:200])
        result = model.predict(horizon=1)
        assert result.point.shape == (1,)
        assert result.point[0] > 0

    def test_multi_step(self, garch_data):
        from volforecast.models.ml_wrappers import TransformerVolForecaster
        model = TransformerVolForecaster(
            n_lags=10, d_model=8, n_heads=2, n_epochs=5, random_state=42,
        )
        model.fit(garch_data.daily_returns[:200])
        result = model.predict(horizon=3)
        assert result.point.shape == (3,)
        assert np.all(result.point > 0)

    def test_model_spec(self):
        from volforecast.models.ml_wrappers import TransformerVolForecaster
        model = TransformerVolForecaster()
        spec = model.model_spec
        assert spec.abbreviation == "Transformer"
        assert spec.family == "ML"
