"""
Tests for AutoVolForecaster v1.

Covers:
- DataProfiler on synthetic data
- CandidateSelector output
- Full pipeline happy path (synthetic GARCH data)
- Edge case: short series
- Edge case: returns-only (no intraday)
"""

from __future__ import annotations

import numpy as np
import pytest

from volforecast.auto.profiler import DataProfiler, DataProfile
from volforecast.auto.selector import CandidateSelector
from volforecast.auto.model_selection import ModelSelector
from volforecast.auto.combination import CombinedForecaster, select_combiner
from volforecast.auto.auto import AutoVolForecaster, AutoForecastResult, auto_fit


# ─── Fixtures ──────────────────────────────────────────────────────

def _make_garch_returns(T: int = 1000, seed: int = 42) -> np.ndarray:
    """Generate simple GARCH(1,1) returns for testing."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 1e-6, 0.08, 0.90
    unc_var = omega / (1.0 - alpha - beta)
    r = np.empty(T, dtype=np.float64)
    h = np.empty(T, dtype=np.float64)
    h[0] = unc_var
    r[0] = rng.normal(0, np.sqrt(h[0]))
    for t in range(1, T):
        h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
        r[t] = rng.normal(0, np.sqrt(h[t]))
    return r


# ─── DataProfiler tests ───────────────────────────────────────────

class TestDataProfiler:

    def test_basic_profile(self):
        returns = _make_garch_returns(500)
        profile = DataProfiler.profile(returns)
        assert isinstance(profile, DataProfile)
        assert profile.T == 500
        assert not profile.has_intraday
        assert not profile.has_realized
        assert 0.0 < profile.hurst_exp < 1.0
        assert isinstance(profile.has_leverage, bool)
        assert isinstance(profile.has_regime_switching, bool)

    def test_short_series(self):
        returns = _make_garch_returns(50)
        profile = DataProfiler.profile(returns)
        assert profile.T == 50

    def test_with_realized_measures(self):
        returns = _make_garch_returns(500)
        rv = returns ** 2  # crude proxy
        profile = DataProfiler.profile(
            returns, realized_measures={"RV": rv}
        )
        assert profile.has_realized
        assert profile.rv is not None


# ─── CandidateSelector tests ──────────────────────────────────────

class TestCandidateSelector:

    def test_baseline_always_selected(self):
        returns = _make_garch_returns(200)
        profile = DataProfiler.profile(returns)
        candidates = CandidateSelector.select(profile)
        names = [c.model_spec.name for c in candidates]
        # Should always have at least GARCH
        assert any("GARCH" in n for n in names)

    def test_family_filter(self):
        returns = _make_garch_returns(500)
        profile = DataProfiler.profile(returns)
        candidates = CandidateSelector.select(profile, model_families=["GARCH"])
        for c in candidates:
            assert c.model_spec.family in ("GARCH",)

    def test_long_series_has_more_candidates(self):
        short = DataProfiler.profile(_make_garch_returns(200))
        long = DataProfiler.profile(_make_garch_returns(1500))
        c_short = CandidateSelector.select(short)
        c_long = CandidateSelector.select(long)
        assert len(c_long) >= len(c_short)


# ─── Combiner selection tests ─────────────────────────────────────

class TestCombinerSelection:

    def test_passthrough_single(self):
        combiner, name = select_combiner(1, 1000, False)
        assert name == "Passthrough"

    def test_equal_weight_short(self):
        combiner, name = select_combiner(3, 500, False)
        assert name == "EqualWeight"

    def test_ewa_default(self):
        combiner, name = select_combiner(3, 1000, False)
        assert name == "EWA"

    def test_fixed_share_regime(self):
        combiner, name = select_combiner(3, 1000, True)
        assert name == "FixedShare"

    def test_explicit_method(self):
        combiner, name = select_combiner(3, 1000, False, "after")
        assert name == "AFTER"


# ─── Integration: full pipeline ───────────────────────────────────

class TestAutoVolForecasterIntegration:

    @pytest.fixture
    def garch_returns(self):
        return _make_garch_returns(800, seed=123)

    def test_fit_returns_result(self, garch_returns):
        """Full pipeline with returns only, GARCH family only (fast)."""
        avf = AutoVolForecaster(
            model_families=["GARCH"],
            min_train=400,
            refit_every=50,
        )
        result = avf.fit(garch_returns)
        assert isinstance(result, AutoForecastResult)
        assert isinstance(result.forecaster, CombinedForecaster)
        assert len(result.component_models) >= 1
        assert result.combiner_name in (
            "Passthrough", "EqualWeight", "EWA", "FixedShare", "AFTER"
        )

    def test_predict_after_fit(self, garch_returns):
        avf = AutoVolForecaster(
            model_families=["GARCH"],
            min_train=400,
            refit_every=50,
        )
        avf.fit(garch_returns)
        fr = avf.predict(horizon=1)
        assert fr.point.shape == (1,)
        assert fr.point[0] > 0

    def test_predict_horizon_gt1_raises(self, garch_returns):
        avf = AutoVolForecaster(
            model_families=["GARCH"],
            min_train=400,
            refit_every=50,
        )
        avf.fit(garch_returns)
        with pytest.raises(NotImplementedError):
            avf.predict(horizon=5)

    def test_update(self, garch_returns):
        avf = AutoVolForecaster(
            model_families=["GARCH"],
            min_train=400,
            refit_every=50,
        )
        avf.fit(garch_returns)
        # Should not raise
        avf.update(np.array([0.01]))

    def test_auto_fit_convenience(self, garch_returns):
        result = auto_fit(
            garch_returns,
            model_families=["GARCH"],
            min_train=400,
            refit_every=50,
        )
        assert isinstance(result, AutoForecastResult)

    def test_not_fitted_raises(self):
        avf = AutoVolForecaster()
        with pytest.raises(RuntimeError):
            avf.predict()
        with pytest.raises(RuntimeError):
            avf.update(np.array([0.01]))
