"""
Tests for benchmark suite: synthetic data generation and runner.
"""

import numpy as np
import pytest

from volforecast.benchmark.synthetic import (
    generate_garch_data,
    generate_jump_diffusion_data,
    generate_sv_data,
)
from volforecast.benchmark.runner import BenchmarkRunner, BenchmarkResult, BenchmarkSuiteResult


class TestGARCHDGP:
    def test_shape(self):
        data = generate_garch_data(T=100, n_intraday=30, seed=42)
        assert data.daily_returns.shape == (100,)
        assert data.true_variance.shape == (100,)
        assert data.intraday_returns.shape == (100, 30)

    def test_positive_variance(self):
        data = generate_garch_data(T=500, seed=42)
        assert np.all(data.true_variance > 0)

    def test_reproducibility(self):
        d1 = generate_garch_data(T=50, seed=99)
        d2 = generate_garch_data(T=50, seed=99)
        assert np.allclose(d1.daily_returns, d2.daily_returns)
        assert np.allclose(d1.true_variance, d2.true_variance)

    def test_rv_close_to_true(self):
        """RV computed from intraday should approximate true variance."""
        from volforecast.realized.measures import realized_variance
        data = generate_garch_data(T=500, n_intraday=78, seed=42)
        rv = np.array([realized_variance(data.intraday_returns[t]) for t in range(500)])
        corr = np.corrcoef(rv, data.true_variance)[0, 1]
        assert corr > 0.5

    def test_dgp_name(self):
        data = generate_garch_data(T=50, seed=42)
        assert data.dgp_name == "GARCH(1,1)"


class TestJumpDiffusionDGP:
    def test_shape(self):
        data = generate_jump_diffusion_data(T=100, n_intraday=30, seed=42)
        assert data.daily_returns.shape == (100,)
        assert data.true_continuous.shape == (100,)
        assert data.true_jumps.shape == (100,)

    def test_jumps_present(self):
        data = generate_jump_diffusion_data(
            T=1000, jump_intensity=0.5, jump_std=0.03, seed=42,
        )
        assert np.sum(data.true_jumps > 0) > 0

    def test_qv_decomposition(self):
        """QV = C + J should hold by construction."""
        data = generate_jump_diffusion_data(T=200, seed=42)
        # true_variance is total conditional var, true_continuous + true_jumps ~ QV
        assert np.all(data.true_continuous >= 0)
        assert np.all(data.true_jumps >= 0)


class TestSVDGP:
    def test_shape(self):
        data = generate_sv_data(T=100, n_intraday=30, seed=42)
        assert data.daily_returns.shape == (100,)
        assert data.true_variance.shape == (100,)
        assert data.intraday_returns.shape == (100, 30)

    def test_positive_variance(self):
        data = generate_sv_data(T=500, seed=42)
        assert np.all(data.true_variance > 0)

    def test_mean_reversion(self):
        """Variance should revert toward theta over long samples."""
        data = generate_sv_data(T=5000, theta=0.04, kappa=5.0, seed=42)
        mean_var = np.mean(data.true_variance)
        assert abs(mean_var - 0.04) < 0.02  # Within 50% of theta


# ─── BenchmarkRunner Integration Tests ───

class TestBenchmarkRunner:
    def test_run_end_to_end(self):
        """Full integration test: generate data, run benchmark with HAR model."""
        from volforecast.models.har import HARForecaster

        data = generate_garch_data(T=150, n_intraday=30, seed=42)
        model = HARForecaster()

        runner = BenchmarkRunner(
            forecasters=[model],
            window_type="expanding",
            window_size=100,
            refit_every=0,
        )
        result = runner.run(
            daily_returns=data.daily_returns,
            intraday_returns=data.intraday_returns,
            true_variance=data.true_variance,
            dgp_name="GARCH(1,1)",
        )

        assert isinstance(result, BenchmarkSuiteResult)
        assert result.dgp_name == "GARCH(1,1)"
        assert result.n_train == 100
        assert result.n_oos == 50
        assert len(result.results) == 1

        br = result.results[0]
        assert isinstance(br, BenchmarkResult)
        assert br.forecasts.shape == (50,)
        assert br.proxies.shape == (50,)
        assert br.mse > 0
        assert np.isfinite(br.qlike)
        assert 0 <= br.mz_r2 <= 1.0

    def test_run_rolling_window(self):
        """Test rolling window mode."""
        from volforecast.models.har import HARForecaster

        data = generate_garch_data(T=150, n_intraday=30, seed=42)
        runner = BenchmarkRunner(
            forecasters=[HARForecaster()],
            window_type="rolling",
            window_size=100,
        )
        result = runner.run(
            daily_returns=data.daily_returns,
            intraday_returns=data.intraday_returns,
            dgp_name="GARCH(1,1)",
        )
        assert result.window_type == "rolling"
        assert result.n_oos == 50
        assert len(result.results) == 1

    def test_run_multiple_models(self):
        """Test with multiple forecasters simultaneously."""
        from volforecast.models.har import HARForecaster
        from volforecast.models.garch import GARCHForecaster

        data = generate_garch_data(T=150, n_intraday=30, seed=42)
        runner = BenchmarkRunner(
            forecasters=[HARForecaster(), GARCHForecaster()],
            window_size=100,
        )
        result = runner.run(
            daily_returns=data.daily_returns,
            intraday_returns=data.intraday_returns,
            true_variance=data.true_variance,
            dgp_name="GARCH(1,1)",
        )
        assert len(result.results) == 2
        names = [r.model_name for r in result.results]
        assert any("HAR" in n for n in names)
        assert any("GARCH" in n for n in names)

    def test_summary_table(self):
        """Test summary_table formatting."""
        from volforecast.models.har import HARForecaster

        data = generate_garch_data(T=150, n_intraday=30, seed=42)
        runner = BenchmarkRunner(
            forecasters=[HARForecaster()],
            window_size=100,
        )
        result = runner.run(
            daily_returns=data.daily_returns,
            intraday_returns=data.intraday_returns,
            true_variance=data.true_variance,
            dgp_name="GARCH(1,1)",
        )
        table = result.summary_table()
        assert isinstance(table, str)
        assert "GARCH(1,1)" in table
        assert "HAR" in table

    def test_refit_every(self):
        """Test periodic refitting."""
        from volforecast.models.har import HARForecaster

        data = generate_garch_data(T=130, n_intraday=30, seed=42)
        runner = BenchmarkRunner(
            forecasters=[HARForecaster()],
            window_size=100,
            refit_every=10,
        )
        result = runner.run(
            daily_returns=data.daily_returns,
            intraday_returns=data.intraday_returns,
            dgp_name="GARCH(1,1)",
        )
        assert result.n_oos == 30
        assert len(result.results) == 1
        assert np.all(np.isfinite(result.results[0].forecasts))
