"""
Benchmark runner for reproducible volatility forecasting experiments.

Supports:
- Expanding window estimation
- Rolling window estimation
- Multiple forecasters evaluated simultaneously
- Automatic realized measure computation
- Result collection with all evaluation metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from volforecast.core.base import BaseForecaster, ForecastResult
from volforecast.realized.measures import (
    realized_variance,
    bipower_variation,
    realized_semivariance,
)
from volforecast.realized.jumps import jump_decomposition
from volforecast.evaluation.losses import mse_loss, qlike_loss
from volforecast.evaluation.tests import diebold_mariano_test, mincer_zarnowitz_test
from volforecast.evaluation.proxy import proxy_noise_correction


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run.

    Attributes
    ----------
    model_name : str
    forecasts : NDArray, shape (n_oos,)
    proxies : NDArray, shape (n_oos,)
    true_variance : NDArray, shape (n_oos,), optional
    mse : float
    qlike : float
    mz_alpha : float
    mz_beta : float
    mz_r2 : float
    """
    model_name: str
    forecasts: NDArray[np.float64]
    proxies: NDArray[np.float64]
    true_variance: Optional[NDArray[np.float64]] = None
    mse: float = 0.0
    qlike: float = 0.0
    mse_vs_true: float = 0.0
    mz_alpha: float = 0.0
    mz_beta: float = 0.0
    mz_r2: float = 0.0
    mz_efficient: bool = False


@dataclass
class BenchmarkSuiteResult:
    """Results from a full benchmark suite."""
    results: list[BenchmarkResult] = field(default_factory=list)
    dgp_name: str = ""
    n_train: int = 0
    n_oos: int = 0
    window_type: str = "expanding"

    def summary_table(self) -> str:
        """Format results as a text table."""
        header = (
            f"{'Model':<25} {'MSE':>12} {'QLIKE':>12} "
            f"{'MSE(true)':>12} {'MZ_R2':>8} {'MZ_eff':>6}"
        )
        lines = [f"Benchmark: {self.dgp_name} | {self.window_type} window "
                 f"| train={self.n_train} oos={self.n_oos}", "=" * len(header), header,
                 "-" * len(header)]
        for r in sorted(self.results, key=lambda x: x.mse):
            lines.append(
                f"{r.model_name:<25} {r.mse:>12.6e} {r.qlike:>12.6f} "
                f"{r.mse_vs_true:>12.6e} {r.mz_r2:>8.4f} "
                f"{'Y' if r.mz_efficient else 'N':>6}"
            )
        return "\n".join(lines)


class BenchmarkRunner:
    """Run reproducible volatility forecasting benchmarks.

    Parameters
    ----------
    forecasters : list of BaseForecaster
        Models to benchmark.
    window_type : str
        "expanding" or "rolling".
    window_size : int
        Initial training window (expanding) or fixed window (rolling).
    refit_every : int
        Re-estimate parameters every N steps. 0 = no refitting after initial.
    """

    def __init__(
        self,
        forecasters: Sequence[BaseForecaster],
        window_type: str = "expanding",
        window_size: int = 500,
        refit_every: int = 0,
    ) -> None:
        self.forecasters = list(forecasters)
        self.window_type = window_type
        self.window_size = window_size
        self.refit_every = refit_every

    def run(
        self,
        daily_returns: NDArray[np.float64],
        intraday_returns: NDArray[np.float64],
        true_variance: Optional[NDArray[np.float64]] = None,
        dgp_name: str = "Unknown",
    ) -> BenchmarkSuiteResult:
        """Run the benchmark.

        Parameters
        ----------
        daily_returns : array, shape (T,)
        intraday_returns : array, shape (T, n_intraday)
        true_variance : array, shape (T,), optional
        dgp_name : str

        Returns
        -------
        BenchmarkSuiteResult
        """
        T = len(daily_returns)
        n_oos = T - self.window_size

        # Pre-compute realized measures for all days
        rv = np.array([realized_variance(intraday_returns[t]) for t in range(T)])
        bv = np.array([bipower_variation(intraday_returns[t]) for t in range(T)])
        rsv = [realized_semivariance(intraday_returns[t]) for t in range(T)]
        rs_pos = np.array([r[0] for r in rsv])
        rs_neg = np.array([r[1] for r in rsv])
        cv = np.minimum(bv, rv)
        jv = np.maximum(rv - bv, 0.0)

        realized = {
            "RV": rv, "BV": bv, "CV": cv, "JV": jv,
            "RS_pos": rs_pos, "RS_neg": rs_neg,
        }

        suite = BenchmarkSuiteResult(
            dgp_name=dgp_name,
            n_train=self.window_size,
            n_oos=n_oos,
            window_type=self.window_type,
        )

        for forecaster in self.forecasters:
            model_name = forecaster.model_spec.name
            forecasts_oos = np.empty(n_oos, dtype=np.float64)

            for t in range(n_oos):
                oos_idx = self.window_size + t

                # Determine training window
                if self.window_type == "expanding":
                    train_start = 0
                else:
                    train_start = max(0, oos_idx - self.window_size)
                train_end = oos_idx

                # Fit or refit
                need_fit = (t == 0) or (
                    self.refit_every > 0 and t % self.refit_every == 0
                )

                if need_fit:
                    train_r = daily_returns[train_start:train_end]
                    train_realized = {
                        k: v[train_start:train_end] for k, v in realized.items()
                    }
                    try:
                        forecaster.fit(train_r, train_realized)
                    except Exception:
                        forecasts_oos[t] = rv[oos_idx - 1]  # Fallback: last RV
                        continue
                else:
                    # Online update
                    try:
                        new_r = daily_returns[oos_idx - 1:oos_idx]
                        new_realized = {
                            k: v[oos_idx - 1:oos_idx] for k, v in realized.items()
                        }
                        forecaster.update(new_r, new_realized)
                    except Exception:
                        pass

                # Predict
                try:
                    result = forecaster.predict(horizon=1)
                    forecasts_oos[t] = result.point[0]
                except Exception:
                    forecasts_oos[t] = rv[oos_idx - 1]

            # Evaluate
            proxies = rv[self.window_size:]
            true_var_oos = true_variance[self.window_size:] if true_variance is not None else None

            mse_val = mse_loss(forecasts_oos, proxies)
            qlike_val = qlike_loss(forecasts_oos, proxies)

            mse_true = 0.0
            if true_var_oos is not None:
                mse_true = mse_loss(forecasts_oos, true_var_oos)

            mz = mincer_zarnowitz_test(forecasts_oos, proxies)

            suite.results.append(BenchmarkResult(
                model_name=model_name,
                forecasts=forecasts_oos,
                proxies=proxies,
                true_variance=true_var_oos,
                mse=mse_val,
                qlike=qlike_val,
                mse_vs_true=mse_true,
                mz_alpha=mz.alpha,
                mz_beta=mz.beta,
                mz_r2=mz.r_squared,
                mz_efficient=mz.efficient,
            ))

        return suite
