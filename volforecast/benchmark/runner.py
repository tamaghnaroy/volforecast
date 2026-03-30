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

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

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

    def summary_table(self, sort_by: str = "qlike") -> str:
        """Format results as a text table.

        Parameters
        ----------
        sort_by : str
            Column to sort by: "qlike" (default) or "mse".
        """
        header = (
            f"{'Model':<25} {'MSE':>12} {'QLIKE':>12} "
            f"{'MSE(true)':>12} {'MZ_R2':>8} {'MZ_eff':>6}"
        )
        sort_key = (lambda x: x.qlike) if sort_by == "qlike" else (lambda x: x.mse)
        lines = [f"Benchmark: {self.dgp_name} | {self.window_type} window "
                 f"| train={self.n_train} oos={self.n_oos} | sorted by {sort_by.upper()}",
                 "=" * len(header), header, "-" * len(header)]
        for r in sorted(self.results, key=sort_key):
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
        intraday_returns: Optional[NDArray[np.float64]] = None,
        precomputed_realized: Optional[dict[str, NDArray[np.float64]]] = None,
        true_variance: Optional[NDArray[np.float64]] = None,
        dgp_name: str = "Unknown",
    ) -> BenchmarkSuiteResult:
        """Run the benchmark.

        Parameters
        ----------
        daily_returns : array, shape (T,)
        intraday_returns : array, shape (T, n_intraday), optional
            If provided, used to compute realized measures.
        precomputed_realized : dict, optional
            Pre-computed realized measures. Must contain at least "RV".
            Missing keys ("BV", "CV", "JV", "RS_pos", "RS_neg") are derived
            from RV automatically. If provided, intraday_returns is ignored.
        true_variance : array, shape (T,), optional
        dgp_name : str

        Returns
        -------
        BenchmarkSuiteResult
        """
        T = len(daily_returns)
        if self.window_size >= T:
            # Clamp window to leave at least 10% OOS (min 10 obs)
            self.window_size = max(T - max(T // 10, 10), 1)
            logger.warning(
                "window_size >= T (%d >= %d). Clamped to %d.",
                self.window_size, T, self.window_size,
            )
        n_oos = T - self.window_size

        # Compute or use pre-computed realized measures
        if precomputed_realized is not None:
            realized = dict(precomputed_realized)  # shallow copy
            if "RV" not in realized:
                raise ValueError("precomputed_realized must contain at least 'RV'")
            rv = realized["RV"]
            # Derive missing keys from RV when not provided
            if "BV" not in realized:
                realized["BV"] = rv.copy()
            if "CV" not in realized:
                realized["CV"] = np.minimum(realized["BV"], rv)
            if "JV" not in realized:
                realized["JV"] = np.maximum(rv - realized["BV"], 0.0)
            if "RS_pos" not in realized:
                realized["RS_pos"] = rv / 2.0
            if "RS_neg" not in realized:
                realized["RS_neg"] = rv / 2.0
        elif intraday_returns is not None:
            # Compute realized measures from intraday returns
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
        else:
            # Daily-returns-only fallback: use squared daily returns as crude
            # RV proxy.  This is a noisy but unbiased estimator under the
            # assumption of zero-mean returns and allows the benchmark-and-
            # select workflow to run without intraday data.
            logger.info(
                "No intraday data or precomputed realized measures provided. "
                "Falling back to squared daily returns as RV proxy."
            )
            r2 = daily_returns ** 2
            realized = {
                "RV": r2,
                "BV": r2.copy(),       # no intraday → BV ≈ RV
                "CV": r2.copy(),       # continuous ≈ total (no jump decomp)
                "JV": np.zeros_like(r2),
                "RS_pos": r2 / 2.0,    # symmetric split
                "RS_neg": r2 / 2.0,
            }

        # Track failed models to exclude from final results (fail-safe behavior)
        failed_models: set[str] = set()

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
                    except Exception as e:
                        logger.warning("%s fit failed at t=%d: %s", model_name, t, e)
                        failed_models.add(model_name)
                        break  # Stop processing this model - it failed
                else:
                    # Online update
                    try:
                        new_r = daily_returns[oos_idx - 1:oos_idx]
                        new_realized = {
                            k: v[oos_idx - 1:oos_idx] for k, v in realized.items()
                        }
                        forecaster.update(new_r, new_realized)
                    except Exception as e:
                        logger.warning("%s update failed at t=%d: %s", model_name, t, e)
                        failed_models.add(model_name)
                        break  # Stop processing this model - it failed

                # Predict
                try:
                    result = forecaster.predict(horizon=1)
                    forecasts_oos[t] = result.point[0]
                except Exception as e:
                    logger.warning("%s predict failed at t=%d: %s", model_name, t, e)
                    failed_models.add(model_name)
                    break  # Stop processing this model - it failed

            # Skip evaluation if model failed during the run
            if model_name in failed_models:
                logger.warning("Excluding %s from results due to failures", model_name)
                continue

            proxies = realized["RV"][self.window_size:]
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

        # Fail-safe: if all models failed, run a GARCH(1,1) fallback so the
        # downstream DM/MCS stages always have at least one result.
        if not suite.results:
            logger.warning(
                "All %d candidate models failed. Running GARCH(1,1) fallback.",
                len(self.forecasters),
            )
            from volforecast.models.garch import GARCHForecaster
            from volforecast.core.base import ModelSpec
            fallback = GARCHForecaster()
            fb_forecasts = np.full(n_oos, np.nan, dtype=np.float64)
            fb_fitted = False
            for t in range(n_oos):
                oos_idx = self.window_size + t
                if self.window_type == "expanding":
                    ts = 0
                else:
                    ts = max(0, oos_idx - self.window_size)
                te = oos_idx
                need_fit = (t == 0) or (
                    self.refit_every > 0 and t % self.refit_every == 0
                )
                try:
                    if need_fit:
                        fallback.fit(
                            daily_returns[ts:te],
                            {k: v[ts:te] for k, v in realized.items()},
                        )
                        fb_fitted = True
                    elif fb_fitted:
                        fallback.update(
                            daily_returns[oos_idx - 1:oos_idx],
                            {k: v[oos_idx - 1:oos_idx] for k, v in realized.items()},
                        )
                    else:
                        continue  # skip until first successful fit
                    res = fallback.predict(horizon=1)
                    fb_forecasts[t] = res.point[0]
                except Exception:
                    # Leave as NaN — do NOT substitute proxy values
                    pass

            # Only include fallback if it produced enough valid forecasts
            valid_mask = ~np.isnan(fb_forecasts)
            n_valid = int(np.sum(valid_mask))
            if n_valid >= max(n_oos // 2, 10):
                # Use only the valid portion for evaluation
                proxies = realized["RV"][self.window_size:]
                fb_valid = fb_forecasts[valid_mask]
                px_valid = proxies[valid_mask]
                true_var_oos = (
                    true_variance[self.window_size:] if true_variance is not None else None
                )
                mse_val = mse_loss(fb_valid, px_valid)
                qlike_val = qlike_loss(fb_valid, px_valid)
                mse_true = 0.0
                if true_var_oos is not None:
                    tv_valid = true_var_oos[valid_mask]
                    mse_true = mse_loss(fb_valid, tv_valid)
                mz = mincer_zarnowitz_test(fb_valid, px_valid)
                suite.results.append(BenchmarkResult(
                    model_name="GARCH(1,1)-fallback",
                    forecasts=fb_forecasts,
                    proxies=proxies,
                    true_variance=true_var_oos,
                    mse=mse_val, qlike=qlike_val, mse_vs_true=mse_true,
                    mz_alpha=mz.alpha, mz_beta=mz.beta,
                    mz_r2=mz.r_squared, mz_efficient=mz.efficient,
                ))
            else:
                logger.error(
                    "GARCH(1,1) fallback also failed (%d/%d valid). "
                    "No models available.", n_valid, n_oos,
                )

        return suite
