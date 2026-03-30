"""
Statistical model selection for AutoVolForecaster (Phase 4).

Prunes dominated models using DM tests and MCS, yielding survivors
for the combination layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from volforecast.benchmark.runner import BenchmarkResult, BenchmarkSuiteResult
from volforecast.evaluation.losses import mse_loss, qlike_loss
from volforecast.evaluation.tests import (
    diebold_mariano_test,
    model_confidence_set,
    mincer_zarnowitz_test,
)
from volforecast.evaluation.proxy import proxy_noise_correction

logger = logging.getLogger(__name__)


@dataclass
class ModelSelectionResult:
    """Result of statistical model selection."""
    mcs_survivors: list[BenchmarkResult]
    eliminated: list[BenchmarkResult]
    primary_loss: str                      # "QLIKE" or "MSE"
    proxy_quality: str
    mz_flags: dict[str, bool] = field(default_factory=dict)  # model_name -> MZ efficient?
    warnings: list[str] = field(default_factory=list)


class ModelSelector:
    """Select models via DM pruning and MCS."""

    def __init__(self, mcs_alpha: float = 0.10, loss_fn: str = "QLIKE") -> None:
        self.mcs_alpha = mcs_alpha
        self.loss_fn = loss_fn

    def select(
        self, suite: BenchmarkSuiteResult,
    ) -> ModelSelectionResult:
        """Run the full selection pipeline.

        Parameters
        ----------
        suite : BenchmarkSuiteResult
            Results from BenchmarkRunner.run().

        Returns
        -------
        ModelSelectionResult
        """
        results = suite.results
        warnings: list[str] = []

        if not results:
            return ModelSelectionResult(
                mcs_survivors=[],
                eliminated=[],
                primary_loss=self.loss_fn,
                proxy_quality="unknown",
                warnings=["No benchmark results to select from."],
            )

        # ── Step 0: Proxy quality check → effective loss ──
        effective_loss = self.loss_fn
        proxy_quality = "good"

        # Use the first model's forecasts/proxies to check proxy quality
        ref = results[0]
        try:
            pnc = proxy_noise_correction(ref.forecasts, ref.proxies, loss_fn="QLIKE")
            snr = pnc.get("signal_to_noise_ratio", 10.0)
            proxy_quality = pnc.get("proxy_quality", "good")
            if snr < 1.0:
                effective_loss = "MSE"
                proxy_quality = pnc.get("proxy_quality", "poor")
                warnings.append(
                    f"Proxy SNR={snr:.2f} < 1. Falling back to MSE for ranking."
                )
                logger.warning("Proxy quality poor (SNR=%.2f). Using MSE.", snr)
        except Exception as e:
            logger.warning("Proxy noise correction failed: %s", e)

        # ── Step 1: Rank by effective loss ──
        if effective_loss == "QLIKE":
            results_sorted = sorted(results, key=lambda r: r.qlike)
        else:
            results_sorted = sorted(results, key=lambda r: r.mse)

        if len(results_sorted) == 1:
            return ModelSelectionResult(
                mcs_survivors=results_sorted,
                eliminated=[],
                primary_loss=effective_loss,
                proxy_quality=proxy_quality,
                warnings=warnings,
            )

        # ── Step 2: Compute per-period losses ──
        n_oos = len(results_sorted[0].forecasts)
        per_period_losses: dict[str, NDArray[np.float64]] = {}
        for r in results_sorted:
            f = r.forecasts
            p = r.proxies
            if effective_loss == "QLIKE":
                f_safe = np.maximum(f, 1e-20)
                losses = p / f_safe + np.log(f_safe)
            else:
                losses = (f - p) ** 2
            per_period_losses[r.model_name] = losses

        # ── Step 3: DM pruning ──
        best = results_sorted[0]
        best_losses = per_period_losses[best.model_name]
        dm_survivors = [best]
        dm_eliminated = []

        for r in results_sorted[1:]:
            try:
                dm = diebold_mariano_test(
                    best_losses,
                    per_period_losses[r.model_name],
                    horizon=1,
                    significance=self.mcs_alpha,
                )
                if dm.p_value < self.mcs_alpha and dm.mean_loss_diff < 0:
                    dm_eliminated.append(r)
                    logger.info("DM eliminated %s (p=%.4f)", r.model_name, dm.p_value)
                else:
                    dm_survivors.append(r)
            except Exception as e:
                logger.warning("DM test failed for %s: %s. Keeping.", r.model_name, e)
                dm_survivors.append(r)

        # ── Step 4: MCS ──
        if len(dm_survivors) >= 2:
            loss_matrix = np.column_stack([
                per_period_losses[r.model_name] for r in dm_survivors
            ])
            try:
                mcs_result = model_confidence_set(
                    loss_matrix, alpha=self.mcs_alpha,
                )
                mcs_survivors = []
                mcs_eliminated_extra = []
                for i, r in enumerate(dm_survivors):
                    if i in mcs_result.included:
                        mcs_survivors.append(r)
                    else:
                        mcs_eliminated_extra.append(r)
                dm_eliminated.extend(mcs_eliminated_extra)
            except Exception as e:
                logger.warning("MCS failed: %s. Keeping DM survivors.", e)
                mcs_survivors = dm_survivors
        else:
            mcs_survivors = dm_survivors

        # ── Step 5: Fallback ──
        if not mcs_survivors:
            warnings.append("MCS eliminated all models. Keeping top-3 by loss.")
            mcs_survivors = results_sorted[:3]
            dm_eliminated = [r for r in results_sorted[3:]]

        # ── Step 6: MZ check ──
        mz_flags: dict[str, bool] = {}
        for r in mcs_survivors:
            try:
                mz = mincer_zarnowitz_test(r.forecasts, r.proxies)
                mz_flags[r.model_name] = mz.efficient
                if not mz.efficient:
                    warnings.append(
                        f"{r.model_name} is not MZ-efficient "
                        f"(alpha={mz.alpha:.4f}, beta={mz.beta:.4f})"
                    )
            except Exception as e:
                logger.warning("MZ test failed for %s: %s", r.model_name, e)
                mz_flags[r.model_name] = False

        logger.info(
            "Model selection: %d survivors from %d candidates",
            len(mcs_survivors), len(results),
        )

        return ModelSelectionResult(
            mcs_survivors=mcs_survivors,
            eliminated=dm_eliminated,
            primary_loss=effective_loss,
            proxy_quality=proxy_quality,
            mz_flags=mz_flags,
            warnings=warnings,
        )
