"""
AutoVolForecaster — main entry point (Phase 6).

Assembles the full pipeline: profile → select → benchmark → prune → combine.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from volforecast.core.base import BaseForecaster, ForecastResult, ModelSpec
from volforecast.auto.profiler import DataProfiler, DataProfile
from volforecast.auto.selector import CandidateSelector
from volforecast.auto.model_selection import ModelSelector, ModelSelectionResult
from volforecast.auto.combination import CombinedForecaster, select_combiner
from volforecast.benchmark.runner import BenchmarkRunner

logger = logging.getLogger(__name__)


def _capture_provenance(config: dict) -> dict:
    """Capture full experiment provenance: git, environment, config."""
    prov: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cli_args": sys.argv,
        "config": config,
    }
    try:
        import volforecast
        prov["volforecast_version"] = volforecast.__version__
    except Exception:
        pass
    try:
        import scipy
        prov["scipy"] = scipy.__version__
    except Exception:
        pass
    # Git commit + dirty status
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        prov["git_commit"] = git_hash
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        prov["git_dirty"] = len(dirty) > 0
    except Exception:
        prov["git_commit"] = "unknown"
        prov["git_dirty"] = None
    return prov


@dataclass
class AutoForecastResult:
    """Result of AutoVolForecaster.fit().

    This is the **single source of truth** for the experiment.
    All analysis scripts and README generation should read from
    `to_dict()` / `save()` rather than re-deriving results.
    """
    forecaster: BaseForecaster
    profile: DataProfile
    selection: ModelSelectionResult
    combiner_name: str
    component_models: list[str]
    initial_weights: NDArray[np.float64]
    benchmark_summary: str
    proxy_quality: str
    warnings: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dictionary (the canonical result object)."""
        sel = self.selection

        def _jsonify(v):
            """Convert numpy scalars to Python natives for JSON."""
            if isinstance(v, (np.bool_, np.generic)):
                return v.item()
            return v

        p = self.profile
        return {
            "provenance": self.provenance,
            "profile": {
                "T": int(p.T),
                "hurst_exp": float(p.hurst_exp),
                "has_long_memory": bool(p.has_long_memory),
                "has_rough_vol": bool(p.has_rough_vol),
                "has_leverage": bool(p.has_leverage),
                "has_jumps": bool(p.has_jumps),
                "jump_fraction": float(p.jump_fraction),
                "excess_kurtosis": float(p.excess_kurtosis),
                "heavy_tails": bool(p.heavy_tails),
                "has_regime_switching": bool(p.has_regime_switching),
                "has_intraday": bool(p.has_intraday),
                "has_realized": bool(p.has_realized),
            },
            "selection": {
                "primary_loss": str(sel.primary_loss),
                "proxy_quality": str(sel.proxy_quality),
                "mcs_survivors": [s.model_name for s in sel.mcs_survivors],
                "eliminated": [e.model_name for e in sel.eliminated],
                "mz_flags": {k: bool(v) for k, v in sel.mz_flags.items()},
                "candidate_metrics": [
                    {
                        "model": r.model_name,
                        "mse": float(r.mse),
                        "qlike": float(r.qlike),
                        "mz_r2": float(r.mz_r2),
                        "mz_efficient": bool(r.mz_efficient),
                    }
                    for r in sel.mcs_survivors + sel.eliminated
                ],
            },
            "combination": {
                "combiner_name": str(self.combiner_name),
                "component_models": list(self.component_models),
                "initial_weights": [float(w) for w in self.initial_weights],
            },
            "benchmark_summary": str(self.benchmark_summary),
            "proxy_quality": str(self.proxy_quality),
            "warnings": [str(w) for w in self.warnings],
        }

    def save(self, path: str) -> None:
        """Save canonical result to JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Saved canonical result to %s", path)

    @staticmethod
    def load_summary(path: str) -> dict:
        """Load a previously saved canonical result."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


class AutoVolForecaster:
    """Automated volatility model selection and combination.

    Given a return series (and optionally intraday data), automatically:
    1. Profiles the data characteristics
    2. Selects candidate model families
    3. Benchmarks all candidates with expanding-window OOS evaluation
    4. Prunes dominated models via DM tests and MCS
    5. Combines survivors with an online combiner
    6. Returns a production-ready BaseForecaster-compatible object

    Parameters
    ----------
    model_families : list of str, optional
        Restrict to these families. None = auto-select.
    combination_method : str
        "auto", "ewa", "fixed_share", "after", "equal".
    loss_fn : str
        Primary ranking loss: "QLIKE" (default) or "MSE".
    window_type : str
        "expanding" (default) or "rolling".
    min_train : int, optional
        Minimum training window. None = max(252, T//3).
    refit_every : int
        Re-estimate every N steps during benchmark.
    mcs_alpha : float
        MCS significance level.
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        model_families: Optional[list[str]] = None,
        combination_method: str = "auto",
        loss_fn: str = "QLIKE",
        window_type: str = "expanding",
        min_train: Optional[int] = None,
        refit_every: int = 21,
        mcs_alpha: float = 0.10,
        random_state: Optional[int] = None,
    ) -> None:
        self.model_families = model_families
        self.combination_method = combination_method
        self.loss_fn = loss_fn
        self.window_type = window_type
        self.min_train = min_train
        self.refit_every = refit_every
        self.mcs_alpha = mcs_alpha
        self.random_state = random_state

        self._result: Optional[AutoForecastResult] = None
        self._forecaster: Optional[CombinedForecaster] = None

    def fit(
        self,
        returns: NDArray[np.float64],
        intraday_returns: Optional[NDArray[np.float64]] = None,
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
    ) -> AutoForecastResult:
        """Run the full auto-selection pipeline.

        Parameters
        ----------
        returns : array, shape (T,)
            Daily log-returns.
        intraday_returns : array, shape (T, n_intraday), optional
        realized_measures : dict, optional
            Pre-computed realized measures (must contain at least "RV").

        Returns
        -------
        AutoForecastResult
        """
        returns = np.asarray(returns, dtype=np.float64)
        T = len(returns)
        warnings_list: list[str] = []

        logger.info("AutoVolForecaster.fit() on T=%d series", T)

        # ── Phase 1: Profile ──
        logger.info("Phase 1: Profiling data...")
        profile = DataProfiler.profile(returns, intraday_returns, realized_measures)
        logger.info(
            "Profile: T=%d, H=%.3f, leverage=%s, jumps=%s (%.2f), "
            "regime=%s, heavy_tails=%s",
            profile.T, profile.hurst_exp, profile.has_leverage,
            profile.has_jumps, profile.jump_fraction,
            profile.has_regime_switching, profile.heavy_tails,
        )

        # ── Phase 2: Select candidates ──
        logger.info("Phase 2: Selecting candidates...")
        candidates = CandidateSelector.select(profile, self.model_families)
        logger.info("Selected %d candidates", len(candidates))

        # ── Phase 3: Benchmark ──
        logger.info("Phase 3: Running benchmark...")
        window_size = self.min_train or max(252, T // 3)
        runner = BenchmarkRunner(
            forecasters=candidates,
            window_type=self.window_type,
            window_size=window_size,
            refit_every=self.refit_every,
        )

        suite = runner.run(
            daily_returns=returns,
            intraday_returns=intraday_returns,
            precomputed_realized=realized_measures,
        )

        if not suite.results:
            raise RuntimeError(
                "All models including GARCH fallback failed. "
                "Cannot proceed with model selection. "
                f"T={T}, window_size={window_size}, "
                f"n_candidates={len(candidates)}"
            )

        benchmark_summary = suite.summary_table(sort_by="qlike")
        logger.info("Benchmark complete: %d models survived", len(suite.results))

        # ── Phase 4: Model selection ──
        logger.info("Phase 4: Statistical model selection...")
        selector = ModelSelector(mcs_alpha=self.mcs_alpha, loss_fn=self.loss_fn)
        selection = selector.select(suite)
        warnings_list.extend(selection.warnings)
        logger.info(
            "Selection: %d MCS survivors, loss=%s, proxy=%s",
            len(selection.mcs_survivors), selection.primary_loss,
            selection.proxy_quality,
        )

        # ── Phase 5: Combination ──
        logger.info("Phase 5: Building combined forecaster...")
        survivors = selection.mcs_survivors
        n_experts = len(survivors)

        # Re-fit survivor models on the full training data
        # (benchmark only used expanding window; now use full series)
        survivor_forecasters: list[BaseForecaster] = []
        for sr in survivors:
            # Find the original candidate by name
            for cand in candidates:
                if cand.model_spec.name == sr.model_name:
                    try:
                        if realized_measures is not None:
                            cand.fit(returns, realized_measures)
                        elif intraday_returns is not None:
                            # Build realized measures from intraday
                            rm = {}
                            if profile.rv is not None:
                                rm["RV"] = profile.rv
                            if profile.bv is not None:
                                rm["BV"] = profile.bv
                            if profile.jv is not None:
                                rm["JV"] = profile.jv
                            if profile.cv is not None:
                                rm["CV"] = profile.cv
                            cand.fit(returns, rm if rm else None)
                        else:
                            cand.fit(returns)
                        survivor_forecasters.append(cand)
                    except Exception as e:
                        logger.warning(
                            "Final fit of %s failed: %s", sr.model_name, e
                        )
                        warnings_list.append(
                            f"Final fit of {sr.model_name} failed: {e}"
                        )
                    break

        if not survivor_forecasters:
            raise RuntimeError(
                "All MCS survivors failed final fit. Cannot build combined forecaster."
            )

        combiner, combiner_name = select_combiner(
            n_experts=len(survivor_forecasters),
            T=T,
            has_regime_switching=profile.has_regime_switching,
            combination_method=self.combination_method,
        )

        combined = CombinedForecaster(
            components=survivor_forecasters,
            combiner=combiner,
            combiner_name=combiner_name,
        )
        combined._fitted = True

        self._forecaster = combined
        # Capture provenance
        provenance = _capture_provenance({
            "model_families": self.model_families,
            "combination_method": self.combination_method,
            "loss_fn": self.loss_fn,
            "window_type": self.window_type,
            "min_train": self.min_train,
            "refit_every": self.refit_every,
            "mcs_alpha": self.mcs_alpha,
            "random_state": self.random_state,
            "T": T,
            "window_size_used": window_size,
        })

        self._result = AutoForecastResult(
            forecaster=combined,
            profile=profile,
            selection=selection,
            combiner_name=combiner_name,
            component_models=[c.model_spec.name for c in survivor_forecasters],
            initial_weights=combiner.weights,
            benchmark_summary=benchmark_summary,
            proxy_quality=selection.proxy_quality,
            warnings=warnings_list,
            provenance=provenance,
        )

        logger.info(
            "AutoVolForecaster ready: %d components, combiner=%s",
            len(survivor_forecasters), combiner_name,
        )
        return self._result

    def predict(self, horizon: int = 1) -> ForecastResult:
        """Produce combined forecast. Delegates to CombinedForecaster."""
        if self._forecaster is None:
            raise RuntimeError("Not fitted. Call fit() first.")
        return self._forecaster.predict(horizon=horizon)

    def update(
        self,
        new_returns: NDArray[np.float64],
        new_realized: Optional[dict[str, NDArray[np.float64]]] = None,
    ) -> None:
        """Online update. Delegates to CombinedForecaster."""
        if self._forecaster is None:
            raise RuntimeError("Not fitted. Call fit() first.")
        self._forecaster.update(new_returns, new_realized)

    @property
    def result(self) -> Optional[AutoForecastResult]:
        return self._result


def auto_fit(
    returns: NDArray[np.float64],
    intraday_returns: Optional[NDArray[np.float64]] = None,
    **kwargs: Any,
) -> AutoForecastResult:
    """Convenience function: create AutoVolForecaster, fit, and return result.

    Parameters
    ----------
    returns : array, shape (T,)
    intraday_returns : array, optional
    **kwargs
        Passed to AutoVolForecaster.__init__.

    Returns
    -------
    AutoForecastResult
    """
    avf = AutoVolForecaster(**kwargs)
    return avf.fit(returns, intraday_returns)
