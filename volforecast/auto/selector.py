"""
Candidate selector for AutoVolForecaster (Phase 2).

Maps a DataProfile to an ordered list of BaseForecaster instances.
"""

from __future__ import annotations

import logging
from typing import Optional

from volforecast.auto.profiler import DataProfile
from volforecast.core.base import BaseForecaster

logger = logging.getLogger(__name__)


class CandidateSelector:
    """Select candidate forecasters based on a DataProfile."""

    @staticmethod
    def select(
        profile: DataProfile,
        model_families: Optional[list[str]] = None,
    ) -> list[BaseForecaster]:
        """Return instantiated forecasters appropriate for the data.

        Parameters
        ----------
        profile : DataProfile
            Result of DataProfiler.profile().
        model_families : list of str, optional
            If provided, restrict to these families (e.g., ["GARCH", "HAR"]).
            None = auto-select all applicable families.

        Returns
        -------
        list[BaseForecaster]
        """
        from volforecast.models import (
            GARCHForecaster,
            EWMAForecaster,
            ARCHForecaster,
            EGARCHForecaster,
            GJRGARCHForecaster,
            APARCHForecaster,
            FIGARCHForecaster,
            CGARCHForecaster,
            HARForecaster,
            HARJForecaster,
            HARCJForecaster,
            SHARForecaster,
            HEAVYForecaster,
            RealizedGARCHForecaster,
            GARCHMIDASForecaster,
            SVForecaster,
            SVJForecaster,
            GASVolForecaster,
            MSVolForecaster,
            RFVolForecaster,
        )

        candidates: list[BaseForecaster] = []
        T = profile.T

        def _family_ok(family: str) -> bool:
            if model_families is None:
                return True
            return family in model_families

        # ── Always included (baseline) ──
        if _family_ok("GARCH"):
            candidates.append(GARCHForecaster())
            candidates.append(EWMAForecaster())
            candidates.append(ARCHForecaster())

        # ── Conditional on leverage ──
        if profile.has_leverage and _family_ok("GARCH"):
            candidates.append(EGARCHForecaster())
            candidates.append(GJRGARCHForecaster())
            candidates.append(APARCHForecaster())

        # ── Conditional on long memory (H > 0.6) ──
        if profile.has_long_memory and _family_ok("GARCH"):
            candidates.append(FIGARCHForecaster())
            candidates.append(CGARCHForecaster())

        # ── Conditional on roughness (H < 0.5 and T >= 500) ──
        if profile.has_rough_vol and T >= 500 and _family_ok("RoughVol"):
            try:
                from volforecast.models.rough_vol import (
                    RoughBergomiForecaster,
                    RoughHestonForecaster,
                )
                candidates.append(RoughBergomiForecaster())
                candidates.append(RoughHestonForecaster())
            except ImportError:
                logger.info("Rough-vol models not available, skipping.")

        # ── Conditional on intraday / realized data ──
        if (profile.has_intraday or profile.has_realized) and _family_ok("HAR"):
            candidates.append(HARForecaster())
            candidates.append(SHARForecaster())
            if profile.has_jumps:
                candidates.append(HARJForecaster())
                candidates.append(HARCJForecaster())
            candidates.append(HEAVYForecaster())
            candidates.append(RealizedGARCHForecaster())
            if T >= 500:
                candidates.append(GARCHMIDASForecaster())

        # ── Conditional on heavy tails / regime ──
        if profile.heavy_tails and T >= 500 and _family_ok("SV"):
            candidates.append(SVForecaster())
            if profile.has_jumps:
                candidates.append(SVJForecaster())

        if _family_ok("GAS"):
            candidates.append(GASVolForecaster())

        if profile.has_regime_switching and _family_ok("MS"):
            candidates.append(MSVolForecaster())

        # ── Conditional on T >= 1000 (ML models) ──
        if T >= 1000 and _family_ok("ML"):
            candidates.append(RFVolForecaster())
            try:
                from volforecast.models.ml_wrappers import (
                    LSTMVolForecaster,
                    TransformerVolForecaster,
                )
                candidates.append(LSTMVolForecaster())
                candidates.append(TransformerVolForecaster())
            except ImportError:
                logger.info("PyTorch ML models not available, skipping.")

        if not candidates:
            logger.warning("No candidates selected, falling back to GARCH baseline.")
            candidates.append(GARCHForecaster())

        logger.info(
            "Selected %d candidate models for T=%d", len(candidates), T
        )
        return candidates
