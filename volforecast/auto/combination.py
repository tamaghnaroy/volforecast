"""
CombinedForecaster for AutoVolForecaster (Phase 5).

Wraps MCS survivors into an online combiner, implementing BaseForecaster.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from volforecast.core.base import BaseForecaster, ForecastResult, ModelSpec
from volforecast.core.targets import VolatilityTarget, TargetSpec, COND_VAR_1STEP
from volforecast.combination.online import (
    BaseCombiner,
    EqualWeightCombiner,
    EWACombiner,
    FixedShareCombiner,
    AFTERCombiner,
)

logger = logging.getLogger(__name__)


class CombinedForecaster(BaseForecaster):
    """Online combined forecaster wrapping multiple BaseForecaster experts.

    Implements the BaseForecaster protocol so the combined model can be
    used interchangeably with any single model.
    """

    def __init__(
        self,
        components: list[BaseForecaster],
        combiner: BaseCombiner,
        combiner_name: str = "EWA",
    ) -> None:
        self._components = list(components)
        self._combiner = combiner
        self._combiner_name = combiner_name
        self._fitted = False
        self._last_forecasts: Optional[NDArray[np.float64]] = None

        names = ", ".join(c.model_spec.name for c in self._components)
        self._model_spec = ModelSpec(
            name=f"Auto[{names}]/{combiner_name}",
            abbreviation="AUTO",
            family="COMBO",
            target=VolatilityTarget.CONDITIONAL_VARIANCE,
            reference="Timmermann (2006)",
        )

    @property
    def model_spec(self) -> ModelSpec:
        return self._model_spec

    @property
    def weights(self) -> NDArray[np.float64]:
        return self._combiner.weights

    @property
    def components(self) -> list[BaseForecaster]:
        return list(self._components)

    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> CombinedForecaster:
        """Fit all component models and initialise combiner weights equally."""
        for comp in self._components:
            try:
                comp.fit(returns, realized_measures, **kwargs)
            except Exception as e:
                logger.warning("Component %s fit failed: %s", comp.model_spec.name, e)

        # Reset combiner weights to equal
        K = len(self._components)
        self._combiner._weights = np.ones(K, dtype=np.float64) / K
        self._combiner._t = 0
        self._fitted = True
        return self

    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        """Produce combined forecast.

        In v1, only horizon=1 is supported.
        """
        if horizon != 1:
            raise NotImplementedError(
                f"CombinedForecaster v1 supports horizon=1 only, got {horizon}"
            )
        if not self._fitted:
            raise RuntimeError("CombinedForecaster not fitted. Call fit() first.")

        # Get forecasts from all components
        expert_forecasts = np.empty(len(self._components), dtype=np.float64)
        for i, comp in enumerate(self._components):
            try:
                res = comp.predict(horizon=1)
                expert_forecasts[i] = res.point[0]
            except Exception as e:
                logger.warning(
                    "Component %s predict failed: %s. Using NaN.",
                    comp.model_spec.name, e,
                )
                expert_forecasts[i] = np.nan

        # Replace NaN with mean of valid forecasts
        valid = ~np.isnan(expert_forecasts)
        if valid.any():
            fill = np.mean(expert_forecasts[valid])
            expert_forecasts[~valid] = fill
        else:
            expert_forecasts[:] = 0.0

        # Store for use in update()
        self._last_forecasts = expert_forecasts.copy()

        # Combine
        combined = self._combiner.combine(expert_forecasts)

        return ForecastResult(
            point=np.array([combined], dtype=np.float64),
            target_spec=COND_VAR_1STEP,
            model_name=self._model_spec.name,
        )

    def update(
        self,
        new_returns: NDArray[np.float64],
        new_realized: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> None:
        """Online update: score pre-update forecasts, then update components.

        The ordering is critical to avoid look-ahead contamination:
        1. Use the forecasts captured during the last predict() call.
        2. Extract the realization.
        3. Update combiner weights.
        4. Update component models.
        """
        # Step 1: Get pre-update forecasts (from last predict call)
        if self._last_forecasts is None:
            # No prior predict — get forecasts now before updating
            expert_forecasts = np.empty(len(self._components), dtype=np.float64)
            for i, comp in enumerate(self._components):
                try:
                    res = comp.predict(horizon=1)
                    expert_forecasts[i] = res.point[0]
                except Exception:
                    expert_forecasts[i] = np.nan
            valid = ~np.isnan(expert_forecasts)
            if valid.any():
                expert_forecasts[~valid] = np.mean(expert_forecasts[valid])
            self._last_forecasts = expert_forecasts

        forecasts_t = self._last_forecasts

        # Step 2: Extract realization
        if new_realized is not None and "RV" in new_realized:
            realization = float(new_realized["RV"][-1])
        else:
            # Fallback: squared return
            realization = float(new_returns[-1] ** 2)

        # Step 3: Update combiner weights
        self._combiner.update(forecasts_t, realization)

        # Step 4: Update component models
        for comp in self._components:
            try:
                comp.update(new_returns, new_realized)
            except Exception as e:
                logger.warning(
                    "Component %s update failed: %s", comp.model_spec.name, e
                )

        # Clear cached forecasts
        self._last_forecasts = None


def select_combiner(
    n_experts: int,
    T: int,
    has_regime_switching: bool,
    combination_method: str = "auto",
) -> tuple[BaseCombiner, str]:
    """Select the appropriate combiner based on data characteristics.

    Parameters
    ----------
    n_experts : int
    T : int
        Series length.
    has_regime_switching : bool
    combination_method : str
        "auto", "ewa", "fixed_share", "after", "equal".

    Returns
    -------
    (combiner, name)
    """
    if combination_method != "auto":
        mapping = {
            "ewa": lambda: (EWACombiner(n_experts, loss_fn="QLIKE"), "EWA"),
            "fixed_share": lambda: (
                FixedShareCombiner(n_experts, loss_fn="QLIKE"), "FixedShare"
            ),
            "after": lambda: (AFTERCombiner(n_experts, loss_fn="QLIKE"), "AFTER"),
            "equal": lambda: (
                EqualWeightCombiner(n_experts, loss_fn="QLIKE"), "EqualWeight"
            ),
        }
        if combination_method in mapping:
            return mapping[combination_method]()

    # Auto selection heuristic
    if n_experts == 1:
        return EqualWeightCombiner(1, loss_fn="QLIKE"), "Passthrough"

    if T < 750:
        return EqualWeightCombiner(n_experts, loss_fn="QLIKE"), "EqualWeight"

    if has_regime_switching:
        return FixedShareCombiner(n_experts, loss_fn="QLIKE"), "FixedShare"

    return EWACombiner(n_experts, loss_fn="QLIKE"), "EWA"
