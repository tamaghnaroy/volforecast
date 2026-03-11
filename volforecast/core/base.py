"""
Base forecaster interface.

All volatility forecasting models implement BaseForecaster, ensuring:
- Explicit declaration of target variable
- Standardized fit/predict/update API
- Online (streaming) update capability
- Metadata for knowledge graph integration
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from volforecast.core.targets import VolatilityTarget, TargetSpec


@dataclass
class ForecastResult:
    """Container for forecast output.

    Attributes
    ----------
    point : NDArray[np.float64]
        Point forecasts, shape (n_horizons,) or (n_periods, n_horizons).
    variance : Optional[NDArray[np.float64]]
        Forecast variance / uncertainty, same shape as point.
    target_spec : TargetSpec
        What latent object these forecasts target.
    model_name : str
        Identifier of the model that produced these forecasts.
    metadata : dict
        Additional info (e.g., parameters, convergence).
    """
    point: NDArray[np.float64]
    target_spec: TargetSpec
    model_name: str
    variance: Optional[NDArray[np.float64]] = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSpec:
    """Static metadata for knowledge-graph integration.

    Attributes
    ----------
    name : str
        Full model name (e.g., "GJR-GARCH(1,1)").
    abbreviation : str
        Short identifier (e.g., "GJR").
    family : str
        Model family (e.g., "GARCH", "HAR", "SV", "ML", "COMBO").
    target : VolatilityTarget
        Primary target variable.
    assumptions : tuple[str, ...]
        Key model assumptions.
    complexity : str
        Computational complexity description.
    reference : str
        Seminal reference.
    extends : tuple[str, ...]
        Models this extends (for knowledge graph edges).
    """
    name: str
    abbreviation: str
    family: str
    target: VolatilityTarget
    assumptions: tuple[str, ...] = ()
    complexity: str = "O(T)"
    reference: str = ""
    extends: tuple[str, ...] = ()


class BaseForecaster(abc.ABC):
    """Abstract base class for all volatility forecasters.

    Subclasses must implement:
    - model_spec (property): static model metadata
    - fit(): estimate parameters from data
    - predict(): produce forecasts
    - update(): online update with new observation(s)
    """

    @property
    @abc.abstractmethod
    def model_spec(self) -> ModelSpec:
        """Return static model specification for knowledge graph."""
        ...

    @abc.abstractmethod
    def fit(
        self,
        returns: NDArray[np.float64],
        realized_measures: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> "BaseForecaster":
        """Fit model parameters.

        Parameters
        ----------
        returns : array, shape (T,)
            Return series (log returns or simple returns).
        realized_measures : dict, optional
            Dictionary of realized measures keyed by name
            (e.g., {"RV": rv_array, "BV": bv_array}).
        **kwargs
            Model-specific options.

        Returns
        -------
        self
        """
        ...

    @abc.abstractmethod
    def predict(self, horizon: int = 1, **kwargs: Any) -> ForecastResult:
        """Produce volatility forecasts.

        Parameters
        ----------
        horizon : int
            Number of steps ahead.
        **kwargs
            Model-specific prediction options.

        Returns
        -------
        ForecastResult
        """
        ...

    @abc.abstractmethod
    def update(
        self,
        new_returns: NDArray[np.float64],
        new_realized: Optional[dict[str, NDArray[np.float64]]] = None,
        **kwargs: Any,
    ) -> None:
        """Online update with new observation(s).

        Parameters
        ----------
        new_returns : array, shape (n,)
            New return observations.
        new_realized : dict, optional
            New realized measure observations.
        """
        ...

    def get_params(self) -> dict[str, Any]:
        """Return estimated parameters as a dictionary."""
        return {}

    def __repr__(self) -> str:
        spec = self.model_spec
        return f"{spec.name} [{spec.family}] -> {spec.target.name}"
