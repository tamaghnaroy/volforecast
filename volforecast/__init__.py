"""
VolForecast: Research-grade volatility forecasting library.

Modular experts with shared interface, Numba-optimized realized measures,
online aggregation, adaptive expert weighting, RL-based forecast combination,
and academically grounded evaluation correcting for forecast-target mismatch.
"""

__version__ = "0.1.0"

from volforecast.core.targets import VolatilityTarget, TargetSpec
from volforecast.core.base import BaseForecaster, ForecastResult, ModelSpec

# Convenience re-exports for top-level access
from volforecast import models
from volforecast import realized
from volforecast import combination
from volforecast import evaluation
from volforecast import knowledge
from volforecast import benchmark
