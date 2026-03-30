"""
AutoVolForecaster — automated volatility model selection and combination.

Given a return series (and optionally intraday data), automatically profiles
the data, selects candidate models, benchmarks them, prunes via MCS, and
combines survivors into a single online-updatable forecaster.
"""

from volforecast.auto.profiler import DataProfiler, DataProfile
from volforecast.auto.selector import CandidateSelector
from volforecast.auto.model_selection import ModelSelector, ModelSelectionResult
from volforecast.auto.combination import CombinedForecaster
from volforecast.auto.auto import AutoVolForecaster, AutoForecastResult, auto_fit

__all__ = [
    "AutoVolForecaster",
    "AutoForecastResult",
    "auto_fit",
    "DataProfiler",
    "DataProfile",
    "CandidateSelector",
    "ModelSelector",
    "ModelSelectionResult",
    "CombinedForecaster",
]
