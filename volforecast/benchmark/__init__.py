"""
Benchmark suite for reproducible volatility forecasting experiments.

Provides:
- Synthetic data generation (GARCH-DGP, jump-diffusion, SV)
- Standard benchmark protocols (expanding/rolling window)
- Result collection and comparison tables
"""

from volforecast.benchmark.synthetic import (
    generate_garch_data,
    generate_jump_diffusion_data,
    generate_sv_data,
)
from volforecast.benchmark.runner import BenchmarkRunner
