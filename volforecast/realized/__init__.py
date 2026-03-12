"""
Realized volatility measures and jump decomposition.

Numba-optimized implementations of:
- Realized Variance (RV)
- Bipower Variation (BV)
- Median Realized Variance (MedRV)
- MinRV
- Realized Kernel (RK) with Parzen kernel
- Two-Scale Realized Variance (TSRV)
- Pre-Averaging Estimator
- Jump tests (BNS, AJ, CPR)
- Signed jump variation decomposition
"""

from volforecast.realized.measures import (
    realized_variance,
    realized_variance_series,
    bipower_variation,
    bipower_variation_series,
    median_rv,
    min_rv,
    realized_kernel,
    tsrv,
    pre_averaging,
    realized_semivariance,
    realized_semivariance_series,
)
from volforecast.realized.jumps import (
    bns_jump_test,
    jump_variation,
    continuous_variation,
    jump_decomposition,
)
