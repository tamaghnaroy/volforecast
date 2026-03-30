"""
Academically grounded volatility forecast evaluation.

Key features:
- Loss functions robust to proxy noise (Patton, 2011)
- Diebold-Mariano test with HAC standard errors
- Model Confidence Set (Hansen, Lunde, Nason, 2011)
- Mincer-Zarnowitz regressions
- Explicit proxy-target mismatch correction
"""

from volforecast.evaluation.losses import (
    mse_loss,
    qlike_loss,
    mae_loss,
    mse_log_loss,
    patton_robust_loss,
    heterogeneous_loss,
)
from volforecast.evaluation.tests import (
    diebold_mariano_test,
    mincer_zarnowitz_test,
    model_confidence_set,
    hit_rate_test,
    dq_test,
)
from volforecast.evaluation.proxy import (
    proxy_noise_correction,
    attenuation_bias_correction,
    hansen_lunde_adjustment,
)
from volforecast.evaluation.conformal import (
    SplitConformalVol,
    OnlineConformalVol,
    coverage_diagnostic,
)
