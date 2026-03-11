"""
Volatility forecasting model implementations.

Model families:
- GARCH: GARCH(1,1), EGARCH, GJR-GARCH, APARCH, CGARCH, HEAVY, Realized GARCH
- HAR: HAR-RV, HAR-RV-J, HAR-RV-CJ, SHAR
- SV: Stochastic Volatility (quasi-likelihood)
- ML: LSTM, Random Forest wrappers
"""

from volforecast.models.garch import (
    GARCHForecaster,
    EGARCHForecaster,
    GJRGARCHForecaster,
    APARCHForecaster,
    CGARCHForecaster,
)
from volforecast.models.har import (
    HARForecaster,
    HARJForecaster,
    HARCJForecaster,
    SHARForecaster,
)
from volforecast.models.realized_garch import RealizedGARCHForecaster
