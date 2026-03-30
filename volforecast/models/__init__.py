"""
Volatility forecasting model implementations.

Model families:
- GARCH: ARCH, EWMA, GARCH(1,1), EGARCH, GJR-GARCH, APARCH, CGARCH,
         FIGARCH, HEAVY, Realized GARCH, GARCH-MIDAS
- HAR: HAR-RV, HAR-RV-J, HAR-RV-CJ, SHAR
- SV: Stochastic Volatility, SV with Jumps
- GAS: Score-driven (GAS/DCS) volatility
- MS: Markov-switching volatility
- Quantile: CAViaR
- ML: Random Forest, LSTM, Transformer wrappers
"""

from volforecast.models.garch import (
    GARCHForecaster,
    EGARCHForecaster,
    GJRGARCHForecaster,
    APARCHForecaster,
    CGARCHForecaster,
    ARCHForecaster,
    EWMAForecaster,
)
from volforecast.models.har import (
    HARForecaster,
    HARJForecaster,
    HARCJForecaster,
    SHARForecaster,
)
from volforecast.models.realized_garch import RealizedGARCHForecaster
from volforecast.models.figarch import FIGARCHForecaster
from volforecast.models.heavy import HEAVYForecaster
from volforecast.models.sv import SVForecaster, SVJForecaster
from volforecast.models.gas import GASVolForecaster
from volforecast.models.markov_switching import MSVolForecaster
from volforecast.models.midas import GARCHMIDASForecaster
from volforecast.models.caviar import CAViaRForecaster
from volforecast.models.ml_wrappers import RFVolForecaster

# Optional PyTorch-dependent imports
try:
    from volforecast.models.ml_wrappers import LSTMVolForecaster, TransformerVolForecaster
except ImportError:
    pass
