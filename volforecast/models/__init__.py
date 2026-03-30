"""
Volatility forecasting model implementations.

Model families:
- GARCH: ARCH, EWMA, GARCH(1,1), EGARCH, GJR-GARCH, APARCH, CGARCH,
         FIGARCH, HEAVY, Realized GARCH, GARCH-MIDAS
- HAR: HAR-RV, HAR-RV-J, HAR-RV-CJ, SHAR, HAR-IV
- SV: Stochastic Volatility, SV with Jumps
- GAS: Score-driven (GAS/DCS) volatility
- MS: Markov-switching volatility, MSGARCH
- Quantile: CAViaR
- ML: Random Forest, LSTM, Transformer, DeepVol
- Multivariate: DCC-GARCH, Copula-GARCH
- RoughVol: Rough Bergomi, Rough Heston
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
    HARIVForecaster,
)
from volforecast.models.realized_garch import RealizedGARCHForecaster
from volforecast.models.figarch import FIGARCHForecaster
from volforecast.models.heavy import HEAVYForecaster
from volforecast.models.sv import SVForecaster, SVJForecaster
from volforecast.models.gas import GASVolForecaster
from volforecast.models.markov_switching import MSVolForecaster, MSGARCHForecaster
from volforecast.models.midas import GARCHMIDASForecaster
from volforecast.models.caviar import CAViaRForecaster
from volforecast.models.ml_wrappers import RFVolForecaster

from volforecast.models.multivariate import DCCGARCHForecaster
from volforecast.models.copula_garch import CopulaGARCHForecaster
from volforecast.models.rough_vol import RoughBergomiForecaster, RoughHestonForecaster

# Optional PyTorch-dependent imports
try:
    from volforecast.models.ml_wrappers import LSTMVolForecaster, TransformerVolForecaster
    from volforecast.models.deep_vol import DeepVolForecaster
except ImportError:
    pass
