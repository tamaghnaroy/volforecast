"""
Forecast combination and online aggregation.

Methods:
- Equal Weight (Bates & Granger, 1969)
- Inverse MSE weighting (Stock & Watson, 2004)
- AFTER: Aggregated Forecast Through Exponential Re-weighting (Yang, 2004)
- EWA: Exponentially Weighted Average (Vovk, 1990)
- Fixed-Share expert aggregation (Herbster & Warmuth, 1998)
- RL-based adaptive combination (PPO/DQN)
"""

from volforecast.combination.online import (
    EqualWeightCombiner,
    InverseMSECombiner,
    AFTERCombiner,
    EWACombiner,
    FixedShareCombiner,
)
from volforecast.combination.rl_combiner import RLCombiner
