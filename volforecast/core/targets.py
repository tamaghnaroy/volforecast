"""
Volatility target taxonomy.

Distinguishes the latent objects that forecasters may target:
- CONDITIONAL_VARIANCE: E[r_t^2 | F_{t-1}], the GARCH-type object
- INTEGRATED_VARIANCE: IV_t = int_0^1 sigma_s^2 ds, continuous-time total
- CONTINUOUS_VARIATION: C_t = IV_t without jump component (QV - JV)
- JUMP_VARIATION: J_t = sum of squared jumps in [t-1, t]
- QUADRATIC_VARIATION: QV_t = C_t + J_t, total variation including jumps

The forecast-target mismatch problem (Patton, 2011; Hansen & Lunde, 2006):
  When the forecast target sigma^2_* differs from the proxy hat{sigma}^2,
  ranking of forecasters can be distorted under standard loss functions.
  Only the MSE and QLIKE loss families are *robust* to noise in the proxy
  (Patton, 2011, Theorem 1).
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional


class VolatilityTarget(Enum):
    """Enumeration of distinct volatility targets."""
    CONDITIONAL_VARIANCE = auto()
    INTEGRATED_VARIANCE = auto()
    CONTINUOUS_VARIATION = auto()
    JUMP_VARIATION = auto()
    QUADRATIC_VARIATION = auto()


@dataclass(frozen=True)
class TargetSpec:
    """Specification of what a forecaster targets and how it relates to proxies.

    Attributes
    ----------
    target : VolatilityTarget
        The latent object being forecast.
    horizon : int
        Forecast horizon in periods (1 = one-step-ahead).
    proxy_description : str
        Description of recommended proxy for evaluation.
    robust_losses : tuple[str, ...]
        Loss functions robust to proxy noise for this target (Patton, 2011).
    """
    target: VolatilityTarget
    horizon: int = 1
    proxy_description: str = "RV_t (realized variance)"
    robust_losses: tuple[str, ...] = ("MSE", "QLIKE")
    notes: Optional[str] = None


# Pre-built target specs for common use cases
COND_VAR_1STEP = TargetSpec(
    target=VolatilityTarget.CONDITIONAL_VARIANCE,
    horizon=1,
    proxy_description="RV_t or squared return (noisy)",
    robust_losses=("MSE", "QLIKE"),
    notes="GARCH-family models target this. Proxy mismatch with RV is first-order."
)

IV_1STEP = TargetSpec(
    target=VolatilityTarget.INTEGRATED_VARIANCE,
    horizon=1,
    proxy_description="RV_t (Barndorff-Nielsen & Shephard, 2002)",
    robust_losses=("MSE", "QLIKE"),
    notes="HAR-RV targets this. RV is consistent but noisy proxy."
)

CV_1STEP = TargetSpec(
    target=VolatilityTarget.CONTINUOUS_VARIATION,
    horizon=1,
    proxy_description="BV_t or MedRV_t (jump-robust)",
    robust_losses=("MSE", "QLIKE"),
    notes="HAR-RV-CJ continuous component. Use BV or MedRV as proxy."
)

JV_1STEP = TargetSpec(
    target=VolatilityTarget.JUMP_VARIATION,
    horizon=1,
    proxy_description="max(RV_t - BV_t, 0) or J_t from jump test",
    robust_losses=("MSE",),
    notes="Jump variation proxy is inherently noisy; threshold-based."
)
