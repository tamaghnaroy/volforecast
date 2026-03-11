"""
Online forecast combination algorithms.

All combiners operate sequentially: at each step t they
1. Receive expert forecasts f_{1,t}, ..., f_{K,t}
2. Produce combined forecast hat{sigma}^2_t = sum w_{k,t} * f_{k,t}
3. Observe realization y_t (proxy for latent volatility)
4. Update weights w_{k,t+1}

References
----------
- Bates & Granger (1969). "The Combination of Forecasts." OR Quarterly.
- Stock & Watson (2004). "Combination forecasts of output growth..." JBES.
- Yang (2004). "Combining forecasting procedures..." Econometric Theory.
- Vovk (1990). "Aggregating strategies." COLT.
- Herbster & Warmuth (1998). "Tracking the Best Expert." Machine Learning.
- Cesa-Bianchi & Lugosi (2006). "Prediction, Learning, and Games." Cambridge.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from numba import njit


@dataclass
class CombinerState:
    """Tracks combiner internal state for diagnostics."""
    weights_history: list[NDArray[np.float64]] = field(default_factory=list)
    combined_history: list[float] = field(default_factory=list)
    loss_history: list[NDArray[np.float64]] = field(default_factory=list)
    cumulative_loss: Optional[NDArray[np.float64]] = None


class BaseCombiner(abc.ABC):
    """Abstract base for forecast combiners."""

    def __init__(self, n_experts: int, loss_fn: str = "MSE") -> None:
        self.n_experts = n_experts
        self.loss_fn = loss_fn
        self._weights = np.ones(n_experts, dtype=np.float64) / n_experts
        self._state = CombinerState()
        self._t = 0

    @property
    def weights(self) -> NDArray[np.float64]:
        return self._weights.copy()

    def combine(self, forecasts: NDArray[np.float64]) -> float:
        """Produce combined forecast from expert forecasts.

        Parameters
        ----------
        forecasts : array, shape (K,)
            Expert forecasts for current period.

        Returns
        -------
        float
            Combined forecast.
        """
        f = np.asarray(forecasts, dtype=np.float64)
        combined = float(np.dot(self._weights, f))
        self._state.combined_history.append(combined)
        self._state.weights_history.append(self._weights.copy())
        return combined

    @abc.abstractmethod
    def update(self, forecasts: NDArray[np.float64], realization: float) -> None:
        """Update weights after observing realization.

        Parameters
        ----------
        forecasts : array, shape (K,)
            Expert forecasts that were combined.
        realization : float
            Observed proxy value (e.g., realized variance).
        """
        ...

    def _compute_losses(
        self, forecasts: NDArray[np.float64], realization: float,
    ) -> NDArray[np.float64]:
        """Compute per-expert losses."""
        f = np.asarray(forecasts, dtype=np.float64)
        if self.loss_fn == "MSE":
            return (f - realization) ** 2
        elif self.loss_fn == "QLIKE":
            # QLIKE: y/f + log(f) (robust to proxy noise, Patton 2011)
            f_safe = np.maximum(f, 1e-20)
            return realization / f_safe + np.log(f_safe)
        elif self.loss_fn == "MAE":
            return np.abs(f - realization)
        else:
            return (f - realization) ** 2

    def get_state(self) -> CombinerState:
        return self._state


class EqualWeightCombiner(BaseCombiner):
    """Equal-weight forecast combination (Bates & Granger, 1969).

    w_k = 1/K for all k and all t. Surprisingly hard to beat in practice.
    """

    def __init__(self, n_experts: int, loss_fn: str = "MSE") -> None:
        super().__init__(n_experts, loss_fn)

    def update(self, forecasts: NDArray[np.float64], realization: float) -> None:
        losses = self._compute_losses(forecasts, realization)
        self._state.loss_history.append(losses)
        self._t += 1
        # Weights stay equal


class InverseMSECombiner(BaseCombiner):
    """Inverse-MSE weighting (Stock & Watson, 2004).

    w_k,t proportional to 1/L_k,t where L_k,t is the cumulative loss of expert k.
    Uses rolling window for non-stationary environments.

    Parameters
    ----------
    n_experts : int
    window : int or None
        Rolling window size. None uses expanding window.
    """

    def __init__(
        self, n_experts: int, window: Optional[int] = None, loss_fn: str = "MSE",
    ) -> None:
        super().__init__(n_experts, loss_fn)
        self.window = window
        self._all_losses: list[NDArray[np.float64]] = []

    def update(self, forecasts: NDArray[np.float64], realization: float) -> None:
        losses = self._compute_losses(forecasts, realization)
        self._all_losses.append(losses)
        self._state.loss_history.append(losses)
        self._t += 1

        # Compute cumulative/rolling loss per expert
        if self.window is not None and len(self._all_losses) > self.window:
            recent = self._all_losses[-self.window:]
        else:
            recent = self._all_losses

        cum_loss = np.mean(recent, axis=0)
        cum_loss = np.maximum(cum_loss, 1e-20)

        inv_loss = 1.0 / cum_loss
        self._weights = inv_loss / np.sum(inv_loss)


class AFTERCombiner(BaseCombiner):
    """AFTER: Aggregated Forecast Through Exponential Re-weighting (Yang, 2004).

    Uses AIC/BIC-inspired exponential weighting with automatic rate selection.

    w_k,t proportional to exp(-eta * sum_{s=1}^{t-1} l_k,s)

    where eta is chosen adaptively. Achieves minimax optimal combination rate.
    """

    def __init__(self, n_experts: int, loss_fn: str = "MSE") -> None:
        super().__init__(n_experts, loss_fn)
        self._cum_losses = np.zeros(n_experts, dtype=np.float64)

    def update(self, forecasts: NDArray[np.float64], realization: float) -> None:
        losses = self._compute_losses(forecasts, realization)
        self._cum_losses += losses
        self._state.loss_history.append(losses)
        self._t += 1

        if self._t < 2:
            return

        # Adaptive learning rate: eta = sqrt(2 * log(K) / t)
        K = self.n_experts
        eta = np.sqrt(2.0 * np.log(max(K, 2)) / self._t)

        # Exponential weights with numerical stability
        log_w = -eta * self._cum_losses
        log_w -= np.max(log_w)  # Shift for numerical stability
        w = np.exp(log_w)
        self._weights = w / np.sum(w)


@njit(cache=True)
def _ewa_update(
    weights: NDArray[np.float64],
    losses: NDArray[np.float64],
    eta: float,
) -> NDArray[np.float64]:
    """Numba-optimized EWA weight update."""
    K = weights.shape[0]
    log_w = np.empty(K, dtype=np.float64)
    for k in range(K):
        log_w[k] = np.log(max(weights[k], 1e-300)) - eta * losses[k]
    max_lw = log_w[0]
    for k in range(1, K):
        if log_w[k] > max_lw:
            max_lw = log_w[k]
    s = 0.0
    for k in range(K):
        log_w[k] -= max_lw
        log_w[k] = np.exp(log_w[k])
        s += log_w[k]
    for k in range(K):
        log_w[k] /= s
    return log_w


class EWACombiner(BaseCombiner):
    """Exponentially Weighted Average (Vovk, 1990).

    Online learning algorithm with regret bound O(sqrt(T log K)).

    w_{k,t+1} proportional to w_{k,t} * exp(-eta * l_{k,t})

    Parameters
    ----------
    n_experts : int
    eta : float or None
        Learning rate. If None, uses theoretically optimal rate.
    """

    def __init__(
        self, n_experts: int, eta: Optional[float] = None, loss_fn: str = "MSE",
    ) -> None:
        super().__init__(n_experts, loss_fn)
        self.eta = eta
        self._auto_eta = eta is None

    def update(self, forecasts: NDArray[np.float64], realization: float) -> None:
        losses = self._compute_losses(forecasts, realization)
        self._state.loss_history.append(losses)
        self._t += 1

        # Auto learning rate: eta = sqrt(8 * log(K) / T)
        if self._auto_eta:
            eta = np.sqrt(8.0 * np.log(max(self.n_experts, 2)) / max(self._t, 1))
        else:
            eta = self.eta

        self._weights = _ewa_update(self._weights, losses, eta)


@njit(cache=True)
def _fixed_share_update(
    weights: NDArray[np.float64],
    losses: NDArray[np.float64],
    eta: float,
    alpha: float,
) -> NDArray[np.float64]:
    """Numba-optimized Fixed-Share weight update."""
    K = weights.shape[0]

    # Step 1: EWA update
    log_w = np.empty(K, dtype=np.float64)
    for k in range(K):
        log_w[k] = np.log(max(weights[k], 1e-300)) - eta * losses[k]
    max_lw = log_w[0]
    for k in range(1, K):
        if log_w[k] > max_lw:
            max_lw = log_w[k]
    s = 0.0
    w_temp = np.empty(K, dtype=np.float64)
    for k in range(K):
        w_temp[k] = np.exp(log_w[k] - max_lw)
        s += w_temp[k]
    for k in range(K):
        w_temp[k] /= s

    # Step 2: Fixed-share mixing
    # w_new = (1 - alpha) * w_temp + alpha * (1/K) * sum(w_temp) = (1 - alpha) * w_temp + alpha/K
    pool = 0.0
    for k in range(K):
        pool += alpha * w_temp[k]
    w_new = np.empty(K, dtype=np.float64)
    for k in range(K):
        w_new[k] = (1.0 - alpha) * w_temp[k] + pool / K
    # Normalize
    s2 = 0.0
    for k in range(K):
        s2 += w_new[k]
    for k in range(K):
        w_new[k] /= s2
    return w_new


class FixedShareCombiner(BaseCombiner):
    """Fixed-Share Expert Aggregation (Herbster & Warmuth, 1998).

    Extends EWA by allowing expert switching: at each step, a fraction alpha
    of each expert's weight is redistributed uniformly. This tracks the best
    *sequence* of experts, not just the single best expert.

    Regret bound: O(sqrt(T * (m * log K + m * log(T/m)))) where m = number
    of expert switches.

    Parameters
    ----------
    n_experts : int
    alpha : float
        Share/mixing rate in (0, 1). Higher = more switching.
    eta : float or None
        Learning rate. None = auto.
    """

    def __init__(
        self,
        n_experts: int,
        alpha: float = 0.01,
        eta: Optional[float] = None,
        loss_fn: str = "MSE",
    ) -> None:
        super().__init__(n_experts, loss_fn)
        self.alpha = alpha
        self.eta = eta
        self._auto_eta = eta is None

    def update(self, forecasts: NDArray[np.float64], realization: float) -> None:
        losses = self._compute_losses(forecasts, realization)
        self._state.loss_history.append(losses)
        self._t += 1

        if self._auto_eta:
            eta = np.sqrt(8.0 * np.log(max(self.n_experts, 2)) / max(self._t, 1))
        else:
            eta = self.eta

        self._weights = _fixed_share_update(self._weights, losses, eta, self.alpha)
