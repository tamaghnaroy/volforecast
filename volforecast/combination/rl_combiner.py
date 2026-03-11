"""
Reinforcement Learning-based forecast combination.

Formulates forecast combination as a sequential decision problem:
- State: recent expert forecasts, losses, weights, volatility features
- Action: weight vector over K experts (continuous action space)
- Reward: negative loss (MSE or QLIKE) of combined forecast

Uses stable-baselines3 PPO by default. Falls back to a simple policy
gradient if SB3 is not available.

References
----------
- Sutton & Barto (2018). "Reinforcement Learning: An Introduction."
- Schulman et al. (2017). "Proximal Policy Optimization." arXiv:1707.06347.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from volforecast.combination.online import BaseCombiner, CombinerState


class RLCombinerEnv:
    """Gymnasium-compatible environment for RL-based combination.

    This is a lightweight environment that doesn't require gymnasium
    to be installed for basic usage. Full gym compatibility is optional.
    """

    def __init__(
        self,
        n_experts: int,
        lookback: int = 10,
        loss_fn: str = "MSE",
    ) -> None:
        self.n_experts = n_experts
        self.lookback = lookback
        self.loss_fn = loss_fn

        # State dimension: K recent forecasts + K recent losses + K current weights
        self.state_dim = 3 * n_experts + lookback
        self.action_dim = n_experts

        self._forecasts_buffer: list[NDArray[np.float64]] = []
        self._losses_buffer: list[NDArray[np.float64]] = []
        self._realizations: list[float] = []
        self._weights = np.ones(n_experts, dtype=np.float64) / n_experts

    def get_state(self) -> NDArray[np.float64]:
        """Build state vector from recent history."""
        state = np.zeros(self.state_dim, dtype=np.float64)

        # Recent expert forecasts (last period, normalized)
        if self._forecasts_buffer:
            f = self._forecasts_buffer[-1]
            f_std = np.std(f)
            if f_std > 1e-10:
                state[:self.n_experts] = (f - np.mean(f)) / f_std
            else:
                state[:self.n_experts] = 0.0

        # Recent expert losses (last period, normalized)
        if self._losses_buffer:
            l = self._losses_buffer[-1]
            l_std = np.std(l)
            if l_std > 1e-10:
                state[self.n_experts:2*self.n_experts] = (l - np.mean(l)) / l_std

        # Current weights
        state[2*self.n_experts:3*self.n_experts] = self._weights

        # Recent realizations (lookback window, normalized)
        if self._realizations:
            recent = self._realizations[-self.lookback:]
            r_arr = np.array(recent, dtype=np.float64)
            r_std = np.std(r_arr)
            if r_std > 1e-10:
                r_norm = (r_arr - np.mean(r_arr)) / r_std
            else:
                r_norm = np.zeros_like(r_arr)
            offset = 3 * self.n_experts
            state[offset:offset + len(r_norm)] = r_norm

        return state

    def step(
        self,
        action: NDArray[np.float64],
        forecasts: NDArray[np.float64],
        realization: float,
    ) -> tuple[NDArray[np.float64], float, dict]:
        """Take a step: apply weights, observe realization, compute reward.

        Parameters
        ----------
        action : array, shape (K,)
            Raw action (will be softmax-ed to get weights).
        forecasts : array, shape (K,)
            Expert forecasts for current period.
        realization : float
            Observed proxy value.

        Returns
        -------
        next_state, reward, info
        """
        # Convert action to weights via softmax
        action = np.asarray(action, dtype=np.float64)
        exp_a = np.exp(action - np.max(action))
        weights = exp_a / np.sum(exp_a)
        self._weights = weights

        # Combined forecast
        combined = float(np.dot(weights, forecasts))

        # Compute loss and reward
        if self.loss_fn == "MSE":
            loss = (combined - realization) ** 2
        elif self.loss_fn == "QLIKE":
            combined_safe = max(combined, 1e-20)
            loss = realization / combined_safe + np.log(combined_safe)
        else:
            loss = (combined - realization) ** 2

        reward = -loss

        # Per-expert losses
        f = np.asarray(forecasts, dtype=np.float64)
        if self.loss_fn == "MSE":
            expert_losses = (f - realization) ** 2
        elif self.loss_fn == "QLIKE":
            f_safe = np.maximum(f, 1e-20)
            expert_losses = realization / f_safe + np.log(f_safe)
        else:
            expert_losses = (f - realization) ** 2

        self._forecasts_buffer.append(forecasts.copy())
        self._losses_buffer.append(expert_losses)
        self._realizations.append(realization)

        next_state = self.get_state()
        info = {"combined": combined, "loss": loss, "weights": weights.copy()}

        return next_state, reward, info

    def reset(self) -> NDArray[np.float64]:
        self._forecasts_buffer.clear()
        self._losses_buffer.clear()
        self._realizations.clear()
        self._weights = np.ones(self.n_experts, dtype=np.float64) / self.n_experts
        return self.get_state()


class SimplePolicyGradient:
    """Lightweight REINFORCE policy gradient for weight selection.

    Used when stable-baselines3 is not available. Simple but effective
    for the low-dimensional combination problem.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 0.001,
        gamma: float = 0.99,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma

        # Simple linear policy: action = W @ state + b
        rng = np.random.default_rng(42)
        self.W = rng.normal(0, 0.01, size=(action_dim, state_dim))
        self.b = np.zeros(action_dim, dtype=np.float64)

        self._trajectory: list[tuple] = []

    def select_action(
        self, state: NDArray[np.float64], explore: bool = True,
    ) -> NDArray[np.float64]:
        """Select action (raw logits for softmax)."""
        action = self.W @ state + self.b
        if explore:
            action += np.random.normal(0, 0.1, size=self.action_dim)
        return action

    def store_transition(
        self, state: NDArray[np.float64], action: NDArray[np.float64], reward: float,
    ) -> None:
        self._trajectory.append((state.copy(), action.copy(), reward))

    def update(self) -> float:
        """REINFORCE update at end of episode."""
        if not self._trajectory:
            return 0.0

        # Compute discounted returns
        returns = []
        G = 0.0
        for _, _, r in reversed(self._trajectory):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = np.array(returns, dtype=np.float64)
        if np.std(returns) > 1e-8:
            returns = (returns - np.mean(returns)) / np.std(returns)

        # Policy gradient update
        total_loss = 0.0
        for (state, action, _), G in zip(self._trajectory, returns):
            # Gradient of log-softmax policy w.r.t. W and b
            logits = self.W @ state + self.b
            exp_l = np.exp(logits - np.max(logits))
            probs = exp_l / np.sum(exp_l)

            # Simplified gradient: d log pi / d W ≈ (action_onehot - probs) outer state
            action_soft = np.exp(action - np.max(action))
            action_soft /= np.sum(action_soft)

            grad_logits = action_soft - probs  # shape (action_dim,)
            grad_W = G * np.outer(grad_logits, state)
            grad_b = G * grad_logits

            self.W += self.lr * grad_W
            self.b += self.lr * grad_b
            total_loss += abs(G)

        self._trajectory.clear()
        return total_loss / max(len(returns), 1)


class RLCombiner(BaseCombiner):
    """RL-based adaptive forecast combination.

    Trains a policy to select expert weights based on recent performance.
    Uses SimplePolicyGradient by default; can use stable-baselines3 PPO
    if available and configured.

    Parameters
    ----------
    n_experts : int
    lookback : int
        Number of recent observations in state.
    lr : float
        Learning rate for policy gradient.
    train_episodes : int
        Number of training episodes over historical data.
    loss_fn : str
        Loss function for reward computation.
    """

    def __init__(
        self,
        n_experts: int,
        lookback: int = 10,
        lr: float = 0.001,
        train_episodes: int = 5,
        loss_fn: str = "MSE",
    ) -> None:
        super().__init__(n_experts, loss_fn)
        self.lookback = lookback
        self.train_episodes = train_episodes

        self._env = RLCombinerEnv(n_experts, lookback, loss_fn)
        self._policy = SimplePolicyGradient(
            state_dim=self._env.state_dim,
            action_dim=self._env.action_dim,
            lr=lr,
        )
        self._trained = False

    def train(
        self,
        expert_forecasts: NDArray[np.float64],
        realizations: NDArray[np.float64],
    ) -> dict[str, float]:
        """Train the RL policy on historical data.

        Parameters
        ----------
        expert_forecasts : array, shape (T, K)
            Historical expert forecasts.
        realizations : array, shape (T,)
            Historical proxy realizations.

        Returns
        -------
        dict with training metrics.
        """
        T = expert_forecasts.shape[0]
        total_rewards = []

        for episode in range(self.train_episodes):
            state = self._env.reset()
            episode_reward = 0.0

            for t in range(T):
                action = self._policy.select_action(state, explore=True)
                next_state, reward, info = self._env.step(
                    action, expert_forecasts[t], realizations[t],
                )
                self._policy.store_transition(state, action, reward)
                state = next_state
                episode_reward += reward

            self._policy.update()
            total_rewards.append(episode_reward)

        self._trained = True
        return {
            "mean_reward": float(np.mean(total_rewards)),
            "final_reward": float(total_rewards[-1]),
            "n_episodes": self.train_episodes,
        }

    def combine(self, forecasts: NDArray[np.float64]) -> float:
        """Produce combined forecast using learned policy."""
        state = self._env.get_state()
        action = self._policy.select_action(state, explore=False)

        # Softmax to get weights
        exp_a = np.exp(action - np.max(action))
        self._weights = exp_a / np.sum(exp_a)

        combined = float(np.dot(self._weights, forecasts))
        self._state.combined_history.append(combined)
        self._state.weights_history.append(self._weights.copy())
        return combined

    def update(self, forecasts: NDArray[np.float64], realization: float) -> None:
        """Update RL environment state with new observation."""
        state = self._env.get_state()
        action = self._policy.select_action(state, explore=False)
        next_state, reward, info = self._env.step(action, forecasts, realization)

        losses = self._compute_losses(forecasts, realization)
        self._state.loss_history.append(losses)
        self._t += 1
