"""
Tests for forecast combination and online aggregation.
"""

import numpy as np
import pytest

from volforecast.combination.online import (
    EqualWeightCombiner,
    InverseMSECombiner,
    AFTERCombiner,
    EWACombiner,
    FixedShareCombiner,
)
from volforecast.combination.rl_combiner import RLCombiner, RLCombinerEnv


# ─── Fixtures ───

@pytest.fixture
def expert_forecasts():
    """3 experts, 100 periods."""
    rng = np.random.default_rng(42)
    T, K = 100, 3
    true_var = np.abs(rng.normal(0.0001, 0.00003, size=T))
    # Expert 1: good, Expert 2: biased, Expert 3: noisy
    f1 = true_var + rng.normal(0, 0.00001, size=T)
    f2 = true_var * 1.5
    f3 = true_var + rng.normal(0, 0.00005, size=T)
    forecasts = np.column_stack([np.maximum(f1, 1e-10),
                                  np.maximum(f2, 1e-10),
                                  np.maximum(f3, 1e-10)])
    return forecasts, true_var


# ─── Equal Weight Tests ───

class TestEqualWeight:
    def test_weights_uniform(self):
        c = EqualWeightCombiner(n_experts=4)
        assert np.allclose(c.weights, 0.25)

    def test_combine(self):
        c = EqualWeightCombiner(n_experts=3)
        f = np.array([1.0, 2.0, 3.0])
        combined = c.combine(f)
        assert np.isclose(combined, 2.0)

    def test_weights_stay_equal(self, expert_forecasts):
        forecasts, realizations = expert_forecasts
        c = EqualWeightCombiner(n_experts=3)
        for t in range(len(realizations)):
            c.combine(forecasts[t])
            c.update(forecasts[t], realizations[t])
        assert np.allclose(c.weights, 1.0 / 3)


# ─── Inverse MSE Tests ───

class TestInverseMSE:
    def test_good_expert_gets_more_weight(self, expert_forecasts):
        forecasts, realizations = expert_forecasts
        c = InverseMSECombiner(n_experts=3)
        for t in range(len(realizations)):
            c.combine(forecasts[t])
            c.update(forecasts[t], realizations[t])
        # Expert 0 (good) should have highest weight
        assert c.weights[0] > c.weights[1]

    def test_rolling_window(self, expert_forecasts):
        forecasts, realizations = expert_forecasts
        c = InverseMSECombiner(n_experts=3, window=20)
        for t in range(len(realizations)):
            c.combine(forecasts[t])
            c.update(forecasts[t], realizations[t])
        assert np.isclose(np.sum(c.weights), 1.0)


# ─── AFTER Tests ───

class TestAFTER:
    def test_weights_sum_to_one(self, expert_forecasts):
        forecasts, realizations = expert_forecasts
        c = AFTERCombiner(n_experts=3)
        for t in range(len(realizations)):
            c.combine(forecasts[t])
            c.update(forecasts[t], realizations[t])
        assert np.isclose(np.sum(c.weights), 1.0, atol=1e-10)

    def test_good_expert_concentration(self, expert_forecasts):
        forecasts, realizations = expert_forecasts
        c = AFTERCombiner(n_experts=3)
        for t in range(len(realizations)):
            c.combine(forecasts[t])
            c.update(forecasts[t], realizations[t])
        # Expert 0 should have more weight than expert 1 (biased)
        assert c.weights[0] > c.weights[1]


# ─── EWA Tests ───

class TestEWA:
    def test_weights_sum_to_one(self, expert_forecasts):
        forecasts, realizations = expert_forecasts
        c = EWACombiner(n_experts=3)
        for t in range(len(realizations)):
            c.combine(forecasts[t])
            c.update(forecasts[t], realizations[t])
        assert np.isclose(np.sum(c.weights), 1.0, atol=1e-10)

    def test_custom_eta(self, expert_forecasts):
        forecasts, realizations = expert_forecasts
        c = EWACombiner(n_experts=3, eta=0.5)
        for t in range(len(realizations)):
            c.combine(forecasts[t])
            c.update(forecasts[t], realizations[t])
        assert np.isclose(np.sum(c.weights), 1.0, atol=1e-10)


# ─── Fixed-Share Tests ───

class TestFixedShare:
    def test_weights_sum_to_one(self, expert_forecasts):
        forecasts, realizations = expert_forecasts
        c = FixedShareCombiner(n_experts=3, alpha=0.02)
        for t in range(len(realizations)):
            c.combine(forecasts[t])
            c.update(forecasts[t], realizations[t])
        assert np.isclose(np.sum(c.weights), 1.0, atol=1e-10)

    def test_higher_alpha_more_uniform(self, expert_forecasts):
        forecasts, realizations = expert_forecasts
        c_low = FixedShareCombiner(n_experts=3, alpha=0.001)
        c_high = FixedShareCombiner(n_experts=3, alpha=0.3)
        for t in range(len(realizations)):
            c_low.combine(forecasts[t])
            c_low.update(forecasts[t], realizations[t])
            c_high.combine(forecasts[t])
            c_high.update(forecasts[t], realizations[t])
        # Higher alpha should produce more uniform weights
        entropy_low = -np.sum(c_low.weights * np.log(c_low.weights + 1e-20))
        entropy_high = -np.sum(c_high.weights * np.log(c_high.weights + 1e-20))
        assert entropy_high > entropy_low

    def test_state_tracking(self, expert_forecasts):
        forecasts, realizations = expert_forecasts
        c = FixedShareCombiner(n_experts=3, alpha=0.01)
        for t in range(20):
            c.combine(forecasts[t])
            c.update(forecasts[t], realizations[t])
        state = c.get_state()
        assert len(state.weights_history) == 20
        assert len(state.loss_history) == 20


# ─── RL Combiner Tests ───

class TestRLCombiner:
    def test_env_state_shape(self):
        env = RLCombinerEnv(n_experts=3, lookback=5)
        state = env.get_state()
        assert state.shape == (env.state_dim,)

    def test_env_step(self):
        env = RLCombinerEnv(n_experts=3, lookback=5)
        action = np.array([0.1, 0.2, 0.3])
        forecasts = np.array([0.0001, 0.00015, 0.0002])
        realization = 0.00012
        next_state, reward, info = env.step(action, forecasts, realization)
        assert next_state.shape == (env.state_dim,)
        assert reward <= 0  # Reward is negative loss
        assert "combined" in info
        assert "weights" in info

    def test_rl_combiner_train(self, expert_forecasts):
        forecasts, realizations = expert_forecasts
        c = RLCombiner(n_experts=3, lookback=5, train_episodes=2)
        metrics = c.train(forecasts[:50], realizations[:50])
        assert "mean_reward" in metrics
        assert metrics["n_episodes"] == 2

    def test_rl_combiner_combine_after_train(self, expert_forecasts):
        forecasts, realizations = expert_forecasts
        c = RLCombiner(n_experts=3, lookback=5, train_episodes=2)
        c.train(forecasts[:50], realizations[:50])
        combined = c.combine(forecasts[50])
        assert combined > 0
        assert np.isclose(np.sum(c.weights), 1.0, atol=1e-8)
