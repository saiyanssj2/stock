# -*- coding: utf-8 -*-
"""
Stage 2: Reinforcement Learning Agent (Policy Gradient + TCN Backbone).

Agent nhận state từ TradingEnv, dùng pretrained TCN backbone để extract
market features chất lượng cao, rồi policy head quyết định action.

Architecture:
- TCN Backbone (frozen, từ Stage 1): raw features → encoded 64-dim
- Policy Head: Linear(64+3 → 128) → ReLU → Linear(128 → NUM_ACTIONS) + Softmax
- Value Head: Linear(64+3 → 128) → ReLU → Linear(128 → 1)

So với v1 (raw features → 2 FC layers): policy giờ "nhìn được chart"
nhờ TCN đã được train để nhận diện patterns ở Stage 1.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.wf_trainer.config import WFConfig
from engine.wf_trainer.rl_env import NUM_ACTIONS, TradingEnv

logger = logging.getLogger(__name__)


@dataclass
class RLTrainResult:
    """Kết quả Stage 2 RL training."""

    episodes_completed: int = 0
    avg_reward: float = 0.0
    avg_return: float = 0.0
    best_return: float = 0.0
    duration_seconds: float = 0.0


class PolicyNetwork(nn.Module):
    """
    Policy + Value network cho RL agent (v2 — dùng TCN backbone).

    Input: observation vector (state_dim = num_features + 3)
    Internal: dùng TCN backbone để encode market features → 64-dim
    Output: action probabilities + state value

    Nếu không có backbone (fallback), dùng FC layers như v1.
    """

    def __init__(self, state_dim: int, action_dim: int = NUM_ACTIONS, hidden_dim: int = 128):
        super().__init__()
        self._state_dim = state_dim
        self._market_dim = state_dim - 3  # Số features thị trường (không tính portfolio state)
        self._backbone = None  # Sẽ được set sau bởi train_rl()
        self._backbone_out_dim = 64  # TCN output dimension

        # Policy/Value heads nhận: backbone_output(64) + portfolio_state(3) = 67
        head_input_dim = self._backbone_out_dim + 3

        self.shared = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim // 2, action_dim)
        self.value_head = nn.Linear(hidden_dim // 2, 1)

        # Fallback FC (dùng khi không có backbone)
        self._fallback = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

    def set_backbone(self, backbone: nn.Module, lookback: int = 60) -> None:
        """
        Gắn TCN backbone (frozen) vào policy.

        Args:
            backbone: StockEvalNet model (chỉ dùng TCN + attention layers)
            lookback: Số phiên lookback mà backbone cần
        """
        self._backbone = backbone
        self._backbone.eval()
        # Freeze backbone — không train lại weights TCN
        for param in self._backbone.parameters():
            param.requires_grad = False
        self._lookback = lookback

    def _encode_market_features(self, market_features: torch.Tensor) -> torch.Tensor:
        """
        Dùng TCN backbone để encode market features → 64-dim vector.

        Args:
            market_features: (batch, market_dim) — features 1 ngày

        Returns:
            (batch, 64) — encoded features

        Lưu ý: backbone cần input (batch, lookback, features) nhưng RL
        chỉ có 1 ngày. Workaround: repeat features thành window giả.
        Backbone sẽ cho output kém hơn window thật nhưng vẫn tốt hơn raw.
        """
        batch_size = market_features.shape[0]
        # Tạo pseudo-window: repeat ngày hiện tại lookback lần
        # Đây là compromise — backbone được train trên sequences thật,
        # nhưng vẫn extract được feature patterns tốt hơn raw Linear
        pseudo_window = market_features.unsqueeze(1).expand(-1, self._lookback, -1)

        with torch.no_grad():
            # Chạy TCN backbone (không cần grad — frozen)
            x = pseudo_window.transpose(1, 2)  # (batch, features, lookback)
            x = self._backbone.tcn(x)  # (batch, 64, lookback)
            x = x.transpose(1, 2)  # (batch, lookback, 64)
            attn_out, _ = self._backbone.attention(x, x, x)
            x = self._backbone.layer_norm(x + attn_out)
            x = x.transpose(1, 2)  # (batch, 64, lookback)
            x = self._backbone.pool(x)  # (batch, 64, 1)
            x = x.squeeze(-1)  # (batch, 64)

        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: state tensor (batch, state_dim) = [market_features | portfolio_state]

        Returns:
            (action_probs, state_value)
        """
        if self._backbone is not None:
            # Tách market features và portfolio state
            market_feat = x[:, :self._market_dim]
            portfolio_state = x[:, self._market_dim:]  # (batch, 3)

            # Encode qua TCN backbone
            encoded = self._encode_market_features(market_feat)  # (batch, 64)

            # Nối encoded + portfolio state
            h_input = torch.cat([encoded, portfolio_state], dim=1)  # (batch, 67)
            h = self.shared(h_input)
        else:
            # Fallback: không có backbone, dùng FC
            h = self._fallback(x)

        logits = self.policy_head(h)
        probs = F.softmax(logits, dim=-1)
        value = self.value_head(h)
        return probs, value

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Chọn action dựa trên policy (sampling).

        Returns:
            (action, log_prob, value)
        """
        state_t = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            probs, value = self.forward(state_t)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()


def train_rl(
    env_features_list: List[np.ndarray],
    env_prices_list: List[np.ndarray],
    config: WFConfig,
    tcn_model: Optional[nn.Module] = None,
) -> Tuple[PolicyNetwork, RLTrainResult]:
    """
    Chạy RL training trên nhiều episodes (nhiều symbols).

    Dùng REINFORCE with baseline (policy gradient).
    Mỗi episode = 1 symbol's data (chọn ngẫu nhiên).

    Cải tiến v2:
    - Tăng episodes (300 thay vì 50) → policy converge tốt hơn
    - Dùng TCN backbone để extract features → policy "nhìn" chart patterns

    Args:
        env_features_list: List features arrays (mỗi symbol 1 array)
        env_prices_list: List close prices arrays
        config: WFConfig
        tcn_model: StockEvalNet đã train ở Stage 1 (optional, dùng làm backbone)

    Returns:
        (trained_policy, result)
    """
    import time

    start_time = time.time()

    if not env_features_list:
        return PolicyNetwork(config.num_features + 3), RLTrainResult()

    # Tạo policy network
    state_dim = config.num_features + 3  # features + portfolio state
    policy = PolicyNetwork(state_dim=state_dim)

    # Gắn TCN backbone (frozen) nếu có
    if tcn_model is not None:
        policy.set_backbone(tcn_model, lookback=config.lookback)
        logger.info("[RL] TCN backbone attached (frozen) — policy uses encoded features")

    # Load policy checkpoint nếu có (tiếp tục học)
    policy_path = Path(config.checkpoint_dir) / "rl_policy.pt"
    if policy_path.exists():
        try:
            saved_state = torch.load(policy_path, map_location="cpu", weights_only=True)
            # Chỉ load policy heads, không load backbone-related weights
            # (vì backbone có thể thay đổi giữa cycles)
            current_state = policy.state_dict()
            compatible_keys = {k: v for k, v in saved_state.items()
                              if k in current_state and current_state[k].shape == v.shape}
            if compatible_keys:
                current_state.update(compatible_keys)
                policy.load_state_dict(current_state)
                logger.info(f"[RL] Loaded {len(compatible_keys)}/{len(saved_state)} policy weights")
        except Exception as e:
            logger.info(f"[RL] Không load được policy checkpoint ({e}), dùng random init")

    # Chỉ train policy heads (không train backbone)
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=config.rl_lr)

    all_rewards = []
    all_returns = []
    best_return = -float("inf")

    for episode in range(config.rl_episodes):
        # Chọn symbol ngẫu nhiên
        idx = episode % len(env_features_list)
        features = env_features_list[idx]
        prices = env_prices_list[idx]

        env = TradingEnv(
            features=features,
            close_prices=prices,
            initial_capital=config.rl_initial_capital,
            transaction_cost=config.rl_transaction_cost,
        )

        # Collect trajectory
        states, actions, rewards = [], [], []
        obs = env.reset()
        done = False

        while not done:
            action, _, _ = policy.select_action(obs)
            next_obs, reward, done = env.step(action)

            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            obs = next_obs

        if not states:
            continue

        # Compute discounted returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + config.rl_gamma * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Policy gradient update
        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long)

        probs, values_t = policy(states_t)
        dist = torch.distributions.Categorical(probs)
        log_probs_t = dist.log_prob(actions_t)
        values_t = values_t.squeeze(-1)

        # Advantages = returns - baseline (value)
        advantages = (returns_t - values_t.detach())

        # Loss = policy loss + value loss - entropy bonus
        policy_loss = -(log_probs_t * advantages).mean()
        value_loss = F.mse_loss(values_t, returns_t)
        entropy_bonus = dist.entropy().mean() * 0.01
        loss = policy_loss + 0.5 * value_loss - entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)
        optimizer.step()

        # Metrics
        episode_reward = sum(rewards)
        episode_return = env.get_final_metrics()["total_return"]
        all_rewards.append(episode_reward)
        all_returns.append(episode_return)
        best_return = max(best_return, episode_return)

    duration = time.time() - start_time
    avg_reward = np.mean(all_rewards) if all_rewards else 0.0
    avg_return = np.mean(all_returns) if all_returns else 0.0

    logger.info(
        f"[RL] Done: {config.rl_episodes} episodes, "
        f"avg_reward={avg_reward:.4f}, avg_return={avg_return:.2%}, "
        f"best_return={best_return:.2%}, {duration:.1f}s"
    )

    result = RLTrainResult(
        episodes_completed=config.rl_episodes,
        avg_reward=avg_reward,
        avg_return=avg_return,
        best_return=best_return,
        duration_seconds=duration,
    )

    return policy, result
