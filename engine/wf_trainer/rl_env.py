# -*- coding: utf-8 -*-
"""
Market Environment cho Reinforcement Learning.

Simulates: portfolio state + market data → agent quyết định BUY/HOLD/SELL.
Tuân thủ luật TTCK VN: T+2.5 settlement, ±7% biên độ, lot size 100.

State: [market_features(61), cash_ratio, position_ratio, unrealized_pnl]
Action: discrete {0=HOLD, 1=BUY_25%, 2=BUY_50%, 3=BUY_100%, 4=SELL_50%, 5=SELL_100%}
Reward: daily_portfolio_return - transaction_cost - drawdown_penalty
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Số actions discrete
NUM_ACTIONS = 6
ACTION_NAMES = ["HOLD", "BUY_25%", "BUY_50%", "BUY_100%", "SELL_50%", "SELL_100%"]

# Luật TTCK VN
LOT_SIZE = 100
PRICE_LIMIT_PCT = 7.0
SETTLEMENT_DAYS = 3
PRICE_SCALE = 1000.0  # CSV lưu đơn vị 1000 VND


@dataclass
class EnvState:
    """Trạng thái environment tại 1 thời điểm."""

    step: int = 0
    cash: float = 0.0
    shares: int = 0
    avg_buy_price: float = 0.0
    buy_step: int = -999  # Step cuối cùng đã mua (cho T+2.5)
    portfolio_value: float = 0.0
    peak_value: float = 0.0


class TradingEnv:
    """
    Market environment cho RL agent.

    Một episode = đi qua toàn bộ data của 1 symbol theo thời gian.
    Agent nhận state mỗi ngày, ra action, nhận reward.
    """

    def __init__(
        self,
        features: np.ndarray,
        close_prices: np.ndarray,
        initial_capital: float = 1_000_000_000.0,
        transaction_cost: float = 0.0015,
    ):
        """
        Args:
            features: shape (T, num_features) — normalized features
            close_prices: shape (T,) — giá close raw (đơn vị 1000 VND)
            initial_capital: Vốn ban đầu (VND)
            transaction_cost: Phí giao dịch (0.15%)
        """
        self.features = features
        self.close_prices = close_prices
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.T = len(features)

        # State dimensions: features + [cash_ratio, position_ratio, unrealized_pnl_ratio]
        self.state_dim = features.shape[1] + 3
        self.action_dim = NUM_ACTIONS

        self.state = EnvState()
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment về trạng thái ban đầu."""
        self.state = EnvState(
            step=0,
            cash=self.initial_capital,
            shares=0,
            avg_buy_price=0.0,
            buy_step=-999,
            portfolio_value=self.initial_capital,
            peak_value=self.initial_capital,
        )
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Thực hiện 1 action, tiến 1 ngày.

        Args:
            action: 0-5 (HOLD, BUY_25%, BUY_50%, BUY_100%, SELL_50%, SELL_100%)

        Returns:
            (observation, reward, done)
        """
        s = self.state
        current_price_vnd = self.close_prices[s.step] * PRICE_SCALE
        prev_portfolio = s.portfolio_value

        # Thực hiện action
        cost = 0.0
        if action in (1, 2, 3) and s.cash > 0:
            # BUY
            pct = {1: 0.25, 2: 0.50, 3: 1.0}[action]
            buy_amount = s.cash * pct
            max_shares = int(buy_amount // current_price_vnd)
            shares_to_buy = (max_shares // LOT_SIZE) * LOT_SIZE

            if shares_to_buy >= LOT_SIZE:
                cost_amount = shares_to_buy * current_price_vnd
                cost = cost_amount * self.transaction_cost
                s.cash -= (cost_amount + cost)
                # Cập nhật giá mua trung bình
                total_shares = s.shares + shares_to_buy
                if total_shares > 0:
                    s.avg_buy_price = (
                        (s.avg_buy_price * s.shares + current_price_vnd * shares_to_buy)
                        / total_shares
                    )
                s.shares = total_shares
                s.buy_step = s.step

        elif action in (4, 5) and s.shares > 0:
            # SELL — check settlement T+2.5
            if (s.step - s.buy_step) >= SETTLEMENT_DAYS:
                pct = {4: 0.50, 5: 1.0}[action]
                shares_to_sell = int(s.shares * pct)
                shares_to_sell = (shares_to_sell // LOT_SIZE) * LOT_SIZE

                if shares_to_sell >= LOT_SIZE:
                    revenue = shares_to_sell * current_price_vnd
                    cost = revenue * self.transaction_cost
                    s.cash += (revenue - cost)
                    s.shares -= shares_to_sell
                    if s.shares == 0:
                        s.avg_buy_price = 0.0

        # Tiến 1 ngày
        s.step += 1
        done = s.step >= self.T

        # Tính portfolio value mới
        if not done:
            new_price_vnd = self.close_prices[s.step] * PRICE_SCALE
        else:
            new_price_vnd = current_price_vnd

        s.portfolio_value = s.cash + s.shares * new_price_vnd
        s.peak_value = max(s.peak_value, s.portfolio_value)

        # === Reward ===
        # Daily return
        daily_return = (s.portfolio_value - prev_portfolio) / max(prev_portfolio, 1.0)

        # Drawdown penalty
        drawdown = (s.peak_value - s.portfolio_value) / max(s.peak_value, 1.0)
        drawdown_penalty = drawdown * 0.1

        # Transaction cost đã trừ ở trên, thêm penalty nhẹ cho action
        action_penalty = 0.0001 if action != 0 else 0.0

        reward = daily_return - drawdown_penalty - action_penalty

        obs = self._get_observation() if not done else np.zeros(self.state_dim)
        return obs, reward, done

    def _get_observation(self) -> np.ndarray:
        """Tạo observation vector: market features + portfolio state."""
        s = self.state
        if s.step >= self.T:
            return np.zeros(self.state_dim, dtype=np.float32)

        market_features = self.features[s.step]

        # Portfolio state (normalized)
        total_value = max(s.portfolio_value, 1.0)
        cash_ratio = s.cash / total_value
        position_value = s.shares * self.close_prices[s.step] * PRICE_SCALE
        position_ratio = position_value / total_value

        # Unrealized PnL ratio
        if s.shares > 0 and s.avg_buy_price > 0:
            current_price_vnd = self.close_prices[s.step] * PRICE_SCALE
            unrealized_pnl = (current_price_vnd - s.avg_buy_price) / s.avg_buy_price
        else:
            unrealized_pnl = 0.0

        portfolio_state = np.array([cash_ratio, position_ratio, unrealized_pnl], dtype=np.float32)
        obs = np.concatenate([market_features.astype(np.float32), portfolio_state])
        return obs

    def get_final_metrics(self) -> dict:
        """Trả về metrics cuối episode."""
        s = self.state
        total_return = (s.portfolio_value - self.initial_capital) / self.initial_capital
        max_drawdown = (s.peak_value - s.portfolio_value) / max(s.peak_value, 1.0)
        return {
            "total_return": total_return,
            "final_value": s.portfolio_value,
            "max_drawdown": max_drawdown,
        }
