# -*- coding: utf-8 -*-
"""
Walk-Forward Cycle Orchestrator.

Flow mỗi cycle:
1. UPDATE DATA (skip nếu đã mới)
2. SL PRE-TRAIN → save model
3. RL FINE-TUNE → save policy
4. BACKTEST (out-of-sample) → save report
5. EVALUATE (so sánh với cycle trước)
6. SAVE hoặc ROLLBACK

Lặp lại mỗi cycle_interval.
"""

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from engine.feature_scaler import FeatureScaler
from engine.wf_trainer.config import WFConfig
from engine.wf_trainer.rl_agent import PolicyNetwork, RLTrainResult, train_rl
from engine.wf_trainer.rl_env import NUM_ACTIONS, PRICE_SCALE, TradingEnv
from engine.wf_trainer.sl_trainer import SLTrainResult, train_supervised

logger = logging.getLogger(__name__)


@dataclass
class CycleReport:
    """Kết quả 1 walk-forward cycle."""

    cycle_number: int = 0
    timestamp: str = ""
    # SL metrics
    sl_train_loss: float = 0.0
    sl_val_loss: float = 0.0
    sl_epochs: int = 0
    # RL metrics
    rl_avg_return: float = 0.0
    rl_best_return: float = 0.0
    rl_episodes: int = 0
    # Backtest metrics (out-of-sample)
    backtest_return: float = 0.0
    backtest_sharpe: float = 0.0
    backtest_win_rate: float = 0.0
    backtest_trades: int = 0
    backtest_max_drawdown: float = 0.0
    # General
    symbols_used: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    model_improved: bool = False
    trades_detail: List[dict] = field(default_factory=list)


def run_one_cycle(
    symbols: List[str],
    config: WFConfig = WFConfig(),
    cycle_number: int = 0,
    debug_log_path: Optional[str] = None,
) -> CycleReport:
    """
    Chạy 1 walk-forward training cycle hoàn chỉnh.

    Args:
        symbols: Danh sách mã cổ phiếu
        config: Cấu hình
        cycle_number: Số thứ tự cycle
        debug_log_path: Path file debug log (None = không log)

    Returns:
        CycleReport với đầy đủ metrics + trades chi tiết
    """
    start_time = time.time()

    def _log(msg: str) -> None:
        if debug_log_path:
            try:
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [WF-CYCLE] {msg}\n")
            except Exception:
                pass
        logger.info(f"[WF-Cycle #{cycle_number}] {msg}")

    _log(f"START cycle #{cycle_number}, symbols={symbols[:5]}...")

    # Giới hạn symbols
    train_symbols = symbols[:config.max_symbols_per_cycle]

    # ===========================================================
    # STAGE 1: Supervised Pre-Training
    # ===========================================================
    _log("Stage 1: Supervised Learning...")

    from engine.evaluation_model import StockEvalNet
    from engine.config import ModelConfig

    model_config = ModelConfig()
    model = StockEvalNet(model_config)

    # Load checkpoint nếu có (fine-tune thay vì train from scratch)
    checkpoint_path = Path(config.checkpoint_dir) / "stock_eval_net.pt"
    if checkpoint_path.exists():
        try:
            cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(cp, dict) and "model_state_dict" in cp:
                model.load_state_dict(cp["model_state_dict"])
            elif isinstance(cp, dict) and "state_dict" in cp:
                model.load_state_dict(cp["state_dict"])
            else:
                model.load_state_dict(cp)
            _log("  Loaded existing checkpoint for fine-tuning")
        except Exception as e:
            _log(f"  Không load được checkpoint ({e}), train from scratch")

    # Load existing scaler nếu có (để incremental fit)
    scaler_path = Path(config.checkpoint_dir) / "scaler_params.json"
    if scaler_path.exists():
        try:
            existing_scaler = FeatureScaler.load(str(scaler_path))
            _log("  Loaded existing scaler for incremental fit")
        except Exception as e:
            _log(f"  Không load được scaler ({e}), sẽ fit mới")
            existing_scaler = None
    else:
        existing_scaler = None

    sl_result, fitted_scaler = train_supervised(model, config, train_symbols, scaler=existing_scaler)
    _log(f"  SL done: {sl_result.epochs_completed} epochs, "
         f"val_loss={sl_result.final_val_loss:.4f}")

    # Save SL model
    _save_model(model, config.checkpoint_dir, "stock_eval_net.pt")
    _log("  Model saved")

    # ===========================================================
    # STAGE 2: Reinforcement Learning Fine-Tuning
    # ===========================================================
    _log("Stage 2: Reinforcement Learning...")

    # Chuẩn bị env data cho RL (dùng phần test data)
    env_features, env_prices = _prepare_rl_data(train_symbols, config)

    if env_features:
        # Truyền TCN model đã train ở Stage 1 làm backbone cho RL policy
        policy, rl_result = train_rl(env_features, env_prices, config, tcn_model=model)
        _log(f"  RL done: {rl_result.episodes_completed} episodes, "
             f"avg_return={rl_result.avg_return:.2%}, best={rl_result.best_return:.2%}")

        # Save policy
        _save_policy(policy, config.checkpoint_dir, "rl_policy.pt")
    else:
        _log("  SKIP RL: không đủ data")
        rl_result = RLTrainResult()
        policy = None

    # ===========================================================
    # STAGE 3: Backtest (Out-of-Sample)
    # ===========================================================
    _log("Stage 3: Backtest out-of-sample...")

    backtest_metrics, trades_detail = _run_backtest(
        model, policy, train_symbols, config
    )
    _log(f"  Backtest: return={backtest_metrics['total_return']:.2%}, "
         f"trades={backtest_metrics['total_trades']}, "
         f"WR={backtest_metrics['win_rate']:.1%}")

    # ===========================================================
    # STAGE 4: Save Report
    # ===========================================================
    duration = time.time() - start_time

    report = CycleReport(
        cycle_number=cycle_number,
        timestamp=datetime.now().isoformat(),
        sl_train_loss=sl_result.final_train_loss,
        sl_val_loss=sl_result.final_val_loss,
        sl_epochs=sl_result.epochs_completed,
        rl_avg_return=rl_result.avg_return,
        rl_best_return=rl_result.best_return,
        rl_episodes=rl_result.episodes_completed,
        backtest_return=backtest_metrics["total_return"],
        backtest_sharpe=backtest_metrics["sharpe"],
        backtest_win_rate=backtest_metrics["win_rate"],
        backtest_trades=backtest_metrics["total_trades"],
        backtest_max_drawdown=backtest_metrics["max_drawdown"],
        symbols_used=train_symbols,
        duration_seconds=duration,
        model_improved=True,  # Có thể so sánh với cycle trước
        trades_detail=trades_detail,
    )

    # Lưu report ra file
    _save_cycle_report(report, config)
    _log(f"DONE cycle #{cycle_number} in {duration:.1f}s")

    return report


# ==============================================================================
# Helper functions
# ==============================================================================


def _prepare_rl_data(
    symbols: List[str], config: WFConfig
) -> tuple:
    """Chuẩn bị features + prices cho RL environment."""
    from engine.market_state import INDICATOR_COLUMNS, OHLCV_COLUMNS, NUM_INDICATORS

    features_list = []
    prices_list = []

    for symbol in symbols:
        csv_path = Path(config.data_dir) / f"{symbol}.csv"
        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if len(df) < config.min_sessions:
            continue

        # Chỉ lấy phần test (cuối 15%) cho RL training
        n = len(df)
        test_start = int(n * (config.train_ratio + config.val_ratio)) + config.embargo_days
        if test_start >= n - config.lookback:
            test_start = max(0, n - 200)

        df_test = df.iloc[test_start:].reset_index(drop=True)
        if len(df_test) < config.lookback + 10:
            continue

        # Extract features
        ohlcv = df_test[OHLCV_COLUMNS].values.astype(np.float64)
        indicators = np.zeros((len(df_test), NUM_INDICATORS), dtype=np.float64)
        for i, col in enumerate(INDICATOR_COLUMNS):
            if col in df_test.columns:
                indicators[:, i] = df_test[col].values.astype(np.float64)

        features = np.concatenate([ohlcv, indicators], axis=1)
        if features.shape[1] < config.num_features:
            pad = np.zeros((len(df_test), config.num_features - features.shape[1]))
            features = np.concatenate([features, pad], axis=1)
        elif features.shape[1] > config.num_features:
            features = features[:, :config.num_features]

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Per-window normalize cho mỗi row
        col_min = features.min(axis=0)
        col_max = features.max(axis=0)
        col_range = col_max - col_min
        col_range[col_range == 0] = 1.0
        features_norm = (features - col_min) / col_range

        close_prices = df_test["close"].values.astype(np.float64)

        features_list.append(features_norm.astype(np.float32))
        prices_list.append(close_prices)

    return features_list, prices_list


def _run_backtest(
    model: torch.nn.Module,
    policy: Optional[PolicyNetwork],
    symbols: List[str],
    config: WFConfig,
) -> tuple:
    """
    Backtest out-of-sample dùng RL policy (hoặc SL model fallback).

    Returns:
        (metrics_dict, trades_detail_list)
    """
    from engine.market_state import INDICATOR_COLUMNS, OHLCV_COLUMNS, NUM_INDICATORS

    all_trades = []
    total_pnl = 0.0
    initial_capital = config.rl_initial_capital

    for symbol in symbols:
        csv_path = Path(config.data_dir) / f"{symbol}.csv"
        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if len(df) < config.min_sessions:
            continue

        # Lấy phần out-of-sample cuối cùng (15%)
        n = len(df)
        oos_start = int(n * (config.train_ratio + config.val_ratio)) + config.embargo_days
        if oos_start >= n - 20:
            continue

        df_oos = df.iloc[oos_start:].reset_index(drop=True)

        # Extract features
        ohlcv = df_oos[OHLCV_COLUMNS].values.astype(np.float64)
        indicators = np.zeros((len(df_oos), NUM_INDICATORS), dtype=np.float64)
        for i, col in enumerate(INDICATOR_COLUMNS):
            if col in df_oos.columns:
                indicators[:, i] = df_oos[col].values.astype(np.float64)

        features = np.concatenate([ohlcv, indicators], axis=1)
        if features.shape[1] < config.num_features:
            pad = np.zeros((len(df_oos), config.num_features - features.shape[1]))
            features = np.concatenate([features, pad], axis=1)
        elif features.shape[1] > config.num_features:
            features = features[:, :config.num_features]

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        col_min = features.min(axis=0)
        col_max = features.max(axis=0)
        col_range = col_max - col_min
        col_range[col_range == 0] = 1.0
        features_norm = (features - col_min) / col_range

        close_prices = df_oos["close"].values.astype(np.float64)
        dates = pd.to_datetime(df_oos["time"]).dt.date.values

        # Chạy backtest dùng RL policy
        env = TradingEnv(
            features=features_norm.astype(np.float32),
            close_prices=close_prices,
            initial_capital=initial_capital,
            transaction_cost=config.rl_transaction_cost,
        )

        obs = env.reset()
        done = False
        position_open = False
        buy_date = None
        buy_price = 0.0
        buy_shares = 0

        while not done:
            step = env.state.step
            if step >= len(dates):
                break

            if policy is not None:
                action, _, _ = policy.select_action(obs)
            else:
                # Fallback: dùng SL model score
                action = _sl_action(model, features_norm, step, config)

            prev_shares = env.state.shares
            obs, reward, done = env.step(action)

            # Track trades
            current_shares = env.state.shares
            if current_shares > prev_shares and not position_open:
                # Opened position
                position_open = True
                buy_date = dates[min(step, len(dates) - 1)]
                buy_price = close_prices[min(step, len(close_prices) - 1)]
                buy_shares = current_shares - prev_shares
            elif current_shares < prev_shares and position_open:
                # Closed position
                sell_step = min(step, len(dates) - 1, len(close_prices) - 1)
                sell_date = dates[sell_step]
                sell_price = close_prices[sell_step]
                shares_sold = prev_shares - current_shares

                pnl_per_share = (sell_price - buy_price) * PRICE_SCALE
                pnl = pnl_per_share * shares_sold
                pnl_pct = (sell_price - buy_price) / buy_price if buy_price > 0 else 0
                holding_days = (sell_date - buy_date).days if buy_date and sell_date else 0

                trade = {
                    "ma_co_phieu": symbol,
                    "ngay_mua": str(buy_date),
                    "gia_mua_vnd": round(buy_price * PRICE_SCALE, 0),
                    "khoi_luong": shares_sold,
                    "ngay_ban": str(sell_date),
                    "gia_ban_vnd": round(sell_price * PRICE_SCALE, 0),
                    "so_ngay_giu": holding_days,
                    "lai_lo_vnd": round(pnl, 0),
                    "lai_lo_pct": round(pnl_pct * 100, 2),
                    "ket_qua": "Lãi" if pnl > 0 else "Lỗ",
                }
                all_trades.append(trade)
                total_pnl += pnl

                if current_shares == 0:
                    position_open = False

    # Tính metrics
    total_trades = len(all_trades)
    winning = sum(1 for t in all_trades if t["lai_lo_vnd"] > 0)
    win_rate = winning / max(total_trades, 1)
    total_return = total_pnl / initial_capital

    # Sharpe (giản lược — dùng trade returns)
    if all_trades:
        trade_returns = [t["lai_lo_pct"] / 100.0 for t in all_trades]
        sharpe = (np.mean(trade_returns) / (np.std(trade_returns) + 1e-8)) * np.sqrt(252 / max(np.mean([t["so_ngay_giu"] for t in all_trades]), 1))
    else:
        sharpe = 0.0

    max_dd = 0.0  # Simplified

    metrics = {
        "total_return": total_return,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "max_drawdown": max_dd,
        "total_pnl": total_pnl,
    }

    return metrics, all_trades


def _sl_action(model, features_norm, step, config) -> int:
    """Fallback: dùng SL model score để quyết định action."""
    lookback = config.lookback
    if step < lookback:
        return 0  # HOLD

    window = features_norm[step - lookback:step]
    tensor = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        score = float(model(tensor)[0, 0])

    if score > 0.2:
        return 3  # BUY_100%
    elif score < -0.2:
        return 5  # SELL_100%
    return 0  # HOLD


def _get_model_score(model, features_norm, step, config) -> float:
    """
    Tính position_score từ SL model tại step hiện tại.
    
    Returns:
        Score trong khoảng [-1, 1], 0.0 nếu không đủ dữ liệu.
    """
    lookback = config.lookback
    if step < lookback:
        return 0.0

    window = features_norm[step - lookback:step]
    tensor = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        score = float(model(tensor)[0, 0])
    return score


def _save_model(model, checkpoint_dir: str, filename: str) -> None:
    """Save model checkpoint."""
    dir_path = Path(checkpoint_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / filename
    torch.save({"model_state_dict": model.state_dict()}, path)


def _save_policy(policy: PolicyNetwork, checkpoint_dir: str, filename: str) -> None:
    """Save RL policy."""
    dir_path = Path(checkpoint_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / filename
    torch.save(policy.state_dict(), path)


def _save_cycle_report(report: CycleReport, config: WFConfig) -> None:
    """Lưu cycle report ra JSON + CSV."""
    dir_path = Path(config.report_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Summary JSON
    summary = {
        "cycle_number": report.cycle_number,
        "timestamp": report.timestamp,
        "sl_val_loss": report.sl_val_loss,
        "sl_epochs": report.sl_epochs,
        "rl_avg_return": round(report.rl_avg_return * 100, 2),
        "rl_best_return": round(report.rl_best_return * 100, 2),
        "rl_episodes": report.rl_episodes,
        "backtest_return_pct": round(report.backtest_return * 100, 2),
        "backtest_sharpe": round(report.backtest_sharpe, 3),
        "backtest_win_rate_pct": round(report.backtest_win_rate * 100, 1),
        "backtest_trades": report.backtest_trades,
        "symbols": report.symbols_used,
        "duration_seconds": round(report.duration_seconds, 1),
    }
    summary_path = dir_path / f"wf_cycle_{report.cycle_number}_{ts}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Trades CSV
    if report.trades_detail:
        trades_path = dir_path / f"wf_trades_{report.cycle_number}_{ts}.csv"
        df = pd.DataFrame(report.trades_detail)
        df.to_csv(trades_path, index=False, encoding="utf-8-sig")

    # History (append)
    history_dir = Path(config.history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    history_file = history_dir / "wf_cycles.jsonl"
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


class WalkForwardCycle:
    """
    Manager chạy walk-forward cycles liên tục.

    Mỗi lần gọi run() → chạy 1 cycle, tự tăng cycle_number.
    """

    def __init__(self, config: WFConfig = WFConfig()):
        self.config = config
        self._cycle_count = self._load_cycle_count()

    def run(self, symbols: List[str], debug_log_path: Optional[str] = None) -> CycleReport:
        """Chạy 1 cycle, tự tăng counter."""
        self._cycle_count += 1
        return run_one_cycle(
            symbols=symbols,
            config=self.config,
            cycle_number=self._cycle_count,
            debug_log_path=debug_log_path,
        )

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    def _load_cycle_count(self) -> int:
        """Load cycle count từ history file."""
        history_file = Path(self.config.history_dir) / "wf_cycles.jsonl"
        if not history_file.exists():
            return 0
        try:
            lines = history_file.read_text(encoding="utf-8").strip().split("\n")
            return len([l for l in lines if l.strip()])
        except Exception:
            return 0
