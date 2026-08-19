"""
BacktestEngineWorker - Worker process cho backtest (manual và auto).

Cung cấp:
- run_manual(): Backtest thủ công với symbol, date range, initial capital
- run_auto(): Backtest tự động cho Auto_Learner trên nhiều symbols
- Enforce luật TTCK Việt Nam (T+2.5, ±7%, lot 100)

Strategy mặc định: Simple Moving Average Crossover (STUB)
Có thể mở rộng sau với ML models thực tế.

References: Req 7.1, 7.2, 7.4, 7.5
"""

import logging
import math
import os
import random
import time
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config.vn_market_rules import LOT_SIZE, PRICE_LIMIT_PCT
from engine.workers.vn_rules import (
    compute_earliest_sell_date,
    enforce_vn_rules,
    is_trading_day,
)
from models.backtest_models import (
    AutoBacktestResult,
    BacktestResult,
    ManualBacktestParams,
    Trade,
)

logger = logging.getLogger(__name__)

# Đường dẫn thư mục data mặc định
DATA_DIR = Path("data")

# Hệ số nhân giá: CSV lưu đơn vị 1000 VND (ví dụ 76.5 = 76,500 VND)
PRICE_SCALE = 1000.0

# Tham số MA crossover mặc định (STUB strategy)
SHORT_MA_PERIOD = 9
LONG_MA_PERIOD = 20

# Số benchmark strategies để so sánh (Req 7.4)
NUM_BENCHMARK_STRATEGIES = 4


class BacktestEngineWorker:
    """
    Worker process cho backtest (manual và auto).

    Chạy mô phỏng giao dịch trên dữ liệu lịch sử, enforce luật TTCK VN,
    và tính các metrics hiệu suất.

    Attributes:
        data_dir: Thư mục chứa CSV data
        price_scale: Hệ số nhân giá từ CSV sang VND
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        price_scale: float = PRICE_SCALE,
    ) -> None:
        """
        Khởi tạo BacktestEngineWorker.

        Args:
            data_dir: Thư mục chứa file CSV OHLCV (mặc định: data/)
            price_scale: Hệ số nhân giá (mặc định: 1000.0)
        """
        self.data_dir = data_dir or DATA_DIR
        self.price_scale = price_scale
        # Cache ML model — load 1 lần, dùng cho tất cả symbols trong cycle
        self._ml_model = None
        self._model_loaded = False

    def reload_model(self) -> None:
        """
        Force reload model checkpoint cho cycle tiếp theo.

        Gọi sau khi retrain xong để backtest kế tiếp dùng weights mới.
        """
        self._model_loaded = False
        self._ml_model = None

    def run_manual(self, params: ManualBacktestParams) -> BacktestResult:
        """
        Chạy backtest thủ công cho 1 symbol.

        Flow:
        1. Load OHLCV data từ CSV
        2. Filter theo date range
        3. Generate signals (MA crossover stub)
        4. Execute trades với VN rules enforcement
        5. Tính metrics và equity curve

        Args:
            params: Tham số backtest (symbol, start_date, end_date, initial_capital)

        Returns:
            BacktestResult với equity curve, metrics, và danh sách trades
        """
        # Load data
        df = self._load_data(params.symbol)
        if df is None or df.empty:
            logger.error(f"Không tìm thấy data cho symbol {params.symbol}")
            return self._empty_result(params)

        # Filter date range
        df_period = self._filter_date_range(df, params.start_date, params.end_date)
        if df_period.empty:
            logger.warning(
                f"Không có data trong khoảng {params.start_date} - {params.end_date} "
                f"cho {params.symbol}"
            )
            return self._empty_result(params)

        # Generate signals
        signals, confidences = self._generate_signals(df_period)

        # Execute trades
        trades, equity_curve = self._execute_trades(
            df_period=df_period,
            signals=signals,
            confidences=confidences,
            symbol=params.symbol,
            initial_capital=params.initial_capital,
            enforce_rules=params.enforce_vn_rules,
        )

        # Tính metrics
        metrics = self._calculate_metrics(trades, params.initial_capital, equity_curve)

        final_capital = equity_curve[-1] if equity_curve else params.initial_capital

        return BacktestResult(
            symbol=params.symbol,
            start_date=params.start_date,
            end_date=params.end_date,
            initial_capital=params.initial_capital,
            final_capital=final_capital,
            total_return=metrics["total_return"],
            sharpe_ratio=metrics["sharpe_ratio"],
            win_rate=metrics["win_rate"],
            max_drawdown=metrics["max_drawdown"],
            total_trades=len(trades),
            trades=trades,
            equity_curve=equity_curve,
        )

    def run_auto(
        self,
        symbols: List[str],
        cycle_number: int = 1,
        progress_callback: Optional[callable] = None,
    ) -> AutoBacktestResult:
        """
        Chạy auto-backtest cho Auto_Learner trên tất cả symbols.

        Iterate qua từng symbol, chạy backtest, aggregate metrics,
        và so sánh với benchmark strategies.

        Args:
            symbols: Danh sách mã cổ phiếu cần backtest
            cycle_number: Số thứ tự auto-learning cycle
            progress_callback: Callback gọi sau mỗi symbol để report progress.
                Signature: callback(completed: int, total: int, current_symbol: str)

        Returns:
            AutoBacktestResult với metrics tổng hợp và benchmark comparison
        """
        start_time = time.time()

        all_trades: List[Trade] = []
        symbol_returns: List[float] = []
        symbol_sharpes: List[float] = []
        symbol_win_rates: List[float] = []
        tested_symbols: List[str] = []
        total_symbols = len(symbols)

        for idx, symbol in enumerate(symbols):
            # Tạo params mặc định cho auto backtest (dùng toàn bộ data)
            df = self._load_data(symbol)
            if df is None or df.empty:
                logger.warning(f"Bỏ qua symbol {symbol}: không có data")
                continue

            # Lấy date range từ data available
            dates = pd.to_datetime(df["time"])
            start_date = dates.min().date()
            end_date = dates.max().date()

            params = ManualBacktestParams(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=100_000_000.0,  # 100 triệu VND mặc định
                enforce_vn_rules=True,
            )

            result = self.run_manual(params)
            tested_symbols.append(symbol)
            all_trades.extend(result.trades)
            symbol_returns.append(result.total_return)
            symbol_sharpes.append(result.sharpe_ratio)
            symbol_win_rates.append(result.win_rate)

            # Report progress sau mỗi symbol
            if progress_callback is not None:
                progress_callback(idx + 1, total_symbols, symbol)

        # Aggregate metrics
        overall_return = (
            float(np.mean(symbol_returns)) if symbol_returns else 0.0
        )
        overall_sharpe = (
            float(np.mean(symbol_sharpes)) if symbol_sharpes else 0.0
        )
        overall_win_rate = (
            float(np.mean(symbol_win_rates)) if symbol_win_rates else 0.0
        )

        # So sánh với benchmark strategies (STUB: random benchmark results)
        benchmark_results = self._generate_benchmark_results(overall_sharpe)
        strategies_beaten = sum(
            1 for b in benchmark_results if overall_sharpe > b
        )

        duration = time.time() - start_time

        return AutoBacktestResult(
            cycle_number=cycle_number,
            symbols_tested=tested_symbols,
            overall_sharpe=overall_sharpe,
            overall_win_rate=overall_win_rate,
            overall_return=overall_return,
            strategies_beaten=strategies_beaten,
            benchmark_results=benchmark_results,
            trades=all_trades,
            duration_seconds=duration,
        )

    # ==========================================================================
    # Data Loading
    # ==========================================================================

    def _load_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load dữ liệu OHLCV từ file CSV.

        Args:
            symbol: Mã cổ phiếu (ví dụ: "FPT")

        Returns:
            DataFrame với columns time, open, high, low, close, volume
            hoặc None nếu file không tồn tại
        """
        csv_path = self.data_dir / f"{symbol}.csv"
        if not csv_path.exists():
            logger.error(f"File data không tồn tại: {csv_path}")
            return None

        try:
            df = pd.read_csv(csv_path)
            # Validate required columns
            required_cols = ["time", "open", "high", "low", "close", "volume"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                logger.error(
                    f"File {csv_path} thiếu columns: {missing}"
                )
                return None
            return df
        except Exception as e:
            logger.error(f"Lỗi đọc file {csv_path}: {e}")
            return None

    # ==========================================================================
    # Date Filtering
    # ==========================================================================

    def _filter_date_range(
        self, df: pd.DataFrame, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """
        Filter DataFrame theo khoảng thời gian.

        Args:
            df: DataFrame gốc với column 'time'
            start_date: Ngày bắt đầu
            end_date: Ngày kết thúc

        Returns:
            DataFrame đã filter, reset index
        """
        dates = pd.to_datetime(df["time"])
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        mask = (dates >= start_ts) & (dates <= end_ts)
        return df.loc[mask].reset_index(drop=True)

    # ==========================================================================
    # Signal Generation — ML Model (với EMA fallback)
    # ==========================================================================

    def _generate_signals(self, df: pd.DataFrame) -> Tuple[List[str], List[float]]:
        """
        Generate tín hiệu mua/bán dựa trên ML model (StockEvalNet).

        Dùng StockEvalNet predict score cho mỗi sliding window:
        - Score > +threshold → BUY
        - Score < -threshold → SELL
        - Else → HOLD

        Fallback về EMA crossover nếu model không load được.

        Args:
            df: DataFrame OHLCV đã filter theo date range

        Returns:
            Tuple (List signals, List confidences):
            - signals: "BUY", "SELL", hoặc "HOLD" cho mỗi ngày
            - confidences: độ tin cậy (0-100%) cho mỗi ngày
        """
        # Thử dùng ML model
        try:
            return self._generate_signals_ml(df)
        except Exception as e:
            logger.warning(
                f"[BacktestWorker] ML signal generation failed ({e}), "
                f"fallback về EMA crossover"
            )
            return self._generate_signals_ema(df)

    def _generate_signals_ml(self, df: pd.DataFrame) -> Tuple[List[str], List[float]]:
        """
        Generate signals bằng ML model (StockEvalNet).

        Build features → sliding window → model.predict() → threshold → signal.

        Args:
            df: DataFrame OHLCV

        Returns:
            Tuple (List signals, List confidences) cho mỗi ngày

        Raises:
            Exception: Nếu model không load được hoặc feature construction fails
        """
        from analysis import add_indicators
        from engine.config import ModelConfig
        from engine.market_state import (
            INDICATOR_COLUMNS,
            NUM_INDICATORS,
            OHLCV_COLUMNS,
        )

        model_config = ModelConfig()
        lookback = model_config.lookback  # 60

        # Thresholds cho signal generation
        buy_threshold = 0.15   # Score > 0.15 → BUY
        sell_threshold = -0.15  # Score < -0.15 → SELL

        # Tính indicators (cần tối thiểu đủ data cho indicators lớn nhất)
        min_rows_for_indicators = 200  # EMA_200 cần ít nhất 200 rows
        if len(df) < max(lookback, 60):
            # Data quá ngắn cho ML → fallback EMA
            raise ValueError(
                f"Data quá ngắn ({len(df)} rows) cho ML signal generation"
            )

        try:
            df_ind = add_indicators(df)
        except Exception as e:
            raise ValueError(f"Lỗi tính indicators: {e}")

        # Extract raw features (OHLCV + indicators)
        ohlcv = df_ind[OHLCV_COLUMNS].values.astype(np.float64)
        indicator_data = np.full(
            (len(df_ind), NUM_INDICATORS), np.nan, dtype=np.float64
        )
        for i, col in enumerate(INDICATOR_COLUMNS):
            if col in df_ind.columns:
                indicator_data[:, i] = df_ind[col].values.astype(np.float64)

        raw_features = np.concatenate([ohlcv, indicator_data], axis=1)

        # Pad/trim đến num_features
        num_features = model_config.num_features
        if raw_features.shape[1] < num_features:
            padding = np.zeros(
                (raw_features.shape[0], num_features - raw_features.shape[1]),
                dtype=np.float64,
            )
            raw_features = np.concatenate([raw_features, padding], axis=1)
        elif raw_features.shape[1] > num_features:
            raw_features = raw_features[:, :num_features]

        # Forward-fill NaN rồi zero-fill
        df_features = pd.DataFrame(raw_features)
        df_features = df_features.ffill().fillna(0.0)
        raw_features = df_features.values

        # Load model (dùng cache ở instance level)
        if not self._model_loaded:
            import torch
            from engine.config import ModelConfig as MC2
            from engine.evaluation_model import StockEvalNet

            checkpoint_path = self._find_model_checkpoint()
            if checkpoint_path is None:
                raise FileNotFoundError("Không tìm thấy model checkpoint")

            mc = MC2()
            model = StockEvalNet(mc)
            cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            # Xử lý nhiều format checkpoint
            if isinstance(cp, dict) and "model_state_dict" in cp:
                model.load_state_dict(cp["model_state_dict"])
            elif isinstance(cp, dict) and "state_dict" in cp:
                model.load_state_dict(cp["state_dict"])
            else:
                model.load_state_dict(cp)

            model.eval()
            self._ml_model = model
            self._model_loaded = True

        ml_model = self._ml_model

        signals: List[str] = []
        confidences: List[float] = []
        n_rows = len(df_ind)

        # Dùng per-window min-max normalization (nhất quán với hard_example training)
        # Phương pháp này self-contained, hoạt động đúng cho mọi symbol
        # và mọi price range khác nhau.

        for i in range(n_rows):
            if i < lookback - 1:
                # Chưa đủ lookback → HOLD
                signals.append("HOLD")
                confidences.append(0.0)
                continue

            # Extract window [i - lookback + 1, i + 1)
            window = raw_features[i - lookback + 1: i + 1]  # shape (lookback, num_features)

            # Replace NaN/Inf trước khi normalize
            window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

            # Per-window min-max normalize (self-contained, multi-symbol compatible)
            col_min = window.min(axis=0)
            col_max = window.max(axis=0)
            col_range = col_max - col_min
            col_range[col_range == 0] = 1.0
            window_norm = (window - col_min) / col_range

            # Predict
            import torch as torch_inf
            window_tensor = torch_inf.from_numpy(
                window_norm.astype(np.float32)
            ).unsqueeze(0)  # shape (1, lookback, num_features)
            with torch_inf.no_grad():
                score = ml_model(window_tensor)  # shape (1, 1)
            score_val = float(score[0, 0])

            # Tính confidence từ score: confidence = min(abs(score) * 1.5, 1.0) * 100
            confidence = min(abs(score_val) * 1.5, 1.0) * 100.0

            # Generate signal từ score
            if score_val > buy_threshold:
                signals.append("BUY")
            elif score_val < sell_threshold:
                signals.append("SELL")
            else:
                signals.append("HOLD")
            confidences.append(confidence)

        return signals, confidences

    def _find_model_checkpoint(self) -> Optional[str]:
        """
        Tìm model checkpoint cho inference.

        Ưu tiên: stock_eval_net.pt → checkpoint_epoch_*.pt

        Returns:
            Path đến checkpoint hoặc None
        """
        models_dir = Path("engine/models")
        if not models_dir.exists():
            return None

        # Ưu tiên stock_eval_net.pt (retrained model)
        stock_eval = models_dir / "stock_eval_net.pt"
        if stock_eval.exists():
            return str(stock_eval)

        # Fallback: epoch checkpoint
        epoch_checkpoints = sorted(models_dir.glob("checkpoint_epoch_*.pt"))
        if epoch_checkpoints:
            return str(epoch_checkpoints[-1])

        return None

    def _generate_signals_ema(self, df: pd.DataFrame) -> Tuple[List[str], List[float]]:
        """
        Generate tín hiệu mua/bán dựa trên EMA crossover (fallback strategy).

        Args:
            df: DataFrame OHLCV đã filter theo date range

        Returns:
            Tuple (List signals, List confidences):
            - signals: "BUY", "SELL", hoặc "HOLD" cho mỗi ngày
            - confidences: độ tin cậy (0-100%) - cố định 50% cho EMA crossover
        """
        close = df["close"].values
        signals: List[str] = []
        confidences: List[float] = []

        # Tính EMA ngắn và dài
        short_ema = self._calculate_ema(close, SHORT_MA_PERIOD)
        long_ema = self._calculate_ema(close, LONG_MA_PERIOD)

        for i in range(len(close)):
            if i < LONG_MA_PERIOD:
                signals.append("HOLD")
                confidences.append(0.0)
            elif short_ema[i] > long_ema[i] and short_ema[i - 1] <= long_ema[i - 1]:
                signals.append("BUY")
                # Confidence dựa vào độ lệch giữa 2 EMA
                ema_diff_pct = abs(short_ema[i] - long_ema[i]) / long_ema[i] * 100
                confidences.append(min(50.0 + ema_diff_pct * 10, 100.0))
            elif short_ema[i] < long_ema[i] and short_ema[i - 1] >= long_ema[i - 1]:
                signals.append("SELL")
                ema_diff_pct = abs(short_ema[i] - long_ema[i]) / long_ema[i] * 100
                confidences.append(min(50.0 + ema_diff_pct * 10, 100.0))
            else:
                signals.append("HOLD")
                confidences.append(0.0)

        return signals, confidences

    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Tính Exponential Moving Average.

        Args:
            data: Mảng giá close
            period: Chu kỳ EMA

        Returns:
            Mảng EMA cùng kích thước với data
        """
        ema = np.zeros_like(data, dtype=float)
        multiplier = 2.0 / (period + 1)

        # Giá trị đầu = SMA của period đầu tiên
        if len(data) >= period:
            ema[period - 1] = np.mean(data[:period])
            for i in range(period, len(data)):
                ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]
        else:
            # Không đủ data, dùng giá close trực tiếp
            ema[:] = data[:]

        return ema

    # ==========================================================================
    # Trade Execution
    # ==========================================================================

    def _execute_trades(
        self,
        df_period: pd.DataFrame,
        signals: List[str],
        confidences: List[float],
        symbol: str,
        initial_capital: float,
        enforce_rules: bool = True,
    ) -> Tuple[List[Trade], List[float]]:
        """
        Thực thi giao dịch dựa trên signals, enforce VN rules.

        Args:
            df_period: DataFrame đã filter theo date range
            signals: List tín hiệu BUY/SELL/HOLD
            confidences: List độ tin cậy (%) tương ứng với signals
            symbol: Mã cổ phiếu
            initial_capital: Vốn ban đầu (VND)
            enforce_rules: Có áp dụng luật TTCK VN không

        Returns:
            Tuple (danh sách trades đã đóng, equity curve)
        """
        cash = initial_capital
        position_shares = 0
        position_buy_price = 0.0
        position_buy_date: Optional[date] = None
        position_buy_idx = 0
        position_confidence = 0.0
        position_cash_before = 0.0
        position_cash_after = 0.0

        trades: List[Trade] = []
        equity_curve: List[float] = []

        for i in range(len(df_period)):
            row = df_period.iloc[i]
            current_price = float(row["close"])
            current_price_vnd = current_price * self.price_scale
            current_date = pd.to_datetime(row["time"]).date()

            # Tính giá trị portfolio hiện tại
            portfolio_value = cash + (position_shares * current_price_vnd)
            equity_curve.append(portfolio_value)

            signal = signals[i] if i < len(signals) else "HOLD"
            confidence = confidences[i] if i < len(confidences) else 0.0

            # Giá tham chiếu (giá close hôm trước)
            ref_price = (
                float(df_period.iloc[i - 1]["close"]) if i > 0
                else float(row["open"])
            )

            if signal == "BUY" and position_shares == 0:
                # Kiểm tra biên độ giá
                if enforce_rules and not self._check_price_limit(
                    current_price, ref_price
                ):
                    continue

                # Lưu cash trước khi mua
                cash_before_buy = cash

                # Tính số lượng cổ phiếu mua được
                max_shares = int(cash // current_price_vnd)
                shares = (max_shares // LOT_SIZE) * LOT_SIZE

                if shares >= LOT_SIZE:
                    cost = shares * current_price_vnd
                    cash -= cost
                    position_shares = shares
                    position_buy_price = current_price
                    position_buy_date = current_date
                    position_buy_idx = i
                    position_confidence = confidence
                    position_cash_before = cash_before_buy
                    position_cash_after = cash

            elif signal == "SELL" and position_shares > 0:
                # Kiểm tra settlement period (T+2.5 = 3 ngày giao dịch)
                if enforce_rules:
                    days_held = i - position_buy_idx
                    if days_held < 3:
                        # Trong settlement period → skip
                        continue

                # Kiểm tra biên độ giá
                if enforce_rules and not self._check_price_limit(
                    current_price, ref_price
                ):
                    continue

                # Thực hiện bán
                revenue = position_shares * current_price_vnd
                cash += revenue

                buy_price_vnd = position_buy_price * self.price_scale
                pnl = (current_price_vnd - buy_price_vnd) * position_shares
                pnl_pct = (
                    (current_price - position_buy_price) / position_buy_price
                    if position_buy_price > 0
                    else 0.0
                )

                trade = Trade(
                    symbol=symbol,
                    buy_date=position_buy_date,
                    sell_date=current_date,
                    buy_price=position_buy_price,
                    sell_price=current_price,
                    shares=position_shares,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    holding_days=(current_date - position_buy_date).days,
                    confidence=position_confidence,
                    cash_before=position_cash_before,
                    cash_after=position_cash_after,
                )

                # Enforce VN rules (điều chỉnh lot size, sell date nếu cần)
                if enforce_rules:
                    trade = enforce_vn_rules(trade)

                trades.append(trade)

                # Reset position
                position_shares = 0
                position_buy_price = 0.0
                position_buy_date = None
                position_confidence = 0.0
                position_cash_before = 0.0
                position_cash_after = 0.0

        return trades, equity_curve

    def _check_price_limit(self, price: float, reference_price: float) -> bool:
        """
        Kiểm tra giá có nằm trong biên độ ±7% không.

        Args:
            price: Giá hiện tại
            reference_price: Giá tham chiếu (close hôm trước)

        Returns:
            True nếu trong biên độ cho phép
        """
        if reference_price <= 0:
            return False
        change_pct = abs(price - reference_price) / reference_price * 100.0
        return change_pct <= PRICE_LIMIT_PCT

    # ==========================================================================
    # Metrics Calculation
    # ==========================================================================

    def _calculate_metrics(
        self,
        trades: List[Trade],
        initial_capital: float,
        equity_curve: List[float],
    ) -> dict:
        """
        Tính các metrics hiệu suất backtest.

        Metrics bao gồm:
        - total_return: Tổng lợi nhuận (%)
        - sharpe_ratio: Tỷ số Sharpe (annualized)
        - win_rate: Tỷ lệ thắng (%)
        - max_drawdown: Drawdown tối đa (%)

        Args:
            trades: Danh sách trades đã đóng
            initial_capital: Vốn ban đầu
            equity_curve: Đường equity theo thời gian

        Returns:
            Dict chứa total_return, sharpe_ratio, win_rate, max_drawdown
        """
        # Total return
        if equity_curve and initial_capital > 0:
            final_value = equity_curve[-1]
            total_return = ((final_value - initial_capital) / initial_capital) * 100.0
        else:
            total_return = 0.0

        # Win rate
        if trades:
            winning = sum(1 for t in trades if t.pnl is not None and t.pnl > 0)
            win_rate = (winning / len(trades)) * 100.0
        else:
            win_rate = 0.0

        # Max drawdown
        max_drawdown = self._compute_max_drawdown(equity_curve)

        # Sharpe ratio
        sharpe_ratio = self._compute_sharpe_ratio(equity_curve)

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
        }

    def _compute_max_drawdown(self, equity_curve: List[float]) -> float:
        """
        Tính maximum drawdown (%) từ equity curve.

        Args:
            equity_curve: List giá trị portfolio theo thời gian

        Returns:
            Max drawdown dưới dạng phần trăm (số dương)
        """
        if len(equity_curve) < 2:
            return 0.0

        peak = equity_curve[0]
        max_dd = 0.0

        for value in equity_curve:
            if value > peak:
                peak = value
            if peak > 0:
                drawdown = (peak - value) / peak * 100.0
                max_dd = max(max_dd, drawdown)

        return max_dd

    def _compute_sharpe_ratio(self, equity_curve: List[float]) -> float:
        """
        Tính Sharpe ratio annualized (risk-free rate = 0%).

        Args:
            equity_curve: List giá trị portfolio theo thời gian

        Returns:
            Sharpe ratio (annualized với 252 ngày giao dịch/năm)
        """
        if len(equity_curve) < 2:
            return 0.0

        values = np.array(equity_curve)
        # Daily returns
        daily_returns = np.diff(values) / values[:-1]

        if len(daily_returns) == 0:
            return 0.0

        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns, ddof=1)

        if std_return == 0.0 or np.isnan(std_return):
            return 0.0

        # Annualize: nhân sqrt(252)
        sharpe = (mean_return / std_return) * math.sqrt(252.0)
        return float(sharpe)

    # ==========================================================================
    # Benchmark Comparison (STUB)
    # ==========================================================================

    def _generate_benchmark_results(self, ai_sharpe: float) -> List[float]:
        """
        Generate kết quả benchmark strategies để so sánh (STUB).

        Tạo 4 Sharpe ratios giả lập cho các strategies:
        - Wyckoff
        - Technical Analysis
        - Momentum
        - Mean Reversion

        Thực tế sẽ chạy backtest riêng cho mỗi strategy.
        Hiện tại dùng random values xung quanh ai_sharpe để test flow.

        Args:
            ai_sharpe: Sharpe ratio của AI strategy

        Returns:
            List 4 Sharpe ratios (benchmark strategies)
        """
        # STUB: Random benchmarks phân bố quanh giá trị trung bình
        base = max(abs(ai_sharpe), 0.5)
        benchmarks = [
            random.uniform(-base, base * 1.5),  # Wyckoff
            random.uniform(-base * 0.5, base),  # Technical
            random.uniform(-base * 0.3, base * 1.2),  # Momentum
            random.uniform(-base * 0.8, base * 0.8),  # Mean Reversion
        ]
        return benchmarks

    # ==========================================================================
    # Helpers
    # ==========================================================================

    def _empty_result(self, params: ManualBacktestParams) -> BacktestResult:
        """
        Tạo kết quả trống khi không có data hoặc lỗi.

        Args:
            params: Tham số backtest gốc

        Returns:
            BacktestResult với tất cả metrics = 0
        """
        return BacktestResult(
            symbol=params.symbol,
            start_date=params.start_date,
            end_date=params.end_date,
            initial_capital=params.initial_capital,
            final_capital=params.initial_capital,
            total_return=0.0,
            sharpe_ratio=0.0,
            win_rate=0.0,
            max_drawdown=0.0,
            total_trades=0,
            trades=[],
            equity_curve=[params.initial_capital],
        )
