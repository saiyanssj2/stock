"""
RecommendationEngine - Tạo khuyến nghị cổ phiếu từ model đã train.

Scan VN30 + watchlist (data/code.txt), assign confidence score,
sort descending, và hỗ trợ hot-swap model khi training xong.

Feature Scaling Consistency:
- Load scaler params từ file (saved lúc training)
- Dùng CÙNG normalization với training để đảm bảo prediction nhất quán
- Fallback về per-window normalize nếu scaler file chưa có

References:
- Req 6.1: Scan VN30 + watchlist
- Req 6.2: Confidence score [0.0, 1.0]
- Req 6.3: Display action, confidence, position_score, holding days
- Req 6.4: No strong recommendations khi tất cả < 0.5
- Req 6.5: Luôn dùng latest model version
- Req 6.6: Hot-swap model → regenerate recommendations
- Req 6.7: Refresh recommendations mỗi Trading_Day sau data update
"""

import hashlib
import logging
import os
import random
from datetime import date
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from config.settings import CONFIDENCE_THRESHOLD, WATCHLIST_PATH
from config.vn_market_rules import VN30_SYMBOLS
from engine.feature_scaler import FeatureScaler, scaler_exists
from engine.workers.vn_rules import add_trading_days, compute_buy_date, compute_earliest_sell_date
from models.portfolio_models import PortfolioState
from models.recommendation_models import Action, HoldingInfo, Recommendation

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Tạo recommendations từ model đã train.

    Scan VN30 + watchlist, assign confidence score cho mỗi symbol,
    sort descending by confidence. Hỗ trợ hot-swap model và cache.

    Attributes:
        _model_path: Đường dẫn đến model file (None = stub mode)
        _model_version: Phiên bản model hiện tại
        _recommendations: Cache danh sách recommendations
    """

    def __init__(self, model_path: Optional[str] = None, data_dir: Optional[str] = None) -> None:
        """
        Khởi tạo RecommendationEngine.

        Args:
            model_path: Đường dẫn model file. None = dùng heuristic stub.
            data_dir: Thư mục chứa file CSV data. None = dùng "data/".
        """
        self._model_path: Optional[str] = model_path
        self._data_dir: str = data_dir or "data"
        self._model_version: str = self._compute_model_version(model_path)
        self._recommendations: List[Recommendation] = []
        
        # Feature scaler để đảm bảo normalization consistency
        self._scaler: Optional[FeatureScaler] = None
        self._scaler_loaded: bool = False

    def _compute_model_version(self, model_path: Optional[str]) -> str:
        """
        Tính model version từ file path.

        Nếu model file tồn tại → dùng hash nội dung.
        Nếu không → trả về "stub-v1".

        Args:
            model_path: Đường dẫn model file

        Returns:
            String version identifier
        """
        if model_path and os.path.exists(model_path):
            # Dùng file modification time + size làm version đơn giản
            stat = os.stat(model_path)
            raw = f"{model_path}:{stat.st_size}:{stat.st_mtime}"
            return hashlib.md5(raw.encode()).hexdigest()[:8]
        return "stub-v1"

    def get_tracked_symbols(self) -> List[str]:
        """
        Đọc VN30 + watchlist, trả về combined unique list.

        VN30 luôn có sẵn từ config. Watchlist đọc từ data/code.txt.
        Nếu file watchlist không tồn tại → chỉ dùng VN30.
        Loại trừ các chỉ số thị trường (VNINDEX, VN30, HNX, UPCOM, HNX30).

        Returns:
            Danh sách symbol duy nhất (không trùng lặp), sorted alphabetically
        """
        # Các chỉ số thị trường — không phải cổ phiếu, không trade được
        INDEX_SYMBOLS = {"VNINDEX", "VN30", "HNX", "UPCOM", "HNX30"}

        symbols: set[str] = set(VN30_SYMBOLS)

        # Đọc watchlist từ file
        watchlist_symbols = self._read_watchlist()
        symbols.update(watchlist_symbols)

        # Loại trừ các chỉ số thị trường
        symbols = symbols - INDEX_SYMBOLS

        return sorted(symbols)

    def _read_watchlist(self) -> List[str]:
        """
        Đọc danh sách symbol từ file watchlist.

        File format: mỗi symbol phân cách bởi dấu phẩy hoặc xuống dòng.
        Bỏ qua các symbol rỗng hoặc chỉ chứa whitespace.
        Handle gracefully khi file không tồn tại.

        Returns:
            Danh sách symbol từ watchlist (có thể rỗng)
        """
        watchlist_path = Path(WATCHLIST_PATH)
        if not watchlist_path.exists():
            return []

        try:
            content = watchlist_path.read_text(encoding="utf-8")
            # Parse: split by comma and newline, strip whitespace
            raw_symbols = content.replace("\n", ",").split(",")
            return [
                s.strip()
                for s in raw_symbols
                if s.strip() and s.strip().isalpha()
            ]
        except (OSError, IOError):
            return []

    def scan_symbols(
        self,
        cash: float = 1_000_000_000.0,
        holdings: Optional[dict] = None,
        portfolio_state: Optional["PortfolioState"] = None,
    ) -> List[Recommendation]:
        """
        Scan VN30 + watchlist, generate recommendations dựa trên ML model.

        Logic portfolio-aware:
        - Nếu có tiền mặt → recommend MUA các mã có score cao
        - Nếu đang giữ mã → recommend BÁN nếu score < 0 (tín hiệu yếu)
        - Không recommend BÁN mã mình không cầm
        - Nếu đang giữ nhưng chưa qua settlement period → GIỮ (không bán được)

        Args:
            cash: Tiền mặt hiện có (VND). Mặc định 1 tỉ. Bỏ qua nếu có portfolio_state.
            holdings: Dict {symbol: HoldingInfo} hoặc {symbol: int (shares)}.
                None = không giữ gì. Bỏ qua nếu có portfolio_state.
            portfolio_state: PortfolioState object. Nếu có → ưu tiên dùng thay cash/holdings.

        Returns:
            Danh sách Recommendation đã sort theo confidence giảm dần
        """
        # Nếu có portfolio_state → extract cash và holdings từ đó
        if portfolio_state is not None:
            cash = portfolio_state.cash
            holdings = {}
            for h in portfolio_state.holdings:
                holdings[h.symbol] = HoldingInfo(
                    shares=h.shares,
                    buy_price=h.buy_price,
                    buy_date=h.buy_date,
                )
        elif holdings is None:
            holdings = {}

        # Normalize holdings: hỗ trợ cả dict[str, int] legacy và dict[str, HoldingInfo]
        normalized_holdings: dict[str, HoldingInfo] = {}
        for sym, val in holdings.items():
            if isinstance(val, HoldingInfo):
                normalized_holdings[sym] = val
            elif isinstance(val, int):
                normalized_holdings[sym] = HoldingInfo(shares=val)
            else:
                normalized_holdings[sym] = HoldingInfo(shares=int(val))

        # Tính tổng giá trị portfolio để giới hạn 20% mỗi vị thế
        portfolio_value = cash
        for sym, info in normalized_holdings.items():
            price = self._get_latest_close_price(sym)
            if price is not None:
                portfolio_value += info.shares * price * 1000

        symbols = self.get_tracked_symbols()
        today = date.today()
        recommendations: List[Recommendation] = []

        # Score tất cả symbols bằng ML model
        scored_symbols = []
        for symbol in symbols:
            score, confidence = self._predict_with_model(symbol)
            scored_symbols.append((symbol, score, confidence))

        # Sort theo score giảm dần
        scored_symbols.sort(key=lambda x: x[1], reverse=True)

        has_cash = cash > 10_000_000  # Coi có tiền nếu > 10M

        for symbol, score, confidence in scored_symbols:
            holding = normalized_holdings.get(symbol)
            is_holding = holding is not None and holding.shares > 0

            # Kiểm tra settlement period: nếu có ngày mua và chưa qua T+2.5 → không bán được
            in_settlement = False
            if is_holding and holding.buy_date is not None:
                from engine.workers.vn_rules import is_within_settlement_period
                in_settlement = is_within_settlement_period(holding.buy_date, today)

            # Logic portfolio-aware:
            if has_cash and not is_holding and score > 0.15:
                # Có tiền + không cầm + score tốt → MUA
                action = Action.BUY
            elif is_holding and score < -0.15 and not in_settlement:
                # Đang giữ + score xấu + đã qua settlement → BÁN
                action = Action.SELL
            elif is_holding and score < -0.15 and in_settlement:
                # Đang giữ + score xấu + chưa qua settlement → GIỮ (không bán được)
                action = Action.HOLD
            elif is_holding:
                # Đang giữ + score không xấu → GIỮ
                action = Action.HOLD
            else:
                # Không giữ + score không đủ tốt → skip
                continue

            # Tính ngày mua T+1 và ngày bán sớm nhất T+3
            buy_date = compute_buy_date(today)
            earliest_sell_date = compute_earliest_sell_date(buy_date)
            expected_holding_days = 3 + int((1.0 - confidence) * 10)

            # Đọc giá hiện tại
            recommended_buy_price = self._get_latest_close_price(symbol)

            # Tính recommended_shares và recommended_sell_date
            recommended_shares: Optional[int] = None
            recommended_sell_date: Optional[date] = None

            if action == Action.BUY and recommended_buy_price is not None:
                # Max 20% tổng giá trị portfolio cho 1 vị thế
                max_amount = portfolio_value * 0.20
                # Số CP tối đa (lot size = 100)
                max_shares = int(max_amount / (recommended_buy_price * 1000)) // 100 * 100
                # Không vượt quá tiền mặt hiện có (trừ phí 0.15%)
                affordable_shares = int(cash / (recommended_buy_price * 1000 * 1.0015)) // 100 * 100
                recommended_shares = min(max_shares, affordable_shares)
                if recommended_shares < 100:
                    recommended_shares = None  # Không đủ tiền mua 1 lot

                # Ngày nên bán = buy_date + expected_holding_days ngày giao dịch
                recommended_sell_date = add_trading_days(buy_date, expected_holding_days)

            elif action == Action.SELL and is_holding:
                # Bán toàn bộ vị thế hiện tại
                recommended_shares = holding.shares
                # Ngày bán = ngày bán sớm nhất (T+3)
                recommended_sell_date = earliest_sell_date

            rec = Recommendation(
                symbol=symbol,
                action=action,
                confidence=confidence,
                position_score=score,
                expected_holding_days=expected_holding_days,
                earliest_sell_date=earliest_sell_date,
                analysis_date=today,
                buy_date=buy_date,
                model_version=self._model_version,
                recommended_buy_price=recommended_buy_price,
                recommended_shares=recommended_shares,
                recommended_sell_date=recommended_sell_date,
            )
            recommendations.append(rec)

        # Sort descending by confidence
        recommendations.sort(key=lambda r: r.confidence, reverse=True)
        self._recommendations = recommendations
        return recommendations

    def _predict_with_model(self, symbol: str) -> tuple:
        """
        Predict position score cho symbol bằng ML model (StockEvalNet).

        Load data CSV → extract features → normalize (dùng saved scaler) → model forward pass.
        Fallback về hash-based heuristic nếu model/data không có.

        QUAN TRỌNG: Dùng CÙNG scaler params với training để đảm bảo consistency.
        Nếu scaler file chưa có → fallback về per-window normalize + warning.

        Args:
            symbol: Mã cổ phiếu

        Returns:
            Tuple (position_score [-1,1], confidence [0,1])
        """
        try:
            import torch
            from engine.config import ModelConfig
            from engine.evaluation_model import StockEvalNet
            from engine.market_state import INDICATOR_COLUMNS, OHLCV_COLUMNS, NUM_INDICATORS

            model_config = ModelConfig()
            lookback = model_config.lookback  # 60

            # Load CSV
            csv_path = Path(self._data_dir) / f"{symbol}.csv"
            if not csv_path.exists():
                return self._stub_predict(symbol)

            df = pd.read_csv(csv_path)
            if len(df) < lookback:
                return self._stub_predict(symbol)

            # Extract features (lấy lookback dòng cuối)
            ohlcv_cols = [c for c in OHLCV_COLUMNS if c in df.columns]
            if len(ohlcv_cols) < 5:
                return self._stub_predict(symbol)

            # === Lấy lookback dòng cuối ===
            df_window = df.iloc[-lookback:].reset_index(drop=True)

            ohlcv = df_window[OHLCV_COLUMNS].values.astype(np.float64)
            indicator = np.full((lookback, NUM_INDICATORS), 0.0, dtype=np.float64)
            for i, col in enumerate(INDICATOR_COLUMNS):
                if col in df_window.columns:
                    indicator[:, i] = df_window[col].values.astype(np.float64)

            raw = np.concatenate([ohlcv, indicator], axis=1)

            # Pad/trim số features
            num_features = model_config.num_features
            if raw.shape[1] < num_features:
                padding = np.zeros((lookback, num_features - raw.shape[1]), dtype=np.float64)
                raw = np.concatenate([raw, padding], axis=1)
            elif raw.shape[1] > num_features:
                raw = raw[:, :num_features]

            # === NORMALIZE dùng saved scaler (QUAN TRỌNG cho consistency) ===
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            raw_norm = self._normalize_features(raw)

            # Load model (cache)
            if not hasattr(self, "_ml_model") or self._ml_model is None:
                model_path = Path("engine/models/stock_eval_net.pt")
                if not model_path.exists():
                    return self._stub_predict(symbol)
                model = StockEvalNet(model_config)
                cp = torch.load(model_path, map_location="cpu", weights_only=False)
                if isinstance(cp, dict) and "model_state_dict" in cp:
                    model.load_state_dict(cp["model_state_dict"])
                elif isinstance(cp, dict) and "state_dict" in cp:
                    model.load_state_dict(cp["state_dict"])
                else:
                    model.load_state_dict(cp)
                model.eval()
                self._ml_model = model

            # Forward pass
            tensor = torch.from_numpy(raw_norm.astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                score = float(self._ml_model(tensor)[0, 0])

            # Score → confidence mapping: abs(score) là confidence
            confidence = min(abs(score) * 1.5, 1.0)

            return score, confidence

        except Exception as e:
            logger.debug(f"ML predict failed for {symbol}: {e}")
            return self._stub_predict(symbol)

    def _normalize_features(self, raw: np.ndarray) -> np.ndarray:
        """
        Normalize features dùng saved scaler hoặc fallback per-window.
        
        Args:
            raw: Array shape (lookback, num_features), đã xử lý NaN
            
        Returns:
            Normalized array shape (lookback, num_features)
        """
        # Lazy load scaler (chỉ load 1 lần)
        if not self._scaler_loaded:
            self._load_scaler()
        
        if self._scaler is not None and self._scaler.is_fitted:
            # Dùng saved scaler — CÙNG normalization với training
            return self._scaler.transform(raw)
        else:
            # Fallback: per-window normalize (có thể gây inconsistency)
            logger.debug("[Recommend] Scaler chưa có, dùng per-window normalize (fallback)")
            col_min = raw.min(axis=0)
            col_max = raw.max(axis=0)
            col_range = col_max - col_min
            col_range[col_range == 0] = 1.0
            return ((raw - col_min) / col_range).astype(np.float32)
    
    def _load_scaler(self) -> None:
        """
        Load scaler từ file (lazy loading).
        
        Thử load từ engine/models/scaler_params.json.
        Nếu không có, _scaler = None và sẽ dùng fallback.
        """
        self._scaler_loaded = True
        
        if not scaler_exists():
            logger.warning(
                "[Recommend] Scaler file chưa có. Recommendation sẽ dùng per-window normalize. "
                "Chạy training ít nhất 1 cycle để tạo scaler."
            )
            self._scaler = None
            return
        
        try:
            self._scaler = FeatureScaler.load()
            logger.info(
                f"[Recommend] Loaded scaler: {self._scaler.params.num_features} features, "
                f"fitted at {self._scaler.params.fitted_at}"
            )
        except Exception as e:
            logger.warning(f"[Recommend] Không load được scaler ({e}), dùng fallback")
            self._scaler = None
    
    def reload_scaler(self) -> bool:
        """
        Force reload scaler từ file.
        
        Gọi sau khi training xong để dùng scaler mới.
        
        Returns:
            True nếu load thành công
        """
        self._scaler_loaded = False
        self._load_scaler()
        return self._scaler is not None and self._scaler.is_fitted

    def _get_latest_close_price(self, symbol: str) -> Optional[float]:
        """
        Đọc giá đóng cửa gần nhất từ file CSV của symbol.

        Giá trong CSV lưu theo đơn vị 1000 VND (VD: 76.5 = 76,500 VND).
        Trả về giá gốc từ CSV (chưa nhân 1000).

        Args:
            symbol: Mã cổ phiếu

        Returns:
            Giá đóng cửa gần nhất (đơn vị 1000 VND), None nếu không đọc được.
        """
        csv_path = Path(self._data_dir) / f"{symbol}.csv"
        if not csv_path.exists():
            return None

        try:
            # Chỉ đọc cột close, lấy dòng cuối
            df = pd.read_csv(csv_path, usecols=["close"])
            if df.empty:
                return None
            return float(df["close"].iloc[-1])
        except Exception as e:
            logger.debug(f"Không đọc được giá cho {symbol}: {e}")
            return None

    def _determine_action(self, position_score: float) -> Action:
        """
        Xác định hành động khuyến nghị dựa trên position_score.

        - position_score > 0.3: BUY
        - position_score < -0.3: SELL
        - Còn lại: HOLD

        Args:
            position_score: Điểm vị thế [-1.0, 1.0]

        Returns:
            Action enum (BUY, HOLD, SELL)
        """
        if position_score > 0.3:
            return Action.BUY
        elif position_score < -0.3:
            return Action.SELL
        return Action.HOLD

    def _stub_predict(self, symbol: str) -> tuple:
        """
        Fallback prediction khi ML model không khả dụng.

        Dùng hash-based heuristic cho consistent scores.

        Returns:
            Tuple (position_score, confidence)
        """
        seed_str = f"{self._model_version}:{symbol}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        position_score = rng.uniform(-0.5, 0.5)
        confidence = rng.uniform(0.2, 0.6)
        return position_score, confidence

    def refresh(self) -> None:
        """
        Re-generate recommendations.

        Gọi sau khi data update hoặc model hot-swap.
        Load lại model và scaler nếu có thay đổi, rồi scan lại toàn bộ symbols.
        """
        # Cập nhật model version (hỗ trợ hot-swap)
        self._model_version = self._compute_model_version(self._model_path)
        
        # Reload scaler để dùng params mới nhất từ training
        self.reload_scaler()
        
        # Clear cached model để force reload
        self._ml_model = None
        
        # Re-scan tất cả symbols
        self.scan_symbols()

    def get_recommendations(self) -> List[Recommendation]:
        """
        Trả về cached recommendations.

        Nếu chưa scan lần nào, trả về list rỗng.
        Caller nên gọi scan_symbols() hoặc refresh() trước.

        Returns:
            Danh sách Recommendation đã cache (sorted by confidence desc)
        """
        return self._recommendations

    def has_strong_recommendations(self) -> bool:
        """
        Kiểm tra có recommendation nào >= CONFIDENCE_THRESHOLD không.

        Req 6.4: Nếu tất cả confidence < 0.5 → "no strong recommendations".

        Returns:
            True nếu có ít nhất 1 recommendation với confidence >= threshold
        """
        return any(
            rec.confidence >= CONFIDENCE_THRESHOLD
            for rec in self._recommendations
        )

    def hot_swap_model(self, new_model_path: str) -> None:
        """
        Hot-swap sang model mới và regenerate recommendations.

        Req 6.6: Khi training xong, swap model → tự động refresh.

        Args:
            new_model_path: Đường dẫn đến model file mới
        """
        self._model_path = new_model_path
        self.refresh()
