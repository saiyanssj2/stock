"""
DataPipeline - Module fetch và validate market data cho trading platform.

Quản lý việc cập nhật dữ liệu OHLCV cho tất cả tracked symbols (VN30 + watchlist).
Hỗ trợ error isolation: nếu một symbol fail, batch vẫn tiếp tục chạy.

References:
- Req 8.1: One-click data update cho tất cả tracked symbols
- Req 8.2: Validate data completeness (time, open, high, low, close, volume)
- Req 8.4: Error isolation - skip failed symbols, continue batch
"""

import math
import os
import time
from typing import Callable, List, Optional

import pandas as pd

from config.settings import RATE_LIMIT, WATCHLIST_PATH
from config.vn_market_rules import VN30_SYMBOLS
from models.data_models import UpdateResult


# Cột OHLCV bắt buộc phải có trong DataFrame
REQUIRED_COLUMNS: List[str] = ["time", "open", "high", "low", "close", "volume"]


class DataPipeline:
    """
    Fetch và validate market data với rate limiting.

    Đọc danh sách tracked symbols từ VN30 + watchlist file,
    validate dữ liệu OHLCV, và cập nhật batch với error isolation.

    Attributes:
        data_dir: Đường dẫn thư mục chứa file CSV data
    """

    def __init__(self, data_dir: str = "data") -> None:
        """
        Khởi tạo DataPipeline.

        Args:
            data_dir: Đường dẫn thư mục chứa file CSV data (mặc định: "data")
        """
        self.data_dir = data_dir

    def get_tracked_symbols(self) -> List[str]:
        """
        Đọc danh sách symbols cần theo dõi từ VN30 + watchlist file.

        Kết hợp VN30_SYMBOLS với các symbol trong WATCHLIST_PATH (data/code.txt).
        Tự động deduplicate và bỏ qua entries không hợp lệ.
        Xử lý graceful khi file watchlist không tồn tại.

        Returns:
            Danh sách symbols đã deduplicate, sắp xếp theo alphabet
        """
        # Bắt đầu với VN30
        symbols: set = set(VN30_SYMBOLS)

        # Đọc watchlist file
        watchlist_path = WATCHLIST_PATH
        if os.path.exists(watchlist_path):
            try:
                with open(watchlist_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # Parse: hỗ trợ cả comma-separated và newline-separated
                for token in content.replace("\n", ",").split(","):
                    symbol = token.strip().upper()
                    # Bỏ qua entries rỗng hoặc index (VNINDEX, VN30, HNX, UPCOM)
                    if symbol and symbol.isalpha() and len(symbol) <= 7:
                        # Loại trừ tên sàn không có data OHLCV
                        invalid_symbols = {"HNX", "UPCOM"}
                        if symbol not in invalid_symbols:
                            symbols.add(symbol)
            except (IOError, OSError):
                # File không đọc được → chỉ dùng VN30
                pass

        return sorted(symbols)

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate DataFrame có đầy đủ cột OHLCV bắt buộc.

        Kiểm tra DataFrame chứa tất cả required columns:
        time, open, high, low, close, volume.

        Args:
            df: DataFrame cần validate

        Returns:
            True nếu tất cả required columns tồn tại, False nếu thiếu
        """
        if df is None or df.empty:
            return False

        df_columns = [col.lower() for col in df.columns]
        return all(col in df_columns for col in REQUIRED_COLUMNS)

    def update_symbol(self, symbol: str) -> bool:
        """
        Cập nhật data cho một symbol qua vnstock API.

        Tải dữ liệu mới từ API, merge với data cũ (nếu có),
        tính indicators, và lưu ra CSV.

        Args:
            symbol: Mã cổ phiếu cần update (VD: "FPT", "VNM")

        Returns:
            True nếu update thành công, False nếu thất bại

        Raises:
            Không raise exception - trả về False khi có lỗi
        """
        import logging
        from datetime import date, timedelta

        logger = logging.getLogger(__name__)
        csv_path = os.path.join(self.data_dir, f"{symbol}.csv")

        try:
            from vnstock.api.quote import Quote
            from analysis import add_indicators

            # Xác định ngày bắt đầu tải
            fetch_start = "2014-01-01"
            if os.path.exists(csv_path):
                try:
                    df_existing = pd.read_csv(csv_path, usecols=["time"])
                    if len(df_existing) > 0:
                        last_date = pd.to_datetime(df_existing["time"].iloc[-1]).date()
                        # Xác định ngày target (ngày giao dịch gần nhất nên có data)
                        # Trước 9h sáng → target = hôm qua (chưa có data hôm nay)
                        # Sau 9h sáng → target = hôm nay
                        now = __import__("datetime").datetime.now()
                        if now.hour < 9:
                            target_date = date.today() - timedelta(days=1)
                        else:
                            target_date = date.today()
                        # Bỏ qua cuối tuần (thứ 7, CN ko có data)
                        while target_date.weekday() >= 5:
                            target_date -= timedelta(days=1)
                        # Chỉ skip nếu data đã có SAU ngày target (ngày mai+)
                        # Nếu last_date == target → vẫn fetch lại (data có thể chưa final)
                        if last_date > target_date:
                            return True
                        # Tải từ ngày cuối (overlap 1 ngày)
                        fetch_start = str(last_date - timedelta(days=1))
                except Exception:
                    pass

            fetch_end = str(date.today())

            # Gọi API với retry
            df_new = None
            max_retries = 3
            for attempt in range(max_retries + 1):
                try:
                    q = Quote(symbol=symbol, source="VCI")
                    df_new = q.history(start=fetch_start, end=fetch_end, interval="1D")
                    break
                except Exception as e:
                    err_msg = str(e).lower()
                    # Log ra debug file
                    _debug_path = os.path.join("pipeline_debug.log")
                    try:
                        with open(_debug_path, "a", encoding="utf-8") as _df:
                            from datetime import datetime as _dtnow
                            _df.write(f"[{_dtnow.now():%Y-%m-%d %H:%M:%S}]   [UPDATE] {symbol} attempt {attempt+1}/{max_retries+1} ERROR: {str(e)[:120]}\n")
                    except Exception:
                        pass

                    if attempt < max_retries:
                        if ("rate" in err_msg or "limit" in err_msg or "429" in err_msg
                                or "giới hạn" in err_msg or "10054" in err_msg
                                or "forcibly closed" in err_msg or "connection" in err_msg):
                            # Parse thời gian chờ từ message nếu có (VD: "Chờ 13 giây")
                            import re
                            wait_match = re.search(r"chờ\s+(\d+)\s+giây", err_msg)
                            if not wait_match:
                                wait_match = re.search(r"wait\s+(\d+)", err_msg)
                            wait_time = int(wait_match.group(1)) + 2 if wait_match else 45
                            logger.info(f"[DataPipeline] Rate limit {symbol}, chờ {wait_time}s (attempt {attempt+1})")
                            try:
                                with open(_debug_path, "a", encoding="utf-8") as _df:
                                    _df.write(f"[{_dtnow.now():%Y-%m-%d %H:%M:%S}]   [UPDATE] {symbol} → chờ {wait_time}s rồi retry...\n")
                            except Exception:
                                pass
                            time.sleep(wait_time)
                        else:
                            time.sleep(5)
                    else:
                        logger.warning(f"[DataPipeline] Fetch {symbol} thất bại sau {max_retries+1} lần: {e}")
                        return False

            if df_new is None or len(df_new) == 0:
                logger.warning(f"[DataPipeline] {symbol}: không có dữ liệu mới")
                return False

            df_new.columns = [c.lower() for c in df_new.columns]
            df_new["time"] = pd.to_datetime(df_new["time"])
            base_cols = ["time", "open", "high", "low", "close", "volume"]

            # Merge với data cũ nếu có
            if os.path.exists(csv_path):
                try:
                    df_old = pd.read_csv(csv_path)
                    df_old["time"] = pd.to_datetime(df_old["time"])
                    df = pd.concat([df_old[base_cols], df_new[base_cols]], ignore_index=True)
                except Exception:
                    df = df_new[base_cols].copy()
            else:
                df = df_new[base_cols].copy()

            # Deduplicate: giữ dòng CUỐI (data mới từ API chính xác hơn data cũ)
            df = df.drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                df = add_indicators(df)
            df.round(4).to_csv(csv_path, index=False)
            logger.info(f"[DataPipeline] {symbol}: {len(df)} phiên, đến {df['time'].iloc[-1].date()}")
            # Log ra debug file
            try:
                with open("pipeline_debug.log", "a", encoding="utf-8") as _df:
                    from datetime import datetime as _dtnow2
                    _df.write(f"[{_dtnow2.now():%Y-%m-%d %H:%M:%S}]   [UPDATE] {symbol} OK → {df['time'].iloc[-1].date()} ({len(df)} phiên)\n")
            except Exception:
                pass
            return True

        except ImportError as e:
            logger.error(f"[DataPipeline] Import error: {e}")
            return False
        except Exception as e:
            logger.error(f"[DataPipeline] Lỗi update {symbol}: {e}")
            return False

    def update_all(
        self, callback: Optional[Callable[[str, float], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> UpdateResult:
        """
        Cập nhật data cho tất cả tracked symbols với rate limiting và error isolation.

        Rate limit: nghỉ 3s giữa mỗi request, nghỉ 65s sau mỗi 18 requests
        (giới hạn API: 20 requests/phút).

        Args:
            callback: Hàm callback(symbol, progress_pct) cho progress reporting.
            should_stop: Hàm callback() trả về True nếu cần dừng ngay.

        Returns:
            UpdateResult chứa thống kê success/failure của batch
        """
        symbols = self.get_tracked_symbols()
        total = len(symbols)

        success_count = 0
        failed_symbols: List[str] = []
        errors: dict = {}

        start_time = time.time()

        # Rate limiting: 20 requests/phút = 1 request mỗi 3s
        # Nghỉ 3.1s sau mỗi request để không bao giờ chạm limit
        api_call_count = 0

        for i, symbol in enumerate(symbols):
            # === Kiểm tra yêu cầu dừng ===
            if should_stop is not None and should_stop():
                break

            # Kiểm tra xem symbol có cần gọi API không (dùng cùng logic với update_symbol)
            csv_path = os.path.join(self.data_dir, f"{symbol}.csv")
            needs_api = True
            if os.path.exists(csv_path):
                try:
                    from datetime import date, timedelta
                    import datetime as _dt_mod
                    df_tail = pd.read_csv(csv_path, usecols=["time"])
                    if len(df_tail) > 0:
                        last_date = pd.to_datetime(df_tail["time"].iloc[-1]).date()
                        # Cùng logic target date với update_symbol
                        now = _dt_mod.datetime.now()
                        target_date = date.today()
                        if now.hour < 9:
                            target_date = date.today() - timedelta(days=1)
                        while target_date.weekday() >= 5:
                            target_date -= timedelta(days=1)
                        if last_date >= target_date:
                            # Chỉ skip nếu data đã có SAU target (ngày mai+)
                            # last_date == target → vẫn gọi API (data có thể chưa final)
                            if last_date > target_date:
                                needs_api = False
                except Exception:
                    pass

            # Nghỉ 3.1s sau mỗi API call (20 req/phút = 1 req mỗi 3s)
            if needs_api and api_call_count > 0:
                time.sleep(3.1)

            try:
                result = self.update_symbol(symbol)
                if result:
                    success_count += 1
                    if needs_api:
                        api_call_count += 1
                else:
                    failed_symbols.append(symbol)
                    errors[symbol] = f"Update failed for {symbol}"
                    if needs_api:
                        api_call_count += 1
            except Exception as e:
                failed_symbols.append(symbol)
                errors[symbol] = str(e)
                if needs_api:
                    api_call_count += 1

            # Report progress qua callback
            if callback is not None:
                progress_pct = ((i + 1) / total) * 100.0 if total > 0 else 100.0
                callback(symbol, progress_pct)

        duration = time.time() - start_time

        return UpdateResult(
            total_symbols=total,
            success_count=success_count,
            failed_symbols=failed_symbols,
            errors=errors,
            duration_seconds=duration,
        )

    def estimate_update_duration(self, num_symbols: int) -> float:
        """
        Ước tính thời gian update dựa trên rate limit.

        Tính toán: ceil(num_symbols / RATE_LIMIT) phút.
        Mỗi phút tối đa RATE_LIMIT requests.

        Args:
            num_symbols: Số lượng symbols cần update

        Returns:
            Thời gian ước tính tính bằng phút (float)
        """
        if num_symbols <= 0:
            return 0.0
        return float(math.ceil(num_symbols / RATE_LIMIT))
