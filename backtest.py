import pandas as pd
import time
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from vnstock import Listing
from vnstock.api.quote import Quote
from analysis import add_indicators

def _safe_history(symbol: str, start: str, end: str, retries: int = 3) -> pd.DataFrame | None:
    for i in range(retries):
        try:
            q = Quote(symbol=symbol, source="VCI")
            df = q.history(start=start, end=end, interval="1D")
            return df
        except Exception as e:
            if "rate limit" in str(e).lower() or "Rate limit" in str(e):
                time.sleep(15)
            else:
                return None
    return None

def _backtest_one(symbol: str, scan_date: date, filters: dict) -> dict | None:
    try:
        # Lấy đủ dữ liệu: 200 ngày trước scan_date + 20 ngày sau để tính T+5/10/20
        start = str(scan_date - timedelta(days=200))
        end   = str(scan_date + timedelta(days=40))

        q = Quote(symbol=symbol, source="VCI")
        df = _safe_history(symbol, start, end)
        if df is None or len(df) < 60:
            return None

        df.columns = [c.lower() for c in df.columns]
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

        # Tìm index của ngày quét (ngày giao dịch gần nhất <= scan_date)
        scan_dt = pd.Timestamp(scan_date)
        past = df[df["time"] <= scan_dt]
        if len(past) < 60:
            return None
        idx = past.index[-1]

        # Tính chỉ báo chỉ trên dữ liệu tới ngày quét
        df_past = df.iloc[:idx + 1].copy()
        avg_vol = df_past["volume"].tail(20).mean()
        if avg_vol < filters.get("min_volume", 100_000):
            return None

        df_past = add_indicators(df_past)
        last = df_past.iloc[-1]
        prev = df_past.iloc[-2]

        score = 0
        reasons = []

        rsi = last.get("RSI_14")
        if pd.notna(rsi) and filters.get("rsi") and rsi < filters["rsi_max"]:
            score += 1
            reasons.append(f"RSI={rsi:.0f}")

        macd = last.get("MACD")
        macd_sig = last.get("MACD_signal")
        macd_prev = prev.get("MACD")
        macd_sig_prev = prev.get("MACD_signal")
        if all(pd.notna(x) for x in [macd, macd_sig, macd_prev, macd_sig_prev]):
            if filters.get("macd_cross") and macd > macd_sig and macd_prev <= macd_sig_prev:
                score += 2
                reasons.append("MACD cắt lên")
            elif filters.get("macd") and macd > macd_sig:
                score += 1
                reasons.append("MACD>Signal")

        ema20 = last.get("EMA_20")
        ema50 = last.get("EMA_50")
        ema20_prev = prev.get("EMA_20")
        ema50_prev = prev.get("EMA_50")
        if all(pd.notna(x) for x in [ema20, ema50, ema20_prev, ema50_prev]):
            if filters.get("ema_cross") and ema20 > ema50 and ema20_prev <= ema50_prev:
                score += 2
                reasons.append("EMA20 cắt EMA50")
            elif filters.get("ema") and ema20 > ema50:
                score += 1
                reasons.append("EMA20>EMA50")

        bb_lower = last.get("BB_lower")
        close = last.get("close")
        if pd.notna(bb_lower) and pd.notna(close) and filters.get("bb") and close < bb_lower:
            score += 1
            reasons.append("Dưới BB lower")

        if score < filters.get("min_score", 2):
            return None

        # Tính giá tại T+5, T+10, T+20
        future = df.iloc[idx + 1:].reset_index(drop=True)

        def pct_at(n):
            if len(future) >= n:
                return round((future.iloc[n - 1]["close"] - close) / close * 100, 2)
            return None

        t5  = pct_at(5)
        t10 = pct_at(10)
        t20 = pct_at(20)

        return {
            "Mã": symbol,
            "Giá quét": round(close, 1),
            "Điểm": score,
            "Tín hiệu": ", ".join(reasons),
            "T+5 (%)": t5,
            "T+10 (%)": t10,
            "T+20 (%)": t20,
        }
    except Exception:
        return None

def run_backtest(scan_date: date, filters: dict, max_workers: int = 30) -> tuple[pd.DataFrame, dict]:
    from vnstock import Listing
    listing = Listing()
    industry_df = listing.symbols_by_industries()

    # Moi nganh lay top 5 ma co volume lon nhat (proxy von hoa)
    symbols_per_industry = (
        industry_df.groupby("industry_name")["symbol"]
        .apply(list)
        .to_dict()
    )

    # Lay volume trung binh 20 phien de chon top 5 moi nganh
    from vnstock.api.quote import Quote
    from datetime import timedelta

    def get_avg_vol(symbol):
        try:
            df = _safe_history(symbol, str(scan_date - timedelta(days=40)), str(scan_date))
            if df is None or len(df) < 5:
                return 0
            df.columns = [c.lower() for c in df.columns]
            return df["volume"].tail(20).mean()
        except Exception:
            return 0

    selected = []
    for industry, syms in symbols_per_industry.items():
        with ThreadPoolExecutor(max_workers=5) as ex:
            vols = list(ex.map(get_avg_vol, syms))
        ranked = sorted(zip(syms, vols), key=lambda x: x[1], reverse=True)
        top5 = [s for s, _ in ranked[:5]]
        selected.extend([(s, industry) for s in top5])
        time.sleep(1)  # tranh rate limit giua cac nganh

    symbols = [s for s, _ in selected]
    industry_map = {s: ind for s, ind in selected}
    total = len(symbols)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_backtest_one, s, scan_date, filters): s for s in symbols}
        for f in as_completed(futures):
            r = f.result()
            if r:
                r["Ngành"] = industry_map.get(r["Mã"], "")
                results.append(r)

    if not results:
        return pd.DataFrame(), {}

    df = pd.DataFrame(results)[["Mã", "Ngành", "Giá quét", "Điểm", "Tín hiệu", "T+5 (%)", "T+10 (%)", "T+20 (%)"]]
    df = df.sort_values("Điểm", ascending=False).reset_index(drop=True)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_backtest_one, s, scan_date, filters): s for s in symbols}
        for f in as_completed(futures):
            r = f.result()
            if r:
                results.append(r)

    if not results:
        return pd.DataFrame(), {}

    df = pd.DataFrame(results).sort_values("Điểm", ascending=False).reset_index(drop=True)

    # Thong ke
    stats = {}
    for col in ["T+5 (%)", "T+10 (%)", "T+20 (%)"]:
        valid = df[col].dropna()
        if len(valid) == 0:
            continue
        stats[col] = {
            "win_rate": round((valid > 0).sum() / len(valid) * 100, 1),
            "avg_return": round(valid.mean(), 2),
            "best": round(valid.max(), 2),
            "worst": round(valid.min(), 2),
            "count": len(valid),
        }

    return df, stats
