import pandas as pd
import time
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from vnstock import Listing
from vnstock.api.quote import Quote
from analysis import add_indicators

def _safe_history(symbol, start, end, retries=3):
    for i in range(retries):
        try:
            q = Quote(symbol=symbol, source="VCI")
            return q.history(start=start, end=end, interval="1D")
        except Exception as e:
            if "rate limit" in str(e).lower() or "Rate limit" in str(e):
                time.sleep(15)
            else:
                return None
    return None

def get_all_symbols() -> list[str]:
    df = Listing().all_symbols()
    return df["symbol"].tolist()

def _scan_one(symbol: str, start: str, end: str, filters: dict) -> dict | None:
    try:
        df = _safe_history(symbol, start, end)
        if df is None or len(df) < 60:
            return None
        df.columns = [c.lower() for c in df.columns]
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

        # Lọc volume tối thiểu
        avg_vol = df["volume"].tail(20).mean()
        if avg_vol < filters.get("min_volume", 100_000):
            return None

        df = add_indicators(df)
        last = df.iloc[-1]
        prev = df.iloc[-2]

        score = 0
        reasons = []

        rsi = last.get("RSI_14")
        if pd.notna(rsi) and filters.get("rsi"):
            if rsi < filters["rsi_max"]:
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
        if pd.notna(bb_lower) and pd.notna(close) and filters.get("bb"):
            if close < bb_lower:
                score += 1
                reasons.append("Dưới BB lower")

        if score < filters.get("min_score", 2):
            return None

        return {
            "Mã": symbol,
            "Giá": round(close, 1) if pd.notna(close) else None,
            "RSI": round(rsi, 1) if pd.notna(rsi) else None,
            "MACD": round(macd, 2) if pd.notna(macd) else None,
            "Vol TB20": int(avg_vol),
            "Điểm": score,
            "Tín hiệu": ", ".join(reasons),
        }
    except Exception:
        return None

def scan_all(filters: dict, max_workers: int = 30) -> pd.DataFrame:
    end = str(date.today())
    start = str(date.today() - timedelta(days=200))
    symbols = get_all_symbols()

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_scan_one, s, start, end, filters): s for s in symbols}
        for f in as_completed(futures):
            r = f.result()
            if r:
                results.append(r)

    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results).sort_values("Điểm", ascending=False).reset_index(drop=True)
    return df
