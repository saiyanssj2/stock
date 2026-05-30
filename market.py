import yfinance as yf
from vnstock.api.quote import Quote
from analysis import add_indicators
import pandas as pd


def get_world_indices():
    symbols = {
        "S&P 500": "^GSPC",
        "Nasdaq": "^IXIC",
        "Dow Jones": "^DJI",
    }
    result = []
    for name, sym in symbols.items():
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="2d")
            if len(hist) >= 2:
                prev, curr = hist["Close"].iloc[-2], hist["Close"].iloc[-1]
                change = curr - prev
                pct = change / prev * 100
                result.append({"name": name, "price": curr, "change": change, "pct": pct})
        except Exception:
            pass
    return result


def get_vn_indices():
    indices = ["VNINDEX", "VN30"]
    result = []
    for symbol in indices:
        try:
            q = Quote(symbol=symbol, source="VCI")
            df = q.history(start="2024-01-01", end="2099-12-31", interval="1D")
            if df is None or len(df) < 2:
                continue
            df.columns = [c.lower() for c in df.columns]
            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time").reset_index(drop=True)
            df = add_indicators(df)

            last = df.iloc[-1]
            prev = df.iloc[-2]
            close = last["close"]
            change = close - prev["close"]
            pct = change / prev["close"] * 100

            score = 0
            if last["EMA_20"] > last["EMA_50"]: score += 1
            else: score -= 1
            if close > last["EMA_20"]: score += 1
            else: score -= 1
            rsi = last["RSI_14"]
            if pd.notna(rsi):
                if rsi > 55: score += 1
                elif rsi < 45: score -= 1
            macd, sig = last["MACD"], last["MACD_signal"]
            if pd.notna(macd) and pd.notna(sig):
                if macd > sig: score += 1
                else: score -= 1
            obv_ema = df["OBV"].ewm(span=20).mean().iloc[-1]
            if last["OBV"] > obv_ema: score += 1
            else: score -= 1

            if score >= 2: trend = "🟢"
            elif score <= -2: trend = "🔴"
            else: trend = "🟡"

            result.append({
                "name": symbol, "close": close,
                "change": change, "pct": pct,
                "trend": trend, "score": score,
            })
        except Exception:
            pass
    return result
