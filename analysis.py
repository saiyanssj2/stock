import pandas as pd
import ta

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    df["EMA_20"] = ta.trend.ema_indicator(close, window=20)
    df["EMA_50"] = ta.trend.ema_indicator(close, window=50)
    df["RSI_14"] = ta.momentum.rsi(close, window=14)

    macd = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()

    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["STOCH_k"] = stoch.stoch()
    df["STOCH_d"] = stoch.stoch_signal()

    df["ATR_14"] = ta.volatility.average_true_range(high, low, close, window=14)
    df["OBV"] = ta.volume.on_balance_volume(close, volume)

    return df

def get_signal(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    signals = {}

    rsi = last.get("RSI_14")
    if pd.notna(rsi):
        if rsi < 30:
            signals["RSI"] = ("MUA", f"RSI = {rsi:.1f} — Quá bán")
        elif rsi > 70:
            signals["RSI"] = ("BÁN", f"RSI = {rsi:.1f} — Quá mua")
        else:
            signals["RSI"] = ("TRUNG LẬP", f"RSI = {rsi:.1f}")

    macd = last.get("MACD")
    signal = last.get("MACD_signal")
    if pd.notna(macd) and pd.notna(signal):
        if macd > signal:
            signals["MACD"] = ("MUA", f"MACD ({macd:.2f}) > Signal ({signal:.2f})")
        else:
            signals["MACD"] = ("BÁN", f"MACD ({macd:.2f}) < Signal ({signal:.2f})")

    ema20 = last.get("EMA_20")
    ema50 = last.get("EMA_50")
    if pd.notna(ema20) and pd.notna(ema50):
        if ema20 > ema50:
            signals["EMA Cross"] = ("MUA", f"EMA20 ({ema20:.0f}) > EMA50 ({ema50:.0f})")
        else:
            signals["EMA Cross"] = ("BÁN", f"EMA20 ({ema20:.0f}) < EMA50 ({ema50:.0f})")

    close = last.get("close")
    bb_upper = last.get("BB_upper")
    bb_lower = last.get("BB_lower")
    if pd.notna(close) and pd.notna(bb_upper) and pd.notna(bb_lower):
        if close > bb_upper:
            signals["Bollinger"] = ("BÁN", f"Giá ({close:.0f}) vượt dải trên ({bb_upper:.0f})")
        elif close < bb_lower:
            signals["Bollinger"] = ("MUA", f"Giá ({close:.0f}) dưới dải dưới ({bb_lower:.0f})")
        else:
            signals["Bollinger"] = ("TRUNG LẬP", "Giá trong dải Bollinger")

    return signals
