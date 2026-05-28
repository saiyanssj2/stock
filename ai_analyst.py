import pandas as pd
import os

def analyze_from_csv(csv_path: str, symbol: str, target_date: str) -> dict | None:
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Lay du lieu den ngay target
    target_dt = pd.Timestamp(target_date)
    past = df[df["time"] <= target_dt]
    if len(past) < 5:
        return None

    last = past.iloc[-1]
    prev = past.iloc[-2]

    signals = []
    score_buy = 0
    score_sell = 0

    close = last["close"]
    ema20 = last.get("EMA_20")
    ema50 = last.get("EMA_50")
    rsi = last.get("RSI_14")
    macd = last.get("MACD")
    macd_sig = last.get("MACD_signal")
    macd_hist = last.get("MACD_hist")
    bb_upper = last.get("BB_upper")
    bb_lower = last.get("BB_lower")
    stoch_k = last.get("STOCH_k")
    stoch_d = last.get("STOCH_d")
    volume = last.get("volume")
    vol_avg = past["volume"].tail(20).mean()

    prev_macd = prev.get("MACD")
    prev_macd_sig = prev.get("MACD_signal")
    prev_ema20 = prev.get("EMA_20")
    prev_ema50 = prev.get("EMA_50")

    # --- EMA ---
    if pd.notna(ema20) and pd.notna(ema50):
        if ema20 > ema50:
            if pd.notna(prev_ema20) and prev_ema20 <= prev_ema50:
                signals.append(("🟢", "EMA20 vừa cắt lên EMA50 — tín hiệu đảo chiều tăng mạnh"))
                score_buy += 2
            else:
                signals.append(("🟢", f"EMA20 ({ema20:.2f}) > EMA50 ({ema50:.2f}) — uptrend"))
                score_buy += 1
        else:
            if pd.notna(prev_ema20) and prev_ema20 >= prev_ema50:
                signals.append(("🔴", "EMA20 vừa cắt xuống EMA50 — tín hiệu đảo chiều giảm mạnh"))
                score_sell += 2
            else:
                signals.append(("🔴", f"EMA20 ({ema20:.2f}) < EMA50 ({ema50:.2f}) — downtrend"))
                score_sell += 1

    # --- RSI ---
    if pd.notna(rsi):
        if rsi < 30:
            signals.append(("🟢", f"RSI = {rsi:.1f} — vùng quá bán, khả năng bật tăng"))
            score_buy += 2
        elif rsi > 70:
            signals.append(("🔴", f"RSI = {rsi:.1f} — vùng quá mua, cẩn thận điều chỉnh"))
            score_sell += 2
        elif rsi < 45:
            signals.append(("🔴", f"RSI = {rsi:.1f} — momentum yếu"))
            score_sell += 1
        elif rsi > 55:
            signals.append(("🟢", f"RSI = {rsi:.1f} — momentum tốt"))
            score_buy += 1
        else:
            signals.append(("🟡", f"RSI = {rsi:.1f} — trung tính"))

    # --- MACD ---
    if pd.notna(macd) and pd.notna(macd_sig):
        if pd.notna(prev_macd) and pd.notna(prev_macd_sig):
            if macd > macd_sig and prev_macd <= prev_macd_sig:
                signals.append(("🟢", f"MACD vừa cắt lên Signal — tín hiệu mua mạnh"))
                score_buy += 2
            elif macd < macd_sig and prev_macd >= prev_macd_sig:
                signals.append(("🔴", f"MACD vừa cắt xuống Signal — tín hiệu bán mạnh"))
                score_sell += 2
            elif macd > macd_sig:
                signals.append(("🟢", f"MACD ({macd:.3f}) > Signal ({macd_sig:.3f})"))
                score_buy += 1
            else:
                signals.append(("🔴", f"MACD ({macd:.3f}) < Signal ({macd_sig:.3f})"))
                score_sell += 1

        if pd.notna(macd_hist):
            prev_hist = prev.get("MACD_hist")
            if pd.notna(prev_hist):
                if macd_hist > 0 and prev_hist <= 0:
                    signals.append(("🟢", "MACD Histogram vừa chuyển dương — momentum tăng"))
                    score_buy += 1
                elif macd_hist < 0 and prev_hist >= 0:
                    signals.append(("🔴", "MACD Histogram vừa chuyển âm — momentum giảm"))
                    score_sell += 1

    # --- Bollinger Bands ---
    if pd.notna(bb_upper) and pd.notna(bb_lower) and pd.notna(close):
        bb_mid = (bb_upper + bb_lower) / 2
        if close < bb_lower:
            signals.append(("🟢", f"Giá ({close:.2f}) dưới BB Lower ({bb_lower:.2f}) — oversold, khả năng bật"))
            score_buy += 2
        elif close > bb_upper:
            signals.append(("🔴", f"Giá ({close:.2f}) vượt BB Upper ({bb_upper:.2f}) — overbought"))
            score_sell += 1
        else:
            pct = (close - bb_lower) / (bb_upper - bb_lower) * 100
            signals.append(("🟡", f"Giá trong dải BB ({pct:.0f}% từ đáy dải)"))

    # --- Stochastic ---
    if pd.notna(stoch_k) and pd.notna(stoch_d):
        if stoch_k < 20:
            signals.append(("🟢", f"Stochastic %K = {stoch_k:.1f} — vùng quá bán sâu"))
            score_buy += 1
        elif stoch_k > 80:
            signals.append(("🔴", f"Stochastic %K = {stoch_k:.1f} — vùng quá mua"))
            score_sell += 1

    # --- Volume ---
    if pd.notna(volume) and pd.notna(vol_avg) and vol_avg > 0:
        vol_ratio = volume / vol_avg
        if vol_ratio > 1.5:
            signals.append(("🟢", f"Volume đột biến {vol_ratio:.1f}x TB20 — dòng tiền vào mạnh"))
            score_buy += 1
        elif vol_ratio < 0.5:
            signals.append(("🟡", f"Volume thấp {vol_ratio:.1f}x TB20 — thiếu xác nhận"))

    # --- Ket luan ---
    total = score_buy + score_sell
    if total == 0:
        verdict = "🟡 TRUNG LẬP"
        confidence = 0
    else:
        confidence = round(score_buy / total * 100)
        if score_buy >= 5 and score_buy > score_sell * 1.5:
            verdict = "🟢 MUA"
        elif score_sell >= 5 and score_sell > score_buy * 1.5:
            verdict = "🔴 BÁN / TRÁNH"
        elif score_buy > score_sell:
            verdict = "🟡 NGHIÊNG VỀ MUA"
        elif score_sell > score_buy:
            verdict = "🟡 NGHIÊNG VỀ BÁN"
        else:
            verdict = "🟡 TRUNG LẬP"

    # Stop loss / target
    atr = last.get("ATR_14")
    sl = round(close - 2 * atr, 2) if pd.notna(atr) else None
    tp = round(close + 3 * atr, 2) if pd.notna(atr) else None

    return {
        "symbol": symbol,
        "date": str(last["time"].date()),
        "close": close,
        "verdict": verdict,
        "score_buy": score_buy,
        "score_sell": score_sell,
        "confidence": confidence,
        "signals": signals,
        "stop_loss": sl,
        "take_profit": tp,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_sig,
        "ema20": ema20,
        "ema50": ema50,
    }
