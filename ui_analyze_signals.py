import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def show_signals(df, last, prev, close, ema20, ema50, rsi, macd, macd_s, macd_h,
                 bb_u, bb_l, stoch, atr, vol, vol_avg):
    signals = []
    score = 0

    def add(icon, desc, s):
        signals.append((icon, desc, s))
        return s

    score += add("🟢" if ema20 > ema50 else "🔴",
                 f"EMA20 {'>' if ema20 > ema50 else '<'} EMA50 — {'Uptrend (xu hướng tăng)' if ema20 > ema50 else 'Downtrend (xu hướng giảm)'} trung hạn",
                 +1 if ema20 > ema50 else -1)
    score += add("🟢" if close > ema20 else "🔴",
                 f"Giá {'trên' if close > ema20 else 'dưới'} EMA20 — Trung bình động 20 ({ema20:.2f})",
                 +1 if close > ema20 else -1)

    if rsi < 30:        score += add("🟢", f"RSI = {rsi:.1f} — Oversold (quá bán)", +2)
    elif rsi > 70:      score += add("🔴", f"RSI = {rsi:.1f} — Overbought (quá mua)", -2)
    elif rsi < 45:      score += add("🔴", f"RSI = {rsi:.1f} — Momentum yếu (đà yếu)", -1)
    elif rsi > 55:      score += add("🟢", f"RSI = {rsi:.1f} — Momentum tốt (đà mạnh)", +1)
    else:               add("🟡", f"RSI = {rsi:.1f} — Neutral (trung tính)", 0)

    if pd.notna(macd) and pd.notna(macd_s):
        prev_macd, prev_macd_s = prev["MACD"], prev["MACD_signal"]
        if macd > macd_s and prev_macd <= prev_macd_s:
            score += add("🟢", "MACD vừa cắt lên Signal — Bullish crossover (tín hiệu mua mạnh)", +2)
        elif macd < macd_s and prev_macd >= prev_macd_s:
            score += add("🔴", "MACD vừa cắt xuống Signal — Bearish crossover (tín hiệu bán mạnh)", -2)
        elif macd > macd_s:
            score += add("🟢", f"MACD ({macd:.2f}) > Signal ({macd_s:.2f}) — Bullish (tích cực)", +1)
        else:
            score += add("🔴", f"MACD ({macd:.2f}) < Signal ({macd_s:.2f}) — Bearish (tiêu cực)", -1)

        prev_h = prev["MACD_hist"]
        if macd_h < 0 and prev_h >= 0:
            score += add("🔴", "MACD Histogram vừa chuyển âm — Bearish (tiêu cực)", -1)
        elif macd_h > 0 and prev_h <= 0:
            score += add("🟢", "MACD Histogram vừa chuyển dương — Bullish (tích cực)", +1)
        elif macd_h < 0:
            if abs(macd - macd_s) > abs(prev_macd - prev_macd_s):
                score += add("🔴", f"MACD Histogram âm nới rộng — Expanding bearish ({macd_h:.2f})", -1)
            else:
                add("🟡", f"MACD Histogram âm thu hẹp — Contracting ({macd_h:.2f})", 0)

    if pd.notna(bb_u) and pd.notna(bb_l):
        bb_pct = (close - bb_l) / (bb_u - bb_l) * 100
        if close < bb_l:    score += add("🟢", f"Giá dưới BB Lower ({bb_l:.2f}) — Oversold mạnh (quá bán sâu)", +2)
        elif close > bb_u:  score += add("🔴", f"Giá vượt BB Upper ({bb_u:.2f}) — Overbought (quá mua)", -1)
        elif bb_pct < 20:   score += add("🟢", f"Giá gần BB Lower — Support zone (vùng hỗ trợ) ({bb_pct:.0f}%)", +1)
        elif bb_pct > 80:   add("🟡", f"Giá gần BB Upper — Resistance zone (vùng kháng cự) ({bb_pct:.0f}%)", 0)
        else:               add("🟡", f"Giá trong dải BB — Bollinger Bands ({bb_pct:.0f}%)", 0)

    if pd.notna(stoch):
        if stoch < 20:   score += add("🟢", f"Stochastic %K = {stoch:.1f} — Oversold sâu (quá bán)", +1)
        elif stoch > 80: score += add("🔴", f"Stochastic %K = {stoch:.1f} — Overbought (quá mua)", -1)
        else:            add("🟡", f"Stochastic %K = {stoch:.1f} — Neutral (trung tính)", 0)

    vol_ratio = vol / vol_avg
    if vol_ratio > 1.5:   score += add("🟢", f"Volume surge — Khối lượng đột biến {vol_ratio:.1f}x TB20", +1)
    elif vol_ratio < 0.7: add("🟡", f"Low volume — Khối lượng thấp {vol_ratio:.1f}x TB20", 0)
    else:                 add("🟡", f"Volume bình thường (Normal) {vol_ratio:.1f}x TB20", 0)

    trend_5 = df["close"].iloc[-1] - df["close"].iloc[-5]
    if trend_5 > 0: score += add("🟢", f"Xu hướng 5 phiên: +{trend_5:.2f} — tăng", +1)
    else:           score += add("🔴", f"Xu hướng 5 phiên: {trend_5:.2f} — giảm", -1)

    return signals, score


def show_verdict(score, wyckoff_total):
    total = score + wyckoff_total
    if total >= 5:    verdict, color = "🟢 MUA — Tín hiệu tích cực mạnh", "success"
    elif total >= 2:  verdict, color = "🟢 NGHIÊNG VỀ MUA", "success"
    elif total >= -1: verdict, color = "🟡 TRUNG LẬP — Chờ thêm tín hiệu", "warning"
    elif total >= -4: verdict, color = "🔴 NGHIÊNG VỀ BÁN — Thận trọng", "error"
    else:             verdict, color = "🔴 TRÁNH / BÁN — Tín hiệu tiêu cực", "error"

    msg = f"**{verdict}** &nbsp;|&nbsp; Kỹ thuật: **{score:+d}** | Wyckoff: **{wyckoff_total:+d}** | Tổng: **{total:+d}**"
    if color == "success": st.success(msg)
    elif color == "error":  st.error(msg)
    else:                   st.warning(msg)


def show_signal_detail(signals, close, ema20, ema50, bb_u, bb_l, atr):
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**📊 Chi tiết tín hiệu:**")
        for icon, desc, s in signals:
            sign = f"`{'+' if s > 0 else ''}{s}`" if s != 0 else "`  0`"
            st.markdown(f"{icon} {sign} {desc}")

    with col_r:
        st.markdown("**🎯 Vùng giá quan trọng:**")
        if pd.notna(atr):
            sl  = round(close - 2 * atr, 2)
            tp1 = round(ema20, 2)
            tp2 = round(bb_u, 2)
            st.markdown(f"🔴 Stop Loss — Cắt lỗ (2x ATR): **{sl:,.2f}** ({(sl-close)/close*100:+.1f}%)")
            st.markdown(f"🟡 Take Profit 1 — Chốt lời 1 (EMA20): **{tp1:,.2f}** ({(tp1-close)/close*100:+.1f}%)")
            st.markdown(f"🟢 Take Profit 2 — Chốt lời 2 (BB Upper): **{tp2:,.2f}** ({(tp2-close)/close*100:+.1f}%)")
        st.markdown("---")
        st.markdown(f"⬆️ Resistance — Kháng cự 1: EMA20 = **{ema20:,.2f}**")
        st.markdown(f"⬆️ Resistance — Kháng cự 2: BB Upper = **{bb_u:,.2f}**")
        st.markdown(f"⬇️ Support — Hỗ trợ 1: BB Lower = **{bb_l:,.2f}**")
        st.markdown(f"⬇️ Support — Hỗ trợ 2: EMA50 = **{ema50:,.2f}**")


def show_chart(df):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03,
                        subplot_titles=("Giá", "Volume", "RSI"))
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Giá", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["EMA_20"], name="EMA20", line=dict(color="orange", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["EMA_50"], name="EMA50", line=dict(color="blue", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_upper"], name="BB Upper", line=dict(color="gray", dash="dash", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_lower"], name="BB Lower",
                             line=dict(color="gray", dash="dash", width=1),
                             fill="tonexty", fillcolor="rgba(128,128,128,0.1)"), row=1, col=1)
    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df["time"], y=df["volume"], name="Volume", marker_color=colors, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["RSI_14"], name="RSI", line=dict(color="purple", width=1.2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.update_layout(height=700, xaxis_rangeslider_visible=False,
                      template="plotly_dark", legend=dict(orientation="h", y=1.02),
                      margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)
