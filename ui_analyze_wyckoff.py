import streamlit as st
import pandas as pd
import ta


def calc_wyckoff(df, last, vol, vol_avg):
    d20 = df.tail(20)
    d60 = df.tail(60)

    high60 = d60['high'].max()
    low60  = d60['low'].min()
    high20 = d20['high'].max()
    low20  = d20['low'].min()
    range60 = high60 - low60

    close      = last['close']
    obv_ema    = ta.trend.ema_indicator(df['OBV'], window=20).iloc[-1]
    obv        = last['OBV']
    obv_rising = obv > obv_ema
    obv_5      = df['OBV'].iloc[-1] - df['OBV'].iloc[-5]
    vol_ratio  = vol / vol_avg
    spread     = last['high'] - last['low']
    spread_avg = (d20['high'] - d20['low']).mean()
    close_pos  = (close - last['low']) / spread if spread > 0 else 0.5
    price_trend20 = df['close'].iloc[-1] - df['close'].iloc[-20]
    vol_trend_up  = df['volume'].tail(5).mean() > vol_avg
    near_low  = close <= low60 + range60 * 0.25
    near_high = close >= high60 - range60 * 0.25

    spring   = last['low'] < low20 * 0.995 and close > low20 * 0.995 and vol_ratio < 1.2
    upthrust = last['high'] > high20 * 1.005 and close < high20 * 1.005 and vol_ratio > 1.0
    sc       = near_low and vol_ratio > 2.5 and spread > spread_avg * 1.5 and close_pos > 0.3

    wide   = spread > spread_avg * 1.3
    narrow = spread < spread_avg * 0.7

    # VSA
    if wide and close_pos > 0.6 and vol_ratio > 1.2:
        vsa_label, vsa_score = "🟢 Demand mạnh — Lực cầu (Effort↑ Result↑)", 2
    elif wide and close_pos < 0.4 and vol_ratio > 1.2:
        vsa_label, vsa_score = "🔴 Supply mạnh — Lực cung (Effort↑ Result↓)", -2
    elif narrow and vol_ratio > 1.3:
        vsa_label, vsa_score = "🟡 No Result — Không kết quả, lực cản mạnh", -1
    elif narrow and vol_ratio < 0.7 and close_pos > 0.5:
        vsa_label, vsa_score = "🟢 No Supply — Không có cung, cạn kiệt", 1
    elif narrow and vol_ratio < 0.7:
        vsa_label, vsa_score = "🟡 No Demand / No Supply — Không cầu/cung, thiếu động lực", 0
    else:
        vsa_label, vsa_score = f"🟡 Volume bình thường (Khối lượng {vol_ratio:.1f}x TB20)", 0

    # Phase
    if near_low and obv_rising and vol_trend_up:
        phase, phase_score = "🔵 Accumulation — Tích lũy (Phase C/D)", 2
    elif sc:
        phase, phase_score = "🟡 Selling Climax (SC) — Bán tháo đỉnh điểm, cung đang bị hấp thụ", 1
    elif near_low and not obv_rising:
        phase, phase_score = "🔴 Markdown / Phase A — Xu hướng giảm chưa dừng", -1
    elif near_high and not obv_rising and vol_ratio > 1.2:
        phase, phase_score = "🟠 Distribution — Phân phối (Phase B/C)", -2
    elif near_high and obv_rising:
        phase, phase_score = "🟢 Markup — Tăng giá (Phase D)", 1
    elif not near_low and not near_high and price_trend20 > 0 and obv_rising:
        phase, phase_score = "🟢 Markup đang hình thành — Tăng giá (Phase D)", 1
    elif not near_low and not near_high and price_trend20 < 0 and not obv_rising:
        phase, phase_score = "🔴 Markdown — Giảm giá (Phase A)", -1
    else:
        phase, phase_score = "🟡 Ranging — Dao động ngang (Phase B)", 0

    spring_score = 0
    if spring:
        phase += " + 🟢 Spring (Bẫy giảm)"
        spring_score = 2
    elif upthrust:
        phase += " + 🔴 Upthrust (Bẫy tăng)"
        spring_score = -2

    obv_score = 1 if (obv_rising and obv_5 > 0) else (-1 if (not obv_rising and obv_5 < 0) else 0)
    total = phase_score + vsa_score + obv_score + spring_score

    return {
        'phase': phase, 'vsa_label': vsa_label,
        'high60': high60, 'low60': low60,
        'obv_rising': obv_rising, 'vol_ratio': vol_ratio,
        'total': total, 'sc': sc,
    }


def show_wyckoff(df, last, vol, vol_avg):
    w = calc_wyckoff(df, last, vol, vol_avg)

    st.caption("⚠️ Phân tích kỹ thuật mô tả trạng thái **tại ngày được chọn** dựa trên dữ liệu đã có. "
               "Đây là công cụ hỗ trợ, không phải dự đoán tương lai. "
               "Một số sự kiện như Selling Climax chỉ xác nhận được sau khi phiên kế tiếp tăng mạnh.")
    st.markdown(f"**📊 Wyckoff:** {w['phase']} &nbsp;|&nbsp; {w['vsa_label']}")

    wc1, wc2, wc3, wc4 = st.columns(4)
    wc1.metric("Đỉnh 60 phiên (Resistance)", f"{w['high60']:,.2f}")
    wc2.metric("Đáy 60 phiên (Support)", f"{w['low60']:,.2f}")
    wc3.metric("OBV vs EMA20 (Dòng tiền)", "↑ Tích lũy (Accumulation)" if w['obv_rising'] else "↓ Phân phối (Distribution)")
    wc4.metric("Vol/TB20 (Khối lượng tương đối)", f"{w['vol_ratio']:.1f}x")

    return w['total']
