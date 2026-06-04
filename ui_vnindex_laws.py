import streamlit as st
import pandas as pd


def show_three_laws(c, df_vn):
    st.markdown("**⚖️ 3 Quy luật nền tảng:**")
    law1_col, law2_col, law3_col = st.columns(3)

    # Quy luat 1: Cung - Cau
    if c['wide_spread'] and c['close_upper'] and c['vol_ratio'] > 1.2:
        law1 = ("🟢", "Cung < Cầu", f"Nến rộng đóng cao, vol {c['vol_ratio']:.1f}x — Cầu đang thắng")
    elif c['wide_spread'] and c['close_lower'] and c['vol_ratio'] > 1.2:
        law1 = ("🔴", "Cung > Cầu", f"Nến rộng đóng thấp, vol {c['vol_ratio']:.1f}x — Cung đang thắng")
    elif c['narrow_spread'] and c['vol_ratio'] < 0.7:
        law1 = ("🟡", "Cung ≈ Cầu", f"Biên độ hẹp, vol thấp {c['vol_ratio']:.1f}x — Cân bằng")
    else:
        law1 = ("🟡", "Cung/Cầu chưa rõ", f"Vol {c['vol_ratio']:.1f}x TB20, cần thêm xác nhận")
    with law1_col:
        st.markdown(f"{law1[0]} **Cung — Cầu:** {law1[1]}")
        st.caption(law1[2])

    # Quy luat 2: No luc - Ket qua
    price_change_pct = abs(c['close'] - df_vn['close'].iloc[-2]) / df_vn['close'].iloc[-2] * 100
    if c['vol_ratio'] > 1.3 and price_change_pct > 0.5:
        law2 = ("🟢", "Nỗ lực = Kết quả", f"Vol cao {c['vol_ratio']:.1f}x, giá di chuyển {price_change_pct:.1f}% — nhất quán")
    elif c['vol_ratio'] > 1.3 and price_change_pct < 0.3:
        law2 = ("🔴", "Nỗ lực ≠ Kết quả", f"Vol cao {c['vol_ratio']:.1f}x nhưng giá ít di chuyển — lực cản mạnh")
    elif c['vol_ratio'] < 0.7 and price_change_pct < 0.3:
        law2 = ("🟡", "Nỗ lực thấp", f"Vol thấp {c['vol_ratio']:.1f}x, giá ít biến động — thiếu động lực")
    else:
        law2 = ("🟡", "Bình thường", f"Vol {c['vol_ratio']:.1f}x, giá {price_change_pct:.1f}%")
    with law2_col:
        st.markdown(f"{law2[0]} **Nỗ lực — Kết quả:** {law2[1]}")
        st.caption(law2[2])

    # Quy luat 3: Nhan - Qua
    accum_days = 0
    for i in range(len(df_vn) - 1, max(len(df_vn) - 60, 0), -1):
        r = df_vn.iloc[i]
        if c['low_60'] <= r['close'] <= c['low_60'] + c['range_60'] * 0.35:
            accum_days += 1
        else:
            break
    if accum_days >= 10:
        law3 = ("🟢", f"Tích lũy {accum_days} phiên", "Nền tảng đủ lớn → tiềm năng tăng mạnh")
    elif accum_days >= 5:
        law3 = ("🟡", f"Tích lũy {accum_days} phiên", "Nền tảng đang hình thành — cần thêm thời gian")
    else:
        law3 = ("🟡", "Chưa có nền tảng rõ", "Giá chưa sideway đủ lâu để tạo đà")
    with law3_col:
        st.markdown(f"{law3[0]} **Nhân — Quả:** {law3[1]}")
        st.caption(law3[2])


def show_composite_man(c):
    close, ema200 = c['close'], c['ema200']
    obv_rising, near_high = c['obv_rising'], c['near_high']
    near_low, price_trend_20 = c['near_low'], c['price_trend_20']

    st.markdown("**🎭 Composite Man đang ở chu kỳ:**")
    if close > ema200 and obv_rising and price_trend_20 > 0:
        cycle, cycle_desc, cycle_color = "🟢 Markup (Tăng giá — Đẩy giá)", "Tiền lớn đã gom xong, đang đẩy giá lên để bán cho nhà đầu tư nhỏ lẻ", "success"
    elif close > ema200 and not obv_rising and near_high:
        cycle, cycle_desc, cycle_color = "🟠 Distribution (Phân phối — Xả hàng)", "Tiền lớn đang phân phối — xả hàng cho nhà đầu tư nhỏ lẻ đang FOMO (sợ bỏ lỡ) mua đỉnh", "warning"
    elif close < ema200 and not obv_rising and price_trend_20 < 0:
        cycle, cycle_desc, cycle_color = "🔴 Markdown (Giảm giá)", "Tiền lớn đã thoát hàng, để giá rơi tự do — nhà đầu tư nhỏ lẻ đang bán tháo", "error"
    elif close < ema200 and (obv_rising or near_low):
        cycle, cycle_desc, cycle_color = "🔵 Accumulation (Tích lũy — Gom hàng)", "Tiền lớn đang âm thầm gom hàng từ nhà đầu tư nhỏ lẻ hoảng loạn bán", "info"
    else:
        cycle, cycle_desc, cycle_color = "🟡 Transition (Chuyển tiếp)", "Chưa xác định rõ chu kỳ — thị trường đang trong giai đoạn chuyển đổi", "warning"

    if cycle_color == "success":   st.success(f"**{cycle}** — {cycle_desc}")
    elif cycle_color == "error":   st.error(f"**{cycle}** — {cycle_desc}")
    elif cycle_color == "info":    st.info(f"**{cycle}** — {cycle_desc}")
    else:                          st.warning(f"**{cycle}** — {cycle_desc}")
    return cycle


def show_price_zones(c, total_score, atr):
    st.markdown("**📐 Vùng giá Wyckoff:**")
    st.markdown(f"- 🔵 Resistance — Đỉnh 60 phiên (Kháng cự): **{c['high_60']:,.2f}**")
    st.markdown(f"- 🔵 Support — Đáy 60 phiên (Hỗ trợ): **{c['low_60']:,.2f}**")
    st.markdown(f"- 🟡 Creek — EMA50 (Ngưỡng kháng cự trung hạn): **{c['ema50']:,.2f}**")
    st.markdown(f"- 🟢 Ice — EMA200 (Nền tăng dài hạn): **{c['ema200']:,.2f}**")
    if pd.notna(atr):
        sl = round(c['close'] - 2 * atr, 2)
        st.markdown("---")
        st.markdown(f"🔴 Stop Loss — Cắt lỗ (2x ATR): **{sl:,.2f}** ({(sl-c['close'])/c['close']*100:+.1f}%)")
        if total_score >= 2:
            tp = round(c['high_60'], 2)
            st.markdown(f"🟢 Target (đỉnh 60 phiên): **{tp:,.2f}** ({(tp-c['close'])/c['close']*100:+.1f}%)")
    st.markdown("---")
    if total_score >= 3:   st.success("✅ Wyckoff: **Tích lũy xong — Có thể mua**")
    elif total_score >= 1: st.info("🔵 Wyckoff: **Đang tích lũy — Theo dõi breakout**")
    elif total_score >= -1: st.warning("🟡 Wyckoff: **Chưa rõ pha — Chờ xác nhận**")
    else:                  st.error("🔴 Wyckoff: **Phân phối / Markdown — Tránh mua**")
