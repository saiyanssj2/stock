import streamlit as st
import pandas as pd
from ui_vnindex_wyckoff import calc_vnindex_wyckoff, show_wyckoff_signals, show_subphase
from ui_vnindex_laws import show_three_laws, show_composite_man, show_price_zones


def show_vnindex(load_vnindex_analysis):
    st.subheader("📊 Phân tích VNINDEX — Wyckoff")

    df_vn = load_vnindex_analysis()
    if df_vn is None:
        st.warning("Không có dữ liệu VNINDEX.")
        return

    last  = df_vn.iloc[-1]
    prev  = df_vn.iloc[-2]
    df_20 = df_vn.tail(20)
    df_60 = df_vn.tail(60)
    atr   = last['ATR_14']

    c = calc_vnindex_wyckoff(df_vn, last, prev, df_20, df_60)

    # Wyckoff signals + phase
    signals, phase, phase_detail, phase_color, total_score = show_wyckoff_signals(c, df_vn)

    # Header
    msg = f"**VNINDEX {last['time'].date()} — {c['close']:,.2f}**"
    if phase_color == "success":   st.success(msg)
    elif phase_color == "error":   st.error(msg)
    elif phase_color == "info":    st.info(msg)
    else:                          st.warning(msg)

    st.markdown(f"### {phase}")
    st.caption(phase_detail)
    st.caption("⚠️ Phân tích mô tả trạng thái **tại ngày cuối cùng có dữ liệu**, không dự đoán tương lai. "
               "Selling Climax chỉ xác nhận được sau khi phiên kế tiếp tăng mạnh. "
               "Markdown và SC có cùng dấu hiệu trong ngày — chỉ khác nhau ở phiên sau.")

    # 3 Quy luat
    show_three_laws(c, df_vn)

    st.divider()

    # Composite Man + sub-phase
    cycle = show_composite_man(c)
    st.markdown("**📍 Đang ở sub-phase:**")
    show_subphase(c, cycle, phase, df_vn)

    st.divider()

    # Chi tiet Wyckoff + vung gia
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.markdown("**🔬 Phân tích Wyckoff:**")
        for icon, event, detail in signals:
            st.markdown(f"{icon} **{event}**")
            st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;{detail}")
    with col_w2:
        show_price_zones(c, total_score, atr)
