import streamlit as st
import pandas as pd
import os
from datetime import date as date_cls
from ui_analyze_realtime import show_realtime
from ui_analyze_wyckoff import show_wyckoff
from ui_analyze_signals import show_signals, show_verdict, show_signal_detail, show_chart


def show_analyze(BASE_DIR):
    avail = sorted(set(
        f.replace('_full.csv', '').replace('.csv', '')
        for f in os.listdir(BASE_DIR)
        if f.endswith('.csv') and not f.startswith('.')
    ))

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        symbol_input = st.selectbox("Mã cổ phiếu", avail,
                                    index=avail.index('VNINDEX') if 'VNINDEX' in avail else 0)
        symbol_input = symbol_input.upper()

    csv_path = os.path.join(BASE_DIR, f"{symbol_input}_full.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(BASE_DIR, f"{symbol_input}.csv")
    csv_exists = os.path.exists(csv_path)

    with c2:
        if csv_exists:
            df_tmp = pd.read_csv(csv_path, usecols=["time"])
            df_tmp["time"] = pd.to_datetime(df_tmp["time"])
            min_date = df_tmp["time"].min().date()
            max_date = df_tmp["time"].max().date()
            analyze_date = st.date_input("Ngày phân tích", value=max_date, min_value=min_date, max_value=max_date)
            st.caption(f"📂 CSV: {min_date} → {max_date}")
        else:
            analyze_date = st.date_input("Ngày phân tích", value=date_cls.today())
            st.caption(f"⚠️ Chưa có file {symbol_input}.csv")
    with c3:
        st.write("")
        st.write("")
        run_btn = st.button("🔍 Phân tích", type="primary", disabled=not csv_exists)

    if not run_btn:
        return

    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df = df[df["time"] <= pd.Timestamp(str(analyze_date))]

    if len(df) < 5:
        st.error("Không đủ dữ liệu.")
        return

    last    = df.iloc[-1]
    prev    = df.iloc[-2]
    close   = last["close"]
    ema20   = last["EMA_20"]
    ema50   = last["EMA_50"]
    rsi     = last["RSI_14"]
    macd    = last["MACD"]
    macd_s  = last["MACD_signal"]
    macd_h  = last["MACD_hist"]
    bb_u    = last["BB_upper"]
    bb_l    = last["BB_lower"]
    stoch   = last["STOCH_k"]
    atr     = last["ATR_14"]
    vol     = last["volume"]
    vol_avg = df["volume"].tail(20).mean()

    st.subheader(f"{symbol_input} — {last['time'].date()} — Giá: {close:,.2f}")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("EMA20 (Trung bình động 20)", f"{ema20:,.2f}", delta=f"{close-ema20:+.2f}", delta_color="normal")
    m2.metric("EMA50 (Trung bình động 50)", f"{ema50:,.2f}", delta=f"{close-ema50:+.2f}", delta_color="normal")
    m3.metric("RSI (Sức mạnh tương đối)", f"{rsi:.1f}")
    m4.metric("MACD (Hội tụ/Phân kỳ TB)", f"{macd:.2f}", delta=f"Signal {macd_s:.2f}", delta_color="off")
    m5.metric("Volume / TB20 (Khối lượng)", f"{vol/vol_avg*100:.0f}%")

    # Realtime
    if str(analyze_date) == str(date_cls.today()):
        show_realtime(symbol_input)
    else:
        st.info("📅 Dữ liệu cầu/cung, khối ngoại chỉ hiển thị khi phân tích ngày hôm nay.")

    st.divider()

    # Wyckoff
    wyckoff_total = show_wyckoff(df, last, vol, vol_avg)

    st.divider()

    # Tín hiệu kỹ thuật
    signals, score = show_signals(df, last, prev, close, ema20, ema50, rsi, macd, macd_s, macd_h,
                                  bb_u, bb_l, stoch, atr, vol, vol_avg)
    show_verdict(score, wyckoff_total)
    show_signal_detail(signals, close, ema20, ema50, bb_u, bb_l, atr)

    st.divider()
    show_chart(df)
