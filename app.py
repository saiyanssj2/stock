import streamlit as st
from datetime import date, timedelta
from news import get_news_vn, get_news_intl, get_world_indices, get_vn_indices
from ai_analyst import analyze_from_csv
import os
import pandas as pd

st.set_page_config(page_title="Phan Tich Ky Thuat", layout="wide")
st.title("📈 Phân Tích Kỹ Thuật Chứng Khoán")

@st.cache_data(ttl=10800)
def load_world_indices():
    return get_world_indices()

@st.cache_data(ttl=10800)
def load_vn_indices():
    return get_vn_indices()

@st.cache_data(ttl=10800)
def load_news():
    return get_news_vn(), get_news_intl()

# --- Thi truong the gioi ---
st.subheader("🌍 Thị trường thế giới")
with st.status("Đang tải dữ liệu...", expanded=False) as status:
    st.write("📡 Đang lấy S&P500, Nasdaq, Dow Jones...")
    indices = load_world_indices()
    st.write("🇻🇳 Đang tính chỉ báo VNINDEX, VN30...")
    vn_indices = load_vn_indices()
    status.update(label="✅ Tải xong dữ liệu thị trường", state="complete")

if indices:
    cols = st.columns(len(indices))
    for col, idx in zip(cols, indices):
        col.metric(label=idx["name"], value=f"{idx['price']:,.2f}",
                   delta=f"{idx['change']:+.2f} ({idx['pct']:+.2f}%)", delta_color="normal")
else:
    st.warning("Không tải được dữ liệu thị trường thế giới.")

# --- Thi truong VN ---
st.subheader("🇻🇳 Thị trường Việt Nam")
if vn_indices:
    cols = st.columns(len(vn_indices))
    for col, idx in zip(cols, vn_indices):
        col.metric(
            label=f"{idx['name']} {idx['trend']}",
            value=f"{idx['close']:,.2f}",
            delta=f"{idx['change']:+.2f} ({idx['pct']:+.2f}%)",
            delta_color="normal"
        )
        col.caption(f"RSI {idx['rsi']:.0f} | MACD {'>' if idx['macd'] > idx['macd_signal'] else '<'} Signal | ADX {idx['adx']:.0f} | EMA20 {'>' if idx['ema20'] > idx['ema50'] else '<'} EMA50")

    main = next((i for i in vn_indices if i["name"] == "VNINDEX"), None)
    if main:
        details_str = " | ".join(main["details"])
        if "🔴" in main["trend"]:
            st.warning(f"⚠️ VNINDEX xu hướng **giảm** (score={main['score']}) — {details_str}")
        elif "🟢" in main["trend"]:
            st.success(f"✅ VNINDEX xu hướng **tăng** (score={main['score']}) — {details_str}")
        else:
            st.info(f"🟡 VNINDEX **trung lập** (score={main['score']}) — {details_str}")
else:
    st.warning("Không tải được dữ liệu thị trường VN.")

st.divider()

# --- Tin tuc ---
with st.status("Đang tải tin tức...", expanded=False) as status:
    st.write("📰 Đang lấy tin CafeF, VnEconomy...")
    st.write("🌐 Đang lấy tin Reuters, Bloomberg, CNBC...")
    vn_news, intl_news = load_news()
    status.update(label=f"✅ Tin tức: {len(vn_news)} VN + {len(intl_news)} quốc tế", state="complete")

col1, col2 = st.columns(2)
with col1:
    st.subheader("📰 Tin trong ngày — Việt Nam")
    if vn_news:
        for item in vn_news:
            st.markdown(f"`{item['published']}` [{item['title']}]({item['link']}) — `{item['source']}`")
    else:
        st.info("Chưa có tin tức trong ngày.")
with col2:
    st.subheader("🌐 Tin quốc tế")
    if intl_news:
        for item in intl_news:
            st.markdown(f"`{item['published']}` [{item['title']}]({item['link']}) — `{item['source']}`")
    else:
        st.info("Chưa có tin quốc tế trong ngày.")

st.divider()

# --- Phan tich AI ---
st.subheader("🤖 Phân tích kỹ thuật tự động")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    symbol_input = st.text_input("Mã cổ phiếu", value="VIX").upper()

csv_path = os.path.join(BASE_DIR, f"{symbol_input}_full.csv")
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
        analyze_date = st.date_input("Ngày phân tích", value=date.today())
        st.caption(f"⚠️ Chưa có file {symbol_input}_full.csv")
with c3:
    st.write("")
    st.write("")
    run_analysis = st.button("🔍 Phân tích", type="primary", disabled=not csv_exists)

if run_analysis:
    with st.status("Đang phân tích...", expanded=True) as status:
        st.write(f"📂 Đọc dữ liệu {symbol_input} từ CSV...")
        st.write(f"🔢 Tính toán EMA, RSI, MACD, Bollinger, Stochastic...")
        result = analyze_from_csv(csv_path, symbol_input, str(analyze_date))
        if result:
            status.update(label=f"✅ Phân tích xong {symbol_input} ngày {result['date']}", state="complete")
        else:
            status.update(label="❌ Không đủ dữ liệu", state="error")

    if result is None:
        st.error("Không đủ dữ liệu để phân tích.")
    else:
        st.subheader(f"{result['symbol']} ngày {result['date']} — Giá: {result['close']:.2f}")

        verdict = result['verdict']
        msg = f"**{verdict}** | Điểm MUA: {result['score_buy']} | Điểm BÁN: {result['score_sell']}"
        if "🟢" in verdict:
            st.success(msg)
        elif "🔴" in verdict:
            st.error(msg)
        else:
            st.warning(msg)

        if result['stop_loss'] and result['take_profit']:
            c1, c2 = st.columns(2)
            c1.metric("Stop Loss (2x ATR)", f"{result['stop_loss']:.2f}",
                      delta=f"{(result['stop_loss']-result['close'])/result['close']*100:+.1f}%", delta_color="inverse")
            c2.metric("Take Profit (3x ATR)", f"{result['take_profit']:.2f}",
                      delta=f"{(result['take_profit']-result['close'])/result['close']*100:+.1f}%", delta_color="normal")

        st.markdown("**Chi tiết tín hiệu:**")
        for icon, desc in result['signals']:
            st.markdown(f"{icon} {desc}")
