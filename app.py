import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, timedelta
from market import get_world_indices, get_vn_indices
import os
import pandas as pd

st.set_page_config(page_title="Phan Tich Ky Thuat", layout="wide")
st.title("📈 Phân Tích Kỹ Thuật Chứng Khoán")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data(ttl=10800)
def load_world_indices():
    return get_world_indices()

@st.cache_data(ttl=10800)
def load_vn_indices():
    return get_vn_indices()

@st.cache_data(ttl=10800)
def load_vnindex_analysis():
    """Phan tich VNINDEX tu CSV"""
    BASE = os.path.dirname(os.path.abspath(__file__))
    for fname in ['VNINDEX_full.csv', 'VNINDEX.csv']:
        path = os.path.join(BASE, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').reset_index(drop=True)
            return df
    return None

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

    main = next((i for i in vn_indices if i["name"] == "VNINDEX"), None)
    if main:
        if "🔴" in main["trend"]:
            st.warning(f"⚠️ VNINDEX xu hướng **giảm** (score={main['score']})")
        elif "🟢" in main["trend"]:
            st.success(f"✅ VNINDEX xu hướng **tăng** (score={main['score']})")
        else:
            st.info(f"🟡 VNINDEX **trung lập** (score={main['score']})")
else:
    st.warning("Không tải được dữ liệu thị trường VN.")

# ===================== PHAN TICH VNINDEX TU DONG =====================
st.subheader("📊 Phân tích VNINDEX")

df_vn = load_vnindex_analysis()
if df_vn is not None:
    last = df_vn.iloc[-1]
    prev = df_vn.iloc[-2]

    close  = last['close']
    ema20  = last['EMA_20']
    ema50  = last['EMA_50']
    rsi    = last['RSI_14']
    macd   = last['MACD']
    macd_s = last['MACD_signal']
    macd_h = last['MACD_hist']
    bb_u   = last['BB_upper']
    bb_l   = last['BB_lower']
    atr    = last['ATR_14']
    obv    = last['OBV']
    vol    = last['volume']
    vol_avg = df_vn['volume'].tail(20).mean()

    # Tinh OBV EMA
    import ta
    obv_ema = ta.trend.ema_indicator(df_vn['OBV'], window=20).iloc[-1]
    obv_trend = obv > obv_ema

    # Tinh diem
    signals = []
    score = 0

    def _add(icon, desc, s):
        signals.append((icon, desc, s))
        return s

    score += _add("🟢" if ema20 > ema50 else "🔴",
                  f"EMA20 {'>' if ema20 > ema50 else '<'} EMA50 — {'uptrend' if ema20 > ema50 else 'downtrend'} trung hạn",
                  +1 if ema20 > ema50 else -1)
    score += _add("🟢" if close > ema20 else "🔴",
                  f"Giá {'trên' if close > ema20 else 'dưới'} EMA20 ({ema20:.2f})",
                  +1 if close > ema20 else -1)
    if rsi < 30:
        score += _add("🟢", f"RSI = {rsi:.1f} — quá bán", +2)
    elif rsi > 70:
        score += _add("🔴", f"RSI = {rsi:.1f} — quá mua", -2)
    elif rsi < 45:
        score += _add("🔴", f"RSI = {rsi:.1f} — momentum yếu", -1)
    elif rsi > 55:
        score += _add("🟢", f"RSI = {rsi:.1f} — momentum tốt", +1)
    else:
        _add("🟡", f"RSI = {rsi:.1f} — trung tính", 0)

    prev_macd, prev_macd_s = prev['MACD'], prev['MACD_signal']
    if macd > macd_s and prev_macd <= prev_macd_s:
        score += _add("🟢", "MACD vừa cắt lên Signal — tín hiệu mua mạnh", +2)
    elif macd < macd_s and prev_macd >= prev_macd_s:
        score += _add("🔴", "MACD vừa cắt xuống Signal — tín hiệu bán mạnh", -2)
    elif macd > macd_s:
        score += _add("🟢", f"MACD ({macd:.2f}) > Signal ({macd_s:.2f})", +1)
    else:
        score += _add("🔴", f"MACD ({macd:.2f}) < Signal ({macd_s:.2f}) — bearish", -1)

    prev_h = prev['MACD_hist']
    if macd_h < 0 and prev_h >= 0:
        score += _add("🔴", "MACD Histogram vừa chuyển âm", -1)
    elif macd_h > 0 and prev_h <= 0:
        score += _add("🟢", "MACD Histogram vừa chuyển dương", +1)
    elif macd_h < 0:
        gap, prev_gap = abs(macd - macd_s), abs(prev_macd - prev_macd_s)
        if gap > prev_gap:
            score += _add("🔴", f"MACD Histogram âm nới rộng ({macd_h:.2f})", -1)
        else:
            _add("🟡", f"MACD Histogram âm thu hẹp ({macd_h:.2f}) — có thể đảo chiều", 0)

    bb_pct = (close - bb_l) / (bb_u - bb_l) * 100
    if close < bb_l:
        score += _add("🟢", f"Giá dưới BB Lower ({bb_l:.2f}) — oversold", +2)
    elif close > bb_u:
        score += _add("🔴", f"Giá vượt BB Upper ({bb_u:.2f}) — overbought", -1)
    elif bb_pct < 20:
        score += _add("🟢", f"Giá gần BB Lower ({bb_pct:.0f}%) — vùng hỗ trợ", +1)
    else:
        _add("🟡", f"Giá trong dải BB ({bb_pct:.0f}%)", 0)

    score += _add("🟢" if obv_trend else "🔴",
                  "OBV > EMA20 — dòng tiền vào" if obv_trend else "OBV < EMA20 — dòng tiền ra",
                  +1 if obv_trend else -1)

    vol_ratio = vol / vol_avg
    if vol_ratio > 1.5:
        score += _add("🟢", f"Volume đột biến {vol_ratio:.1f}x TB20", +1)
    elif vol_ratio < 0.7:
        _add("🟡", f"Volume thấp {vol_ratio:.1f}x TB20 — thiếu xác nhận", 0)
    else:
        _add("🟡", f"Volume {vol_ratio:.1f}x TB20 — bình thường", 0)

    trend_5 = df_vn['close'].iloc[-1] - df_vn['close'].iloc[-5]
    score += _add("🟢" if trend_5 > 0 else "🔴",
                  f"Xu hướng 5 phiên: {trend_5:+.2f} điểm",
                  +1 if trend_5 > 0 else -1)

    # Verdict
    if score >= 4:
        verdict, color = "🟢 MUA — Tín hiệu tích cực mạnh", "success"
    elif score >= 2:
        verdict, color = "🟢 NGHIÊNG VỀ MUA", "success"
    elif score >= 0:
        verdict, color = "🟡 TRUNG LẬP — Chờ thêm tín hiệu", "warning"
    elif score >= -2:
        verdict, color = "🔴 NGHIÊNG VỀ BÁN — Thận trọng", "error"
    else:
        verdict, color = "🔴 TRÁNH / BÁN — Tín hiệu tiêu cực", "error"

    msg = f"**VNINDEX {last['time'].date()} — {close:,.2f}** &nbsp;|&nbsp; {verdict} &nbsp;|&nbsp; Điểm: **{score:+d}**"
    if color == "success":
        st.success(msg)
    elif color == "error":
        st.error(msg)
    else:
        st.warning(msg)

    # Chi tiet + Khuyen nghi
    col_sig, col_rec = st.columns(2)
    with col_sig:
        st.markdown("**📊 Tín hiệu:**")
        for icon, desc, s in signals:
            sign = f"`{'+' if s > 0 else ''}{s}`" if s != 0 else "`  0`"
            st.markdown(f"{icon} {sign} {desc}")

    with col_rec:
        st.markdown("**🎯 Khuyến nghị:**")
        sl = round(close - 2 * atr, 2) if pd.notna(atr) else None
        tp1 = round(ema20, 2)
        tp2 = round(bb_u, 2)

        # Nhan xet thong minh theo logic
        reasons = []
        if macd < macd_s and abs(macd - macd_s) > abs(prev_macd - prev_macd_s):
            reasons.append("⚠️ MACD đang diverge — chưa có tín hiệu đảo chiều")
        if rsi > 35 and close > bb_l + (bb_u - bb_l) * 0.15:
            reasons.append(f"⚠️ Giá chưa chạm vùng hỗ trợ mạnh (BB Lower {bb_l:.0f}, EMA50 {ema50:.0f})")
        if not obv_trend:
            reasons.append("⚠️ OBV giảm — dòng tiền thực sự đang rút")

        if score <= -1 and reasons:
            st.markdown("**Chưa nên mua ngay. Lý do:**")
            for r in reasons:
                st.markdown(f"- {r}")
            st.markdown("**Kịch bản theo dõi:**")
            st.markdown(f"- Về **{bb_l:.0f}-{ema50:.0f}** (BB Lower + EMA50) → mua thăm dò")
            st.markdown(f"- RSI về **40-42** + MACD histogram thu hẹp → tín hiệu tốt hơn")
            if sl:
                st.markdown(f"- Stoploss: **{sl:,.2f}** | Target 1: **{tp1:,.2f}** | Target 2: **{tp2:,.2f}**")
        elif score >= 2:
            st.markdown("✅ Tín hiệu tích cực — có thể cân nhắc mua")
            if sl:
                st.markdown(f"- Stoploss: **{sl:,.2f}** | Target 1: **{tp1:,.2f}** | Target 2: **{tp2:,.2f}**")
        else:
            st.markdown("🟡 Thị trường trung lập — chờ tín hiệu rõ hơn")
            st.markdown(f"- Hỗ trợ: **{bb_l:.0f}** (BB Lower) | **{ema50:.0f}** (EMA50)")
            st.markdown(f"- Kháng cự: **{ema20:.0f}** (EMA20) | **{bb_u:.0f}** (BB Upper)")

st.divider()

# --- Tabs ---
tab_analyze, tab_update = st.tabs(["🤖 Phân tích kỹ thuật chi tiết", "📥 Cập nhật dữ liệu"])

with tab_update:
    st.subheader("📥 Cập nhật dữ liệu CSV")
    st.caption("Tải dữ liệu OHLCV + tính toán 50+ chỉ báo kỹ thuật, lưu vào file CSV.")

    u1, u2, u3, u4 = st.columns([2, 2, 2, 1])
    with u1:
        update_symbol = st.text_input("Ðạ cổ phiếu", value="VIX", key="update_sym").upper()
    with u2:
        update_start = st.date_input("Từ ngày", value=date.today() - timedelta(days=365), key="update_start")
    with u3:
        update_end = st.date_input("Đến ngày", value=date.today(), key="update_end")
    with u4:
        st.write("")
        st.write("")
        run_update = st.button("📥 Cập nhật", type="primary", key="btn_update")

    # Hien thi file hien co
    csv_files = [f for f in os.listdir(BASE_DIR) if f.endswith('.csv') and not f.startswith('.')]
    if csv_files:
        st.caption(f"📂 File CSV hiện có: {', '.join(sorted(csv_files))}")

    if run_update:
        csv_out = os.path.join(BASE_DIR, f"{update_symbol}.csv")
        with st.status(f"Đang cập nhật {update_symbol}...", expanded=True) as status:
            try:
                from vnstock.api.quote import Quote
                from analysis import add_indicators

                st.write(f"📡 Đang tải dữ liệu {update_symbol} từ {update_start} đến {update_end}...")
                q = Quote(symbol=update_symbol, source='VCI')
                df_new = q.history(start=str(update_start), end=str(update_end), interval='1D')

                if df_new is None or len(df_new) == 0:
                    status.update(label="❌ Không có dữ liệu", state="error")
                    st.error(f"Không tìm thấy dữ liệu cho mã {update_symbol}")
                else:
                    df_new.columns = [c.lower() for c in df_new.columns]
                    df_new['time'] = pd.to_datetime(df_new['time'])

                    # Gop voi file cu neu co
                    base_cols = ['time','open','high','low','close','volume']
                    if os.path.exists(csv_out):
                        st.write(f"📂 Ghép với dữ liệu cũ...")
                        df_old = pd.read_csv(csv_out)
                        df_old['time'] = pd.to_datetime(df_old['time'])
                        df = pd.concat([df_old[base_cols], df_new[base_cols]], ignore_index=True)
                    else:
                        df = df_new[base_cols].copy()

                    df = df.drop_duplicates('time').sort_values('time').reset_index(drop=True)

                    st.write(f"🔢 Tính toán 50+ chỉ báo kỹ thuật...")
                    df = add_indicators(df)
                    df.round(4).to_csv(csv_out, index=False)

                    status.update(label=f"✅ Cập nhật xong {update_symbol}", state="complete")
                    st.success(f"✅ **{update_symbol}**: {len(df)} phiên | Từ {df.time.iloc[0].date()} đến {df.time.iloc[-1].date()} | {len(df.columns)} cột")
                    st.caption(f"📂 Lưu tại: {csv_out}")

            except Exception as e:
                status.update(label=f"❌ Lỗi: {e}", state="error")
                st.error(f"Lỗi: {e}")

with tab_analyze:
    st.subheader("🤖 Phân tích kỹ thuật chi tiết")

# Lay danh sach ma co CSV
avail = [f.replace('_full.csv','').replace('.csv','') for f in os.listdir(BASE_DIR)
         if f.endswith('.csv') and not f.startswith('.')]
avail = sorted(set(avail))

c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    symbol_input = st.selectbox("Mã cổ phiếu", avail, index=avail.index('VNINDEX') if 'VNINDEX' in avail else 0)
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
        analyze_date = st.date_input("Ngày phân tích", value=date.today())
        st.caption(f"⚠️ Chưa có file {symbol_input}_full.csv")
with c3:
    st.write("")
    st.write("")
    run_btn = st.button("🔍 Phân tích", type="primary", disabled=not csv_exists)

if run_btn:
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df = df[df["time"] <= pd.Timestamp(str(analyze_date))]

    if len(df) < 5:
        st.error("Không đủ dữ liệu.")
        st.stop()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    close  = last["close"]
    ema20  = last["EMA_20"]
    ema50  = last["EMA_50"]
    rsi    = last["RSI_14"]
    macd   = last["MACD"]
    macd_s = last["MACD_signal"]
    macd_h = last["MACD_hist"]
    bb_u   = last["BB_upper"]
    bb_l   = last["BB_lower"]
    stoch  = last["STOCH_k"]
    atr    = last["ATR_14"]
    vol    = last["volume"]
    vol_avg = df["volume"].tail(20).mean()

    # Header
    st.subheader(f"{symbol_input} — {last['time'].date()} — Giá: {close:,.2f}")

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("EMA20", f"{ema20:,.2f}", delta=f"{close-ema20:+.2f}", delta_color="normal")
    m2.metric("EMA50", f"{ema50:,.2f}", delta=f"{close-ema50:+.2f}", delta_color="normal")
    m3.metric("RSI", f"{rsi:.1f}")
    m4.metric("MACD", f"{macd:.2f}", delta=f"Signal {macd_s:.2f}", delta_color="off")
    m5.metric("Volume / TB20", f"{vol/vol_avg*100:.0f}%")

    # --- Realtime data tu price_board ---
    from datetime import date as date_cls
    is_today = (str(analyze_date) == str(date_cls.today()))

    if is_today:
        try:
            from vnstock import Trading, Finance
            t_board = Trading(symbol=symbol_input, source='VCI')
            pb = t_board.price_board()
            pb.columns = ['_'.join(c) for c in pb.columns]
            row = pb.iloc[0]

            st.markdown("**📊 Dữ liệu phiên hôm nay:**")
            r1, r2, r3, r4 = st.columns(4)
            ref   = row.get('listing_ref_price') or 0
            price_rt = row.get('match_match_price') or 0
            pct_rt = (price_rt - ref) / ref * 100 if ref else 0
            high  = row.get('match_highest') or 0
            low_  = row.get('match_lowest') or 0
            acc_vol = row.get('match_accumulated_volume') or 0
            acc_val = row.get('match_accumulated_value') or 0
            r1.metric("Giá khớp", f"{price_rt:,.0f}", delta=f"{pct_rt:+.2f}%", delta_color="normal")
            r2.metric("Cao / Thấp", f"{high:,.0f} / {low_:,.0f}")
            r3.metric("KL khớp", f"{acc_vol/1e6:.1f}M CP")
            r4.metric("GT khớp", f"{acc_val/1e3:.0f} tỷ")

            st.markdown("**📊 Cung cầu (3 bước giá):**")
            col_b, col_a = st.columns(2)
            with col_b:
                st.markdown("**🟢 Bên mua (Cầu)**")
                for i in [1,2,3]:
                    bp = row.get(f'bid_ask_bid_{i}_price') or 0
                    bv = row.get(f'bid_ask_bid_{i}_volume') or 0
                    if bp:
                        bar = '█' * min(int(bv/50000), 20)
                        st.markdown(f"`{bp:,.0f}` &nbsp; {bv:,.0f} CP &nbsp; {bar}")
            with col_a:
                st.markdown("**🔴 Bên bán (Cung)**")
                for i in [1,2,3]:
                    ap = row.get(f'bid_ask_ask_{i}_price') or 0
                    av = row.get(f'bid_ask_ask_{i}_volume') or 0
                    if ap:
                        bar = '█' * min(int(av/50000), 20)
                        st.markdown(f"`{ap:,.0f}` &nbsp; {av:,.0f} CP &nbsp; {bar}")

            total_bid = sum(row.get(f'bid_ask_bid_{i}_volume') or 0 for i in [1,2,3])
            total_ask = sum(row.get(f'bid_ask_ask_{i}_volume') or 0 for i in [1,2,3])
            if total_bid + total_ask > 0:
                bid_pct = total_bid / (total_bid + total_ask) * 100
                if bid_pct > 65:
                    st.success(f"🟢 Cầu mạnh hơn cung ({bid_pct:.0f}% / {100-bid_pct:.0f}%)")
                elif bid_pct < 35:
                    st.error(f"🔴 Cung mạnh hơn cầu ({bid_pct:.0f}% / {100-bid_pct:.0f}%)")
                else:
                    st.info(f"🟡 Cung cầu cân bằng ({bid_pct:.0f}% / {100-bid_pct:.0f}%)")

            bid1_v = row.get('bid_ask_bid_1_volume') or 0
            ask1_v = row.get('bid_ask_ask_1_volume') or 0
            if bid1_v == 0 and total_bid < 10000:
                st.warning("⚠️ Cạn cầu — không còn lệnh mua gần giá khớp")
            if ask1_v == 0 and total_ask < 10000:
                st.warning("⚠️ Cạn cung — không còn lệnh bán gần giá khớp")

            st.markdown("**🌐 Khối ngoại:**")
            fb_vol = row.get('match_foreign_buy_volume') or 0
            fs_vol = row.get('match_foreign_sell_volume') or 0
            fb_val = (row.get('match_foreign_buy_value') or 0) / 1e9
            fs_val = (row.get('match_foreign_sell_value') or 0) / 1e9
            net_vol = fb_vol - fs_vol
            net_val = fb_val - fs_val
            room = row.get('match_current_room') or 0
            total_room = row.get('match_total_room') or 1
            room_pct = room / total_room * 100
            kn1, kn2, kn3, kn4 = st.columns(4)
            kn1.metric("Mua ròng KL", f"{net_vol:+,.0f} CP")
            kn2.metric("Mua ròng GT", f"{net_val:+.1f} tỷ")
            kn3.metric("NN Mua", f"{fb_vol:,.0f} CP")
            kn4.metric("NN Bán", f"{fs_vol:,.0f} CP")
            st.caption(f"Room nước ngoài còn lại: {room:,.0f} CP ({room_pct:.1f}%)")
            if net_val > 0:
                st.success(f"✅ Khối ngoại **mua ròng** {net_val:+.1f} tỷ")
            elif net_val < -1:
                st.error(f"⚠️ Khối ngoại **bán ròng** {net_val:.1f} tỷ")

            try:
                f_api = Finance(symbol=symbol_input, source='VCI')
                df_ratio = f_api.ratio(period='quarter', lang='en')
                if not df_ratio.empty:
                    pe_row = df_ratio[df_ratio['item_en'].str.contains('P/E', na=False)]
                    pb_row = df_ratio[df_ratio['item_en'].str.contains('P/B', na=False)]
                    val_cols = [c for c in df_ratio.columns if c not in ['item','item_en','item_id']]
                    if val_cols and (not pe_row.empty or not pb_row.empty):
                        last_col = val_cols[-1]
                        st.markdown("**💰 Định giá:**")
                        d1, d2 = st.columns(2)
                        if not pe_row.empty:
                            d1.metric("P/E", f"{pe_row[last_col].values[0]:.1f}x")
                        if not pb_row.empty:
                            d2.metric("P/B", f"{pb_row[last_col].values[0]:.1f}x")
            except Exception:
                pass

        except Exception as e:
            st.caption(f"Không tải được dữ liệu realtime: {e}")
    else:
        st.info("📅 Dữ liệu cầu/cung, khối ngoại chỉ hiển thị khi phân tích ngày hôm nay.")

    st.divider()

    # Tinh diem
    signals = []
    score = 0

    def add(icon, desc, s):
        signals.append((icon, desc, s))
        return s

    score += add("🟢" if ema20 > ema50 else "🔴",
                 f"EMA20 {'>' if ema20 > ema50 else '<'} EMA50 — {'uptrend' if ema20 > ema50 else 'downtrend'} trung hạn",
                 +1 if ema20 > ema50 else -1)

    score += add("🟢" if close > ema20 else "🔴",
                 f"Giá {'trên' if close > ema20 else 'dưới'} EMA20 ({ema20:.2f})",
                 +1 if close > ema20 else -1)

    if rsi < 30:
        score += add("🟢", f"RSI = {rsi:.1f} — quá bán, khả năng bật tăng", +2)
    elif rsi > 70:
        score += add("🔴", f"RSI = {rsi:.1f} — quá mua, cẩn thận điều chỉnh", -2)
    elif rsi < 45:
        score += add("🔴", f"RSI = {rsi:.1f} — momentum yếu", -1)
    elif rsi > 55:
        score += add("🟢", f"RSI = {rsi:.1f} — momentum tốt", +1)
    else:
        add("🟡", f"RSI = {rsi:.1f} — trung tính", 0)

    if pd.notna(macd) and pd.notna(macd_s):
        prev_macd, prev_macd_s = prev["MACD"], prev["MACD_signal"]
        if macd > macd_s and prev_macd <= prev_macd_s:
            score += add("🟢", "MACD vừa cắt lên Signal — tín hiệu mua mạnh", +2)
        elif macd < macd_s and prev_macd >= prev_macd_s:
            score += add("🔴", "MACD vừa cắt xuống Signal — tín hiệu bán mạnh", -2)
        elif macd > macd_s:
            score += add("🟢", f"MACD ({macd:.2f}) > Signal ({macd_s:.2f})", +1)
        else:
            score += add("🔴", f"MACD ({macd:.2f}) < Signal ({macd_s:.2f}) — bearish", -1)

        prev_h = prev["MACD_hist"]
        if macd_h < 0 and prev_h >= 0:
            score += add("🔴", "MACD Histogram vừa chuyển âm — momentum giảm", -1)
        elif macd_h > 0 and prev_h <= 0:
            score += add("🟢", "MACD Histogram vừa chuyển dương — momentum tăng", +1)
        elif macd_h < 0:
            gap = abs(macd - macd_s)
            prev_gap = abs(prev_macd - prev_macd_s)
            if gap > prev_gap:
                score += add("🔴", f"MACD Histogram âm và nới rộng ({macd_h:.2f})", -1)
            else:
                add("🟡", f"MACD Histogram âm nhưng thu hẹp ({macd_h:.2f}) — có thể đảo chiều", 0)

    if pd.notna(bb_u) and pd.notna(bb_l):
        bb_pct = (close - bb_l) / (bb_u - bb_l) * 100
        if close < bb_l:
            score += add("🟢", f"Giá dưới BB Lower ({bb_l:.2f}) — oversold mạnh", +2)
        elif close > bb_u:
            score += add("🔴", f"Giá vượt BB Upper ({bb_u:.2f}) — overbought", -1)
        elif bb_pct < 20:
            score += add("🟢", f"Giá gần BB Lower ({bb_pct:.0f}% từ đáy dải) — vùng hỗ trợ", +1)
        elif bb_pct > 80:
            add("🟡", f"Giá gần BB Upper ({bb_pct:.0f}% từ đáy dải) — cẩn thận", 0)
        else:
            add("🟡", f"Giá trong dải BB ({bb_pct:.0f}% từ đáy dải)", 0)

    if pd.notna(stoch):
        if stoch < 20:
            score += add("🟢", f"Stochastic %K = {stoch:.1f} — quá bán sâu", +1)
        elif stoch > 80:
            score += add("🔴", f"Stochastic %K = {stoch:.1f} — quá mua", -1)
        else:
            add("🟡", f"Stochastic %K = {stoch:.1f} — trung tính", 0)

    vol_ratio = vol / vol_avg
    if vol_ratio > 1.5:
        score += add("🟢", f"Volume đột biến {vol_ratio:.1f}x TB20 — xác nhận tín hiệu", +1)
    elif vol_ratio < 0.7:
        add("🟡", f"Volume thấp {vol_ratio:.1f}x TB20 — thiếu xác nhận", 0)
    else:
        add("🟡", f"Volume {vol_ratio:.1f}x TB20 — bình thường", 0)

    trend_5 = df["close"].iloc[-1] - df["close"].iloc[-5]
    if trend_5 > 0:
        score += add("🟢", f"Xu hướng 5 phiên: +{trend_5:.2f} — tăng", +1)
    else:
        score += add("🔴", f"Xu hướng 5 phiên: {trend_5:.2f} — giảm liên tiếp", -1)

    # Verdict
    if score >= 4:
        verdict, color = "🟢 MUA — Tín hiệu tích cực mạnh", "success"
    elif score >= 2:
        verdict, color = "🟢 NGHIÊNG VỀ MUA — Cân nhắc mua thăm dò", "success"
    elif score >= 0:
        verdict, color = "🟡 TRUNG LẬP — Chờ thêm tín hiệu", "warning"
    elif score >= -2:
        verdict, color = "🔴 NGHIÊNG VỀ BÁN — Thận trọng", "error"
    else:
        verdict, color = "🔴 TRÁNH / BÁN — Tín hiệu tiêu cực", "error"

    msg = f"**{verdict}** &nbsp;|&nbsp; Tổng điểm: **{score:+d}**"
    if color == "success":
        st.success(msg)
    elif color == "error":
        st.error(msg)
    else:
        st.warning(msg)

    # Chi tiet tin hieu
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**📊 Chi tiết tín hiệu:**")
        for icon, desc, s in signals:
            sign = f"`{'+' if s > 0 else ''}{s}`" if s != 0 else "`  0`"
            st.markdown(f"{icon} {sign} {desc}")

    with col_r:
        st.markdown("**🎯 Vùng giá quan trọng:**")
        if pd.notna(atr):
            sl = round(close - 2 * atr, 2)
            tp1 = round(ema20, 2)
            tp2 = round(bb_u, 2)
            st.markdown(f"🔴 Stop Loss (2x ATR): **{sl:,.2f}** ({(sl-close)/close*100:+.1f}%)")
            st.markdown(f"🟡 Take Profit 1 (EMA20): **{tp1:,.2f}** ({(tp1-close)/close*100:+.1f}%)")
            st.markdown(f"🟢 Take Profit 2 (BB Upper): **{tp2:,.2f}** ({(tp2-close)/close*100:+.1f}%)")
        st.markdown("---")
        st.markdown(f"⬆️ Kháng cự 1: EMA20 = **{ema20:,.2f}**")
        st.markdown(f"⬆️ Kháng cự 2: BB Upper = **{bb_u:,.2f}**")
        st.markdown(f"⬇️ Hỗ trợ 1: BB Lower = **{bb_l:,.2f}**")
        st.markdown(f"⬇️ Hỗ trợ 2: EMA50 = **{ema50:,.2f}**")

    # Bieu do
    st.divider()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03,
                        subplot_titles=("Giá", "Volume", "RSI"))

    fig.add_trace(go.Candlestick(x=df["time"], open=df["open"], high=df["high"],
                                  low=df["low"], close=df["close"], name="Giá",
                                  increasing_line_color="#26a69a", decreasing_line_color="#ef5350"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["EMA_20"], name="EMA20",
                              line=dict(color="orange", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["EMA_50"], name="EMA50",
                              line=dict(color="blue", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_upper"], name="BB Upper",
                              line=dict(color="gray", dash="dash", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_lower"], name="BB Lower",
                              line=dict(color="gray", dash="dash", width=1),
                              fill="tonexty", fillcolor="rgba(128,128,128,0.1)"), row=1, col=1)

    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df["time"], y=df["volume"], name="Volume",
                          marker_color=colors, showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=df["time"], y=df["RSI_14"], name="RSI",
                              line=dict(color="purple", width=1.2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_layout(height=700, xaxis_rangeslider_visible=False,
                      template="plotly_dark", legend=dict(orientation="h", y=1.02),
                      margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)
