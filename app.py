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
        # Lay du lieu tu CSV de tinh cac chi so
        _df = load_vnindex_analysis()
        if _df is not None:
            import ta as _ta
            _last = _df.iloc[-1]
            _prev = _df.iloc[-2]
            _df20 = _df.tail(20)
            _df60 = _df.tail(60)
            _close  = _last['close']
            _ema20  = _last['EMA_20']
            _ema50  = _last['EMA_50']
            _ema200 = _last['EMA_200']
            _rsi    = _last['RSI_14']
            _macd   = _last['MACD']
            _macd_s = _last['MACD_signal']
            _macd_h = _last['MACD_hist']
            _bb_u   = _last['BB_upper']
            _bb_l   = _last['BB_lower']
            _obv    = _last['OBV']
            _obv_ema = _ta.trend.ema_indicator(_df['OBV'], window=20).iloc[-1]
            _vol    = _last['volume']
            _vol_avg = _df['volume'].tail(20).mean()
            _vol_ratio = _vol / _vol_avg
            _trend5 = _df['close'].iloc[-1] - _df['close'].iloc[-5]
            _prev_macd = _prev['MACD']
            _prev_macd_s = _prev['MACD_signal']
            _prev_h = _prev['MACD_hist']

            # Tinh score tung tieu chi
            _s = {}
            _s['ema'] = 1 if _ema20 > _ema50 else -1
            _s['price_ema20'] = 1 if _close > _ema20 else -1
            if _rsi < 30: _s['rsi'] = 2
            elif _rsi > 70: _s['rsi'] = -2
            elif _rsi < 45: _s['rsi'] = -1
            elif _rsi > 55: _s['rsi'] = 1
            else: _s['rsi'] = 0
            if _macd > _macd_s and _prev_macd <= _prev_macd_s: _s['macd'] = 2
            elif _macd < _macd_s and _prev_macd >= _prev_macd_s: _s['macd'] = -2
            elif _macd > _macd_s: _s['macd'] = 1
            else: _s['macd'] = -1
            if _macd_h < 0 and _prev_h >= 0: _s['hist'] = -1
            elif _macd_h > 0 and _prev_h <= 0: _s['hist'] = 1
            elif _macd_h < 0:
                _s['hist'] = -1 if abs(_macd - _macd_s) > abs(_prev_macd - _prev_macd_s) else 0
            else: _s['hist'] = 0
            _bb_pct = (_close - _bb_l) / (_bb_u - _bb_l) * 100 if (_bb_u - _bb_l) > 0 else 50
            if _close < _bb_l: _s['bb'] = 2
            elif _close > _bb_u: _s['bb'] = -1
            elif _bb_pct < 20: _s['bb'] = 1
            else: _s['bb'] = 0
            _s['obv'] = 1 if _obv > _obv_ema else -1
            _s['vol'] = 1 if _vol_ratio > 1.5 else 0
            _s['trend5'] = 1 if _trend5 > 0 else -1

            # Wyckoff phase score
            _high60 = _df60['high'].max()
            _low60  = _df60['low'].min()
            _range60 = _high60 - _low60
            _price_trend20 = _df['close'].iloc[-1] - _df['close'].iloc[-20]
            _vol_trend_up = _df['volume'].tail(5).mean() > _df['volume'].tail(20).mean()
            _near_low  = _close <= _low60 + _range60 * 0.25
            _near_high = _close >= _high60 - _range60 * 0.25
            _obv_rising = _obv > _obv_ema
            _obv5 = _df['OBV'].iloc[-1] - _df['OBV'].iloc[-5]
            _spread = _last['high'] - _last['low']
            _spread_avg = (_df20['high'] - _df20['low']).mean()
            _wide = _spread > _spread_avg * 1.3
            _narrow = _spread < _spread_avg * 0.7
            _close_pos = (_close - _last['low']) / _spread if _spread > 0 else 0.5
            if _near_low and _obv_rising and _vol_trend_up: _wyckoff_s = 2
            elif _near_low and not _obv_rising: _wyckoff_s = -1
            elif _near_high and not _obv_rising and _vol_ratio > 1.2: _wyckoff_s = -2
            elif _near_high and _obv_rising: _wyckoff_s = 2
            elif not _near_low and not _near_high and _price_trend20 > 0 and _obv_rising: _wyckoff_s = 1
            elif not _near_low and not _near_high and _price_trend20 < 0 and not _obv_rising: _wyckoff_s = -1
            else: _wyckoff_s = 0
            if _wide and _close_pos > 0.6 and _vol_ratio > 1.2: _wyckoff_s += 1
            elif _wide and _close_pos < 0.4 and _vol_ratio > 1.2: _wyckoff_s -= 1

            # Khoi ngoai (lay tu price_board neu co)
            _net_val = None
            try:
                from vnstock import Trading
                _pb = Trading(symbol='VNINDEX', source='VCI').price_board()
                _pb.columns = ['_'.join(c) for c in _pb.columns]
                _row = _pb.iloc[0]
                _fb_val = (_row.get('match_foreign_buy_value') or 0) / 1e9
                _fs_val = (_row.get('match_foreign_sell_value') or 0) / 1e9
                _net_val = _fb_val - _fs_val
            except Exception:
                pass

            # Tong score
            _total = sum(_s.values()) + _wyckoff_s
            if _net_val is not None:
                _total += 1 if _net_val > 0 else (-1 if _net_val < -100 else 0)

            # Hien thi bang tieu chi
            def _icon(v): return "🟢" if v > 0 else ("🔴" if v < 0 else "🟡")

            rows = [
                (_icon(_s['ema']),      f"EMA20 {'>' if _ema20 > _ema50 else '<'} EMA50"),
                (_icon(_s['price_ema20']), f"Giá {'trên' if _close > _ema20 else 'dưới'} EMA20 ({_ema20:,.0f})"),
                (_icon(_s['rsi']),      f"RSI = {_rsi:.1f}"),
                (_icon(_s['macd']),     f"MACD {_macd:.1f} vs Signal {_macd_s:.1f}"),
                (_icon(_s['hist']),     f"Histogram {_macd_h:.2f} {'nới rộng' if abs(_macd-_macd_s)>abs(_prev_macd-_prev_macd_s) else 'thu hẹp'}"),
                (_icon(_s['bb']),       f"BB {_bb_pct:.0f}% {'(gần Lower)' if _bb_pct < 20 else '(gần Upper)' if _bb_pct > 80 else ''}"),
                (_icon(_s['obv']),      f"OBV {'>' if _obv > _obv_ema else '<'} EMA20"),
                (_icon(_s['vol']),      f"Volume {_vol_ratio:.1f}x TB20"),
                (_icon(_s['trend5']),   f"Xu hướng 5 phiên: {_trend5:+.1f} điểm"),
            ]
            if _net_val is not None:
                _nv_icon = "🟢" if _net_val > 0 else ("🔴" if _net_val < -100 else "🟡")
                rows.append((_nv_icon, f"NN {'mua' if _net_val > 0 else 'bán'} ròng {abs(_net_val):.0f} tỷ"))
            _wy_icon = "🟢" if _wyckoff_s > 0 else ("🔴" if _wyckoff_s < 0 else "🟡")
            rows.append((_wy_icon, f"Wyckoff: {'Tích lũy/Markup' if _wyckoff_s > 0 else 'Phân phối/Markdown' if _wyckoff_s < 0 else 'Ranging'}"))

            # Verdict
            if _total >= 6: _verdict, _vc = "✅ MUA — Tín hiệu mạnh", "success"
            elif _total >= 3: _verdict, _vc = "✅ Nghiêng về MUA", "success"
            elif _total >= 0: _verdict, _vc = "🟡 Trung lập — Chờ xác nhận", "warning"
            elif _total >= -3: _verdict, _vc = "⚠️ Nghiêng về BÁN", "error"
            else: _verdict, _vc = "🔴 TRÁNH — Tín hiệu tiêu cực", "error"

            # Hien thi
            with st.expander(f"{'✅' if _vc=='success' else '⚠️' if _vc=='error' else '🟡'} VNINDEX {_verdict} (điểm: {_total:+d})", expanded=False):
                for ic, desc in rows:
                    st.markdown(f"{ic} {desc}")
        else:
            if "🔴" in main["trend"]:
                st.warning(f"⚠️ VNINDEX xu hướng **giảm** (score={main['score']})")
            elif "🟢" in main["trend"]:
                st.success(f"✅ VNINDEX xu hướng **tăng** (score={main['score']})")
            else:
                st.info(f"🟡 VNINDEX **trung lập** (score={main['score']})")
else:
    st.warning("Không tải được dữ liệu thị trường VN.")

# ===================== PHAN TICH VNINDEX WYCKOFF =====================
st.subheader("📊 Phân tích VNINDEX — Wyckoff")

df_vn = load_vnindex_analysis()
if df_vn is not None:
    import ta

    last  = df_vn.iloc[-1]
    prev  = df_vn.iloc[-2]
    df_20 = df_vn.tail(20)
    df_60 = df_vn.tail(60)

    close   = last['close']
    high    = last['high']
    low     = last['low']
    vol     = last['volume']
    vol_avg = df_vn['volume'].tail(20).mean()
    vol_avg60 = df_vn['volume'].tail(60).mean()
    atr     = last['ATR_14']
    ema20   = last['EMA_20']
    ema50   = last['EMA_50']
    ema200  = last['EMA_200']
    obv     = last['OBV']
    obv_ema = ta.trend.ema_indicator(df_vn['OBV'], window=20).iloc[-1]
    rsi     = last['RSI_14']
    bb_u    = last['BB_upper']
    bb_l    = last['BB_lower']
    bb_mid  = last['BB_middle']

    # --- Wyckoff: xac dinh Phase ---
    # Lay high/low 60 phien
    high_60 = df_60['high'].max()
    low_60  = df_60['low'].min()
    high_20 = df_20['high'].max()
    low_20  = df_20['low'].min()
    range_60 = high_60 - low_60

    # Volume analysis
    vol_ratio     = vol / vol_avg
    vol_ratio_60  = vol / vol_avg60
    vol_trend_up  = df_vn['volume'].tail(5).mean() > df_vn['volume'].tail(20).mean()  # vol tang dan
    price_trend_5 = df_vn['close'].iloc[-1] - df_vn['close'].iloc[-5]
    price_trend_20 = df_vn['close'].iloc[-1] - df_vn['close'].iloc[-20]

    # Spread (biên độ nến)
    spread     = high - low
    spread_avg = (df_20['high'] - df_20['low']).mean()
    narrow_spread = spread < spread_avg * 0.7  # nến biên độ hẹp
    wide_spread   = spread > spread_avg * 1.3  # nến biên độ rộng

    # Closing position (giá đóng cửa ở đâu trong nến)
    close_pos = (close - low) / (high - low) if (high - low) > 0 else 0.5
    close_upper = close_pos > 0.6   # đóng cửa phần trên nến → strength
    close_lower = close_pos < 0.4   # đóng cửa phần dưới nến → weakness

    # OBV trend
    obv_rising = obv > obv_ema
    obv_5 = df_vn['OBV'].iloc[-1] - df_vn['OBV'].iloc[-5]

    # --- Xac dinh Wyckoff Phase ---
    wyckoff_signals = []

    def wadd(icon, event, detail):
        wyckoff_signals.append((icon, event, detail))

    # 1. PRICE STRUCTURE
    if close > ema200:
        wadd("🟢", "Cấu trúc giá", f"Giá ({close:,.0f}) trên EMA200 ({ema200:,.0f}) — xu hướng dài hạn tăng")
        struct_score = 1
    elif close > ema50:
        wadd("🟡", "Cấu trúc giá", f"Giá trên EMA50 ({ema50:,.0f}) nhưng dưới EMA200 — trung hạn tích cực")
        struct_score = 0
    else:
        wadd("🔴", "Cấu trúc giá", f"Giá ({close:,.0f}) dưới EMA50 ({ema50:,.0f}) và EMA200 ({ema200:,.0f}) — xu hướng yếu")
        struct_score = -1

    # 2. SUPPLY / DEMAND qua Volume + Spread
    if wide_spread and close_upper and vol_ratio > 1.2:
        wadd("🟢", "Demand mạnh (Effort↑ Result↑)", f"Nến rộng, đóng cửa cao, volume {vol_ratio:.1f}x TB — lực cầu áp đảo")
        vsa_score = 2
    elif wide_spread and close_lower and vol_ratio > 1.2:
        wadd("🔴", "Supply mạnh (Effort↑ Result↓)", f"Nến rộng, đóng cửa thấp, volume {vol_ratio:.1f}x TB — lực cung áp đảo")
        vsa_score = -2
    elif narrow_spread and vol_ratio > 1.3:
        wadd("🟡", "No Result (Effort↑ Result↓)", f"Volume cao {vol_ratio:.1f}x nhưng biên độ hẹp — lực cản mạnh, cần thận trọng")
        vsa_score = -1
    elif narrow_spread and vol_ratio < 0.7:
        wadd("🟡", "No Demand / No Supply", f"Volume thấp {vol_ratio:.1f}x, biên độ hẹp — thị trường thiếu động lực")
        vsa_score = 0
    else:
        wadd("🟡", "Volume bình thường", f"Volume {vol_ratio:.1f}x TB20, biên độ nến bình thường")
        vsa_score = 0

    # 3. OBV — ACCUMULATION / DISTRIBUTION
    if obv_rising and obv_5 > 0:
        wadd("🟢", "OBV tích lũy (Accumulation)", f"OBV tăng, dòng tiền thực sự đang vào — smart money mua")
        obv_score = 1
    elif not obv_rising and obv_5 < 0:
        wadd("🔴", "OBV phân phối (Distribution)", f"OBV giảm, dòng tiền rút ra — smart money bán")
        obv_score = -1
    else:
        wadd("🟡", "OBV trung lập", f"OBV chưa xác nhận rõ xu hướng")
        obv_score = 0

    # 4. PHASE DETECTION
    near_low  = close <= low_60 + range_60 * 0.25
    near_high = close >= high_60 - range_60 * 0.25
    in_range  = not near_low and not near_high

    if near_low and obv_rising and vol_trend_up:
        phase = "🔵 Phase C/D — Accumulation (Tích lũy)"
        phase_detail = "Giá vùng đáy 60 phiên, OBV tăng, volume tăng dần — dấu hiệu smart money đang gom hàng"
        phase_color = "info"
        phase_score = 2
    elif near_low and not obv_rising:
        phase = "🔴 Phase A/B — Markdown / Selling Climax"
        phase_detail = "Giá vùng đáy nhưng OBV chưa xác nhận — có thể vẫn đang trong giai đoạn bán tháo"
        phase_color = "error"
        phase_score = -1
    elif near_high and not obv_rising and vol_ratio > 1.2:
        phase = "🟠 Phase B/C — Distribution (Phân phối)"
        phase_detail = "Giá vùng đỉnh 60 phiên, OBV giảm, volume cao — dấu hiệu smart money đang xả hàng"
        phase_color = "warning"
        phase_score = -2
    elif near_high and obv_rising:
        phase = "🟢 Phase D — Markup (Tăng giá)"
        phase_detail = "Giá vùng đỉnh, OBV vẫn tăng — xu hướng tăng còn tiếp diễn"
        phase_color = "success"
        phase_score = 2
    elif in_range and price_trend_20 > 0 and obv_rising:
        phase = "🟢 Phase D — Markup (Tăng giá)"
        phase_detail = "Giá tăng 20 phiên, OBV xác nhận — xu hướng tăng đang hình thành"
        phase_color = "success"
        phase_score = 1
    elif in_range and price_trend_20 < 0 and not obv_rising:
        phase = "🔴 Phase A — Markdown (Giảm giá)"
        phase_detail = "Giá giảm 20 phiên, OBV xác nhận — xu hướng giảm đang diễn ra"
        phase_color = "error"
        phase_score = -1
    else:
        phase = "🟡 Phase B — Ranging (Tích lũy / Phân phối chưa rõ)"
        phase_detail = "Giá dao động trong vùng, chưa xác định được hướng — chờ breakout"
        phase_color = "warning"
        phase_score = 0

    # 5. SPRING / UPTHRUST
    spring = low < low_20 * 0.995 and close > low_20 * 0.995 and vol_ratio > 1.0
    upthrust = high > high_20 * 1.005 and close < high_20 * 1.005 and vol_ratio > 1.0
    if spring:
        wadd("🟢", "Spring (Bẫy giảm)", f"Giá xuyên đáy 20 phiên rồi phục hồi — tín hiệu Wyckoff Spring, khả năng đảo chiều tăng")
        spring_score = 2
    elif upthrust:
        wadd("🔴", "Upthrust (Bẫy tăng)", f"Giá xuyên đỉnh 20 phiên rồi quay đầu — tín hiệu Wyckoff Upthrust, khả năng đảo chiều giảm")
        spring_score = -2
    else:
        spring_score = 0

    # --- Tong hop ---
    total_score = struct_score + vsa_score + obv_score + phase_score + spring_score

    # --- Hien thi ---
    msg = f"**VNINDEX {last['time'].date()} — {close:,.2f}**"
    if phase_color == "success":
        st.success(msg)
    elif phase_color == "error":
        st.error(msg)
    elif phase_color == "info":
        st.info(msg)
    else:
        st.warning(msg)

    # Phase box
    st.markdown(f"### {phase}")
    st.caption(phase_detail)

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.markdown("**🔬 Phân tích Wyckoff:**")
        for icon, event, detail in wyckoff_signals:
            st.markdown(f"{icon} **{event}**")
            st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;{detail}")

    with col_w2:
        st.markdown("**📐 Vùng giá Wyckoff:**")
        st.markdown(f"- 🔵 Đỉnh 60 phiên (Resistance): **{high_60:,.2f}**")
        st.markdown(f"- 🔵 Đáy 60 phiên (Support): **{low_60:,.2f}**")
        st.markdown(f"- 🟡 EMA50 (Creek): **{ema50:,.2f}**")
        st.markdown(f"- 🟢 EMA200 (Ice): **{ema200:,.2f}**")
        if pd.notna(atr):
            sl = round(close - 2 * atr, 2)
            st.markdown("---")
            st.markdown(f"🔴 Stop Loss (2x ATR): **{sl:,.2f}** ({(sl-close)/close*100:+.1f}%)")
            if total_score >= 2:
                tp = round(high_60, 2)
                st.markdown(f"🟢 Target (đỉnh 60 phiên): **{tp:,.2f}** ({(tp-close)/close*100:+.1f}%)")

        st.markdown("---")
        if total_score >= 3:
            st.success("✅ Wyckoff: **Tích lũy xong — Có thể mua**")
        elif total_score >= 1:
            st.info("🔵 Wyckoff: **Đang tích lũy — Theo dõi breakout**")
        elif total_score >= -1:
            st.warning("🟡 Wyckoff: **Chưa rõ pha — Chờ xác nhận**")
        else:
            st.error("🔴 Wyckoff: **Phân phối / Markdown — Tránh mua**")

st.divider()

# --- Tabs ---
tab_analyze, tab_scanner, tab_update = st.tabs(["🤖 Phân tích kỹ thuật chi tiết", "🔍 Bộ lọc cổ phiếu", "📥 Cập nhật dữ liệu"])

with tab_scanner:
    st.subheader("🔍 Bộ lọc cổ phiếu — Cạn cung / Sắp bùng nổ")
    st.caption("Quét tất cả file CSV có sẵn, tìm cổ phiếu đang tích lũy xong theo Wyckoff.")

    # Cac tieu chi loc
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        min_score = st.slider("Điểm tối thiểu", 3, 10, 5)
    with fc2:
        top_n = st.selectbox("Hiển thị top", [10, 20, 30, 50], index=1)
    with fc3:
        exclude_index = st.checkbox("Bỏ chỉ số (VNINDEX, VN30...)", value=True)

    run_scan = st.button("🔍 Quét ngay", type="primary", key="btn_scan")

    if run_scan:
        # Lay danh sach CSV
        csv_list = [f for f in os.listdir(BASE_DIR) if f.endswith('.csv') and not f.startswith('.')]
        if exclude_index:
            index_names = {'VNINDEX', 'VN30', 'HNX', 'UPCOM', 'HNX30'}
            csv_list = [f for f in csv_list if f.replace('_full.csv','').replace('.csv','').upper() not in index_names]

        results = []
        progress = st.progress(0, text="Đang quét...")

        for i, fname in enumerate(csv_list):
            progress.progress((i+1)/len(csv_list), text=f"Quét {fname}...")
            try:
                sym = fname.replace('_full.csv','').replace('.csv','')
                _p = os.path.join(BASE_DIR, fname)
                _d = pd.read_csv(_p)
                _d['time'] = pd.to_datetime(_d['time'])
                _d = _d.sort_values('time').reset_index(drop=True)
                if len(_d) < 60: continue

                _last = _d.iloc[-1]
                _prev = _d.iloc[-2]
                _d20  = _d.tail(20)
                _d60  = _d.tail(60)
                _d5   = _d.tail(5)

                close   = _last['close']
                high    = _last['high']
                low     = _last['low']
                vol     = _last['volume']
                vol_avg20 = _d['volume'].tail(20).mean()
                vol_avg5  = _d['volume'].tail(5).mean()
                ema20   = _last['EMA_20']
                ema50   = _last['EMA_50']
                rsi     = _last['RSI_14']
                macd_h  = _last['MACD_hist']
                prev_h  = _prev['MACD_hist']
                atr     = _last['ATR_14']
                atr_5ago = _d.iloc[-6]['ATR_14'] if len(_d) > 6 else atr
                obv     = _last['OBV']
                obv_ema = _d['OBV'].ewm(span=20).mean().iloc[-1]
                obv_5   = _d['OBV'].iloc[-1] - _d['OBV'].iloc[-5]
                bb_u    = _last['BB_upper']
                bb_l    = _last['BB_lower']

                high_60 = _d60['high'].max()
                low_60  = _d60['low'].min()
                high_20 = _d20['high'].max()
                low_20  = _d20['low'].min()
                range_60 = high_60 - low_60

                spread     = high - low
                spread_avg = (_d20['high'] - _d20['low']).mean()
                vol_ratio  = vol / vol_avg20
                close_pos  = (close - low) / spread if spread > 0 else 0.5
                price_trend20 = _d['close'].iloc[-1] - _d['close'].iloc[-20]
                vol_trend_up  = vol_avg5 > vol_avg20
                obv_rising    = obv > obv_ema
                near_low  = close <= low_60 + range_60 * 0.25
                near_high = close >= high_60 - range_60 * 0.25

                score = 0
                tags  = []

                # 1. No Supply — cạn cung
                if vol_avg5 < vol_avg20 * 0.6 and close >= _prev['close']:
                    score += 2; tags.append("🟢 No Supply")

                # 2. Biên độ hẹp dần
                if pd.notna(atr) and pd.notna(atr_5ago) and atr < atr_5ago * 0.8:
                    score += 1; tags.append("🟢 ATR thu hẹp")

                # 3. Nến hẹp đóng cao
                if spread < spread_avg * 0.7 and close_pos > 0.6:
                    score += 1; tags.append("🟢 Nến hẹp đóng cao")

                # 4. OBV tăng (smart money gom)
                if obv_rising and obv_5 > 0:
                    score += 2; tags.append("🟢 OBV tăng")

                # 5. Cấu trúc giá tăng
                if close > ema20 and ema20 > ema50:
                    score += 1; tags.append("🟢 EMA tăng")
                elif close > ema20:
                    score += 0

                # 6. Spring pattern
                if low < low_20 * 0.995 and close > low_20 * 0.995 and vol_ratio > 1.0:
                    score += 3; tags.append("🟢 Spring")

                # 7. MACD hist vừa chuyển dương
                if pd.notna(macd_h) and pd.notna(prev_h) and macd_h > 0 and prev_h <= 0:
                    score += 2; tags.append("🟢 MACD đảo chiều")

                # 8. RSI vùng tốt (45-60)
                if pd.notna(rsi) and 40 <= rsi <= 60:
                    score += 1; tags.append(f"🟢 RSI {rsi:.0f}")

                # 9. Gần kháng cự (sắp breakout)
                if close >= high_20 * 0.97 and close < high_20 * 1.01:
                    score += 1; tags.append("🟢 Gần breakout")

                # 10. Wyckoff Accumulation
                if near_low and obv_rising and vol_trend_up:
                    score += 2; tags.append("🔵 Wyckoff Accum")
                elif not near_low and not near_high and price_trend20 > 0 and obv_rising:
                    score += 1; tags.append("🟢 Wyckoff Markup")

                # Loai tru tin hieu xau
                if close < ema50: score -= 1
                if vol_ratio > 2.0 and close_pos < 0.3: score -= 2  # Volume cao đóng thấp = supply
                if near_high and not obv_rising: score -= 1  # Đỉnh + OBV giảm = phân phối

                if score >= min_score:
                    results.append({
                        'Mã': sym.upper(),
                        'Điểm': score,
                        'Giá': close,
                        'RSI': round(rsi, 1) if pd.notna(rsi) else None,
                        'Vol/TB20': round(vol_ratio, 2),
                        'Tín hiệu': ' | '.join(tags),
                    })
            except Exception:
                continue

        progress.empty()

        if results:
            df_res = pd.DataFrame(results).sort_values('Điểm', ascending=False).head(top_n)
            st.success(f"✅ Tìm thấy **{len(df_res)}** mã (tổng quét {len(csv_list)} file CSV)")
            st.dataframe(
                df_res,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Điểm': st.column_config.ProgressColumn('Điểm', min_value=0, max_value=15),
                    'Giá': st.column_config.NumberColumn('Giá', format="%.2f"),
                    'Vol/TB20': st.column_config.NumberColumn('Vol/TB20', format="%.2f"),
                }
            )
        else:
            st.warning(f"Không tìm thấy mã nào đạt điểm ≥ {min_score}. Thử giảm điểm tối thiểu.")


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

    # --- Wyckoff Analysis ---
    import ta as _ta2
    _d20w = df.tail(20)
    _d60w = df.tail(60)
    _high60w = _d60w['high'].max()
    _low60w  = _d60w['low'].min()
    _high20w = _d20w['high'].max()
    _low20w  = _d20w['low'].min()
    _range60w = _high60w - _low60w
    _obv_emaw = _ta2.trend.ema_indicator(df['OBV'], window=20).iloc[-1]
    _obvw     = last['OBV']
    _obv_risingw = _obvw > _obv_emaw
    _obv5w    = df['OBV'].iloc[-1] - df['OBV'].iloc[-5]
    _vol_avgw = df['volume'].tail(20).mean()
    _vol_ratiow = vol / _vol_avgw
    _spreadw  = last['high'] - last['low']
    _spread_avgw = (_d20w['high'] - _d20w['low']).mean()
    _close_posw  = (_spreadw > 0) and ((close - last['low']) / _spreadw > 0.6)
    _price_trend20w = df['close'].iloc[-1] - df['close'].iloc[-20]
    _vol_trend_upw  = df['volume'].tail(5).mean() > _vol_avgw
    _near_loww  = close <= _low60w + _range60w * 0.25
    _near_highw = close >= _high60w - _range60w * 0.25
    _springw    = last['low'] < _low20w * 0.995 and close > _low20w * 0.995 and _vol_ratiow > 1.0
    _upthrustw  = last['high'] > _high20w * 1.005 and close < _high20w * 1.005 and _vol_ratiow > 1.0

    # Phase
    if _near_loww and _obv_risingw and _vol_trend_upw:
        _phasew, _phase_colorw = "🔵 Accumulation (Phase C/D)", "info"
    elif _near_loww and not _obv_risingw:
        _phasew, _phase_colorw = "🔴 Markdown / Selling Climax (Phase A/B)", "error"
    elif _near_highw and not _obv_risingw and _vol_ratiow > 1.2:
        _phasew, _phase_colorw = "🟠 Distribution (Phase B/C)", "warning"
    elif _near_highw and _obv_risingw:
        _phasew, _phase_colorw = "🟢 Markup (Phase D)", "success"
    elif not _near_loww and not _near_highw and _price_trend20w > 0 and _obv_risingw:
        _phasew, _phase_colorw = "🟢 Markup đang hình thành (Phase D)", "success"
    elif not _near_loww and not _near_highw and _price_trend20w < 0 and not _obv_risingw:
        _phasew, _phase_colorw = "🔴 Markdown (Phase A)", "error"
    else:
        _phasew, _phase_colorw = "🟡 Ranging (Phase B)", "warning"
    if _springw: _phasew += " + 🟢 Spring"
    if _upthrustw: _phasew += " + 🔴 Upthrust"

    st.markdown(f"**📊 Wyckoff:** {_phasew}")
    wc1, wc2, wc3, wc4 = st.columns(4)
    wc1.metric("Đỉnh 60 phiên", f"{_high60w:,.2f}")
    wc2.metric("Đáy 60 phiên", f"{_low60w:,.2f}")
    wc3.metric("OBV vs EMA20", "↑ Tích lũy" if _obv_risingw else "↓ Phân phối")
    wc4.metric("Vol/TB20", f"{_vol_ratiow:.1f}x")

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
