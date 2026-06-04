import streamlit as st
import pandas as pd
import ta
import os


def show_market(load_world_indices, load_vn_indices, load_vnindex_analysis):
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
    if not vn_indices:
        st.warning("Không tải được dữ liệu thị trường VN.")
        return

    cols = st.columns(len(vn_indices))
    for col, idx in zip(cols, vn_indices):
        col.metric(
            label=f"{idx['name']} {idx['trend']}",
            value=f"{idx['close']:,.2f}",
            delta=f"{idx['change']:+.2f} ({idx['pct']:+.2f}%)",
            delta_color="normal"
        )

    main = next((i for i in vn_indices if i["name"] == "VNINDEX"), None)
    if not main:
        return

    _df = load_vnindex_analysis()
    if _df is None:
        if "🔴" in main["trend"]:
            st.warning(f"⚠️ VNINDEX xu hướng **giảm** (score={main['score']})")
        elif "🟢" in main["trend"]:
            st.success(f"✅ VNINDEX xu hướng **tăng** (score={main['score']})")
        else:
            st.info(f"🟡 VNINDEX **trung lập** (score={main['score']})")
        return

    _last = _df.iloc[-1]
    _prev = _df.iloc[-2]
    _df20 = _df.tail(20)
    _df60 = _df.tail(60)
    _close  = _last['close']
    _ema20  = _last['EMA_20']
    _ema50  = _last['EMA_50']
    _rsi    = _last['RSI_14']
    _macd   = _last['MACD']
    _macd_s = _last['MACD_signal']
    _macd_h = _last['MACD_hist']
    _bb_u   = _last['BB_upper']
    _bb_l   = _last['BB_lower']
    _obv    = _last['OBV']
    _obv_ema = ta.trend.ema_indicator(_df['OBV'], window=20).iloc[-1]
    _vol    = _last['volume']
    _vol_avg = _df['volume'].tail(20).mean()
    _vol_ratio = _vol / _vol_avg
    _trend5 = _df['close'].iloc[-1] - _df['close'].iloc[-5]
    _prev_macd   = _prev['MACD']
    _prev_macd_s = _prev['MACD_signal']
    _prev_h      = _prev['MACD_hist']

    # Score tung tieu chi
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

    # Wyckoff phase score — FIX: tách VSA riêng, Markup near_high +1
    _high60 = _df60['high'].max()
    _low60  = _df60['low'].min()
    _range60 = _high60 - _low60
    _price_trend20 = _df['close'].iloc[-1] - _df['close'].iloc[-20]
    _vol_trend_up = _df['volume'].tail(5).mean() > _df['volume'].tail(20).mean()
    _near_low  = _close <= _low60 + _range60 * 0.25
    _near_high = _close >= _high60 - _range60 * 0.25
    _obv_rising = _obv > _obv_ema
    _spread = _last['high'] - _last['low']
    _spread_avg = (_df20['high'] - _df20['low']).mean()
    _narrow = _spread < _spread_avg * 0.7
    _wide   = _spread > _spread_avg * 1.3
    _close_pos = (_close - _last['low']) / _spread if _spread > 0 else 0.5
    _obv5 = _df['OBV'].iloc[-1] - _df['OBV'].iloc[-5]

    # Phase score
    if _near_low and _obv_rising and _vol_trend_up: _wyckoff_s = 2
    elif _near_low and not _obv_rising: _wyckoff_s = -1
    elif _near_high and not _obv_rising and _vol_ratio > 1.2: _wyckoff_s = -2
    elif _near_high and _obv_rising: _wyckoff_s = 1  # FIX: +1 thay vì +2
    elif not _near_low and not _near_high and _price_trend20 > 0 and _obv_rising: _wyckoff_s = 1
    elif not _near_low and not _near_high and _price_trend20 < 0 and not _obv_rising: _wyckoff_s = -1
    else: _wyckoff_s = 0

    # FIX: VSA là tiêu chí riêng, không modify _wyckoff_s
    if _wide and _close_pos > 0.6 and _vol_ratio > 1.2: _vsa_s = 1
    elif _wide and _close_pos < 0.4 and _vol_ratio > 1.2: _vsa_s = -1
    elif _narrow and _vol_ratio < 0.7 and _close_pos > 0.5: _vsa_s = 1  # No Supply thật
    else: _vsa_s = 0

    # Spring/Upthrust — FIX: Spring cần volume thấp
    _high20 = _df20['high'].max()
    _low20  = _df20['low'].min()
    _spring   = _last['low'] < _low20 * 0.995 and _close > _low20 * 0.995 and _vol_ratio < 1.2
    _upthrust = _last['high'] > _high20 * 1.005 and _close < _high20 * 1.005 and _vol_ratio > 1.0
    _spring_s = 2 if _spring else (-2 if _upthrust else 0)

    _wyckoff_total = _wyckoff_s + _vsa_s + _spring_s

    # Khoi ngoai
    _net_val = None
    try:
        from vnstock import Trading
        _pb = Trading(symbol='VNINDEX', source='VCI').price_board()
        _pb.columns = ['_'.join(c) for c in _pb.columns]
        _row = _pb.iloc[0]
        _net_val = (_row.get('match_foreign_buy_value') or 0) / 1e9 - (_row.get('match_foreign_sell_value') or 0) / 1e9
    except Exception:
        pass

    _total = sum(_s.values()) + _wyckoff_total
    if _net_val is not None:
        _total += 1 if _net_val > 0 else (-1 if _net_val < -100 else 0)

    def _icon(v): return "🟢" if v > 0 else ("🔴" if v < 0 else "🟡")

    rows = [
        (_icon(_s['ema']),         f"EMA20 {'>' if _ema20 > _ema50 else '<'} EMA50 — {'Uptrend (tăng)' if _ema20 > _ema50 else 'Downtrend (giảm)'}"),
        (_icon(_s['price_ema20']), f"Giá {'trên' if _close > _ema20 else 'dưới'} EMA20 — Trung bình động 20 ({_ema20:,.0f})"),
        (_icon(_s['rsi']),         f"RSI (Sức mạnh tương đối) = {_rsi:.1f}"),
        (_icon(_s['macd']),        f"MACD (Hội tụ/Phân kỳ) {_macd:.1f} vs Signal {_macd_s:.1f}"),
        (_icon(_s['hist']),        f"Histogram {_macd_h:.2f} {'nới rộng (expanding)' if abs(_macd-_macd_s)>abs(_prev_macd-_prev_macd_s) else 'thu hẹp (contracting)'}"),
        (_icon(_s['bb']),          f"BB (Bollinger Bands) {_bb_pct:.0f}% {'(gần Lower — hỗ trợ)' if _bb_pct < 20 else '(gần Upper — kháng cự)' if _bb_pct > 80 else ''}"),
        (_icon(_s['obv']),         f"OBV (Dòng tiền) {'>' if _obv > _obv_ema else '<'} EMA20"),
        (_icon(_s['vol']),         f"Volume (Khối lượng) {_vol_ratio:.1f}x TB20"),
        (_icon(_s['trend5']),      f"Trend 5 phiên (Xu hướng ngắn hạn): {_trend5:+.1f} điểm"),
    ]
    if _net_val is not None:
        rows.append(("🟢" if _net_val > 0 else ("🔴" if _net_val < -100 else "🟡"),
                     f"NN {'mua' if _net_val > 0 else 'bán'} ròng {abs(_net_val):.0f} tỷ"))
    rows.append(("🟢" if _wyckoff_total > 0 else ("🔴" if _wyckoff_total < 0 else "🟡"),
                 f"Wyckoff: {'Tích lũy/Markup' if _wyckoff_total > 0 else 'Phân phối/Markdown' if _wyckoff_total < 0 else 'Ranging'} ({_wyckoff_total:+d})"))

    if _total >= 6: _verdict, _vc = "✅ MUA — Tín hiệu mạnh", "success"
    elif _total >= 3: _verdict, _vc = "✅ Nghiêng về MUA", "success"
    elif _total >= 0: _verdict, _vc = "🟡 Trung lập — Chờ xác nhận", "warning"
    elif _total >= -3: _verdict, _vc = "⚠️ Nghiêng về BÁN", "error"
    else: _verdict, _vc = "🔴 TRÁNH — Tín hiệu tiêu cực", "error"

    with st.expander(f"{'✅' if _vc=='success' else '⚠️' if _vc=='error' else '🟡'} VNINDEX {_verdict} (điểm: {_total:+d})", expanded=False):
        for ic, desc in rows:
            st.markdown(f"{ic} {desc}")
