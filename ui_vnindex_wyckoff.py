import streamlit as st
import pandas as pd
import ta


def calc_vnindex_wyckoff(df_vn, last, prev, df_20, df_60):
    close   = last['close']
    high    = last['high']
    low     = last['low']
    vol     = last['volume']
    vol_avg = df_vn['volume'].tail(20).mean()
    ema20   = last['EMA_20']
    ema50   = last['EMA_50']
    ema200  = last['EMA_200']
    obv     = last['OBV']
    obv_ema = ta.trend.ema_indicator(df_vn['OBV'], window=20).iloc[-1]
    bb_u    = last['BB_upper']
    bb_l    = last['BB_lower']

    high_60  = df_60['high'].max()
    low_60   = df_60['low'].min()
    high_20  = df_20['high'].max()
    low_20   = df_20['low'].min()
    range_60 = high_60 - low_60

    vol_ratio     = vol / vol_avg
    vol_trend_up  = df_vn['volume'].tail(5).mean() > vol_avg
    vol_5_avg     = df_vn['volume'].tail(5).mean()
    vol_declining = vol_5_avg < vol_avg * 0.8
    price_trend_20 = df_vn['close'].iloc[-1] - df_vn['close'].iloc[-20]

    spread      = high - low
    spread_avg  = (df_20['high'] - df_20['low']).mean()
    wide_spread   = spread > spread_avg * 1.3
    narrow_spread = spread < spread_avg * 0.7
    close_pos   = (close - low) / spread if spread > 0 else 0.5
    close_upper = close_pos > 0.6
    close_lower = close_pos < 0.4

    obv_rising = obv > obv_ema
    obv_5      = df_vn['OBV'].iloc[-1] - df_vn['OBV'].iloc[-5]

    price_recovering = close > df_vn['close'].iloc[-3]
    big_vol_recently = df_vn['volume'].tail(5).max() > vol_avg * 2

    near_low  = close <= low_60 + range_60 * 0.25
    near_high = close >= high_60 - range_60 * 0.25
    in_range  = not near_low and not near_high

    spring   = low < low_20 * 0.995 and close > low_20 * 0.995 and vol_ratio < 1.2
    upthrust = high > high_20 * 1.005 and close < high_20 * 1.005 and vol_ratio > 1.0

    return dict(
        close=close, high=high, low=low, vol=vol, vol_avg=vol_avg,
        ema20=ema20, ema50=ema50, ema200=ema200, obv=obv, obv_ema=obv_ema,
        bb_u=bb_u, bb_l=bb_l, high_60=high_60, low_60=low_60, range_60=range_60,
        vol_ratio=vol_ratio, vol_trend_up=vol_trend_up, vol_declining=vol_declining,
        price_trend_20=price_trend_20, spread=spread, spread_avg=spread_avg,
        wide_spread=wide_spread, narrow_spread=narrow_spread,
        close_pos=close_pos, close_upper=close_upper, close_lower=close_lower,
        obv_rising=obv_rising, obv_5=obv_5,
        price_recovering=price_recovering, big_vol_recently=big_vol_recently,
        near_low=near_low, near_high=near_high, in_range=in_range,
        spring=spring, upthrust=upthrust,
    )


def show_wyckoff_signals(c, df_vn):
    signals = []
    def wadd(icon, event, detail):
        signals.append((icon, event, detail))

    # 1. Price structure
    if c['close'] > c['ema200']:
        wadd("🟢", "Cấu trúc giá", f"Giá ({c['close']:,.0f}) trên EMA200 ({c['ema200']:,.0f}) — xu hướng dài hạn tăng")
        struct_score = 1
    elif c['close'] > c['ema50']:
        wadd("🟡", "Cấu trúc giá", f"Giá trên EMA50 ({c['ema50']:,.0f}) nhưng dưới EMA200 — trung hạn tích cực")
        struct_score = 0
    else:
        wadd("🔴", "Cấu trúc giá", f"Giá ({c['close']:,.0f}) dưới EMA50 ({c['ema50']:,.0f}) và EMA200 ({c['ema200']:,.0f}) — xu hướng yếu")
        struct_score = -1

    # 2. VSA
    if c['wide_spread'] and c['close_upper'] and c['vol_ratio'] > 1.2:
        wadd("🟢", "Demand (Lực cầu) mạnh — Nỗ lực↑ Kết quả↑", f"Nến rộng, đóng cửa cao, khối lượng {c['vol_ratio']:.1f}x TB — lực cầu áp đảo")
        vsa_score = 2
    elif c['wide_spread'] and c['close_lower'] and c['vol_ratio'] > 1.2:
        wadd("🔴", "Supply (Lực cung) mạnh — Nỗ lực↑ Kết quả↓", f"Nến rộng, đóng cửa thấp, khối lượng {c['vol_ratio']:.1f}x TB — lực cung áp đảo")
        vsa_score = -2
    elif c['narrow_spread'] and c['vol_ratio'] > 1.3:
        wadd("🟡", "No Result (Không có kết quả) — Nỗ lực↑ Kết quả↓", f"Khối lượng cao {c['vol_ratio']:.1f}x nhưng biên độ hẹp — lực cản mạnh")
        vsa_score = -1
    elif c['narrow_spread'] and c['vol_ratio'] < 0.7 and c['close_pos'] > 0.5:
        wadd("🟢", "No Supply (Không có cung) — cung cạn kiệt", f"Khối lượng thấp {c['vol_ratio']:.1f}x, biên độ hẹp, đóng cửa cao — không có lực bán")
        vsa_score = 1
    elif c['narrow_spread'] and c['vol_ratio'] < 0.7:
        wadd("🟡", "No Demand / No Supply (Không cầu / Không cung)", f"Khối lượng thấp {c['vol_ratio']:.1f}x, biên độ hẹp — thiếu động lực")
        vsa_score = 0
    else:
        wadd("🟡", "Khối lượng bình thường", f"Khối lượng {c['vol_ratio']:.1f}x TB20, biên độ nến bình thường")
        vsa_score = 0

    # 3. OBV
    if c['obv_rising'] and c['obv_5'] > 0:
        wadd("🟢", "OBV (Dòng tiền tích lũy) — Accumulation (Tích lũy)", "OBV tăng, dòng tiền thực sự đang vào — tiền lớn đang mua")
        obv_score = 1
    elif not c['obv_rising'] and c['obv_5'] < 0:
        wadd("🔴", "OBV (Dòng tiền phân phối) — Distribution (Phân phối)", "OBV giảm, dòng tiền rút ra — tiền lớn đang bán")
        obv_score = -1
    else:
        wadd("🟡", "OBV (Dòng tiền) trung lập", "OBV chưa xác nhận rõ xu hướng")
        obv_score = 0

    # 4. Phase
    if c['near_low'] and c['obv_rising'] and c['vol_trend_up']:
        phase, phase_detail, phase_color, phase_score = "🔵 Phase C/D (Giai đoạn C/D) — Accumulation (Tích lũy)", "Giá vùng đáy 60 phiên, OBV tăng, khối lượng tăng dần — tiền lớn đang gom hàng", "info", 2
    elif c['near_low'] and c['vol_ratio'] > 2.5 and c['spread'] > c['spread_avg'] * 1.5 and c['close_pos'] > 0.3:
        phase, phase_detail, phase_color, phase_score = "🟡 Selling Climax (SC) — Bán tháo đỉnh điểm, cung đang bị hấp thụ", "Volume đột biến cực lớn, giá giảm mạnh nhưng đóng cửa phục hồi — cung đang bị hấp thụ. SC chỉ xác nhận được khi phiên kế tiếp tăng mạnh.", "warning", 1
    elif c['near_low'] and not c['obv_rising']:
        phase, phase_detail, phase_color, phase_score = "🔴 Phase A/B (Giai đoạn A/B) — Markdown / Selling Climax (Bán tháo đỉnh điểm)", "Giá vùng đáy nhưng OBV chưa xác nhận — có thể vẫn đang trong giai đoạn bán tháo", "error", -1
    elif c['near_high'] and not c['obv_rising'] and c['vol_ratio'] > 1.2:
        phase, phase_detail, phase_color, phase_score = "🟠 Phase B/C (Giai đoạn B/C) — Distribution (Phân phối)", "Giá vùng đỉnh 60 phiên, OBV giảm, khối lượng cao — tiền lớn đang xả hàng", "warning", -2
    elif c['near_high'] and c['obv_rising']:
        phase, phase_detail, phase_color, phase_score = "🟢 Phase D (Giai đoạn D) — Markup (Tăng giá)", "Giá vùng đỉnh, OBV vẫn tăng — xu hướng tăng tiếp diễn (không phải điểm mua mới)", "success", 1
    elif c['in_range'] and c['price_trend_20'] > 0 and c['obv_rising']:
        phase, phase_detail, phase_color, phase_score = "🟢 Phase D (Giai đoạn D) — Markup (Tăng giá)", "Giá tăng 20 phiên, OBV xác nhận — xu hướng tăng đang hình thành", "success", 1
    elif c['in_range'] and c['price_trend_20'] < 0 and not c['obv_rising']:
        phase, phase_detail, phase_color, phase_score = "🔴 Phase A (Giai đoạn A) — Markdown (Giảm giá)", "Giá giảm 20 phiên, OBV xác nhận — xu hướng giảm đang diễn ra", "error", -1
    else:
        phase, phase_detail, phase_color, phase_score = "🟡 Phase B (Giai đoạn B) — Ranging (Dao động ngang)", "Giá dao động trong vùng, chưa xác định được hướng — chờ breakout (phá vỡ)", "warning", 0

    # 5. Spring / Upthrust
    if c['spring']:
        wadd("🟢", "Spring (Bẫy giảm / Đảo chiều tăng)", "Giá xuyên đáy 20 phiên rồi phục hồi với khối lượng thấp — cung cạn kiệt, khả năng đảo chiều tăng")
        spring_score = 2
    elif c['upthrust']:
        wadd("🔴", "Upthrust (Bẫy tăng / Đảo chiều giảm)", "Giá xuyên đỉnh 20 phiên rồi quay đầu với khối lượng cao — cung xuất hiện, khả năng đảo chiều giảm")
        spring_score = -2
    else:
        spring_score = 0

    total_score = struct_score + vsa_score + obv_score + phase_score + spring_score
    return signals, phase, phase_detail, phase_color, total_score


def show_subphase(c, cycle, phase, df_vn):
    if "Accumulation" in cycle or "Accumulation" in phase:
        if c['near_low'] and c['vol_ratio'] > 2.0 and c['close_lower']:
            sub = ("SC (Selling Climax)",
                   "🔴 Đang xảy ra Selling Climax — volume cực lớn, giá giảm mạnh đóng thấp. "
                   "Retail hoảng loạn bán, smart money đang hấp thụ. Chưa nên mua ngay.")
        elif c['near_low'] and c['big_vol_recently'] and c['price_recovering']:
            sub = ("AR (Automatic Rally)",
                   "🟡 Đang trong Automatic Rally — giá bật sau SC do cung cạn tạm thời. "
                   "Đây là kháng cự tạm thời, chưa phải điểm mua an toàn.")
        elif c['near_low'] and c['vol_declining'] and not c['obv_rising']:
            sub = ("ST (Secondary Test)",
                   "🟡 Đang test lại vùng SC (Secondary Test) — volume thấp hơn SC là tín hiệu tốt. "
                   "Nếu giá không phá đáy SC → xác nhận Phase A kết thúc.")
        elif c['near_low'] and c['vol_declining'] and c['obv_rising'] and not c['spring']:
            sub = ("LPS (Last Point of Support)",
                   "🟢 Có thể là LPS — giá pullback nhẹ, volume thấp, OBV tăng. "
                   "Đây là cơ hội mua tốt nếu đã có SOS trước đó.")
        elif c['spring']:
            sub = ("Spring (Phase C)",
                   "🟢⭐ SPRING — Tín hiệu mạnh nhất! Giá xuyên đáy rồi phục hồi với volume thấp. "
                   "Cung đã cạn kiệt hoàn toàn. Đây là điểm mua tốt nhất trong Accumulation.")
        elif c['obv_rising'] and c['vol_trend_up'] and c['price_trend_20'] > 0:
            sub = ("SOS (Sign of Strength)",
                   "🟢 SOS — Giá breakout với volume tăng, OBV xác nhận. "
                   "Accumulation kết thúc, Markup bắt đầu. Có thể mua hoặc mua thêm.")
        elif not c['obv_rising'] and c['vol_ratio'] < 1.0:
            sub = ("Phase B (Sideway)",
                   "🟡 Phase B — Giá đang sideway, volume giảm dần. "
                   "Smart money đang âm thầm gom hàng. Chờ Spring hoặc SOS để xác nhận.")
        else:
            sub = ("Phase A/B (Chưa rõ)", "🟡 Chưa xác định rõ sub-phase. Cần quan sát thêm volume và hành vi giá.")

    elif "Distribution" in cycle or "Distribution" in phase:
        if c['near_high'] and c['vol_ratio'] > 2.0 and c['close_upper']:
            sub = ("BC (Buying Climax)",
                   "🔴 Đang xảy ra Buying Climax — volume cực lớn, giá tăng mạnh. "
                   "Retail FOMO mua đỉnh, smart money đang xả hàng. Không nên mua thêm.")
        elif c['near_high'] and c['big_vol_recently'] and c['close'] < df_vn['close'].iloc[-3]:
            sub = ("AR (Automatic Reaction)",
                   "🟡 Automatic Reaction sau BC — giá giảm do cầu cạn tạm thời. "
                   "Xác định vùng hỗ trợ dưới của trading range.")
        elif c['near_high'] and c['vol_declining'] and not c['obv_rising']:
            sub = ("ST (Secondary Test)",
                   "🟡 Secondary Test vùng BC — volume thấp hơn, giá không vượt đỉnh cũ. "
                   "Xác nhận lực cầu đã yếu đi.")
        elif c['upthrust']:
            sub = ("UTAD (Upthrust After Distribution)",
                   "🔴⭐ UTAD — Bẫy tăng cuối cùng! Giá vượt đỉnh rồi quay đầu nhanh. "
                   "Smart money xả nốt hàng cho retail FOMO. Đây là điểm bán tốt nhất.")
        elif not c['obv_rising'] and c['price_trend_20'] < 0 and c['vol_ratio'] > 1.0:
            sub = ("SOW (Sign of Weakness)",
                   "🔴 SOW — Giá giảm mạnh với volume lớn, OBV giảm. "
                   "Phân phối gần xong, Markdown sắp bắt đầu. Nên thoát hàng.")
        elif not c['obv_rising'] and c['vol_declining']:
            sub = ("LPSY (Last Point of Supply)",
                   "🔴 LPSY — Rally yếu, volume thấp, spread hẹp. "
                   "Cơ hội bán/short cuối trước khi Markdown.")
        else:
            sub = ("Phase B Distribution",
                   "🟡 Phase B Distribution — Giá sideway vùng đỉnh. "
                   "Smart money đang xả hàng từ từ. Không nên mua thêm.")

    elif "Markup" in cycle:
        if c['price_trend_20'] > 0 and c['obv_rising'] and c['vol_trend_up']:
            sub = ("Markup mạnh (Phase D)",
                   "🟢 Xu hướng tăng được xác nhận đầy đủ — giá tăng, OBV tăng, volume tăng. "
                   "Có thể giữ hoặc mua thêm tại các pullback.")
        elif c['price_trend_20'] > 0 and c['vol_declining']:
            sub = ("BU (Back Up / Pullback)",
                   "🟢 Pullback sau breakout — volume giảm khi giá điều chỉnh là tín hiệu tốt. "
                   "Đây là cơ hội mua thêm (Back Up to creek).")
        else:
            sub = ("Markup đang hình thành", "🟡 Markup chưa xác nhận hoàn toàn — cần volume tăng theo giá để xác nhận.")

    elif "Markdown" in cycle:
        if c['price_trend_20'] < 0 and not c['obv_rising'] and c['vol_ratio'] > 1.2:
            sub = ("Markdown mạnh",
                   "🔴 Xu hướng giảm được xác nhận — giá giảm, OBV giảm, volume cao. "
                   "Tránh mua, chờ dấu hiệu Selling Climax để xem xét.")
        elif c['near_low'] and c['vol_ratio'] > 1.5 and c['close_lower']:
            sub = ("Selling Climax (SC) đang hình thành",
                   "🔴 Có thể đang hình thành SC — volume đột biến, giá giảm mạnh. "
                   "Chưa nên mua, chờ giá phục hồi và volume giảm để xác nhận đáy.")
        else:
            sub = ("Markdown tiếp diễn", "🔴 Xu hướng giảm chưa có dấu hiệu dừng. Tránh bắt đáy.")
    else:
        sub = ("Chưa xác định", "🟡 Thị trường đang trong giai đoạn chuyển tiếp. Cần quan sát thêm.")

    st.markdown(f"**→ {sub[0]}**")
    st.info(sub[1])
