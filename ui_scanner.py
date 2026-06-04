import streamlit as st
import pandas as pd
import os


def show_scanner(BASE_DIR):
    st.subheader("🔍 Bộ lọc cổ phiếu — Cạn cung / Sắp bùng nổ")
    st.caption("Quét tất cả file CSV có sẵn, tìm cổ phiếu đang tích lũy xong theo Wyckoff.")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        min_score = st.slider("Điểm tối thiểu", 3, 10, 5)
    with fc2:
        top_n = st.selectbox("Hiển thị top", [10, 20, 30, 50], index=1)
    with fc3:
        exclude_index = st.checkbox("Bỏ chỉ số (VNINDEX, VN30...)", value=True)

    if not st.button("🔍 Quét ngay", type="primary", key="btn_scan"):
        return

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
            _d = pd.read_csv(os.path.join(BASE_DIR, fname))
            _d['time'] = pd.to_datetime(_d['time'])
            _d = _d.sort_values('time').reset_index(drop=True)
            if len(_d) < 60: continue

            _last = _d.iloc[-1]
            _prev = _d.iloc[-2]
            _d20  = _d.tail(20)
            _d60  = _d.tail(60)

            close     = _last['close']
            high      = _last['high']
            low       = _last['low']
            vol       = _last['volume']
            vol_avg20 = _d['volume'].tail(20).mean()
            vol_avg5  = _d['volume'].tail(5).mean()
            ema20     = _last['EMA_20']
            ema50     = _last['EMA_50']
            rsi       = _last['RSI_14']
            macd_h    = _last['MACD_hist']
            prev_h    = _prev['MACD_hist']
            atr       = _last['ATR_14']
            atr_5ago  = _d.iloc[-6]['ATR_14'] if len(_d) > 6 else atr
            obv       = _last['OBV']
            obv_ema   = _d['OBV'].ewm(span=20).mean().iloc[-1]
            obv_5     = _d['OBV'].iloc[-1] - _d['OBV'].iloc[-5]

            high_60  = _d60['high'].max()
            low_60   = _d60['low'].min()
            high_20  = _d20['high'].max()
            low_20   = _d20['low'].min()
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

            if vol_avg5 < vol_avg20 * 0.6 and close >= _prev['close']:
                score += 2; tags.append("🟢 No Supply (Không có cung)")
            if pd.notna(atr) and pd.notna(atr_5ago) and atr < atr_5ago * 0.8:
                score += 1; tags.append("🟢 ATR thu hẹp (Volatility giảm)")
            if spread < spread_avg * 0.7 and close_pos > 0.6:
                score += 1; tags.append("🟢 Nến hẹp đóng cao (Narrow bar, close high)")
            if obv_rising and obv_5 > 0:
                score += 2; tags.append("🟢 OBV tăng (Dòng tiền vào)")
            if close > ema20 and ema20 > ema50:
                score += 1; tags.append("🟢 EMA tăng (Uptrend)")
            if low < low_20 * 0.995 and close > low_20 * 0.995 and vol_ratio > 1.0:
                score += 3; tags.append("🟢 Spring (Bẫy giảm)")
            if pd.notna(macd_h) and pd.notna(prev_h) and macd_h > 0 and prev_h <= 0:
                score += 2; tags.append("🟢 MACD đảo chiều (Bullish crossover)")
            if pd.notna(rsi) and 40 <= rsi <= 60:
                score += 1; tags.append(f"🟢 RSI trung tính (Neutral) {rsi:.0f}")
            if close >= high_20 * 0.97 and close < high_20 * 1.01:
                score += 1; tags.append("🟢 Gần breakout (Phá vỡ kháng cự)")
            if near_low and obv_rising and vol_trend_up:
                score += 2; tags.append("🔵 Wyckoff Accum (Tích lũy)")
            elif not near_low and not near_high and price_trend20 > 0 and obv_rising:
                score += 1; tags.append("🟢 Wyckoff Markup (Tăng giá)")

            # Tru diem xau
            if close < ema50: score -= 1
            if vol_ratio > 2.0 and close_pos < 0.3: score -= 2
            if near_high and not obv_rising: score -= 1

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
