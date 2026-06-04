import streamlit as st
import pandas as pd
import os
from datetime import date, timedelta


def show_update(BASE_DIR):
    st.subheader("📥 Cập nhật dữ liệu CSV")
    st.caption("Tải dữ liệu OHLCV + tính toán 50+ chỉ báo kỹ thuật, lưu vào file CSV.")

    u1, u2, u3, u4 = st.columns([2, 2, 2, 1])
    with u1:
        update_symbol = st.text_input("Mã cổ phiếu", value="VIX", key="update_sym").upper()
    with u2:
        update_start = st.date_input("Từ ngày", value=date.today() - timedelta(days=365), key="update_start")
    with u3:
        update_end = st.date_input("Đến ngày", value=date.today(), key="update_end")
    with u4:
        st.write("")
        st.write("")
        run_update = st.button("📥 Cập nhật", type="primary", key="btn_update")

    csv_files = [f for f in os.listdir(BASE_DIR) if f.endswith('.csv') and not f.startswith('.')]
    if csv_files:
        st.caption(f"📂 File CSV hiện có: {', '.join(sorted(csv_files))}")

    if not run_update:
        return

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
                return

            df_new.columns = [c.lower() for c in df_new.columns]
            df_new['time'] = pd.to_datetime(df_new['time'])
            base_cols = ['time','open','high','low','close','volume']

            if os.path.exists(csv_out):
                st.write("📂 Ghép với dữ liệu cũ...")
                df_old = pd.read_csv(csv_out)
                df_old['time'] = pd.to_datetime(df_old['time'])
                df = pd.concat([df_old[base_cols], df_new[base_cols]], ignore_index=True)
            else:
                df = df_new[base_cols].copy()

            df = df.drop_duplicates('time').sort_values('time').reset_index(drop=True)
            st.write("🔢 Tính toán 50+ chỉ báo kỹ thuật...")
            df = add_indicators(df)
            df.round(4).to_csv(csv_out, index=False)

            status.update(label=f"✅ Cập nhật xong {update_symbol}", state="complete")
            st.success(f"✅ **{update_symbol}**: {len(df)} phiên | {df.time.iloc[0].date()} → {df.time.iloc[-1].date()} | {len(df.columns)} cột")
            st.caption(f"📂 Lưu tại: {csv_out}")

        except Exception as e:
            status.update(label=f"❌ Lỗi: {e}", state="error")
            st.error(f"Lỗi: {e}")
