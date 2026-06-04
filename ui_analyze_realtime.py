import streamlit as st


def show_realtime(symbol_input):
    try:
        from vnstock import Trading, Finance
        pb = Trading(symbol=symbol_input, source='VCI').price_board()
        pb.columns = ['_'.join(c) for c in pb.columns]
        row = pb.iloc[0]

        ref      = row.get('listing_ref_price') or 0
        price_rt = row.get('match_match_price') or 0
        pct_rt   = (price_rt - ref) / ref * 100 if ref else 0
        acc_vol  = row.get('match_accumulated_volume') or 0
        acc_val  = row.get('match_accumulated_value') or 0

        if price_rt == 0:
            st.info("📅 Thị trường đóng cửa hoặc chưa có dữ liệu realtime.")
            return

        st.markdown("**📊 Dữ liệu phiên hôm nay:**")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Giá khớp", f"{price_rt:,.0f}", delta=f"{pct_rt:+.2f}%", delta_color="normal")
        r2.metric("Cao / Thấp", f"{row.get('match_highest') or 0:,.0f} / {row.get('match_lowest') or 0:,.0f}")
        r3.metric("KL khớp", f"{acc_vol/1e6:.1f}M CP")
        r4.metric("GT khớp", f"{acc_val/1e9:.0f} tỷ")

        st.markdown("**📊 Cung cầu (3 bước giá):**")
        col_b, col_a = st.columns(2)
        with col_b:
            st.markdown("**🟢 Bên mua (Cầu)**")
            for i in [1, 2, 3]:
                bp = row.get(f'bid_ask_bid_{i}_price') or 0
                bv = row.get(f'bid_ask_bid_{i}_volume') or 0
                if bp:
                    st.markdown(f"`{bp:,.0f}` &nbsp; {bv:,.0f} CP &nbsp; {'█'*min(int(bv/50000),20)}")
        with col_a:
            st.markdown("**🔴 Bên bán (Cung)**")
            for i in [1, 2, 3]:
                ap = row.get(f'bid_ask_ask_{i}_price') or 0
                av = row.get(f'bid_ask_ask_{i}_volume') or 0
                if ap:
                    st.markdown(f"`{ap:,.0f}` &nbsp; {av:,.0f} CP &nbsp; {'█'*min(int(av/50000),20)}")

        total_bid = sum(row.get(f'bid_ask_bid_{i}_volume') or 0 for i in [1, 2, 3])
        total_ask = sum(row.get(f'bid_ask_ask_{i}_volume') or 0 for i in [1, 2, 3])
        if total_bid + total_ask > 0:
            bid_pct = total_bid / (total_bid + total_ask) * 100
            if bid_pct > 65:   st.success(f"🟢 Cầu mạnh hơn cung ({bid_pct:.0f}% / {100-bid_pct:.0f}%)")
            elif bid_pct < 35: st.error(f"🔴 Cung mạnh hơn cầu ({bid_pct:.0f}% / {100-bid_pct:.0f}%)")
            else:              st.info(f"🟡 Cung cầu cân bằng ({bid_pct:.0f}% / {100-bid_pct:.0f}%)")

        st.markdown("**🌐 Khối ngoại:**")
        fb_val  = (row.get('match_foreign_buy_value') or 0) / 1e9
        fs_val  = (row.get('match_foreign_sell_value') or 0) / 1e9
        net_val = fb_val - fs_val
        kn1, kn2, kn3, kn4 = st.columns(4)
        kn1.metric("Mua ròng KL", f"{(row.get('match_foreign_buy_volume') or 0)-(row.get('match_foreign_sell_volume') or 0):+,.0f} CP")
        kn2.metric("Mua ròng GT", f"{net_val:+.1f} tỷ")
        kn3.metric("NN Mua", f"{row.get('match_foreign_buy_volume') or 0:,.0f} CP")
        kn4.metric("NN Bán", f"{row.get('match_foreign_sell_volume') or 0:,.0f} CP")
        if net_val > 0:    st.success(f"✅ Khối ngoại **mua ròng** {net_val:+.1f} tỷ")
        elif net_val < -1: st.error(f"⚠️ Khối ngoại **bán ròng** {net_val:.1f} tỷ")

        try:
            f_api    = Finance(symbol=symbol_input, source='VCI')
            df_ratio = f_api.ratio(period='quarter', lang='en')
            if not df_ratio.empty:
                pe_row   = df_ratio[df_ratio['item_en'].str.contains('P/E', na=False)]
                pb_row   = df_ratio[df_ratio['item_en'].str.contains('P/B', na=False)]
                val_cols = [c for c in df_ratio.columns if c not in ['item', 'item_en', 'item_id']]
                if val_cols and (not pe_row.empty or not pb_row.empty):
                    st.markdown("**💰 Định giá:**")
                    d1, d2 = st.columns(2)
                    if not pe_row.empty: d1.metric("P/E", f"{pe_row[val_cols[-1]].values[0]:.1f}x")
                    if not pb_row.empty: d2.metric("P/B", f"{pb_row[val_cols[-1]].values[0]:.1f}x")
        except Exception:
            pass
    except Exception as e:
        st.caption(f"Không tải được dữ liệu realtime: {e}")
