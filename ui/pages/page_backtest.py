# -*- coding: utf-8 -*-
"""
Backtest page — Chạy backtest manual với model hiện tại.

User nhập: vốn ban đầu + khoảng ngày.
System: dùng RL policy hiện tại chạy trên 70 symbols đã train.
Kết quả: tổng return, WR, danh sách trades.
"""

import json
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch

from config.preferences import load_preferences_from_file


def _get_symbols() -> List[str]:
    """Lấy danh sách symbols từ preferences, loại trừ các chỉ số thị trường."""
    # Các chỉ số thị trường — không phải cổ phiếu, không trade được
    INDEX_SYMBOLS = {"VNINDEX", "VN30", "HNX", "UPCOM", "HNX30"}
    
    prefs = load_preferences_from_file()
    if prefs and prefs.get("selected_symbols"):
        # Loại trừ các chỉ số thị trường khỏi danh sách trade
        return [s for s in prefs["selected_symbols"] if s not in INDEX_SYMBOLS]
    # Fallback: tất cả CSV, loại trừ index
    csv_files = list(Path("data").glob("*.csv"))
    return sorted([f.stem for f in csv_files if f.stem.isalpha() and f.stem not in INDEX_SYMBOLS])


def _run_backtest(
    initial_capital: float,
    start_date: date,
    end_date: date,
    symbols: List[str],
) -> dict:
    """
    Chạy backtest dùng RL policy — 1 portfolio chung cho tất cả symbols.

    Mỗi ngày: scan tất cả mã → policy quyết định mua/bán → tiền lấy từ 1 ví.
    
    QUAN TRỌNG: 
    1. Load cả TCN backbone (giống training) để policy encode features đúng cách
    2. Normalize per-period (giống training normalize per-slice)
    """
    from engine.market_state import INDICATOR_COLUMNS, OHLCV_COLUMNS, NUM_INDICATORS
    from engine.wf_trainer.config import WFConfig
    from engine.wf_trainer.rl_agent import PolicyNetwork
    from engine.wf_trainer.rl_env import PRICE_SCALE, LOT_SIZE, SETTLEMENT_DAYS
    from engine.evaluation_model import StockEvalNet
    from engine.config import ModelConfig

    config = WFConfig()
    model_config = ModelConfig()

    # === Load TCN backbone (giống training) ===
    tcn_path = Path(config.checkpoint_dir) / "stock_eval_net.pt"
    tcn_model = None
    if tcn_path.exists():
        try:
            tcn_model = StockEvalNet(model_config)
            cp = torch.load(tcn_path, map_location="cpu", weights_only=False)
            if isinstance(cp, dict) and "model_state_dict" in cp:
                tcn_model.load_state_dict(cp["model_state_dict"])
            elif isinstance(cp, dict) and "state_dict" in cp:
                tcn_model.load_state_dict(cp["state_dict"])
            else:
                tcn_model.load_state_dict(cp)
            tcn_model.eval()
        except Exception as e:
            tcn_model = None
            # Không fail, sẽ dùng fallback

    # === Load RL policy ===
    policy_path = Path(config.checkpoint_dir) / "rl_policy.pt"
    if not policy_path.exists():
        return {"error": "Chưa có model. Hãy train trước."}

    state_dim = config.num_features + 3
    policy = PolicyNetwork(state_dim=state_dim)
    
    # Gắn TCN backbone VÀO policy (QUAN TRỌNG - giống training)
    if tcn_model is not None:
        policy.set_backbone(tcn_model, lookback=config.lookback)
    
    try:
        saved_state = torch.load(policy_path, map_location="cpu", weights_only=True)
        current_state = policy.state_dict()
        compatible_keys = {k: v for k, v in saved_state.items()
                          if k in current_state and current_state[k].shape == v.shape}
        if compatible_keys:
            current_state.update(compatible_keys)
            policy.load_state_dict(current_state)
    except Exception as e:
        return {"error": f"Lỗi load model: {e}"}

    policy.eval()

    # === Load và chuẩn bị data cho tất cả symbols ===
    symbol_data = {}  # symbol -> {features_norm, close_prices, dates, date_to_idx}

    for symbol in symbols:
        csv_path = Path("data") / f"{symbol}.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
            df["time"] = pd.to_datetime(df["time"])
        except Exception:
            continue

        # === Filter trước, rồi normalize CHỈ trên period đang backtest ===
        # (Giống cách training normalize trên từng slice riêng)
        mask = (df["time"].dt.date >= start_date) & (df["time"].dt.date <= end_date)
        df_period = df[mask].reset_index(drop=True)
        
        if len(df_period) < 10:
            continue

        # Extract features từ period đã filter
        ohlcv = df_period[OHLCV_COLUMNS].values.astype(np.float64)
        indicators = np.zeros((len(df_period), NUM_INDICATORS), dtype=np.float64)
        for i, col in enumerate(INDICATOR_COLUMNS):
            if col in df_period.columns:
                indicators[:, i] = df_period[col].values.astype(np.float64)

        features = np.concatenate([ohlcv, indicators], axis=1)
        if features.shape[1] < config.num_features:
            pad = np.zeros((len(df_period), config.num_features - features.shape[1]))
            features = np.concatenate([features, pad], axis=1)
        elif features.shape[1] > config.num_features:
            features = features[:, :config.num_features]

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize CHỈ trên period này (giống training làm với mỗi slice)
        col_min = features.min(axis=0)
        col_max = features.max(axis=0)
        col_range = col_max - col_min
        col_range[col_range == 0] = 1.0
        features_norm = ((features - col_min) / col_range).astype(np.float32)

        close_prices = df_period["close"].values.astype(np.float64)
        dates_arr = df_period["time"].dt.date.values

        # Map date → index trong period
        date_to_idx = {d: i for i, d in enumerate(dates_arr)}

        symbol_data[symbol] = {
            "features_norm": features_norm,
            "close_prices": close_prices,
            "dates": dates_arr,
            "date_to_idx": date_to_idx,
        }

    if not symbol_data:
        return {"error": "Không có data cho khoảng thời gian này."}

    # === Chạy backtest theo ngày — 1 portfolio chung ===
    # Tìm tất cả trading days
    all_dates = sorted(set(d for sd in symbol_data.values() for d in sd["dates"]))

    cash = initial_capital
    pending_cash = []  # [(available_date, amount)] — tiền bán chờ T+2
    holdings = {}  # symbol -> {shares, avg_price, buy_date, buy_step_date}
    all_trades = []
    total_pnl = 0.0
    buy_order_counter = 0  # Đếm thứ tự mua

    for current_date in all_dates:
        # Cộng tiền bán đã đủ T+2
        available_now = [p for p in pending_cash if p[0] <= current_date]
        for _, amount in available_now:
            cash += amount
        pending_cash = [p for p in pending_cash if p[0] > current_date]

        # === Bước 1: Xử lý SELL trước (giải phóng vốn) ===
        for symbol, sd in symbol_data.items():
            if current_date not in sd["date_to_idx"]:
                continue
            if symbol not in holdings or holdings[symbol]["shares"] <= 0:
                continue

            idx = sd["date_to_idx"][current_date]
            current_price = sd["close_prices"][idx]
            current_price_vnd = current_price * PRICE_SCALE

            # Observation
            market_features = sd["features_norm"][idx]
            position_value = holdings[symbol]["shares"] * current_price_vnd
            total_value = cash + sum(
                h["shares"] * symbol_data[s]["close_prices"][symbol_data[s]["date_to_idx"].get(current_date, -1)] * PRICE_SCALE
                for s, h in holdings.items()
                if current_date in symbol_data[s]["date_to_idx"]
            )
            total_value = max(total_value, 1.0)
            cash_ratio = cash / total_value
            position_ratio = position_value / total_value
            unrealized_pnl = (current_price_vnd - holdings[symbol]["avg_price"]) / max(holdings[symbol]["avg_price"], 1.0)

            portfolio_state = np.array([cash_ratio, position_ratio, unrealized_pnl], dtype=np.float32)
            obs = np.concatenate([market_features, portfolio_state])
            action, _, _ = policy.select_action(obs)

            if action in (4, 5):
                h = holdings[symbol]
                days_held = (current_date - h["buy_step_date"]).days if h.get("buy_step_date") else 999
                if days_held >= SETTLEMENT_DAYS:
                    pct = {4: 0.50, 5: 1.0}[action]
                    shares_to_sell = int(h["shares"] * pct)
                    shares_to_sell = (shares_to_sell // LOT_SIZE) * LOT_SIZE

                    if shares_to_sell >= LOT_SIZE:
                        revenue = shares_to_sell * current_price_vnd
                        fee = revenue * config.rl_transaction_cost
                        net_revenue = revenue - fee

                        # Tiền bán về sau T+2
                        current_idx_in_dates = all_dates.index(current_date)
                        settle_idx = min(current_idx_in_dates + 2, len(all_dates) - 1)
                        settle_date = all_dates[settle_idx]
                        pending_cash.append((settle_date, net_revenue))

                        pnl = (current_price_vnd - h["avg_price"]) * shares_to_sell
                        pnl_pct = (current_price_vnd - h["avg_price"]) / max(h["avg_price"], 1.0)
                        holding_days = (current_date - h["buy_date"]).days

                        # Số lượng còn lại sau khi bán
                        shares_remaining = h["shares"] - shares_to_sell
                        
                        trade = {
                            "Mã": symbol,
                            "Ngày mua": str(h["buy_date"]),
                            "Giá mua": f"{h['avg_price']:,.0f}₫",
                            "SL mua": h.get("total_shares_bought", h["shares"]),
                            "Tổng mua": f"{h.get('total_cost', h['shares'] * h['avg_price']):,.0f}₫",
                            "Confidence": f"{h.get('confidence', 0):.1f}%",
                            "Cash lúc mua": f"{h.get('cash_before', 0):,.0f}₫",
                            "Ngày bán": str(current_date),
                            "Giá bán": f"{current_price_vnd:,.0f}₫",
                            "SL bán": shares_to_sell,
                            "Còn giữ": shares_remaining,
                            "Số ngày giữ": holding_days,
                            "Lãi/Lỗ": f"{pnl:,.0f}₫",
                            "% Lãi/Lỗ": f"{pnl_pct*100:.1f}%",
                            "Kết quả": "✅ Lãi" if pnl > 0 else "❌ Lỗ",
                            "_buy_order": h.get("buy_order", 0),  # Ẩn, dùng để sort
                        }
                        all_trades.append(trade)
                        total_pnl += pnl

                        h["shares"] -= shares_to_sell
                        if h["shares"] <= 0:
                            del holdings[symbol]
                        else:
                            holdings[symbol] = h

        # === Bước 2: Tính BUY score cho tất cả symbols, chọn top mã tốt nhất ===
        buy_candidates = []  # [(symbol, buy_prob)]

        for symbol, sd in symbol_data.items():
            if current_date not in sd["date_to_idx"]:
                continue
            # Không mua thêm mã đang hold (tránh dồn vào 1 mã)
            if symbol in holdings:
                continue

            idx = sd["date_to_idx"][current_date]
            current_price = sd["close_prices"][idx]
            current_price_vnd = current_price * PRICE_SCALE

            market_features = sd["features_norm"][idx]

            total_value = max(cash, 1.0)
            cash_ratio = 1.0
            position_ratio = 0.0
            unrealized_pnl = 0.0

            portfolio_state = np.array([cash_ratio, position_ratio, unrealized_pnl], dtype=np.float32)
            obs = np.concatenate([market_features, portfolio_state])

            # Lấy probabilities
            state_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                probs, _ = policy(state_t)
            probs_np = probs.squeeze(0).numpy()

            # Tổng prob cho BUY actions (1,2,3)
            buy_prob = float(probs_np[1] + probs_np[2] + probs_np[3])
            buy_candidates.append((symbol, buy_prob))

        # Sắp xếp theo confidence giảm dần
        buy_candidates.sort(key=lambda x: x[1], reverse=True)

        # Mua top 3 mã tốt nhất — dùng ~100% cash
        if buy_candidates and cash > 100_000:
            # Chọn tối đa 3 mã, chia đều cash
            top_n = min(3, len(buy_candidates))
            picks_to_buy = buy_candidates[:top_n]

            for i, (symbol, buy_prob) in enumerate(picks_to_buy):
                if cash <= 100_000:
                    break

                sd = symbol_data[symbol]
                idx = sd["date_to_idx"][current_date]
                current_price_vnd = sd["close_prices"][idx] * PRICE_SCALE

                # Chia đều cash cho số mã còn lại
                picks_left = top_n - i
                budget_for_this = cash / picks_left

                max_shares = int(budget_for_this // current_price_vnd)
                shares_to_buy = (max_shares // LOT_SIZE) * LOT_SIZE

                if shares_to_buy >= LOT_SIZE:
                    cost_amount = shares_to_buy * current_price_vnd
                    fee = cost_amount * config.rl_transaction_cost
                    total_cost = cost_amount + fee

                    if total_cost <= cash:
                        cash_before_buy = cash
                        cash -= total_cost
                        cash_after_buy = cash

                        if symbol in holdings:
                            old = holdings[symbol]
                            total_shares = old["shares"] + shares_to_buy
                            holdings[symbol] = {
                                "shares": total_shares,
                                "avg_price": (old["avg_price"] * old["shares"] + current_price_vnd * shares_to_buy) / total_shares,
                                "buy_date": old["buy_date"],
                                "buy_step_date": current_date,
                                "confidence": old.get("confidence", buy_prob * 100),
                                "cash_before": old.get("cash_before", cash_before_buy),
                                "cash_after": cash_after_buy,
                                "buy_order": old.get("buy_order", buy_order_counter),
                                "total_shares_bought": old.get("total_shares_bought", old["shares"]) + shares_to_buy,
                                "total_cost": old.get("total_cost", old["shares"] * old["avg_price"]) + total_cost,
                            }
                        else:
                            buy_order_counter += 1
                            holdings[symbol] = {
                                "shares": shares_to_buy,
                                "avg_price": current_price_vnd,
                                "buy_date": current_date,
                                "buy_step_date": current_date,
                                "confidence": buy_prob * 100,
                                "cash_before": cash_before_buy,
                                "cash_after": cash_after_buy,
                                "buy_order": buy_order_counter,
                                "total_shares_bought": shares_to_buy,
                                "total_cost": total_cost,
                            }

    # Tính metrics
    num_trades = len(all_trades)
    winning = sum(1 for t in all_trades if "✅" in t["Kết quả"])
    win_rate = winning / max(num_trades, 1)
    total_return = total_pnl / initial_capital

    return {
        "total_return": total_return,
        "win_rate": win_rate,
        "num_trades": num_trades,
        "total_pnl": total_pnl,
        "trades": all_trades,
        "winning": winning,
        "losing": num_trades - winning,
    }


def page_backtest() -> None:
    """Trang Backtest."""
    st.title("📈 Backtest — Mô phỏng giao dịch")

    st.caption("Dùng model hiện tại chạy mô phỏng mua/bán trên toàn bộ symbols đã train.")

    # === Form nhập ===
    with st.form("backtest_form"):
        col1, col2 = st.columns(2)

        with col1:
            initial_capital = st.number_input(
                "💰 Vốn ban đầu (VND)",
                min_value=10_000_000,
                max_value=100_000_000_000,
                value=1_000_000_000,
                step=100_000_000,
                help="Vốn dùng để backtest",
            )
            symbols = _get_symbols()
            st.caption(f"🎯 {len(symbols)} mã: {', '.join(symbols[:10])}...")

        with col2:
            start_date = st.date_input(
                "📅 Ngày bắt đầu",
                value=date.today() - timedelta(days=180),
            )
            end_date = st.date_input(
                "📅 Ngày kết thúc",
                value=date.today(),
            )

        st.caption("🇻🇳 Luật TTCK VN: T+3, ±7%, lô 100 cổ phiếu, phí 0.15%")

        submitted = st.form_submit_button("🚀 Chạy Backtest", type="primary", use_container_width=True)

    if submitted:
        if start_date >= end_date:
            st.error("❌ Ngày bắt đầu phải trước ngày kết thúc!")
            return

        with st.spinner(f"Đang chạy backtest {len(symbols)} mã từ {start_date} → {end_date}..."):
            result = _run_backtest(initial_capital, start_date, end_date, symbols)

        if "error" in result:
            st.error(f"❌ {result['error']}")
            return

        # === Kết quả ===
        st.divider()
        st.subheader("📊 Kết quả Backtest")

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tổng Return", f"{result['total_return']*100:.1f}%")
        m2.metric("Win Rate", f"{result['win_rate']*100:.1f}%")
        m3.metric("Tổng Trades", f"{result['num_trades']}")
        m4.metric("Lãi/Lỗ ròng", f"{result['total_pnl']:,.0f}₫")

        # Chi tiết thắng/thua
        c1, c2 = st.columns(2)
        c1.metric("✅ Trades thắng", result['winning'])
        c2.metric("❌ Trades thua", result['losing'])

        # Bảng trades
        if result["trades"]:
            st.divider()
            st.subheader("📋 Chi tiết Trades")
            df_trades = pd.DataFrame(result["trades"])
            # Sort theo thứ tự mua, sau đó ẩn cột _buy_order
            if "_buy_order" in df_trades.columns:
                df_trades = df_trades.sort_values("_buy_order").drop(columns=["_buy_order"])
            st.dataframe(df_trades, use_container_width=True, hide_index=True)
        else:
            st.warning("Không có trades nào trong khoảng thời gian này.")


# Entry point
page_backtest()
