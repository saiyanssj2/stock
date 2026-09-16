# -*- coding: utf-8 -*-
"""
Recommendations page - Portfolio Tracker + Khuyến nghị cổ phiếu.

Hiển thị:
- Section 1: Tóm tắt portfolio (tổng tài sản, tiền mặt, giá trị CP, lãi/lỗ)
- Section 2: Bảng holdings hiện tại với nút Bán
- Section 3: Khuyến nghị AI với nút Thực hiện MUA/BÁN

References: Req 6.1, 6.3, 6.4, 5.3
"""

from datetime import date
from typing import Dict, List, Optional

import streamlit as st

from config.settings import CONFIDENCE_THRESHOLD
from engine.portfolio_storage import load_portfolio, save_portfolio
from engine.recommendation_engine import RecommendationEngine
from models.portfolio_models import PortfolioHolding, PortfolioState
from models.recommendation_models import Action, Recommendation

# Phí giao dịch 0.15%
TRANSACTION_FEE_RATE: float = 0.0015

# Mapping Action → display info (icon, màu, label)
_ACTION_DISPLAY: Dict[Action, Dict[str, str]] = {
    Action.BUY: {"icon": "🟢", "color": "green", "label": "MUA"},
    Action.HOLD: {"icon": "🟡", "color": "orange", "label": "GIỮ"},
    Action.SELL: {"icon": "🔴", "color": "red", "label": "BÁN"},
}


def _get_recommendation_engine() -> RecommendationEngine:
    """
    Lấy hoặc tạo RecommendationEngine từ session_state (singleton).

    Returns:
        RecommendationEngine instance (cached trong session)
    """
    if "recommendation_engine" not in st.session_state:
        st.session_state.recommendation_engine = RecommendationEngine()
    return st.session_state.recommendation_engine


def _get_portfolio() -> PortfolioState:
    """
    Load portfolio từ file. Luôn đọc từ file để đảm bảo persistence.

    Returns:
        PortfolioState (tạo mặc định nếu file chưa có)
    """
    state = load_portfolio()
    if state is None:
        # File corrupt → tạo mới
        state = PortfolioState(
            cash=1_000_000_000.0,
            holdings=[],
            initial_capital=1_000_000_000.0,
        )
        save_portfolio(state)
    return state


def _get_current_price(symbol: str) -> Optional[float]:
    """
    Đọc giá đóng cửa gần nhất từ CSV (đơn vị 1000 VND).

    Args:
        symbol: Mã cổ phiếu

    Returns:
        Giá (đơn vị 1000 VND), None nếu không đọc được
    """
    engine = _get_recommendation_engine()
    return engine._get_latest_close_price(symbol)


def _calculate_portfolio_stock_value(holdings: List[PortfolioHolding]) -> float:
    """
    Tính tổng giá trị cổ phiếu theo giá hiện tại (VND).

    Args:
        holdings: Danh sách vị thế

    Returns:
        Tổng giá trị (VND)
    """
    total = 0.0
    for h in holdings:
        price = _get_current_price(h.symbol)
        if price is not None:
            total += h.shares * price * 1000
        else:
            # Fallback dùng giá mua nếu không đọc được giá hiện tại
            total += h.shares * h.buy_price * 1000
    return total


def _execute_buy(portfolio: PortfolioState, symbol: str, shares: int, price: float) -> None:
    """
    Thực hiện lệnh MUA: trừ tiền mặt, thêm vào holdings, lưu file.

    Args:
        portfolio: Trạng thái portfolio hiện tại
        symbol: Mã cổ phiếu
        shares: Số CP mua (bội số 100)
        price: Giá mua (đơn vị 1000 VND)
    """
    # Tính tổng chi phí (giá * số CP * 1000 + phí)
    cost = shares * price * 1000
    fee = cost * TRANSACTION_FEE_RATE
    total_cost = cost + fee

    if total_cost > portfolio.cash:
        st.error("❌ Không đủ tiền mặt để thực hiện lệnh mua!")
        return

    # Trừ tiền mặt
    portfolio.cash -= total_cost

    # Thêm hoặc cộng dồn vào holdings
    existing = next((h for h in portfolio.holdings if h.symbol == symbol), None)
    if existing is not None:
        # Tính giá trung bình mới
        total_shares = existing.shares + shares
        existing.buy_price = (
            (existing.buy_price * existing.shares + price * shares) / total_shares
        )
        existing.shares = total_shares
        existing.buy_date = date.today()
    else:
        portfolio.holdings.append(
            PortfolioHolding(
                symbol=symbol,
                shares=shares,
                buy_price=price,
                buy_date=date.today(),
            )
        )

    save_portfolio(portfolio)


def _execute_sell(portfolio: PortfolioState, symbol: str, shares: int) -> None:
    """
    Thực hiện lệnh BÁN: cộng tiền mặt, xóa/giảm holdings, lưu file.

    Args:
        portfolio: Trạng thái portfolio hiện tại
        symbol: Mã cổ phiếu
        shares: Số CP bán
    """
    holding = next((h for h in portfolio.holdings if h.symbol == symbol), None)
    if holding is None:
        st.error(f"❌ Không tìm thấy {symbol} trong portfolio!")
        return

    if shares > holding.shares:
        shares = holding.shares

    # Lấy giá hiện tại
    current_price = _get_current_price(symbol)
    if current_price is None:
        st.error(f"❌ Không đọc được giá hiện tại cho {symbol}!")
        return

    # Cộng tiền mặt (trừ phí)
    revenue = shares * current_price * 1000
    fee = revenue * TRANSACTION_FEE_RATE
    portfolio.cash += revenue - fee

    # Giảm/xóa holding
    if shares >= holding.shares:
        portfolio.holdings.remove(holding)
    else:
        holding.shares -= shares

    save_portfolio(portfolio)


def _render_portfolio_summary(portfolio: PortfolioState) -> None:
    """
    Render Section 1: Tóm tắt portfolio.

    Hiển thị tổng tài sản, tiền mặt, giá trị CP, lãi/lỗ %.
    """
    st.subheader("💼 Tóm tắt Portfolio")

    stock_value = _calculate_portfolio_stock_value(portfolio.holdings)
    total_value = portfolio.cash + stock_value
    profit_loss = total_value - portfolio.initial_capital
    profit_loss_pct = (profit_loss / portfolio.initial_capital * 100) if portfolio.initial_capital > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tổng tài sản", f"{total_value:,.0f}₫")
    with col2:
        st.metric("Tiền mặt", f"{portfolio.cash:,.0f}₫")
    with col3:
        st.metric("Giá trị cổ phiếu", f"{stock_value:,.0f}₫")
    with col4:
        delta_color = "normal" if profit_loss >= 0 else "inverse"
        st.metric(
            "Lãi/Lỗ",
            f"{profit_loss:+,.0f}₫",
            delta=f"{profit_loss_pct:+.2f}%",
            delta_color=delta_color,
        )




def _render_holdings_table(portfolio: PortfolioState) -> None:
    """
    Render Section 2: Bảng holdings hiện tại + form nhập tài sản.

    Hiển thị: Mã | Số CP | Giá mua | Giá hiện tại | Lãi/Lỗ | Ngày mua | Nút Bán
    """
    st.subheader("📊 Cổ phiếu đang nắm giữ")

    # Form nhập tài sản (tiền mặt + vị thế) — lưu 1 lần
    with st.expander("⚙️ Cài đặt tài sản hiện có"):
        st.caption("Nhập tiền mặt và các vị thế cổ phiếu hiện có, sau đó **Lưu tất cả** một lần.")

        # --- Tiền mặt ---
        st.markdown("**💰 Tiền mặt**")
        # Khởi tạo session state cho tiền mặt
        if "_setup_cash" not in st.session_state:
            st.session_state["_setup_cash"] = portfolio.cash

        new_cash = st.number_input(
            "Tiền mặt hiện có (₫)",
            min_value=0.0,
            value=st.session_state["_setup_cash"],
            step=1_000_000.0,
            format="%.0f",
            key="setup_cash_input",
        )
        st.session_state["_setup_cash"] = new_cash

        st.divider()

        # --- Danh sách vị thế ---
        st.markdown("**📈 Vị thế cổ phiếu**")

        # Khởi tạo session state cho danh sách vị thế đang nhập
        if "_setup_holdings" not in st.session_state:
            # Load từ portfolio hiện tại
            st.session_state["_setup_holdings"] = [
                {
                    "symbol": h.symbol,
                    "shares": h.shares,
                    "price": h.buy_price * 1000,  # Chuyển về VND
                    "date": h.buy_date,
                }
                for h in portfolio.holdings
            ]

        holdings_list = st.session_state["_setup_holdings"]

        # Hiển thị các vị thế đã nhập
        to_remove = []
        for idx, h in enumerate(holdings_list):
            cols = st.columns([1.2, 1, 1.2, 1.2, 0.5])
            with cols[0]:
                h["symbol"] = st.text_input(
                    "Mã CK",
                    value=h["symbol"],
                    key=f"setup_sym_{idx}",
                    label_visibility="collapsed" if idx > 0 else "visible",
                ).strip().upper()
            with cols[1]:
                h["shares"] = st.number_input(
                    "Số CP",
                    min_value=1,
                    value=h["shares"],
                    step=1,
                    key=f"setup_shares_{idx}",
                    label_visibility="collapsed" if idx > 0 else "visible",
                )
            with cols[2]:
                h["price"] = st.number_input(
                    "Giá TB (₫)",
                    min_value=100.0,
                    value=float(h["price"]),
                    step=100.0,
                    format="%.2f",
                    key=f"setup_price_{idx}",
                    label_visibility="collapsed" if idx > 0 else "visible",
                )
            with cols[3]:
                h["date"] = st.date_input(
                    "Ngày mua",
                    value=h["date"],
                    key=f"setup_date_{idx}",
                    label_visibility="collapsed" if idx > 0 else "visible",
                )
            with cols[4]:
                if idx > 0:
                    st.write("")  # Spacer cho alignment
                if st.button("🗑️", key=f"remove_holding_{idx}", help="Xóa vị thế này"):
                    to_remove.append(idx)

        # Xóa các vị thế được đánh dấu
        for idx in reversed(to_remove):
            holdings_list.pop(idx)
        if to_remove:
            st.rerun()

        # Nút thêm vị thế mới
        if st.button("➕ Thêm vị thế", key="add_new_holding"):
            holdings_list.append({
                "symbol": "",
                "shares": 100,
                "price": 50000.0,
                "date": date.today(),
            })
            st.rerun()

        st.divider()

        # --- Tính tổng và lưu ---
        total_stock_value = sum(h["shares"] * h["price"] for h in holdings_list if h["symbol"])
        total_assets = new_cash + total_stock_value

        col_summary, col_save = st.columns([3, 1])
        with col_summary:
            st.markdown(
                f"**Tổng tài sản:** {total_assets:,.0f}₫ "
                f"(Tiền mặt: {new_cash:,.0f}₫ + CP: {total_stock_value:,.0f}₫)"
            )
        with col_save:
            if st.button("💾 Lưu tất cả", key="save_all_assets", type="primary"):
                # Tạo portfolio mới
                new_holdings = []
                for h in holdings_list:
                    if h["symbol"] and h["shares"] > 0:
                        new_holdings.append(
                            PortfolioHolding(
                                symbol=h["symbol"],
                                shares=h["shares"],
                                buy_price=h["price"] / 1000.0,  # Chuyển về đơn vị 1000 VND
                                buy_date=h["date"],
                            )
                        )

                portfolio.cash = new_cash
                portfolio.holdings = new_holdings
                portfolio.initial_capital = total_assets  # Vốn ban đầu = tổng tài sản nhập vào
                save_portfolio(portfolio)

                # Clear session state
                st.session_state.pop("_setup_cash", None)
                st.session_state.pop("_setup_holdings", None)

                st.success(f"✅ Đã lưu: {new_cash:,.0f}₫ tiền mặt + {len(new_holdings)} vị thế")
                st.rerun()

    if not portfolio.holdings:
        st.info("Chưa có vị thế nào. Thêm thủ công ở trên hoặc thực hiện MUA từ khuyến nghị bên dưới.")
        return

    # Header
    cols = st.columns([1.0, 0.8, 1.2, 1.2, 1.2, 1.0, 0.8])
    with cols[0]:
        st.caption("**Mã CK**")
    with cols[1]:
        st.caption("**Số CP**")
    with cols[2]:
        st.caption("**Giá mua**")
    with cols[3]:
        st.caption("**Giá hiện tại**")
    with cols[4]:
        st.caption("**Lãi/Lỗ**")
    with cols[5]:
        st.caption("**Ngày mua**")
    with cols[6]:
        st.caption("**Thao tác**")

    # Rows
    for idx, holding in enumerate(portfolio.holdings):
        cols = st.columns([1.0, 0.8, 1.2, 1.2, 1.2, 1.0, 0.8])
        current_price = _get_current_price(holding.symbol)

        with cols[0]:
            st.markdown(f"**{holding.symbol}**")
        with cols[1]:
            st.write(f"{holding.shares:,}")
        with cols[2]:
            st.write(f"{holding.buy_price * 1000:,.0f}₫")
        with cols[3]:
            if current_price is not None:
                st.write(f"{current_price * 1000:,.0f}₫")
            else:
                st.write("—")
        with cols[4]:
            if current_price is not None:
                pnl = (current_price - holding.buy_price) * holding.shares * 1000
                pnl_pct = (current_price / holding.buy_price - 1) * 100
                color = "green" if pnl >= 0 else "red"
                st.markdown(f":{color}[{pnl:+,.0f}₫ ({pnl_pct:+.1f}%)]")
            else:
                st.write("—")
        with cols[5]:
            st.write(holding.buy_date.strftime("%d/%m/%Y"))
        with cols[6]:
            if st.button("🔴 Bán", key=f"sell_holding_{idx}"):
                _execute_sell(portfolio, holding.symbol, holding.shares)
                st.rerun()


def _render_recommendations(portfolio: PortfolioState) -> None:
    """
    Render Section 3: Khuyến nghị AI với nút thực hiện MUA/BÁN.
    """
    st.subheader("⭐ Khuyến nghị cổ phiếu")
    st.caption("Dựa trên phân tích AI — có tính đến portfolio hiện tại")

    engine = _get_recommendation_engine()

    # Nút phân tích
    col_refresh, col_info = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 Phân tích", type="primary", help="Scan lại toàn bộ mã cổ phiếu"):
            recommendations = engine.scan_symbols(portfolio_state=portfolio)
            st.session_state["_rec_cache"] = recommendations
            st.rerun()

    # Lấy recommendations (từ cache hoặc scan mới)
    recommendations = st.session_state.get("_rec_cache") or engine.get_recommendations()
    if not recommendations:
        recommendations = engine.scan_symbols(portfolio_state=portfolio)
        st.session_state["_rec_cache"] = recommendations

    with col_info:
        if recommendations:
            st.caption(
                f"Model: `{recommendations[0].model_version}` | "
                f"Ngày phân tích: {recommendations[0].analysis_date.strftime('%d/%m/%Y')} | "
                f"Ngày mua: {recommendations[0].buy_date.strftime('%d/%m/%Y')}"
            )

    st.divider()

    # Summary metrics
    buy_count = sum(1 for r in recommendations if r.action == Action.BUY)
    hold_count = sum(1 for r in recommendations if r.action == Action.HOLD)
    sell_count = sum(1 for r in recommendations if r.action == Action.SELL)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tổng mã phân tích", len(recommendations))
    with col2:
        st.metric("🟢 MUA", buy_count)
    with col3:
        st.metric("🟡 GIỮ", hold_count)
    with col4:
        st.metric("🔴 BÁN", sell_count)

    st.divider()

    # Kiểm tra ngưỡng confidence (Req 6.4)
    if not engine.has_strong_recommendations():
        st.warning(
            "⚠️ **Không có khuyến nghị mạnh**\n\n"
            f"Tất cả mã cổ phiếu có độ tin cậy dưới ngưỡng {CONFIDENCE_THRESHOLD:.0%}. "
            "Nên chờ tín hiệu rõ ràng hơn trước khi giao dịch."
        )

    # Header bảng khuyến nghị
    header_cols = st.columns([1.0, 0.8, 1.1, 0.9, 1.1, 1.0, 0.9, 0.8])
    with header_cols[0]:
        st.caption("**Mã CK**")
    with header_cols[1]:
        st.caption("**Hành động**")
    with header_cols[2]:
        st.caption("**Giá mua**")
    with header_cols[3]:
        st.caption("**Số CP**")
    with header_cols[4]:
        st.caption("**Nên bán ngày**")
    with header_cols[5]:
        st.caption("**Tín hiệu AI**")
    with header_cols[6]:
        st.caption("**Độ tin cậy**")
    with header_cols[7]:
        st.caption("**Thao tác**")

    # Render từng recommendation
    for idx, rec in enumerate(recommendations):
        action_info = _ACTION_DISPLAY.get(
            rec.action,
            {"icon": "⚪", "color": "gray", "label": rec.action.value.upper()},
        )

        row_cols = st.columns([1.0, 0.8, 1.1, 0.9, 1.1, 1.0, 0.9, 0.8])

        with row_cols[0]:
            st.markdown(f"**{rec.symbol}**")

        with row_cols[1]:
            st.markdown(f"{action_info['icon']} :{action_info['color']}[{action_info['label']}]")

        with row_cols[2]:
            if rec.recommended_buy_price is not None:
                st.write(f"{rec.recommended_buy_price * 1000:,.0f}₫")
            else:
                st.write("—")

        with row_cols[3]:
            if rec.recommended_shares is not None and rec.recommended_buy_price is not None:
                cost_vnd = rec.recommended_shares * rec.recommended_buy_price * 1000
                st.write(f"{rec.recommended_shares:,} CP")
                st.caption(f"~{cost_vnd / 1_000_000:.0f}tr")
            else:
                st.write("—")

        with row_cols[4]:
            if rec.recommended_sell_date is not None:
                st.write(rec.recommended_sell_date.strftime("%d/%m/%Y"))
            else:
                st.write("—")

        with row_cols[5]:
            pos_text = f"{rec.position_score:+.2f}"
            if rec.position_score > 0:
                st.markdown(f"🟢 {pos_text}")
            elif rec.position_score < 0:
                st.markdown(f"🔴 {pos_text}")
            else:
                st.markdown(f"⚪ {pos_text}")

        with row_cols[6]:
            st.progress(rec.confidence, text=f"{rec.confidence:.0%}")

        with row_cols[7]:
            if rec.action == Action.BUY and rec.recommended_shares and rec.recommended_buy_price:
                with st.popover("🟢 MUA", width="stretch"):
                    # Hiển thị chi tiết khuyến nghị
                    st.caption(f"**{rec.symbol}** — Giá: {rec.recommended_buy_price * 1000:,.0f}₫")
                    # Cho phép chỉnh số lượng
                    buy_shares = st.number_input(
                        "Số CP mua",
                        min_value=100,
                        max_value=rec.recommended_shares * 2,
                        value=rec.recommended_shares,
                        step=100,
                        key=f"buy_qty_{idx}",
                    )
                    # Hiển thị chi phí
                    cost = buy_shares * rec.recommended_buy_price * 1000
                    fee = cost * TRANSACTION_FEE_RATE
                    total = cost + fee
                    st.caption(f"Chi phí: {cost:,.0f}₫ + phí {fee:,.0f}₫ = **{total:,.0f}₫**")
                    st.caption(f"Tiền mặt còn: {portfolio.cash - total:,.0f}₫")
                    # Nút xác nhận
                    if total <= portfolio.cash:
                        if st.button("✅ Xác nhận MUA", key=f"confirm_buy_{idx}", type="primary"):
                            _execute_buy(portfolio, rec.symbol, buy_shares, rec.recommended_buy_price)
                            st.session_state.pop("_rec_cache", None)
                            st.rerun()
                    else:
                        st.error("Không đủ tiền mặt!")
            elif rec.action == Action.SELL and rec.recommended_shares:
                if st.button("🔴 BÁN", key=f"exec_sell_{idx}"):
                    _execute_sell(portfolio, rec.symbol, rec.recommended_shares)
                    st.session_state.pop("_rec_cache", None)
                    st.rerun()


def page_recommendations() -> None:
    """
    Render trang Recommendations với Portfolio Tracker.

    Flow:
    1. Load portfolio từ file
    2. Render tóm tắt portfolio
    3. Render bảng holdings
    4. Render khuyến nghị AI + nút thực hiện

    References: Req 6.1, 6.3, 6.4, 5.3
    """
    st.title("⭐ Khuyến nghị & Portfolio")
    st.caption("Quản lý portfolio và khuyến nghị dựa trên phân tích AI")

    # Luôn load từ file để đảm bảo persistence
    portfolio = _get_portfolio()

    # Section 1: Tóm tắt portfolio
    _render_portfolio_summary(portfolio)

    st.divider()

    # Section 2: Bảng holdings
    _render_holdings_table(portfolio)

    st.divider()

    # Section 3: Khuyến nghị AI
    _render_recommendations(portfolio)


# Streamlit page entry point — chạy khi được load bởi st.navigation
import streamlit.runtime.scriptrunner as _sr

try:
    _ctx = _sr.get_script_run_ctx()
except Exception:
    _ctx = None

if _ctx is not None:
    page_recommendations()
