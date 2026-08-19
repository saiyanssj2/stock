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

    # Nút cài đặt vốn ban đầu
    with st.expander("⚙️ Cài đặt vốn ban đầu"):
        new_capital = st.number_input(
            "Vốn ban đầu (VND)",
            min_value=0.0,
            value=portfolio.initial_capital,
            step=100_000_000.0,
            format="%.0f",
            key="initial_capital_input",
        )
        if st.button("💾 Lưu vốn ban đầu", key="save_capital_btn"):
            portfolio.initial_capital = new_capital
            portfolio.cash = new_capital
            portfolio.holdings = []
            save_portfolio(portfolio)
            st.success("✅ Đã cập nhật vốn ban đầu và reset portfolio!")
            st.rerun()


def _render_holdings_table(portfolio: PortfolioState) -> None:
    """
    Render Section 2: Bảng holdings hiện tại + form nhập thủ công.

    Hiển thị: Mã | Số CP | Giá mua | Giá hiện tại | Lãi/Lỗ | Ngày mua | Nút Bán
    """
    st.subheader("📊 Cổ phiếu đang nắm giữ")

    # Form nhập vị thế thủ công
    with st.expander("➕ Thêm vị thế thủ công (đã mua ngoài app)"):
        col_sym, col_shares, col_price, col_date = st.columns(4)
        with col_sym:
            manual_symbol = st.text_input("Mã CK", key="manual_symbol", placeholder="VD: FPT").strip().upper()
        with col_shares:
            manual_shares = st.number_input("Số CP", min_value=100, step=100, value=100, key="manual_shares")
        with col_price:
            manual_price = st.number_input("Giá mua (₫)", min_value=1000.0, step=1000.0, value=50000.0, format="%.0f", key="manual_price")
        with col_date:
            manual_date = st.date_input("Ngày mua", value=date.today(), key="manual_date")

        if st.button("💾 Lưu vị thế", key="save_manual_holding"):
            if manual_symbol and manual_shares >= 100:
                # Giá nhập là VND thực, chuyển về đơn vị 1000 VND
                price_1000 = manual_price / 1000.0
                existing = next((h for h in portfolio.holdings if h.symbol == manual_symbol), None)
                if existing is not None:
                    # Cập nhật giá trung bình
                    total_shares = existing.shares + manual_shares
                    existing.buy_price = (
                        (existing.buy_price * existing.shares + price_1000 * manual_shares) / total_shares
                    )
                    existing.shares = total_shares
                    existing.buy_date = manual_date
                else:
                    portfolio.holdings.append(
                        PortfolioHolding(
                            symbol=manual_symbol,
                            shares=manual_shares,
                            buy_price=price_1000,
                            buy_date=manual_date,
                        )
                    )
                # Trừ tiền mặt tương ứng
                cost = manual_shares * manual_price
                portfolio.cash -= cost
                save_portfolio(portfolio)
                st.success(f"✅ Đã thêm {manual_symbol} {manual_shares} CP, giá {manual_price:,.0f}₫")
                st.rerun()
            else:
                st.error("❌ Vui lòng nhập mã CK và số CP ≥ 100")

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
