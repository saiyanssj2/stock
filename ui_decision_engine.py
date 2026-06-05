"""
Decision Engine UI tab for the Streamlit application.

Provides a Streamlit interface for:
- Selecting a VN30 stock symbol for analysis
- Running the DecisionEngine.analyze() pipeline
- Displaying the DecisionReport: signal badge, confidence, position score
- Showing top 5 indicator contributions as a bar chart
- Showing top 3 scenarios as a ranked list
- Backtesting and strategy comparison with equity curves and trade list
- Displaying model info and training status

Requirements: 11.1, 11.4, 11.5, 11.6, 11.7, 9.6, 10.6, 10.7, 7.4, 7.5, 7.6
"""

import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.config import Action, ComparisonResult, ConfigError, DataError, ModelError, TrainingConfig

# VN30 component stocks (plus VNINDEX for context)
VN30_SYMBOLS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR",
    "HDB", "HPG", "MBB", "MSN", "MWG", "PLX", "POW", "SAB",
    "SHB", "SSB", "SSI", "STB", "TCB", "TPB", "VCB", "VHM",
    "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]


def show_decision_engine(base_dir: str) -> None:
    """Render the Decision Engine UI tab.

    Parameters
    ----------
    base_dir : str
        Path to the data directory containing CSV files.
    """
    st.header("🧠 Decision Engine")
    st.caption("Phân tích quyết định giao dịch bằng AI (Minimax + Neural Network)")

    # --- Model Info & Training Status (sidebar) ---
    _display_model_info_and_training_status(base_dir)

    # --- Symbol selection ---
    # Filter to only show symbols that have CSV data available
    available_symbols = _get_available_symbols(base_dir)

    if not available_symbols:
        st.warning(
            "⚠️ Không tìm thấy dữ liệu CSV nào. "
            "Vui lòng chạy cập nhật dữ liệu trước."
        )
        return

    selected_symbol = st.selectbox(
        "Chọn mã cổ phiếu",
        available_symbols,
        index=0,
        key="de_symbol_select",
    )

    # --- Sub-tabs: Analysis and Backtest ---
    tab_analysis, tab_backtest = st.tabs(["📊 Phân tích", "📈 Backtest"])

    with tab_analysis:
        _show_analysis_tab(base_dir, selected_symbol)

    with tab_backtest:
        _show_backtest_tab(base_dir, selected_symbol)


# ==============================================================================
# Analysis Sub-Tab
# ==============================================================================


def _show_analysis_tab(base_dir: str, selected_symbol: str) -> None:
    """Render the analysis sub-tab content.

    Parameters
    ----------
    base_dir : str
        Path to the data directory.
    selected_symbol : str
        The stock symbol selected by the user.
    """
    analyze_btn = st.button("🔍 Phân tích", type="primary", key="de_analyze_btn")

    if not analyze_btn:
        st.info("👆 Nhấn **Phân tích** để bắt đầu phân tích cho mã đã chọn.")
        return

    # --- Run analysis ---
    with st.spinner("Đang phân tích... (có thể mất vài giây)"):
        try:
            from engine.decision_engine import DecisionEngine

            engine = DecisionEngine(base_dir)
            report = engine.analyze(selected_symbol)
        except DataError as e:
            st.error(f"❌ Lỗi dữ liệu: {e.message}")
            if e.details.get("suggestion"):
                st.info(f"💡 {e.details['suggestion']}")
            return
        except ModelError as e:
            st.error(f"❌ Lỗi mô hình: {e.message}")
            return
        except Exception as e:
            st.error(f"❌ Lỗi không xác định: {str(e)}")
            return

    # --- Display results ---
    st.divider()
    _display_signal_badge(report.recommended_action, report.symbol)
    _display_metrics(report)
    st.divider()
    _display_indicator_contributions(report.top_indicators)
    st.divider()
    _display_scenarios(report.top_scenarios)


# ==============================================================================
# Backtest Sub-Tab
# ==============================================================================


def _show_backtest_tab(base_dir: str, selected_symbol: str) -> None:
    """Render the backtest sub-tab with strategy comparison.

    Provides:
    - Date range selector (default 1 year)
    - Initial capital input (default 100M VND)
    - Strategy comparison table with metrics
    - Overlaid equity curves chart
    - Trade list in expandable sections

    Parameters
    ----------
    base_dir : str
        Path to the data directory.
    selected_symbol : str
        The stock symbol selected by the user.

    Requirements: 11.7, 9.6, 10.6, 10.7
    """
    st.subheader("📈 Backtest & So sánh chiến lược")
    st.caption(
        "So sánh hiệu suất AI Engine với các chiến lược truyền thống "
        "(Wyckoff, Technical, Momentum, Mean Reversion)"
    )

    # --- Inputs: date range and initial capital ---
    col_start, col_end, col_capital = st.columns(3)

    default_end = date.today()
    default_start = default_end - timedelta(days=365)

    with col_start:
        start_date = st.date_input(
            "Ngày bắt đầu",
            value=default_start,
            key="bt_start_date",
        )
    with col_end:
        end_date = st.date_input(
            "Ngày kết thúc",
            value=default_end,
            key="bt_end_date",
        )
    with col_capital:
        initial_capital = st.number_input(
            "Vốn ban đầu (VND)",
            min_value=10_000_000,
            max_value=10_000_000_000,
            value=100_000_000,
            step=10_000_000,
            format="%d",
            key="bt_initial_capital",
        )

    # Run backtest button
    run_backtest_btn = st.button(
        "🚀 Chạy Backtest", type="primary", key="bt_run_btn"
    )

    if not run_backtest_btn:
        st.info(
            "👆 Thiết lập khoảng thời gian, vốn ban đầu và nhấn "
            "**Chạy Backtest** để so sánh chiến lược."
        )
        return

    # Validate date range
    if start_date >= end_date:
        st.error("❌ Ngày bắt đầu phải trước ngày kết thúc.")
        return

    # --- Run comparison ---
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    with st.spinner("Đang chạy backtest... (có thể mất 1-2 phút)"):
        try:
            from engine.decision_engine import DecisionEngine

            engine = DecisionEngine(base_dir)
            comparison = engine.compare(
                selected_symbol, start_str, end_str, initial_capital=float(initial_capital)
            )
        except DataError as e:
            st.error(f"❌ Lỗi dữ liệu: {e.message}")
            if e.details.get("suggestion"):
                st.info(f"💡 {e.details['suggestion']}")
            return
        except ConfigError as e:
            st.error(f"❌ Lỗi cấu hình: {e.message}")
            return
        except Exception as e:
            st.error(f"❌ Lỗi không xác định: {str(e)}")
            return

    # --- Display results ---
    st.divider()
    _display_backtest_results(comparison, selected_symbol)


def _display_backtest_results(
    comparison: ComparisonResult, symbol: str
) -> None:
    """Display full backtest comparison results.

    Parameters
    ----------
    comparison : ComparisonResult
        The comparison results from DecisionEngine.compare().
    symbol : str
        The stock symbol that was backtested.
    """
    # Show excluded strategies warning if any
    if comparison.excluded_strategies:
        for name in comparison.excluded_strategies:
            reason = comparison.exclusion_reasons.get(name, "Không đủ tín hiệu")
            st.warning(f"⚠️ Chiến lược **{name}** bị loại: {reason}")

    if not comparison.results:
        st.warning("Không có chiến lược nào hoàn thành backtest trong khoảng thời gian này.")
        return

    # --- Strategy Comparison Table ---
    _display_comparison_table(comparison)

    st.divider()

    # --- Equity Curves ---
    _display_equity_curves(comparison, symbol)

    st.divider()

    # --- Trade Lists ---
    _display_trade_lists(comparison)


def _display_comparison_table(comparison: ComparisonResult) -> None:
    """Display strategy comparison table with key metrics.

    Shows: total return %, annualized return %, win rate %, max drawdown %, Sharpe ratio.

    Parameters
    ----------
    comparison : ComparisonResult
        The comparison results.
    """
    st.subheader("📋 Bảng so sánh chiến lược")

    rows = []
    for name, result in comparison.results.items():
        rows.append(
            {
                "Chiến lược": name,
                "Tổng lợi nhuận (%)": f"{result.total_return_pct:.2f}",
                "Lợi nhuận/năm (%)": f"{result.annualized_return_pct:.2f}",
                "Tỷ lệ thắng (%)": f"{result.win_rate:.1f}",
                "Max Drawdown (%)": f"{result.max_drawdown:.2f}",
                "Sharpe Ratio": f"{result.sharpe_ratio:.3f}",
            }
        )

    df_table = pd.DataFrame(rows)
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    # Highlight info
    st.caption(
        f"Khoảng thời gian: {comparison.date_range_start} → {comparison.date_range_end} | "
        f"Vốn ban đầu: {comparison.initial_capital:,.0f} VND"
    )


def _display_equity_curves(comparison: ComparisonResult, symbol: str) -> None:
    """Display overlaid equity curves for all strategies.

    Uses plotly line chart with distinct colors per strategy.

    Parameters
    ----------
    comparison : ComparisonResult
        The comparison results.
    symbol : str
        The stock symbol for the chart title.
    """
    st.subheader("📉 Đường equity (Equity Curves)")

    # Distinct color palette for strategies
    color_palette = [
        "#2196F3",  # Blue - AI Engine
        "#FF9800",  # Orange - Wyckoff
        "#4CAF50",  # Green - Technical
        "#9C27B0",  # Purple - Momentum
        "#F44336",  # Red - Mean Reversion
    ]

    fig = go.Figure()

    for i, (name, result) in enumerate(comparison.results.items()):
        if result.equity_curve is not None and len(result.equity_curve) > 0:
            color = color_palette[i % len(color_palette)]
            fig.add_trace(
                go.Scatter(
                    x=result.equity_curve.index,
                    y=result.equity_curve.values,
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=2),
                )
            )

    fig.update_layout(
        title=f"Equity Curves - {symbol}",
        xaxis_title="Ngày",
        yaxis_title="Giá trị danh mục (VND)",
        height=450,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
    )

    # Format y-axis with comma separators
    fig.update_yaxes(tickformat=",")

    st.plotly_chart(fig, use_container_width=True)


def _display_trade_lists(comparison: ComparisonResult) -> None:
    """Display trade lists for each strategy in expandable sections.

    Shows entry/exit prices, shares, PnL for each trade.

    Parameters
    ----------
    comparison : ComparisonResult
        The comparison results.
    """
    st.subheader("📝 Danh sách giao dịch")

    for name, result in comparison.results.items():
        trades = result.trades
        trade_count = len(trades)
        label = f"📋 {name} — {trade_count} giao dịch"

        with st.expander(label):
            if not trades:
                st.info("Không có giao dịch nào được thực hiện.")
                continue

            trade_rows = []
            for t in trades:
                entry_date_str = (
                    t.entry_date.strftime("%Y-%m-%d")
                    if t.entry_date is not None
                    else "N/A"
                )
                exit_date_str = (
                    t.exit_date.strftime("%Y-%m-%d")
                    if t.exit_date is not None
                    else "N/A"
                )
                trade_rows.append(
                    {
                        "Ngày mua": entry_date_str,
                        "Ngày bán": exit_date_str,
                        "Giá mua": f"{t.entry_price:,.0f}",
                        "Giá bán": f"{t.exit_price:,.0f}",
                        "Số CP": f"{t.shares:,}",
                        "Lãi/Lỗ (VND)": f"{t.pnl:,.0f}",
                        "Lãi/Lỗ (%)": f"{t.pnl_pct:+.2f}%",
                    }
                )

            df_trades = pd.DataFrame(trade_rows)
            st.dataframe(df_trades, use_container_width=True, hide_index=True)


# ==============================================================================
# Analysis Display Helpers
# ==============================================================================


def _get_available_symbols(base_dir: str) -> list:
    """Get VN30 symbols that have CSV data available.

    Checks for {symbol}.csv or {symbol}_full.csv in base_dir.

    Parameters
    ----------
    base_dir : str
        Path to the data directory.

    Returns
    -------
    list
        Sorted list of available VN30 symbols.
    """
    available = []
    for symbol in VN30_SYMBOLS:
        csv_path = os.path.join(base_dir, f"{symbol}_full.csv")
        if os.path.exists(csv_path):
            available.append(symbol)
            continue
        csv_path = os.path.join(base_dir, f"{symbol}.csv")
        if os.path.exists(csv_path):
            available.append(symbol)
    return sorted(available)


def _display_signal_badge(action: Action, symbol: str) -> None:
    """Display the signal badge with appropriate color.

    Parameters
    ----------
    action : Action
        The recommended action (BUY, HOLD, SELL).
    symbol : str
        The stock symbol.
    """
    color_map = {
        Action.BUY: "green",
        Action.HOLD: "gray",
        Action.SELL: "red",
    }
    label_map = {
        Action.BUY: "🟢 MUA (BUY)",
        Action.HOLD: "⚪ GIỮ (HOLD)",
        Action.SELL: "🔴 BÁN (SELL)",
    }
    color = color_map.get(action, "gray")
    label = label_map.get(action, "HOLD")

    st.markdown(
        f"### Tín hiệu cho {symbol}: "
        f"<span style='color:{color}; font-weight:bold; font-size:1.2em;'>"
        f"{label}</span>",
        unsafe_allow_html=True,
    )


def _display_metrics(report) -> None:
    """Display confidence score and position score as metrics.

    Parameters
    ----------
    report : DecisionReport
        The decision report containing confidence and position_score.
    """
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Độ tin cậy (Confidence)",
            value=f"{report.confidence:.1%}",
        )
        # Also show as a progress bar for visual clarity
        st.progress(report.confidence)

    with col2:
        # Position score in [-1, 1], show as metric with sign
        score_display = f"{report.position_score:+.3f}"
        st.metric(
            label="Điểm vị thế (Position Score)",
            value=score_display,
        )
        # Visual indicator: map [-1, 1] to [0, 1] for progress bar
        normalized_score = (report.position_score + 1.0) / 2.0
        st.progress(normalized_score)
        st.caption("-1.0 (Bất lợi) ← → +1.0 (Thuận lợi)")


def _display_indicator_contributions(indicators: list) -> None:
    """Display top 5 indicator contributions as a horizontal bar chart.

    Parameters
    ----------
    indicators : list of IndicatorContribution
        Top indicators sorted by absolute contribution.
    """
    st.subheader("📊 Top 5 chỉ báo đóng góp")

    if not indicators:
        st.info("Không có dữ liệu chỉ báo.")
        return

    # Build horizontal bar chart using plotly (consistent with project)
    names = [ind.name for ind in indicators]
    contributions = [ind.contribution for ind in indicators]

    fig = go.Figure(
        go.Bar(
            x=contributions,
            y=names,
            orientation="h",
            marker_color=[
                "green" if c > 0 else "red" for c in contributions
            ],
        )
    )
    fig.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=10, b=10),
        xaxis_title="Mức đóng góp",
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Also show as a table with values
    with st.expander("📋 Chi tiết chỉ báo"):
        detail_data = pd.DataFrame(
            {
                "Chỉ báo": [ind.name for ind in indicators],
                "Giá trị hiện tại": [f"{ind.value:.4f}" for ind in indicators],
                "Mức đóng góp": [f"{ind.contribution:.4f}" for ind in indicators],
            }
        )
        st.dataframe(detail_data, use_container_width=True, hide_index=True)


def _display_scenarios(scenarios: list) -> None:
    """Display top 3 scenarios as a ranked list.

    Parameters
    ----------
    scenarios : list of ScenarioResult
        Top scenarios from the search module.
    """
    st.subheader("🎯 Top 3 kịch bản")

    if not scenarios:
        st.info("Không có kịch bản nào được khám phá.")
        return

    for i, scenario in enumerate(scenarios, 1):
        # Medal emoji for ranking
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, "▪️")

        # Format action sequence
        action_str = " → ".join(a.value for a in scenario.action_sequence)

        # Score color
        score = scenario.leaf_score
        if score > 0.3:
            score_color = "green"
        elif score < -0.3:
            score_color = "red"
        else:
            score_color = "orange"

        st.markdown(
            f"{medal} **Kịch bản {i}** — Điểm: "
            f"<span style='color:{score_color};font-weight:bold;'>"
            f"{score:+.3f}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;Chuỗi hành động: `{action_str}`")
        if scenario.description:
            st.markdown(
                f"&nbsp;&nbsp;&nbsp;&nbsp;_{scenario.description}_"
            )


# ==============================================================================
# Model Info & Training Status
# ==============================================================================


def _display_model_info_and_training_status(base_dir: str) -> None:
    """Display model version, last training timestamp, and training progress.

    Shows in a sidebar section:
    - Model version and last training timestamp
    - Training progress when active (epoch, loss, ETA)
    - Notification when training paused due to GPU constraints

    Parameters
    ----------
    base_dir : str
        Path to the data directory.

    Requirements: 7.4, 7.5, 7.6
    """
    with st.sidebar:
        st.subheader("🤖 Thông tin Model")

        # Load model info
        model_path = Path(base_dir) / "engine" / "models" / "stock_eval_net.pt"
        if model_path.exists():
            mod_time = datetime.fromtimestamp(model_path.stat().st_mtime)
            st.metric("Phiên bản", f"v{mod_time.strftime('%Y%m%d')}")
            st.caption(f"Cập nhật: {mod_time.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.info("Chưa có model. Chạy Training Pipeline để tạo.")

        # Check for active training
        training_log_path = Path(base_dir) / "engine" / "models" / "training_log.jsonl"
        if training_log_path.exists():
            try:
                lines = training_log_path.read_text(encoding="utf-8").strip().split("\n")
                if lines:
                    last_entry = json.loads(lines[-1])
                    if "epoch" in last_entry:
                        # Check if training is recent (within last 60 seconds)
                        log_time = last_entry.get("timestamp", "")
                        st.caption(
                            f"Epoch gần nhất: {last_entry.get('epoch', 'N/A')} | "
                            f"Loss: {last_entry.get('loss', 'N/A')}"
                        )
            except (json.JSONDecodeError, OSError, KeyError):
                pass

        st.divider()
