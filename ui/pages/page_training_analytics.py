# -*- coding: utf-8 -*-
"""
Training Analytics Page - Trực quan hóa số liệu training model.

Hiển thị:
- Biểu đồ Return, Win Rate, val_loss theo thời gian
- Milestones và improvement phases
- AI insights giải thích xu hướng model
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine.training_analytics import (
    load_all_cycles,
    calculate_moving_averages,
    detect_improvement_phases,
    get_milestones,
    get_summary_stats,
    analyze_trend,
)


def _create_return_chart(df: pd.DataFrame) -> go.Figure:
    """Tạo biểu đồ Return % theo cycle."""
    fig = go.Figure()
    
    # Return line
    fig.add_trace(go.Scatter(
        x=df["cycle"],
        y=df["return_pct"],
        mode="lines",
        name="Return %",
        line=dict(color="#2ecc71", width=1),
        opacity=0.6,
    ))
    
    # Moving average
    if "return_pct_ma20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["cycle"],
            y=df["return_pct_ma20"],
            mode="lines",
            name="MA20",
            line=dict(color="#27ae60", width=2),
        ))
    
    fig.update_layout(
        title="📈 Backtest Return % theo Cycle",
        xaxis_title="Cycle",
        yaxis_title="Return %",
        hovermode="x unified",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    
    return fig


def _create_win_rate_chart(df: pd.DataFrame) -> go.Figure:
    """Tạo biểu đồ Win Rate theo cycle."""
    fig = go.Figure()
    
    # Win rate line
    fig.add_trace(go.Scatter(
        x=df["cycle"],
        y=df["win_rate"],
        mode="lines",
        name="Win Rate %",
        line=dict(color="#3498db", width=1),
        opacity=0.6,
    ))
    
    # Moving average
    if "win_rate_ma20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["cycle"],
            y=df["win_rate_ma20"],
            mode="lines",
            name="MA20",
            line=dict(color="#2980b9", width=2),
        ))
    
    # Reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="orange", 
                  annotation_text="Ngưỡng 70%", annotation_position="right")
    fig.add_hline(y=75, line_dash="dash", line_color="green",
                  annotation_text="Mục tiêu 75%", annotation_position="right")
    
    fig.update_layout(
        title="🎯 Win Rate % theo Cycle",
        xaxis_title="Cycle",
        yaxis_title="Win Rate %",
        hovermode="x unified",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    
    return fig


def _create_val_loss_chart(df: pd.DataFrame) -> go.Figure:
    """Tạo biểu đồ val_loss theo cycle."""
    fig = go.Figure()
    
    # val_loss line
    fig.add_trace(go.Scatter(
        x=df["cycle"],
        y=df["val_loss"],
        mode="lines",
        name="val_loss",
        line=dict(color="#e74c3c", width=1),
        opacity=0.6,
    ))
    
    # Moving average
    if "val_loss_ma20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["cycle"],
            y=df["val_loss_ma20"],
            mode="lines",
            name="MA20",
            line=dict(color="#c0392b", width=2),
        ))
    
    fig.update_layout(
        title="📉 Validation Loss theo Cycle",
        xaxis_title="Cycle",
        yaxis_title="val_loss",
        hovermode="x unified",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    
    return fig


def _create_rl_metrics_chart(df: pd.DataFrame) -> go.Figure:
    """Tạo biểu đồ RL metrics theo cycle."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("RL Avg Return %", "RL Best Return %"),
        horizontal_spacing=0.1,
    )
    
    # RL avg return
    fig.add_trace(go.Scatter(
        x=df["cycle"],
        y=df["rl_avg_return"],
        mode="lines",
        name="Avg Return",
        line=dict(color="#9b59b6", width=1),
    ), row=1, col=1)
    
    if "rl_avg_return_ma20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["cycle"],
            y=df["rl_avg_return_ma20"],
            mode="lines",
            name="Avg MA20",
            line=dict(color="#8e44ad", width=2),
        ), row=1, col=1)
    
    # RL best return
    fig.add_trace(go.Scatter(
        x=df["cycle"],
        y=df["rl_best_return"],
        mode="lines",
        name="Best Return",
        line=dict(color="#f39c12", width=1),
    ), row=1, col=2)
    
    fig.update_layout(
        title="🤖 Reinforcement Learning Metrics",
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    
    return fig


def _create_sharpe_chart(df: pd.DataFrame) -> go.Figure:
    """Tạo biểu đồ Sharpe ratio theo cycle."""
    fig = go.Figure()
    
    # Sharpe line
    fig.add_trace(go.Scatter(
        x=df["cycle"],
        y=df["sharpe"],
        mode="lines",
        name="Sharpe Ratio",
        line=dict(color="#1abc9c", width=1),
        opacity=0.6,
    ))
    
    # Moving average
    if "sharpe_ma20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["cycle"],
            y=df["sharpe_ma20"],
            mode="lines",
            name="MA20",
            line=dict(color="#16a085", width=2),
        ))
    
    # Reference lines
    fig.add_hline(y=1.0, line_dash="dash", line_color="orange",
                  annotation_text="Sharpe 1.0", annotation_position="right")
    fig.add_hline(y=2.0, line_dash="dash", line_color="green",
                  annotation_text="Sharpe 2.0 (Excellent)", annotation_position="right")
    
    fig.update_layout(
        title="📊 Sharpe Ratio theo Cycle",
        xaxis_title="Cycle",
        yaxis_title="Sharpe",
        hovermode="x unified",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    
    return fig


def _create_combined_overview(df: pd.DataFrame) -> go.Figure:
    """Tạo biểu đồ tổng quan với nhiều metrics (normalized)."""
    fig = go.Figure()
    
    # Normalize các metrics về scale 0-100
    df_norm = df.copy()
    
    # Return: chia cho max rồi nhân 100
    max_return = df["return_pct"].max()
    if max_return > 0:
        df_norm["return_norm"] = (df["return_pct"] / max_return) * 100
    
    # Win rate: giữ nguyên (đã là %)
    df_norm["wr_norm"] = df["win_rate"]
    
    # val_loss: invert (loss thấp = tốt)
    max_loss = df["val_loss"].max()
    min_loss = df["val_loss"].min()
    if max_loss > min_loss:
        df_norm["loss_norm"] = 100 - ((df["val_loss"] - min_loss) / (max_loss - min_loss)) * 100
    
    # Sharpe: scale lên
    max_sharpe = df["sharpe"].max()
    if max_sharpe > 0:
        df_norm["sharpe_norm"] = (df["sharpe"] / max_sharpe) * 100
    
    # Plot
    fig.add_trace(go.Scatter(
        x=df_norm["cycle"],
        y=df_norm.get("return_norm", df["return_pct"]),
        mode="lines",
        name="Return (normalized)",
        line=dict(color="#2ecc71", width=2),
    ))
    
    fig.add_trace(go.Scatter(
        x=df_norm["cycle"],
        y=df_norm["wr_norm"],
        mode="lines",
        name="Win Rate",
        line=dict(color="#3498db", width=2),
    ))
    
    fig.add_trace(go.Scatter(
        x=df_norm["cycle"],
        y=df_norm.get("loss_norm", 50),
        mode="lines",
        name="val_loss (inverted)",
        line=dict(color="#e74c3c", width=2),
    ))
    
    fig.add_trace(go.Scatter(
        x=df_norm["cycle"],
        y=df_norm.get("sharpe_norm", 50),
        mode="lines",
        name="Sharpe (normalized)",
        line=dict(color="#1abc9c", width=2),
    ))
    
    fig.update_layout(
        title="🔄 Tổng quan Metrics (Normalized 0-100)",
        xaxis_title="Cycle",
        yaxis_title="Score (0-100)",
        hovermode="x unified",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    
    return fig


def _render_trend_indicator(trend: str, description: str) -> None:
    """Render trend indicator với icon và màu sắc."""
    if trend == "up":
        st.success(f"📈 {description}")
    elif trend == "down":
        st.error(f"📉 {description}")
    else:
        st.info(f"➡️ {description}")


def _generate_ai_insights(df: pd.DataFrame, stats: dict, phases: list) -> list[str]:
    """Tạo AI insights dựa trên phân tích dữ liệu."""
    insights = []
    
    if not stats:
        return ["Không đủ dữ liệu để phân tích"]
    
    # Insight 1: Overall progress
    total_cycles = stats.get("total_cycles", 0)
    if total_cycles > 100:
        first_50_avg = df.head(50)["return_pct"].mean() if len(df) >= 50 else 0
        last_50_avg = df.tail(50)["return_pct"].mean()
        improvement = ((last_50_avg - first_50_avg) / max(first_50_avg, 1)) * 100
        
        if improvement > 100:
            insights.append(
                f"🚀 **Tiến bộ vượt bậc**: Return trung bình tăng {improvement:.0f}% "
                f"từ {first_50_avg:.0f}% (50 cycles đầu) lên {last_50_avg:.0f}% (50 cycles gần nhất). "
                "Model đã học được patterns hiệu quả từ dữ liệu thị trường."
            )
        elif improvement > 50:
            insights.append(
                f"📈 **Cải thiện tốt**: Return tăng {improvement:.0f}% so với giai đoạn đầu. "
                "Fine-tuning liên tục đang phát huy tác dụng."
            )
    
    # Insight 2: Win rate analysis
    current_wr = stats.get("current_avg_win_rate", 0)
    if current_wr >= 75:
        insights.append(
            f"🎯 **Win Rate xuất sắc**: {current_wr:.1f}% - Model dự đoán đúng 3/4 trades. "
            "Đây là mức rất tốt cho trading model, cho thấy SL đã capture được features quan trọng."
        )
    elif current_wr >= 70:
        insights.append(
            f"✅ **Win Rate tốt**: {current_wr:.1f}% - Vượt ngưỡng 70% cho thấy model "
            "có khả năng phân biệt tốt giữa cơ hội tốt và xấu."
        )
    
    # Insight 3: val_loss trend
    return_trend = analyze_trend(df, "return_pct", 30)
    loss_trend = analyze_trend(df, "val_loss", 30)
    
    if loss_trend["trend"] == "stable" and loss_trend.get("recent_avg", 0) < 0.15:
        insights.append(
            f"📉 **val_loss ổn định**: ~{loss_trend.get('recent_avg', 0):.4f} - "
            "Model không bị overfit, đang generalize tốt trên validation set. "
            "Điều này quan trọng để đảm bảo performance trên dữ liệu mới."
        )
    
    # Insight 4: Improvement phases
    if phases:
        recent_phase = phases[-1]
        insights.append(
            f"🔥 **Giai đoạn cải tiến gần nhất** (Cycles {recent_phase['cycles']}): "
            f"Return +{recent_phase['return_improvement_pct']}%, "
            f"Win Rate +{recent_phase['win_rate_improvement']}%. "
            f"Nguyên nhân: {recent_phase['reason']}"
        )
    
    # Insight 5: RL contribution
    rl_trend = analyze_trend(df, "rl_avg_return", 30)
    if rl_trend["trend"] == "up":
        insights.append(
            "🤖 **RL đang cải thiện**: Policy network đang tìm được chiến lược tốt hơn "
            "qua các episodes. Avg return tăng cho thấy reward shaping hiệu quả."
        )
    
    # Insight 6: Sharpe analysis
    current_sharpe = stats.get("current_avg_sharpe", 0)
    if current_sharpe >= 2.0:
        insights.append(
            f"📊 **Sharpe Ratio xuất sắc**: {current_sharpe:.2f} - "
            "Return/Risk ratio rất tốt. Model không chỉ sinh lời mà còn kiểm soát rủi ro hiệu quả."
        )
    elif current_sharpe >= 1.5:
        insights.append(
            f"📊 **Sharpe Ratio tốt**: {current_sharpe:.2f} - "
            "Đạt mức chấp nhận được cho quỹ đầu tư chuyên nghiệp (>1.0)."
        )
    
    # Insight 7: Training efficiency
    total_hours = stats.get("total_training_hours", 0)
    avg_duration = stats.get("avg_cycle_duration_min", 0)
    if total_hours > 100:
        insights.append(
            f"⏱️ **Đầu tư training**: {total_hours:.0f} giờ, "
            f"trung bình {avg_duration:.0f} phút/cycle. "
            f"Qua {total_cycles} cycles, model đã học từ hàng triệu data points."
        )
    
    return insights if insights else ["Tiếp tục training để có thêm insights"]


def _format_analytics_for_copy(df: pd.DataFrame, stats: dict, milestones: list, phases: list, insights: list) -> str:
    """Format toàn bộ analytics thành text để copy."""
    lines = []
    lines.append("=" * 60)
    lines.append("📊 TRAINING ANALYTICS SUMMARY")
    lines.append("=" * 60)
    lines.append("")
    
    # Summary
    lines.append("📈 TỔNG QUAN PERFORMANCE")
    lines.append("-" * 40)
    lines.append(f"Total Cycles: {stats.get('total_cycles', 0):,}")
    lines.append(f"Avg Return (50 cycles): {stats.get('current_avg_return', 0):,.0f}%")
    lines.append(f"Avg Win Rate: {stats.get('current_avg_win_rate', 0):.1f}%")
    lines.append(f"Avg Sharpe: {stats.get('current_avg_sharpe', 0):.2f}")
    lines.append(f"Avg val_loss: {stats.get('current_avg_val_loss', 0):.4f}")
    lines.append("")
    
    # Trends
    lines.append("📊 XU HƯỚNG HIỆN TẠI (30 cycles gần nhất)")
    lines.append("-" * 40)
    return_trend = analyze_trend(df, "return_pct", 30)
    wr_trend = analyze_trend(df, "win_rate", 30)
    loss_trend = analyze_trend(df, "val_loss", 30)
    sharpe_trend = analyze_trend(df, "sharpe", 30)
    lines.append(f"Return: {return_trend['description']}")
    lines.append(f"Win Rate: {wr_trend['description']}")
    lines.append(f"val_loss: {loss_trend['description']}")
    lines.append(f"Sharpe: {sharpe_trend['description']}")
    lines.append("")
    
    # Milestones
    if milestones:
        lines.append("🏆 MILESTONES")
        lines.append("-" * 40)
        for m in milestones[:6]:
            lines.append(f"{m['icon']} {m['type']}: Cycle #{m['cycle']} - {m['value']} ({m['date']})")
        lines.append("")
    
    # Recent cycles
    lines.append("📋 10 CYCLES GẦN NHẤT")
    lines.append("-" * 40)
    recent = df.tail(10)[["cycle", "return_pct", "win_rate", "val_loss", "sharpe", "trades"]]
    lines.append(f"{'Cycle':>6} {'Return%':>10} {'WR%':>8} {'val_loss':>10} {'Sharpe':>8} {'Trades':>7}")
    for _, row in recent.iterrows():
        lines.append(f"{int(row['cycle']):>6} {row['return_pct']:>10.1f} {row['win_rate']:>8.1f} {row['val_loss']:>10.4f} {row['sharpe']:>8.2f} {int(row['trades']):>7}")
    lines.append("")
    
    # AI Insights
    if insights:
        lines.append("🧠 AI INSIGHTS")
        lines.append("-" * 40)
        for insight in insights:
            # Bỏ markdown formatting
            clean = insight.replace("**", "").replace("_", "")
            lines.append(clean)
            lines.append("")
    
    # Improvement phases
    if phases:
        lines.append("🔥 CÁC GIAI ĐOẠN CẢI TIẾN")
        lines.append("-" * 40)
        for p in phases:
            lines.append(f"Phase {p['phase']}: Cycles {p['cycles']}")
            lines.append(f"  Return +{p['return_improvement_pct']}%, WR +{p['win_rate_improvement']}%")
            lines.append(f"  Nguyên nhân: {p['reason']}")
        lines.append("")
    
    # Footer
    lines.append("=" * 60)
    lines.append(f"Training từ {stats.get('first_cycle_date', 'N/A')} đến {stats.get('last_cycle_date', 'N/A')}")
    lines.append(f"Tổng thời gian: {stats.get('total_training_hours', 0):.0f} giờ")
    
    return "\n".join(lines)


def page_training_analytics() -> None:
    """Trang Training Analytics - Trực quan hóa số liệu training."""
    st.title("📊 Training Analytics")
    st.caption("Phân tích chi tiết quá trình training model từ cycle 1 đến hiện tại")
    
    # Load data
    with st.spinner("Đang load dữ liệu training..."):
        df = load_all_cycles()
    
    if df.empty:
        st.warning("Không tìm thấy dữ liệu training. Hãy chạy pipeline trước.")
        return
    
    # Tính moving averages
    df = calculate_moving_averages(df, window=20)
    
    # Get stats và phases
    stats = get_summary_stats(df)
    milestones = get_milestones(df)
    phases = detect_improvement_phases(df)
    
    # === Summary Cards ===
    st.subheader("📈 Tổng quan Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Cycles",
            f"{stats.get('total_cycles', 0):,}",
            help="Tổng số training cycles đã hoàn thành"
        )
    
    with col2:
        change = stats.get("return_change_vs_prev", 0)
        st.metric(
            "Avg Return (50 cycles)",
            f"{stats.get('current_avg_return', 0):,.0f}%",
            delta=f"{change:+.0f}%" if change else None,
            help="Return trung bình 50 cycles gần nhất"
        )
    
    with col3:
        wr_change = stats.get("wr_change_vs_prev", 0)
        st.metric(
            "Avg Win Rate",
            f"{stats.get('current_avg_win_rate', 0):.1f}%",
            delta=f"{wr_change:+.1f}%" if wr_change else None,
            help="Win rate trung bình 50 cycles gần nhất"
        )
    
    with col4:
        st.metric(
            "Avg Sharpe",
            f"{stats.get('current_avg_sharpe', 0):.2f}",
            help="Sharpe ratio trung bình 50 cycles gần nhất"
        )
    
    # === Nút Copy Analytics ===
    # Tạo insights trước để dùng cho copy
    insights = _generate_ai_insights(df, stats, phases)
    
    # Format text để copy
    copy_text = _format_analytics_for_copy(df, stats, milestones, phases, insights)
    
    # Nút copy sử dụng st.code với label
    with st.expander("📋 Copy Analytics", expanded=False):
        st.text_area(
            "Chọn tất cả (Ctrl+A) rồi copy (Ctrl+C):",
            value=copy_text,
            height=400,
            key="analytics_copy_text"
        )
        st.caption("💡 Click vào text area → Ctrl+A → Ctrl+C để copy toàn bộ")
    
    # === Milestones ===
    st.divider()
    st.subheader("🏆 Milestones")
    
    milestone_cols = st.columns(len(milestones) if len(milestones) <= 6 else 6)
    for i, milestone in enumerate(milestones[:6]):
        with milestone_cols[i % 6]:
            st.markdown(f"""
            **{milestone['icon']} {milestone['type']}**  
            Cycle #{milestone['cycle']}  
            **{milestone['value']}**  
            _{milestone['date']}_
            """)
    
    # === Trend Analysis ===
    st.divider()
    st.subheader("📊 Xu hướng hiện tại")
    
    trend_col1, trend_col2, trend_col3, trend_col4 = st.columns(4)
    
    with trend_col1:
        trend = analyze_trend(df, "return_pct", 30)
        _render_trend_indicator(trend["trend"], f"Return: {trend['description']}")
    
    with trend_col2:
        trend = analyze_trend(df, "win_rate", 30)
        _render_trend_indicator(trend["trend"], f"Win Rate: {trend['description']}")
    
    with trend_col3:
        trend = analyze_trend(df, "val_loss", 30)
        # Invert logic cho val_loss (giảm = tốt)
        inverted_trend = "up" if trend["trend"] == "down" else ("down" if trend["trend"] == "up" else "stable")
        _render_trend_indicator(inverted_trend, f"val_loss: {trend['description']}")
    
    with trend_col4:
        trend = analyze_trend(df, "sharpe", 30)
        _render_trend_indicator(trend["trend"], f"Sharpe: {trend['description']}")
    
    # === Charts ===
    st.divider()
    st.subheader("📈 Biểu đồ chi tiết")
    
    # Tabs cho các biểu đồ
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Tổng quan", "Return", "Win Rate", "val_loss", "RL Metrics", "Sharpe"
    ])
    
    with tab1:
        st.plotly_chart(_create_combined_overview(df), use_container_width=True)
    
    with tab2:
        st.plotly_chart(_create_return_chart(df), use_container_width=True)
    
    with tab3:
        st.plotly_chart(_create_win_rate_chart(df), use_container_width=True)
    
    with tab4:
        st.plotly_chart(_create_val_loss_chart(df), use_container_width=True)
    
    with tab5:
        st.plotly_chart(_create_rl_metrics_chart(df), use_container_width=True)
    
    with tab6:
        st.plotly_chart(_create_sharpe_chart(df), use_container_width=True)
    
    # === AI Insights ===
    st.divider()
    st.subheader("🧠 AI Insights - Phân tích & Diễn giải")
    
    # insights đã được tạo ở trên (cho nút copy)
    for insight in insights:
        st.markdown(insight)
        st.markdown("---")
    
    # === Improvement Phases ===
    if phases:
        st.divider()
        st.subheader("🔥 Các giai đoạn cải tiến đáng kể")
        
        phases_df = pd.DataFrame(phases)
        phases_df = phases_df.rename(columns={
            "phase": "Giai đoạn",
            "cycles": "Cycles",
            "return_improvement_pct": "Return +%",
            "win_rate_improvement": "WR +%",
            "avg_return": "Avg Return",
            "avg_win_rate": "Avg WR",
            "reason": "Nguyên nhân"
        })
        
        st.dataframe(
            phases_df,
            use_container_width=True,
            hide_index=True,
        )
    
    # === Raw Data (expandable) ===
    with st.expander("📋 Xem dữ liệu thô"):
        st.dataframe(
            df[["cycle", "timestamp", "return_pct", "win_rate", "val_loss", 
                "sharpe", "rl_avg_return", "trades", "duration_sec"]].tail(100),
            use_container_width=True,
            hide_index=True,
        )
    
    # Footer
    st.caption(
        f"Dữ liệu từ {stats.get('first_cycle_date', 'N/A')} đến {stats.get('last_cycle_date', 'N/A')} | "
        f"Tổng thời gian training: {stats.get('total_training_hours', 0):.0f} giờ"
    )


# Entry point cho st.Page
page_training_analytics()
