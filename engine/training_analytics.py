# -*- coding: utf-8 -*-
"""
Training Analytics Module - Load và xử lý dữ liệu training cycles.

Cung cấp các functions để:
- Load tất cả wf_cycle_*.json files
- Tính toán các chỉ số phân tích (moving averages, trends)
- Phát hiện các improvement phases và milestones
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Đường dẫn mặc định đến thư mục reports
REPORTS_DIR = Path("data/engine/reports")


@dataclass
class TrainingCycle:
    """Dữ liệu một training cycle."""
    cycle_number: int
    timestamp: datetime
    sl_val_loss: float
    sl_epochs: int
    rl_avg_return: float
    rl_best_return: float
    rl_episodes: int
    backtest_return_pct: float
    backtest_sharpe: float
    backtest_win_rate_pct: float
    backtest_trades: int
    symbols_count: int
    duration_seconds: float


def load_all_cycles(reports_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load tất cả training cycles từ JSON files.
    
    Args:
        reports_dir: Thư mục chứa wf_cycle_*.json files
        
    Returns:
        DataFrame với các cột metrics theo từng cycle
    """
    if reports_dir is None:
        reports_dir = REPORTS_DIR
    
    reports_dir = Path(reports_dir)
    if not reports_dir.exists():
        return pd.DataFrame()
    
    # Tìm tất cả wf_cycle_*.json files
    cycle_files = sorted(
        reports_dir.glob("wf_cycle_*.json"),
        key=lambda f: int(f.stem.split("_")[2])  # Sort theo cycle number
    )
    
    if not cycle_files:
        return pd.DataFrame()
    
    records = []
    for file_path in cycle_files:
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            
            # Parse timestamp
            ts_str = data.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                timestamp = datetime.now()
            
            # Đếm số symbols
            symbols = data.get("symbols", [])
            symbols_count = len(symbols) if isinstance(symbols, list) else 0
            
            record = {
                "cycle": data.get("cycle_number", 0),
                "timestamp": timestamp,
                "val_loss": data.get("sl_val_loss", 0),
                "sl_epochs": data.get("sl_epochs", 0),
                "rl_avg_return": data.get("rl_avg_return", 0),
                "rl_best_return": data.get("rl_best_return", 0),
                "rl_episodes": data.get("rl_episodes", 0),
                "return_pct": data.get("backtest_return_pct", 0),
                "sharpe": data.get("backtest_sharpe", 0),
                "win_rate": data.get("backtest_win_rate_pct", 0),
                "trades": data.get("backtest_trades", 0),
                "symbols_count": symbols_count,
                "duration_sec": data.get("duration_seconds", 0),
            }
            records.append(record)
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Skip file lỗi, tiếp tục
            continue
    
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    df = df.sort_values("cycle").reset_index(drop=True)
    return df


def calculate_moving_averages(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Tính moving averages cho các metrics chính.
    
    Args:
        df: DataFrame từ load_all_cycles()
        window: Số cycles cho moving average
        
    Returns:
        DataFrame với thêm các cột MA
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Moving averages cho các metrics chính
    metrics = ["return_pct", "win_rate", "val_loss", "rl_avg_return", "sharpe"]
    
    for metric in metrics:
        if metric in df.columns:
            df[f"{metric}_ma{window}"] = df[metric].rolling(window=window, min_periods=1).mean()
    
    return df


def detect_improvement_phases(df: pd.DataFrame, threshold_pct: float = 20.0) -> list[dict]:
    """
    Phát hiện các giai đoạn cải tiến đáng kể của model.
    
    Args:
        df: DataFrame từ load_all_cycles()
        threshold_pct: Ngưỡng % cải tiến để tính là "significant"
        
    Returns:
        List các phases với thông tin chi tiết
    """
    if df.empty or len(df) < 10:
        return []
    
    phases = []
    
    # Chia thành các chunks 50 cycles
    chunk_size = 50
    n_chunks = len(df) // chunk_size
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        
        chunk = df.iloc[start_idx:end_idx]
        
        if len(chunk) < 10:
            continue
        
        # Tính metrics trung bình
        avg_return = chunk["return_pct"].mean()
        avg_win_rate = chunk["win_rate"].mean()
        avg_val_loss = chunk["val_loss"].mean()
        
        # So sánh với chunk trước
        if i > 0:
            prev_chunk = df.iloc[(i-1)*chunk_size : start_idx]
            prev_return = prev_chunk["return_pct"].mean()
            prev_win_rate = prev_chunk["win_rate"].mean()
            
            return_improvement = ((avg_return - prev_return) / max(prev_return, 1)) * 100
            wr_improvement = avg_win_rate - prev_win_rate
            
            if return_improvement > threshold_pct or wr_improvement > 5:
                phases.append({
                    "phase": i + 1,
                    "cycles": f"{chunk['cycle'].min()} - {chunk['cycle'].max()}",
                    "return_improvement_pct": round(return_improvement, 1),
                    "win_rate_improvement": round(wr_improvement, 1),
                    "avg_return": round(avg_return, 1),
                    "avg_win_rate": round(avg_win_rate, 1),
                    "avg_val_loss": round(avg_val_loss, 4),
                    "reason": _infer_improvement_reason(return_improvement, wr_improvement, avg_val_loss)
                })
    
    return phases


def _infer_improvement_reason(return_impr: float, wr_impr: float, val_loss: float) -> str:
    """Suy luận nguyên nhân cải tiến dựa trên metrics."""
    reasons = []
    
    if return_impr > 50:
        reasons.append("Đột phá return do RL policy tốt hơn")
    elif return_impr > 20:
        reasons.append("Return cải thiện nhờ fine-tuning hiệu quả")
    
    if wr_impr > 10:
        reasons.append("Win rate tăng mạnh - model dự đoán chính xác hơn")
    elif wr_impr > 5:
        reasons.append("Win rate ổn định tăng")
    
    if val_loss < 0.135:
        reasons.append("val_loss thấp - SL converge tốt")
    
    if not reasons:
        reasons.append("Cải tiến tổng hợp từ nhiều yếu tố")
    
    return "; ".join(reasons)


def get_milestones(df: pd.DataFrame) -> list[dict]:
    """
    Tìm các milestone quan trọng (peak return, peak WR, etc.)
    
    Args:
        df: DataFrame từ load_all_cycles()
        
    Returns:
        List các milestones
    """
    if df.empty:
        return []
    
    milestones = []
    
    # Peak return
    idx_max_return = df["return_pct"].idxmax()
    row = df.loc[idx_max_return]
    milestones.append({
        "type": "Peak Return",
        "cycle": int(row["cycle"]),
        "value": f"{row['return_pct']:.1f}%",
        "date": row["timestamp"].strftime("%Y-%m-%d"),
        "icon": "🏆"
    })
    
    # Peak win rate
    idx_max_wr = df["win_rate"].idxmax()
    row = df.loc[idx_max_wr]
    milestones.append({
        "type": "Peak Win Rate",
        "cycle": int(row["cycle"]),
        "value": f"{row['win_rate']:.1f}%",
        "date": row["timestamp"].strftime("%Y-%m-%d"),
        "icon": "🎯"
    })
    
    # Best val_loss (lowest)
    idx_min_loss = df["val_loss"].idxmin()
    row = df.loc[idx_min_loss]
    milestones.append({
        "type": "Best val_loss",
        "cycle": int(row["cycle"]),
        "value": f"{row['val_loss']:.4f}",
        "date": row["timestamp"].strftime("%Y-%m-%d"),
        "icon": "📉"
    })
    
    # Best Sharpe
    idx_max_sharpe = df["sharpe"].idxmax()
    row = df.loc[idx_max_sharpe]
    milestones.append({
        "type": "Best Sharpe",
        "cycle": int(row["cycle"]),
        "value": f"{row['sharpe']:.3f}",
        "date": row["timestamp"].strftime("%Y-%m-%d"),
        "icon": "📊"
    })
    
    # First cycle vượt 1000% return
    above_1000 = df[df["return_pct"] >= 1000]
    if not above_1000.empty:
        first_1000 = above_1000.iloc[0]
        milestones.append({
            "type": "First 1000%+ Return",
            "cycle": int(first_1000["cycle"]),
            "value": f"{first_1000['return_pct']:.1f}%",
            "date": first_1000["timestamp"].strftime("%Y-%m-%d"),
            "icon": "🚀"
        })
    
    # First cycle vượt 70% WR
    above_70wr = df[df["win_rate"] >= 70]
    if not above_70wr.empty:
        first_70 = above_70wr.iloc[0]
        milestones.append({
            "type": "First 70%+ Win Rate",
            "cycle": int(first_70["cycle"]),
            "value": f"{first_70['win_rate']:.1f}%",
            "date": first_70["timestamp"].strftime("%Y-%m-%d"),
            "icon": "✅"
        })
    
    return milestones


def get_summary_stats(df: pd.DataFrame) -> dict:
    """
    Tính toán summary statistics.
    
    Args:
        df: DataFrame từ load_all_cycles()
        
    Returns:
        Dict với các thống kê tổng hợp
    """
    if df.empty:
        return {}
    
    # Lấy 50 cycles gần nhất để so sánh với 50 cycles trước đó
    recent_50 = df.tail(50)
    prev_50 = df.iloc[-100:-50] if len(df) >= 100 else df.head(50)
    
    return {
        "total_cycles": len(df),
        "first_cycle_date": df["timestamp"].min().strftime("%Y-%m-%d"),
        "last_cycle_date": df["timestamp"].max().strftime("%Y-%m-%d"),
        
        # Current performance (last 50 cycles)
        "current_avg_return": round(recent_50["return_pct"].mean(), 1),
        "current_avg_win_rate": round(recent_50["win_rate"].mean(), 1),
        "current_avg_val_loss": round(recent_50["val_loss"].mean(), 4),
        "current_avg_sharpe": round(recent_50["sharpe"].mean(), 3),
        
        # All-time
        "all_time_max_return": round(df["return_pct"].max(), 1),
        "all_time_max_win_rate": round(df["win_rate"].max(), 1),
        "all_time_min_val_loss": round(df["val_loss"].min(), 4),
        "all_time_max_sharpe": round(df["sharpe"].max(), 3),
        
        # Improvement vs previous 50
        "return_change_vs_prev": round(recent_50["return_pct"].mean() - prev_50["return_pct"].mean(), 1),
        "wr_change_vs_prev": round(recent_50["win_rate"].mean() - prev_50["win_rate"].mean(), 1),
        
        # Training stats
        "total_training_hours": round(df["duration_sec"].sum() / 3600, 1),
        "avg_cycle_duration_min": round(df["duration_sec"].mean() / 60, 1),
    }


def analyze_trend(df: pd.DataFrame, metric: str, window: int = 30) -> dict:
    """
    Phân tích xu hướng của một metric.
    
    Args:
        df: DataFrame từ load_all_cycles()
        metric: Tên cột metric cần phân tích
        window: Số cycles để tính trend
        
    Returns:
        Dict với trend analysis
    """
    if df.empty or metric not in df.columns or len(df) < window:
        return {"trend": "unknown", "slope": 0, "description": "Không đủ dữ liệu"}
    
    recent = df[metric].tail(window).values
    x = np.arange(len(recent))
    
    # Linear regression để tính slope
    slope = np.polyfit(x, recent, 1)[0]
    
    # Tính % change
    start_val = recent[0]
    end_val = recent[-1]
    pct_change = ((end_val - start_val) / max(abs(start_val), 1)) * 100
    
    # Xác định trend
    if slope > 0 and pct_change > 5:
        trend = "up"
        description = f"Đang tăng (+{pct_change:.1f}% trong {window} cycles)"
    elif slope < 0 and pct_change < -5:
        trend = "down"
        description = f"Đang giảm ({pct_change:.1f}% trong {window} cycles)"
    else:
        trend = "stable"
        description = f"Ổn định ({pct_change:+.1f}% trong {window} cycles)"
    
    return {
        "trend": trend,
        "slope": round(slope, 4),
        "pct_change": round(pct_change, 1),
        "description": description,
        "recent_avg": round(np.mean(recent), 2),
        "recent_std": round(np.std(recent), 2),
    }
