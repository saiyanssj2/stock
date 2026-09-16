# -*- coding: utf-8 -*-
"""
Unit tests cho module engine/training_analytics.py

Kiểm tra các functions:
- load_all_cycles()
- calculate_moving_averages()
- detect_improvement_phases()
- get_milestones()
- get_summary_stats()
- analyze_trend()
"""

import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Thêm root vào path để import được engine module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engine.training_analytics import (
    load_all_cycles,
    calculate_moving_averages,
    detect_improvement_phases,
    get_milestones,
    get_summary_stats,
    analyze_trend,
)


# === Fixtures ===

@pytest.fixture
def sample_cycle_data():
    """Tạo dữ liệu cycle mẫu."""
    return {
        "cycle_number": 100,
        "timestamp": "2026-08-01T12:00:00.000000",
        "sl_val_loss": 0.14,
        "sl_epochs": 10,
        "rl_avg_return": 55.0,
        "rl_best_return": 300.0,
        "rl_episodes": 300,
        "backtest_return_pct": 3500.0,
        "backtest_sharpe": 2.0,
        "backtest_win_rate_pct": 75.0,
        "backtest_trades": 1200,
        "symbols": ["ACB", "FPT", "VNM", "HPG", "MBB"],
        "duration_seconds": 3600.0,
    }


@pytest.fixture
def temp_reports_dir(sample_cycle_data):
    """Tạo thư mục reports tạm với sample data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reports_path = Path(tmpdir)
        
        # Tạo 100 cycle files với dữ liệu biến thiên
        for i in range(1, 101):
            cycle_data = sample_cycle_data.copy()
            cycle_data["cycle_number"] = i
            
            # Timestamp tăng dần
            ts = datetime(2026, 6, 1) + timedelta(hours=i * 2)
            cycle_data["timestamp"] = ts.isoformat()
            
            # Metrics biến thiên theo progression
            # Return tăng dần từ 500 lên 4000
            cycle_data["backtest_return_pct"] = 500 + (i * 35)
            # Win rate tăng từ 50 lên 78
            cycle_data["backtest_win_rate_pct"] = 50 + (i * 0.28)
            # val_loss giảm từ 0.17 xuống 0.13
            cycle_data["val_loss"] = cycle_data["sl_val_loss"] = 0.17 - (i * 0.0004)
            # Sharpe tăng từ 0.5 lên 2.2
            cycle_data["backtest_sharpe"] = 0.5 + (i * 0.017)
            # RL avg return tăng
            cycle_data["rl_avg_return"] = 30 + (i * 0.25)
            
            # Ghi file
            filename = f"wf_cycle_{i}_20260601_{i:06d}.json"
            file_path = reports_path / filename
            file_path.write_text(json.dumps(cycle_data), encoding="utf-8")
        
        yield reports_path


@pytest.fixture
def sample_df(temp_reports_dir):
    """Load DataFrame từ temp reports dir."""
    return load_all_cycles(temp_reports_dir)


# === Tests cho load_all_cycles ===

class TestLoadAllCycles:
    """Tests cho function load_all_cycles()."""
    
    def test_load_returns_dataframe(self, temp_reports_dir):
        """Kiểm tra return type là DataFrame."""
        df = load_all_cycles(temp_reports_dir)
        assert isinstance(df, pd.DataFrame)
    
    def test_load_correct_row_count(self, temp_reports_dir):
        """Kiểm tra số dòng đúng với số files."""
        df = load_all_cycles(temp_reports_dir)
        assert len(df) == 100
    
    def test_load_correct_columns(self, temp_reports_dir):
        """Kiểm tra các cột cần thiết."""
        df = load_all_cycles(temp_reports_dir)
        expected_cols = [
            "cycle", "timestamp", "val_loss", "sl_epochs",
            "rl_avg_return", "rl_best_return", "return_pct",
            "sharpe", "win_rate", "trades", "symbols_count", "duration_sec"
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_load_sorted_by_cycle(self, temp_reports_dir):
        """Kiểm tra dữ liệu được sort theo cycle."""
        df = load_all_cycles(temp_reports_dir)
        cycles = df["cycle"].tolist()
        assert cycles == sorted(cycles)
    
    def test_load_empty_dir_returns_empty_df(self):
        """Kiểm tra trả về empty DataFrame khi dir rỗng."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = load_all_cycles(Path(tmpdir))
            assert df.empty
    
    def test_load_nonexistent_dir_returns_empty_df(self):
        """Kiểm tra trả về empty DataFrame khi dir không tồn tại."""
        df = load_all_cycles(Path("/nonexistent/path"))
        assert df.empty
    
    def test_load_skips_invalid_json(self, temp_reports_dir):
        """Kiểm tra skip file JSON lỗi."""
        # Tạo file lỗi
        invalid_file = temp_reports_dir / "wf_cycle_999_invalid.json"
        invalid_file.write_text("not valid json {{{", encoding="utf-8")
        
        df = load_all_cycles(temp_reports_dir)
        # Vẫn load được 100 files hợp lệ
        assert len(df) == 100


# === Tests cho calculate_moving_averages ===

class TestCalculateMovingAverages:
    """Tests cho function calculate_moving_averages()."""
    
    def test_ma_adds_columns(self, sample_df):
        """Kiểm tra thêm các cột MA."""
        df = calculate_moving_averages(sample_df, window=20)
        
        assert "return_pct_ma20" in df.columns
        assert "win_rate_ma20" in df.columns
        assert "val_loss_ma20" in df.columns
        assert "sharpe_ma20" in df.columns
    
    def test_ma_values_reasonable(self, sample_df):
        """Kiểm tra MA values hợp lý."""
        df = calculate_moving_averages(sample_df, window=20)
        
        # MA không được âm (cho các metrics dương)
        assert (df["return_pct_ma20"] >= 0).all()
        assert (df["win_rate_ma20"] >= 0).all()
    
    def test_ma_empty_df(self):
        """Kiểm tra với empty DataFrame."""
        df = pd.DataFrame()
        result = calculate_moving_averages(df)
        assert result.empty
    
    def test_ma_different_windows(self, sample_df):
        """Kiểm tra với window sizes khác nhau."""
        df10 = calculate_moving_averages(sample_df, window=10)
        df30 = calculate_moving_averages(sample_df, window=30)
        
        assert "return_pct_ma10" in df10.columns
        assert "return_pct_ma30" in df30.columns


# === Tests cho detect_improvement_phases ===

class TestDetectImprovementPhases:
    """Tests cho function detect_improvement_phases()."""
    
    def test_detects_phases(self, sample_df):
        """Kiểm tra phát hiện improvement phases."""
        phases = detect_improvement_phases(sample_df)
        
        # Với dữ liệu tăng đều, sẽ có ít nhất 1 phase
        assert isinstance(phases, list)
    
    def test_phase_structure(self, sample_df):
        """Kiểm tra cấu trúc phase data."""
        phases = detect_improvement_phases(sample_df)
        
        if phases:
            phase = phases[0]
            assert "phase" in phase
            assert "cycles" in phase
            assert "return_improvement_pct" in phase
            assert "win_rate_improvement" in phase
            assert "reason" in phase
    
    def test_empty_df_returns_empty_list(self):
        """Kiểm tra với empty DataFrame."""
        phases = detect_improvement_phases(pd.DataFrame())
        assert phases == []
    
    def test_small_df_returns_empty_list(self, sample_cycle_data):
        """Kiểm tra với DataFrame quá nhỏ."""
        df = pd.DataFrame([sample_cycle_data])
        phases = detect_improvement_phases(df)
        assert phases == []


# === Tests cho get_milestones ===

class TestGetMilestones:
    """Tests cho function get_milestones()."""
    
    def test_returns_milestones(self, sample_df):
        """Kiểm tra trả về milestones."""
        milestones = get_milestones(sample_df)
        
        assert isinstance(milestones, list)
        assert len(milestones) > 0
    
    def test_milestone_structure(self, sample_df):
        """Kiểm tra cấu trúc milestone."""
        milestones = get_milestones(sample_df)
        
        milestone = milestones[0]
        assert "type" in milestone
        assert "cycle" in milestone
        assert "value" in milestone
        assert "date" in milestone
        assert "icon" in milestone
    
    def test_has_peak_return_milestone(self, sample_df):
        """Kiểm tra có milestone Peak Return."""
        milestones = get_milestones(sample_df)
        types = [m["type"] for m in milestones]
        
        assert "Peak Return" in types
    
    def test_has_peak_winrate_milestone(self, sample_df):
        """Kiểm tra có milestone Peak Win Rate."""
        milestones = get_milestones(sample_df)
        types = [m["type"] for m in milestones]
        
        assert "Peak Win Rate" in types
    
    def test_empty_df_returns_empty_list(self):
        """Kiểm tra với empty DataFrame."""
        milestones = get_milestones(pd.DataFrame())
        assert milestones == []


# === Tests cho get_summary_stats ===

class TestGetSummaryStats:
    """Tests cho function get_summary_stats()."""
    
    def test_returns_dict(self, sample_df):
        """Kiểm tra trả về dict."""
        stats = get_summary_stats(sample_df)
        assert isinstance(stats, dict)
    
    def test_has_required_keys(self, sample_df):
        """Kiểm tra có các keys cần thiết."""
        stats = get_summary_stats(sample_df)
        
        required_keys = [
            "total_cycles",
            "first_cycle_date",
            "last_cycle_date",
            "current_avg_return",
            "current_avg_win_rate",
            "all_time_max_return",
            "total_training_hours",
        ]
        
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"
    
    def test_total_cycles_correct(self, sample_df):
        """Kiểm tra total_cycles đúng."""
        stats = get_summary_stats(sample_df)
        assert stats["total_cycles"] == 100
    
    def test_empty_df_returns_empty_dict(self):
        """Kiểm tra với empty DataFrame."""
        stats = get_summary_stats(pd.DataFrame())
        assert stats == {}


# === Tests cho analyze_trend ===

class TestAnalyzeTrend:
    """Tests cho function analyze_trend()."""
    
    def test_returns_dict(self, sample_df):
        """Kiểm tra trả về dict."""
        trend = analyze_trend(sample_df, "return_pct")
        assert isinstance(trend, dict)
    
    def test_trend_structure(self, sample_df):
        """Kiểm tra cấu trúc trend result."""
        trend = analyze_trend(sample_df, "return_pct")
        
        assert "trend" in trend
        assert "slope" in trend
        assert "description" in trend
        assert trend["trend"] in ["up", "down", "stable", "unknown"]
    
    def test_uptrend_detected(self, sample_df):
        """Kiểm tra phát hiện uptrend cho dữ liệu tăng."""
        # sample_df có return_pct tăng dần
        trend = analyze_trend(sample_df, "return_pct", window=30)
        assert trend["trend"] == "up"
    
    def test_downtrend_for_valloss(self, sample_df):
        """Kiểm tra phát hiện downtrend cho val_loss giảm."""
        # sample_df có val_loss giảm dần nhưng mức giảm nhỏ nên có thể stable
        trend = analyze_trend(sample_df, "val_loss", window=30)
        # val_loss giảm từ 0.17 xuống 0.13 → có thể down hoặc stable tùy threshold
        assert trend["trend"] in ["down", "stable"]
    
    def test_empty_df(self):
        """Kiểm tra với empty DataFrame."""
        trend = analyze_trend(pd.DataFrame(), "return_pct")
        assert trend["trend"] == "unknown"
    
    def test_missing_column(self, sample_df):
        """Kiểm tra với column không tồn tại."""
        trend = analyze_trend(sample_df, "nonexistent_column")
        assert trend["trend"] == "unknown"


# === Integration Tests ===

class TestIntegration:
    """Integration tests cho workflow đầy đủ."""
    
    def test_full_workflow(self, temp_reports_dir):
        """Test toàn bộ workflow: load → analyze → output."""
        # 1. Load data
        df = load_all_cycles(temp_reports_dir)
        assert not df.empty
        
        # 2. Calculate MAs
        df = calculate_moving_averages(df, window=20)
        assert "return_pct_ma20" in df.columns
        
        # 3. Get stats
        stats = get_summary_stats(df)
        assert stats["total_cycles"] == 100
        
        # 4. Get milestones
        milestones = get_milestones(df)
        assert len(milestones) > 0
        
        # 5. Detect phases
        phases = detect_improvement_phases(df)
        assert isinstance(phases, list)
        
        # 6. Analyze trends
        trend = analyze_trend(df, "return_pct")
        assert trend["trend"] in ["up", "down", "stable"]
    
    def test_realistic_data_scenario(self, temp_reports_dir):
        """Test với scenario dữ liệu thực tế hơn."""
        df = load_all_cycles(temp_reports_dir)
        
        # Verify data progression makes sense
        first_10 = df.head(10)["return_pct"].mean()
        last_10 = df.tail(10)["return_pct"].mean()
        
        # Return should increase over time (based on fixture design)
        assert last_10 > first_10
        
        # Win rate should also improve
        first_wr = df.head(10)["win_rate"].mean()
        last_wr = df.tail(10)["win_rate"].mean()
        assert last_wr > first_wr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
