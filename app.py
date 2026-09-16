"""
Stock Trading Platform - Multi-page Streamlit Application.

Router chính dùng st.navigation() để điều hướng giữa các pages.
Training chỉ chạy khi user nhấn Start trong trang Training.
"""
import json
import logging
import threading
from datetime import datetime
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)

# Log app start
_startup_log = Path("pipeline_debug.log")
with open(_startup_log, "a", encoding="utf-8") as _f:
    _f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] APP SCRIPT LOADED (Streamlit render)\n")

# Cấu hình page - phải gọi đầu tiên
st.set_page_config(
    page_title="Stock Trading Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === Hiển thị trạng thái pipeline (nếu đang chạy) ===
_status_file = Path("data/engine/status/auto_pipeline_status.json")
if _status_file.exists():
    try:
        _status = json.loads(_status_file.read_text(encoding="utf-8"))
        if _status.get("state") == "running":
            st.info(f"⏳ {_status.get('message', 'Pipeline đang chạy...')}", icon="🔄")
    except Exception:
        pass


# ==============================================================================
# Auto-refresh: nếu pipeline đang chạy → rerun mỗi 10s
# ==============================================================================
_pipeline_running = False
if _status_file.exists():
    try:
        _ps = json.loads(_status_file.read_text(encoding="utf-8"))
        _pipeline_running = _ps.get("state") == "running"
    except Exception:
        pass

if _pipeline_running:
    @st.fragment(run_every=10)
    def _auto_refresh():
        """Fragment tự refresh mỗi 10s khi pipeline đang chạy."""
        pass
    _auto_refresh()


# ==============================================================================
# Navigation
# ==============================================================================
pages = [
    st.Page("ui/pages/page_training.py", title="Training", icon="🧠", default=True),
    st.Page("ui/pages/page_training_analytics.py", title="Analytics", icon="📊"),
    st.Page("ui/pages/page_backtest.py", title="Backtest", icon="📈"),
    st.Page("ui/pages/page_recommendations.py", title="Recommendations", icon="⭐"),
    st.Page("ui/pages/page_settings.py", title="Settings", icon="⚙️"),
]

nav = st.navigation(pages)
nav.run()
