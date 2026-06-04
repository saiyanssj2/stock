import streamlit as st
import os
import pandas as pd
from market import get_world_indices, get_vn_indices
from ui_market import show_market
from ui_vnindex import show_vnindex
from ui_analyze import show_analyze
from ui_scanner import show_scanner
from ui_update import show_update
from ui_wyckoff_edu import show_wyckoff_edu

st.set_page_config(page_title="Phan Tich Ky Thuat", layout="wide")
st.title("📈 Phân Tích Kỹ Thuật Chứng Khoán")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data(ttl=10800)
def load_world_indices():
    return get_world_indices()


@st.cache_data(ttl=10800)
def load_vn_indices():
    return get_vn_indices()


@st.cache_data(ttl=10800)
def load_vnindex_analysis():
    for fname in ['VNINDEX_full.csv', 'VNINDEX.csv']:
        path = os.path.join(BASE_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['time'] = pd.to_datetime(df['time'])
            return df.sort_values('time').reset_index(drop=True)
    return None


# --- Thi truong ---
show_market(load_world_indices, load_vn_indices, load_vnindex_analysis)

st.divider()

# --- VNINDEX Wyckoff ---
show_vnindex(load_vnindex_analysis)

st.divider()

# --- Tabs ---
tab_analyze, tab_scanner, tab_wyckoff, tab_update = st.tabs([
    "🤖 Phân tích kỹ thuật chi tiết",
    "🔍 Bộ lọc cổ phiếu",
    "📚 Kiến thức Wyckoff",
    "📥 Cập nhật dữ liệu"
])

with tab_analyze:
    show_analyze(BASE_DIR)

with tab_scanner:
    show_scanner(BASE_DIR)

with tab_wyckoff:
    show_wyckoff_edu()

with tab_update:
    show_update(BASE_DIR)
