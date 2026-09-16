"""
Unit tests cho app.py router và navigation structure.
Kiểm tra cấu hình pages, imports.
"""
import ast
import importlib
import os
import sys

import pytest


# Đường dẫn project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAppStructure:
    """Test cấu trúc file app.py."""

    def test_app_py_exists(self) -> None:
        """app.py phải tồn tại ở project root."""
        app_path = os.path.join(PROJECT_ROOT, "app.py")
        assert os.path.exists(app_path), "app.py không tồn tại"

    def test_app_py_valid_syntax(self) -> None:
        """app.py phải có syntax hợp lệ."""
        app_path = os.path.join(PROJECT_ROOT, "app.py")
        with open(app_path, encoding="utf-8") as f:
            source = f.read()
        # Nếu parse thành công → syntax OK
        ast.parse(source)

    def test_app_py_uses_set_page_config(self) -> None:
        """app.py phải gọi st.set_page_config với layout='wide'."""
        app_path = os.path.join(PROJECT_ROOT, "app.py")
        with open(app_path, encoding="utf-8") as f:
            source = f.read()
        assert "set_page_config" in source
        assert 'layout="wide"' in source or "layout='wide'" in source

    def test_app_py_uses_st_navigation(self) -> None:
        """app.py phải dùng st.navigation cho multi-page routing."""
        app_path = os.path.join(PROJECT_ROOT, "app.py")
        with open(app_path, encoding="utf-8") as f:
            source = f.read()
        assert "st.navigation" in source
        assert "st.Page" in source

    def test_app_py_defines_all_required_pages(self) -> None:
        """app.py phải define đủ 4 pages: Training, Backtest, Recommendations, Settings."""
        app_path = os.path.join(PROJECT_ROOT, "app.py")
        with open(app_path, encoding="utf-8") as f:
            source = f.read()
        required_pages = ["Training", "Backtest", "Recommendations", "Settings"]
        for page in required_pages:
            assert page in source, f"Page '{page}' không được define trong app.py"


class TestPageFiles:
    """Test các page files tồn tại và có syntax hợp lệ."""

    PAGE_FILES = [
        "ui/pages/page_training.py",
        "ui/pages/page_backtest.py",
        "ui/pages/page_recommendations.py",
        "ui/pages/page_settings.py",
    ]

    @pytest.mark.parametrize("page_file", PAGE_FILES)
    def test_page_file_exists(self, page_file: str) -> None:
        """Mỗi page file phải tồn tại."""
        full_path = os.path.join(PROJECT_ROOT, page_file)
        assert os.path.exists(full_path), f"{page_file} không tồn tại"

    @pytest.mark.parametrize("page_file", PAGE_FILES)
    def test_page_file_valid_syntax(self, page_file: str) -> None:
        """Mỗi page file phải có syntax hợp lệ."""
        full_path = os.path.join(PROJECT_ROOT, page_file)
        with open(full_path, encoding="utf-8") as f:
            source = f.read()
        ast.parse(source)

    @pytest.mark.parametrize("page_file", PAGE_FILES)
    def test_page_file_imports_streamlit(self, page_file: str) -> None:
        """Mỗi page file phải import streamlit."""
        full_path = os.path.join(PROJECT_ROOT, page_file)
        with open(full_path, encoding="utf-8") as f:
            source = f.read()
        assert "import streamlit" in source or "from streamlit" in source
