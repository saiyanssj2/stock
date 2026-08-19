"""
Conftest cho test/engine/diagnostics/ — đảm bảo project root trên sys.path.
"""

import os
import sys

# Thêm project root vào sys.path để import engine modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
