import sys, os, traceback
from pathlib import Path

# Make env.*, reward.*, agents.* (from repo root) and orchestrator_widget
# (from demo/) importable before demo/streamlit_app.py starts.
_root = Path(__file__).resolve().parent
for _p in (str(_root), str(_root / "demo")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(str(_root))

import streamlit as st   # needed only for the error fallback below

_demo = _root / "demo" / "streamlit_app.py"
try:
    # Run demo/streamlit_app.py exactly as if it were the entry-point script.
    # __file__ must point to the demo file so its own Path(__file__) logic
    # resolves paths correctly (e.g. demo/assets, orchestrator_widget).
    exec(
        compile(_demo.read_text(encoding="utf-8"), str(_demo), "exec"),
        {"__file__": str(_demo), "__name__": "__main__"},
    )
except SystemExit:
    pass
except Exception as exc:
    st.error(f"SpindleFlow failed to load: {exc}")
    st.code(traceback.format_exc())
