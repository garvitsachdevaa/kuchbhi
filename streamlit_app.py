import sys, os, traceback
from pathlib import Path
import streamlit as st

print("[SpindleFlow] wrapper starting", flush=True)

_root = Path(__file__).resolve().parent

# Page config fires before anything else so the browser gets a live response.
st.set_page_config(
    page_title="SpindleFlow RL",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Show a visible loading message while the heavy ML imports run (~20-30 s).
_banner = st.empty()
_banner.info("⏳ Loading SpindleFlow RL — first load takes ~30 s on CPU…")

# Silence the duplicate set_page_config inside demo/streamlit_app.py.
_real_spc = st.set_page_config
st.set_page_config = lambda *a, **kw: None

for _p in (str(_root), str(_root / "demo")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(str(_root))

_demo = _root / "demo" / "streamlit_app.py"
print(f"[SpindleFlow] running exec on {_demo}", flush=True)

try:
    exec(
        compile(_demo.read_text(encoding="utf-8"), str(_demo), "exec"),
        {"__file__": str(_demo), "__name__": "__main__"},
    )
    print("[SpindleFlow] exec completed OK", flush=True)
    _banner.empty()
except SystemExit:
    _banner.empty()
except Exception as exc:
    print(f"[SpindleFlow] ERROR: {exc}", file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)
    _banner.empty()
    st.error(f"SpindleFlow failed to load: {exc}")
    st.code(traceback.format_exc())
finally:
    st.set_page_config = _real_spc
