import sys, os, traceback
from pathlib import Path
import streamlit as st

# ── 1. Page config fires immediately so the browser gets a live response
#       before the 10-30 s ML imports below.
st.set_page_config(
    page_title="SpindleFlow RL",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 2. Silence the duplicate set_page_config call inside demo/streamlit_app.py
#       (it also has one at module level — would raise StreamlitAPIException).
_real_spc = st.set_page_config
st.set_page_config = lambda *a, **kw: None

# ── 3. Make env.*, reward.*, agents.* and orchestrator_widget importable.
_root = Path(__file__).resolve().parent
for _p in (str(_root), str(_root / "demo")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(str(_root))

# ── 4. Run demo/streamlit_app.py exactly as __main__.
#       Its st.set_page_config() call is now a no-op; main() draws the UI.
_demo = _root / "demo" / "streamlit_app.py"
try:
    exec(
        compile(_demo.read_text(encoding="utf-8"), str(_demo), "exec"),
        {"__file__": str(_demo), "__name__": "__main__"},
    )
except SystemExit:
    pass
except Exception as exc:
    st.error(f"SpindleFlow failed to load: {exc}")
    st.code(traceback.format_exc())
finally:
    st.set_page_config = _real_spc   # restore for subsequent Streamlit re-runs
