import streamlit as st
import sys, os, traceback
from pathlib import Path

# Set page config immediately so the browser gets a response before the
# heavy ML imports below.  demo/streamlit_app.py also calls set_page_config
# at module level — we monkey-patch it to a no-op so the duplicate call
# (on first import) doesn't raise a StreamlitAPIException.
st.set_page_config(
    page_title="SpindleFlow RL",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)
_orig_spc = st.set_page_config
st.set_page_config = lambda *a, **kw: None   # silence double-call from demo

root = Path(__file__).resolve().parent
demo_dir = root / "demo"
for p in (str(root), str(demo_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(str(root))


@st.cache_resource(show_spinner="Loading SpindleFlow RL…")
def _load_app():
    """Cache the demo module so the 10–20 s ML import only happens once."""
    import demo.streamlit_app as _app   # noqa: PLC0415
    return _app


try:
    st.set_page_config = _orig_spc      # restore before main() uses st.*
    _app = _load_app()
    _app.main()
except SystemExit:
    pass
except Exception as exc:
    st.error(f"SpindleFlow failed to load: {exc}")
    st.code(traceback.format_exc())
