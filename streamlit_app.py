import sys, os, traceback
from pathlib import Path
import streamlit as st

_root = Path(__file__).resolve().parent

for _p in (str(_root), str(_root / "demo")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(str(_root))

st.set_page_config(
    page_title="SpindleFlow RL",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner="⏳ Loading SpindleFlow RL — first load ~30 s on CPU…")
def _warm():
    """
    Pre-load every heavy module into sys.modules exactly once.
    cache_resource flushes the spinner to the browser BEFORE this runs,
    so the user sees feedback and Streamlit stays alive during the 30 s wait.
    Returns None on success, error string on failure.
    """
    try:
        import torch  # noqa: F401
        import numpy  # noqa: F401
        import gymnasium  # noqa: F401
        import stable_baselines3  # noqa: F401
        import sb3_contrib  # noqa: F401
        import sentence_transformers  # noqa: F401
        import plotly  # noqa: F401
        from env.spindleflow_env import SpindleFlowEnv  # noqa: F401
        from env.state import EpisodeState  # noqa: F401
        from env.specialist_registry import SpecialistRegistry  # noqa: F401
        from orchestrator_widget import render_orchestrator  # noqa: F401
        return None
    except Exception as exc:
        return traceback.format_exc()


_err = _warm()

if _err:
    st.error("SpindleFlow failed to import dependencies.")
    st.code(_err)
else:
    _real_spc = st.set_page_config
    st.set_page_config = lambda *a, **kw: None
    _demo = _root / "demo" / "streamlit_app.py"
    try:
        exec(
            compile(_demo.read_text(encoding="utf-8"), str(_demo), "exec"),
            {"__file__": str(_demo), "__name__": "__main__"},
        )
    except SystemExit:
        pass
    except Exception as exc:
        st.error(f"SpindleFlow demo crashed: {exc}")
        st.code(traceback.format_exc())
    finally:
        st.set_page_config = _real_spc
