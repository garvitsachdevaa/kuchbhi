import streamlit as st
import sys, os, traceback
from pathlib import Path
import importlib.util

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
os.chdir(str(root))

try:
    demo_file = root / "demo" / "streamlit_app.py"
    spec = importlib.util.spec_from_file_location("spindleflow_demo", str(demo_file))
    mod  = importlib.util.module_from_spec(spec)
    mod.__file__ = str(demo_file)          # demo's own sys.path logic resolves correctly
    sys.modules["spindleflow_demo"] = mod
    spec.loader.exec_module(mod)           # runs demo/streamlit_app.py in its own context
    mod.main()
except SystemExit:
    pass
except BaseException as e:
    st.error(f"SpindleFlow failed to load: {e}")
    st.code(traceback.format_exc())
