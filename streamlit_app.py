import sys, os, traceback
import streamlit as st
from pathlib import Path
import importlib.util

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
os.chdir(str(root))

demo_file = root / "demo" / "streamlit_app.py"

try:
    spec = importlib.util.spec_from_file_location("spindleflow_demo", str(demo_file))
    mod  = importlib.util.module_from_spec(spec)
    mod.__file__ = str(demo_file)
    sys.modules["spindleflow_demo"] = mod
    spec.loader.exec_module(mod)
    mod.main()
except Exception as e:
    st.error(f"SpindleFlow failed to load: {e}")
    st.code(traceback.format_exc())
