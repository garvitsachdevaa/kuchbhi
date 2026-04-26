import streamlit as st
import sys, os, traceback
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
os.chdir(str(root))

try:
    demo = root / "demo" / "streamlit_app.py"
    with open(demo, encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, str(demo), "exec"), {"__file__": str(demo), "__name__": "__main__"})
except SystemExit:
    pass
except BaseException as e:
    st.error(f"SpindleFlow failed to load: {e}")
    st.code(traceback.format_exc())
