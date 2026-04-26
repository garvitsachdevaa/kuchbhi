import sys
from pathlib import Path

# Ensure the project root is on sys.path so all local packages are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Entry point — delegate to the full demo app
from demo.streamlit_app import main
main()
