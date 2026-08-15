"""
pipeline/load.py
----------------
Thin shim — delegates to the real implementation in load/load_to_postgres.py.

Usage:
    python pipeline/load.py [--dataset FD001] [--force] [--verbose]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from load.load_to_postgres import main  # noqa: E402

if __name__ == "__main__":
    main()
