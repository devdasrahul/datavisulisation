"""
pipeline/quality.py
-------------------
Thin shim — delegates to the real implementation in quality/run_checks.py.

Usage:
    python pipeline/quality.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quality.run_checks import main  # noqa: E402

if __name__ == "__main__":
    main()
