"""
pipeline/transform.py
----------------------
Thin shim — delegates to the real implementation in transform/transform_batches.py.

Usage:
    python pipeline/transform.py [--dataset FD001] [--force] [--dry-run]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transform.transform_batches import main  # noqa: E402

if __name__ == "__main__":
    main()
