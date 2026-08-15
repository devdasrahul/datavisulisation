"""
scripts/download_data.py
─────────────────────────
Thin shim — delegates to the real implementation in ingest/download_data.py.

Usage:
  python scripts/download_data.py [--dest PATH] [--dataset FD001 FD002 ...]
"""

# Make `python scripts/download_data.py` work from the project root
# by adding the project root to sys.path before importing the ingest package.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.download_data import main  # noqa: E402

if __name__ == "__main__":
    main()
