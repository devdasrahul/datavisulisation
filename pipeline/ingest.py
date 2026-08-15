"""
pipeline/ingest.py
------------------
Thin shim — delegates to the real implementation in ingest/ingest_batches.py.

Usage:
    python pipeline/ingest.py [--dataset FD001] [--batch-size 20] [--dry-run]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.ingest_batches import main  # noqa: E402

if __name__ == "__main__":
    main()
