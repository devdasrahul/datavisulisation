"""
pipeline/run_pipeline.py
─────────────────────────
CLI entry point — runs ingest → transform → load → quality checks in sequence.

Usage:
  python pipeline/run_pipeline.py [--batch-date YYYY-MM-DD] [--dry-run]

Flags:
  --batch-date  Synthetic date to replay (default: today)
  --dry-run     Process data but skip all DB writes (useful for testing)

TODO (Phase 2):
  - Wire up ingest, transform, load, quality as sequential steps
  - Write pipeline start/end timestamps to pipeline_run_log
  - Exit with code 1 on any step failure
"""

import argparse
import sys
from datetime import date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the predictive maintenance ETL pipeline."
    )
    parser.add_argument(
        "--batch-date",
        type=date.fromisoformat,
        default=date.today(),
        help="Synthetic batch date to replay (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process data locally but skip all database writes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[pipeline] batch_date={args.batch_date}  dry_run={args.dry_run}")
    print(
        "[pipeline] TODO: implement ingest -> transform -> load -> quality steps (Phase 2)"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
