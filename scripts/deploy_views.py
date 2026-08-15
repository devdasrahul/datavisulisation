"""
scripts/deploy_views.py
────────────────────────
Deploy analytical SQL views to Postgres.
Runs sql/views.sql as CREATE OR REPLACE VIEW statements.
Idempotent — safe to re-run after any view changes.

Usage:
  python scripts/deploy_views.py
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def main():
    db_url = os.environ.get("DATABASE_URL", "").strip()
    if not db_url:
        log.error("DATABASE_URL environment variable is not set.")
        sys.exit(1)

    engine = create_engine(db_url)
    views_sql_path = PROJECT_ROOT / "sql" / "views.sql"

    if not views_sql_path.exists():
        log.error(f"Cannot find views file: {views_sql_path}")
        sys.exit(1)

    sql_script = views_sql_path.read_text(encoding="utf-8")

    # Simple split by ';' works for these DDL statements
    statements = [s.strip() for s in sql_script.split(";") if s.strip()]

    log.info("=" * 60)
    log.info(f"Deploying {len(statements)} view statements from sql/views.sql")
    log.info("=" * 60)

    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))

    log.info("Views deployed successfully.")


if __name__ == "__main__":
    main()
