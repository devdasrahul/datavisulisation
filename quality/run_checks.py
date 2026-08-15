"""
quality/run_checks.py

Phase 3 — Quality Assurance

Runs a suite of data quality checks against Postgres to make sure the ETL actually worked.
It logs everything into `pipeline_run_log` so we have a paper trail, and crashes the GitHub 
Action (exit code 1) if anything critical fails.

Checks:
  1. Freshness: Are we getting data every 24 hours like we expect?
  2. Cycle Gaps: Did we skip any cycles for an engine?
  3. Null Rate: Is the sensor data coming in clean?
  4. Out-of-Range Rate: Are the readings physically possible? (We don't fail the pipeline for this, just log it)
  5. Batch Size: Did we actually load anything?
  6. Referential Integrity: Do our fact rows map to real engines and batches?
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def get_engine():
    db_url = os.environ.get("DATABASE_URL", "").strip()
    if not db_url:
        log.error("DATABASE_URL environment variable is not set.")
        sys.exit(1)
    return create_engine(db_url)


def run_checks() -> int:
    """Runs the full DQ suite, logs to Postgres, and returns how many checks failed."""
    engine = get_engine()
    run_started_at = datetime.now(timezone.utc)
    
    # Get the most recent batch date for logging purposes
    with engine.connect() as conn:
        batch_date = conn.execute(text("SELECT MAX(batch_date) FROM dim_batch WHERE status = 'loaded'")).scalar()
        if not batch_date:
            batch_date = run_started_at.date() # Fallback

    failures = 0
    results = []

    log.info("=" * 60)
    log.info("QUALITY CHECKS")
    log.info("=" * 60)

    with engine.begin() as conn:
        # ── 1. Freshness Check ────────────────────────────────────────────────
        # Expected cadence: 24 hours. (Tolerance: 30 hours to be safe)
        freshness_sql = """
            SELECT EXTRACT(EPOCH FROM (NOW() - MAX(loaded_at))) / 3600.0 AS hours_since
            FROM dim_batch
            WHERE status = 'loaded'
        """
        hours_since = conn.execute(text(freshness_sql)).scalar()
        if hours_since is None:
            hours_since = 999.0  # No batches loaded yet
            
        threshold = 30.0
        passed = float(hours_since) <= threshold
        if not passed: failures += 1
        
        results.append({
            "check_name": "freshness_sla",
            "metric_value": float(hours_since),
            "threshold": threshold,
            "passed": passed,
            "notes": f"{hours_since:.2f} hours since last successful batch"
        })

        # ── 2. Cycle Gap Check ────────────────────────────────────────────────
        # No missing cycles per unit. max(cycle) == count(distinct cycle)
        gap_sql = """
            SELECT COUNT(*) AS units_with_gaps
            FROM (
                SELECT unit_id
                FROM fact_sensor_readings
                GROUP BY unit_id
                HAVING COUNT(DISTINCT cycle) <> MAX(cycle)
            ) t
        """
        units_with_gaps = conn.execute(text(gap_sql)).scalar() or 0
        threshold = 0.0
        passed = (units_with_gaps == 0)
        if not passed: failures += 1

        results.append({
            "check_name": "cycle_gap_check",
            "metric_value": float(units_with_gaps),
            "threshold": threshold,
            "passed": passed,
            "notes": f"{units_with_gaps} unit(s) have missing intermediate cycles"
        })

        # ── 3. Null Rate Check ────────────────────────────────────────────────
        # Null rate on raw reading_value below 1%
        null_sql = """
            SELECT SUM(CASE WHEN reading_value IS NULL THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0)
            FROM fact_sensor_readings
        """
        null_rate = conn.execute(text(null_sql)).scalar() or 0.0
        threshold = 1.0
        passed = float(null_rate) <= threshold
        if not passed: failures += 1

        results.append({
            "check_name": "null_rate_per_column",
            "metric_value": float(null_rate),
            "threshold": threshold,
            "passed": passed,
            "notes": f"Reading value null rate is {null_rate:.4f}%"
        })

        # ── 4. Out-of-Range Check ─────────────────────────────────────────────
        # % of sensor readings outside expected_min/max
        # We don't fail the pipeline on this (threshold 100%) but we log it as an observable metric
        oor_sql = """
            SELECT SUM(CASE WHEN f.reading_value < s.expected_min OR f.reading_value > s.expected_max THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0)
            FROM fact_sensor_readings f
            JOIN dim_sensor s ON f.sensor_id = s.sensor_id
        """
        oor_rate = conn.execute(text(oor_sql)).scalar() or 0.0
        threshold = 100.0  # Just for tracking
        passed = float(oor_rate) <= threshold
        if not passed: failures += 1

        results.append({
            "check_name": "out_of_range_rate",
            "metric_value": float(oor_rate),
            "threshold": threshold,
            "passed": passed,
            "notes": f"{oor_rate:.2f}% of readings outside dim_sensor bounds"
        })

        # ── 5. Batch Size Check ───────────────────────────────────────────────
        # Most recent batch should have > 0 rows
        size_sql = """
            SELECT row_count
            FROM dim_batch
            WHERE status = 'loaded'
            ORDER BY loaded_at DESC
            LIMIT 1
        """
        latest_batch_size = conn.execute(text(size_sql)).scalar()
        if latest_batch_size is None:
            latest_batch_size = 0
            
        threshold = 1.0 # Minimum 1 row
        passed = latest_batch_size >= threshold
        if not passed: failures += 1

        results.append({
            "check_name": "batch_row_count",
            "metric_value": float(latest_batch_size),
            "threshold": threshold,
            "passed": passed,
            "notes": f"Latest loaded batch had {latest_batch_size} rows"
        })

        # ── 6. Referential Integrity ──────────────────────────────────────────
        # Every fact row has a valid unit_id, sensor_id, batch_id
        # (Though PG constraints enforce this, it's good to verify explicitly)
        ri_sql = """
            SELECT COUNT(*)
            FROM fact_sensor_readings
            WHERE unit_id NOT IN (SELECT unit_id FROM dim_equipment)
               OR sensor_id NOT IN (SELECT sensor_id FROM dim_sensor)
               OR batch_id NOT IN (SELECT batch_id FROM dim_batch)
        """
        orphans = conn.execute(text(ri_sql)).scalar() or 0
        threshold = 0.0
        passed = (orphans == 0)
        if not passed: failures += 1

        results.append({
            "check_name": "referential_integrity",
            "metric_value": float(orphans),
            "threshold": threshold,
            "passed": passed,
            "notes": f"Found {orphans} orphaned fact rows"
        })

        # ── Log Results to DB ─────────────────────────────────────────────────
        run_finished_at = datetime.now(timezone.utc)
        
        insert_sql = text("""
            INSERT INTO pipeline_run_log 
                (batch_date, run_started_at, run_finished_at, check_name, metric_value, threshold, passed, notes)
            VALUES 
                (:batch_date, :run_started_at, :run_finished_at, :check_name, :metric_value, :threshold, :passed, :notes)
        """)
        
        for res in results:
            # Print console summary
            status_str = "PASS" if res["passed"] else "FAIL"
            log.info(f"[{status_str}] {res['check_name']:<25} | Metric: {res['metric_value']:<8.2f} | Threshold: {res['threshold']:<8.2f} | {res['notes']}")
            
            # Write to DB
            conn.execute(insert_sql, {
                "batch_date": batch_date,
                "run_started_at": run_started_at,
                "run_finished_at": run_finished_at,
                "check_name": res["check_name"],
                "metric_value": res["metric_value"],
                "threshold": res["threshold"],
                "passed": res["passed"],
                "notes": res["notes"]
            })

    log.info("=" * 60)
    if failures == 0:
        log.info("All quality checks PASSED.")
    else:
        log.error(f"{failures} quality check(s) FAILED.")
        
    return failures


def main():
    t0 = time.perf_counter()
    failures = run_checks()
    elapsed = time.perf_counter() - t0
    
    log.info(f"Quality checks completed in {elapsed:.2f}s")
    
    if failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
