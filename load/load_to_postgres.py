"""
load/load_to_postgres.py

Grabs the Gold Parquet files and pushes them into the Postgres warehouse.

It handles:
  - Making sure all 21 sensors are defined in `dim_sensor`.
  - Upserting the engine details into `dim_equipment`.
  - Logging the batch load into `dim_batch`.
  - Bulk-inserting the actual readings into `fact_sensor_readings`.

Everything is idempotent. If a file was already loaded, it skips it automatically.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Numeric, SmallInteger, BigInteger, Date, DateTime, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.sql import select

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
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_AGG_DIR = PROJECT_ROOT / "data" / "aggregated"
ALL_DATASETS = ["FD001", "FD002", "FD003", "FD004"]

# Mirrors transform_batches.py SENSOR_BOUNDS
SENSOR_BOUNDS: dict[int, tuple[str, float, float]] = {
    1: ("sensor_1", 443.5266, 520.1434),
    2: ("sensor_2", 533.3502, 646.6998),
    3: ("sensor_3", 1236.2664, 1624.3736),
    4: ("sensor_4", 1015.4156, 1449.8444),
    5: ("sensor_5", 3.6958, 14.8342),
    6: ("sensor_6", 5.3920, 21.9280),
    7: ("sensor_7", 128.4196, 564.2004),
    8: ("sensor_8", 1905.2954, 2397.9746),
    9: ("sensor_9", 7960.4760, 9264.8440),
    10: ("sensor_10", 0.9226, 1.3074),
    11: ("sensor_11", 35.9842, 48.7658),
    12: ("sensor_12", 121.2350, 531.2550),
    13: ("sensor_13", 2020.3526, 2397.7374),
    14: ("sensor_14", 7839.5346, 8298.4554),
    15: ("sensor_15", 8.2811, 11.1215),
    16: ("sensor_16", 0.0198, 0.0302),
    17: ("sensor_17", 301.0600, 401.9400),
    18: ("sensor_18", 1905.5400, 2397.4600),
    19: ("sensor_19", 84.6286, 100.3014),
    20: ("sensor_20", 9.5968, 39.9232),
    21: ("sensor_21", 5.6589, 23.9417),
}


# ---------------------------------------------------------------------------
# SQLAlchemy Schema Definitions
# ---------------------------------------------------------------------------
metadata = MetaData()

dim_equipment = Table('dim_equipment', metadata,
    Column('unit_id', Integer, primary_key=True),
    Column('dataset_source', String(10)),
    Column('max_cycles_observed', Integer),
    Column('failure_cycle', Integer),
    Column('first_seen_at', DateTime(timezone=True)),
    Column('last_updated_at', DateTime(timezone=True))
)

dim_sensor = Table('dim_sensor', metadata,
    Column('sensor_id', SmallInteger, primary_key=True),
    Column('sensor_name', String(30), unique=True),
    Column('expected_min', Numeric(12,4)),
    Column('expected_max', Numeric(12,4))
)

dim_batch = Table('dim_batch', metadata,
    Column('batch_id', BigInteger, primary_key=True),
    Column('batch_date', Date),
    Column('ingested_at', DateTime(timezone=True)),
    Column('source_file', String(60)),
    Column('dataset', String(10)),
    Column('row_count', Integer),
    Column('status', String(10)),
    Column('loaded_at', DateTime(timezone=True))
)

fact_sensor_readings = Table('fact_sensor_readings', metadata,
    Column('reading_id', BigInteger, primary_key=True),
    Column('unit_id', Integer),
    Column('sensor_id', SmallInteger),
    Column('batch_id', BigInteger),
    Column('cycle', Integer),
    Column('reading_value', Numeric(14,6)),
    Column('rolling_avg_7', Numeric(14,6)),
    Column('rate_of_change', Numeric(14,6)),
    Column('loaded_at', DateTime(timezone=True))
)


def get_engine():
    db_url = os.environ.get("DATABASE_URL", "").strip()
    if not db_url:
        log.error("DATABASE_URL environment variable is not set.")
        sys.exit(1)
    
    # Optional performance tuning for SQLAlchemy with PostgreSQL
    return create_engine(
        db_url, 
        pool_size=10, 
        max_overflow=20
    )


# ---------------------------------------------------------------------------
# Loader Functions
# ---------------------------------------------------------------------------

def ensure_dim_sensor(conn):
    """Makes sure our sensor reference table is populated. Safe to run repeatedly."""
    log.info("Ensuring dim_sensor is populated...")
    
    sensor_records = [
        {
            "sensor_id": s_id, 
            "sensor_name": name, 
            "expected_min": min_v, 
            "expected_max": max_v
        }
        for s_id, (name, min_v, max_v) in SENSOR_BOUNDS.items()
    ]

    stmt = insert(dim_sensor).values(sensor_records)
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=['sensor_id'],
        set_={
            'sensor_name': stmt.excluded.sensor_name,
            'expected_min': stmt.excluded.expected_min,
            'expected_max': stmt.excluded.expected_max
        }
    )
    conn.execute(upsert_stmt)
    log.info("  dim_sensor OK.")


def get_loaded_batches(conn) -> set[str]:
    """Pulls a list of files we've already loaded so we don't duplicate work."""
    stmt = select(dim_batch.c.source_file).where(dim_batch.c.status == 'loaded')
    result = conn.execute(stmt).fetchall()
    return {row[0] for row in result}


def load_gold_file(conn, gold_path: Path, dataset: str, force: bool, loaded_files: set[str]) -> tuple[bool, int]:
    """
    Pushes a single Gold Parquet file into Postgres.
    We do the equipment, batch logging, and fact table inserts all in one transaction 
    so if something breaks, we don't get partial data.
    """
    source_file = gold_path.name
    
    if source_file in loaded_files and not force:
        log.debug("  [SKIP] %s is already loaded.", source_file)
        return False, 0
    
    # ── Read Parquet ──────────────────────────────────────────────────────────
    try:
        df = pl.read_parquet(gold_path)
    except Exception as exc:
        log.error("  [ERROR] Cannot read %s: %s", gold_path.name, exc)
        return False, 0
    
    if len(df) == 0:
        log.warning("  [SKIP] %s is empty.", source_file)
        return False, 0

    unit_id = df["unit_id"][0]
    max_cycle = df["cycle"].max()
    row_count = len(df)
    now_utc = datetime.now(timezone.utc)

    # Convert dataframe to a list of dicts for SQLAlchemy bulk insert
    # Keep only the columns present in the fact table
    records = df.select([
        "unit_id", "sensor_id", "cycle", "reading_value", 
        "rolling_avg_7", "rate_of_change"
    ]).to_dicts()

    # We do all updates in one transaction (the connection passed in manages it)
    log.info("Loading %s (Unit %d, %d rows)...", source_file, unit_id, row_count)

    # 1. Upsert dim_equipment
    eq_stmt = insert(dim_equipment).values([{
        "unit_id": unit_id,
        "dataset_source": dataset,
        "max_cycles_observed": max_cycle,
        "first_seen_at": now_utc,
        "last_updated_at": now_utc
    }])
    eq_upsert = eq_stmt.on_conflict_do_update(
        index_elements=['unit_id'],
        set_={
            'max_cycles_observed': func.greatest(dim_equipment.c.max_cycles_observed, eq_stmt.excluded.max_cycles_observed),
            'last_updated_at': now_utc
        }
    )
    conn.execute(eq_upsert)

    # 2. Insert into dim_batch
    # Notice we insert as 'loaded' immediately because it's part of the same transaction
    batch_stmt = insert(dim_batch).values(
        batch_date=now_utc.date(),
        ingested_at=now_utc,
        source_file=source_file,
        dataset=dataset,
        row_count=row_count,
        status='loaded',
        loaded_at=now_utc
    ).returning(dim_batch.c.batch_id)
    batch_id = conn.execute(batch_stmt).scalar()

    # 3. Add batch_id to records and loaded_at
    for row in records:
        row["batch_id"] = batch_id
        row["loaded_at"] = now_utc

    # 4. Upsert into fact_sensor_readings
    # using ON CONFLICT (unit_id, cycle, sensor_id) DO UPDATE
    fact_stmt = insert(fact_sensor_readings).values(records)
    fact_upsert = fact_stmt.on_conflict_do_update(
        index_elements=['unit_id', 'cycle', 'sensor_id'],
        set_={
            'batch_id': fact_stmt.excluded.batch_id,
            'reading_value': fact_stmt.excluded.reading_value,
            'rolling_avg_7': fact_stmt.excluded.rolling_avg_7,
            'rate_of_change': fact_stmt.excluded.rate_of_change,
            'loaded_at': fact_stmt.excluded.loaded_at
        }
    )
    
    conn.execute(fact_upsert)
    
    return True, row_count


def run_loader(agg_dir: Path, datasets: list[str], force: bool) -> None:
    engine = get_engine()
    
    total_files_loaded = 0
    total_rows_loaded = 0

    log.info("=" * 60)
    log.info("LOAD  Gold -> PostgreSQL")
    log.info("=" * 60)

    # 1. Start by ensuring dim_sensor is complete
    # We'll use a single connection with a manual transaction loop for safety
    with engine.begin() as conn:
        ensure_dim_sensor(conn)

    # Fetch loaded batches across the whole DB
    with engine.connect() as conn:
        loaded_files = get_loaded_batches(conn)

    # Discover files
    for dataset in datasets:
        dataset_dir = agg_dir / f"dataset={dataset}"
        if not dataset_dir.exists():
            continue
            
        for unit_dir in sorted(dataset_dir.iterdir()):
            if not unit_dir.is_dir() or not unit_dir.name.startswith("unit="):
                continue
                
            for gold_file in sorted(unit_dir.glob("gold_*.parquet")):
                
                # Use engine.begin() to start an implicit transaction per file
                try:
                    with engine.begin() as conn:
                        loaded, rows = load_gold_file(conn, gold_file, dataset, force, loaded_files)
                        if loaded:
                            total_files_loaded += 1
                            total_rows_loaded += rows
                except Exception as exc:
                    log.error("  [ERROR] Transaction failed for %s. Rolled back. %s", gold_file.name, exc)

    log.info("")
    log.info("=" * 60)
    log.info("  LOAD SUMMARY")
    log.info("=" * 60)
    log.info(f"  Gold files loaded : {total_files_loaded}")
    log.info(f"  Total rows upserted: {total_rows_loaded:,}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load Gold Parquet files into the PostgreSQL warehouse.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--agg-dir",
        type=Path,
        default=DEFAULT_AGG_DIR,
        help="Root directory of Gold (aggregated) Parquet files.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=ALL_DATASETS,
        default=None,
        metavar="FDxxx",
        help="Restrict load to specific C-MAPSS subsets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-load files that are already recorded as loaded in dim_batch.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def main() -> None:
    import io as _io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    datasets = args.dataset or ALL_DATASETS

    log.info(
        "Starting load  |  datasets=%s  force=%s",
        datasets, args.force
    )

    t0 = time.perf_counter()
    run_loader(
        agg_dir=args.agg_dir,
        datasets=datasets,
        force=args.force
    )
    elapsed = time.perf_counter() - t0

    log.info("Load complete in %.1fs", elapsed)


if __name__ == "__main__":
    main()
