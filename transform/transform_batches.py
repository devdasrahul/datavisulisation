"""
transform/transform_batches.py

Takes the raw Bronze Parquet files, cleans them up (Silver), and engineers
features like rolling averages (Gold).

The process:
    Bronze -> [Clean] -> Silver
    Silver -> [Enrich] -> Gold

Silver keeps the data wide (one column per sensor). We deduplicate rows, 
drop the metadata columns we added during ingest, and flag any readings that 
fall outside expected physical bounds (we don't delete them, just flag them).

Gold melts the data into a long format (fact-table ready) and calculates
the 7-cycle rolling average and rate-of-change for each sensor.
Because rolling averages need historical context, the Gold file for an engine 
is fully recomputed whenever a new Silver batch arrives.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import polars as pl
from dotenv import load_dotenv

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

DEFAULT_RAW_DIR   = PROJECT_ROOT / "data" / "raw"
DEFAULT_CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned"
DEFAULT_AGG_DIR   = PROJECT_ROOT / "data" / "aggregated"

ALL_DATASETS = ["FD001", "FD002", "FD003", "FD004"]


def _rel(path: Path) -> Path:
    """Return path relative to PROJECT_ROOT, or the full path if outside it."""
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path

# Bronze metadata columns added by ingest_batches.py — dropped before Silver write
BRONZE_META_COLS = {"_ingested_at", "_source_file", "_batch_idx"}

# The 26 canonical sensor-data columns (matches ingest_batches.RAW_COLUMNS)
RAW_COLUMNS: list[str] = [
    "unit_id", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1",  "sensor_2",  "sensor_3",  "sensor_4",  "sensor_5",
    "sensor_6",  "sensor_7",  "sensor_8",  "sensor_9",  "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21",
]

SENSOR_COLS:  list[str] = [c for c in RAW_COLUMNS if c.startswith("sensor_")]
SETTING_COLS: list[str] = ["op_setting_1", "op_setting_2", "op_setting_3"]
ID_COLS:      list[str] = ["unit_id", "cycle"]

# Expected sensor bounds — mirrors the seed data in sql/schema.sql (dim_sensor).
# Used for anomaly flagging in the Silver clean step.
# Dict format: sensor_name -> (expected_min, expected_max)
SENSOR_BOUNDS: dict[str, tuple[float, float]] = {
    "sensor_1": (443.5266, 520.1434),
    "sensor_2": (533.3502, 646.6998),
    "sensor_3": (1236.2664, 1624.3736),
    "sensor_4": (1015.4156, 1449.8444),
    "sensor_5": (3.6958, 14.8342),
    "sensor_6": (5.3920, 21.9280),
    "sensor_7": (128.4196, 564.2004),
    "sensor_8": (1905.2954, 2397.9746),
    "sensor_9": (7960.4760, 9264.8440),
    "sensor_10": (0.9226, 1.3074),
    "sensor_11": (35.9842, 48.7658),
    "sensor_12": (121.2350, 531.2550),
    "sensor_13": (2020.3526, 2397.7374),
    "sensor_14": (7839.5346, 8298.4554),
    "sensor_15": (8.2811, 11.1215),
    "sensor_16": (0.0198, 0.0302),
    "sensor_17": (301.0600, 401.9400),
    "sensor_18": (1905.5400, 2397.4600),
    "sensor_19": (84.6286, 100.3014),
    "sensor_20": (9.5968, 39.9232),
    "sensor_21": (5.6589, 23.9417),
}

# Rolling window size (matches PRD Section 7 and the SQL view definition)
ROLLING_WINDOW = 7


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BatchReport:
    """Collects statistics for one Bronze -> Silver transform."""
    bronze_path:     str = ""
    silver_path:     str = ""
    rows_in:         int = 0
    rows_out:        int = 0
    duplicates_dropped: int = 0
    null_counts:     dict[str, int] = field(default_factory=dict)
    anomaly_counts:  dict[str, int] = field(default_factory=dict)
    skipped:         bool = False
    error:           str | None = None


@dataclass
class RunStats:
    """Aggregate statistics across all batches in one run."""
    batches_transformed: int = 0
    batches_skipped:     int = 0
    batches_errored:     int = 0
    rows_in_total:       int = 0
    rows_out_silver:     int = 0
    rows_out_gold:       int = 0
    units_gold_updated:  int = 0
    total_duplicates:    int = 0
    total_nulls:         int = 0
    total_anomalies:     int = 0


# ---------------------------------------------------------------------------
# Manifest helpers (parallel structure to ingest manifest)
# ---------------------------------------------------------------------------

def _manifest_path(clean_dir: Path) -> Path:
    return clean_dir / ".transform_manifest.json"


def load_manifest(clean_dir: Path) -> dict:
    path = _manifest_path(clean_dir)
    if path.exists():
        try:
            with path.open(encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, KeyError):
            log.warning("Transform manifest corrupted — starting fresh: %s", path)
    return {"version": 1, "batches": {}}


def save_manifest(clean_dir: Path, manifest: dict) -> None:
    clean_dir.mkdir(parents=True, exist_ok=True)
    path    = _manifest_path(clean_dir)
    tmp     = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Bronze file discovery
# ---------------------------------------------------------------------------

def find_bronze_files(raw_dir: Path, datasets: list[str]) -> Generator[tuple[str, int, Path], None, None]:
    """
    Yield (dataset, unit_id, parquet_path) for every Bronze Parquet file
    found under raw_dir, filtered to the requested datasets.
    """
    for dataset in datasets:
        dataset_dir = raw_dir / f"dataset={dataset}"
        if not dataset_dir.exists():
            log.debug("No bronze directory found: %s", dataset_dir)
            continue
        for unit_dir in sorted(dataset_dir.iterdir()):
            if not unit_dir.is_dir() or not unit_dir.name.startswith("unit="):
                continue
            try:
                unit_id = int(unit_dir.name.split("=")[1])
            except (IndexError, ValueError):
                log.warning("Unexpected unit directory name: %s", unit_dir.name)
                continue
            for pq_file in sorted(unit_dir.glob("batch=*.parquet")):
                yield dataset, unit_id, pq_file


# ---------------------------------------------------------------------------
# STEP 1: Bronze → Silver (cleaning)
# ---------------------------------------------------------------------------

def clean_batch(
    bronze_path: Path,
    clean_dir: Path,
    dataset: str,
    unit_id: int,
    dry_run: bool,
) -> BatchReport:
    """
    Cleans a single Bronze file and saves it to Silver.
    Strips out the pipeline metadata cols, enforces data types, deduplicates, 
    and checks sensor bounds to flag anomalies.
    """
    report = BatchReport(bronze_path=str(_rel(bronze_path)))

    # ── Read Bronze ──────────────────────────────────────────────────────────
    try:
        df = pl.read_parquet(bronze_path)
    except Exception as exc:
        report.error = f"Read error: {exc}"
        log.error("  [ERROR] Cannot read %s: %s", bronze_path.name, exc)
        return report

    report.rows_in = len(df)

    # ── 1. Strip Bronze-only metadata columns ────────────────────────────────
    meta_cols_present = [c for c in BRONZE_META_COLS if c in df.columns]
    if meta_cols_present:
        df = df.drop(meta_cols_present)

    # ── 2. Ensure all 26 RAW_COLUMNS present ────────────────────────────────
    for col in RAW_COLUMNS:
        if col not in df.columns:
            log.warning("  Missing column %s in %s — filling with null", col, bronze_path.name)
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    # Reorder to canonical order
    df = df.select(RAW_COLUMNS)

    # ── 3. Enforce dtypes ────────────────────────────────────────────────────
    cast_exprs = []
    for col in ID_COLS:
        if df[col].dtype != pl.Int32:
            cast_exprs.append(pl.col(col).cast(pl.Int32, strict=False))
    for col in SETTING_COLS + SENSOR_COLS:
        if df[col].dtype != pl.Float64:
            cast_exprs.append(pl.col(col).cast(pl.Float64, strict=False))
    if cast_exprs:
        df = df.with_columns(cast_exprs)

    # ── 4. Drop exact duplicate rows (same unit_id + cycle) ──────────────────
    before_dedup = len(df)
    df = df.unique(subset=["unit_id", "cycle"], keep="first", maintain_order=True)
    report.duplicates_dropped = before_dedup - len(df)
    if report.duplicates_dropped:
        log.warning(
            "  Dropped %d duplicate (unit_id, cycle) rows from %s",
            report.duplicates_dropped, bronze_path.name,
        )

    # ── 5. Count nulls per column ────────────────────────────────────────────
    null_counts: dict[str, int] = {}
    for col in df.columns:
        n_null = df[col].null_count()
        if n_null > 0:
            null_counts[col] = n_null
    report.null_counts = null_counts
    if null_counts:
        log.warning(
            "  Nulls found in %s: %s",
            bronze_path.name,
            {k: v for k, v in sorted(null_counts.items(), key=lambda x: -x[1])},
        )

    # ── 6. Flag out-of-range readings ────────────────────────────────────────
    anomaly_exprs: list[pl.Expr] = []
    anomaly_counts: dict[str, int] = {}

    for sensor_col, (lo, hi) in SENSOR_BOUNDS.items():
        if sensor_col not in df.columns:
            continue
        out_of_range = (pl.col(sensor_col) < lo) | (pl.col(sensor_col) > hi)
        n_anomalies = df.filter(out_of_range.fill_null(False)).height
        if n_anomalies:
            anomaly_counts[sensor_col] = n_anomalies
        anomaly_exprs.append(out_of_range.alias(f"_oor_{sensor_col}"))

    # Combine all per-sensor OOR flags into a single is_anomaly boolean
    if anomaly_exprs:
        df = df.with_columns(anomaly_exprs)
        oor_cols = [f"_oor_{s}" for s in SENSOR_BOUNDS if f"_oor_{s}" in df.columns]
        # is_anomaly = True if ANY sensor reading is out of range
        combined_oor = pl.lit(False)
        for c in oor_cols:
            combined_oor = combined_oor | pl.col(c).fill_null(False)
        df = df.with_columns(combined_oor.alias("is_anomaly"))
        df = df.drop(oor_cols)  # keep the row, drop per-sensor flag cols
    else:
        df = df.with_columns(pl.lit(False).alias("is_anomaly"))

    report.anomaly_counts = anomaly_counts
    if anomaly_counts:
        log.warning(
            "  Out-of-range readings in %s: %s",
            bronze_path.name,
            {k: v for k, v in sorted(anomaly_counts.items(), key=lambda x: -x[1])},
        )

    report.rows_out = len(df)

    # ── Write Silver Parquet ─────────────────────────────────────────────────
    silver_path = (
        clean_dir
        / f"dataset={dataset}"
        / f"unit={unit_id}"
        / bronze_path.name  # preserve original filename for easy lineage
    )
    report.silver_path = str(_rel(silver_path))

    if not dry_run:
        silver_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(silver_path, compression="snappy")
        log.info(
            "  Silver -> %-55s  %d rows  (dupes=%d  nulls=%d  anomalies=%d)",
            _rel(silver_path),
            len(df),
            report.duplicates_dropped,
            sum(null_counts.values()),
            sum(anomaly_counts.values()),
        )
    else:
        log.info(
            "  [DRY-RUN] Silver would write %d rows -> %s",
            len(df), _rel(silver_path),
        )

    return report


# ---------------------------------------------------------------------------
# STEP 2: Silver → Gold (feature engineering)
# ---------------------------------------------------------------------------

def compute_gold(
    clean_dir: Path,
    agg_dir: Path,
    dataset: str,
    unit_id: int,
    dry_run: bool,
    transformed_at: datetime,
) -> int:
    """
    Grabs all the Silver files for a specific engine, melts them down, 
    and calculates the rolling averages and rate of change. 
    
    This completely overwrites the Gold file for the engine each time it runs, 
    because we need the full history to calculate the rolling windows correctly.
    """
    silver_dir = clean_dir / f"dataset={dataset}" / f"unit={unit_id}"
    silver_files = sorted(silver_dir.glob("batch=*.parquet"))

    if not silver_files:
        log.warning("  No silver files found for %s unit %d", dataset, unit_id)
        return 0

    # ── Load and concatenate all Silver files for this unit ──────────────────
    frames: list[pl.DataFrame] = []
    for sf in silver_files:
        try:
            frames.append(pl.read_parquet(sf))
        except Exception as exc:
            log.error("  Skipping corrupt silver file %s: %s", sf.name, exc)

    if not frames:
        return 0

    silver_df = pl.concat(frames)

    # Drop duplicates that may span batch boundaries (same unit_id + cycle)
    silver_df = silver_df.unique(subset=["unit_id", "cycle"], keep="first", maintain_order=True)

    # ── Melt (unpivot) wide -> long: one row per (unit_id, cycle, sensor) ────
    # Keep is_anomaly at the row level — will be True if ANY sensor on that cycle
    # was out-of-range. This means the flag applies to the whole cycle, not a
    # specific sensor reading (acceptable for the fact table grain).
    keep_cols = ID_COLS + SETTING_COLS + ["is_anomaly"]
    available_keep = [c for c in keep_cols if c in silver_df.columns]

    long_df = silver_df.unpivot(
        on=SENSOR_COLS,
        index=available_keep,
        variable_name="sensor_name",
        value_name="reading_value",
    )

    # Extract integer sensor_id from "sensor_N" -> N
    long_df = long_df.with_columns(
        pl.col("sensor_name")
          .str.extract(r"sensor_(\d+)", 1)
          .cast(pl.Int32)
          .alias("sensor_id")
    )

    # ── Sort for window operations ────────────────────────────────────────────
    # CRITICAL: the DataFrame MUST be sorted by (unit_id, sensor_id, cycle)
    # before applying .over() with rolling/diff operations in Polars.
    long_df = long_df.sort(["unit_id", "sensor_id", "cycle"])

    # ── Compute engineered features ───────────────────────────────────────────
    long_df = long_df.with_columns([
        # 7-cycle rolling mean per unit+sensor.
        # min_periods=1 means partial windows at the start of each unit's life
        # still produce a value (avg of available readings) rather than null.
        pl.col("reading_value")
          .rolling_mean(window_size=ROLLING_WINDOW, min_periods=1)
          .over(["unit_id", "sensor_id"])
          .alias("rolling_avg_7"),

        # Cycle-over-cycle difference (rate of change) per unit+sensor.
        # First cycle for each unit+sensor will be null — expected behaviour.
        pl.col("reading_value")
          .diff(n=1)
          .over(["unit_id", "sensor_id"])
          .alias("rate_of_change"),
    ])

    # Add transform lineage column
    long_df = long_df.with_columns(
        pl.lit(transformed_at.isoformat()).alias("_transformed_at")
    )

    # ── Reorder columns to match fact table grain ─────────────────────────────
    col_order = (
        ["unit_id", "cycle", "sensor_id", "sensor_name", "reading_value",
         "rolling_avg_7", "rate_of_change"]
        + [c for c in available_keep if c not in ("unit_id", "cycle")]
        + ["_transformed_at"]
    )
    col_order = [c for c in col_order if c in long_df.columns]
    long_df = long_df.select(col_order)

    rows_out = len(long_df)

    # ── Write Gold Parquet ────────────────────────────────────────────────────
    ts_str    = transformed_at.strftime("%Y%m%dT%H%M%S")
    gold_dir  = agg_dir / f"dataset={dataset}" / f"unit={unit_id}"
    gold_path = gold_dir / f"gold_{ts_str}.parquet"

    if not dry_run:
        # Remove previous Gold files for this unit (full recompute)
        if gold_dir.exists():
            for old in gold_dir.glob("gold_*.parquet"):
                old.unlink()
                log.debug("  Removed stale gold file: %s", old.name)

        gold_dir.mkdir(parents=True, exist_ok=True)
        long_df.write_parquet(gold_path, compression="snappy")
        log.info(
            "  Gold   -> %-55s  %d rows  (%d cycles x 21 sensors)",
            _rel(gold_path),
            rows_out,
            rows_out // 21 if rows_out >= 21 else rows_out,
        )
    else:
        log.info(
            "  [DRY-RUN] Gold would write %d rows -> %s",
            rows_out, _rel(gold_path),
        )

    return rows_out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_transform(
    raw_dir: Path,
    clean_dir: Path,
    agg_dir: Path,
    datasets: list[str],
    force: bool,
    dry_run: bool,
) -> RunStats:
    """
    The main driver. Finds new Bronze files, cleans them, and then recomputes 
    Gold for any engines that got new data.
    """
    clean_dir.mkdir(parents=True, exist_ok=True)
    agg_dir.mkdir(parents=True, exist_ok=True)

    manifest         = load_manifest(clean_dir)
    transformed_at   = datetime.now(tz=timezone.utc)
    stats            = RunStats()
    reports: list[BatchReport] = []

    # Track which (dataset, unit_id) pairs got new Silver this run
    units_needing_gold: set[tuple[str, int]] = set()

    log.info("=" * 60)
    log.info("TRANSFORM  Bronze -> Silver")
    log.info("=" * 60)

    # ── Step 1: Bronze -> Silver ──────────────────────────────────────────────
    for dataset, unit_id, bronze_path in find_bronze_files(raw_dir, datasets):
        manifest_key = str(_rel(bronze_path))

        if manifest_key in manifest["batches"] and not force:
            log.debug("  [SKIP] Already transformed: %s", bronze_path.name)
            stats.batches_skipped += 1
            stats.rows_out_silver += manifest["batches"][manifest_key].get("rows_out_silver", 0)
            continue

        log.info("Processing: dataset=%s unit=%d file=%s", dataset, unit_id, bronze_path.name)
        report = clean_batch(
            bronze_path=bronze_path,
            clean_dir=clean_dir,
            dataset=dataset,
            unit_id=unit_id,
            dry_run=dry_run,
        )
        reports.append(report)

        if report.error:
            stats.batches_errored += 1
            continue

        # Update manifest
        if not dry_run:
            manifest["batches"][manifest_key] = {
                "dataset":           dataset,
                "unit_id":           unit_id,
                "bronze_path":       manifest_key,
                "silver_path":       report.silver_path,
                "transformed_at":    transformed_at.isoformat(),
                "rows_in":           report.rows_in,
                "rows_out_silver":   report.rows_out,
                "duplicates_dropped": report.duplicates_dropped,
                "null_counts":       report.null_counts,
                "anomaly_counts":    report.anomaly_counts,
            }

        stats.batches_transformed += 1
        stats.rows_in_total       += report.rows_in
        stats.rows_out_silver     += report.rows_out
        stats.total_duplicates    += report.duplicates_dropped
        stats.total_nulls         += sum(report.null_counts.values())
        stats.total_anomalies     += sum(report.anomaly_counts.values())

        units_needing_gold.add((dataset, unit_id))

    # Persist manifest after Silver step
    if not dry_run and stats.batches_transformed > 0:
        save_manifest(clean_dir, manifest)

    # ── Step 2: Silver -> Gold ────────────────────────────────────────────────
    if units_needing_gold:
        log.info("")
        log.info("=" * 60)
        log.info("TRANSFORM  Silver -> Gold  (%d unit(s))", len(units_needing_gold))
        log.info("=" * 60)

        for dataset, unit_id in sorted(units_needing_gold):
            log.info("Enriching: dataset=%s unit=%d", dataset, unit_id)
            try:
                gold_rows = compute_gold(
                    clean_dir=clean_dir,
                    agg_dir=agg_dir,
                    dataset=dataset,
                    unit_id=unit_id,
                    dry_run=dry_run,
                    transformed_at=transformed_at,
                )
                stats.rows_out_gold      += gold_rows
                stats.units_gold_updated += 1
            except Exception as exc:
                log.error("  [ERROR] Gold transform failed for %s unit %d: %s", dataset, unit_id, exc)
                stats.batches_errored += 1
    else:
        log.info("No new Silver batches — Gold layer is up to date.")

    return stats


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(stats: RunStats, reports: list[BatchReport] | None = None) -> None:
    prefix = "[DRY-RUN] " if False else ""  # dry_run flag is in stats via callers
    print()
    print("=" * 60)
    print("  TRANSFORM SUMMARY")
    print("=" * 60)
    print(f"  Batches transformed  : {stats.batches_transformed}")
    print(f"  Batches skipped      : {stats.batches_skipped}  (already in manifest)")
    print(f"  Batches errored      : {stats.batches_errored}")
    print()
    print(f"  Rows in (Bronze)     : {stats.rows_in_total:,}")
    print(f"  Rows out (Silver)    : {stats.rows_out_silver:,}")
    print(f"  Rows out (Gold)      : {stats.rows_out_gold:,}")
    print(f"  Units Gold updated   : {stats.units_gold_updated}")
    print()
    print(f"  Data quality:")
    print(f"    Duplicates dropped : {stats.total_duplicates:,}")
    print(f"    Null values found  : {stats.total_nulls:,}")
    print(f"    Anomalous readings : {stats.total_anomalies:,}  (flagged, not removed)")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Silver + Gold transform: clean Bronze Parquet and compute "
            "rolling avg / rate-of-change features."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Root directory of Bronze (raw) Parquet files.",
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=DEFAULT_CLEAN_DIR,
        help="Root output directory for Silver (cleaned) Parquet files.",
    )
    parser.add_argument(
        "--agg-dir",
        type=Path,
        default=DEFAULT_AGG_DIR,
        help="Root output directory for Gold (aggregated) Parquet files.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=ALL_DATASETS,
        default=None,
        metavar="FDxxx",
        help="Restrict transform to specific C-MAPSS subsets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transform batches already recorded in the manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and log transforms but write no Parquet files.",
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
        "Starting transform  |  datasets=%s  force=%s  dry_run=%s",
        datasets, args.force, args.dry_run,
    )

    t0    = time.perf_counter()
    stats = run_transform(
        raw_dir=args.raw_dir,
        clean_dir=args.clean_dir,
        agg_dir=args.agg_dir,
        datasets=datasets,
        force=args.force,
        dry_run=args.dry_run,
    )
    elapsed = time.perf_counter() - t0

    print_summary(stats)
    log.info("Transform complete in %.1fs", elapsed)

    if stats.batches_errored > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
