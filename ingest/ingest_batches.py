"""
ingest/ingest_batches.py

Reads the raw NASA C-MAPSS text files, chops them into small chronological batches 
(simulating a daily feed), and saves them as partitioned Parquet files under data/raw/.

We track everything in `data/raw/.manifest.json` to keep this idempotent. If you rerun 
the script, it just skips files it already knows about instead of duplicating data.

If DATABASE_URL is set, it also registers the batches in Postgres (`dim_batch`) so the 
load script knows they're waiting.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator

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

DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "source"
DEFAULT_DEST_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH_TPL = "{dest_dir}/.manifest.json"
DEFAULT_BATCH_SIZE = 20

# All four C-MAPSS subsets
ALL_DATASETS = ["FD001", "FD002", "FD003", "FD004"]

# Canonical column names for the 26 usable columns in each NASA train file.
# Cols 27-28 (if present) are trailing zeros from NASA's export and are dropped.
RAW_COLUMNS: list[str] = [
    "unit_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]

# Polars schema for strict type assignment at read time (keeps bronze data typed
# correctly without any value transformation — dtypes match the raw signal types)
RAW_SCHEMA: dict[str, pl.DataType] = {
    "unit_id": pl.Int32,
    "cycle": pl.Int32,
    "op_setting_1": pl.Float64,
    "op_setting_2": pl.Float64,
    "op_setting_3": pl.Float64,
    **{f"sensor_{i}": pl.Float64 for i in range(1, 22)},
}


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _manifest_path(dest_dir: Path) -> Path:
    return dest_dir / ".manifest.json"


def load_manifest(dest_dir: Path) -> dict:
    """Loads the ingest tracking manifest, creating a fresh one if it's missing or corrupted."""
    path = _manifest_path(dest_dir)
    if path.exists():
        try:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            return data
        except (json.JSONDecodeError, KeyError):
            log.warning("Manifest file corrupted — starting fresh: %s", path)
    return {"version": 1, "batches": {}}


def save_manifest(dest_dir: Path, manifest: dict) -> None:
    """Persist the manifest atomically (write to .tmp then rename)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = _manifest_path(dest_dir)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    tmp.replace(path)


def make_batch_key(dataset: str, unit_id: int, batch_idx: int) -> str:
    """Stable logical key for one (dataset, unit, batch_index) triple."""
    return f"{dataset}_unit{unit_id:04d}_batch{batch_idx:06d}"


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def find_source_files(source_dir: Path, datasets: list[str]) -> list[tuple[str, Path]]:
    """
    Return a list of (dataset_name, file_path) tuples for each requested dataset.
    Only train_FD00X.txt files are ingested — test files are kept separate.
    """
    found: list[tuple[str, Path]] = []
    for ds in datasets:
        candidate = source_dir / f"train_{ds}.txt"
        if candidate.exists():
            found.append((ds, candidate))
        else:
            log.warning(
                "Source file not found (run download_data.py first): %s", candidate
            )
    if not found:
        log.error("No source files found in %s for datasets %s", source_dir, datasets)
        sys.exit(1)
    return found


# ---------------------------------------------------------------------------
# Read raw file
# ---------------------------------------------------------------------------


def read_raw_file(path: Path) -> pl.DataFrame:
    """
    Pulls a NASA C-MAPSS train file into Polars.
    These files don't have headers and the column counts sometimes vary (NASA leaves
    trailing spaces that get parsed as empty columns). We trim the junk, map the 26 real
    sensor columns, and strictly type them here at the very edge of the pipeline.
    """
    log.info("Reading source file: %s", path.name)

    # Read without specifying a schema first, to detect actual column count
    raw = pl.read_csv(
        path,
        separator=" ",
        has_header=False,
        infer_schema_length=50,
        truncate_ragged_lines=True,  # some NASA files have trailing whitespace
        ignore_errors=True,
    )

    # Drop extra trailing columns (27, 28 — NASA trailing zeros)
    n_cols = raw.width
    if n_cols > 26:
        log.debug("  Dropping %d trailing column(s) from %s", n_cols - 26, path.name)
        raw = raw.select(raw.columns[:26])

    if raw.width != 26:
        log.error(
            "  Unexpected column count %d in %s (expected 26). "
            "Check the source file format.",
            raw.width,
            path.name,
        )
        sys.exit(1)

    # Assign canonical column names
    raw = raw.rename(dict(zip(raw.columns, RAW_COLUMNS)))

    # Cast to expected dtypes (keeps values identical, just typed correctly)
    cast_exprs = [pl.col(name).cast(dtype) for name, dtype in RAW_SCHEMA.items()]
    raw = raw.with_columns(cast_exprs)

    log.info("  Loaded %d rows, %d units", len(raw), raw["unit_id"].n_unique())
    return raw


# ---------------------------------------------------------------------------
# Batch splitting
# ---------------------------------------------------------------------------


def split_into_batches(
    unit_df: pl.DataFrame, batch_size: int
) -> Iterator[tuple[int, pl.DataFrame]]:
    """
    Yield (batch_index, batch_df) pairs by slicing a unit's cycles
    into consecutive windows of `batch_size` rows.

    Cycles are sorted ascending before slicing so earlier cycles
    always appear in earlier batches (simulating chronological arrival).
    The last batch may be smaller than batch_size.
    """
    # Sort by cycle to guarantee chronological order
    ordered = unit_df.sort("cycle")
    total = len(ordered)
    batch_idx = 0
    offset = 0
    while offset < total:
        chunk = ordered.slice(offset, batch_size)
        yield batch_idx, chunk
        offset += batch_size
        batch_idx += 1


# ---------------------------------------------------------------------------
# Parquet writer
# ---------------------------------------------------------------------------


def write_parquet(
    batch_df: pl.DataFrame,
    dest_dir: Path,
    dataset: str,
    unit_id: int,
    batch_idx: int,
    ingested_at: datetime,
    source_file: str,
    dry_run: bool,
) -> Path:
    """
    Write one batch to a Hive-partitioned Parquet file and return its path.

    Path pattern:
        <dest_dir>/dataset=<DS>/unit=<id>/batch=<YYYYMMDDTHHMMSS>.parquet

    Two lineage metadata columns are added:
        _ingested_at  : ISO-8601 timestamp string (when the ingest script ran)
        _source_file  : basename of the originating NASA text file

    These columns are prefixed with '_' to distinguish them from raw sensor
    columns and are ignored by all transform / analytics steps.
    """
    ts_str = ingested_at.strftime("%Y%m%dT%H%M%S")
    out_dir = dest_dir / f"dataset={dataset}" / f"unit={unit_id}"
    out_path = out_dir / f"batch={ts_str}_{batch_idx:06d}.parquet"

    # Append lineage metadata as literal string columns
    enriched = batch_df.with_columns(
        [
            pl.lit(ingested_at.isoformat()).alias("_ingested_at"),
            pl.lit(source_file).alias("_source_file"),
            pl.lit(batch_idx).cast(pl.Int32).alias("_batch_idx"),
        ]
    )

    if dry_run:
        try:
            display_path = out_path.relative_to(PROJECT_ROOT)
        except ValueError:
            display_path = out_path
        log.debug(
            "  [DRY-RUN] Would write %d rows -> %s",
            len(enriched),
            display_path,
        )
        return out_path

    out_dir.mkdir(parents=True, exist_ok=True)
    enriched.write_parquet(out_path, compression="snappy")
    return out_path


# ---------------------------------------------------------------------------
# Optional Postgres dim_batch registration
# ---------------------------------------------------------------------------


def _try_db_register(
    batch_date: date,
    source_file: str,
    dataset: str,
    row_count: int,
    ingested_at: datetime,
    dry_run: bool,
) -> int | None:
    """
    Logs the batch in Postgres so the downstream loader knows it's ready.
    Fails silently if the DB isn't configured (like when testing locally).
    """
    db_url = os.environ.get("DATABASE_URL", "").strip()
    if not db_url or dry_run:
        return None

    try:
        import psycopg2

        conn = psycopg2.connect(db_url, connect_timeout=8)
        conn.autocommit = False
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO dim_batch (batch_date, ingested_at, source_file, dataset, row_count, status)
            VALUES (%s, %s, %s, %s, %s, 'staged')
            RETURNING batch_id
            """,
            (batch_date, ingested_at, source_file, dataset, row_count),
        )
        batch_id: int = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return batch_id
    except Exception as exc:  # noqa: BLE001
        log.warning("dim_batch registration skipped (DB error): %s", exc)
        return None


# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------


def run_ingest(
    datasets: list[str],
    batch_size: int,
    batch_date: date,
    source_dir: Path,
    dest_dir: Path,
    force: bool,
    dry_run: bool,
) -> None:
    """
    Main ingest loop: iterate over datasets → units → batches,
    writing Parquet files and updating the manifest.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(dest_dir)

    # ── Counters for the end-of-run summary ──────────────────────────────────
    stats = {
        "files_processed": 0,
        "batches_written": 0,
        "batches_skipped": 0,
        "rows_written": 0,
        "rows_skipped": 0,
        "errors": 0,
    }

    ingested_at = datetime.now(tz=timezone.utc)
    source_files = find_source_files(source_dir, datasets)

    for dataset, file_path in source_files:
        log.info("=" * 56)
        log.info("Dataset: %s  |  File: %s", dataset, file_path.name)
        log.info("=" * 56)

        try:
            full_df = read_raw_file(file_path)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to read %s: %s", file_path.name, exc)
            stats["errors"] += 1
            continue

        stats["files_processed"] += 1
        unit_ids: list[int] = sorted(full_df["unit_id"].unique().to_list())

        for unit_id in unit_ids:
            unit_df = full_df.filter(pl.col("unit_id") == unit_id)
            n_cycles = len(unit_df)
            n_batches = (n_cycles + batch_size - 1) // batch_size  # ceil div

            log.info(
                "  Unit %3d | %4d cycles | %d batch(es) of %d",
                unit_id,
                n_cycles,
                n_batches,
                batch_size,
            )

            for batch_idx, batch_df in split_into_batches(unit_df, batch_size):
                key = make_batch_key(dataset, unit_id, batch_idx)

                # ── Idempotency check ─────────────────────────────────────
                if key in manifest["batches"] and not force:
                    existing = manifest["batches"][key]
                    log.debug(
                        "    [SKIP] %s  (written %s, %d rows)",
                        key,
                        existing.get("written_at", "?"),
                        existing.get("row_count", 0),
                    )
                    stats["batches_skipped"] += 1
                    stats["rows_skipped"] += existing.get("row_count", 0)
                    continue

                # ── Write Parquet ─────────────────────────────────────────
                try:
                    out_path = write_parquet(
                        batch_df=batch_df,
                        dest_dir=dest_dir,
                        dataset=dataset,
                        unit_id=unit_id,
                        batch_idx=batch_idx,
                        ingested_at=ingested_at,
                        source_file=file_path.name,
                        dry_run=dry_run,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.error("    [ERROR] writing %s: %s", key, exc)
                    stats["errors"] += 1
                    continue

                row_count = len(batch_df)

                # ── Register in dim_batch (optional, non-blocking) ────────
                db_batch_id = _try_db_register(
                    batch_date=batch_date,
                    source_file=file_path.name,
                    dataset=dataset,
                    row_count=row_count,
                    ingested_at=ingested_at,
                    dry_run=dry_run,
                )

                # ── Update manifest ───────────────────────────────────────
                cycle_min = int(batch_df["cycle"].min())
                cycle_max = int(batch_df["cycle"].max())

                manifest["batches"][key] = {
                    "dataset": dataset,
                    "unit_id": unit_id,
                    "batch_idx": batch_idx,
                    "source_file": file_path.name,
                    "parquet_path": str(out_path.relative_to(PROJECT_ROOT)),
                    "written_at": ingested_at.isoformat(),
                    "batch_date": str(batch_date),
                    "row_count": row_count,
                    "cycle_min": cycle_min,
                    "cycle_max": cycle_max,
                    "db_batch_id": db_batch_id,
                    "dry_run": dry_run,
                }

                log.info(
                    "    [OK ] %-40s  %3d rows  cycles %d–%d%s",
                    out_path.relative_to(dest_dir),
                    row_count,
                    cycle_min,
                    cycle_max,
                    "  [DRY-RUN]" if dry_run else "",
                )

                stats["batches_written"] += 1
                stats["rows_written"] += row_count

        # ── Persist manifest after each dataset (crash safety) ────────────
        if not dry_run:
            save_manifest(dest_dir, manifest)
            log.debug("Manifest updated: %s", _manifest_path(dest_dir))

    # ── Final summary ─────────────────────────────────────────────────────────
    _print_summary(stats, batch_size, batch_date, dest_dir, dry_run)

    if stats["errors"] > 0:
        log.error("%d error(s) occurred during ingest.", stats["errors"])
        sys.exit(1)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def _print_summary(
    stats: dict,
    batch_size: int,
    batch_date: date,
    dest_dir: Path,
    dry_run: bool,
) -> None:
    prefix = "[DRY-RUN] " if dry_run else ""
    print()
    print("=" * 56)
    print(f"  {prefix}INGEST SUMMARY")
    print("=" * 56)
    print(f"  Batch date        : {batch_date}")
    print(f"  Batch size        : {batch_size} cycles / unit")
    print(f"  Output dir        : {dest_dir}")
    print()
    print(f"  Source files read : {stats['files_processed']}")
    print(f"  Batches written   : {stats['batches_written']}")
    print(f"  Rows written      : {stats['rows_written']:,}")
    print(
        f"  Batches skipped   : {stats['batches_skipped']}  "
        f"(already in manifest — use --force to overwrite)"
    )
    print(f"  Rows skipped      : {stats['rows_skipped']:,}")
    print(f"  Errors            : {stats['errors']}")
    print("=" * 56)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bronze-layer ingestion: split NASA C-MAPSS files into "
            "partitioned Parquet batches under data/raw/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=ALL_DATASETS,
        default=None,
        metavar="FDxxx",
        help="Which C-MAPSS subsets to ingest. Defaults to all that are present in --source-dir.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("BATCH_SIZE", DEFAULT_BATCH_SIZE)),
        metavar="N",
        help="Number of cycles per batch per engine unit.",
    )
    parser.add_argument(
        "--batch-date",
        type=date.fromisoformat,
        default=date.today(),
        metavar="YYYY-MM-DD",
        help="Synthetic arrival date recorded in the manifest and dim_batch.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing raw NASA .txt source files.",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=DEFAULT_DEST_DIR,
        help="Root output directory for partitioned Parquet files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process and overwrite batches already recorded in the manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and split data but do not write Parquet files or touch the DB.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging (prints every batch skip/write).",
    )
    return parser.parse_args()


def main() -> None:
    # Force UTF-8 stdout (prevents cp1252 UnicodeEncodeError on Windows)
    import io as _io

    if hasattr(sys.stdout, "buffer"):
        sys.stdout = _io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )

    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # If no --dataset given, default to all datasets present in source_dir
    datasets = args.dataset or [
        ds for ds in ALL_DATASETS if (args.source_dir / f"train_{ds}.txt").exists()
    ]
    if not datasets:
        log.error(
            "No source files found in %s. Run `python ingest/download_data.py` first.",
            args.source_dir,
        )
        sys.exit(1)

    log.info(
        "Starting ingest  |  datasets=%s  batch_size=%d  dry_run=%s",
        datasets,
        args.batch_size,
        args.dry_run,
    )

    t0 = time.perf_counter()
    run_ingest(
        datasets=datasets,
        batch_size=args.batch_size,
        batch_date=args.batch_date,
        source_dir=args.source_dir,
        dest_dir=args.dest_dir,
        force=args.force,
        dry_run=args.dry_run,
    )
    elapsed = time.perf_counter() - t0
    log.info("Ingest complete in %.1fs", elapsed)


if __name__ == "__main__":
    main()
