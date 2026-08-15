"""
Smoke test for transform/transform_batches.py
Run with: python -m transform.test_transform_smoke
"""

import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from transform.transform_batches import (
    RAW_COLUMNS,
    SENSOR_COLS,
    BRONZE_META_COLS,
    clean_batch,
    compute_gold,
    load_manifest,
    save_manifest,
    run_transform,
    RunStats,
)

RNG = random.Random(42)


def make_synthetic_bronze(dest: Path, dataset: str, unit_id: int, n_cycles: int = 55) -> Path:
    """Create a realistic Bronze Parquet with proper column structure."""
    cycles = list(range(1, n_cycles + 1))
    n = n_cycles

    data: dict = {
        "unit_id": pl.Series([unit_id] * n, dtype=pl.Int32),
        "cycle":   pl.Series(cycles, dtype=pl.Int32),
        "op_setting_1": pl.Series([RNG.uniform(0, 0.5) for _ in range(n)], dtype=pl.Float64),
        "op_setting_2": pl.Series([RNG.uniform(0, 0.5) for _ in range(n)], dtype=pl.Float64),
        "op_setting_3": pl.Series([100.0] * n, dtype=pl.Float64),
    }

    # sensor_1: in-range except cycle 10 (index 9) which is deliberately OOR
    s1_vals = [RNG.uniform(420.0, 480.0) for _ in range(n)]
    if n > 9:
        s1_vals[9] = 999.0   # out-of-range at cycle 10
    data["sensor_1"] = pl.Series(s1_vals, dtype=pl.Float64)
    data["sensor_2"] = pl.Series([RNG.uniform(610.0, 640.0) for _ in range(n)], dtype=pl.Float64)
    data["sensor_3"] = pl.Series([RNG.uniform(1565.0, 1615.0) for _ in range(n)], dtype=pl.Float64)
    for i in range(4, 22):
        data[f"sensor_{i}"] = pl.Series(
            [float(RNG.randint(100, 900)) for _ in range(n)], dtype=pl.Float64
        )

    # Bronze metadata columns
    data["_ingested_at"] = pl.Series(["2026-08-14T06:00:00+00:00"] * n, dtype=pl.Utf8)
    data["_source_file"] = pl.Series([f"train_{dataset}.txt"] * n,       dtype=pl.Utf8)
    data["_batch_idx"]   = pl.Series([0] * n, dtype=pl.Int32)

    df = pl.DataFrame(data)

    # Add one exact duplicate row (unit_id=1, cycle=5 → index 4)
    dup_row = df.slice(4, 1)
    df = pl.concat([df, dup_row])

    out_dir = dest / f"dataset={dataset}" / f"unit={unit_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "batch=20260814T060000_000000.parquet"
    # Use BytesIO round-trip to avoid Polars sink_parquet issue on some builds
    import io as _io
    buf = _io.BytesIO()
    df.write_parquet(buf)
    buf.seek(0)
    out_path.write_bytes(buf.read())
    return out_path


# ── Test 1: clean_batch ───────────────────────────────────────────────────────
print("Test 1: clean_batch (Bronze -> Silver)")
with tempfile.TemporaryDirectory() as tmpdir:
    raw_dir   = Path(tmpdir) / "raw"
    clean_dir = Path(tmpdir) / "cleaned"

    bronze_path = make_synthetic_bronze(raw_dir, "FD001", unit_id=1, n_cycles=55)

    report = clean_batch(
        bronze_path=bronze_path,
        clean_dir=clean_dir,
        dataset="FD001",
        unit_id=1,
        dry_run=False,
    )

    assert report.error is None, f"clean_batch error: {report.error}"
    assert report.rows_in == 56,     f"Expected 56 rows in (55 + 1 dup), got {report.rows_in}"
    assert report.rows_out == 55,    f"Expected 55 rows out (dup dropped), got {report.rows_out}"
    assert report.duplicates_dropped == 1, f"Expected 1 dup dropped, got {report.duplicates_dropped}"
    assert "sensor_1" in report.anomaly_counts, "Expected sensor_1 anomaly not detected"

    # Verify Silver file exists and has correct structure
    silver_files = list(clean_dir.rglob("*.parquet"))
    assert len(silver_files) == 1, f"Expected 1 Silver file, got {len(silver_files)}"

    silver_df = pl.read_parquet(silver_files[0])
    assert "is_anomaly" in silver_df.columns, "is_anomaly column missing from Silver"
    assert len(silver_df) == 55
    # Bronze metadata cols must be stripped
    for meta_col in BRONZE_META_COLS:
        assert meta_col not in silver_df.columns, f"Bronze meta col {meta_col} not stripped"
    # All 26 raw cols must be present
    for col in RAW_COLUMNS:
        assert col in silver_df.columns, f"Missing column: {col}"
    # Check anomaly flag is correct (cycle 10 had OOR sensor_1)
    anomaly_rows = silver_df.filter(pl.col("is_anomaly"))
    assert len(anomaly_rows) > 0, "No anomaly rows found (expected cycle=10 to be flagged)"

    print(f"  rows_in={report.rows_in}  rows_out={report.rows_out}  "
          f"dupes_dropped={report.duplicates_dropped}  anomalies={report.anomaly_counts}")
    print("  Silver file columns:", silver_df.columns)
    print("  PASS")


# ── Test 2: compute_gold ──────────────────────────────────────────────────────
print("\nTest 2: compute_gold (Silver -> Gold)")
with tempfile.TemporaryDirectory() as tmpdir:
    raw_dir   = Path(tmpdir) / "raw"
    clean_dir = Path(tmpdir) / "cleaned"
    agg_dir   = Path(tmpdir) / "aggregated"

    # Produce Silver first
    bronze_path = make_synthetic_bronze(raw_dir, "FD001", unit_id=1, n_cycles=55)
    clean_batch(bronze_path, clean_dir, "FD001", 1, dry_run=False)

    gold_rows = compute_gold(
        clean_dir=clean_dir,
        agg_dir=agg_dir,
        dataset="FD001",
        unit_id=1,
        dry_run=False,
        transformed_at=datetime.now(tz=timezone.utc),
    )

    # 55 cycles * 21 sensors = 1155 rows
    assert gold_rows == 55 * 21, f"Expected {55*21} gold rows, got {gold_rows}"

    gold_files = list(agg_dir.rglob("*.parquet"))
    assert len(gold_files) == 1, f"Expected 1 Gold file, got {len(gold_files)}"

    gold_df = pl.read_parquet(gold_files[0])
    assert "rolling_avg_7"  in gold_df.columns, "rolling_avg_7 missing from Gold"
    assert "rate_of_change" in gold_df.columns, "rate_of_change missing from Gold"
    assert "sensor_id"      in gold_df.columns, "sensor_id missing from Gold"
    assert "sensor_name"    in gold_df.columns, "sensor_name missing from Gold"

    # Spot-check rolling avg for sensor_1 on unit_id=1: first 7 rows should converge
    s1 = gold_df.filter(
        (pl.col("unit_id") == 1) & (pl.col("sensor_id") == 1)
    ).sort("cycle")
    assert s1["rolling_avg_7"][0] is not None, "rolling_avg_7 null at cycle 1 (min_periods=1 broken)"
    assert s1["rate_of_change"][0] is None,    "rate_of_change should be null at cycle 1"
    assert s1["rate_of_change"][1] is not None, "rate_of_change should have a value at cycle 2"

    print(f"  gold_rows={gold_rows}  ({55}cx{21}s)")
    print(f"  Gold columns: {gold_df.columns}")
    print(f"  sensor_1 rolling_avg_7 at cycle 1: {s1['rolling_avg_7'][0]:.4f}")
    print(f"  sensor_1 rate_of_change at cycle 1: {s1['rate_of_change'][0]}")
    print(f"  sensor_1 rate_of_change at cycle 2: {s1['rate_of_change'][1]:.4f}")
    print("  PASS")


# ── Test 3: idempotency via manifest ──────────────────────────────────────────
print("\nTest 3: run_transform idempotency")
with tempfile.TemporaryDirectory() as tmpdir:
    raw_dir   = Path(tmpdir) / "raw"
    clean_dir = Path(tmpdir) / "cleaned"
    agg_dir   = Path(tmpdir) / "aggregated"

    make_synthetic_bronze(raw_dir, "FD001", unit_id=1, n_cycles=30)

    # First run
    stats1 = run_transform(raw_dir, clean_dir, agg_dir, ["FD001"], force=False, dry_run=False)
    assert stats1.batches_transformed == 1
    assert stats1.batches_skipped    == 0

    # Second run — should be a no-op
    stats2 = run_transform(raw_dir, clean_dir, agg_dir, ["FD001"], force=False, dry_run=False)
    assert stats2.batches_transformed == 0
    assert stats2.batches_skipped    == 1
    print(f"  Run 1: transformed={stats1.batches_transformed}  skipped={stats1.batches_skipped}")
    print(f"  Run 2: transformed={stats2.batches_transformed}  skipped={stats2.batches_skipped}")
    print("  PASS")


print()
print("All transform smoke tests PASSED.")
