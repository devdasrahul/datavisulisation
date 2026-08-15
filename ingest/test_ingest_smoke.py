"""Quick smoke test for ingest_batches.py — run with: python ingest/test_ingest_smoke.py"""

import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from ingest.ingest_batches import (
    RAW_COLUMNS,
    load_manifest,
    make_batch_key,
    save_manifest,
    split_into_batches,
    write_parquet,
)

# --- Build a tiny 3-unit, 55-cycle synthetic DataFrame ---
rows = []
for unit in [1, 2, 3]:
    for cycle in range(1, 56):
        row = [unit, cycle, 0.0, 0.0, 100.0] + [
            float(random.randint(100, 999)) for _ in range(21)
        ]
        rows.append(row)

schema = {
    c: (pl.Int32 if c in ("unit_id", "cycle") else pl.Float64) for c in RAW_COLUMNS
}
df = pl.DataFrame(rows, schema=schema, orient="row")

# --- Test split_into_batches ---
unit1 = df.filter(pl.col("unit_id") == 1)
batches = list(split_into_batches(unit1, batch_size=20))

print(f"Unit 1 ({len(unit1)} cycles) -> {len(batches)} batches")
assert len(batches) == 3, f"Expected 3 batches, got {len(batches)}"

for bi, bdf in batches:
    cmin = bdf["cycle"].min()
    cmax = bdf["cycle"].max()
    print(f"  Batch {bi}: {len(bdf)} rows, cycles {cmin}-{cmax}")

# Verify batch sizes: 20, 20, 15
assert len(batches[0][1]) == 20
assert len(batches[1][1]) == 20
assert len(batches[2][1]) == 15

# --- Test write_parquet in dry-run mode (no files written) ---
with tempfile.TemporaryDirectory() as tmpdir:
    dest = Path(tmpdir)
    for bi, bdf in batches:
        p = write_parquet(
            bdf,
            dest,
            "FD001",
            1,
            bi,
            datetime.now(tz=timezone.utc),
            "train_FD001.txt",
            dry_run=True,
        )
        print(f"  [dry-run] Would write: {p.name}")
    assert not any(dest.rglob("*.parquet")), "Dry-run wrote files unexpectedly"
    print("  dry-run: no files written (correct)")

# --- Test actual Parquet write ---
with tempfile.TemporaryDirectory() as tmpdir:
    dest = Path(tmpdir)
    written = []
    for bi, bdf in batches:
        p = write_parquet(
            bdf,
            dest,
            "FD001",
            1,
            bi,
            datetime.now(tz=timezone.utc),
            "train_FD001.txt",
            dry_run=False,
        )
        written.append(p)

    parquet_files = list(dest.rglob("*.parquet"))
    assert (
        len(parquet_files) == 3
    ), f"Expected 3 Parquet files, got {len(parquet_files)}"

    # Read one back and verify columns + lineage metadata
    sample = pl.read_parquet(parquet_files[0])
    assert "unit_id" in sample.columns
    assert "_ingested_at" in sample.columns
    assert "_source_file" in sample.columns
    assert "_batch_idx" in sample.columns
    print(f"  Parquet columns: {sample.columns}")
    print(f"  Parquet shape  : {sample.shape}")
    print("  Parquet write  : OK")

# --- Test manifest round-trip ---
with tempfile.TemporaryDirectory() as tmpdir:
    dest = Path(tmpdir)
    m = load_manifest(dest)
    key = make_batch_key("FD001", 1, 0)
    m["batches"][key] = {"row_count": 20}
    save_manifest(dest, m)

    m2 = load_manifest(dest)
    assert key in m2["batches"], "Manifest round-trip FAILED"
    assert m2["batches"][key]["row_count"] == 20
    print(f"  Manifest key   : {key}")
    print("  Manifest round-trip: OK")

print()
print("All smoke tests passed.")
