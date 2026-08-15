"""
ingest/download_data.py

Grabs the NASA C-MAPSS dataset and drops it into data/source/. 
Also prints out a quick summary of the files so we can verify the download worked and the columns look right.

The raw text files don't have headers and are space-delimited. 
Columns:
  1     : unit_id (engine number)
  2     : cycle (time index)
  3-5   : op_settings
  6-26  : sensor measurements (temps, speeds, pressures, etc.)
  27-28 : (train files only) useless trailing zeros that we'll drop later.

RUL_*.txt just holds the actual remaining useful life for the test engines.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

# ── Constants ─────────────────────────────────────────────────────────────────

# Primary download URL — NASA open data portal (no auth required)
NASA_ZIP_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"

# Fallback mirror in case the NASA portal is unreachable
FALLBACK_ZIP_URL = (
    "https://ti.arc.nasa.gov/m/project/prognostic-repository/CMAPSSData.zip"
)

# Expected files inside the zip (NASA packages all files flat or in CMAPSSData/)
TRAIN_FILES = [f"train_FD00{i}.txt" for i in range(1, 5)]
RUL_FILES = [f"RUL_FD00{i}.txt" for i in range(1, 5)]
TARGET_FILES = TRAIN_FILES + RUL_FILES

# Canonical column names for the 26 usable columns in train/test files
SENSOR_NAMES = [
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

# Root of the project — resolve relative to this file regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = PROJECT_ROOT / "data" / "source"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _progress_bar(desc: str, total: int, width: int = 40) -> callable:
    """Minimal in-terminal progress bar that works without tqdm."""

    def update(downloaded: int) -> None:
        pct = min(downloaded / total, 1.0) if total else 0
        filled = int(width * pct)
        bar = "█" * filled + "░" * (width - filled)
        mb_dl = downloaded / 1_048_576
        mb_tot = total / 1_048_576
        sys.stdout.write(f"\r  {desc}: [{bar}] {mb_dl:.1f}/{mb_tot:.1f} MB")
        sys.stdout.flush()
        if downloaded >= total:
            sys.stdout.write("\n")

    return update


def _download_zip(url: str) -> bytes:
    """Stream-download a zip from *url* with a progress bar. Returns raw bytes."""
    print(f"  → Connecting to: {url}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("Content-Length", 0))
    update = _progress_bar("Downloading", total)
    buf = io.BytesIO()
    downloaded = 0

    for chunk in resp.iter_content(chunk_size=65_536):
        buf.write(chunk)
        downloaded += len(chunk)
        update(downloaded)

    buf.seek(0)
    return buf.read()


def _extract_targets(zip_bytes: bytes, dest: Path) -> list[Path]:
    """
    Pulls just the target files out of the zip.
    NASA sometimes nests these in a sub-folder, so this just finds them regardless of depth.
    """
    extracted: list[Path] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        name_map: dict[str, str] = {}  # basename → full zip path
        for name in zf.namelist():
            basename = Path(name).name
            if basename in TARGET_FILES:
                name_map[basename] = name

        missing = set(TARGET_FILES) - set(name_map.keys())
        if missing:
            print(f"\n  ⚠  Files not found inside zip: {sorted(missing)}")
            print(
                "     The zip structure may have changed. Check the NASA portal manually."
            )

        for basename, zip_path in name_map.items():
            out_path = dest / basename
            if out_path.exists():
                print(f"  ✓  {basename:30s} already exists — skipped")
                extracted.append(out_path)
                continue
            data = zf.read(zip_path)
            out_path.write_bytes(data)
            print(f"  ✓  {basename:30s} extracted  ({len(data)/1_048_576:.2f} MB)")
            extracted.append(out_path)

    return extracted


# ── Summary printer ───────────────────────────────────────────────────────────


def _summarise_file(path: Path) -> None:
    """
    Quick sanity check: load the file, print its shape and column names,
    so we can confirm the layout didn't change unexpectedly.
    """
    print(f"\n{'─'*60}")
    print(f"  📄  {path.name}")
    print(f"{'─'*60}")

    is_rul = path.name.startswith("RUL_")

    if is_rul:
        # RUL files: single column — true remaining useful life per test unit
        df = pd.read_csv(path, sep=r"\s+", header=None, names=["rul"])
        print(f"  Shape   : {df.shape[0]:,} rows × {df.shape[1]} column")
        print(f"  Columns : {list(df.columns)}")
        print(f"  Dtypes  : {dict(df.dtypes)}")
        print(
            f"  RUL range: min={df['rul'].min():.0f}  max={df['rul'].max():.0f}  "
            f"mean={df['rul'].mean():.1f}"
        )
    else:
        # Train/test files: 26+ space-delimited columns, no header
        raw = pd.read_csv(path, sep=r"\s+", header=None)
        n_cols = raw.shape[1]

        # The NASA txt files sometimes include 1–2 trailing all-zero columns.
        # We keep only the 26 meaningful columns and rename them.
        usable = raw.iloc[:, :26].copy()
        usable.columns = SENSOR_NAMES

        print(
            f"  Raw cols  : {n_cols}  (26 usable, {n_cols - 26} trailing zero-col(s) dropped)"
        )
        print(f"  Shape     : {usable.shape[0]:,} rows × {usable.shape[1]} columns")
        print(f"\n  Column layout:")
        print(f"    unit_id         — engine unit number")
        print(f"    cycle           — operational cycle (time index)")
        print(f"    op_setting_1–3  — 3 flight condition settings")
        print(f"    sensor_1–21     — 21 raw sensor measurements")
        print(f"\n  Units observed: {sorted(usable['unit_id'].unique())}")
        print(f"  Cycles per unit:")
        cycle_stats = usable.groupby("unit_id")["cycle"].max()
        print(
            f"    min={cycle_stats.min()}  max={cycle_stats.max()}  "
            f"mean={cycle_stats.mean():.1f}"
        )

        print(f"\n  Sensor value ranges (min / max):")
        sensor_cols = [c for c in SENSOR_NAMES if c.startswith("sensor_")]
        range_df = (
            usable[sensor_cols]
            .agg(["min", "max"])
            .T.rename(columns={"min": "min", "max": "max"})
        )
        print(
            range_df.to_string(
                float_format=lambda x: f"{x:>10.4f}",
                col_space=12,
            )
        )

    print(f"\n  File size: {path.stat().st_size / 1_048_576:.2f} MB")


# ── Main ──────────────────────────────────────────────────────────────────────


def download(dest: Path | None = None, datasets: list[str] | None = None) -> None:
    """
    Download and extract the C-MAPSS dataset, then print a per-file summary.

    Args:
        dest:     Directory to write files into (default: data/source/).
        datasets: Which FDxxx subsets to process in the summary,
                  e.g. ["FD001"]. None = all four.
    """
    dest = dest or DEFAULT_DEST
    dest.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  NASA C-MAPSS Dataset Downloader")
    print("=" * 60)

    # ── Check which files are already present ─────────────────────────────
    already_have = [f for f in TARGET_FILES if (dest / f).exists()]
    need_download = [f for f in TARGET_FILES if f not in already_have]

    if not need_download:
        print(f"\n  All {len(TARGET_FILES)} target files already present in:")
        print(f"  {dest}\n  Skipping download.\n")
    else:
        print(
            f"\n  {len(already_have)} file(s) already cached, "
            f"{len(need_download)} to download.\n"
        )

        # ── Attempt primary URL, then fallback ────────────────────────────
        zip_bytes: bytes | None = None
        for url in (NASA_ZIP_URL, FALLBACK_ZIP_URL):
            try:
                zip_bytes = _download_zip(url)
                print(f"  Download complete ({len(zip_bytes)/1_048_576:.1f} MB).\n")
                break
            except requests.RequestException as exc:
                print(f"\n  ✗  Failed ({url}):\n     {exc}")
                print("  Trying fallback URL...\n")

        if zip_bytes is None:
            print(
                "\n  ✗  Both download URLs failed.\n"
                "  Manual download instructions:\n"
                "    1. Visit: https://www.nasa.gov/content/"
                "prognostics-center-of-excellence-data-set-repository\n"
                "    2. Download the C-MAPSS zip and extract into data/source/\n"
            )
            sys.exit(1)

        # ── Extract target files ──────────────────────────────────────────
        print("  Extracting target files...")
        _extract_targets(zip_bytes, dest)

    # ── Print per-file summary ────────────────────────────────────────────
    filter_tags = {f"FD00{d[-1]}" for d in datasets} if datasets else None

    files_to_summarise = sorted(
        p
        for f in TARGET_FILES
        if (filter_tags is None or any(tag in f for tag in filter_tags))
        and (p := dest / f).exists()
    )

    if not files_to_summarise:
        print("  ⚠  No files found to summarise. Check the dest directory.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  FILE SUMMARIES  ({len(files_to_summarise)} files)")
    print(f"{'='*60}")

    for path in files_to_summarise:
        _summarise_file(path)

    print(f"\n{'='*60}")
    print(f"  ✅  Done. Files saved to: {dest}")
    print(f"{'='*60}\n")


def main() -> None:
    # Windows terminals sometimes crash on UTF-8 prints if cp1252 is the default.
    # This forces UTF-8 output to avoid that mess.
    import io

    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )

    parser = argparse.ArgumentParser(
        description="Download NASA C-MAPSS dataset and print per-file summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Directory to save downloaded files.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=["FD001", "FD002", "FD003", "FD004"],
        default=None,
        metavar="FDxxx",
        help="Restrict the post-download summary to specific subsets "
        "(e.g. --dataset FD001 FD002). All four are always downloaded.",
    )
    args = parser.parse_args()
    download(dest=args.dest, datasets=args.dataset)


if __name__ == "__main__":
    main()
