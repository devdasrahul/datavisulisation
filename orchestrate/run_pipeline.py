"""
orchestrate/run_pipeline.py

The main controller for the ETL pipeline. 
Runs everything in sequence: Ingest -> Transform -> Load -> Quality Checks.

It logs everything to `pipeline.log` and the terminal. If any step fails, 
it throws an error and stops immediately so we don't accidentally load bad data.
"""

import logging
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = PROJECT_ROOT / "pipeline.log"

# Configure dual logging (console + file)
logger = logging.getLogger("orchestrator")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# File Handler
fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
fh.setFormatter(formatter)
logger.addHandler(fh)

# Console Handler
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)


STAGES = [
    {"name": "1. INGEST (Bronze)", "cmd": [sys.executable, "ingest/ingest_batches.py"]},
    {
        "name": "2. TRANSFORM (Silver/Gold)",
        "cmd": [sys.executable, "transform/transform_batches.py"],
    },
    {
        "name": "3. LOAD (Postgres DWH)",
        "cmd": [sys.executable, "load/load_to_postgres.py"],
    },
    {"name": "4. QUALITY CHECKS", "cmd": [sys.executable, "quality/run_checks.py"]},
]


def run_stage(stage: dict) -> bool:
    """Spins up a script in a subprocess and pipes its output to our central logger."""
    name = stage["name"]
    cmd = stage["cmd"]

    logger.info(f"STARTING: {name}")

    try:
        # Run process, pipe stdout and stderr to the log
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )

        # Log the output from the subprocess
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                logger.info(f"  | {line}")

        logger.info(f"COMPLETED: {name}\n")
        return True

    except subprocess.CalledProcessError as e:
        # Log the error output
        if e.stdout:
            for line in e.stdout.strip().split("\n"):
                logger.error(f"  | {line}")

        logger.error(f"FAILED: {name} (Exit code: {e.returncode})\n")
        return False
    except FileNotFoundError:
        logger.error(f"FAILED: {name} (Could not find script: {' '.join(cmd)})\n")
        return False


def main():
    logger.info("=" * 60)
    logger.info("STARTING ETL PIPELINE RUN")
    logger.info("=" * 60)

    for stage in STAGES:
        success = run_stage(stage)
        if not success:
            logger.error("PIPELINE HALTED due to failure in stage: " + stage["name"])
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
