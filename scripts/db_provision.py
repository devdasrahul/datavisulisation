"""
scripts/db_provision.py
────────────────────────
Phase 1 — Provision the PostgreSQL schema.

Runs all DDL files in sql/schema/ in order against the target Postgres instance.
Idempotent — safe to re-run on an already-provisioned database.

Usage:
  python scripts/db_provision.py

Reads DATABASE_URL from .env (or environment).

TODO (Phase 1):
  - Implement run_ddl_files() to execute sql/schema/*.sql in sorted order
  - Add a --check flag to verify all tables exist without modifying the DB
"""
