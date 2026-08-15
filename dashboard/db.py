"""
dashboard/db.py
────────────────
Phase 3 — Postgres connection helper for Streamlit.

Provides:
  - get_connection(): @st.cache_resource connection pool
  - query(sql): runs a SQL string, returns a pandas DataFrame

TODO (Phase 3):
  - Implement get_connection() reading DATABASE_URL from st.secrets / env
  - Add error handling and retry logic for free-tier cold starts
"""
