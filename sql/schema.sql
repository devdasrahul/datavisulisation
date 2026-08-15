-- =============================================================================
-- sql/schema.sql
-- Predictive Maintenance ETL Pipeline — PostgreSQL Warehouse DDL
--
-- PRD Reference: Section 7 (Data Model)
-- Star schema: 1 fact table + 3 dimension tables + 1 operational log table
--
-- Execution: idempotent — safe to run on an already-provisioned database.
--   All tables use CREATE TABLE IF NOT EXISTS.
--   All indexes use CREATE INDEX IF NOT EXISTS.
--   Run via: python db/init_db.py
--
-- Table dependency order (top → bottom):
--   1. dim_equipment    — engine unit registry
--   2. dim_sensor       — sensor metadata & expected value bounds
--   3. dim_batch        — ingest batch audit log
--   4. fact_sensor_readings — central fact table (FKs to all three dims)
--   5. pipeline_run_log — data quality check results per pipeline run
-- =============================================================================


-- ---------------------------------------------------------------------------
-- 0. Extensions
-- ---------------------------------------------------------------------------
-- btree_gin enables GIN indexing on btree-compatible types (int, text),
-- useful for multi-column quality-check queries on pipeline_run_log.
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS pg_trgm;


-- ---------------------------------------------------------------------------
-- 1. dim_equipment
--    One row per physical engine unit.
--    Populated during ingest. Updated as new cycles are observed.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_equipment (
    unit_id               INTEGER       PRIMARY KEY,          -- engine unit number (matches NASA file col 1)
    dataset_source        VARCHAR(10)   NOT NULL,             -- e.g. 'FD001', 'FD002', 'FD003', 'FD004'
    max_cycles_observed   INTEGER       DEFAULT 0,            -- highest cycle number seen so far (updated on load)
    failure_cycle         INTEGER       DEFAULT NULL,         -- cycle at which this unit failed (NULL if unknown / still running)
    first_seen_at         TIMESTAMPTZ   NOT NULL DEFAULT NOW(), -- when this unit was first ingested
    last_updated_at       TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE  dim_equipment                IS 'Engine unit registry — one row per turbofan unit in the NASA C-MAPSS dataset.';
COMMENT ON COLUMN dim_equipment.unit_id        IS 'Engine unit identifier (col 1 in raw NASA files).';
COMMENT ON COLUMN dim_equipment.dataset_source IS 'C-MAPSS subset this unit belongs to (FD001–FD004).';
COMMENT ON COLUMN dim_equipment.failure_cycle  IS 'Cycle at which the unit reached end-of-life. Used to compute RUL.';

-- Support fast lookups by dataset when filtering to a single FD subset
CREATE INDEX IF NOT EXISTS idx_dim_equipment_dataset
    ON dim_equipment (dataset_source);


-- ---------------------------------------------------------------------------
-- 2. dim_sensor
--    One row per sensor channel (21 total in C-MAPSS).
--    Pre-seeded — sensor definitions do not change at runtime.
--    expected_min / expected_max are used by the data quality checks to
--    flag out-of-range readings in the fact table.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_sensor (
    sensor_id       SMALLINT      PRIMARY KEY,           -- 1–21 (matches sensor_N column index)
    sensor_name     VARCHAR(30)   NOT NULL UNIQUE,       -- short identifier, e.g. 'T24', 'Nf', 'P30'
    description     TEXT,                               -- human-readable description from NASA docs
    unit_of_measure VARCHAR(20),                        -- engineering unit, e.g. 'degR', 'psia', 'rpm'
    expected_min    NUMERIC(12,4),                      -- lower bound for DQ out-of-range check
    expected_max    NUMERIC(12,4)                       -- upper bound for DQ out-of-range check
);

COMMENT ON TABLE  dim_sensor             IS 'Sensor channel definitions for the 21 NASA C-MAPSS measurement channels.';
COMMENT ON COLUMN dim_sensor.expected_min IS 'Below this value a reading is flagged as out-of-range by quality checks.';
COMMENT ON COLUMN dim_sensor.expected_max IS 'Above this value a reading is flagged as out-of-range by quality checks.';

-- No additional indexes needed — sensor_id is PK (already B-tree indexed)
-- and the table has only 21 rows so full-table scans are negligible.


-- ---------------------------------------------------------------------------
-- 3. dim_batch
--    One row per synthetic ingest batch execution.
--    Created by the ingest step. Updated to 'loaded' or 'failed' by the
--    load step. Drives idempotency — the load step only processes batches
--    where status = 'staged'.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_batch (
    batch_id     BIGSERIAL     PRIMARY KEY,
    batch_date   DATE          NOT NULL,                -- synthetic "arrival" date (--batch-date CLI arg)
    ingested_at  TIMESTAMPTZ   NOT NULL DEFAULT NOW(), -- when the ingest step created this row
    source_file  VARCHAR(60)   NOT NULL,               -- e.g. 'train_FD001.txt'
    dataset      VARCHAR(10)   NOT NULL,               -- e.g. 'FD001'
    row_count    INTEGER       DEFAULT 0,              -- number of raw rows in this batch
    status       VARCHAR(10)   NOT NULL DEFAULT 'staged'  -- 'staged' | 'loaded' | 'failed'
                 CHECK (status IN ('staged', 'loaded', 'failed')),
    loaded_at    TIMESTAMPTZ   DEFAULT NULL,           -- set when status transitions to 'loaded'
    notes        TEXT          DEFAULT NULL            -- optional error message if status = 'failed'
);

COMMENT ON TABLE  dim_batch           IS 'Ingest batch audit log — one row per synthetic batch execution.';
COMMENT ON COLUMN dim_batch.status    IS 'Lifecycle state: staged → loaded (success) or failed (error).';
COMMENT ON COLUMN dim_batch.batch_date IS 'Synthetic arrival date used to replay the static dataset as daily batches.';

-- Freshness check queries filter by ingested_at DESC. Partial index on loaded batches
CREATE INDEX IF NOT EXISTS idx_dim_batch_ingested_at
    ON dim_batch (ingested_at DESC);

-- Load step queries: SELECT batches WHERE status = 'staged'
CREATE INDEX IF NOT EXISTS idx_dim_batch_status
    ON dim_batch (status)
    WHERE status = 'staged';

-- Dataset-level batch history
CREATE INDEX IF NOT EXISTS idx_dim_batch_dataset
    ON dim_batch (dataset, batch_date DESC);


-- ---------------------------------------------------------------------------
-- 4. fact_sensor_readings
--    Central fact table — one row per (engine unit, cycle, sensor).
--    Grain: one measurement per sensor per cycle per unit.
--
--    The UNIQUE constraint on (unit_id, cycle, sensor_id) enables
--    ON CONFLICT DO UPDATE upserts from the load step (idempotency).
--
--    rolling_avg_7 and rate_of_change are computed in the Gold transform
--    and stored here so SQL views can read them without re-computing via
--    heavy window functions on every dashboard query.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_sensor_readings (
    reading_id      BIGSERIAL     PRIMARY KEY,

    -- Dimension FKs
    unit_id         INTEGER       NOT NULL
                    REFERENCES dim_equipment (unit_id) ON DELETE RESTRICT,
    sensor_id       SMALLINT      NOT NULL
                    REFERENCES dim_sensor    (sensor_id) ON DELETE RESTRICT,
    batch_id        BIGINT        NOT NULL
                    REFERENCES dim_batch     (batch_id)  ON DELETE RESTRICT,

    -- Core measurement
    cycle           INTEGER       NOT NULL  CHECK (cycle > 0),  -- operational cycle (time axis)
    reading_value   NUMERIC(14,6) NOT NULL,                     -- raw sensor reading

    -- Pre-computed Gold-layer features (from Polars transform)
    rolling_avg_7   NUMERIC(14,6) DEFAULT NULL,  -- 7-cycle rolling mean (per unit + sensor)
    rate_of_change  NUMERIC(14,6) DEFAULT NULL,  -- diff vs. previous cycle (per unit + sensor)

    -- Audit
    loaded_at       TIMESTAMPTZ   NOT NULL DEFAULT NOW(),

    -- Upsert guard — one reading per (unit, cycle, sensor)
    CONSTRAINT uq_fact_unit_cycle_sensor
        UNIQUE (unit_id, cycle, sensor_id)
);

COMMENT ON TABLE  fact_sensor_readings               IS 'Central fact table — one row per sensor reading per cycle per engine unit.';
COMMENT ON COLUMN fact_sensor_readings.cycle         IS 'Operational cycle number (time axis within a unit lifetime).';
COMMENT ON COLUMN fact_sensor_readings.rolling_avg_7 IS '7-cycle rolling average, computed in Polars Gold transform (ROWS BETWEEN 6 PRECEDING AND CURRENT ROW).';
COMMENT ON COLUMN fact_sensor_readings.rate_of_change IS 'Difference vs. previous cycle value (lag-1 diff), per unit+sensor.';

-- ── Indexes tuned for window-function and dashboard query patterns ──────────

-- Primary window-function index: PARTITION BY unit_id, sensor_id ORDER BY cycle
-- Covers: rolling avg, rate-of-change, RUL, degradation rank views
CREATE INDEX IF NOT EXISTS idx_fact_unit_sensor_cycle
    ON fact_sensor_readings (unit_id, sensor_id, cycle);

-- RUL + per-unit timeline queries: WHERE unit_id = X ORDER BY cycle
CREATE INDEX IF NOT EXISTS idx_fact_unit_cycle
    ON fact_sensor_readings (unit_id, cycle);

-- Batch-level queries: WHERE batch_id = X (used by DQ checks + load idempotency)
CREATE INDEX IF NOT EXISTS idx_fact_batch_id
    ON fact_sensor_readings (batch_id);

-- Out-of-range rate queries: filter on sensor_id + loaded_at window
CREATE INDEX IF NOT EXISTS idx_fact_sensor_loaded
    ON fact_sensor_readings (sensor_id, loaded_at DESC);

-- Partial index for recent data (last 30 days) — speeds up dashboard freshness queries
-- Note: this index stays small even as the table grows.
CREATE INDEX IF NOT EXISTS idx_fact_recent
    ON fact_sensor_readings (loaded_at DESC);


-- ---------------------------------------------------------------------------
-- 5. pipeline_run_log
--    One row per data quality check per pipeline execution.
--    Written by pipeline/quality.py at the end of every run.
--    Read by the dashboard "Pipeline Health" panel.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline_run_log (
    log_id          BIGSERIAL    PRIMARY KEY,
    batch_date      DATE         NOT NULL,                -- synthetic batch date this run processed
    run_started_at  TIMESTAMPTZ  NOT NULL,
    run_finished_at TIMESTAMPTZ  DEFAULT NULL,
    check_name      VARCHAR(60)  NOT NULL,               -- e.g. 'freshness_sla', 'null_rate_per_column'
    metric_value    NUMERIC(14,6) DEFAULT NULL,          -- measured value (e.g. 0.012 = 1.2% null rate)
    threshold       NUMERIC(14,6) DEFAULT NULL,          -- configured threshold for this check
    passed          BOOLEAN      NOT NULL,               -- TRUE = check passed, FALSE = failed
    notes           TEXT         DEFAULT NULL            -- optional detail / error message
);

COMMENT ON TABLE  pipeline_run_log            IS 'Per-check DQ results written at the end of every pipeline execution.';
COMMENT ON COLUMN pipeline_run_log.check_name IS 'Identifier for the quality check: freshness_sla | cycle_gap_check | null_rate_per_column | out_of_range_rate | batch_row_count | referential_integrity.';
COMMENT ON COLUMN pipeline_run_log.passed     IS 'Whether the check passed its configured threshold. FALSE triggers a non-zero exit code in CI.';

-- Dashboard health-trend queries: ORDER BY run_started_at DESC, GROUP BY check_name
CREATE INDEX IF NOT EXISTS idx_log_run_started
    ON pipeline_run_log (run_started_at DESC);

CREATE INDEX IF NOT EXISTS idx_log_check_name_run
    ON pipeline_run_log (check_name, run_started_at DESC);

-- GIN index for fast text searches on check_name + notes (ad-hoc debugging)
CREATE INDEX IF NOT EXISTS idx_log_gin_check
    ON pipeline_run_log USING GIN (check_name gin_trgm_ops)
    WITH (fastupdate = off);


-- ---------------------------------------------------------------------------
-- 6. Seed data — dim_sensor
--    21 C-MAPSS sensor definitions, sourced from the NASA dataset
--    description paper (Saxena & Goebel, 2008).
--    expected_min / expected_max are approximate operational bounds.
--    they are used by the out-of-range DQ check and the v_out_of_range_rate view.
--    INSERT OR IGNORE pattern: safe to re-run on an already-seeded database.
-- ---------------------------------------------------------------------------
INSERT INTO dim_sensor (sensor_id, sensor_name, description, unit_of_measure, expected_min, expected_max)
VALUES
    -- Physical / thermodynamic sensors
    ( 1, 'T2',         'Total temperature at fan inlet',                      'degR',    408.0,   492.0),
    ( 2, 'T24',        'Total temperature at LPC outlet',                     'degR',    600.0,   645.0),
    ( 3, 'T30',        'Total temperature at HPC outlet',                     'degR',   1560.0,  1620.0),
    ( 4, 'T50',        'Total temperature at LPT outlet',                     'degR',   1390.0,  1500.0),
    ( 5, 'P2',         'Pressure at fan inlet',                               'psia',     14.0,    15.0),
    ( 6, 'P15',        'Total pressure in bypass-duct',                       'psia',     20.0,    24.0),
    ( 7, 'P30',        'Total pressure at HPC outlet',                        'psia',    550.0,   600.0),
    -- Speed / ratio sensors
    ( 8, 'Nf',         'Physical fan speed',                                  'rpm',    2385.0,  2390.0),
    ( 9, 'Nc',         'Physical core speed',                                 'rpm',    9030.0,  9090.0),
    (10, 'epr',        'Engine pressure ratio (P50/P2)',                      'ratio',     1.0,    1.5),
    -- Combustion / flow sensors
    (11, 'Ps30',       'Static pressure at HPC outlet',                       'psia',    46.0,    50.0),
    (12, 'phi',        'Ratio of fuel flow to Ps30',                          'pph/psi',  520.0,   540.0),
    (13, 'NRf',        'Corrected fan speed',                                 'rpm',    2388.0,  2392.0),
    (14, 'NRc',        'Corrected core speed',                                'rpm',    8130.0,  8160.0),
    (15, 'BPR',        'Bypass ratio',                                        'ratio',     8.3,    8.6),
    (16, 'farB',       'Burner fuel-air ratio',                               'ratio',     0.02,   0.04),
    (17, 'htBleed',    'Bleed enthalpy',                                      'BTU/lb',  388.0,   400.0),
    -- Demand / virtual sensors
    (18, 'Nf_dmd',     'Demanded fan speed',                                  'rpm',    2388.0,  2392.0),
    (19, 'PCNfR_dmd',  'Demanded corrected fan speed',                        'rpm',     100.0,   100.0),
    (20, 'W31',        'HPT coolant bleed flow',                              'lbm/s',    38.0,    40.0),
    (21, 'W32',        'LPT coolant bleed flow',                              'lbm/s',    23.0,    24.0)
ON CONFLICT (sensor_id) DO NOTHING;

-- =============================================================================
-- End of schema.sql
-- =============================================================================
