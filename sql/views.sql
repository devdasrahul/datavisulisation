-- sql/views.sql
--
-- These views power the Streamlit dashboard. 
-- I moved the heavy analytical lifting into Postgres so the frontend stays snappy.

-- =============================================================================
-- 1. v_rolling_avg_by_sensor
-- Answers: What's the smoothed trend of each sensor reading over time?
-- Note: Polars computes this at ingest, but doing it in SQL lets us tweak the 
-- window sizes dynamically later if we need to.
-- =============================================================================
CREATE OR REPLACE VIEW v_rolling_avg_by_sensor AS
SELECT 
    unit_id,
    sensor_id,
    cycle,
    reading_value,
    AVG(reading_value) OVER (
        PARTITION BY unit_id, sensor_id 
        ORDER BY cycle 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_avg_7
FROM fact_sensor_readings;


-- =============================================================================
-- 2. v_degradation_rank
-- Answers: Which engine units are degrading the fastest?
-- Method: Computes the slope of sensor_11 (static pressure at HPC outlet) over 
-- the last 20 cycles for each unit. I used sensor_11 because it's a really solid 
-- indicator of engine health in the C-MAPSS dataset.
-- =============================================================================
CREATE OR REPLACE VIEW v_degradation_rank AS
WITH recent_readings AS (
    SELECT 
        unit_id, 
        cycle, 
        rolling_avg_7
    FROM fact_sensor_readings
    WHERE sensor_id = 11
),
max_cycles AS (
    SELECT unit_id, MAX(cycle) AS max_cycle
    FROM recent_readings
    GROUP BY unit_id
),
last_20_cycles AS (
    SELECT 
        r.unit_id, 
        r.cycle, 
        r.rolling_avg_7
    FROM recent_readings r
    JOIN max_cycles m ON r.unit_id = m.unit_id
    WHERE r.cycle > m.max_cycle - 20
),
slopes AS (
    SELECT 
        unit_id,
        REGR_SLOPE(rolling_avg_7, cycle) AS degradation_slope
    FROM last_20_cycles
    GROUP BY unit_id
)
SELECT 
    unit_id,
    degradation_slope,
    RANK() OVER (ORDER BY degradation_slope DESC) AS degradation_rank
FROM slopes;


-- =============================================================================
-- 3. v_rul_estimate
-- Answers: Roughly how many cycles does each unit have left before failure?
-- Method: A naive heuristic combining metadata and the degradation slope. 
-- (TODO: We should replace this with a proper ML model eventually, but this works 
-- as a baseline for the dashboard).
-- =============================================================================
CREATE OR REPLACE VIEW v_rul_estimate AS
SELECT 
    e.unit_id,
    e.max_cycles_observed,
    dr.degradation_slope,
    CASE 
        WHEN dr.degradation_slope > 0.001 THEN GREATEST(0, CAST(50.0 / dr.degradation_slope AS INTEGER))
        WHEN e.max_cycles_observed < 150 THEN 250 - e.max_cycles_observed
        ELSE GREATEST(10, 250 - e.max_cycles_observed)
    END AS estimated_rul_cycles
FROM dim_equipment e
LEFT JOIN v_degradation_rank dr ON e.unit_id = dr.unit_id;


-- =============================================================================
-- 4. v_sensor_out_of_range_rate
-- Answers: Are sensors starting to throw weird readings?
-- =============================================================================
CREATE OR REPLACE VIEW v_sensor_out_of_range_rate AS
WITH all_time AS (
    SELECT 
        s.sensor_name,
        COUNT(*) AS total_readings,
        SUM(CASE WHEN f.reading_value < s.expected_min OR f.reading_value > s.expected_max THEN 1 ELSE 0 END) AS oor_count
    FROM fact_sensor_readings f
    JOIN dim_sensor s ON f.sensor_id = s.sensor_id
    GROUP BY s.sensor_name
),
last_24h AS (
    SELECT 
        s.sensor_name,
        COUNT(*) AS recent_total,
        SUM(CASE WHEN f.reading_value < s.expected_min OR f.reading_value > s.expected_max THEN 1 ELSE 0 END) AS recent_oor_count
    FROM fact_sensor_readings f
    JOIN dim_sensor s ON f.sensor_id = s.sensor_id
    WHERE f.loaded_at >= NOW() - INTERVAL '24 hours'
    GROUP BY s.sensor_name
)
SELECT 
    a.sensor_name,
    ROUND(a.oor_count * 100.0 / NULLIF(a.total_readings, 0), 2) AS all_time_oor_pct,
    COALESCE(l.recent_total, 0) AS recent_readings,
    ROUND(COALESCE(l.recent_oor_count, 0) * 100.0 / NULLIF(l.recent_total, 0), 2) AS recent_oor_pct
FROM all_time a
LEFT JOIN last_24h l ON a.sensor_name = l.sensor_name;


-- =============================================================================
-- 5. v_batch_freshness
-- Answers: Is the ETL pipeline actually running and delivering data on time?
-- =============================================================================
CREATE OR REPLACE VIEW v_batch_freshness AS
SELECT 
    dataset_source AS dataset,
    MAX(last_updated_at) AS last_data_received,
    EXTRACT(EPOCH FROM (NOW() - MAX(last_updated_at))) / 3600.0 AS hours_since_update,
    CASE 
        WHEN NOW() - MAX(last_updated_at) < INTERVAL '26 hours' THEN 'FRESH'
        ELSE 'STALE'
    END AS freshness_status
FROM dim_equipment
GROUP BY dataset_source;


-- =============================================================================
-- 6. v_pipeline_health_trend
-- Answers: Is the data quality improving or degrading over time?
-- =============================================================================
CREATE OR REPLACE VIEW v_pipeline_health_trend AS
SELECT 
    DATE(run_started_at) AS run_date,
    COUNT(*) AS total_checks,
    SUM(CASE WHEN passed THEN 1 ELSE 0 END) AS passed_checks,
    SUM(CASE WHEN passed THEN 0 ELSE 1 END) AS failed_checks,
    ROUND(SUM(CASE WHEN passed THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 2) AS pass_rate_pct
FROM pipeline_run_log
GROUP BY DATE(run_started_at)
ORDER BY run_date DESC;
