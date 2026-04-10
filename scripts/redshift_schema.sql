-- ==========================================================
-- redshift_schema.sql
-- HealthPredict AI — Amazon Redshift Schema
-- Run this in Redshift Query Editor v2 after cluster is AVAILABLE
-- Database: healthdb   User: adminuser
-- ==========================================================

-- ── Schema ─────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS healthpredict;

-- ── Table 1: prediction_log ────────────────────────────────
-- Stores every real-time prediction from the Lambda function.
-- distkey(patient_id)  → efficient per-patient queries
-- sortkey(pred_timestamp) → chronological scan optimization
CREATE TABLE IF NOT EXISTS healthpredict.prediction_log (
    id                  VARCHAR(36)     DEFAULT CONCAT(GETDATE()::VARCHAR, '-', RANDOM()::VARCHAR),
    patient_id          VARCHAR(64)     NOT NULL,
    pred_timestamp      TIMESTAMP       NOT NULL DEFAULT GETDATE(),
    pregnancies         INTEGER,
    glucose             DOUBLE PRECISION,
    blood_pressure      DOUBLE PRECISION,
    skin_thickness      DOUBLE PRECISION,
    insulin             DOUBLE PRECISION,
    bmi                 DOUBLE PRECISION,
    diabetes_pedigree   DOUBLE PRECISION,
    age                 INTEGER,
    risk_score          DOUBLE PRECISION NOT NULL,
    risk_level          VARCHAR(8)       NOT NULL,   -- LOW / MEDIUM / HIGH
    model_version       VARCHAR(32)      DEFAULT 'v1',
    model_endpoint      VARCHAR(128),
    request_id          VARCHAR(64),
    created_at          TIMESTAMP        DEFAULT GETDATE()
)
DISTSTYLE KEY
DISTKEY (patient_id)
SORTKEY (pred_timestamp);

-- ── Table 2: patient_data_processed ───────────────────────
-- Loaded by the Glue ETL job — full processed dataset including
-- engineered features. Used for analytical queries via Athena + Redshift.
CREATE TABLE IF NOT EXISTS healthpredict.patient_data_processed (
    record_id           BIGINT          IDENTITY(1,1),
    pregnancies         DOUBLE PRECISION,
    glucose             DOUBLE PRECISION,
    blood_pressure      DOUBLE PRECISION,
    skin_thickness      DOUBLE PRECISION,
    insulin             DOUBLE PRECISION,
    bmi                 DOUBLE PRECISION,
    diabetes_pedigree   DOUBLE PRECISION,
    age                 DOUBLE PRECISION,
    outcome             INTEGER,
    bmi_category        VARCHAR(16),
    age_group           VARCHAR(16),
    glucose_risk        VARCHAR(20),
    glucose_bmi_interaction DOUBLE PRECISION,
    etl_timestamp       TIMESTAMP       DEFAULT GETDATE()
)
DISTSTYLE EVEN
SORTKEY (age, outcome);

-- ── Table 3: model_registry ────────────────────────────────
-- Versioned catalog of every trained model.
CREATE TABLE IF NOT EXISTS healthpredict.model_registry (
    model_id            VARCHAR(64)     NOT NULL,
    model_version       VARCHAR(32)     NOT NULL,
    training_job_name   VARCHAR(128),
    pipeline_arn        VARCHAR(256),
    model_s3_uri        VARCHAR(512),
    auc_score           DOUBLE PRECISION,
    f1_score            DOUBLE PRECISION,
    training_records    INTEGER,
    validation_records  INTEGER,
    endpoint_name       VARCHAR(128),
    approval_status     VARCHAR(32)     DEFAULT 'PendingManualApproval',
    is_active           BOOLEAN         DEFAULT FALSE,
    created_at          TIMESTAMP       DEFAULT GETDATE(),
    approved_at         TIMESTAMP,
    approved_by         VARCHAR(64),
    notes               VARCHAR(512),
    PRIMARY KEY (model_id)
)
DISTSTYLE ALL;

-- ── Table 4: daily_summary ────────────────────────────────
-- Pre-aggregated daily statistics (populated by stored procedure).
-- Reduces dashboard query latency.
CREATE TABLE IF NOT EXISTS healthpredict.daily_summary (
    summary_date        DATE            NOT NULL,
    total_predictions   INTEGER         DEFAULT 0,
    high_risk_count     INTEGER         DEFAULT 0,
    medium_risk_count   INTEGER         DEFAULT 0,
    low_risk_count      INTEGER         DEFAULT 0,
    avg_risk_score      DOUBLE PRECISION,
    avg_glucose         DOUBLE PRECISION,
    avg_bmi             DOUBLE PRECISION,
    avg_age             DOUBLE PRECISION,
    unique_patients     INTEGER         DEFAULT 0,
    last_updated        TIMESTAMP       DEFAULT GETDATE(),
    PRIMARY KEY (summary_date)
)
DISTSTYLE ALL
SORTKEY (summary_date);

-- ── View 1: v_risk_stats ──────────────────────────────────
-- Risk level distribution with patient demographics per category.
CREATE OR REPLACE VIEW healthpredict.v_risk_stats AS
SELECT
    risk_level,
    COUNT(*)                            AS total_predictions,
    ROUND(AVG(risk_score)::NUMERIC, 4)  AS avg_risk_score,
    ROUND(MIN(risk_score)::NUMERIC, 4)  AS min_risk_score,
    ROUND(MAX(risk_score)::NUMERIC, 4)  AS max_risk_score,
    ROUND(AVG(glucose)::NUMERIC, 1)     AS avg_glucose,
    ROUND(AVG(bmi)::NUMERIC, 1)         AS avg_bmi,
    ROUND(AVG(age)::NUMERIC, 1)         AS avg_age,
    COUNT(DISTINCT patient_id)          AS unique_patients
FROM healthpredict.prediction_log
GROUP BY risk_level
ORDER BY risk_level;

-- ── View 2: v_monthly_trend ──────────────────────────────
-- Day-by-day prediction counts and average risk score for last 30 days.
CREATE OR REPLACE VIEW healthpredict.v_monthly_trend AS
SELECT
    DATE(pred_timestamp)                          AS prediction_date,
    COUNT(*)                                      AS total_predictions,
    SUM(CASE WHEN risk_level = 'HIGH'   THEN 1 ELSE 0 END) AS high_risk,
    SUM(CASE WHEN risk_level = 'MEDIUM' THEN 1 ELSE 0 END) AS medium_risk,
    SUM(CASE WHEN risk_level = 'LOW'    THEN 1 ELSE 0 END) AS low_risk,
    ROUND(AVG(risk_score)::NUMERIC, 4)            AS avg_risk_score,
    COUNT(DISTINCT patient_id)                    AS unique_patients
FROM healthpredict.prediction_log
WHERE pred_timestamp >= DATEADD(day, -30, CURRENT_DATE)
GROUP BY DATE(pred_timestamp)
ORDER BY prediction_date DESC;

-- ── View 3: v_high_risk_patients ─────────────────────────
-- Ranked list of patients with at least one HIGH-risk prediction.
-- Useful for clinical follow-up prioritization.
CREATE OR REPLACE VIEW healthpredict.v_high_risk_patients AS
SELECT
    patient_id,
    COUNT(*)                                      AS total_visits,
    SUM(CASE WHEN risk_level = 'HIGH' THEN 1 ELSE 0 END) AS high_risk_count,
    ROUND(MAX(risk_score)::NUMERIC, 4)            AS max_risk_score,
    ROUND(AVG(risk_score)::NUMERIC, 4)            AS avg_risk_score,
    ROUND(AVG(glucose)::NUMERIC, 1)               AS avg_glucose,
    ROUND(AVG(bmi)::NUMERIC, 1)                   AS avg_bmi,
    ROUND(AVG(age)::NUMERIC, 0)                   AS avg_age,
    MIN(pred_timestamp)                           AS first_prediction,
    MAX(pred_timestamp)                           AS latest_prediction
FROM healthpredict.prediction_log
GROUP BY patient_id
HAVING SUM(CASE WHEN risk_level = 'HIGH' THEN 1 ELSE 0 END) > 0
ORDER BY max_risk_score DESC;

-- ── Stored Procedure: sp_update_daily_summary ────────────
-- Refreshes the daily_summary row for a given date.
-- Call after each day's predictions are complete.
CREATE OR REPLACE PROCEDURE healthpredict.sp_update_daily_summary(p_date DATE)
AS $$
DECLARE
    v_total     INTEGER;
    v_high      INTEGER;
    v_medium    INTEGER;
    v_low       INTEGER;
    v_avg_score DOUBLE PRECISION;
    v_avg_gluc  DOUBLE PRECISION;
    v_avg_bmi   DOUBLE PRECISION;
    v_avg_age   DOUBLE PRECISION;
    v_patients  INTEGER;
BEGIN
    SELECT
        COUNT(*),
        SUM(CASE WHEN risk_level = 'HIGH'   THEN 1 ELSE 0 END),
        SUM(CASE WHEN risk_level = 'MEDIUM' THEN 1 ELSE 0 END),
        SUM(CASE WHEN risk_level = 'LOW'    THEN 1 ELSE 0 END),
        AVG(risk_score),
        AVG(glucose),
        AVG(bmi),
        AVG(age::DOUBLE PRECISION),
        COUNT(DISTINCT patient_id)
    INTO v_total, v_high, v_medium, v_low,
         v_avg_score, v_avg_gluc, v_avg_bmi, v_avg_age, v_patients
    FROM healthpredict.prediction_log
    WHERE DATE(pred_timestamp) = p_date;

    DELETE FROM healthpredict.daily_summary WHERE summary_date = p_date;

    INSERT INTO healthpredict.daily_summary
        (summary_date, total_predictions, high_risk_count, medium_risk_count,
         low_risk_count, avg_risk_score, avg_glucose, avg_bmi, avg_age,
         unique_patients, last_updated)
    VALUES
        (p_date, COALESCE(v_total,0), COALESCE(v_high,0), COALESCE(v_medium,0),
         COALESCE(v_low,0), v_avg_score, v_avg_gluc, v_avg_bmi, v_avg_age,
         COALESCE(v_patients,0), GETDATE());
END;
$$ LANGUAGE plpgsql;

-- ── Initial model registry entry ─────────────────────────
INSERT INTO healthpredict.model_registry
    (model_id, model_version, approval_status, is_active, notes)
VALUES
    ('healthpredict-xgb-v1', 'v1', 'PendingManualApproval', FALSE,
     'Initial XGBoost 1.7 model — Pima Indians Diabetes Dataset');

-- ── Grants (for LabRole access) ───────────────────────────
GRANT ALL ON SCHEMA healthpredict TO adminuser;
GRANT ALL ON ALL TABLES IN SCHEMA healthpredict TO adminuser;

-- ── Verification Queries ──────────────────────────────────
-- Run these after executing this script to confirm everything created correctly:
--
-- SELECT table_name, table_type FROM information_schema.tables
-- WHERE table_schema = 'healthpredict' ORDER BY table_name;
-- Expected: 4 tables + 3 views = 7 rows
--
-- SELECT * FROM healthpredict.model_registry;
-- Expected: 1 row with version v1
--
-- CALL healthpredict.sp_update_daily_summary(CURRENT_DATE);
-- SELECT * FROM healthpredict.daily_summary;
-- Expected: 1 row with 0 counts (no predictions yet)
