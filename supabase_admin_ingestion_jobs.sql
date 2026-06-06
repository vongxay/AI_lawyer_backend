-- Admin ingestion job history.
-- Apply after the base schema and before using the admin ingestion page.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       uuid,
    job_type        text NOT NULL,
    source_type     text NOT NULL,
    source_url      text,
    file_name       text,
    status          text NOT NULL DEFAULT 'pending',
    progress        int NOT NULL DEFAULT 0,
    total_items     int,
    processed_items int DEFAULT 0,
    error_count     int DEFAULT 0,
    errors          jsonb NOT NULL DEFAULT '[]'::jsonb,
    config          jsonb NOT NULL DEFAULT '{}'::jsonb,
    result          jsonb NOT NULL DEFAULT '{}'::jsonb,
    started_at      timestamptz,
    completed_at    timestamptz,
    created_by      uuid,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ingestion_jobs_tenant_created_idx
    ON ingestion_jobs(tenant_id, created_at DESC);

CREATE INDEX IF NOT EXISTS ingestion_jobs_status_idx
    ON ingestion_jobs(status, created_at DESC);

CREATE INDEX IF NOT EXISTS ingestion_jobs_result_idx
    ON ingestion_jobs USING gin(result);

ALTER TABLE ingestion_jobs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS ingestion_jobs_admin_read_write ON ingestion_jobs;
CREATE POLICY ingestion_jobs_admin_read_write ON ingestion_jobs
    FOR ALL TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM users
            WHERE users.id = auth.uid()
              AND users.role IN ('admin', 'super_admin')
              AND (
                  ingestion_jobs.tenant_id IS NULL
                  OR users.tenant_id = ingestion_jobs.tenant_id
                  OR users.role = 'super_admin'
              )
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM users
            WHERE users.id = auth.uid()
              AND users.role IN ('admin', 'super_admin')
              AND (
                  ingestion_jobs.tenant_id IS NULL
                  OR users.tenant_id = ingestion_jobs.tenant_id
                  OR users.role = 'super_admin'
              )
        )
    );
