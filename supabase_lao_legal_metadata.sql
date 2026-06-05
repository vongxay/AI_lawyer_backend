-- Lao legal AI metadata and review workflow
-- Apply after ai_lawyer_database.sql / supabase_migration.sql.

-- 1) Rich Lao-law metadata on knowledge tables.
ALTER TABLE laws
    ADD COLUMN IF NOT EXISTS language text DEFAULT 'lo',
    ADD COLUMN IF NOT EXISTS official_source_url text,
    ADD COLUMN IF NOT EXISTS source_authority text DEFAULT 'uploaded',
    ADD COLUMN IF NOT EXISTS law_no text,
    ADD COLUMN IF NOT EXISTS article text,
    ADD COLUMN IF NOT EXISTS gazette_date date,
    ADD COLUMN IF NOT EXISTS effective_date date,
    ADD COLUMN IF NOT EXISTS amended_by text,
    ADD COLUMN IF NOT EXISTS repealed_by text,
    ADD COLUMN IF NOT EXISTS review_status text DEFAULT 'pending_review',
    ADD COLUMN IF NOT EXISTS reviewed_by uuid,
    ADD COLUMN IF NOT EXISTS reviewed_at timestamptz,
    ADD COLUMN IF NOT EXISTS review_notes text;

ALTER TABLE cases
    ADD COLUMN IF NOT EXISTS language text DEFAULT 'lo',
    ADD COLUMN IF NOT EXISTS official_source_url text,
    ADD COLUMN IF NOT EXISTS source_authority text DEFAULT 'uploaded',
    ADD COLUMN IF NOT EXISTS review_status text DEFAULT 'pending_review',
    ADD COLUMN IF NOT EXISTS reviewed_by uuid,
    ADD COLUMN IF NOT EXISTS reviewed_at timestamptz,
    ADD COLUMN IF NOT EXISTS review_notes text;

ALTER TABLE legal_forms
    ADD COLUMN IF NOT EXISTS official_source_url text,
    ADD COLUMN IF NOT EXISTS source_authority text DEFAULT 'uploaded',
    ADD COLUMN IF NOT EXISTS review_status text DEFAULT 'pending_review',
    ADD COLUMN IF NOT EXISTS reviewed_by uuid,
    ADD COLUMN IF NOT EXISTS reviewed_at timestamptz,
    ADD COLUMN IF NOT EXISTS review_notes text;

CREATE INDEX IF NOT EXISTS laws_lao_authority_idx
    ON laws(jurisdiction, review_status, source_authority, effective_date DESC);

CREATE INDEX IF NOT EXISTS laws_lao_article_idx
    ON laws(jurisdiction, law_no, article);

-- 2) Evaluation benchmark tables for legal QA quality.
CREATE TABLE IF NOT EXISTS legal_eval_cases (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    jurisdiction text NOT NULL DEFAULT 'laos',
    language text NOT NULL DEFAULT 'lo',
    category text NOT NULL DEFAULT 'general',
    question text NOT NULL,
    expected_answer text,
    required_citations jsonb NOT NULL DEFAULT '[]',
    fact_pattern jsonb NOT NULL DEFAULT '{}',
    difficulty text NOT NULL DEFAULT 'medium',
    status text NOT NULL DEFAULT 'active',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS legal_eval_runs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_name text NOT NULL,
    jurisdiction text NOT NULL DEFAULT 'laos',
    model text,
    total_cases int NOT NULL DEFAULT 0,
    passed_cases int NOT NULL DEFAULT 0,
    avg_confidence float,
    citation_pass_rate float,
    results jsonb NOT NULL DEFAULT '[]',
    created_by uuid,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- 3) RLS defense in depth. Backend service role bypasses these; frontend REST should still be protected.
ALTER TABLE legal_eval_cases ENABLE ROW LEVEL SECURITY;
ALTER TABLE legal_eval_runs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS legal_eval_cases_admin_all ON legal_eval_cases;
CREATE POLICY legal_eval_cases_admin_all ON legal_eval_cases
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM users
            WHERE users.id = auth.uid()
              AND users.role IN ('admin', 'super_admin')
        )
    );

DROP POLICY IF EXISTS legal_eval_runs_admin_all ON legal_eval_runs;
CREATE POLICY legal_eval_runs_admin_all ON legal_eval_runs
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM users
            WHERE users.id = auth.uid()
              AND users.role IN ('admin', 'super_admin')
        )
    );
