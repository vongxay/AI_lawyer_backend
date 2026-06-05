-- =============================================================================
-- AI Lawyer Backend — Supabase Schema Migration
-- Run once against your Supabase project (SQL Editor or supabase db push)
-- =============================================================================

-- ── Extensions ────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;    -- for faster ILIKE searches


-- =============================================================================
-- CORE KNOWLEDGE BASE
-- =============================================================================

-- ── Laws / Statutes ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS laws (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    title           text NOT NULL,
    full_text       text NOT NULL,
    section         text,
    chapter         text,
    jurisdiction    text NOT NULL DEFAULT 'TH',
    year            int,
    status          text NOT NULL DEFAULT 'ACTIVE'
                        CHECK (status IN ('ACTIVE', 'AMENDED', 'REPEALED')),
    metadata        jsonb DEFAULT '{}',
    embedding       vector(1536),
    created_at      timestamptz DEFAULT now(),
    updated_at      timestamptz DEFAULT now()
);

-- ── Cases / Judgments ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cases (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    case_no         text NOT NULL UNIQUE,
    court           text NOT NULL,
    year            int NOT NULL,
    parties         text,
    summary         text NOT NULL,
    ruling          text NOT NULL,
    ratio_decidendi text,               -- key legal principle — most important chunk
    outcome         text CHECK (outcome IN ('plaintiff_won', 'defendant_won', 'settled', 'dismissed')),
    jurisdiction    text NOT NULL DEFAULT 'TH',
    metadata        jsonb DEFAULT '{}',
    embedding       vector(1536),
    created_at      timestamptz DEFAULT now(),
    updated_at      timestamptz DEFAULT now()
);

-- ── Case Law Graph ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS case_citations (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_case     uuid NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    cited_case      uuid REFERENCES cases(id),
    cited_statute   uuid REFERENCES laws(id),
    relationship    text NOT NULL
                        CHECK (relationship IN ('cites', 'overruled_by', 'distinguished', 'followed', 'applied')),
    year            int,
    notes           text,
    CONSTRAINT cited_something CHECK (cited_case IS NOT NULL OR cited_statute IS NOT NULL)
);

-- ── Legal Forms / Templates ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS legal_forms (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    form_type       text NOT NULL,
    title           text NOT NULL,
    content         text NOT NULL,
    jurisdiction    text NOT NULL DEFAULT 'TH',
    language        text NOT NULL DEFAULT 'TH',
    embedding       vector(1536),
    created_at      timestamptz DEFAULT now()
);


-- =============================================================================
-- USERS & TENANTS
-- =============================================================================

CREATE TABLE IF NOT EXISTS users (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    email           text UNIQUE NOT NULL,
    role            text NOT NULL DEFAULT 'client'
                        CHECK (role IN ('admin', 'lawyer', 'client', 'auditor')),
    tenant_id       uuid NOT NULL DEFAULT gen_random_uuid(),
    plan            text DEFAULT 'free',
    created_at      timestamptz DEFAULT now()
);


-- =============================================================================
-- CASE MANAGEMENT
-- =============================================================================

-- ── Case Memory ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS case_memory (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id         uuid NOT NULL UNIQUE,
    client_id       uuid REFERENCES users(id),
    tenant_id       uuid NOT NULL,
    case_type       text,
    facts_summary   text,
    jurisdiction    text NOT NULL DEFAULT 'TH',
    status          text DEFAULT 'active' CHECK (status IN ('active', 'closed', 'settled')),
    irac_history    jsonb DEFAULT '[]',
    arguments_used  jsonb DEFAULT '[]',
    strategies      jsonb DEFAULT '[]',
    key_citations   jsonb DEFAULT '[]',
    document_ids    uuid[],
    evidence_ids    uuid[],
    created_at      timestamptz DEFAULT now(),
    updated_at      timestamptz DEFAULT now(),
    last_accessed   timestamptz DEFAULT now()
);

-- ── Case Sessions ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS case_sessions (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         uuid REFERENCES users(id),
    case_id         uuid,
    tenant_id       uuid,
    messages        jsonb DEFAULT '[]',
    agents_used     text[],
    confidence      float,
    status          text DEFAULT 'active',
    created_at      timestamptz DEFAULT now(),
    updated_at      timestamptz DEFAULT now()
);

-- ── Documents ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      uuid REFERENCES case_sessions(id),
    case_id         uuid,
    tenant_id       uuid NOT NULL,
    file_name       text NOT NULL,
    file_path       text,
    file_type       text,
    file_size_bytes int,
    analysis_result jsonb,
    created_at      timestamptz DEFAULT now()
);

-- ── Evidence ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS evidence (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id         uuid,
    tenant_id       uuid NOT NULL,
    file_name       text NOT NULL,
    file_path       text,
    evidence_type   text CHECK (evidence_type IN ('image', 'audio', 'video', 'document', 'email', 'other')),
    analysis        jsonb,
    admissibility   jsonb,
    legal_relevance text CHECK (legal_relevance IN ('HIGH', 'MEDIUM', 'LOW', 'NONE')),
    created_at      timestamptz DEFAULT now()
);


-- =============================================================================
-- AUDIT & COMPLIANCE
-- =============================================================================

-- ── Audit Log ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_log (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             uuid,
    tenant_id           uuid,
    agent               text NOT NULL,
    query_hash          text NOT NULL,      -- SHA-256, never store raw query
    confidence          float,
    agents_used         text[],
    processing_time_ms  int,
    escalated_to_expert boolean DEFAULT false,
    ts                  timestamptz DEFAULT now()
);

-- ── Expert Review Queue ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS expert_reviews (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      text,
    user_id         uuid,
    reason          text NOT NULL,
    confidence      float,
    query_preview   text,                   -- first 200 chars, no PII
    status          text DEFAULT 'pending' CHECK (status IN ('pending', 'in_review', 'resolved')),
    reviewer_id     uuid,
    resolution      text,
    created_at      timestamptz DEFAULT now(),
    resolved_at     timestamptz
);

-- ── Citations Log ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS citations_log (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      text,
    citation_ref    text NOT NULL,
    status          text NOT NULL CHECK (status IN ('VERIFIED', 'OUTDATED', 'UNVERIFIED', 'REJECTED')),
    verified_at     timestamptz DEFAULT now()
);

-- ── Feedback ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS feedback (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id          text NOT NULL,
    user_id             uuid,
    rating              int CHECK (rating BETWEEN 1 AND 5),
    comment             text,
    corrected_answer    text,
    created_at          timestamptz DEFAULT now()
);


-- =============================================================================
-- INDEXES
-- =============================================================================

-- Vector (HNSW — fastest ANN for legal search)
CREATE INDEX IF NOT EXISTS laws_hnsw  ON laws  USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS cases_hnsw ON cases USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS forms_hnsw ON legal_forms USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Full-text search (BM25 keyword search)
CREATE INDEX IF NOT EXISTS laws_fts  ON laws
    USING gin(to_tsvector('simple', coalesce(title,'') || ' ' || coalesce(full_text,'')));
CREATE INDEX IF NOT EXISTS cases_fts ON cases
    USING gin(to_tsvector('simple', coalesce(summary,'') || ' ' || coalesce(ruling,'')));

-- Trigram index for fast ILIKE citation lookups
CREATE INDEX IF NOT EXISTS laws_title_trgm  ON laws  USING gin(title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS cases_case_no_trgm ON cases USING gin(case_no gin_trgm_ops);

-- Foreign key / lookup indexes
CREATE INDEX IF NOT EXISTS case_citations_source  ON case_citations(source_case);
CREATE INDEX IF NOT EXISTS case_citations_cited   ON case_citations(cited_case);
CREATE INDEX IF NOT EXISTS case_memory_case_id    ON case_memory(case_id);
CREATE INDEX IF NOT EXISTS case_memory_tenant     ON case_memory(tenant_id);
CREATE INDEX IF NOT EXISTS audit_log_ts           ON audit_log(ts DESC);
CREATE INDEX IF NOT EXISTS audit_log_tenant       ON audit_log(tenant_id);
CREATE INDEX IF NOT EXISTS expert_reviews_status  ON expert_reviews(status);


-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- ── Hybrid Legal Search (Semantic + BM25 + RRF) ───────────────────────────────
CREATE OR REPLACE FUNCTION hybrid_legal_search(
    query_embedding  vector(1536) DEFAULT NULL,
    query_text       text DEFAULT '',
    p_jurisdiction   text DEFAULT NULL,
    match_count      int  DEFAULT 10,
    rrf_k            int  DEFAULT 60
) RETURNS TABLE (
    id           uuid,
    type         text,
    title        text,
    content      text,
    section      text,
    metadata     jsonb,
    final_score  float
) LANGUAGE sql AS $$
    WITH semantic AS (
        SELECT id, 'law' AS type, title,
               full_text AS content, section, metadata,
               ROW_NUMBER() OVER (
                   ORDER BY CASE WHEN query_embedding IS NOT NULL
                            THEN embedding <=> query_embedding
                            ELSE 1.0 END
               ) AS rank
        FROM laws
        WHERE (p_jurisdiction IS NULL OR jurisdiction = p_jurisdiction)
          AND embedding IS NOT NULL
        LIMIT 50
    ),
    keyword AS (
        SELECT id, 'law' AS type, title,
               full_text AS content, section, metadata,
               ROW_NUMBER() OVER (
                   ORDER BY ts_rank(
                       to_tsvector('simple', coalesce(title,'') || ' ' || coalesce(full_text,'')),
                       plainto_tsquery('simple', query_text)
                   ) DESC
               ) AS rank
        FROM laws
        WHERE query_text != ''
          AND to_tsvector('simple', coalesce(title,'') || ' ' || coalesce(full_text,''))
              @@ plainto_tsquery('simple', query_text)
          AND (p_jurisdiction IS NULL OR jurisdiction = p_jurisdiction)
        LIMIT 50
    ),
    -- Same for cases
    semantic_cases AS (
        SELECT id, 'case' AS type, case_no AS title,
               summary AS content, NULL AS section, metadata,
               ROW_NUMBER() OVER (
                   ORDER BY CASE WHEN query_embedding IS NOT NULL
                            THEN embedding <=> query_embedding
                            ELSE 1.0 END
               ) AS rank
        FROM cases
        WHERE (p_jurisdiction IS NULL OR jurisdiction = p_jurisdiction)
          AND embedding IS NOT NULL
        LIMIT 30
    ),
    all_semantic AS (
        SELECT * FROM semantic
        UNION ALL
        SELECT * FROM semantic_cases
    ),
    all_keyword AS (
        SELECT * FROM keyword
    ),
    fused AS (
        SELECT
            COALESCE(s.id, k.id)           AS id,
            COALESCE(s.type, k.type)       AS type,
            COALESCE(s.title, k.title)     AS title,
            COALESCE(s.content, k.content) AS content,
            COALESCE(s.section, k.section) AS section,
            COALESCE(s.metadata, k.metadata) AS metadata,
            COALESCE(1.0 / (rrf_k + s.rank), 0.0) +
            COALESCE(1.0 / (rrf_k + k.rank), 0.0) AS score
        FROM all_semantic s
        FULL OUTER JOIN all_keyword k ON s.id = k.id
    )
    SELECT id, type, title, content, section, metadata, score AS final_score
    FROM fused
    ORDER BY score DESC
    LIMIT match_count;
$$;

-- ── Recursive Precedent Chain ─────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION get_precedent_chain(
    start_case_id uuid,
    max_depth     int DEFAULT 3
) RETURNS TABLE (
    case_id      uuid,
    depth        int,
    relationship text,
    path         uuid[]
) LANGUAGE sql AS $$
    WITH RECURSIVE chain AS (
        SELECT source_case, cited_case, relationship,
               1 AS depth,
               ARRAY[source_case] AS path
        FROM case_citations
        WHERE source_case = start_case_id
          AND cited_case IS NOT NULL

        UNION ALL

        SELECT cc.source_case, cc.cited_case, cc.relationship,
               c.depth + 1,
               c.path || cc.source_case
        FROM case_citations cc
        JOIN chain c ON cc.source_case = c.cited_case
        WHERE c.depth < max_depth
          AND cc.source_case != ALL(c.path)
          AND cc.cited_case IS NOT NULL
    )
    SELECT cited_case, depth, relationship, path FROM chain;
$$;

-- ── Auto-update updated_at ────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

CREATE OR REPLACE TRIGGER laws_updated_at
    BEFORE UPDATE ON laws
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE OR REPLACE TRIGGER cases_updated_at
    BEFORE UPDATE ON cases
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE OR REPLACE TRIGGER case_memory_updated_at
    BEFORE UPDATE ON case_memory
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- =============================================================================
-- ROW LEVEL SECURITY (Tenant Isolation)
-- =============================================================================

ALTER TABLE case_memory    ENABLE ROW LEVEL SECURITY;
ALTER TABLE case_sessions  ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents      ENABLE ROW LEVEL SECURITY;
ALTER TABLE evidence       ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log      ENABLE ROW LEVEL SECURITY;

-- Case Memory: tenant can only see own cases
CREATE POLICY case_memory_tenant_isolation ON case_memory
    USING (tenant_id = (
        SELECT tenant_id FROM users WHERE id = auth.uid()
    ));

-- Documents: tenant isolation
CREATE POLICY documents_tenant_isolation ON documents
    USING (tenant_id = (
        SELECT tenant_id FROM users WHERE id = auth.uid()
    ));

-- Evidence: tenant isolation
CREATE POLICY evidence_tenant_isolation ON evidence
    USING (tenant_id = (
        SELECT tenant_id FROM users WHERE id = auth.uid()
    ));

-- Audit log: admins see all, others see own
CREATE POLICY audit_log_admin_or_own ON audit_log
    USING (
        user_id = auth.uid()
        OR EXISTS (
            SELECT 1 FROM users
            WHERE id = auth.uid() AND role = 'admin'
        )
    );


-- =============================================================================
-- SEED: minimal test data (remove in production)
-- =============================================================================

-- Uncomment to seed with a test admin user:
-- INSERT INTO users (id, email, role, tenant_id, plan)
-- VALUES (gen_random_uuid(), 'admin@ailawyer.test', 'admin', gen_random_uuid(), 'enterprise')
-- ON CONFLICT (email) DO NOTHING;
