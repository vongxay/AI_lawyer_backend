-- Agentic RAG chunk store for large legal documents.
-- Apply after ai_lawyer_database.sql and supabase_lao_legal_metadata.sql.

CREATE EXTENSION IF NOT EXISTS vector;

-- Review metadata used by the admin approval workflow.
ALTER TABLE laws
    ADD COLUMN IF NOT EXISTS review_status text DEFAULT 'pending_review',
    ADD COLUMN IF NOT EXISTS reviewed_by uuid,
    ADD COLUMN IF NOT EXISTS reviewed_at timestamptz,
    ADD COLUMN IF NOT EXISTS review_notes text;

ALTER TABLE cases
    ADD COLUMN IF NOT EXISTS review_status text DEFAULT 'pending_review',
    ADD COLUMN IF NOT EXISTS reviewed_by uuid,
    ADD COLUMN IF NOT EXISTS reviewed_at timestamptz,
    ADD COLUMN IF NOT EXISTS review_notes text;

ALTER TABLE legal_forms
    ADD COLUMN IF NOT EXISTS review_status text DEFAULT 'pending_review',
    ADD COLUMN IF NOT EXISTS reviewed_by uuid,
    ADD COLUMN IF NOT EXISTS reviewed_at timestamptz,
    ADD COLUMN IF NOT EXISTS review_notes text;

CREATE TABLE IF NOT EXISTS document_chunks (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       uuid,
    source_table    text NOT NULL CHECK (source_table IN ('laws', 'cases', 'legal_forms')),
    source_id       uuid NOT NULL,
    document_type   text NOT NULL,
    jurisdiction    text,
    title           text NOT NULL,
    chunk_index     int NOT NULL,
    section_ref     text,
    content         text NOT NULL,
    token_count     int NOT NULL DEFAULT 0,
    status          text NOT NULL DEFAULT 'pending',
    review_status   text NOT NULL DEFAULT 'pending_review',
    metadata        jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding       vector(1536),
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (source_table, source_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS document_chunks_source_idx
    ON document_chunks(source_table, source_id, chunk_index);

CREATE INDEX IF NOT EXISTS document_chunks_jurisdiction_status_idx
    ON document_chunks(jurisdiction, status, review_status);

CREATE INDEX IF NOT EXISTS document_chunks_metadata_idx
    ON document_chunks USING gin(metadata);

CREATE INDEX IF NOT EXISTS document_chunks_fts_idx
    ON document_chunks USING gin(
        to_tsvector('simple', coalesce(title, '') || ' ' || coalesce(section_ref, '') || ' ' || coalesce(content, ''))
    );

CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw
    ON document_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS document_chunks_read ON document_chunks;
CREATE POLICY document_chunks_read ON document_chunks
    FOR SELECT TO authenticated
    USING (
        tenant_id IS NULL
        OR EXISTS (
            SELECT 1 FROM users
            WHERE users.id = auth.uid()
              AND users.tenant_id = document_chunks.tenant_id
        )
        OR EXISTS (
            SELECT 1 FROM users
            WHERE users.id = auth.uid()
              AND users.role IN ('admin', 'super_admin')
        )
    );

DROP POLICY IF EXISTS document_chunks_admin_write ON document_chunks;
CREATE POLICY document_chunks_admin_write ON document_chunks
    FOR ALL TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM users
            WHERE users.id = auth.uid()
              AND users.role IN ('admin', 'super_admin')
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM users
            WHERE users.id = auth.uid()
              AND users.role IN ('admin', 'super_admin')
        )
    );

-- Backfill one coarse chunk per existing document. New ingestion writes fine-grained chunks.
INSERT INTO document_chunks (
    tenant_id, source_table, source_id, document_type, jurisdiction, title,
    chunk_index, section_ref, content, token_count, status, review_status, metadata, embedding
)
SELECT
    l.tenant_id,
    'laws',
    l.id,
    l.doc_type::text,
    l.jurisdiction::text,
    l.title,
    0,
    coalesce(l.section_number, l.metadata->>'article'),
    l.full_text,
    greatest(1, length(l.full_text) / 4),
    l.status::text,
    coalesce(l.review_status, l.metadata->>'review_status', 'approved'),
    l.metadata || jsonb_build_object('backfilled', true, 'source_url', l.source_url),
    l.embedding
FROM laws l
WHERE l.full_text IS NOT NULL AND length(trim(l.full_text)) > 0
ON CONFLICT (source_table, source_id, chunk_index) DO NOTHING;

INSERT INTO document_chunks (
    tenant_id, source_table, source_id, document_type, jurisdiction, title,
    chunk_index, section_ref, content, token_count, status, review_status, metadata, embedding
)
SELECT
    c.tenant_id,
    'cases',
    c.id,
    coalesce(c.doc_type::text, 'case_law'),
    c.jurisdiction::text,
    c.case_no,
    0,
    NULL,
    coalesce(c.ratio_decidendi, c.ruling, c.summary, c.full_text),
    greatest(1, length(coalesce(c.ratio_decidendi, c.ruling, c.summary, c.full_text, '')) / 4),
    c.status::text,
    coalesce(c.review_status, c.metadata->>'review_status', 'approved'),
    c.metadata || jsonb_build_object('backfilled', true, 'source_url', c.source_url),
    c.embedding
FROM cases c
WHERE length(trim(coalesce(c.ratio_decidendi, c.ruling, c.summary, c.full_text, ''))) > 0
ON CONFLICT (source_table, source_id, chunk_index) DO NOTHING;

INSERT INTO document_chunks (
    tenant_id, source_table, source_id, document_type, jurisdiction, title,
    chunk_index, section_ref, content, token_count, status, review_status, metadata, embedding
)
SELECT
    lf.tenant_id,
    'legal_forms',
    lf.id,
    coalesce(lf.form_type, 'form'),
    lf.jurisdiction::text,
    lf.title,
    0,
    NULL,
    lf.content,
    greatest(1, length(lf.content) / 4),
    CASE WHEN lf.is_active THEN 'active' ELSE 'archived' END,
    coalesce(lf.review_status, 'approved'),
    jsonb_build_object('backfilled', true, 'form_type', lf.form_type, 'tags', lf.tags),
    lf.embedding
FROM legal_forms lf
WHERE lf.content IS NOT NULL AND length(trim(lf.content)) > 0
ON CONFLICT (source_table, source_id, chunk_index) DO NOTHING;

CREATE OR REPLACE FUNCTION public.hybrid_document_chunk_search(
    query_text          text,
    query_embedding     vector(1536) DEFAULT NULL,
    p_jurisdiction      text DEFAULT NULL,
    p_status            text DEFAULT 'active',
    p_review_status     text DEFAULT 'approved',
    p_tenant_id         uuid DEFAULT NULL,
    match_count         int DEFAULT 10,
    rrf_k               int DEFAULT 60
)
RETURNS TABLE (
    id              uuid,
    chunk_id        uuid,
    source_table    text,
    title           text,
    content         text,
    doc_type        text,
    jurisdiction    text,
    status          text,
    year            int,
    metadata        jsonb,
    section         text,
    semantic_rank   bigint,
    keyword_rank    bigint,
    final_score     double precision
)
LANGUAGE sql
STABLE
SECURITY INVOKER
SET search_path = public
AS $$
    WITH semantic AS (
        SELECT
            dc.source_id AS id,
            dc.id AS chunk_id,
            dc.source_table,
            dc.title,
            dc.content,
            dc.document_type AS doc_type,
            dc.jurisdiction,
            dc.status,
            NULL::int AS year,
            dc.metadata,
            dc.section_ref AS section,
            ROW_NUMBER() OVER (ORDER BY dc.embedding <=> query_embedding) AS sem_rank
        FROM document_chunks dc
        WHERE query_embedding IS NOT NULL
          AND dc.embedding IS NOT NULL
          AND (p_jurisdiction IS NULL OR dc.jurisdiction = p_jurisdiction)
          AND (p_status IS NULL OR dc.status = p_status)
          AND (p_review_status IS NULL OR dc.review_status = p_review_status)
          AND (dc.tenant_id IS NULL OR (p_tenant_id IS NOT NULL AND dc.tenant_id = p_tenant_id))
        ORDER BY dc.embedding <=> query_embedding
        LIMIT 80
    ),
    keyword AS (
        SELECT
            dc.source_id AS id,
            dc.id AS chunk_id,
            dc.source_table,
            dc.title,
            dc.content,
            dc.document_type AS doc_type,
            dc.jurisdiction,
            dc.status,
            NULL::int AS year,
            dc.metadata,
            dc.section_ref AS section,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank(
                    to_tsvector('simple', coalesce(dc.title, '') || ' ' || coalesce(dc.section_ref, '') || ' ' || coalesce(dc.content, '')),
                    plainto_tsquery('simple', query_text)
                ) DESC
            ) AS kw_rank
        FROM document_chunks dc
        WHERE to_tsvector('simple', coalesce(dc.title, '') || ' ' || coalesce(dc.section_ref, '') || ' ' || coalesce(dc.content, ''))
              @@ plainto_tsquery('simple', query_text)
            AND (p_jurisdiction IS NULL OR dc.jurisdiction = p_jurisdiction)
            AND (p_status IS NULL OR dc.status = p_status)
            AND (p_review_status IS NULL OR dc.review_status = p_review_status)
            AND (dc.tenant_id IS NULL OR (p_tenant_id IS NOT NULL AND dc.tenant_id = p_tenant_id))
        LIMIT 80
    ),
    all_results AS (
        SELECT id, chunk_id, source_table, title, content, doc_type, jurisdiction, status, year, metadata, section, sem_rank::bigint, NULL::bigint AS kw_rank
        FROM semantic
        UNION ALL
        SELECT id, chunk_id, source_table, title, content, doc_type, jurisdiction, status, year, metadata, section, NULL::bigint, kw_rank::bigint
        FROM keyword
    ),
    fused AS (
        SELECT
            id,
            chunk_id,
            source_table,
            title,
            content,
            doc_type,
            jurisdiction,
            status,
            year,
            metadata || jsonb_build_object('chunk_id', chunk_id, 'section', section) AS metadata,
            section,
            MIN(sem_rank) AS semantic_rank,
            MIN(kw_rank) AS keyword_rank,
            COALESCE(SUM(1.0 / (rrf_k + sem_rank)), 0)
              + COALESCE(SUM(1.0 / (rrf_k + kw_rank)), 0) AS final_score
        FROM all_results
        GROUP BY id, chunk_id, source_table, title, content, doc_type, jurisdiction, status, year, metadata, section
    )
    SELECT *
    FROM fused
    ORDER BY final_score DESC
    LIMIT match_count;
$$;
