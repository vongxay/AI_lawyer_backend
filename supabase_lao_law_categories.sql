-- Lao law category taxonomy for admin ingestion and Agentic RAG.
-- Apply after ai_lawyer_database.sql, supabase_lao_legal_metadata.sql,
-- and supabase_agentic_rag_chunks.sql.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS lao_law_categories (
    id              text PRIMARY KEY,
    lao_name        text NOT NULL,
    english_name    text NOT NULL,
    description     text,
    sort_order      int NOT NULL UNIQUE,
    is_active       boolean NOT NULL DEFAULT true,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    CHECK (id IN (
        'constitution_justice',
        'state_security',
        'economy',
        'social_culture',
        'foreign_affairs'
    ))
);

INSERT INTO lao_law_categories (id, lao_name, english_name, description, sort_order)
VALUES
    ('constitution_justice', 'ລັດຖະທຳມະນູນ ແລະ ຍຸຕິທຳ', 'Constitution & Justice', 'Constitutional, court, prosecution, civil and criminal justice laws.', 1),
    ('state_security', 'ປົກຄອງ ແລະ ປ້ອງກັນຄວາມສະຫງົບ', 'Administration & Security', 'State administration, public order, defense, police and internal security.', 2),
    ('economy', 'ເສດຖະກິດ', 'Economy', 'Business, investment, land, tax, banking, labor and commercial regulations.', 3),
    ('social_culture', 'ວັດທະນະທຳ-ສັງຄົມ', 'Social & Culture', 'Education, health, environment, culture, family and social protection.', 4),
    ('foreign_affairs', 'ການຕ່າງປະເທດ', 'Foreign Affairs', 'Treaties, international cooperation, borders and diplomatic affairs.', 5)
ON CONFLICT (id) DO UPDATE SET
    lao_name = EXCLUDED.lao_name,
    english_name = EXCLUDED.english_name,
    description = EXCLUDED.description,
    sort_order = EXCLUDED.sort_order,
    is_active = true,
    updated_at = now();

ALTER TABLE laws
    ADD COLUMN IF NOT EXISTS law_category text,
    ADD COLUMN IF NOT EXISTS language text DEFAULT 'lo';

ALTER TABLE cases
    ADD COLUMN IF NOT EXISTS law_category text,
    ADD COLUMN IF NOT EXISTS language text DEFAULT 'lo';

ALTER TABLE legal_forms
    ADD COLUMN IF NOT EXISTS law_category text,
    ADD COLUMN IF NOT EXISTS metadata jsonb NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE IF EXISTS document_chunks
    ADD COLUMN IF NOT EXISTS law_category text,
    ADD COLUMN IF NOT EXISTS language text DEFAULT 'lo',
    ADD COLUMN IF NOT EXISTS law_no text,
    ADD COLUMN IF NOT EXISTS article text,
    ADD COLUMN IF NOT EXISTS chapter_ref text;

UPDATE laws
SET law_category = metadata->>'law_category'
WHERE law_category IS NULL
  AND metadata ? 'law_category'
  AND metadata->>'law_category' IN (SELECT id FROM lao_law_categories);

UPDATE cases
SET law_category = metadata->>'law_category'
WHERE law_category IS NULL
  AND metadata ? 'law_category'
  AND metadata->>'law_category' IN (SELECT id FROM lao_law_categories);

UPDATE legal_forms
SET law_category = metadata->>'law_category'
WHERE law_category IS NULL
  AND metadata ? 'law_category'
  AND metadata->>'law_category' IN (SELECT id FROM lao_law_categories);

DO $$
BEGIN
    IF to_regclass('public.document_chunks') IS NOT NULL THEN
        UPDATE document_chunks
        SET
            law_category = metadata->>'law_category',
            language = coalesce(metadata->>'language', language),
            law_no = coalesce(metadata->>'law_no', law_no),
            article = coalesce(metadata->>'article', article)
        WHERE metadata ? 'law_category'
          AND metadata->>'law_category' IN (SELECT id FROM lao_law_categories);
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'laws_law_category_fk'
    ) THEN
        ALTER TABLE laws
            ADD CONSTRAINT laws_law_category_fk
            FOREIGN KEY (law_category) REFERENCES lao_law_categories(id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'cases_law_category_fk'
    ) THEN
        ALTER TABLE cases
            ADD CONSTRAINT cases_law_category_fk
            FOREIGN KEY (law_category) REFERENCES lao_law_categories(id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'legal_forms_law_category_fk'
    ) THEN
        ALTER TABLE legal_forms
            ADD CONSTRAINT legal_forms_law_category_fk
            FOREIGN KEY (law_category) REFERENCES lao_law_categories(id);
    END IF;

    IF to_regclass('public.document_chunks') IS NOT NULL
       AND NOT EXISTS (
           SELECT 1 FROM pg_constraint WHERE conname = 'document_chunks_law_category_fk'
       ) THEN
        ALTER TABLE document_chunks
            ADD CONSTRAINT document_chunks_law_category_fk
            FOREIGN KEY (law_category) REFERENCES lao_law_categories(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS lao_law_categories_active_idx
    ON lao_law_categories(is_active, sort_order);

CREATE INDEX IF NOT EXISTS laws_law_category_idx
    ON laws(jurisdiction, law_category, status, review_status, year_be DESC);

CREATE INDEX IF NOT EXISTS cases_law_category_idx
    ON cases(jurisdiction, law_category, status, review_status, year_be DESC);

CREATE INDEX IF NOT EXISTS legal_forms_law_category_idx
    ON legal_forms(jurisdiction, law_category, review_status, is_active);

CREATE INDEX IF NOT EXISTS document_chunks_law_category_idx
    ON document_chunks(jurisdiction, law_category, status, review_status);

CREATE INDEX IF NOT EXISTS document_chunks_law_article_idx
    ON document_chunks(law_category, law_no, article);

DROP FUNCTION IF EXISTS public.hybrid_document_chunk_search(
    text, vector, text, text, text, uuid, int, int
);

CREATE OR REPLACE FUNCTION public.hybrid_document_chunk_search(
    query_text          text,
    query_embedding     vector(1536) DEFAULT NULL,
    p_jurisdiction      text DEFAULT NULL,
    p_status            text DEFAULT 'active',
    p_review_status     text DEFAULT 'approved',
    p_tenant_id         uuid DEFAULT NULL,
    match_count         int DEFAULT 10,
    rrf_k               int DEFAULT 60,
    p_law_category      text DEFAULT NULL
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
            dc.metadata || jsonb_build_object(
                'law_category', dc.law_category,
                'law_no', dc.law_no,
                'article', dc.article,
                'language', dc.language
            ) AS metadata,
            dc.section_ref AS section,
            ROW_NUMBER() OVER (ORDER BY dc.embedding <=> query_embedding) AS sem_rank
        FROM document_chunks dc
        WHERE query_embedding IS NOT NULL
          AND dc.embedding IS NOT NULL
          AND (p_jurisdiction IS NULL OR dc.jurisdiction = p_jurisdiction)
          AND (p_law_category IS NULL OR dc.law_category = p_law_category)
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
            dc.metadata || jsonb_build_object(
                'law_category', dc.law_category,
                'law_no', dc.law_no,
                'article', dc.article,
                'language', dc.language
            ) AS metadata,
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
            AND (p_law_category IS NULL OR dc.law_category = p_law_category)
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
