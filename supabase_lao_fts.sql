-- Lao-aware full-text search for legal document chunks.
-- Apply in Supabase SQL editor AFTER:
--   1. ai_lawyer_database.sql
--   2. supabase_agentic_rag_chunks.sql
--   3. supabase_lao_legal_metadata.sql
--   4. supabase_lao_law_categories.sql
--   5. supabase_fix_hybrid_search_overload.sql (optional, if PGRST203 occurred)
--
-- Why: PostgreSQL 'simple' FTS does not tokenize Lao script well (no word boundaries,
-- tone marks, OCR variants). This migration adds Lao normalization, token boundaries,
-- a maintained search_tsv column, trigram similarity, and updates hybrid search.

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Ensure optional metadata columns exist (added by supabase_lao_law_categories.sql).
ALTER TABLE IF EXISTS public.document_chunks
    ADD COLUMN IF NOT EXISTS law_category text,
    ADD COLUMN IF NOT EXISTS language text DEFAULT 'lo',
    ADD COLUMN IF NOT EXISTS law_no text,
    ADD COLUMN IF NOT EXISTS article text;

-- ── Lao script detection ──────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION public.contains_lao_script(input text)
RETURNS boolean
LANGUAGE sql
IMMUTABLE
STRICT
AS $$
    SELECT coalesce(input, '') ~ E'[\u0E80-\u0EFF]';
$$;

-- ── Normalize Lao legal text for search (mirrors Python normalise_search_text) ─
CREATE OR REPLACE FUNCTION public.normalize_lao_legal_fts_text(input text)
RETURNS text
LANGUAGE plpgsql
IMMUTABLE
STRICT
AS $$
DECLARE
    t text := coalesce(input, '');
BEGIN
    -- NFC-ish cleanup: strip BOM / zero-width / control chars
    t := regexp_replace(t, E'[\\u200B-\\u200F\\uFEFF]', '', 'g');
    t := regexp_replace(t, E'[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F\\x7F]', ' ', 'g');
    t := lower(t);

    -- Remove common Lao tone marks for OCR-tolerant matching
    t := translate(
        t,
        E'\u0EC8\u0EC9\u0ECA\u0ECB\u0ECC\u0ECD',
        ''
    );

    -- High-value OCR normalisations used in retrieval
    t := replace(t, E'\u0EB3', E'\u0EB2');
    t := replace(t, E'\u0E97\u0EB5\u0EA5\u0EB4\u0E99', E'\u0E97\u0EB5\u0E94\u0EB4\u0E99');
    t := replace(t, E'\u0E97\u0EB5\u0E9C\u0EB4\u0E99', E'\u0E97\u0EB5\u0E94\u0EB4\u0E99');
    t := replace(t, E'\u0EAA\u0EB4\u0E94\u0E99\u0ECD\u0EB2\u0EC3\u0E8A', E'\u0EAA\u0EB4\u0E94\u0E99\u0EB2\u0EC3\u0E8A');
    t := replace(t, E'\u0EAA\u0EB4\u0E94\u0E99\u0EB2\u0ECD\u0EC3\u0E8A', E'\u0EAA\u0EB4\u0E94\u0E99\u0EB2\u0EC3\u0E8A');
    t := replace(t, E'\u0E9C\u0EBB\u0E99\u0E9B\u0EB0\u0EC2\u0E97\u0E8D\u0E94', E'\u0E9C\u0EBB\u0E99\u0E9B\u0EB0\u0EC2\u0EAB\u0E8D\u0E94');
    t := replace(t, E'\u0EC3\u0E82', E'\u0EC3\u0E8A');

    t := regexp_replace(t, E'\\s+', ' ', 'g');
    RETURN trim(t);
END;
$$;

-- Insert token boundaries so 'simple' FTS can match Lao words and legal markers.
CREATE OR REPLACE FUNCTION public.lao_legal_add_token_boundaries(input text)
RETURNS text
LANGUAGE sql
IMMUTABLE
STRICT
AS $$
    SELECT trim(regexp_replace(
        regexp_replace(
            regexp_replace(
                regexp_replace(
                    public.normalize_lao_legal_fts_text(input),
                    E'([\\u0E80-\\u0EFF])([0-9])',
                    E'\\1 \\2',
                    'g'
                ),
                E'([0-9])([\\u0E80-\\u0EFF])',
                E'\\1 \\2',
                'g'
            ),
            E'[\\s\\u0EAF\\u0EBB\\u0EBC,.;:()\\[\\]{}''\"]+',
            ' ',
            'g'
        ),
        E'\\s+',
        ' ',
        'g'
    ));
$$;

CREATE OR REPLACE FUNCTION public.lao_legal_document_text(
    title text,
    section_ref text,
    content text
)
RETURNS text
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT public.lao_legal_add_token_boundaries(
        coalesce(title, '') || ' ' || coalesce(section_ref, '') || ' ' || coalesce(content, '')
    );
$$;

CREATE OR REPLACE FUNCTION public.lao_legal_to_tsvector(input text)
RETURNS tsvector
LANGUAGE sql
IMMUTABLE
STRICT
AS $$
    SELECT to_tsvector('simple', public.lao_legal_add_token_boundaries(input));
$$;

-- Build a tsquery from Lao/English mixed queries. Uses AND between tokens for precision.
CREATE OR REPLACE FUNCTION public.lao_legal_to_tsquery(input text)
RETURNS tsquery
LANGUAGE plpgsql
IMMUTABLE
STRICT
AS $$
DECLARE
    tokenized text := public.lao_legal_add_token_boundaries(input);
    tokens text[];
    t text;
    parts text[] := ARRAY[]::text[];
    min_len int := 2;
BEGIN
    IF tokenized IS NULL OR tokenized = '' THEN
        RETURN plainto_tsquery('simple', '');
    END IF;

    tokens := regexp_split_to_array(tokenized, E'\\s+');

    FOREACH t IN ARRAY tokens LOOP
        t := trim(t);
        IF t = '' THEN
            CONTINUE;
        END IF;
        -- Skip very short ASCII noise; keep Lao chars even if short (e.g. digits in articles)
        IF t ~ '^[a-z0-9]+$' AND length(t) < min_len THEN
            CONTINUE;
        END IF;
        -- Escape tsquery special chars
        t := regexp_replace(t, '([&|!():*''\"])', E'\\\\\\1', 'g');
        parts := array_append(parts, t || ':*');
    END LOOP;

    IF array_length(parts, 1) IS NULL THEN
        RETURN plainto_tsquery('simple', tokenized);
    END IF;

    RETURN to_tsquery('simple', array_to_string(parts, ' & '));
EXCEPTION
    WHEN others THEN
        RETURN plainto_tsquery('simple', tokenized);
END;
$$;

CREATE OR REPLACE FUNCTION public.lao_legal_search_document_text(
    title text,
    section_ref text,
    content text
)
RETURNS text
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT public.normalize_lao_legal_fts_text(
        coalesce(title, '') || ' ' || coalesce(section_ref, '') || ' ' || coalesce(content, '')
    );
$$;

-- ── Maintained tsvector column on document_chunks ───────────────────────────
ALTER TABLE public.document_chunks
    ADD COLUMN IF NOT EXISTS search_tsv tsvector;

CREATE OR REPLACE FUNCTION public.document_chunks_search_tsv_trigger()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.search_tsv := public.lao_legal_to_tsvector(
        coalesce(NEW.title, '') || ' ' || coalesce(NEW.section_ref, '') || ' ' || coalesce(NEW.content, '')
    );
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS document_chunks_search_tsv_update ON public.document_chunks;
CREATE TRIGGER document_chunks_search_tsv_update
    BEFORE INSERT OR UPDATE OF title, section_ref, content
    ON public.document_chunks
    FOR EACH ROW
    EXECUTE FUNCTION public.document_chunks_search_tsv_trigger();

-- Backfill existing rows (safe to re-run)
UPDATE public.document_chunks dc
SET search_tsv = public.lao_legal_to_tsvector(
    coalesce(dc.title, '') || ' ' || coalesce(dc.section_ref, '') || ' ' || coalesce(dc.content, '')
)
WHERE dc.search_tsv IS NULL
   OR dc.search_tsv = ''::tsvector;

CREATE INDEX IF NOT EXISTS document_chunks_search_tsv_idx
    ON public.document_chunks USING gin(search_tsv);

CREATE INDEX IF NOT EXISTS document_chunks_search_text_trgm_idx
    ON public.document_chunks USING gin(
        (public.lao_legal_search_document_text(title, section_ref, content)) gin_trgm_ops
    );

-- Drop ambiguous overloads before replacing hybrid search
DROP FUNCTION IF EXISTS public.hybrid_document_chunk_search(
    text, vector, text, text, text, uuid, int, int
);

-- ── Hybrid search with Lao FTS + trigram keyword legs ─────────────────────────
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
    WITH
    normalized_query AS (
        SELECT
            coalesce(query_text, '') AS raw_query,
            public.normalize_lao_legal_fts_text(coalesce(query_text, '')) AS norm_query,
            public.lao_legal_to_tsquery(coalesce(query_text, '')) AS lao_query,
            (
                public.contains_lao_script(coalesce(query_text, ''))
                OR coalesce(p_jurisdiction, '') IN ('laos', 'lao', 'lao-pdr')
            ) AS use_lao_search
    ),
    semantic AS (
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
                ORDER BY ts_rank_cd(
                    coalesce(dc.search_tsv, public.lao_legal_to_tsvector(
                        coalesce(dc.title, '') || ' ' || coalesce(dc.section_ref, '') || ' ' || coalesce(dc.content, '')
                    )),
                    nq.lao_query,
                    32
                ) DESC
            ) AS kw_rank
        FROM document_chunks dc
        CROSS JOIN normalized_query nq
        WHERE nq.use_lao_search
          AND coalesce(dc.search_tsv, public.lao_legal_to_tsvector(
                coalesce(dc.title, '') || ' ' || coalesce(dc.section_ref, '') || ' ' || coalesce(dc.content, '')
              )) @@ nq.lao_query
          AND (p_jurisdiction IS NULL OR dc.jurisdiction = p_jurisdiction)
          AND (p_law_category IS NULL OR dc.law_category = p_law_category)
          AND (p_status IS NULL OR dc.status = p_status)
          AND (p_review_status IS NULL OR dc.review_status = p_review_status)
          AND (dc.tenant_id IS NULL OR (p_tenant_id IS NOT NULL AND dc.tenant_id = p_tenant_id))
        LIMIT 80
    ),
    keyword_simple AS (
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
        CROSS JOIN normalized_query nq
        WHERE NOT nq.use_lao_search
          AND to_tsvector('simple', coalesce(dc.title, '') || ' ' || coalesce(dc.section_ref, '') || ' ' || coalesce(dc.content, ''))
              @@ plainto_tsquery('simple', query_text)
          AND (p_jurisdiction IS NULL OR dc.jurisdiction = p_jurisdiction)
          AND (p_law_category IS NULL OR dc.law_category = p_law_category)
          AND (p_status IS NULL OR dc.status = p_status)
          AND (p_review_status IS NULL OR dc.review_status = p_review_status)
          AND (dc.tenant_id IS NULL OR (p_tenant_id IS NOT NULL AND dc.tenant_id = p_tenant_id))
        LIMIT 80
    ),
    lao_trigram AS (
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
                ORDER BY similarity(
                    public.lao_legal_search_document_text(dc.title, dc.section_ref, dc.content),
                    nq.norm_query
                ) DESC
            ) AS trgm_rank
        FROM document_chunks dc
        CROSS JOIN normalized_query nq
        WHERE nq.use_lao_search
          AND length(nq.norm_query) >= 3
          AND similarity(
                public.lao_legal_search_document_text(dc.title, dc.section_ref, dc.content),
                nq.norm_query
              ) >= 0.08
          AND (p_jurisdiction IS NULL OR dc.jurisdiction = p_jurisdiction)
          AND (p_law_category IS NULL OR dc.law_category = p_law_category)
          AND (p_status IS NULL OR dc.status = p_status)
          AND (p_review_status IS NULL OR dc.review_status = p_review_status)
          AND (dc.tenant_id IS NULL OR (p_tenant_id IS NOT NULL AND dc.tenant_id = p_tenant_id))
        LIMIT 80
    ),
    all_results AS (
        SELECT id, chunk_id, source_table, title, content, doc_type, jurisdiction, status, year, metadata, section,
               sem_rank::bigint AS sem_rank, NULL::bigint AS kw_rank, NULL::bigint AS trgm_rank
        FROM semantic
        UNION ALL
        SELECT id, chunk_id, source_table, title, content, doc_type, jurisdiction, status, year, metadata, section,
               NULL::bigint, kw_rank::bigint, NULL::bigint
        FROM keyword
        UNION ALL
        SELECT id, chunk_id, source_table, title, content, doc_type, jurisdiction, status, year, metadata, section,
               NULL::bigint, kw_rank::bigint, NULL::bigint
        FROM keyword_simple
        UNION ALL
        SELECT id, chunk_id, source_table, title, content, doc_type, jurisdiction, status, year, metadata, section,
               NULL::bigint, NULL::bigint, trgm_rank::bigint
        FROM lao_trigram
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
              + COALESCE(SUM(1.0 / (rrf_k + kw_rank)), 0)
              + COALESCE(SUM(1.0 / (rrf_k + trgm_rank)), 0) AS final_score
        FROM all_results
        GROUP BY id, chunk_id, source_table, title, content, doc_type, jurisdiction, status, year, metadata, section
    )
    SELECT *
    FROM fused
    ORDER BY final_score DESC
    LIMIT match_count;
$$;
