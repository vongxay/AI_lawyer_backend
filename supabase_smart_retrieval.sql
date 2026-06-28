-- ============================================================================
-- supabase_smart_retrieval.sql
-- ----------------------------------------------------------------------------
-- Makes RAG retrieval SEMANTIC-FIRST and INDEX-BACKED so it stops timing out
-- and degrading to dumb keyword matching.
--
-- ROOT CAUSE this migration fixes:
--   The previous hybrid_document_chunk_search combined four CTEs in one query.
--   Two of them could NOT use their indexes and forced a full sequential scan
--   over ~16k chunks, recomputing expensive functions on the full `content`
--   for every row, which blew past Supabase's 10s statement_timeout:
--     1. keyword leg:  COALESCE(dc.search_tsv, lao_legal_to_tsvector(content))
--                      -> the COALESCE wrapper prevents using the GIN index on
--                         search_tsv.
--     2. trigram leg:  similarity(fn(content), q) >= 0.08
--                      -> similarity()>=threshold does NOT use a trgm GIN index
--                         (only the % operator does).
--
-- FIX:
--   * Backfill + NOT NULL guarantee on search_tsv so the GIN index is always
--     usable (no COALESCE needed).
--   * semantic leg uses the HNSW vector index (fast, meaning-based).
--   * keyword leg uses `dc.search_tsv @@ query` directly (GIN index, fast).
--   * Provide dedicated single-leg RPCs so the application can run a clean
--     semantic-first plan and fuse results itself.
--
-- Safe to re-run (idempotent). Run AFTER:
--   supabase_agentic_rag_chunks.sql, supabase_lao_law_categories.sql,
--   supabase_lao_fts.sql
-- ============================================================================

-- ── 0. Guarantee search_tsv is populated for every row ──────────────────────
UPDATE public.document_chunks dc
SET search_tsv = public.lao_legal_to_tsvector(
    coalesce(dc.title, '') || ' ' || coalesce(dc.section_ref, '') || ' ' || coalesce(dc.content, '')
)
WHERE dc.search_tsv IS NULL
   OR dc.search_tsv = ''::tsvector;

-- GIN index on the maintained tsvector (used by the keyword leg).
CREATE INDEX IF NOT EXISTS document_chunks_search_tsv_idx
    ON public.document_chunks USING gin(search_tsv);

-- Composite b-tree to make the metadata filters cheap.
CREATE INDEX IF NOT EXISTS document_chunks_filter_idx
    ON public.document_chunks (jurisdiction, status, review_status, tenant_id);

-- Make sure the HNSW vector index exists (semantic leg).
CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw
    ON public.document_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ── 1. Pure semantic search (HNSW only — fast, meaning-based) ───────────────
CREATE OR REPLACE FUNCTION public.semantic_document_chunk_search(
    query_embedding     vector(1536),
    p_jurisdiction      text DEFAULT NULL,
    p_status            text DEFAULT 'active',
    p_review_status     text DEFAULT 'approved',
    p_tenant_id         uuid DEFAULT NULL,
    match_count         int  DEFAULT 10,
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
            'language', dc.language,
            'chunk_id', dc.id,
            'section', dc.section_ref
        ) AS metadata,
        dc.section_ref AS section,
        ROW_NUMBER() OVER (ORDER BY dc.embedding <=> query_embedding)::bigint AS semantic_rank,
        NULL::bigint AS keyword_rank,
        (1.0 - (dc.embedding <=> query_embedding))::double precision AS final_score
    FROM public.document_chunks dc
    WHERE query_embedding IS NOT NULL
      AND dc.embedding IS NOT NULL
      AND (p_jurisdiction IS NULL OR dc.jurisdiction = p_jurisdiction)
      AND (p_law_category IS NULL OR dc.law_category = p_law_category)
      AND (p_status IS NULL OR dc.status = p_status)
      AND (p_review_status IS NULL OR dc.review_status = p_review_status)
      AND (dc.tenant_id IS NULL OR (p_tenant_id IS NOT NULL AND dc.tenant_id = p_tenant_id))
    ORDER BY dc.embedding <=> query_embedding
    LIMIT match_count;
$$;

-- ── 2. Pure lexical search (GIN tsvector — fast, exact-term) ────────────────
CREATE OR REPLACE FUNCTION public.lexical_document_chunk_search(
    query_text          text,
    p_jurisdiction      text DEFAULT NULL,
    p_status            text DEFAULT 'active',
    p_review_status     text DEFAULT 'approved',
    p_tenant_id         uuid DEFAULT NULL,
    match_count         int  DEFAULT 10,
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
    WITH q AS (
        SELECT public.lao_legal_to_tsquery(coalesce(query_text, '')) AS lao_query
    )
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
            'language', dc.language,
            'chunk_id', dc.id,
            'section', dc.section_ref
        ) AS metadata,
        dc.section_ref AS section,
        NULL::bigint AS semantic_rank,
        ROW_NUMBER() OVER (
            ORDER BY ts_rank_cd(dc.search_tsv, q.lao_query, 32) DESC
        )::bigint AS keyword_rank,
        ts_rank_cd(dc.search_tsv, q.lao_query, 32)::double precision AS final_score
    FROM public.document_chunks dc
    CROSS JOIN q
    WHERE dc.search_tsv @@ q.lao_query
      AND (p_jurisdiction IS NULL OR dc.jurisdiction = p_jurisdiction)
      AND (p_law_category IS NULL OR dc.law_category = p_law_category)
      AND (p_status IS NULL OR dc.status = p_status)
      AND (p_review_status IS NULL OR dc.review_status = p_review_status)
      AND (dc.tenant_id IS NULL OR (p_tenant_id IS NOT NULL AND dc.tenant_id = p_tenant_id))
    ORDER BY final_score DESC
    LIMIT match_count;
$$;

-- ── 3. Hybrid search — semantic + lexical RRF, both index-backed ────────────
-- Same signature as before so the application keeps working, but now FAST.
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
    q AS (
        SELECT public.lao_legal_to_tsquery(coalesce(query_text, '')) AS lao_query
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
            dc.metadata || jsonb_build_object(
                'law_category', dc.law_category,
                'law_no', dc.law_no,
                'article', dc.article,
                'language', dc.language
            ) AS metadata,
            dc.section_ref AS section,
            ROW_NUMBER() OVER (ORDER BY dc.embedding <=> query_embedding) AS sem_rank
        FROM public.document_chunks dc
        WHERE query_embedding IS NOT NULL
          AND dc.embedding IS NOT NULL
          AND (p_jurisdiction IS NULL OR dc.jurisdiction = p_jurisdiction)
          AND (p_law_category IS NULL OR dc.law_category = p_law_category)
          AND (p_status IS NULL OR dc.status = p_status)
          AND (p_review_status IS NULL OR dc.review_status = p_review_status)
          AND (dc.tenant_id IS NULL OR (p_tenant_id IS NOT NULL AND dc.tenant_id = p_tenant_id))
        ORDER BY dc.embedding <=> query_embedding
        LIMIT 50
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
            dc.metadata || jsonb_build_object(
                'law_category', dc.law_category,
                'law_no', dc.law_no,
                'article', dc.article,
                'language', dc.language
            ) AS metadata,
            dc.section_ref AS section,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank_cd(dc.search_tsv, q.lao_query, 32) DESC
            ) AS kw_rank
        FROM public.document_chunks dc
        CROSS JOIN q
        WHERE dc.search_tsv @@ q.lao_query
          AND (p_jurisdiction IS NULL OR dc.jurisdiction = p_jurisdiction)
          AND (p_law_category IS NULL OR dc.law_category = p_law_category)
          AND (p_status IS NULL OR dc.status = p_status)
          AND (p_review_status IS NULL OR dc.review_status = p_review_status)
          AND (dc.tenant_id IS NULL OR (p_tenant_id IS NOT NULL AND dc.tenant_id = p_tenant_id))
        LIMIT 50
    ),
    all_results AS (
        SELECT id, chunk_id, source_table, title, content, doc_type, jurisdiction, status,
               metadata, section, sem_rank::bigint AS sem_rank, NULL::bigint AS kw_rank
        FROM semantic
        UNION ALL
        SELECT id, chunk_id, source_table, title, content, doc_type, jurisdiction, status,
               metadata, section, NULL::bigint, kw_rank::bigint
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
            NULL::int AS year,
            metadata || jsonb_build_object('chunk_id', chunk_id, 'section', section) AS metadata,
            section,
            MIN(sem_rank) AS semantic_rank,
            MIN(kw_rank) AS keyword_rank,
            COALESCE(SUM(1.0 / (rrf_k + sem_rank)), 0)
              + COALESCE(SUM(1.0 / (rrf_k + kw_rank)), 0) AS final_score
        FROM all_results
        GROUP BY id, chunk_id, source_table, title, content, doc_type, jurisdiction, status, section,
                 metadata || jsonb_build_object('chunk_id', chunk_id, 'section', section)
    )
    SELECT
        id, chunk_id, source_table, title, content, doc_type, jurisdiction, status, year,
        metadata, section, semantic_rank, keyword_rank, final_score
    FROM fused
    ORDER BY final_score DESC
    LIMIT match_count;
$$;

-- ── 4. Optional: raise per-statement timeout for the API role ───────────────
-- Vector + GIN searches are fast, but ingest of very large laws can be slow.
-- Uncomment and adjust if you still see 57014 timeouts during heavy ingest.
-- ALTER ROLE authenticator SET statement_timeout = '20s';
-- NOTIFY pgrst, 'reload config';
