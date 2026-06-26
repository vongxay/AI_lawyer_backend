-- Fix PostgREST ambiguity: drop the older 8-parameter overload of hybrid_document_chunk_search.
-- Keep the version with p_law_category from supabase_lao_law_categories.sql.
--
-- Apply in Supabase SQL editor if retrieval RPC calls fail with PGRST203.

DROP FUNCTION IF EXISTS public.hybrid_document_chunk_search(
    text,
    vector,
    text,
    text,
    text,
    uuid,
    int,
    int
);
