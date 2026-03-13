-- ============================================================
-- AI LAWYER SYSTEM — Complete Database Schema
-- Production-Ready | Supabase + PostgreSQL + pgvector
-- Version 2.0 | Copy & paste into Supabase SQL Editor
-- ============================================================
-- EXECUTION ORDER: Run this entire file at once
-- All sections are ordered to avoid dependency errors
-- ============================================================


-- ============================================================
-- SECTION 0: EXTENSIONS
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";     -- UUID generation
CREATE EXTENSION IF NOT EXISTS "vector";         -- pgvector for embeddings
CREATE EXTENSION IF NOT EXISTS "pg_trgm";        -- Trigram for fuzzy text search
CREATE EXTENSION IF NOT EXISTS "unaccent";       -- Remove accents for Thai/Lao search
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements"; -- Query performance monitoring


-- ============================================================
-- SECTION 1: ENUMS (Type Definitions)
-- ============================================================

-- User roles
CREATE TYPE user_role AS ENUM (
    'super_admin',
    'admin',
    'lawyer',
    'client',
    'auditor',
    'expert_reviewer'
);

-- Subscription plans
CREATE TYPE subscription_plan AS ENUM (
    'free',
    'starter',
    'professional',
    'enterprise'
);

-- Legal jurisdictions supported
CREATE TYPE jurisdiction AS ENUM (
    'thailand',
    'laos',
    'cambodia',
    'myanmar',
    'vietnam',
    'international',
    'asean'
);

-- Law/Case document types
CREATE TYPE legal_doc_type AS ENUM (
    'statute',              -- กฎหมาย
    'case_law',             -- คำพิพากษา
    'regulation',           -- กฎกระทรวง
    'royal_decree',         -- พระราชกฤษฎีกา
    'ministerial_order',    -- คำสั่งกระทรวง
    'legal_form',           -- แบบฟอร์มกฎหมาย
    'guideline'             -- แนวทางปฏิบัติ
);

-- Law/case status
CREATE TYPE law_status AS ENUM (
    'active',       -- ยังมีผลบังคับ
    'amended',      -- แก้ไขแล้ว (ใช้ฉบับใหม่)
    'repealed',     -- ยกเลิกแล้ว
    'pending'       -- รอประกาศใช้
);

-- Case outcome (for precedent analysis)
CREATE TYPE case_outcome AS ENUM (
    'plaintiff_won',     -- โจทก์ชนะ
    'defendant_won',     -- จำเลยชนะ
    'settled',           -- เจรจาตกลง
    'dismissed',         -- ยกฟ้อง
    'partial',           -- ชนะบางส่วน
    'remanded',          -- ส่งกลับไต่สวนใหม่
    'unknown'
);

-- Citation relationship types (for case law graph)
CREATE TYPE citation_relationship AS ENUM (
    'cites',            -- อ้างอิงเป็น precedent
    'overruled_by',     -- ถูก overrule
    'distinguished',    -- แยกแยะว่าต่างกัน
    'followed',         -- นำหลักมาใช้ต่อ
    'applied',          -- นำไปปรับใช้
    'criticized',       -- วิจารณ์คำวินิจฉัย
    'explained'         -- อธิบายขยายความ
);

-- Citation verification status
CREATE TYPE citation_status AS ENUM (
    'verified',
    'outdated',
    'unverified',
    'rejected',
    'pending'
);

-- Case session status
CREATE TYPE session_status AS ENUM (
    'active',
    'closed',
    'escalated',
    'archived'
);

-- Legal case status
CREATE TYPE legal_case_status AS ENUM (
    'open',
    'in_progress',
    'closed',
    'settled',
    'won',
    'lost',
    'appealed'
);

-- Risk level
CREATE TYPE risk_level AS ENUM (
    'low',
    'medium',
    'high',
    'critical'
);

-- Evidence types
CREATE TYPE evidence_type AS ENUM (
    'document_pdf',
    'document_word',
    'image',
    'screenshot',
    'audio',
    'video',
    'email',
    'chat_log',
    'contract',
    'scanned_doc'
);

-- Expert review status
CREATE TYPE review_status AS ENUM (
    'pending',
    'in_review',
    'resolved',
    'escalated_further'
);

-- Agent names used in orchestration
CREATE TYPE agent_name AS ENUM (
    'legal_research',
    'irac_reasoning',
    'citation_verification',
    'document_analysis',
    'evidence_analyzer',
    'risk_strategy',
    'query_classifier'
);


-- ============================================================
-- SECTION 2: CORE AUTH & TENANT TABLES
-- ============================================================

-- Tenants (law firms, organizations)
CREATE TABLE tenants (
    id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            text NOT NULL,
    slug            text UNIQUE NOT NULL,
    plan            subscription_plan NOT NULL DEFAULT 'starter',
    is_active       boolean NOT NULL DEFAULT true,
    settings        jsonb NOT NULL DEFAULT '{}',
    -- Settings keys: max_users, max_cases, allowed_jurisdictions, logo_url
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

-- Users
CREATE TABLE users (
    id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    email           text UNIQUE NOT NULL,
    full_name       text NOT NULL,
    role            user_role NOT NULL DEFAULT 'client',
    is_active       boolean NOT NULL DEFAULT true,
    last_login      timestamptz,
    preferences     jsonb NOT NULL DEFAULT '{}',
    -- Preferences keys: language, theme, notification_settings
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

-- Indexes
CREATE INDEX users_tenant_idx   ON users(tenant_id);
CREATE INDEX users_email_idx    ON users(email);
CREATE INDEX users_role_idx     ON users(role);


-- ============================================================
-- SECTION 3: KNOWLEDGE BASE TABLES
-- ============================================================

-- Laws and Statutes (core legal knowledge)
CREATE TABLE laws (
    id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       uuid REFERENCES tenants(id) ON DELETE SET NULL,
    -- NULL tenant_id = shared/global knowledge base

    -- Identity
    title           text NOT NULL,
    short_title     text,               -- ชื่อย่อ
    law_code        text,               -- รหัสกฎหมาย เช่น "ปพพ.", "ปอ."
    doc_type        legal_doc_type NOT NULL DEFAULT 'statute',
    jurisdiction    jurisdiction NOT NULL,

    -- Content
    full_text       text NOT NULL,
    summary         text,               -- AI-generated summary
    section_number  text,               -- มาตราที่ (e.g., "537")
    chapter         text,               -- บท/หมวด
    parent_law_id   uuid REFERENCES laws(id) ON DELETE SET NULL,
    -- parent_law_id: link มาตราย่อยกลับมาตราหลัก

    -- Status & Versioning
    status          law_status NOT NULL DEFAULT 'active',
    effective_date  date,               -- วันที่มีผลบังคับใช้
    repeal_date     date,               -- วันที่ถูกยกเลิก (ถ้ามี)
    amended_by_id   uuid REFERENCES laws(id) ON DELETE SET NULL,
    version         int NOT NULL DEFAULT 1,
    year_be         int,                -- พ.ศ.
    year_ce         int,                -- ค.ศ.

    -- Metadata
    source_url      text,
    tags            text[] DEFAULT '{}',
    metadata        jsonb NOT NULL DEFAULT '{}',

    -- Vector embedding (1536 dims for text-embedding-3-small or multilingual-e5-large)
    embedding       vector(1536),

    -- Audit
    ingested_by     uuid REFERENCES users(id) ON DELETE SET NULL,
    ingested_at     timestamptz NOT NULL DEFAULT now(),
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

-- Laws indexes
CREATE INDEX laws_jurisdiction_idx ON laws(jurisdiction);
CREATE INDEX laws_status_idx       ON laws(status);
CREATE INDEX laws_doc_type_idx     ON laws(doc_type);
CREATE INDEX laws_parent_idx       ON laws(parent_law_id) WHERE parent_law_id IS NOT NULL;
CREATE INDEX laws_tags_idx         ON laws USING gin(tags);
CREATE INDEX laws_metadata_idx     ON laws USING gin(metadata);

-- Full-text search indexes (Thai + English)
CREATE INDEX laws_fts_th ON laws USING gin(
    to_tsvector('simple', coalesce(title,'') || ' ' || coalesce(full_text,''))
);

-- HNSW vector index (fastest recall for legal search)
CREATE INDEX laws_embedding_hnsw ON laws
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);


-- Case Law (court decisions and precedents)
CREATE TABLE cases (
    id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       uuid REFERENCES tenants(id) ON DELETE SET NULL,

    -- Identity
    case_no         text NOT NULL,      -- เลขคดี e.g., "ฎ. 1234/2560"
    case_no_year    int,                -- ปีคดี
    court           text NOT NULL,      -- ชื่อศาล
    court_level     text,               -- ชั้นศาล: แขวง/ชั้นต้น/อุทธรณ์/ฎีกา
    jurisdiction    jurisdiction NOT NULL,

    -- Parties
    plaintiff       text,               -- โจทก์
    defendant       text,               -- จำเลย

    -- Content
    facts           text,               -- ข้อเท็จจริง
    issues          text,               -- ประเด็น
    ruling          text NOT NULL,      -- คำวินิจฉัย
    ratio_decidendi text,               -- หลักกฎหมายสำคัญ (most important for RAG)
    full_text       text,               -- ข้อความทั้งหมด
    summary         text,               -- AI-generated summary

    -- Outcome
    outcome         case_outcome NOT NULL DEFAULT 'unknown',
    outcome_notes   text,

    -- Dates
    decision_date   date,
    year_be         int,
    year_ce         int,

    -- Metadata
    doc_type        legal_doc_type NOT NULL DEFAULT 'case_law',
    status          law_status NOT NULL DEFAULT 'active',
    source_url      text,
    tags            text[] DEFAULT '{}',
    metadata        jsonb NOT NULL DEFAULT '{}',

    -- Vector embedding
    embedding       vector(1536),

    -- Audit
    ingested_by     uuid REFERENCES users(id) ON DELETE SET NULL,
    ingested_at     timestamptz NOT NULL DEFAULT now(),
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),

    UNIQUE (case_no, jurisdiction)
);

-- Cases indexes
CREATE INDEX cases_jurisdiction_idx ON cases(jurisdiction);
CREATE INDEX cases_court_idx        ON cases(court);
CREATE INDEX cases_outcome_idx      ON cases(outcome);
CREATE INDEX cases_year_idx         ON cases(year_be);
CREATE INDEX cases_tags_idx         ON cases USING gin(tags);
CREATE INDEX cases_metadata_idx     ON cases USING gin(metadata);

-- Full-text search
CREATE INDEX cases_fts ON cases USING gin(
    to_tsvector('simple',
        coalesce(case_no,'') || ' ' ||
        coalesce(facts,'')   || ' ' ||
        coalesce(ruling,'')  || ' ' ||
        coalesce(ratio_decidendi,'')
    )
);

-- HNSW vector index
CREATE INDEX cases_embedding_hnsw ON cases
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);


-- Case Law Graph (precedent relationship network)
CREATE TABLE case_citations (
    id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Source (the case that does the citing)
    source_case_id  uuid NOT NULL REFERENCES cases(id) ON DELETE CASCADE,

    -- Target (can be another case or a statute)
    cited_case_id   uuid REFERENCES cases(id) ON DELETE CASCADE,
    cited_law_id    uuid REFERENCES laws(id)  ON DELETE CASCADE,

    relationship    citation_relationship NOT NULL,
    notes           text,               -- บริบทของการอ้างอิง
    year            int,

    created_at      timestamptz NOT NULL DEFAULT now(),

    -- At least one target must be set
    CONSTRAINT citation_has_target CHECK (
        cited_case_id IS NOT NULL OR cited_law_id IS NOT NULL
    )
);

CREATE INDEX case_citations_source_idx ON case_citations(source_case_id);
CREATE INDEX case_citations_case_idx   ON case_citations(cited_case_id)  WHERE cited_case_id IS NOT NULL;
CREATE INDEX case_citations_law_idx    ON case_citations(cited_law_id)   WHERE cited_law_id IS NOT NULL;
CREATE INDEX case_citations_rel_idx    ON case_citations(relationship);


-- Legal Form Templates (สัญญา/แบบฟอร์มมาตรฐาน)
CREATE TABLE legal_forms (
    id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       uuid REFERENCES tenants(id) ON DELETE SET NULL,

    title           text NOT NULL,
    form_type       text NOT NULL,      -- e.g., 'lease', 'employment', 'sale'
    jurisdiction    jurisdiction NOT NULL,
    content         text NOT NULL,      -- Full template text
    variables       jsonb DEFAULT '{}', -- Variable placeholders and descriptions
    version         int NOT NULL DEFAULT 1,
    is_active       boolean NOT NULL DEFAULT true,
    tags            text[] DEFAULT '{}',
    embedding       vector(1536),

    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX legal_forms_type_idx ON legal_forms(form_type);
CREATE INDEX legal_forms_hnsw ON legal_forms
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);


-- ============================================================
-- SECTION 4: CLIENT CASES & SESSION TABLES
-- ============================================================

-- Legal Cases (client cases, not court cases)
CREATE TABLE legal_cases (
    id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    client_id       uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    lawyer_id       uuid REFERENCES users(id) ON DELETE SET NULL,

    -- Case details
    title           text NOT NULL,
    case_type       text NOT NULL,      -- 'contract', 'labor', 'property', 'criminal', etc.
    jurisdiction    jurisdiction NOT NULL DEFAULT 'thailand',
    description     text,
    status          legal_case_status NOT NULL DEFAULT 'open',
    risk_level      risk_level,
    win_probability float CHECK (win_probability BETWEEN 0 AND 1),

    -- Important dates
    incident_date   date,
    filing_date     date,
    next_hearing    date,
    deadline        date,

    -- Financial
    claim_amount    numeric(15,2),
    currency        text DEFAULT 'THB',

    -- Metadata
    court_case_no   text,               -- เลขคดีที่ศาล (if filed)
    court_name      text,
    tags            text[] DEFAULT '{}',
    metadata        jsonb NOT NULL DEFAULT '{}',

    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX legal_cases_tenant_idx    ON legal_cases(tenant_id);
CREATE INDEX legal_cases_client_idx    ON legal_cases(client_id);
CREATE INDEX legal_cases_lawyer_idx    ON legal_cases(lawyer_id) WHERE lawyer_id IS NOT NULL;
CREATE INDEX legal_cases_status_idx    ON legal_cases(status);
CREATE INDEX legal_cases_type_idx      ON legal_cases(case_type);
CREATE INDEX legal_cases_tags_idx      ON legal_cases USING gin(tags);


-- Chat Sessions (each conversation)
CREATE TABLE case_sessions (
    id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    legal_case_id   uuid REFERENCES legal_cases(id) ON DELETE SET NULL,
    user_id         uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    title           text,               -- Auto-generated from first message
    status          session_status NOT NULL DEFAULT 'active',
    query_type      text,               -- 'legal_question', 'document_review', 'case_strategy', etc.
    agents_used     agent_name[] DEFAULT '{}',
    message_count   int NOT NULL DEFAULT 0,
    total_tokens    int NOT NULL DEFAULT 0,
    total_cost_usd  numeric(10,6) NOT NULL DEFAULT 0,

    -- Last interaction summary for quick context
    last_summary    text,

    metadata        jsonb NOT NULL DEFAULT '{}',
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    closed_at       timestamptz
);

CREATE INDEX case_sessions_tenant_idx  ON case_sessions(tenant_id);
CREATE INDEX case_sessions_case_idx    ON case_sessions(legal_case_id) WHERE legal_case_id IS NOT NULL;
CREATE INDEX case_sessions_user_idx    ON case_sessions(user_id);
CREATE INDEX case_sessions_status_idx  ON case_sessions(status);
CREATE INDEX case_sessions_created_idx ON case_sessions(created_at DESC);


-- Messages (individual Q&A turns)
CREATE TABLE messages (
    id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      uuid NOT NULL REFERENCES case_sessions(id) ON DELETE CASCADE,
    tenant_id       uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,

    role            text NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content         text NOT NULL,

    -- AI response metadata (populated when role = 'assistant')
    irac_output     jsonb,              -- Full IRAC JSON (issue/rule/application/conclusion)
    citations       jsonb DEFAULT '[]', -- Array of citation objects
    confidence      float CHECK (confidence BETWEEN 0 AND 1),
    agents_used     agent_name[] DEFAULT '{}',
    model_used      text,               -- e.g., 'claude-opus-4-6', 'gpt-4o'
    tokens_used     int,
    cost_usd        numeric(10,6),
    latency_ms      int,

    -- Retrieval metadata
    retrieved_law_ids  uuid[],
    retrieved_case_ids uuid[],

    -- Escalation
    escalated       boolean NOT NULL DEFAULT false,
    escalation_reason text,

    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX messages_session_idx  ON messages(session_id);
CREATE INDEX messages_tenant_idx   ON messages(tenant_id);
CREATE INDEX messages_role_idx     ON messages(role);
CREATE INDEX messages_created_idx  ON messages(created_at DESC);
CREATE INDEX messages_confidence_idx ON messages(confidence) WHERE role = 'assistant';


-- ============================================================
-- SECTION 5: CASE MEMORY SYSTEM
-- ============================================================

-- Persistent case memory (AI remembers everything about each case)
CREATE TABLE case_memory (
    id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    legal_case_id   uuid NOT NULL REFERENCES legal_cases(id) ON DELETE CASCADE,
    tenant_id       uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,

    -- Living summary (updated after each session)
    facts_summary   text,               -- AI-generated running summary of case facts
    legal_issues    jsonb DEFAULT '[]', -- Key legal issues identified so far
    timeline        jsonb DEFAULT '[]', -- Chronological events: [{date, event, source}]

    -- Legal work product accumulated
    irac_history    jsonb DEFAULT '[]', -- All past IRAC analyses: [{session_id, date, irac}]
    key_citations   jsonb DEFAULT '[]', -- Most important citations for this case
    arguments_used  jsonb DEFAULT '[]', -- Legal arguments constructed: [{arg, strength, session}]
    counter_args    jsonb DEFAULT '[]', -- Counter-arguments identified

    -- Strategy tracking
    strategies      jsonb DEFAULT '[]', -- [{name, pros, cons, cost, win_prob, session_id}]
    recommended_strategy text,

    -- Evidence summary (links to evidence table)
    evidence_summary jsonb DEFAULT '[]',

    -- References to sessions and documents
    session_ids     uuid[] DEFAULT '{}',
    document_ids    uuid[] DEFAULT '{}',
    evidence_ids    uuid[] DEFAULT '{}',

    -- Stats
    total_sessions  int NOT NULL DEFAULT 0,
    last_accessed   timestamptz NOT NULL DEFAULT now(),
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),

    UNIQUE (legal_case_id)  -- One memory record per case
);

CREATE INDEX case_memory_case_idx   ON case_memory(legal_case_id);
CREATE INDEX case_memory_tenant_idx ON case_memory(tenant_id);
CREATE INDEX case_memory_updated_idx ON case_memory(updated_at DESC);


-- ============================================================
-- SECTION 6: DOCUMENTS & EVIDENCE
-- ============================================================

-- Documents (contracts, legal docs, etc.)
CREATE TABLE documents (
    id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    legal_case_id   uuid REFERENCES legal_cases(id) ON DELETE SET NULL,
    session_id      uuid REFERENCES case_sessions(id) ON DELETE SET NULL,
    uploaded_by     uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    -- File info
    file_name       text NOT NULL,
    file_path       text NOT NULL,      -- Supabase Storage path
    file_type       text NOT NULL,      -- 'pdf', 'docx', 'txt', etc.
    file_size_bytes bigint,
    checksum        text,               -- SHA-256 for integrity

    -- Analysis result
    analysis        jsonb,              -- Document Analysis Agent output
    extracted_text  text,               -- OCR / parsed text
    key_clauses     jsonb DEFAULT '[]', -- [{clause, risk_level, notes}]
    risk_flags      jsonb DEFAULT '[]', -- [{issue, severity, recommendation}]

    -- Classification
    doc_category    text,               -- 'contract', 'evidence', 'pleading', etc.
    is_analyzed     boolean NOT NULL DEFAULT false,
    is_privileged   boolean NOT NULL DEFAULT false, -- Attorney-client privilege

    -- Embedding for semantic search within documents
    embedding       vector(1536),

    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX documents_tenant_idx  ON documents(tenant_id);
CREATE INDEX documents_case_idx    ON documents(legal_case_id) WHERE legal_case_id IS NOT NULL;
CREATE INDEX documents_session_idx ON documents(session_id) WHERE session_id IS NOT NULL;
CREATE INDEX documents_type_idx    ON documents(file_type);
CREATE INDEX documents_hnsw ON documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE embedding IS NOT NULL;


-- Evidence (multimodal: image, audio, video, email, etc.)
CREATE TABLE evidence (
    id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    legal_case_id   uuid REFERENCES legal_cases(id) ON DELETE SET NULL,
    session_id      uuid REFERENCES case_sessions(id) ON DELETE SET NULL,
    uploaded_by     uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    -- File info
    file_name       text NOT NULL,
    file_path       text NOT NULL,
    evidence_type   evidence_type NOT NULL,
    file_size_bytes bigint,
    duration_sec    int,                -- For audio/video

    -- Analysis results
    transcript      text,               -- Whisper transcription (audio/video)
    analysis        jsonb,              -- Evidence Analyzer Agent output
    legal_relevance float CHECK (legal_relevance BETWEEN 0 AND 1),
    admissibility   text,               -- 'admissible', 'questionable', 'inadmissible'
    admissibility_notes text,
    key_findings    jsonb DEFAULT '[]', -- [{finding, importance, timestamp}]

    -- Processing status
    is_processed    boolean NOT NULL DEFAULT false,
    processing_error text,

    -- Chain of custody
    acquisition_date date,
    acquisition_notes text,
    is_original     boolean NOT NULL DEFAULT true,

    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX evidence_tenant_idx   ON evidence(tenant_id);
CREATE INDEX evidence_case_idx     ON evidence(legal_case_id) WHERE legal_case_id IS NOT NULL;
CREATE INDEX evidence_type_idx     ON evidence(evidence_type);
CREATE INDEX evidence_processed_idx ON evidence(is_processed);


-- ============================================================
-- SECTION 7: CITATION VERIFICATION LOG
-- ============================================================

CREATE TABLE citations_log (
    id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    message_id      uuid NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    session_id      uuid NOT NULL REFERENCES case_sessions(id) ON DELETE CASCADE,

    -- Citation details
    citation_text   text NOT NULL,      -- Raw citation string from AI
    citation_type   text,               -- 'statute', 'case', 'regulation'
    cited_law_id    uuid REFERENCES laws(id)  ON DELETE SET NULL,
    cited_case_id   uuid REFERENCES cases(id) ON DELETE SET NULL,

    -- Verification result
    status          citation_status NOT NULL DEFAULT 'pending',
    verified_at     timestamptz,
    verified_by     text,               -- 'system' or user_id
    rejection_reason text,              -- If rejected, why

    -- Metadata
    jurisdiction    jurisdiction,
    year_referenced int,
    source_url      text,               -- Link to verified source

    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX citations_log_message_idx ON citations_log(message_id);
CREATE INDEX citations_log_session_idx ON citations_log(session_id);
CREATE INDEX citations_log_status_idx  ON citations_log(status);
CREATE INDEX citations_log_tenant_idx  ON citations_log(tenant_id);
CREATE INDEX citations_log_created_idx ON citations_log(created_at DESC);


-- ============================================================
-- SECTION 8: AUDIT & COMPLIANCE
-- ============================================================

-- Full audit trail (every AI action)
CREATE TABLE audit_log (
    id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    user_id         uuid REFERENCES users(id) ON DELETE SET NULL,
    session_id      uuid REFERENCES case_sessions(id) ON DELETE SET NULL,
    message_id      uuid REFERENCES messages(id) ON DELETE SET NULL,
    legal_case_id   uuid REFERENCES legal_cases(id) ON DELETE SET NULL,

    -- Action details
    action          text NOT NULL,      -- 'query', 'document_upload', 'citation_verify', etc.
    agents_used     agent_name[] DEFAULT '{}',
    model_used      text,
    query_hash      text,               -- SHA-256 of query (for privacy)
    response_hash   text,               -- SHA-256 of response

    -- Performance
    confidence      float,
    tokens_used     int,
    cost_usd        numeric(10,6),
    latency_ms      int,

    -- Outcome
    success         boolean NOT NULL DEFAULT true,
    error_message   text,
    escalated       boolean NOT NULL DEFAULT false,
    citation_rejection_count int NOT NULL DEFAULT 0,

    -- Immutable timestamp
    created_at      timestamptz NOT NULL DEFAULT now()
);

-- Audit log is append-only — no UPDATE or DELETE allowed (enforced by trigger)
CREATE INDEX audit_log_tenant_idx     ON audit_log(tenant_id);
CREATE INDEX audit_log_user_idx       ON audit_log(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX audit_log_session_idx    ON audit_log(session_id) WHERE session_id IS NOT NULL;
CREATE INDEX audit_log_action_idx     ON audit_log(action);
CREATE INDEX audit_log_created_idx    ON audit_log(created_at DESC);
CREATE INDEX audit_log_escalated_idx  ON audit_log(escalated) WHERE escalated = true;


-- Expert Review Queue (human-in-the-loop)
CREATE TABLE expert_reviews (
    id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    session_id      uuid NOT NULL REFERENCES case_sessions(id) ON DELETE CASCADE,
    message_id      uuid NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    legal_case_id   uuid REFERENCES legal_cases(id) ON DELETE SET NULL,

    -- Flagging
    flagged_reason  text NOT NULL,
    confidence_at_flag float,
    citation_rejection_count int NOT NULL DEFAULT 0,
    auto_flagged    boolean NOT NULL DEFAULT true,

    -- Assignment
    reviewer_id     uuid REFERENCES users(id) ON DELETE SET NULL,
    assigned_at     timestamptz,

    -- Resolution
    status          review_status NOT NULL DEFAULT 'pending',
    resolution      text,               -- Expert's conclusion
    corrected_irac  jsonb,              -- Corrected IRAC if applicable
    reviewed_at     timestamptz,

    priority        int NOT NULL DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
    -- 1 = highest priority, 10 = lowest

    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX expert_reviews_tenant_idx   ON expert_reviews(tenant_id);
CREATE INDEX expert_reviews_status_idx   ON expert_reviews(status);
CREATE INDEX expert_reviews_priority_idx ON expert_reviews(priority, created_at);
CREATE INDEX expert_reviews_reviewer_idx ON expert_reviews(reviewer_id) WHERE reviewer_id IS NOT NULL;


-- ============================================================
-- SECTION 9: FEEDBACK & IMPROVEMENT
-- ============================================================

CREATE TABLE feedback (
    id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    session_id      uuid NOT NULL REFERENCES case_sessions(id) ON DELETE CASCADE,
    message_id      uuid NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    user_id         uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    rating          int NOT NULL CHECK (rating BETWEEN 1 AND 5),
    feedback_type   text,               -- 'wrong_law', 'wrong_reasoning', 'wrong_citation', 'other'
    comment         text,
    corrected_answer text,              -- User or expert's correct answer

    -- Processing
    is_processed    boolean NOT NULL DEFAULT false,
    processed_at    timestamptz,
    added_to_dataset boolean NOT NULL DEFAULT false,

    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX feedback_session_idx   ON feedback(session_id);
CREATE INDEX feedback_rating_idx    ON feedback(rating);
CREATE INDEX feedback_processed_idx ON feedback(is_processed) WHERE is_processed = false;
CREATE INDEX feedback_tenant_idx    ON feedback(tenant_id);


-- ============================================================
-- SECTION 10: PERFORMANCE & ANALYTICS
-- ============================================================

-- Query analytics (for monitoring and optimization)
CREATE TABLE query_analytics (
    id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    session_id      uuid NOT NULL REFERENCES case_sessions(id) ON DELETE CASCADE,
    message_id      uuid NOT NULL REFERENCES messages(id) ON DELETE CASCADE,

    query_type      text NOT NULL,
    agents_invoked  agent_name[] DEFAULT '{}',
    agent_latencies jsonb DEFAULT '{}',     -- {agent_name: latency_ms}
    total_latency_ms int,

    -- RAG metrics
    docs_retrieved  int,
    docs_after_rerank int,
    top_similarity  float,
    graph_nodes_expanded int NOT NULL DEFAULT 0,

    -- Quality metrics
    confidence      float,
    citation_count  int NOT NULL DEFAULT 0,
    citation_verified_count int NOT NULL DEFAULT 0,
    citation_rejected_count int NOT NULL DEFAULT 0,

    -- Cost
    total_tokens    int,
    total_cost_usd  numeric(10,6),

    -- Cache
    cache_hit       boolean NOT NULL DEFAULT false,

    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX query_analytics_tenant_idx  ON query_analytics(tenant_id);
CREATE INDEX query_analytics_created_idx ON query_analytics(created_at DESC);
CREATE INDEX query_analytics_type_idx    ON query_analytics(query_type);
CREATE INDEX query_analytics_latency_idx ON query_analytics(total_latency_ms DESC);


-- ============================================================
-- SECTION 11: ROW-LEVEL SECURITY (RLS)
-- ============================================================

-- Enable RLS on all sensitive tables
ALTER TABLE tenants        ENABLE ROW LEVEL SECURITY;
ALTER TABLE users          ENABLE ROW LEVEL SECURITY;
ALTER TABLE legal_cases    ENABLE ROW LEVEL SECURITY;
ALTER TABLE case_sessions  ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages       ENABLE ROW LEVEL SECURITY;
ALTER TABLE case_memory    ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents      ENABLE ROW LEVEL SECURITY;
ALTER TABLE evidence       ENABLE ROW LEVEL SECURITY;
ALTER TABLE citations_log  ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log      ENABLE ROW LEVEL SECURITY;
ALTER TABLE expert_reviews ENABLE ROW LEVEL SECURITY;
ALTER TABLE feedback       ENABLE ROW LEVEL SECURITY;
ALTER TABLE query_analytics ENABLE ROW LEVEL SECURITY;

-- Helper function: get current user's tenant_id
CREATE OR REPLACE FUNCTION get_user_tenant_id()
RETURNS uuid
LANGUAGE sql STABLE SECURITY DEFINER
AS $$
    SELECT tenant_id FROM users WHERE id = auth.uid()
$$;

-- Helper function: get current user's role
CREATE OR REPLACE FUNCTION get_user_role()
RETURNS user_role
LANGUAGE sql STABLE SECURITY DEFINER
AS $$
    SELECT role FROM users WHERE id = auth.uid()
$$;

-- Tenants: users see only their own tenant
CREATE POLICY tenants_isolation ON tenants
    FOR ALL USING (id = get_user_tenant_id());

-- Users: see only users in same tenant
CREATE POLICY users_tenant_isolation ON users
    FOR ALL USING (tenant_id = get_user_tenant_id());

-- Legal cases: tenant isolation
CREATE POLICY legal_cases_tenant ON legal_cases
    FOR ALL USING (tenant_id = get_user_tenant_id());

-- Case sessions: tenant isolation
CREATE POLICY case_sessions_tenant ON case_sessions
    FOR ALL USING (tenant_id = get_user_tenant_id());

-- Messages: tenant isolation
CREATE POLICY messages_tenant ON messages
    FOR ALL USING (tenant_id = get_user_tenant_id());

-- Case memory: tenant isolation
CREATE POLICY case_memory_tenant ON case_memory
    FOR ALL USING (tenant_id = get_user_tenant_id());

-- Documents: tenant isolation + privilege check
CREATE POLICY documents_tenant ON documents
    FOR ALL USING (
        tenant_id = get_user_tenant_id()
        AND (
            NOT is_privileged
            OR get_user_role() IN ('admin', 'super_admin', 'lawyer')
        )
    );

-- Evidence: tenant isolation
CREATE POLICY evidence_tenant ON evidence
    FOR ALL USING (tenant_id = get_user_tenant_id());

-- Citations log: tenant isolation
CREATE POLICY citations_tenant ON citations_log
    FOR ALL USING (tenant_id = get_user_tenant_id());

-- Audit log: admins and auditors only
CREATE POLICY audit_log_restricted ON audit_log
    FOR SELECT USING (
        tenant_id = get_user_tenant_id()
        AND get_user_role() IN ('admin', 'super_admin', 'auditor')
    );

-- Expert reviews: reviewers and admins
CREATE POLICY expert_reviews_access ON expert_reviews
    FOR ALL USING (
        tenant_id = get_user_tenant_id()
        AND get_user_role() IN ('admin', 'super_admin', 'expert_reviewer', 'lawyer')
    );

-- Feedback: users see own feedback, admins see all
CREATE POLICY feedback_access ON feedback
    FOR ALL USING (
        tenant_id = get_user_tenant_id()
        AND (user_id = auth.uid() OR get_user_role() IN ('admin', 'super_admin'))
    );

-- Query analytics: admins only
CREATE POLICY query_analytics_admin ON query_analytics
    FOR SELECT USING (
        tenant_id = get_user_tenant_id()
        AND get_user_role() IN ('admin', 'super_admin')
    );

-- Laws and cases: readable by all (shared knowledge base)
-- But only admins can insert/update
CREATE POLICY laws_read ON laws
    FOR SELECT USING (tenant_id IS NULL OR tenant_id = get_user_tenant_id());

CREATE POLICY laws_write ON laws
    FOR INSERT WITH CHECK (get_user_role() IN ('admin', 'super_admin'));

CREATE POLICY cases_read ON cases
    FOR SELECT USING (tenant_id IS NULL OR tenant_id = get_user_tenant_id());

CREATE POLICY cases_write ON cases
    FOR INSERT WITH CHECK (get_user_role() IN ('admin', 'super_admin'));


-- ============================================================
-- SECTION 12: CORE DATABASE FUNCTIONS
-- ============================================================

-- 12.1: Hybrid legal search (Semantic + BM25 + RRF reranking)
CREATE OR REPLACE FUNCTION hybrid_legal_search(
    query_embedding     vector(1536),
    query_text          text,
    p_jurisdiction      jurisdiction    DEFAULT NULL,
    p_doc_type          legal_doc_type  DEFAULT NULL,
    p_status            law_status      DEFAULT 'active',
    match_count         int             DEFAULT 10,
    rrf_k               int             DEFAULT 60
)
RETURNS TABLE (
    id              uuid,
    source_table    text,
    title           text,
    content         text,
    doc_type        text,
    jurisdiction    text,
    status          text,
    year            int,
    metadata        jsonb,
    semantic_rank   bigint,
    keyword_rank    bigint,
    final_score     float
)
LANGUAGE sql STABLE
AS $$
    -- Laws: semantic search
    WITH law_semantic AS (
        SELECT
            l.id,
            'laws'          AS source_table,
            l.title,
            l.full_text     AS content,
            l.doc_type::text,
            l.jurisdiction::text,
            l.status::text,
            l.year_be       AS year,
            l.metadata,
            ROW_NUMBER() OVER (ORDER BY l.embedding <=> query_embedding) AS sem_rank
        FROM laws l
        WHERE l.embedding IS NOT NULL
          AND (p_jurisdiction IS NULL OR l.jurisdiction = p_jurisdiction)
          AND (p_doc_type IS NULL OR l.doc_type = p_doc_type)
          AND (p_status IS NULL OR l.status = p_status)
        ORDER BY l.embedding <=> query_embedding
        LIMIT 40
    ),

    -- Laws: keyword search (BM25-style with ts_rank)
    law_keyword AS (
        SELECT
            l.id,
            'laws'          AS source_table,
            l.title,
            l.full_text     AS content,
            l.doc_type::text,
            l.jurisdiction::text,
            l.status::text,
            l.year_be       AS year,
            l.metadata,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank(
                    to_tsvector('simple', coalesce(l.title,'') || ' ' || coalesce(l.full_text,'')),
                    plainto_tsquery('simple', query_text)
                ) DESC
            ) AS kw_rank
        FROM laws l
        WHERE to_tsvector('simple', coalesce(l.title,'') || ' ' || coalesce(l.full_text,''))
              @@ plainto_tsquery('simple', query_text)
          AND (p_jurisdiction IS NULL OR l.jurisdiction = p_jurisdiction)
          AND (p_status IS NULL OR l.status = p_status)
        LIMIT 40
    ),

    -- Cases: semantic search
    case_semantic AS (
        SELECT
            c.id,
            'cases'             AS source_table,
            c.case_no           AS title,
            coalesce(c.ratio_decidendi, c.ruling) AS content,
            c.doc_type::text,
            c.jurisdiction::text,
            c.status::text,
            c.year_be           AS year,
            c.metadata,
            ROW_NUMBER() OVER (ORDER BY c.embedding <=> query_embedding) AS sem_rank
        FROM cases c
        WHERE c.embedding IS NOT NULL
          AND (p_jurisdiction IS NULL OR c.jurisdiction = p_jurisdiction)
        ORDER BY c.embedding <=> query_embedding
        LIMIT 40
    ),

    -- Cases: keyword search
    case_keyword AS (
        SELECT
            c.id,
            'cases'             AS source_table,
            c.case_no           AS title,
            coalesce(c.ratio_decidendi, c.ruling) AS content,
            c.doc_type::text,
            c.jurisdiction::text,
            c.status::text,
            c.year_be           AS year,
            c.metadata,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank(
                    to_tsvector('simple',
                        coalesce(c.case_no,'') || ' ' ||
                        coalesce(c.facts,'')   || ' ' ||
                        coalesce(c.ruling,'')  || ' ' ||
                        coalesce(c.ratio_decidendi,'')
                    ),
                    plainto_tsquery('simple', query_text)
                ) DESC
            ) AS kw_rank
        FROM cases c
        WHERE to_tsvector('simple',
                coalesce(c.case_no,'') || ' ' ||
                coalesce(c.facts,'')   || ' ' ||
                coalesce(c.ruling,'')  || ' ' ||
                coalesce(c.ratio_decidendi,'')
              ) @@ plainto_tsquery('simple', query_text)
          AND (p_jurisdiction IS NULL OR c.jurisdiction = p_jurisdiction)
        LIMIT 40
    ),

    -- Reciprocal Rank Fusion: combine all sources
    all_results AS (
        SELECT id, source_table, title, content, doc_type, jurisdiction, status, year, metadata,
               sem_rank::bigint, NULL::bigint AS kw_rank
        FROM law_semantic
        UNION ALL
        SELECT id, source_table, title, content, doc_type, jurisdiction, status, year, metadata,
               NULL::bigint, kw_rank::bigint
        FROM law_keyword
        UNION ALL
        SELECT id, source_table, title, content, doc_type, jurisdiction, status, year, metadata,
               sem_rank::bigint, NULL::bigint
        FROM case_semantic
        UNION ALL
        SELECT id, source_table, title, content, doc_type, jurisdiction, status, year, metadata,
               NULL::bigint, kw_rank::bigint
        FROM case_keyword
    ),

    fused AS (
        SELECT
            id, source_table, title, content, doc_type, jurisdiction, status, year, metadata,
            MIN(sem_rank) AS semantic_rank,
            MIN(kw_rank)  AS keyword_rank,
            COALESCE(SUM(1.0 / (rrf_k + sem_rank)), 0) +
            COALESCE(SUM(1.0 / (rrf_k + kw_rank)),  0) AS final_score
        FROM all_results
        GROUP BY id, source_table, title, content, doc_type, jurisdiction, status, year, metadata
    )

    SELECT * FROM fused
    ORDER BY final_score DESC
    LIMIT match_count;
$$;


-- 12.2: Get precedent chain via recursive graph traversal
CREATE OR REPLACE FUNCTION get_precedent_chain(
    start_case_id   uuid,
    max_depth       int DEFAULT 3
)
RETURNS TABLE (
    case_id         uuid,
    case_no         text,
    court           text,
    year_be         int,
    outcome         text,
    relationship    text,
    depth           int,
    path            uuid[]
)
LANGUAGE sql STABLE
AS $$
    WITH RECURSIVE chain AS (
        -- Base: direct citations from starting case
        SELECT
            cc.cited_case_id            AS case_id,
            cc.relationship::text,
            1                           AS depth,
            ARRAY[start_case_id]        AS path
        FROM case_citations cc
        WHERE cc.source_case_id = start_case_id
          AND cc.cited_case_id IS NOT NULL

        UNION ALL

        -- Recursive: follow the chain
        SELECT
            cc.cited_case_id,
            cc.relationship::text,
            c.depth + 1,
            c.path || cc.source_case_id
        FROM case_citations cc
        JOIN chain c ON cc.source_case_id = c.case_id
        WHERE c.depth < max_depth
          AND cc.cited_case_id IS NOT NULL
          AND cc.cited_case_id != ALL(c.path)  -- prevent infinite loops
    )
    SELECT
        ch.case_id,
        ca.case_no,
        ca.court,
        ca.year_be,
        ca.outcome::text,
        ch.relationship,
        ch.depth,
        ch.path
    FROM chain ch
    JOIN cases ca ON ca.id = ch.case_id
    ORDER BY ch.depth, ca.year_be DESC;
$$;


-- 12.3: Get or create case memory
CREATE OR REPLACE FUNCTION get_or_create_case_memory(
    p_legal_case_id uuid,
    p_tenant_id     uuid
)
RETURNS uuid
LANGUAGE plpgsql
AS $$
DECLARE
    v_memory_id uuid;
BEGIN
    SELECT id INTO v_memory_id
    FROM case_memory
    WHERE legal_case_id = p_legal_case_id;

    IF v_memory_id IS NULL THEN
        INSERT INTO case_memory (legal_case_id, tenant_id)
        VALUES (p_legal_case_id, p_tenant_id)
        RETURNING id INTO v_memory_id;
    END IF;

    -- Update last accessed timestamp
    UPDATE case_memory
    SET last_accessed = now()
    WHERE id = v_memory_id;

    RETURN v_memory_id;
END;
$$;


-- 12.4: Update case memory after session
CREATE OR REPLACE FUNCTION update_case_memory_after_session(
    p_legal_case_id uuid,
    p_session_id    uuid,
    p_irac_result   jsonb,
    p_citations     jsonb,
    p_document_ids  uuid[],
    p_evidence_ids  uuid[]
)
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE case_memory SET
        irac_history    = irac_history || jsonb_build_array(
                            jsonb_build_object(
                                'session_id',   p_session_id,
                                'timestamp',    now(),
                                'irac',         p_irac_result
                            )
                          ),
        key_citations   = (
            SELECT jsonb_agg(DISTINCT elem)
            FROM jsonb_array_elements(key_citations || p_citations) elem
        ),
        session_ids     = array_append(session_ids, p_session_id),
        document_ids    = array_cat(document_ids, p_document_ids),
        evidence_ids    = array_cat(evidence_ids, p_evidence_ids),
        total_sessions  = total_sessions + 1,
        last_accessed   = now(),
        updated_at      = now()
    WHERE legal_case_id = p_legal_case_id;
END;
$$;


-- 12.5: Flag session for expert review
CREATE OR REPLACE FUNCTION flag_for_expert_review(
    p_session_id        uuid,
    p_message_id        uuid,
    p_reason            text,
    p_confidence        float,
    p_rejection_count   int
)
RETURNS uuid
LANGUAGE plpgsql
AS $$
DECLARE
    v_tenant_id     uuid;
    v_case_id       uuid;
    v_review_id     uuid;
    v_priority      int;
BEGIN
    SELECT tenant_id, legal_case_id
    INTO v_tenant_id, v_case_id
    FROM case_sessions WHERE id = p_session_id;

    -- Calculate priority (lower confidence = higher priority)
    v_priority := CASE
        WHEN p_confidence < 0.4 THEN 1
        WHEN p_confidence < 0.5 THEN 2
        WHEN p_confidence < 0.6 THEN 3
        WHEN p_confidence < 0.7 THEN 4
        ELSE 5
    END;

    INSERT INTO expert_reviews (
        tenant_id, session_id, message_id, legal_case_id,
        flagged_reason, confidence_at_flag, citation_rejection_count,
        priority
    )
    VALUES (
        v_tenant_id, p_session_id, p_message_id, v_case_id,
        p_reason, p_confidence, p_rejection_count,
        v_priority
    )
    RETURNING id INTO v_review_id;

    -- Mark session as escalated
    UPDATE case_sessions SET status = 'escalated' WHERE id = p_session_id;

    RETURN v_review_id;
END;
$$;


-- 12.6: Get session analytics summary
CREATE OR REPLACE FUNCTION get_session_stats(p_tenant_id uuid, p_days int DEFAULT 30)
RETURNS TABLE (
    total_sessions      bigint,
    total_queries       bigint,
    avg_confidence      float,
    avg_latency_ms      float,
    escalation_rate     float,
    citation_accuracy   float,
    total_cost_usd      numeric
)
LANGUAGE sql STABLE
AS $$
    SELECT
        COUNT(DISTINCT s.id)                                            AS total_sessions,
        COUNT(m.id)                                                     AS total_queries,
        AVG(m.confidence)                                               AS avg_confidence,
        AVG(m.latency_ms)                                               AS avg_latency_ms,
        AVG(CASE WHEN m.escalated THEN 1.0 ELSE 0.0 END)               AS escalation_rate,
        AVG(
            CASE
                WHEN qa.citation_count > 0
                THEN qa.citation_verified_count::float / qa.citation_count
                ELSE NULL
            END
        )                                                               AS citation_accuracy,
        SUM(s.total_cost_usd)                                           AS total_cost_usd
    FROM case_sessions s
    JOIN messages m ON m.session_id = s.id AND m.role = 'assistant'
    LEFT JOIN query_analytics qa ON qa.message_id = m.id
    WHERE s.tenant_id = p_tenant_id
      AND s.created_at >= now() - (p_days || ' days')::interval;
$$;


-- ============================================================
-- SECTION 13: TRIGGERS
-- ============================================================

-- 13.1: Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

-- Apply to all tables with updated_at
DO $$
DECLARE
    t text;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'tenants', 'users', 'laws', 'cases', 'legal_forms',
        'legal_cases', 'case_sessions', 'case_memory',
        'documents', 'evidence', 'expert_reviews'
    ] LOOP
        EXECUTE format(
            'CREATE TRIGGER trg_%s_updated_at
             BEFORE UPDATE ON %s
             FOR EACH ROW EXECUTE FUNCTION set_updated_at()',
            t, t
        );
    END LOOP;
END;
$$;


-- 13.2: Prevent DELETE/UPDATE on audit_log (immutable)
CREATE OR REPLACE FUNCTION block_audit_mutation()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'audit_log is immutable — no UPDATE or DELETE allowed';
END;
$$;

CREATE TRIGGER trg_audit_no_update
    BEFORE UPDATE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION block_audit_mutation();

CREATE TRIGGER trg_audit_no_delete
    BEFORE DELETE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION block_audit_mutation();


-- 13.3: Auto-increment message count on case_sessions
CREATE OR REPLACE FUNCTION increment_session_message_count()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    UPDATE case_sessions
    SET message_count = message_count + 1,
        updated_at    = now()
    WHERE id = NEW.session_id;
    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_messages_count
    AFTER INSERT ON messages
    FOR EACH ROW EXECUTE FUNCTION increment_session_message_count();


-- 13.4: Auto-flag when citation rejection rate is too high
CREATE OR REPLACE FUNCTION auto_flag_high_rejection()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    v_rejection_count int;
    v_total_count     int;
    v_session_id      uuid;
BEGIN
    v_session_id := NEW.session_id;

    SELECT
        COUNT(*) FILTER (WHERE status = 'rejected'),
        COUNT(*)
    INTO v_rejection_count, v_total_count
    FROM citations_log
    WHERE session_id = v_session_id;

    -- Flag if rejection rate > 20% with at least 5 citations checked
    IF v_total_count >= 5 AND (v_rejection_count::float / v_total_count) > 0.2 THEN
        PERFORM flag_for_expert_review(
            v_session_id,
            NEW.message_id,
            'High citation rejection rate: ' || v_rejection_count || '/' || v_total_count,
            0.5,
            v_rejection_count
        );
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_auto_flag_rejections
    AFTER INSERT OR UPDATE OF status ON citations_log
    FOR EACH ROW
    WHEN (NEW.status = 'rejected')
    EXECUTE FUNCTION auto_flag_high_rejection();


-- 13.5: Auto-close session and update cost
CREATE OR REPLACE FUNCTION sync_session_cost()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    UPDATE case_sessions
    SET total_cost_usd  = total_cost_usd + coalesce(NEW.cost_usd, 0),
        total_tokens    = total_tokens   + coalesce(NEW.tokens_used, 0),
        updated_at      = now()
    WHERE id = NEW.session_id;
    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_messages_cost
    AFTER INSERT ON messages
    FOR EACH ROW
    WHEN (NEW.role = 'assistant')
    EXECUTE FUNCTION sync_session_cost();


-- ============================================================
-- SECTION 14: MATERIALIZED VIEWS (Performance Optimization)
-- ============================================================

-- 14.1: Active laws summary (for quick reference / admin dashboard)
CREATE MATERIALIZED VIEW active_laws_summary AS
SELECT
    id,
    title,
    short_title,
    law_code,
    doc_type,
    jurisdiction,
    section_number,
    year_be,
    status,
    tags
FROM laws
WHERE status = 'active'
  AND embedding IS NOT NULL
WITH DATA;

CREATE UNIQUE INDEX ON active_laws_summary(id);
CREATE INDEX ON active_laws_summary(jurisdiction, doc_type);


-- 14.2: Case law index (for quick precedent lookup)
CREATE MATERIALIZED VIEW case_law_index AS
SELECT
    id,
    case_no,
    court,
    court_level,
    jurisdiction,
    outcome,
    year_be,
    summary,
    tags
FROM cases
WHERE status = 'active'
  AND embedding IS NOT NULL
WITH DATA;

CREATE UNIQUE INDEX ON case_law_index(id);
CREATE INDEX ON case_law_index(jurisdiction, outcome);
CREATE INDEX ON case_law_index(year_be DESC);


-- 14.3: Tenant usage stats (for billing/monitoring)
CREATE MATERIALIZED VIEW tenant_usage_stats AS
SELECT
    t.id        AS tenant_id,
    t.name,
    t.plan,
    COUNT(DISTINCT u.id)   AS user_count,
    COUNT(DISTINCT lc.id)  AS case_count,
    COUNT(DISTINCT cs.id)  AS session_count,
    SUM(cs.total_cost_usd) AS total_cost_usd,
    MAX(cs.created_at)     AS last_activity
FROM tenants t
LEFT JOIN users         u  ON u.tenant_id  = t.id
LEFT JOIN legal_cases   lc ON lc.tenant_id = t.id
LEFT JOIN case_sessions cs ON cs.tenant_id = t.id
GROUP BY t.id, t.name, t.plan
WITH DATA;

CREATE UNIQUE INDEX ON tenant_usage_stats(tenant_id);

-- Refresh function (call via pg_cron or manually)
CREATE OR REPLACE FUNCTION refresh_materialized_views()
RETURNS void LANGUAGE plpgsql AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY active_laws_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY case_law_index;
    REFRESH MATERIALIZED VIEW CONCURRENTLY tenant_usage_stats;
END;
$$;


-- ============================================================
-- SECTION 15: SEED DATA (Initial Setup)
-- ============================================================

-- Default tenant
INSERT INTO tenants (id, name, slug, plan)
VALUES (
    '00000000-0000-0000-0000-000000000001',
    'AI Lawyer Platform',
    'ai-lawyer',
    'enterprise'
) ON CONFLICT DO NOTHING;

-- Default super admin (update email before use)
INSERT INTO users (id, tenant_id, email, full_name, role)
VALUES (
    '00000000-0000-0000-0000-000000000010',
    '00000000-0000-0000-0000-000000000001',
    'admin@ailawyer.local',
    'System Administrator',
    'super_admin'
) ON CONFLICT DO NOTHING;

-- Sample law (ประมวลกฎหมายแพ่งและพาณิชย์ มาตรา 537)
INSERT INTO laws (
    title, short_title, law_code, doc_type,
    jurisdiction, section_number, full_text,
    summary, status, year_be, year_ce
)
VALUES (
    'ประมวลกฎหมายแพ่งและพาณิชย์ มาตรา 537 — สัญญาเช่าทรัพย์',
    'ปพพ. ม.537',
    'CCC',
    'statute',
    'thailand',
    '537',
    'อันว่าเช่าทรัพย์สินนั้น คือสัญญาซึ่งบุคคลคนหนึ่งเรียกว่าผู้ให้เช่า ตกลงให้บุคคลอีกคนหนึ่งเรียกว่าผู้เช่า ได้ใช้หรือได้รับประโยชน์ในทรัพย์สินอย่างใดอย่างหนึ่ง ชั่วระยะเวลาอันมีจำกัด และผู้เช่าตกลงจะให้ค่าเช่าเพื่อการนั้น',
    'กำหนดนิยามและองค์ประกอบของสัญญาเช่าทรัพย์ ได้แก่ ผู้ให้เช่า ผู้เช่า ทรัพย์สิน ระยะเวลา และค่าเช่า',
    'active',
    2535,
    1992
) ON CONFLICT DO NOTHING;


-- ============================================================
-- SECTION 16: INDEXES SUMMARY & MAINTENANCE
-- ============================================================

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS legal_cases_tenant_status_idx
    ON legal_cases(tenant_id, status, created_at DESC);

CREATE INDEX IF NOT EXISTS case_sessions_user_created_idx
    ON case_sessions(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS messages_session_role_idx
    ON messages(session_id, role, created_at DESC);

CREATE INDEX IF NOT EXISTS citations_log_session_status_idx
    ON citations_log(session_id, status, created_at DESC);

CREATE INDEX IF NOT EXISTS audit_log_tenant_created_idx
    ON audit_log(tenant_id, created_at DESC);

-- Partial indexes for common filters
CREATE INDEX IF NOT EXISTS expert_reviews_pending_idx
    ON expert_reviews(priority, created_at)
    WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS feedback_unprocessed_idx
    ON feedback(created_at)
    WHERE is_processed = false;

CREATE INDEX IF NOT EXISTS documents_unanalyzed_idx
    ON documents(created_at)
    WHERE is_analyzed = false;

CREATE INDEX IF NOT EXISTS evidence_unprocessed_idx
    ON evidence(created_at)
    WHERE is_processed = false;


-- ============================================================
-- SECTION 17: COMMENTS (Documentation)
-- ============================================================

COMMENT ON TABLE tenants        IS 'Law firms / organizations using the platform (multi-tenant)';
COMMENT ON TABLE users          IS 'Platform users — roles: super_admin, admin, lawyer, client, auditor, expert_reviewer';
COMMENT ON TABLE laws           IS 'Legal knowledge base: statutes, regulations, royal decrees + embeddings for RAG';
COMMENT ON TABLE cases          IS 'Case law knowledge base: court decisions + precedents + embeddings for RAG';
COMMENT ON TABLE case_citations IS 'Case law graph: tracks cites/overrules/follows relationships between cases and laws';
COMMENT ON TABLE legal_forms    IS 'Template contracts and legal forms for auto-drafting';
COMMENT ON TABLE legal_cases    IS 'Client cases managed through the platform (not court cases)';
COMMENT ON TABLE case_sessions  IS 'Each conversation thread between user and AI Lawyer';
COMMENT ON TABLE messages       IS 'Individual messages with full IRAC output and citation metadata';
COMMENT ON TABLE case_memory    IS 'Persistent AI memory per legal case — accumulates across all sessions';
COMMENT ON TABLE documents      IS 'Uploaded contracts and legal documents with analysis results';
COMMENT ON TABLE evidence       IS 'Multimodal evidence: images, audio, emails, chat logs';
COMMENT ON TABLE citations_log  IS 'Every citation the AI produces is logged and verified here';
COMMENT ON TABLE audit_log      IS 'Immutable audit trail — every AI action logged for compliance';
COMMENT ON TABLE expert_reviews IS 'Human-in-the-loop review queue for low-confidence responses';
COMMENT ON TABLE feedback       IS 'User/expert feedback for continuous model improvement';
COMMENT ON TABLE query_analytics IS 'Per-query performance and quality metrics';

COMMENT ON FUNCTION hybrid_legal_search IS 'Combines semantic (pgvector) + keyword (BM25) search with RRF reranking across laws and cases';
COMMENT ON FUNCTION get_precedent_chain IS 'Recursively traverses the case citation graph up to max_depth levels';
COMMENT ON FUNCTION get_or_create_case_memory IS 'Returns existing case memory or creates new one for the given legal_case_id';
COMMENT ON FUNCTION update_case_memory_after_session IS 'Appends IRAC result and citations to case memory after each session';
COMMENT ON FUNCTION flag_for_expert_review IS 'Creates expert review record and marks session as escalated';


-- ============================================================
-- DONE — AI Lawyer Database v2.0
-- Total: 13 tables · 6+ functions · 4 triggers · 3 mat. views
-- Compatible with: Supabase (PostgreSQL 15+) + pgvector
-- ============================================================