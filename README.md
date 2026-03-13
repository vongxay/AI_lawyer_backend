# ⚖️ AI Lawyer Backend — v2.0

> **Multi-Agent RAG + IRAC Legal Reasoning | FastAPI | Production-Ready**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  React 19 Frontend  (separate repo)                     │
└────────────────────┬────────────────────────────────────┘
                     │ REST / SSE
┌────────────────────▼────────────────────────────────────┐
│  LAYER 1 — API Gateway                                  │
│  FastAPI + JWT Auth + CORS + Rate Limiting              │
│  /api/v1/legal  /documents  /evidence  /memory  /admin  │
└────────────────────┬────────────────────────────────────┘
                     │ Depends(get_workflow_manager)
┌────────────────────▼────────────────────────────────────┐
│  LAYER 2 — Orchestration Engine                         │
│  WorkflowManager: PII → Classify → Plan → Execute       │
│  QueryClassifier + AgentSelector + AuditService         │
└────────────────────┬────────────────────────────────────┘
                     │ Parallel asyncio.TaskGroup
┌────────────────────▼────────────────────────────────────┐
│  LAYER 3 — Dynamic Multi-Agent Engine                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Research │ │ Document │ │ Evidence │ │  Risk &  │  │
│  │  Agent   │ │  Agent   │ │  Agent   │ │ Strategy │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│  ┌────────────────────┐  ┌──────────────────────────┐  │
│  │  IRAC Reasoning    │  │  Citation Verification   │  │
│  │  Agent (core)      │  │  Agent (core)            │  │
│  └────────────────────┘  └──────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  LAYER 4 — Knowledge & Memory                           │
│  RAG Pipeline: Embed → Hybrid Search → Graph → Rerank   │
│  CaseMemoryService: Redis (hot) + Supabase (persistent) │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  LAYER 5 — AI Model Layer                               │
│  Anthropic (Claude Sonnet 4.6 / Sonnet 4)               │
│  OpenAI (GPT-4o vision, GPT-4o-mini, text-embedding)    │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Redis 7+
- Supabase project (or run fully in stub/dev mode without it)

### Local Development

```bash
# 1. Clone and enter project
git clone <repo> && cd ai-lawyer-backend

# 2. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Configure environment
cp .env.example .env
# Edit .env — add your API keys

# 5. Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# 6. Run the server
uvicorn backend.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

### Docker Compose (recommended)

```bash
cp .env.example .env  # fill in API keys
docker compose up --build
```

---

## Project Structure

```
backend/
├── main.py                    # App factory + lifespan management
├── core/
│   ├── config.py              # Pydantic Settings — all env vars
│   ├── database.py            # Async Redis pool + Supabase client
│   ├── security.py            # JWT create/decode + RBAC FastAPI deps
│   ├── exceptions.py          # Domain exceptions + FastAPI handlers
│   └── logging.py             # Structured JSON logging (structlog)
├── agents/
│   ├── base_agent.py          # BaseAgent: timeout, retry, PII, logging
│   ├── research_agent.py      # Legal Research — RAG retrieval + graph
│   ├── reasoning_agent.py     # IRAC Reasoning — Claude Sonnet 4.6
│   ├── verification_agent.py  # Citation Verification — DB + LLM check
│   ├── document_agent.py      # Document Analysis — GPT-4o
│   ├── evidence_agent.py      # Evidence Analyzer — multimodal
│   └── risk_strategy_agent.py # Risk & Strategy — win probability
├── api/
│   ├── schemas.py             # Pydantic v2 request/response models
│   ├── dependencies.py        # FastAPI DI container
│   ├── legal.py               # /api/v1/legal/* routes
│   ├── documents.py           # /api/v1/documents/*
│   ├── evidence.py            # /api/v1/evidence/*
│   ├── memory.py              # /api/v1/memory/*
│   ├── feedback.py            # /api/v1/feedback/*
│   └── admin.py               # /api/v1/admin/* (admin role only)
├── orchestrator/
│   ├── workflow_manager.py    # Central orchestration engine
│   ├── query_classifier.py    # Rule-based query type detection
│   └── agent_selector.py      # Dynamic agent plan selection
├── rag/
│   ├── embedder.py            # Embedding + Redis cache
│   ├── retriever.py           # Hybrid search (pgvector + BM25)
│   ├── graph_expander.py      # Case law graph traversal
│   └── reranker.py            # Cross-encoder reranking
├── memory/
│   └── case_memory.py         # 3-tier memory (Redis → Supabase → local)
└── services/
    ├── llm_service.py         # Anthropic + OpenAI abstraction + retry
    ├── pii_service.py         # PII detection + redaction (Thai/EN)
    └── audit_service.py       # Audit trail + expert review queue
```

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/api/v1/legal/query` | Optional | Full IRAC response |
| `POST` | `/api/v1/legal/query/stream` | Optional | SSE streaming |
| `POST` | `/api/v1/legal/draft` | Optional | Draft legal document |
| `POST` | `/api/v1/legal/citations/verify` | Optional | Verify citations |
| `GET`  | `/api/v1/legal/graph/{case_no}` | Optional | Precedent graph |
| `POST` | `/api/v1/documents/analyze` | Optional | Analyse PDF/DOCX |
| `POST` | `/api/v1/evidence/analyze` | Optional | Analyse evidence files |
| `GET`  | `/api/v1/memory/case/{case_id}` | Required | Case memory |
| `GET`  | `/api/v1/memory/case/{case_id}/timeline` | Required | IRAC timeline |
| `POST` | `/api/v1/feedback/` | Optional | Submit feedback |
| `POST` | `/api/v1/admin/ingest` | Admin only | Ingest knowledge |
| `GET`  | `/api/v1/admin/audit-log` | Admin only | Audit trail |
| `GET`  | `/api/v1/admin/expert-queue` | Admin only | Review queue |
| `GET`  | `/health` | None | Health check |

---

## Agent Plans — Dynamic Selection

| Query Type | Research | IRAC | Verify | Document | Evidence | Risk |
|------------|----------|------|--------|----------|----------|------|
| `legal_question` | ✅ | ✅ | ✅ | — | — | — |
| `document_review` | ✅ | ✅ | ✅ | ✅ | — | — |
| `case_strategy` | ✅ | ✅ | ✅ | — | — | ✅ |
| `evidence_analysis` | ✅ | ✅ | ✅ | — | ✅ | — |
| `draft_document` | ✅ | ✅ | ✅ | ✅ | — | — |

---

## Environment Variables

See `.env.example` for the full list. Minimum required for dev:

```bash
ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY — at least one required
REDIS_URL=redis://localhost:6379/0
```

For production, additionally required:
```bash
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=service-role-key
JWT_SECRET=32-char-random-string
APP_ENV=production
```

---

## Security Architecture

| Layer | Mechanism |
|-------|-----------|
| Authentication | JWT Bearer tokens (HS256, 1hr TTL) |
| Authorization | RBAC: `admin` / `lawyer` / `client` / `auditor` |
| Tenant isolation | Supabase Row Level Security (RLS) on all tables |
| PII protection | Redaction applied BEFORE any LLM API call |
| Data in transit | TLS 1.3 (enforced at Nginx/LB layer) |
| Data at rest | AES-256 via Supabase Storage |
| LLM data policy | Anthropic API (no training) + OpenAI Zero Data Retention |
| Audit trail | Every query hashed + logged to `audit_log` table |

---

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=backend --cov-report=term-missing

# Specific module
pytest tests/test_orchestrator/
pytest tests/test_agents/test_reasoning_agent.py -v
```

---

## Development Roadmap

| Phase | Status | Focus |
|-------|--------|-------|
| 0 — Foundation | ✅ Complete | Schema, auth, DB setup, ingest pipeline |
| 1 — Core RAG + IRAC | ✅ Complete | Full agent pipeline, stub → real LLM |
| 2 — Full Multi-Agent | ✅ Complete | Document, Evidence, Risk agents |
| 3 — Case Memory | ✅ Complete | 3-tier memory system |
| 4 — Governance | ✅ Complete | Audit, PII, expert queue, RBAC |
| 5 — Audio + Scale | 🔲 Next | Whisper, performance tuning, Lao language |
| 6 — Intelligence | 🔲 Future | Fine-tuning, court-level pattern analysis |

---

## Making it Production-Real: Next Steps

1. **Connect Supabase** — Run the SQL schema from the blueprint (`laws`, `cases`, `case_citations`, `case_memory`, `audit_log` tables + RLS policies + pgvector indexes)
2. **Ingest legal data** — Use `POST /api/v1/admin/ingest` with Thai/Lao statute and case law documents
3. **Replace stub Reranker** — Integrate Cohere Rerank API or `bge-reranker-large` for Thai
4. **Add Whisper service** — For audio evidence transcription (AWS Transcribe or self-hosted)
5. **Wire real embeddings** — `text-embedding-3-large` for EN, `multilingual-e5-large` for TH/LA
6. **Set up monitoring** — Sentry for errors, Prometheus + Grafana for agent latency and LLM cost
7. **Deploy** — Docker Compose → Railway/Render for staging, AWS ECS for production
