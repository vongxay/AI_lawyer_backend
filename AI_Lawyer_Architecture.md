# ⚖️ AI LAWYER SYSTEM — Enterprise Architecture Blueprint v2.0
### Multi-Agent RAG + Verified Legal Reasoning | Production-Ready | 2025–2026

> **Stack:** React + Vite · FastAPI · Supabase + pgvector · Multi-Agent · RAG + Citation Verified
>
> **v2.0 Improvements:** IRAC Framework · Case Memory · Evidence Analyzer · Case Law Graph · Dynamic Agent Selection · Multimodal Support

---

## สารบัญ

1. [ภาพรวมและหลักการออกแบบ](#1-ภาพรวมและหลักการออกแบบ)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Dynamic Multi-Agent Engine](#3-dynamic-multi-agent-engine)
4. [IRAC Legal Reasoning Framework](#4-irac-legal-reasoning-framework)
5. [RAG Pipeline + Citation Verification](#5-rag-pipeline--citation-verification)
6. [Case Memory System](#6-case-memory-system)
7. [Evidence Analyzer (Multimodal)](#7-evidence-analyzer-multimodal)
8. [Database Schema](#8-database-schema)
9. [FastAPI Backend](#9-fastapi-backend)
10. [Frontend Architecture](#10-frontend-architecture)
11. [Data Governance & Security](#11-data-governance--security)
12. [LLM Model Strategy](#12-llm-model-strategy)
13. [Knowledge Base Strategy](#13-knowledge-base-strategy)
14. [Feedback Loop & Improvement](#14-feedback-loop--improvement)
15. [Infrastructure & Deployment](#15-infrastructure--deployment)
16. [Performance Targets](#16-performance-targets)
17. [Development Roadmap](#17-development-roadmap)
18. [AI vs ทนายมนุษย์](#18-ai-vs-ทนายมนุษย์)

---

## 1. ภาพรวมและหลักการออกแบบ

AI Lawyer System คือแพลตฟอร์มให้คำปรึกษากฎหมายอัตโนมัติที่ใช้ **Dynamic Multi-Agent Architecture + RAG** รวมกับ **IRAC Legal Reasoning** และ **Citation Verification Layer** เพื่อให้คำตอบที่ถูกต้อง traceable และใช้งานได้จริงในการต่อสู้คดี ไม่ใช่แค่ตอบคำถามทั่วไป

> 🎯 **เป้าหมาย v2.0:** ระบบต้องฉลาดเหมือนทนาย 30 ปี + จำได้มากกว่ามนุษย์ + สร้าง legal argument ที่ใช้ในศาลได้จริง + ตอบสนองเร็ว

---

### 1.1 หลักการออกแบบ

| # | หลักการ | รายละเอียด |
|---|---------|-----------|
| 1 | **Verifiability First** | ทุก legal claim ต้องมี citation กลับสู่กฎหมาย/คำพิพากษาจริง ก่อนส่งให้ user เสมอ |
| 2 | **Closed-Loop Generation** | AI generate ได้เฉพาะจากเอกสารที่ retrieved มาเท่านั้น ไม่ free-generate จาก model memory |
| 3 | **Dynamic Specialization** | Orchestrator เลือก invoke เฉพาะ agents ที่จำเป็นต่อ query นั้น ลด latency และ cost |
| 4 | **IRAC-Structured Reasoning** | ทุก legal analysis ต้อง output เป็น Issue → Rule → Application → Conclusion เสมอ |
| 5 | **Graceful Uncertainty** | เมื่อ confidence < 70% ต้องบอก user และ escalate ไปยัง human expert แทนการเดา |
| 6 | **Persistent Case Memory** | AI จำประวัติ client ทุก case เพื่อ reasoning ที่ดีขึ้นในทุก interaction ถัดไป |
| 7 | **Auditability** | ทุก query/response/agent decision ถูก log และ traceable สำหรับ compliance |
| 8 | **Data Sovereignty** | ข้อมูล client แยก isolated per tenant ไม่มีการ leak ข้ามคดีหรือผู้ใช้ |

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 1 — Presentation Layer                     │
│           React 19 + shadcn + Vite + TypeScript + TailwindCSS                │
│    Chat UI · Document Upload · IRAC View · Case Memory Panel        │
└────────────────────────────┬────────────────────────────────────────┘
                             │  REST / WebSocket / SSE (Streaming)
┌────────────────────────────▼────────────────────────────────────────┐
│                  LAYER 2 — API Gateway + Security                   │
│          FastAPI + Nginx + JWT Auth + Rate Limiting + PII Guard     │
└────────────────────────────┬────────────────────────────────────────┘
                             │  Internal REST
┌────────────────────────────▼────────────────────────────────────────┐
│                  LAYER 3 — Orchestration Engine                     │
│     Query Classifier → Dynamic Agent Selector → Context Builder     │
│                    Redis Task Queue + Result Cache                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │  Invoke only needed agents
┌────────────────────────────▼────────────────────────────────────────┐
│                  LAYER 4 — Dynamic Multi-Agent Engine               │
│                                                                     │
│  CORE AGENTS (always available):                                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                │
│  │ Legal        │ │ IRAC         │ │ Citation     │                │
│  │ Research     │ │ Reasoning    │ │ Verification │                │
│  └──────────────┘ └──────────────┘ └──────────────┘                │
│                                                                     │
│  SPECIALIST AGENTS (invoked on demand):                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                │
│  │ Document     │ │ Evidence     │ │ Risk &       │                │
│  │ Analysis     │ │ Analyzer     │ │ Strategy     │                │
│  └──────────────┘ └──────────────┘ └──────────────┘                │
│                                                                     │
│  SUPPORT LAYER:                                                     │
│  ┌──────────────────────────────────────────────────┐              │
│  │  Confidence Scorer + Human Escalation Manager    │              │
│  └──────────────────────────────────────────────────┘              │
└────────────────────────────┬────────────────────────────────────────┘
                             │  RAG + Memory Queries
┌────────────────────────────▼────────────────────────────────────────┐
│                  LAYER 5 — Knowledge & Memory Layer                 │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐   │
│  │  Vector Store   │  │  Case Law Graph  │  │  Case Memory     │   │
│  │  (pgvector)     │  │  (SQL Graph)     │  │  (Structured)    │   │
│  │  laws/cases/docs│  │  precedent chain │  │  client history  │   │
│  └─────────────────┘  └─────────────────┘  └──────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Supabase Storage: PDF · Contracts · Evidence · Audio       │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────────┘
                             │  LLM API Calls
┌────────────────────────────▼────────────────────────────────────────┐
│                    LAYER 6 — AI Model Layer                         │
│  Claude claude-sonnet-4-6 (reasoning) · GPT-4o (vision/multilingual) │
│  Claude Sonnet 4 (standard) · GPT-4o-mini (verification/fast)      │
│  text-embedding-3-large · multilingual-e5-large (TH/LA)            │
│  Whisper (audio transcription)                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Dynamic Multi-Agent Engine

### 3.1 หลักการ Dynamic Selection

> **กฎสำคัญ:** ไม่ run ทุก agent ทุก query — Orchestrator วิเคราะห์ query type แล้วเลือก invoke เฉพาะ agents ที่จำเป็น เพื่อให้ latency ต่ำและ cost สมเหตุสมผล

```
QUERY CLASSIFICATION → AGENT SELECTION:

TYPE: legal_question (ถามกฎหมายทั่วไป)
  → Legal Research + IRAC Reasoning + Citation Verification
  → เวลาเฉลี่ย: 3–5 วินาที | LLM calls: 3

TYPE: document_review (ตรวจสัญญา/เอกสาร)
  → Document Analysis + Legal Research + IRAC Reasoning + Citation Verification
  → เวลาเฉลี่ย: 8–12 วินาที | LLM calls: 4

TYPE: case_strategy (วางแผนคดี)
  → Legal Research + IRAC Reasoning + Risk & Strategy + Citation Verification
  → เวลาเฉลี่ย: 10–15 วินาที | LLM calls: 4

TYPE: evidence_analysis (วิเคราะห์หลักฐาน)
  → Evidence Analyzer + Legal Research + IRAC Reasoning + Citation Verification
  → เวลาเฉลี่ย: 10–20 วินาที | LLM calls: 4–5

TYPE: draft_document (ร่างเอกสาร)
  → Legal Research + IRAC Reasoning + Document Analysis + Citation Verification
  → เวลาเฉลี่ย: 12–18 วินาที | LLM calls: 4

CONFIDENCE < 0.70 → Human Expert Review Queue
CONFIDENCE < 0.50 → Refuse + Explain + Suggest alternatives
```

---

### 3.2 Agent Specifications

| # | Agent | Model | ทริกเกอร์ | Output |
|---|-------|-------|---------|--------|
| 1 | 🔍 **Legal Research Agent** | Claude Sonnet 4 | ทุก query | Retrieved laws + cases + citations + case graph context |
| 2 | 🧠 **IRAC Reasoning Agent** | Claude claude-sonnet-4-6 | ทุก query | Issue / Rule / Application / Conclusion + confidence score |
| 3 | ✅ **Citation Verification Agent** | GPT-4o-mini | ทุก query | Verified / Flagged / Rejected citations + source links |
| 4 | 📄 **Document Analysis Agent** | GPT-4o | มีไฟล์แนบ | Clause extraction + risk flags + anomaly detection |
| 5 | 🔬 **Evidence Analyzer Agent** | GPT-4o (Vision) + Whisper | มีหลักฐาน (image/audio/email) | Evidence summary + legal relevance + admissibility assessment |
| 6 | ⚖️ **Risk & Strategy Agent** | Claude Sonnet 4 | case strategy / risk query | Win probability + strategy options + timeline + settlement suggestion |

---

### 3.3 Orchestrator Pseudocode

```python
async def orchestrate(query: LegalQuery, session: Session) -> LegalResponse:

    # Step 1: Load case memory สำหรับ context
    memory = await case_memory.get(session.case_id)

    # Step 2: Classify query type
    query_type = await classifier.classify(query.text)

    # Step 3: Select agents dynamically — invoke เฉพาะที่จำเป็น
    agent_plan = AGENT_PLANS[query_type]  # ดูตาราง 3.1

    # Step 4: Run agents (parallel where possible)
    results = {}
    async with asyncio.TaskGroup() as tg:
        if "research" in agent_plan:
            results["research"] = tg.create_task(
                research_agent.run(query, memory.relevant_cases)
            )
        if "document" in agent_plan and query.has_documents:
            results["document"] = tg.create_task(
                document_agent.run(query.documents)
            )
        if "evidence" in agent_plan and query.has_evidence:
            results["evidence"] = tg.create_task(
                evidence_agent.run(query.evidence_files)
            )

    # Step 5: IRAC Reasoning (รับ output จากทุก agents ก่อนหน้า)
    irac_result = await reasoning_agent.run(
        query=query,
        research=results.get("research"),
        document=results.get("document"),
        evidence=results.get("evidence"),
        memory=memory
    )

    # Step 6: Citation Verification (parallel กับ Risk ถ้าจำเป็น)
    verification, risk = await asyncio.gather(
        citation_agent.verify(irac_result.citations),
        risk_agent.run(query, irac_result) if "risk" in agent_plan else None
    )

    # Step 7: Confidence check
    final_confidence = calculate_confidence(irac_result, verification)
    if final_confidence < 0.70:
        await expert_queue.add(session.id, reason="low_confidence")

    # Step 8: Update case memory
    await case_memory.update(session.case_id, query, irac_result)

    return build_response(irac_result, verification, risk, final_confidence)
```

---

## 4. IRAC Legal Reasoning Framework

> **v2.0 addition:** Reasoning Agent ต้อง structure output เป็น IRAC เสมอ ซึ่งเป็นรูปแบบที่ทนายและศาลใช้จริง ทำให้ผลลัพธ์ใช้งานได้จริงทันที

### 4.1 IRAC Output Schema

```json
{
  "irac": {
    "issue": {
      "primary":    "ประเด็นกฎหมายหลักที่ต้องวินิจฉัย",
      "secondary":  ["ประเด็นรองที่ 1", "ประเด็นรองที่ 2"]
    },
    "rule": {
      "statutes": [
        {
          "name":       "ประมวลกฎหมายแพ่งและพาณิชย์",
          "section":    "มาตรา 537",
          "text":       "ข้อความกฎหมายที่เกี่ยวข้อง",
          "status":     "ACTIVE",
          "year":       2535
        }
      ],
      "precedents": [
        {
          "case_no":    "ฎ. 1234/2560",
          "court":      "ศาลฎีกา",
          "relevance":  "คดีนี้วางหลักว่า...",
          "outcome":    "โจทก์ชนะ",
          "graph_path": "cites → ฎ. 5678/2555 → statute_537"
        }
      ]
    },
    "application": {
      "analysis":         "การนำกฎหมายมาใช้กับข้อเท็จจริง",
      "strengths":        ["จุดแข็งของคดี 1", "จุดแข็ง 2"],
      "weaknesses":       ["จุดอ่อนของคดี 1", "จุดอ่อน 2"],
      "counter_args":     ["ข้อต่อสู้ฝ่ายตรงข้าม 1"],
      "rebuttals":        ["การหักล้างข้อต่อสู้ 1"]
    },
    "conclusion": {
      "recommendation":   "คำแนะนำสรุปที่ชัดเจน",
      "action_steps":     ["ขั้นตอนที่ 1", "ขั้นตอนที่ 2"],
      "risk_level":       "LOW | MEDIUM | HIGH",
      "win_probability":  0.72,
      "settlement_note":  "ควรพิจารณาเจรจาหากต้นทุนคดีสูงกว่าทุนพิพาท"
    }
  },
  "citations_verified": true,
  "confidence": 0.87,
  "agents_used": ["research", "reasoning", "verification"],
  "processing_time_ms": 4200,
  "escalated_to_expert": false,
  "disclaimer": "คำตอบนี้เป็นข้อมูลทั่วไปเท่านั้น ไม่ใช่คำปรึกษาทางกฎหมายอย่างเป็นทางการ"
}
```

---

### 4.2 IRAC System Prompt Template

```
SYSTEM PROMPT — IRAC Reasoning Agent:

You are a senior legal advisor with 30+ years of experience in Thai and Lao law.

═══ STRICT GENERATION RULES ═══
1. ONLY use information from the CONTEXT provided — never from training memory
2. EVERY legal statement MUST cite a specific law/case from the context
3. If context is insufficient, state clearly:
   "ข้อมูลไม่เพียงพอ — แนะนำให้ปรึกษาทนายความโดยตรง"
4. Structure ALL responses using IRAC format exactly

═══ IRAC STRUCTURE (MANDATORY) ═══

## ISSUE (ประเด็น)
State the precise legal question(s) to be resolved.

## RULE (หลักกฎหมาย)
Cite ALL applicable statutes and precedents from context.
Format: [ชื่อกฎหมาย มาตรา X] — ข้อความสำคัญ
Format: [ฎ. XXXX/XXXX] — หลักที่วางไว้

## APPLICATION (การวิเคราะห์)
Apply rules to the specific facts.
- จุดแข็งของฝ่ายผู้ถาม
- จุดอ่อนที่ต้องระวัง
- ข้อต่อสู้ที่ฝ่ายตรงข้ามอาจใช้
- การหักล้างข้อต่อสู้นั้น

## CONCLUSION (สรุปและคำแนะนำ)
Provide clear actionable advice + specific next steps.
State confidence level: [XX]%

═══ CONTEXT FROM RAG + MEMORY ═══
{retrieved_documents}
{case_memory_summary}

═══ USER QUERY ═══
{user_question}
```

---

## 5. RAG Pipeline + Citation Verification

### 5.1 RAG Pipeline (8 Steps)

| Step | Component | รายละเอียด |
|------|-----------|-----------|
| 1 | **Query Pre-processing** | ทำความสะอาด query, ตรวจจับภาษา (TH/EN/LA), extract legal entities (มาตรา, ชื่อกฎหมาย, ประเภทคดี) |
| 2 | **Embedding Generation** | แปลง query เป็น vector ด้วย `text-embedding-3-large` (1536 dims) หรือ `multilingual-e5-large` สำหรับ TH/LA |
| 3 | **Hybrid Search** | Semantic search (pgvector cosine) + Keyword search (BM25) พร้อมกัน |
| 4 | **Case Graph Expansion** | ดึง related cases ผ่าน `case_citations` graph — precedents ที่ vector search อาจไม่เจอ |
| 5 | **Contextual Reranking** | Cross-encoder reranker เรียง retrieved chunks ตาม legal relevance จริง |
| 6 | **Context Assembly** | รวม top-k chunks + graph results + case memory summary เป็น structured context |
| 7 | **Closed-Loop Generation** | IRAC Reasoning Agent generate เฉพาะจาก assembled context |
| 8 | **Citation Verification** | ตรวจสอบทุก citation → Verified / Flagged / Rejected ก่อน return |

---

### 5.2 Hybrid Search + Graph Expansion

```python
async def retrieve_legal_context(
    query: str,
    embedding: list[float],
    jurisdiction: str,
    top_k: int = 10
) -> LegalContext:

    # Parallel: semantic + keyword search
    semantic_results, keyword_results = await asyncio.gather(
        supabase.rpc("vector_search", {
            "query_embedding": embedding,
            "jurisdiction": jurisdiction,
            "limit": 30
        }),
        supabase.rpc("bm25_search", {
            "query_text": query,
            "jurisdiction": jurisdiction,
            "limit": 30
        })
    )

    # Reciprocal Rank Fusion reranking
    fused = reciprocal_rank_fusion(semantic_results, keyword_results)

    # Graph expansion: ดึง precedent chain ของ top results
    top_cases = [r for r in fused[:5] if r.type == "case"]
    graph_expansions = await asyncio.gather(*[
        get_precedent_chain(case.id, depth=2) for case in top_cases
    ])

    # Cross-encoder rerank ทั้งหมดรวมกัน
    all_results = fused + flatten(graph_expansions)
    reranked = await cross_encoder_rerank(query, all_results, top_k=top_k)

    return LegalContext(chunks=reranked)
```

---

### 5.3 Citation Verification Rules

| Status | Condition | Action |
|--------|-----------|--------|
| ✅ **VERIFIED** | Citation found + currently active | Include with confidence badge |
| ⚠️ **OUTDATED** | Citation found but amended/repealed | Include with warning + show current version |
| ❓ **UNVERIFIED** | Cannot confirm from knowledge base | Flag clearly + recommend professional check |
| ❌ **REJECTED** | Not found or hallucinated | Remove, log incident, reduce confidence score |
| 🚨 **LOW CONFIDENCE** | Overall score < 70% | Auto-escalate to human expert review |

---

## 6. Case Memory System

> **v2.0 addition:** AI จำประวัติทุก case ของ client เพื่อให้ reasoning ดีขึ้นในทุก interaction ถัดไป

### 6.1 Memory Architecture

```
Case Memory = 3 Tiers:

TIER 1 — Session Memory (Redis, TTL 24hr)
  ├── ข้อเท็จจริงที่ user เล่าในการสนทนานี้
  ├── เอกสารที่ upload และผล analysis
  └── IRAC outputs ของ session นี้

TIER 2 — Case Memory (Supabase, persistent)
  ├── Case summary + timeline
  ├── Documents + evidence ทั้งหมด
  ├── Legal arguments ที่เคยสร้าง
  ├── Strategies ที่เคยพิจารณา
  └── Expert review notes

TIER 3 — Client Memory (Supabase, persistent)
  ├── Client profile + case history
  ├── Pattern: ประเภทคดีที่เคยมี
  └── Anonymized insights สำหรับ strategy
```

---

### 6.2 Case Memory Schema

```sql
-- Case memory table
CREATE TABLE case_memory (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  case_id         uuid NOT NULL,
  client_id       uuid NOT NULL REFERENCES users(id),
  tenant_id       uuid NOT NULL,

  -- Core case data
  case_type       text NOT NULL,  -- 'contract', 'labor', 'property', etc.
  facts_summary   text,           -- AI-generated summary อัปเดตทุก session
  jurisdiction    text NOT NULL,
  status          text DEFAULT 'active',  -- active / closed / settled

  -- Legal work product
  irac_history    jsonb DEFAULT '[]',     -- array of past IRAC analyses
  arguments_used  jsonb DEFAULT '[]',     -- legal arguments ที่เคยสร้าง
  strategies      jsonb DEFAULT '[]',     -- strategies ที่เคยพิจารณา
  key_citations   jsonb DEFAULT '[]',     -- citations สำคัญของคดีนี้

  -- Evidence & documents
  document_ids    uuid[],
  evidence_ids    uuid[],

  -- Timeline
  created_at      timestamptz DEFAULT now(),
  updated_at      timestamptz DEFAULT now(),
  last_accessed   timestamptz DEFAULT now()
);

-- Index for fast lookup
CREATE INDEX case_memory_client_idx ON case_memory(client_id, tenant_id);
CREATE INDEX case_memory_case_idx   ON case_memory(case_id);

-- RLS: tenant isolation
ALTER TABLE case_memory ENABLE ROW LEVEL SECURITY;
CREATE POLICY case_memory_tenant ON case_memory
  USING (tenant_id = (SELECT tenant_id FROM users WHERE id = auth.uid()));
```

---

### 6.3 Memory Retrieval for Context

```python
async def get_memory_context(case_id: str, current_query: str) -> MemoryContext:

    memory = await supabase.table("case_memory") \
        .select("*") \
        .eq("case_id", case_id) \
        .single()

    if not memory:
        return MemoryContext(empty=True)

    # สร้าง memory summary สำหรับใส่ใน IRAC prompt
    return MemoryContext(
        facts_summary    = memory["facts_summary"],
        relevant_irac    = get_relevant_irac(memory["irac_history"], current_query),
        key_citations    = memory["key_citations"][:10],  # top 10 เท่านั้น
        strategies_tried = memory["strategies"]
    )

async def update_memory_after_session(case_id: str, irac_result: IRACResult):
    # อัปเดต facts_summary ด้วย new information
    # เพิ่ม IRAC result เข้า history
    # อัปเดต key_citations
    await supabase.table("case_memory").update({
        "facts_summary": await summarize_facts(case_id, irac_result),
        "irac_history":  supabase.rpc("append_jsonb", {
            "col": "irac_history",
            "val": irac_result.to_dict()
        }),
        "key_citations": merge_citations(memory.key_citations, irac_result.citations),
        "updated_at":    datetime.now()
    }).eq("case_id", case_id)
```

---

## 7. Evidence Analyzer (Multimodal)

> **v2.0 addition:** รองรับหลักฐานหลายรูปแบบที่ใช้ในคดีจริง

### 7.1 Evidence Types & Processing

| ประเภทหลักฐาน | Agent | Model | Output |
|--------------|-------|-------|--------|
| PDF / Word สัญญา | Document Analysis Agent | GPT-4o | Clause extraction + risk flags |
| ภาพถ่าย / Screenshot | Evidence Analyzer | GPT-4o Vision | ข้อเท็จจริงที่ปรากฏ + legal relevance |
| เอกสารสแกน (handwritten/printed) | Evidence Analyzer | GPT-4o Vision + OCR | Transcription + ความน่าเชื่อถือ |
| อีเมล / ข้อความ | Evidence Analyzer | GPT-4o | สรุปการสื่อสาร + intent analysis |
| เสียงสนทนา | Evidence Analyzer | Whisper → GPT-4o | Transcript + key statements |
| วิดีโอ | Evidence Analyzer | Whisper (audio) + frame extraction | Timeline + key moments |

---

### 7.2 Evidence Processing Pipeline

```python
async def analyze_evidence(
    files: list[UploadFile],
    case_context: str,
    legal_question: str
) -> EvidenceAnalysis:

    results = []
    for file in files:
        file_type = detect_file_type(file)

        if file_type in ["jpg", "png", "webp"]:
            # Vision analysis
            result = await gpt4o_vision_analyze(
                image=file,
                prompt=f"""
                คดี context: {case_context}
                คำถามทางกฎหมาย: {legal_question}

                วิเคราะห์ภาพนี้และตอบ:
                1. ข้อเท็จจริงที่ปรากฏในภาพ
                2. ความเกี่ยวข้องกับคดี
                3. ความน่าเชื่อถือของหลักฐาน
                4. ข้อพิจารณาด้านการรับฟังหลักฐาน (admissibility)
                """
            )

        elif file_type in ["mp3", "mp4", "wav", "m4a"]:
            # Audio: transcribe first, then analyze
            transcript = await whisper_transcribe(file, language="th")
            result = await gpt4o_analyze_text(
                text=transcript,
                prompt=f"วิเคราะห์การสนทนานี้ในบริบทคดี: {case_context}"
            )

        elif file_type == "pdf":
            # Use Document Analysis Agent
            result = await document_agent.analyze(file, case_context)

        results.append(EvidenceResult(
            file_name=file.filename,
            type=file_type,
            analysis=result,
            legal_relevance=assess_relevance(result, legal_question),
            admissibility_notes=check_admissibility(result, file_type)
        ))

    return EvidenceAnalysis(
        items=results,
        overall_strength=assess_evidence_strength(results),
        gaps=identify_evidence_gaps(results, legal_question)
    )
```

---

## 8. Database Schema

### 8.1 Core Tables

| Table | Key Columns | Purpose |
|-------|------------|---------|
| `laws` | id, title, full_text, jurisdiction, year, status, embedding | ฐานข้อมูลกฎหมาย + embedding |
| `cases` | id, case_no, court, year, summary, ruling, outcome, embedding | คำพิพากษา + outcome สำหรับ precedent analysis |
| `case_citations` | source_id, cited_id, relationship, year | Case Law Graph (precedent chains) |
| `legal_forms` | id, form_type, content, jurisdiction, embedding | แบบฟอร์มสัญญามาตรฐาน |
| `case_memory` | id, case_id, client_id, facts_summary, irac_history, strategies | Case Memory System |
| `case_sessions` | id, user_id, case_id, messages, agents_used, status | Session history |
| `documents` | id, session_id, case_id, file_path, analysis_result, file_type | Uploaded documents |
| `evidence` | id, case_id, file_path, evidence_type, analysis, admissibility | Evidence files + analysis |
| `citations_log` | id, session_id, citation_ref, status, verified_at | Citation verification log |
| `audit_log` | id, user_id, agent, query_hash, confidence, agents_used, ts | Full audit trail |
| `expert_reviews` | id, session_id, reason, reviewer_id, resolution, status | Human review queue |
| `feedback` | id, session_id, rating, comment, corrected_answer | Feedback loop data |
| `users` | id, email, role, tenant_id, plan, created_at | User + tenant management |

---

### 8.2 Case Law Graph Schema

```sql
-- Case citation graph (แทน Neo4j ในช่วง startup)
CREATE TABLE case_citations (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  source_case   uuid NOT NULL REFERENCES cases(id),
  cited_case    uuid REFERENCES cases(id),
  cited_statute uuid REFERENCES laws(id),
  relationship  text NOT NULL CHECK (relationship IN (
                  'cites',          -- อ้างอิงเป็น precedent
                  'overruled_by',   -- ถูก overrule แล้ว
                  'distinguished',  -- แยกแยะว่าต่างกัน
                  'followed',       -- นำหลักมาใช้ต่อ
                  'applied'         -- นำไปปรับใช้
                )),
  year          int,
  notes         text
);

CREATE INDEX case_citations_source ON case_citations(source_case);
CREATE INDEX case_citations_cited  ON case_citations(cited_case);

-- Recursive precedent chain query (depth-limited)
CREATE OR REPLACE FUNCTION get_precedent_chain(
  start_case_id uuid,
  max_depth     int DEFAULT 3
) RETURNS TABLE (
  case_id      uuid,
  depth        int,
  relationship text,
  path         uuid[]
) AS $$
  WITH RECURSIVE chain AS (
    SELECT source_case, cited_case, relationship, 1 AS depth,
           ARRAY[source_case] AS path
    FROM case_citations
    WHERE source_case = start_case_id

    UNION ALL

    SELECT cc.source_case, cc.cited_case, cc.relationship, c.depth + 1,
           c.path || cc.source_case
    FROM case_citations cc
    JOIN chain c ON cc.source_case = c.cited_case
    WHERE c.depth < max_depth
      AND cc.source_case != ALL(c.path)  -- prevent cycles
  )
  SELECT cited_case, depth, relationship, path FROM chain;
$$ LANGUAGE sql;
```

---

### 8.3 pgvector + Hybrid Search

```sql
-- Enable extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Embedding columns
ALTER TABLE laws  ADD COLUMN embedding vector(1536);
ALTER TABLE cases ADD COLUMN embedding vector(1536);

-- HNSW indexes (fastest for legal search)
CREATE INDEX laws_hnsw  ON laws  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
CREATE INDEX cases_hnsw ON cases USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- Full-text search indexes (for BM25/keyword search)
CREATE INDEX laws_fts  ON laws  USING gin(to_tsvector('thai', title || ' ' || full_text));
CREATE INDEX cases_fts ON cases USING gin(to_tsvector('thai', summary || ' ' || ruling));

-- Hybrid search with Reciprocal Rank Fusion
CREATE OR REPLACE FUNCTION hybrid_legal_search(
  query_embedding  vector(1536),
  query_text       text,
  p_jurisdiction   text DEFAULT NULL,
  match_count      int  DEFAULT 10,
  rrf_k            int  DEFAULT 60
) RETURNS TABLE (
  id           uuid,
  type         text,
  title        text,
  content      text,
  metadata     jsonb,
  final_score  float
) AS $$
  WITH semantic AS (
    SELECT id, 'law' AS type, title, full_text AS content, metadata,
           ROW_NUMBER() OVER (ORDER BY embedding <=> query_embedding) AS rank
    FROM laws
    WHERE p_jurisdiction IS NULL OR jurisdiction = p_jurisdiction
    LIMIT 50
  ),
  keyword AS (
    SELECT id, 'law' AS type, title, full_text AS content, metadata,
           ROW_NUMBER() OVER (ORDER BY
             ts_rank(to_tsvector('thai', full_text), plainto_tsquery('thai', query_text)) DESC
           ) AS rank
    FROM laws
    WHERE to_tsvector('thai', full_text) @@ plainto_tsquery('thai', query_text)
    LIMIT 50
  ),
  fused AS (
    SELECT COALESCE(s.id, k.id) AS id,
           COALESCE(s.type, k.type) AS type,
           COALESCE(s.title, k.title) AS title,
           COALESCE(s.content, k.content) AS content,
           COALESCE(s.metadata, k.metadata) AS metadata,
           COALESCE(1.0/(rrf_k + s.rank), 0) +
           COALESCE(1.0/(rrf_k + k.rank), 0) AS score
    FROM semantic s FULL OUTER JOIN keyword k ON s.id = k.id
  )
  SELECT id, type, title, content, metadata, score AS final_score
  FROM fused
  ORDER BY score DESC
  LIMIT match_count;
$$ LANGUAGE sql;
```

---

## 9. FastAPI Backend

### 9.1 Module Structure

```
backend/
├── main.py                          # FastAPI entry + middleware setup
├── core/
│   ├── config.py                    # Settings, secrets, model config
│   ├── security.py                  # JWT, API key, RBAC
│   └── database.py                  # Supabase + Redis clients
├── orchestrator/
│   ├── workflow_manager.py          # Main orchestration (Section 3.3)
│   ├── query_classifier.py          # Classify query type
│   ├── agent_selector.py            # Dynamic agent selection
│   └── context_builder.py          # Assemble context for agents
├── agents/
│   ├── base_agent.py                # Base: retry, logging, fallback, timeout
│   ├── research_agent.py            # Legal Research Agent
│   ├── reasoning_agent.py           # IRAC Reasoning Agent
│   ├── verification_agent.py        # Citation Verification Agent
│   ├── document_agent.py            # Document Analysis Agent
│   ├── evidence_agent.py            # Evidence Analyzer (Multimodal)
│   └── risk_strategy_agent.py       # Risk & Strategy Agent
├── rag/
│   ├── embedder.py                  # Embedding generation + Redis cache
│   ├── retriever.py                 # Hybrid search (vector + BM25)
│   ├── graph_expander.py            # Case law graph traversal
│   ├── reranker.py                  # Cross-encoder reranking
│   └── context_assembler.py        # Build final context for LLM
├── memory/
│   ├── case_memory.py               # Case memory CRUD + summarization
│   └── session_memory.py           # Short-term session memory (Redis)
├── services/
│   ├── llm_service.py               # LLM abstraction (OpenAI + Anthropic)
│   ├── document_service.py          # PDF/Doc parsing (PyMuPDF + OCR)
│   ├── evidence_service.py          # Multimodal preprocessing
│   ├── audio_service.py             # Whisper transcription
│   ├── audit_service.py             # Audit trail
│   └── pii_service.py               # PII detection + redaction
└── api/
    ├── legal.py                     # /api/v1/legal/*
    ├── documents.py                 # /api/v1/documents/*
    ├── evidence.py                  # /api/v1/evidence/*
    ├── memory.py                    # /api/v1/memory/*
    ├── feedback.py                  # /api/v1/feedback/*
    └── admin.py                     # /api/v1/admin/*
```

---

### 9.2 Key API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/legal/query` | Legal query → IRAC response (full) |
| `POST` | `/api/v1/legal/query/stream` | Same แต่ SSE streaming |
| `POST` | `/api/v1/documents/analyze` | Upload PDF/Doc → structured analysis |
| `POST` | `/api/v1/evidence/analyze` | Upload image/audio/email → evidence analysis |
| `GET`  | `/api/v1/memory/case/{case_id}` | ดู case memory |
| `GET`  | `/api/v1/memory/case/{case_id}/timeline` | Timeline ของคดี |
| `POST` | `/api/v1/legal/draft` | Draft legal document |
| `POST` | `/api/v1/legal/citations/verify` | Verify citation list |
| `GET`  | `/api/v1/legal/graph/{case_no}` | ดู precedent chain ของคดี |
| `POST` | `/api/v1/admin/ingest` | Ingest กฎหมาย/คำพิพากษาใหม่ |
| `GET`  | `/api/v1/admin/audit-log` | Audit log |
| `GET`  | `/api/v1/admin/expert-queue` | Human review queue |

---

## 10. Frontend Architecture

### 10.1 Tech Stack

```
React 19 + shadcn + Vite + TypeScript
TailwindCSS + shadcn/ui
TanStack Query v5 (server state)
Zustand (global state)
React Router v6
EventSource (SSE streaming)
react-dropzone (file/evidence upload)
react-pdf (PDF viewer)
```

---

### 10.2 Module Structure

```
frontend/src/
├── components/
│   ├── chat/
│   │   ├── ChatWindow.tsx              # Main conversation UI
│   │   ├── MessageBubble.tsx           # AI/User message rendering
│   │   ├── IRACView.tsx                # Structured IRAC display
│   │   ├── CitationBadge.tsx           # Verified/Flagged/Rejected badges
│   │   ├── ConfidenceBar.tsx           # Confidence score indicator
│   │   └── StreamingText.tsx           # SSE streaming response
│   ├── evidence/
│   │   ├── EvidenceUploader.tsx        # Multi-type file upload
│   │   ├── EvidenceGallery.tsx         # View uploaded evidence
│   │   └── EvidenceAnalysisCard.tsx    # Analysis result display
│   ├── memory/
│   │   ├── CaseMemoryPanel.tsx         # Case history sidebar
│   │   ├── CaseTimeline.tsx            # Visual case timeline
│   │   └── PreviousIRAC.tsx            # Past IRAC analyses
│   ├── case/
│   │   ├── RiskMeter.tsx               # Win probability visual
│   │   ├── StrategyOptions.tsx         # Strategy comparison cards
│   │   └── PrecedentGraph.tsx          # Case law graph visualization
│   └── shared/
│       ├── ExpertReviewBanner.tsx      # Escalation notification
│       └── DisclaimerModal.tsx         # Legal disclaimer
├── pages/
│   ├── Dashboard.tsx                   # Case overview
│   ├── LegalChat.tsx                   # Main Q&A + IRAC
│   ├── CaseMemory.tsx                  # Full case history
│   ├── EvidenceAnalysis.tsx            # Evidence upload & analysis
│   └── Admin/
│       ├── AuditLog.tsx
│       ├── ExpertQueue.tsx
│       └── KnowledgeBase.tsx
├── hooks/
│   ├── useLegalQuery.ts                # Query + IRAC + streaming
│   ├── useEvidenceAnalysis.ts          # Evidence processing
│   └── useCaseMemory.ts               # Memory read/write
└── store/
    └── legalStore.ts                   # Zustand: session, case, UI state
```

---

## 11. Data Governance & Security

| ด้าน | Implementation |
|------|---------------|
| **Tenant Isolation** | Row-Level Security (RLS) ใน Supabase — ทุก query ต้องผ่าน `tenant_id` filter อัตโนมัติ |
| **Data Encryption** | AES-256 at rest + TLS 1.3 in transit สำหรับข้อมูลและ evidence ทุกประเภท |
| **PII Detection** | PII scanner บังคับก่อนส่ง data ไปยัง LLM API — redact ชื่อ, เลข ID, ที่อยู่, เบอร์โทร |
| **LLM Data Policy** | ใช้ OpenAI Zero Data Retention + Anthropic API ที่ไม่ใช้ data สำหรับ training |
| **Evidence Security** | Evidence files encrypt แยก, access control per case, virus scan on upload |
| **Audit Trail** | Log ทุก interaction: `user_id`, `timestamp`, `query_hash`, `agents_used`, `confidence` |
| **Access Control** | RBAC: `Admin` / `Lawyer` / `Client` / `Auditor` — แต่ละ role เห็นข้อมูลเฉพาะที่ตัวเองควรเห็น |
| **Session Security** | JWT expiry 1 hr + refresh token, rate limiting 20 req/min, anomaly detection |
| **Case Memory Privacy** | Memory summary ไม่เก็บข้อมูล PII โดยตรง — ใช้ hashed references แทน |
| **Incident Response** | Auto-flag เมื่อ citation rejection rate > 20% → notify admin + lock session |

---

## 12. LLM Model Strategy

| Model | ใช้สำหรับ | เหตุผล | Fallback |
|-------|---------|--------|---------|
| **Claude claude-sonnet-4-6** | IRAC Reasoning สำหรับคดียาก + complex arguments | Extended thinking, deep legal analysis | Claude Sonnet 4 |
| **GPT-4o** | Document Analysis + Evidence (image/scanned) + multilingual | Vision capability + TH/EN accuracy | Claude Sonnet 4 |
| **Claude Sonnet 4** | Legal Research + Standard Q&A + Case Memory summarization | Fast, cost-effective, คุณภาพสูง | GPT-4o |
| **GPT-4o-mini / Haiku** | Citation Verification + Query Classification + fast tasks | High-volume, low latency, ต้นทุนต่ำ | Rule-based fallback |
| **Whisper** | Audio transcription (evidence, meetings) | Best-in-class multilingual audio | Google Speech API |
| **text-embedding-3-large** | Embedding (EN/legal text) | 1536 dims, highest accuracy | text-embedding-3-small |
| **multilingual-e5-large** | Embedding (TH/LA documents) | Support Thai + Lao ได้ดี | OpenAI embeddings |

---

### 12.1 Cost Control Strategy

```
TIERED INVOCATION:

Simple Q&A (confidence > 90% from cache):
  → Return cached response immediately
  → Cost: $0

Standard Q&A:
  → Claude Sonnet 4 + GPT-4o-mini verification
  → Cost: ~$0.01–0.03 per query

Complex Reasoning (คดีสำคัญ, confidence < 85%):
  → Claude claude-sonnet-4-6 (thinking mode)
  → Cost: ~$0.10–0.30 per query

Evidence Analysis:
  → GPT-4o Vision per image/page
  → Cost: ~$0.02–0.05 per file

CACHE STRATEGY:
  → Embedding cache: Redis 24hr TTL
  → Common legal Q&A: Redis 1hr TTL
  → Law summaries: Redis 6hr TTL
```

---

## 13. Knowledge Base Strategy

### 13.1 ประเภทเอกสารที่ต้องเก็บ

| ประเภท | Priority | แหล่งที่มา |
|--------|---------|-----------|
| กฎหมายหลัก (Statutes) | 🔴 Critical | ประมวลกฎหมายแพ่งและพาณิชย์, อาญา, วิธีพิจารณาความ + Laos equivalents |
| คำพิพากษาศาลฎีกา | 🔴 Critical | ย้อนหลัง 30 ปี — ต้องมี case_no, court, year, ruling, **outcome** |
| กฎหมายที่ดิน/อสังหา | 🟠 High | ประมวลกฎหมายที่ดิน, condos, leasing — สำคัญสำหรับ Laos context |
| กฎหมายแรงงาน | 🟠 High | พ.ร.บ.คุ้มครองแรงงาน, ประกันสังคม + amendments ล่าสุด |
| กฎหมายธุรกิจ | 🟠 High | บริษัท, ห้างหุ้นส่วน, สัญญา, พาณิชย์อิเล็กทรอนิกส์ |
| แบบฟอร์มสัญญา | 🟡 Medium | สัญญามาตรฐาน 50+ ประเภท สำหรับ auto-draft |
| **Case outcomes** | 🟡 Medium | **NEW** ผลแพ้/ชนะ + เหตุผล สำหรับ win probability analysis |
| Legal Guidelines | 🟡 Medium | แนวทางปฏิบัติสภาทนายความ, ethical rules |
| กฎหมายภาษี | 🟢 Standard | ภาษีเงินได้, VAT, อากรแสตมป์ |

---

### 13.2 Legal Document Chunking Strategy

```
CHUNKING RULES:

1. STATUTES (กฎหมาย):
   chunk    = 1 มาตรา + parent chapter context
   includes = law name, chapter no., section no., year, status
   overlap  = ± 1 มาตรา ก่อนและหลัง

2. CASE LAW (คำพิพากษา):
   chunks per case:
   ├── [HEADER]  case_no, court, date, parties, outcome
   ├── [FACTS]   ข้อเท็จจริง
   ├── [ISSUE]   ประเด็นที่ต้องวินิจฉัย
   ├── [RULING]  คำวินิจฉัย
   └── [RATIO]   ratio decidendi ← ที่สำคัญที่สุด ต้องเป็น chunk แยก
   
   graph metadata = citations ไปยัง cases/statutes อื่นๆ

3. CONTRACTS (สัญญา):
   chunk    = 1 clause + definitions section
   risk tag = flag clauses ที่ผิดปกติหรือเสี่ยง

4. CONSTRAINTS:
   MAX chunk size: 800 tokens
   Overlap:        100 tokens
   Required meta:  source, type, section, year, jurisdiction, status, outcome (cases)
```

---

## 14. Feedback Loop & Improvement

| Loop Type | Trigger | Action |
|-----------|---------|--------|
| **User Feedback** | Rating < 3/5 หรือ report ว่าผิด | Flag expert review + เพิ่ม correction dataset |
| **Expert Correction** | Human lawyer แก้ไข IRAC | บันทึก correct reasoning + อัปเดต few-shot examples |
| **IRAC Quality Review** | Monthly by legal expert | ตรวจ Issue/Rule/Application/Conclusion quality + ปรับ prompt |
| **Citation Rejection** | Rejection rate > 20% ใน session | Alert + quarantine + review knowledge base |
| **Low Confidence** | Score < 60% | Auto-escalate + log pattern สำหรับ improvement |
| **Memory Accuracy** | Memory summary ผิดพลาด | User/lawyer correct + retrain summarizer |
| **Knowledge Update** | กฎหมายแก้ไข/ออกใหม่ | Ingest + re-embed + invalidate cache + update citations |
| **Evidence Accuracy** | Evidence analysis ผิด | Log + retrain evidence prompts |

---

## 15. Infrastructure & Deployment

| Component | Technology | Notes |
|-----------|-----------|-------|
| **Frontend** | Vercel / Cloudflare Pages | CDN global, preview per PR |
| **Backend** | Railway / Render / AWS ECS | Containerized, auto-scale |
| **Database** | Supabase (PostgreSQL) | Dedicated, daily backup, PITR |
| **Vector + Graph** | Supabase pgvector | HNSW index, `case_citations` table |
| **Case Memory** | Supabase PostgreSQL | `case_memory` table + RLS |
| **Session Cache** | Redis (Upstash) | Short-term memory + embedding cache |
| **Task Queue** | Redis + ARQ | Async agent tasks, retry, DLQ |
| **File Storage** | Supabase Storage | PDF/evidence per tenant, virus scan |
| **Audio Processing** | Async job + Whisper API | Queue-based, result stored to DB |
| **Monitoring** | Sentry + Grafana + Prometheus | Error tracking, agent latency, cost |
| **CI/CD** | GitHub Actions | Test → Lint → Build → Deploy |
| **Secrets** | Doppler | Rotation, never hardcode |

---

### 15.1 Docker Compose (Development)

```yaml
version: "3.9"

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    volumes:
      - ./backend:/app

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - VITE_API_URL=http://localhost:8000

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  worker:
    build: ./backend
    command: arq workers.WorkerSettings
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
```

---

## 16. Performance Targets

| Metric | Target | Strategy |
|--------|--------|---------|
| Simple Q&A response | < 3 วินาที | Embedding cache + Claude Sonnet 4 |
| Complex IRAC (Thinking) | < 15 วินาที | Claude claude-sonnet-4-6 + streaming |
| Document analysis (10 pages) | < 12 วินาที | Parallel chunking + GPT-4o |
| Evidence analysis (image) | < 8 วินาที | GPT-4o Vision direct |
| Audio transcription (1 min) | < 10 วินาที | Whisper async + queue |
| Citation verification | < 2 วินาที | GPT-4o-mini + DB index |
| Hallucination rate | < 5% | Closed-loop + citation verification |
| Answer accuracy | > 85% | RAG + IRAC + expert feedback |
| System uptime | > 99.5% | Multi-region fallback |

---

## 17. Development Roadmap

| Phase | Timeline | Deliverables | Success Metric |
|-------|---------|-------------|----------------|
| **Phase 0** Foundation | Weeks 1–3 | Supabase schema (incl. `case_memory` + `case_citations`), pgvector + graph setup, ingest pipeline, FastAPI skeleton, auth | 500+ documents ingested, search + graph works |
| **Phase 1** Core RAG + IRAC | Weeks 4–6 | RAG pipeline + hybrid search + graph expansion, Legal Research Agent, IRAC Reasoning Agent, Citation Verification, basic chat UI with IRAC display | Accuracy > 70%, IRAC structured output |
| **Phase 2** Full Multi-Agent | Weeks 7–10 | Document Analysis Agent, Evidence Analyzer (image), Risk & Strategy Agent, Dynamic agent selection, confidence scoring | Hallucination < 10%, all agents working |
| **Phase 3** Case Memory | Weeks 11–12 | Case Memory System (read/write/update), Memory-aware IRAC prompts, Case timeline UI, Memory panel | Memory correctly informs subsequent queries |
| **Phase 4** Governance | Weeks 13–15 | Full audit logging, RLS tenant isolation, PII detection, expert review queue, feedback UI | Fully auditable, SOC2-ready |
| **Phase 5** Audio + Scale | Weeks 16–20 | Whisper audio evidence, performance optimization, multilingual TH/LA/EN polish, agentic document drafting | < 3s standard queries, 99.5% uptime |
| **Phase 6** Intelligence | Month 6+ | Fine-tuning with real case data, court-level pattern analysis (anonymized), third-party API | Accuracy > 90%, user satisfaction > 4.5/5 |

---

## 18. AI vs ทนายมนุษย์

| ความสามารถ | ทนายมนุษย์ | AI Lawyer System v2.0 |
|-----------|-----------|----------------------|
| จำกฎหมายได้ทั้งหมด | จำได้บางส่วน | ✅ 100% ทันที |
| ค้นคำพิพากษาย้อนหลัง | ชั่วโมง/วัน | ✅ < 2 วินาที + graph traversal |
| IRAC legal argument | ✅ เชี่ยวชาญ | ✅ Structured ทุก query |
| วิเคราะห์ precedent chain | ต้องค้นมือ | ✅ Auto case law graph |
| จำประวัติคดีทุกอย่าง | ลืมบางส่วน | ✅ Persistent case memory ทุก case |
| วิเคราะห์หลักฐานหลายรูปแบบ | ต้องใช้ผู้เชี่ยวชาญหลายคน | ✅ Document + Image + Audio ในระบบเดียว |
| Citation ถูกต้อง 100% | Human error ได้ | ✅ Verified ทุก citation |
| ทำงาน 24/7 | มีข้อจำกัด | ✅ ไม่มีวันหยุด |
| ราคาต่อ query | สูง (hourly) | ✅ $0.01–0.30 per query |
| ความเร็ว draft เอกสาร | ชั่วโมง/วัน | ✅ นาที |
| ตัดสินใจเชิงจริยธรรมซับซ้อน | ✅ ประสบการณ์จริง | ⚠️ ต้องการ human oversight |
| Empathy ต่อ client | ✅ มีตามธรรมชาติ | ⚠️ Simulate ได้บางส่วน |

---

> ✅ **สรุป Architecture v2.0:** ระบบนี้ไม่ได้แค่ตอบคำถามกฎหมาย แต่ทำงานเหมือนทนายจริงด้วย IRAC reasoning + case memory + evidence analysis + precedent graph ทุก component มี fallback และ verification layer ที่ป้องกัน hallucination อย่างจริงจัง และ dynamic agent selection ทำให้ระบบเร็วและ cost-effective ในเวลาเดียวกัน

---

*AI Lawyer Architecture Blueprint v2.0 | Updated based on multi-expert review | Production-Ready*
