"""
agents/reasoning_agent.py
==========================
IRAC Reasoning Agent — CORE agent, the heart of the system.

Uses the configured reasoning LLM with a strict IRAC system prompt.
Output is ALWAYS structured as Issue / Rule / Application / Conclusion.

Key constraints enforced:
- Closed-loop: only generate from retrieved context
- Every legal claim must cite a specific law/case
- If context is insufficient → explicit uncertainty + escalation
- Structured JSON output parsed and validated against IRAC schema
"""
from __future__ import annotations

import json
import re
from typing import Any

from agents.base_agent import BaseAgent
from core.config import get_settings
from core.jurisdiction import (
    contains_lao_script,
    contains_thai_script,
    infer_response_language,
    response_language_instruction,
)
from core.logging import get_logger

log = get_logger(__name__)

LAO_LAND = "\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99"
LAO_ARTICLE = "\u0ea1\u0eb2\u0e94\u0e95\u0eb2"
LAO_RIGHT = "\u0eaa\u0eb4\u0e94"
LAO_LAND_USE_RIGHT = "\u0eaa\u0eb4\u0e94\u0e99\u0eb3\u0ec3\u0e8a\u0ec9"
LAO_LAND_USE_RIGHT_ALT = "\u0eaa\u0eb4\u0e94\u0e99\u0ecd\u0eb2\u0ec3\u0e8a\u0ec9"
LAO_LAND_USE_RIGHT_OCR = "\u0eaa\u0eb4\u0e94\u0e99\u0eb2\u0ecd\u0ec3\u0e8a\u0ec9"
LAO_PROTECTION = "\u0e9b\u0ebb\u0e81\u0e9b\u0ec9\u0ead\u0e87"
LAO_GUARD_RIGHT = "\u0eaa\u0eb4\u0e94\u0e9b\u0ebb\u0e81\u0e9b\u0eb1\u0e81\u0eae\u0eb1\u0e81\u0eaa\u0eb2"
LAO_USE_RIGHT = "\u0eaa\u0eb4\u0e94\u0ec3\u0e8a\u0ec9"
LAO_FRUITS_RIGHT = "\u0eaa\u0eb4\u0e94\u0ec4\u0e94\u0ec9\u0eae\u0eb1\u0e9a\u0edd\u0eb2\u0e81\u0e9c\u0ebb\u0e99"
LAO_TRANSFER_RIGHT = "\u0eaa\u0eb4\u0e94\u0ec2\u0ead\u0e99"
LAO_INHERIT_RIGHT = "\u0eaa\u0eb4\u0e94\u0eaa\u0eb7\u0e9a\u0e97\u0ead\u0e94"
THAI_ARTICLE = "\u0e21\u0e32\u0e15\u0e23\u0e32"

_DISPLAY_TEXT_REPLACEMENTS = (
    ("\u0e81\u0ebb\u0e94\u0e9a\u0ea1\u0eb2\u0e8d", "\u0e81\u0ebb\u0e94\u0edd\u0eb2\u0e8d"),
    ("\u0e9c\u0ebb\u0e99\u0e9b\u0eb0\u0ec2\u0e97\u0e8d\u0e94", "\u0e9c\u0ebb\u0e99\u0e9b\u0eb0\u0ec2\u0eab\u0e8d\u0e94"),
    ("\u0ec3\u0e82\u0ec9", "\u0ec3\u0e8a\u0ec9"),
    ("\u0ec3\u0e82", "\u0ec3\u0e8a"),
    ("\u0e9a\u0e99\u0eb2\u0e81\u0e9c\u0ebb\u0e99", "\u0edd\u0eb2\u0e81\u0e9c\u0ebb\u0e99"),
    ("\u0eaa\u0eb4\u0e94\u0eab\u0ebb\u0e81\u0eab\u0ebb\u0e81", "\u0eaa\u0eb4\u0e94\u0e9b\u0ebb\u0e81\u0e9b\u0eb1\u0e81"),
    ("\u0eaa\u0eb4\u0e94\u0e9b\u0ebb\u0e81 \u0e9b\u0eb1\u0e81", "\u0eaa\u0eb4\u0e94\u0e9b\u0ebb\u0e81\u0e9b\u0eb1\u0e81"),
)

# ── System prompt ──────────────────────────────────────────────────────────────
_IRAC_SYSTEM_PROMPT = """
You are a senior legal advisor with 30+ years of experience in Thai and Lao law.

For Lao PDR legal questions, prioritize legislation from the Lao PDR Official Gazette or ingested official Lao legal documents. Treat Lao PDR as a civil-law jurisdiction where statutes, regulations, decrees, and promulgated legislation are primary. Do not treat court decisions as binding precedent unless the retrieved context explicitly says so. If an English translation conflicts with Lao text, prefer the Lao official text and flag translation uncertainty.
Use LEGAL QUESTION ANALYSIS as the issue-spotting and research brief. Authority hints inside that analysis are search hypotheses only; do not cite or rely on them unless the same authority appears in the retrieved legal context.
Use CONVERSATION MEMORY only to understand prior facts, user goals, follow-up questions, and what has already been explained. Never treat conversation memory as legal authority.

═══ STRICT GENERATION RULES ═══
1. ONLY use information from the CONTEXT block — never from training memory.
2. EVERY legal statement MUST cite a specific law or case from the context.
3. If context is insufficient, reply:
   {"insufficient_context": true, "reason": "brief explanation"}
4. Structure ALL responses using the IRAC JSON format below.
5. Be concise but complete — courts require precision.
6. Return compact JSON only. Do not add markdown fences, commentary, or prose outside JSON.
7. Keep output focused: max 3 items for strengths/weaknesses/counter_args/rebuttals, max 8 action_steps when listing statutory requirements, one sentence per item, statute text <= 500 characters, application.analysis <= 1000 characters, reasoning_notes <= 120 characters.
8. Return minified one-line JSON. Do not pretty-print, indent, or wrap in code fences.
9. If the current question is a follow-up, resolve it using CONVERSATION MEMORY before answering.
10. If QUERY MODE is "general" or RESPONSE STYLE is "plain", answer as legal information, not litigation strategy:
   - Do not discuss win/loss, settlement, case prospects, or settlement unless the user asks about a dispute.
   - Use action_steps for exact statutory requirements, conditions, prohibitions, or checklist items.
   - If the context says an article contains several conditions, enumerate every condition found in the context.
   - If the exact article text is incomplete or OCR-noisy, say that clearly and do not invent missing conditions.
11. For case_strategy/evidence/action_plan only, include litigation risk, counter-arguments, and win_probability.

═══ OUTPUT FORMAT (strict JSON) ═══
{
  "irac": {
    "issue": {
      "primary": "precise legal question",
      "secondary": ["sub-issue 1", "sub-issue 2"]
    },
    "rule": {
      "statutes": [
        {
          "name": "law name",
          "section": "section number",
          "text": "relevant text",
          "status": "ACTIVE|AMENDED|REPEALED",
          "year": 2565
        }
      ],
      "precedents": [
        {
          "case_no": "ฎ. XXXX/XXXX",
          "court": "court name",
          "relevance": "what principle it establishes",
          "outcome": "ผู้ฟ้องชนะ|ผู้ฟ้องแพ้",
          "graph_path": "citation chain note"
        }
      ]
    },
    "application": {
      "analysis": "detailed application of rules to facts",
      "strengths": ["strength 1"],
      "weaknesses": ["weakness 1"],
      "counter_args": ["counter-argument 1"],
      "rebuttals": ["rebuttal 1"]
    },
    "conclusion": {
      "recommendation": "clear actionable advice",
      "action_steps": ["step 1", "step 2"],
      "risk_level": "LOW|MEDIUM|HIGH",
      "win_probability": 0.72,
      "settlement_note": "settlement consideration if applicable"
    }
  },
  "citations": [
    {
      "ref": "law section or case number exactly as shown in CONTEXT",
      "status": "UNVERIFIED",
      "note": "why this citation supports the answer"
    }
  ],
  "confidence": 0.85,
  "reasoning_notes": "internal notes on analysis quality"
}

═══ LANGUAGE ═══
Respond in the same language as the user query (Thai/English/Lao).
Legal citations always use official section/case numbers as-is.
"""

_CONTEXT_TEMPLATE = """
═══ RETRIEVED LEGAL CONTEXT ═══
{retrieved_documents}

═══ CASE MEMORY SUMMARY ═══
{case_memory_summary}

CONVERSATION MEMORY
{conversation_memory}

RESPONSE MODE
query_mode={query_mode}
response_style={response_style}
query_type={query_type}

═══ USER LEGAL QUERY ═══
{question}
"""

_FOCUSED_STATUTORY_SYSTEM_PROMPT = """
You are a senior Lao legal advisor. Answer a statutory question using ONLY the single statutory excerpt provided by the system.

Rules:
1. Do not use outside knowledge.
2. If the excerpt does not answer the question, say the retrieved article is incomplete or insufficient.
3. Keep the answer practical and professional.
4. Return compact JSON only, with this shape:
{"recommendation":"direct answer","analysis":"short legal explanation grounded in the excerpt","action_steps":["step or condition 1"],"confidence":0.85}
5. Do not add markdown, comments, or text outside JSON.
"""

_FOCUSED_STATUTORY_TEMPLATE = """
LAW TITLE:
{title}

SECTION:
{section}

STATUTORY EXCERPT:
{statute_text}

USER QUESTION:
{question}
"""


class IracReasoningAgent(BaseAgent):
    name = "reasoning"

    async def _execute(
        self,
        *,
        question: str,
        research: dict | None = None,
        document: dict | None = None,
        evidence: dict | None = None,
        memory: dict,
        response_language: str | None = None,
        query_mode: str | None = None,
        response_style: str | None = None,
        query_type: str | None = None,
        model_override: str | None = None,
    ) -> dict[str, Any]:
        settings = get_settings()
        reasoning_model = model_override or settings.model_reasoning
        language = infer_response_language(question, response_language)

        # Build context from all upstream agents
        context = self._build_context(
            question=question,
            research=research,
            document=document,
            evidence=evidence,
            memory=memory,
        )

        missing_case_facts_reason = self._missing_case_facts_reason(question, memory)
        if missing_case_facts_reason:
            log.info("reasoning.skipped_llm_missing_case_facts", reason=missing_case_facts_reason)
            parsed = self._insufficient_context_response(question, missing_case_facts_reason)
            parsed["_tokens"] = 0
            return parsed

        focused_response = await self._focused_statutory_response(
            question=question,
            research=research,
            language=language,
            query_mode=query_mode,
            response_style=response_style,
            query_type=query_type,
            model_override=reasoning_model,
        )
        if focused_response:
            log.info("reasoning.focused_statutory_response", question_type=query_type)
            return focused_response

        user_message = _CONTEXT_TEMPLATE.format(
            retrieved_documents=context["docs"],
            case_memory_summary=context["memory"],
            conversation_memory=context["conversation"],
            query_mode=query_mode or "general",
            response_style=response_style or "plain",
            query_type=query_type or "legal_question",
            question=question,
        )

        result = await self._call_llm(
            model=reasoning_model,
            system=f"{_IRAC_SYSTEM_PROMPT}\n\nLANGUAGE OVERRIDE:\n{response_language_instruction(language)}",
            user_message=user_message,
            max_tokens=settings.llm_max_tokens_reasoning,
        )

        parsed = self._parse_irac_response(result.text, question)
        parsed["_tokens"] = result.total_tokens

        log.info(
            "reasoning.done",
            confidence=parsed.get("confidence"),
            has_statutes=len(parsed.get("irac", {}).get("rule", {}).get("statutes", [])),
            has_precedents=len(parsed.get("irac", {}).get("rule", {}).get("precedents", [])),
        )

        return parsed

    def _build_context(
        self,
        *,
        question: str,
        research: dict | None,
        document: dict | None,
        evidence: dict | None,
        memory: dict,
    ) -> dict[str, str]:
        doc_parts: list[str] = []

        if research and research.get("retrieved_documents"):
            settings = get_settings()
            query_analysis = research.get("query_analysis") if isinstance(research.get("query_analysis"), dict) else {}
            if query_analysis:
                safe_analysis = {
                    "jurisdiction": query_analysis.get("jurisdiction"),
                    "practice_area": query_analysis.get("practice_area"),
                    "issue_type": query_analysis.get("issue_type"),
                    "legal_issues": query_analysis.get("legal_issues"),
                    "material_facts": query_analysis.get("material_facts"),
                    "missing_facts": query_analysis.get("missing_facts"),
                    "requested_outcome": query_analysis.get("requested_outcome"),
                    "authority_hints": [
                        {
                            "law_name": hint.get("law_name"),
                            "article": hint.get("article"),
                            "reason": hint.get("reason"),
                        }
                        for hint in (query_analysis.get("authority_hints") or [])[:4]
                        if isinstance(hint, dict)
                    ],
                }
                doc_parts.append(
                    "[LEGAL QUESTION ANALYSIS]\n"
                    f"{json.dumps(safe_analysis, ensure_ascii=False)[:1400]}"
                )
            retrieval = research.get("retrieval") if isinstance(research.get("retrieval"), dict) else {}
            coverage = retrieval.get("coverage") if isinstance(retrieval.get("coverage"), dict) else {}
            if coverage:
                doc_parts.append(
                    "[RETRIEVAL QUALITY] "
                    f"count={coverage.get('count')} "
                    f"statutes={coverage.get('statute_count')} "
                    f"official_sources={coverage.get('official_source_count')} "
                    f"clean_text={coverage.get('clean_text_count')} "
                    f"reason={coverage.get('reason') or 'ok'}"
                )
            chunks = self._prioritise_target_sections(question, research["retrieved_documents"])
            for chunk in chunks[:max(1, settings.reasoning_context_top_k)]:
                section = f" {chunk.get('section')}" if chunk.get("section") else ""
                score = f" score={chunk.get('final_score'):.4f}" if isinstance(chunk.get("final_score"), (int, float)) else ""
                source = f" source={chunk.get('source_url')}" if chunk.get("source_url") else ""
                target = " TARGET_SECTION" if chunk.get("_target_section_match") else ""
                doc_parts.append(
                    f"[{chunk.get('type', 'doc').upper()}{target}] "
                    f"{chunk.get('title', '')}{section}{score}{source} "
                    f"— {chunk.get('content', '')[:max(120, settings.reasoning_context_chunk_chars)]}"
                )

        if document and document.get("clauses"):
            doc_parts.append(f"[DOCUMENT CLAUSES]\n{json.dumps(document['clauses'], ensure_ascii=False)}")

        if evidence and evidence.get("items"):
            doc_parts.append(f"[EVIDENCE]\n{json.dumps(evidence['items'], ensure_ascii=False)}")

        memory_parts: list[str] = []
        conversation_parts: list[str] = []
        if not memory.get("empty"):
            if memory.get("facts_summary"):
                memory_parts.append(f"Case facts: {memory['facts_summary']}")
            highlights = memory.get("memory_highlights") or {}
            if highlights.get("past_strategies"):
                memory_parts.append(f"Past strategies: {highlights['past_strategies']}")
            if memory.get("conversation_summary"):
                conversation_parts.append(str(memory["conversation_summary"])[:2200])
            if memory.get("current_user_state"):
                conversation_parts.append(f"Current user state: {memory['current_user_state']}")
            if memory.get("last_assistant_answer"):
                conversation_parts.append(f"Last assistant answer: {memory['last_assistant_answer'][:700]}")

        return {
            "docs": "\n\n".join(doc_parts) or "No retrieved documents available.",
            "memory": "\n".join(memory_parts) or "No prior case memory.",
            "conversation": "\n".join(conversation_parts) or "No prior conversation memory.",
        }

    def _prioritise_target_sections(self, question: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        targets = self._section_numbers_from_question(question)
        if not targets:
            return chunks

        def has_target(chunk: dict[str, Any]) -> bool:
            haystack = f"{chunk.get('section') or ''}\n{str(chunk.get('content') or '')[:220]}"
            return any(self._matches_section(haystack, target) for target in targets)

        def sort_key(chunk: dict[str, Any]) -> tuple[int, float]:
            base_score = 0.0
            for key in ("final_score", "_rerank_score", "score"):
                try:
                    base_score = float(chunk.get(key))
                    break
                except (TypeError, ValueError):
                    continue
            return (1 if has_target(chunk) else 0, base_score)

        sorted_chunks = sorted(chunks, key=sort_key, reverse=True)
        return [{**chunk, "_target_section_match": has_target(chunk)} for chunk in sorted_chunks]

    def _section_numbers_from_question(self, question: str) -> list[str]:
        matches = re.findall(
            rf"(?:{LAO_ARTICLE}(?:\s*\u0e97\u0eb5)?|{THAI_ARTICLE}|Article|Art\.?|Section|Sec\.?)\s*([0-9]{{1,4}})",
            question,
            flags=re.IGNORECASE,
        )
        targets: list[str] = []
        seen: set[str] = set()
        for match in matches:
            value = str(match).lstrip("0") or "0"
            if value not in seen:
                seen.add(value)
                targets.append(value)
        return targets

    def _matches_section(self, text: str, target: str) -> bool:
        patterns = (
            rf"(?:{LAO_ARTICLE}|{THAI_ARTICLE}|Article|Art\.?|Section|Sec\.?)\s*0*{re.escape(target)}(?:\D|$)",
            rf"^0*{re.escape(target)}(?:\.|\s)",
        )
        return any(re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) for pattern in patterns)

    async def _focused_statutory_response(
        self,
        *,
        question: str,
        research: dict | None,
        language: str,
        query_mode: str | None,
        response_style: str | None,
        query_type: str | None,
        model_override: str | None = None,
    ) -> dict[str, Any] | None:
        if query_type not in {None, "legal_question"}:
            return None
        if query_mode not in {None, "general"}:
            return None
        if response_style not in {None, "plain", "irac"}:
            return None
        if not research or not research.get("retrieved_documents"):
            return None

        candidate = self._focused_statutory_candidate(question, research)
        if not candidate:
            return None

        settings = get_settings()
        direct_payload = self._direct_focused_statutory_payload(
            question=question,
            candidate=candidate,
            language=language,
        )
        if direct_payload:
            response = self._build_focused_statutory_irac(
                question=question,
                candidate=candidate,
                payload=direct_payload,
            )
            response["_tokens"] = 0
            return response

        try:
            result = await self._call_llm(
                model=model_override or settings.model_reasoning,
                system=f"{_FOCUSED_STATUTORY_SYSTEM_PROMPT}\n\nLANGUAGE OVERRIDE:\n{response_language_instruction(language)}",
                user_message=_FOCUSED_STATUTORY_TEMPLATE.format(
                    title=candidate["title"],
                    section=candidate["section"],
                    statute_text=candidate["statute_text"],
                    question=question,
                ),
                max_tokens=min(settings.llm_max_tokens_reasoning, 900),
            )
            payload = self._parse_focused_statutory_payload(result.text)
            response = self._build_focused_statutory_irac(
                question=question,
                candidate=candidate,
                payload=payload,
            )
            response["_tokens"] = result.total_tokens
            return response
        except Exception as exc:  # noqa: BLE001
            log.warning("reasoning.focused_statutory_llm_failed", error=str(exc))
            response = self._build_focused_statutory_irac(
                question=question,
                candidate=candidate,
                payload=self._fallback_focused_statutory_payload(candidate, language),
            )
            response["_tokens"] = 0
            return response

    def _focused_statutory_candidate(self, question: str, research: dict[str, Any]) -> dict[str, Any] | None:
        targets = self._target_sections_from_research(question, research)
        if not targets:
            return self._infer_focused_statutory_candidate(question, research)

        chunks = self._prioritise_chunks_by_targets(research["retrieved_documents"], targets)
        if not chunks:
            return None

        top_chunk = chunks[0]
        section = str(top_chunk.get("section") or top_chunk.get("section_ref") or "").strip()
        content = str(top_chunk.get("content") or "")
        matched_article = self._first_matching_target(section, content, targets)
        if not matched_article:
            return None

        title = str(top_chunk.get("title") or self._best_law_name_from_research(research) or "Retrieved legal source").strip()
        statute_text = self._clean_statute_excerpt(content, max_chars=1600)
        return {
            "chunk": top_chunk,
            "title": title,
            "section": section or f"{LAO_ARTICLE} {matched_article}",
            "article": matched_article,
            "statute_text": statute_text,
            "citation_ref": f"{title} {section or f'{LAO_ARTICLE} {matched_article}'}".strip(),
        }

    def _infer_focused_statutory_candidate(
        self,
        question: str,
        research: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self._is_lao_land_use_rights_protection_question(question):
            return None

        chunks = research.get("retrieved_documents") if isinstance(research.get("retrieved_documents"), list) else []
        for chunk in chunks[:5]:
            section = str(chunk.get("section") or chunk.get("section_ref") or "").strip()
            content = str(chunk.get("content") or "")
            combined = self._clean_display_text(f"{section} {content}")
            if not self._contains_land_use_rights_answer(combined):
                continue

            article = self._article_number_from_text(section, content)
            if not article:
                continue

            title = str(chunk.get("title") or self._best_law_name_from_research(research) or "Retrieved legal source").strip()
            statute_text = self._clean_statute_excerpt(content, max_chars=1600)
            resolved_section = section or f"{LAO_ARTICLE} {article}"
            return {
                "chunk": chunk,
                "title": title,
                "section": resolved_section,
                "article": article,
                "statute_text": statute_text,
                "citation_ref": f"{title} {resolved_section}".strip(),
            }
        return None

    def _parse_focused_statutory_payload(self, text: str) -> dict[str, Any]:
        clean = self._extract_json_payload(text)
        payload = json.loads(clean)
        if not isinstance(payload, dict):
            raise ValueError("Focused statutory answer must be a JSON object")

        if "irac" in payload and isinstance(payload.get("irac"), dict):
            irac = payload["irac"]
            conclusion = irac.get("conclusion") if isinstance(irac.get("conclusion"), dict) else {}
            application = irac.get("application") if isinstance(irac.get("application"), dict) else {}
            payload = {
                "recommendation": conclusion.get("recommendation"),
                "analysis": application.get("analysis"),
                "action_steps": conclusion.get("action_steps"),
                "confidence": payload.get("confidence"),
            }

        recommendation = str(payload.get("recommendation") or "").strip()
        analysis = str(payload.get("analysis") or "").strip()
        action_steps = [
            str(step).strip()
            for step in (payload.get("action_steps") if isinstance(payload.get("action_steps"), list) else [])
            if str(step).strip()
        ][:8]
        raw_text = "\n".join([recommendation, analysis, *action_steps])
        if self._has_degenerate_text(raw_text):
            raise ValueError("Focused statutory answer failed text quality gate")

        recommendation = self._clean_display_text(recommendation)
        analysis = self._clean_display_text(analysis)
        action_steps = [self._clean_display_text(step) for step in action_steps]
        if self._has_degenerate_text("\n".join([recommendation, analysis, *action_steps])):
            raise ValueError("Focused statutory answer failed cleaned text quality gate")

        confidence = self._bounded_confidence(payload.get("confidence"), default=0.84)
        if not recommendation and not analysis:
            raise ValueError("Focused statutory answer is empty")
        return {
            "recommendation": recommendation or analysis,
            "analysis": analysis or recommendation,
            "action_steps": action_steps,
            "confidence": confidence,
        }

    def _build_focused_statutory_irac(
        self,
        *,
        question: str,
        candidate: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        confidence = self._bounded_confidence(payload.get("confidence"), default=0.84)
        return {
            "irac": {
                "issue": {"primary": question, "secondary": []},
                "rule": {
                    "statutes": [
                        {
                            "name": candidate["title"],
                            "section": candidate["section"],
                            "text": self._clean_statute_excerpt(candidate["statute_text"], max_chars=500),
                            "status": "ACTIVE",
                            "year": None,
                        }
                    ],
                    "precedents": [],
                },
                "application": {
                    "analysis": str(payload.get("analysis") or "").strip(),
                    "strengths": [],
                    "weaknesses": [],
                    "counter_args": [],
                    "rebuttals": [],
                },
                "conclusion": {
                    "recommendation": str(payload.get("recommendation") or "").strip(),
                    "action_steps": [
                        str(step).strip()
                        for step in (payload.get("action_steps") if isinstance(payload.get("action_steps"), list) else [])
                        if str(step).strip()
                    ][:8],
                    "action_label": str(payload.get("action_label") or "").strip() or None,
                    "risk_level": "LOW",
                    "win_probability": 0.0,
                    "settlement_note": None,
                },
            },
            "citations": [
                {
                    "ref": candidate["citation_ref"],
                    "status": "UNVERIFIED",
                    "note": "Grounded in the retrieved statutory section.",
                    "chunk_id": candidate.get("chunk", {}).get("chunk_id") or candidate.get("chunk", {}).get("id"),
                    "source_id": candidate.get("chunk", {}).get("source_id"),
                    "source_table": candidate.get("chunk", {}).get("source_table"),
                    "section": candidate["section"],
                }
            ],
            "confidence": confidence,
            "_confidence": confidence,
            "reasoning_notes": "focused statutory answer from retrieved section",
        }

    def _fallback_focused_statutory_payload(self, candidate: dict[str, Any], language: str) -> dict[str, Any]:
        title = candidate["title"]
        section = candidate["section"]
        excerpt = self._clean_statute_excerpt(candidate["statute_text"], max_chars=700)
        if language in {"lo", "th"}:
            return {
                "recommendation": f"{title} {section}: {excerpt}",
                "analysis": excerpt,
                "action_steps": [],
                "confidence": 0.72,
            }
        if language == "lo":
            recommendation = f"ອີງຕາມ {title} {section}, ຄຳຕອບຕ້ອງອີງໃສ່ຂໍ້ຄວາມຂອງມາດຕານີ້."
            analysis = f"ຂໍ້ຄວາມທີ່ຄົ້ນພົບລະບຸວ່າ: {excerpt}"
            action_steps = [
                "ກວດສອບຂໍ້ຄວາມມາດຕາສະບັບເຕັມກ່ອນນຳໄປໃຊ້ກັບກໍລະນີຈິງ.",
                "ຖ້າມີຂໍ້ເທັດຈິງສະເພາະ, ໃຫ້ນຳມາປຽບທຽບກັບເງື່ອນໄຂໃນມາດຕານີ້.",
            ]
        elif language == "th":
            recommendation = f"อ้างอิง {title} {section} คำตอบต้องยึดตามข้อความของมาตรานี้"
            analysis = f"ข้อความที่ค้นพบระบุว่า: {excerpt}"
            action_steps = [
                "ตรวจสอบข้อความมาตราฉบับเต็มก่อนนำไปใช้กับข้อเท็จจริงจริง",
                "หากมีข้อเท็จจริงเฉพาะ ให้นำมาเทียบกับเงื่อนไขในมาตรานี้",
            ]
        else:
            recommendation = f"Based on {title} {section}, the answer must be grounded in this retrieved statutory text."
            analysis = f"The retrieved excerpt states: {excerpt}"
            action_steps = [
                "Check the full official article before applying it to a real matter.",
                "Compare any specific facts against the conditions in this article.",
            ]
        return {
            "recommendation": recommendation,
            "analysis": analysis,
            "action_steps": action_steps,
            "confidence": 0.72,
        }

    def _direct_focused_statutory_payload(
        self,
        *,
        question: str,
        candidate: dict[str, Any],
        language: str,
    ) -> dict[str, Any] | None:
        if language != "lo" and not contains_lao_script(question):
            return None
        if not self._is_lao_land_use_rights_protection_question(question):
            return None
        if not self._contains_land_use_rights_answer(candidate.get("statute_text", "")):
            return None

        citation = f"{candidate['title']} {candidate['section']}".strip()
        rights = [
            LAO_GUARD_RIGHT,
            LAO_USE_RIGHT,
            LAO_FRUITS_RIGHT,
            LAO_TRANSFER_RIGHT,
            LAO_INHERIT_RIGHT,
        ]
        rights_text = ", ".join(rights)
        recommendation = (
            f"\u0ead\u0eb5\u0e87\u0e95\u0eb2\u0ea1 {citation}, "
            f"\u0e9c\u0eb9\u0ec9\u0ec4\u0e94\u0ec9\u0eae\u0eb1\u0e9a{LAO_LAND_USE_RIGHT}{LAO_LAND}"
            f"\u0ec4\u0e94\u0ec9\u0eae\u0eb1\u0e9a\u0e81\u0eb2\u0e99\u0eae\u0eb1\u0e9a\u0e9b\u0eb0\u0e81\u0eb1\u0e99 5 {LAO_RIGHT}: "
            f"{rights_text}."
        )
        analysis = (
            f"{candidate['section']} "
            "\u0ea5\u0eb0\u0e9a\u0eb8\u0ea7\u0ec8\u0eb2"
            "\u0ea5\u0eb1\u0e94"
            f"{LAO_PROTECTION}"
            "\u0eaa\u0eb4\u0e94\u0e9c\u0ebb\u0e99\u0e9b\u0eb0\u0ec2\u0eab\u0e8d\u0e94"
            f"\u0e82\u0ead\u0e87\u0e9c\u0eb9\u0ec9\u0ec4\u0e94\u0ec9\u0eae\u0eb1\u0e9a{LAO_LAND_USE_RIGHT}{LAO_LAND} "
            "\u0ec1\u0ea5\u0eb0\u0eae\u0eb1\u0e9a\u0e9b\u0eb0\u0e81\u0eb1\u0e99"
            f"{LAO_RIGHT}\u0ec0\u0eab\u0ebc\u0ebb\u0ec8\u0eb2\u0e99\u0eb5\u0ec9\u0ec2\u0e94\u0e8d\u0e81\u0ebb\u0e87."
        )
        return {
            "recommendation": recommendation,
            "analysis": analysis,
            "action_steps": rights,
            "action_label": "\u0eaa\u0eb4\u0e94\u0e97\u0eb5\u0ec8\u0ec4\u0e94\u0ec9\u0eae\u0eb1\u0e9a",
            "confidence": 0.84,
        }

    def _is_lao_land_use_rights_protection_question(self, question: str) -> bool:
        clean = self._clean_display_text(question)
        has_land_use_right = any(
            marker in clean
            for marker in (
                LAO_LAND_USE_RIGHT,
                LAO_LAND_USE_RIGHT_ALT,
                LAO_LAND_USE_RIGHT_OCR,
            )
        )
        has_protection_intent = (
            LAO_PROTECTION in clean
            or LAO_GUARD_RIGHT in clean
            or "\u0e9b\u0ebb\u0e81\u0e9b\u0eb1\u0e81" in clean
        )
        return contains_lao_script(clean) and has_land_use_right and has_protection_intent

    def _contains_land_use_rights_answer(self, text: str) -> bool:
        clean = self._clean_display_text(text)
        has_land_use_right = any(
            marker in clean
            for marker in (
                LAO_LAND_USE_RIGHT,
                LAO_LAND_USE_RIGHT_ALT,
                LAO_LAND_USE_RIGHT_OCR,
            )
        )
        return (
            has_land_use_right
            and LAO_GUARD_RIGHT in clean
            and LAO_USE_RIGHT in clean
            and LAO_FRUITS_RIGHT in clean
            and LAO_TRANSFER_RIGHT in clean
            and LAO_INHERIT_RIGHT in clean
        )

    def _article_number_from_text(self, section: str, content: str) -> str | None:
        haystack = f"{section}\n{content[:220]}"
        pattern = rf"(?:{LAO_ARTICLE}|{THAI_ARTICLE}|Article|Art\.?|Section|Sec\.?)\s*0*([0-9]{{1,4}})"
        match = re.search(pattern, haystack, flags=re.IGNORECASE)
        if not match:
            return None
        return match.group(1).lstrip("0") or "0"

    def _best_law_name_from_research(self, research: dict[str, Any]) -> str | None:
        analysis = research.get("query_analysis") if isinstance(research.get("query_analysis"), dict) else {}
        hints = analysis.get("authority_hints") if isinstance(analysis.get("authority_hints"), list) else []
        for hint in hints:
            if not isinstance(hint, dict):
                continue
            law_name = str(hint.get("law_name") or "").strip()
            if law_name:
                return law_name
        return None

    def _bounded_confidence(self, value: Any, *, default: float) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = default
        return max(0.0, min(confidence, 0.92))

    def _target_sections_from_research(self, question: str, research: dict[str, Any]) -> list[str]:
        targets = self._section_numbers_from_question(question)
        analysis = research.get("query_analysis") if isinstance(research.get("query_analysis"), dict) else {}
        hints = analysis.get("authority_hints") if isinstance(analysis.get("authority_hints"), list) else []
        for hint in hints:
            if not isinstance(hint, dict):
                continue
            article = str(hint.get("article") or "").strip()
            if article:
                targets.append(article)

        unique: list[str] = []
        seen: set[str] = set()
        for target in targets:
            match = re.search(r"0*([0-9]{1,4})", str(target))
            if not match:
                continue
            value = match.group(1)
            if value not in seen:
                seen.add(value)
                unique.append(value)
        return unique[:5]

    def _prioritise_chunks_by_targets(
        self,
        chunks: list[dict[str, Any]],
        targets: list[str],
    ) -> list[dict[str, Any]]:
        def target_match(chunk: dict[str, Any]) -> bool:
            haystack = f"{chunk.get('section') or chunk.get('section_ref') or ''}\n{str(chunk.get('content') or '')[:260]}"
            return any(self._matches_section(haystack, target) for target in targets)

        def score(chunk: dict[str, Any]) -> float:
            for key in ("_rerank_score", "final_score", "score"):
                try:
                    return float(chunk.get(key))
                except (TypeError, ValueError):
                    continue
            return 0.0

        sorted_chunks = sorted(chunks, key=lambda chunk: (target_match(chunk), score(chunk)), reverse=True)
        return [chunk for chunk in sorted_chunks if target_match(chunk)]

    def _first_matching_target(self, section: str, content: str, targets: list[str]) -> str | None:
        haystack = f"{section}\n{content[:260]}"
        for target in targets:
            if self._matches_section(haystack, target):
                return target
        return None

    def _clean_statute_excerpt(self, text: str, *, max_chars: int) -> str:
        clean = self._clean_display_text(text or "")
        return clean[:max_chars]

    def _clean_display_text(self, text: str) -> str:
        clean = re.sub(r"\s+", " ", str(text or "")).strip()
        for old, new in _DISPLAY_TEXT_REPLACEMENTS:
            clean = clean.replace(old, new)
        return clean

    def _has_degenerate_text(self, text: str) -> bool:
        clean = str(text or "")
        if not clean.strip():
            return False
        if "\ufffd" in clean:
            return True
        compact = re.sub(r"\s+", "", clean)
        if re.search(r"(.)\1{24,}", compact):
            return True
        if re.search(r"([^\s]{2,16}?)\1{5,}", compact):
            return True
        if re.search(r"(?:^|\s)(\S{2,24})(?:\s+\1){5,}(?:\s|$)", clean):
            return True
        if len(compact) >= 220:
            tail = compact[-180:]
            if tail and (len(set(tail)) / len(tail)) < 0.16:
                return True
        return False

    def _contains_degenerate_text(self, value: Any) -> bool:
        if isinstance(value, str):
            return self._has_degenerate_text(value)
        if isinstance(value, dict):
            return any(self._contains_degenerate_text(item) for item in value.values())
        if isinstance(value, list):
            return any(self._contains_degenerate_text(item) for item in value)
        return False

    def _missing_case_facts_reason(self, question: str, memory: dict[str, Any] | None = None) -> str | None:
        settings = get_settings()
        clean = re.sub(r"\s+", " ", question).strip()
        if len(clean) >= settings.case_analysis_min_fact_chars:
            return None
        if memory and (memory.get("facts_summary") or memory.get("conversation_summary")):
            return None

        lowered = clean.casefold()
        analysis_markers = ("วิเคราะห์", "ວິເຄາະ", "analyze", "analysis")
        case_markers = ("คดี", "ຄະດີ", "case")
        if not (
            any(marker in lowered for marker in analysis_markers)
            and any(marker in lowered for marker in case_markers)
        ):
            return None

        if contains_lao_script(clean):
            return (
                "ຄຳຖາມເປັນການຂໍວິເຄາະຄະດີ ແຕ່ຍັງບໍ່ມີຂໍ້ເທັດຈິງພຽງພໍ "
                "ເຊັ່ນ ຄູ່ກໍລະນີ, ຂໍ້ພິພາດ, ເອກະສານສິດ, ແລະຄຳຖາມກົດໝາຍສະເພາະ."
            )
        if contains_thai_script(clean):
            return (
                "คำถามเป็นการขอวิเคราะห์คดี แต่ยังไม่มีข้อเท็จจริงเพียงพอ เช่น คู่กรณี "
                "ข้อพิพาท เอกสารสิทธิ และประเด็นกฎหมายเฉพาะ."
            )
        return (
            "The query asks for case analysis but does not provide enough facts, parties, dispute details, "
            "documents, or specific legal issues."
        )

    def _parse_irac_response(self, text: str, question: str) -> dict[str, Any]:
        """Parse LLM response into IRAC dict, with graceful fallback."""
        # Try direct JSON parse
        try:
            # Strip markdown code fences if present
            clean = self._extract_json_payload(text)
            data = json.loads(clean)
            return self._normalise_parsed_irac(data, question)

        except (json.JSONDecodeError, ValueError):
            log.warning("reasoning.json_parse_failed", raw_length=len(text))
            repaired = self._try_parse_repaired_json(text)
            if repaired:
                return self._normalise_parsed_irac(repaired, question)
            if self._looks_like_structured_json(text) or self._looks_like_prompt_leak(text):
                return self._structured_parse_failure_response(question)
            return self._fallback_response(question, text)

    def _normalise_parsed_irac(self, data: dict[str, Any], question: str) -> dict[str, Any]:
        if data.get("insufficient_context"):
            log.warning("reasoning.insufficient_context", reason=data.get("reason"))
            return self._insufficient_context_response(question, data.get("reason", ""))

        normalised_irac = self._normalise_irac_shape(data.get("irac"), question)
        if self._contains_prompt_leak(normalised_irac):
            log.warning("reasoning.prompt_leak_in_structured_response")
            return self._structured_parse_failure_response(question)
        if self._contains_degenerate_text(normalised_irac):
            log.warning("reasoning.degenerate_text_in_structured_response")
            return self._structured_parse_failure_response(question)

        data = {
            **data,
            "irac": normalised_irac,
        }
        confidence = float(data.get("confidence", 0.75))
        return {**data, "_confidence": confidence}

    def _extract_json_payload(self, text: str) -> str:
        clean = self._strip_json_fences(text)
        if clean.startswith("{") and clean.endswith("}"):
            return clean

        start = clean.find("{")
        end = clean.rfind("}")
        if start >= 0 and end > start:
            return clean[start : end + 1]
        return clean

    def _try_parse_repaired_json(self, text: str) -> dict[str, Any] | None:
        clean = self._strip_json_fences(text)
        start = clean.find("{")
        if start >= 0:
            clean = clean[start:].strip()

        decoder = json.JSONDecoder()
        try:
            parsed, _ = decoder.raw_decode(clean)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        repaired = self._close_incomplete_json(clean)
        if not repaired:
            return None

        try:
            parsed = json.loads(repaired)
        except (json.JSONDecodeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None

    def _strip_json_fences(self, text: str) -> str:
        return re.sub(r"```(?:json)?\s*|\s*```", "", text.strip(), flags=re.IGNORECASE)

    def _normalise_irac_shape(self, value: Any, question: str) -> dict[str, Any]:
        irac = value if isinstance(value, dict) else {}
        issue = irac.get("issue") if isinstance(irac.get("issue"), dict) else {}
        rule = irac.get("rule") if isinstance(irac.get("rule"), dict) else {}
        application = irac.get("application") if isinstance(irac.get("application"), dict) else {}
        conclusion = irac.get("conclusion") if isinstance(irac.get("conclusion"), dict) else {}

        return {
            "issue": {
                "primary": self._clean_display_text(str(issue.get("primary") or question)),
                "secondary": self._string_list(issue.get("secondary")),
            },
            "rule": {
                "statutes": rule.get("statutes") if isinstance(rule.get("statutes"), list) else [],
                "precedents": rule.get("precedents") if isinstance(rule.get("precedents"), list) else [],
            },
            "application": {
                "analysis": self._clean_display_text(str(application.get("analysis") or "")),
                "strengths": self._string_list(application.get("strengths")),
                "weaknesses": self._string_list(application.get("weaknesses")),
                "counter_args": self._string_list(application.get("counter_args")),
                "rebuttals": self._string_list(application.get("rebuttals")),
            },
            "conclusion": {
                "recommendation": self._clean_display_text(str(conclusion.get("recommendation") or "")),
                "action_steps": self._string_list(conclusion.get("action_steps")),
                "risk_level": self._risk_level(conclusion.get("risk_level")),
                "win_probability": self._probability(conclusion.get("win_probability")),
                "settlement_note": self._clean_display_text(conclusion.get("settlement_note")) if isinstance(conclusion.get("settlement_note"), str) else None,
            },
        }

    def _string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [self._clean_display_text(str(item)) for item in value if str(item).strip()]

    def _risk_level(self, value: Any) -> str:
        risk = str(value or "MEDIUM").strip().upper()
        return risk if risk in {"LOW", "MEDIUM", "HIGH"} else "MEDIUM"

    def _probability(self, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, number))

    def _close_incomplete_json(self, text: str) -> str | None:
        clean = text.strip()
        if not clean.startswith("{"):
            return None

        stack: list[str] = []
        in_string = False
        escaped = False

        for char in clean:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif char in "}]":
                if not stack or stack[-1] != char:
                    return None
                stack.pop()

        if in_string:
            clean += '"'

        return clean + "".join(reversed(stack))

    def _looks_like_structured_json(self, text: str) -> bool:
        clean = text.strip()
        return clean.startswith("{") or "```json" in clean.lower() or '"irac"' in clean

    def _looks_like_prompt_leak(self, text: str) -> bool:
        lowered = text.casefold()
        markers = (
            "context block",
            "retrieved legal context",
            "strict generation rules",
            "output format",
            "legal question analysis",
            "conversation memory",
            "response mode",
            "return compact json",
            "only use information from the context",
            "only use information from the context block",
        )
        return any(marker in lowered for marker in markers)

    def _contains_prompt_leak(self, value: Any) -> bool:
        if isinstance(value, str):
            return self._looks_like_prompt_leak(value)
        if isinstance(value, dict):
            return any(self._contains_prompt_leak(item) for item in value.values())
        if isinstance(value, list):
            return any(self._contains_prompt_leak(item) for item in value)
        return False

    def _safe_insufficient_reason(self, reason: str) -> str:
        clean = str(reason or "").strip()
        if not clean or self._looks_like_prompt_leak(clean):
            return ""
        return clean[:600]

    def _structured_parse_failure_response(self, question: str) -> dict[str, Any]:
        if contains_lao_script(question):
            reason = (
                "ຄຳຕອບໂຄງສ້າງຈາກແບບຈຳລອງບໍ່ສົມບູນ ລະບົບຈຶ່ງບໍ່ຄວນນຳຂໍ້ມູນດິບມາສະແດງເປັນຄຳປຶກສາ."
            )
        elif contains_thai_script(question):
            reason = (
                "คำตอบแบบมีโครงสร้างจากโมเดลไม่สมบูรณ์ ระบบจึงไม่ควรนำข้อมูลดิบมาแสดงเป็นคำปรึกษา."
            )
        else:
            reason = (
                "The model returned an incomplete structured answer, so the raw payload was withheld."
            )
        return self._insufficient_context_response(question, reason)

    def _insufficient_context_texts(self, question: str, reason: str) -> dict[str, Any]:
        if contains_lao_script(question):
            return {
                "analysis": (
                    "ຂໍ້ມູນກົດໝາຍໃນຖານຂໍ້ມູນຍັງບໍ່ພຽງພໍສຳລັບການວິເຄາະແບບ IRAC. "
                    f"ເຫດຜົນ: {reason or 'ບໍ່ພົບເອກະສານກົດໝາຍທີ່ກ່ຽວຂ້ອງ.'}"
                ),
                "recommendation": (
                    "ຕອນນີ້ລະບົບບໍ່ຄວນໃຫ້ຄຳຕອບທາງກົດໝາຍແບບຢືນຢັນ "
                    "ເພາະຍັງບໍ່ມີແຫຼ່ງອ້າງອີງກົດໝາຍລາວທີ່ກວດສອບໄດ້ໃນລະບົບ."
                ),
                "action_steps": [
                    "ເພີ່ມ/ອະນຸມັດເອກະສານກົດໝາຍລາວທີ່ເປັນທາງການເຂົ້າຖານຂໍ້ມູນ",
                    "ລອງຖາມພ້ອມຊື່ກົດໝາຍ ຫຼື ມາດຕາ ຖ້າມີ",
                    "ປຶກສາທະນາຍຄວາມທ້ອງຖິ່ນກ່ອນດຳເນີນການຈິງ",
                ],
            }

        if contains_thai_script(question):
            return {
                "analysis": (
                    "ข้อมูลกฎหมายในฐานข้อมูลยังไม่เพียงพอสำหรับวิเคราะห์แบบ IRAC "
                    f"เหตุผล: {reason or 'ไม่พบเอกสารกฎหมายที่เกี่ยวข้อง'}"
                ),
                "recommendation": (
                    "ตอนนี้ระบบไม่ควรให้คำตอบทางกฎหมายแบบยืนยัน เพราะยังไม่มีแหล่งอ้างอิงที่ตรวจสอบได้ในบริบทนี้"
                ),
                "action_steps": [
                    "เพิ่มหรืออนุมัติเอกสารกฎหมายที่เกี่ยวข้องในฐานข้อมูล",
                    "ลองถามพร้อมชื่อกฎหมายหรือมาตราที่ต้องการตรวจ",
                    "ปรึกษาทนายความก่อนดำเนินการจริง",
                ],
            }

        return {
            "analysis": (
                "The legal knowledge base does not contain enough retrieved authority for a grounded IRAC answer. "
                f"Reason: {reason or 'No relevant legal documents were found.'}"
            ),
            "recommendation": (
                "The system should not provide a definitive legal answer until official, reviewable legal sources are available."
            ),
            "action_steps": [
                "Ingest and approve official legal documents for the relevant jurisdiction.",
                "Ask again with the law name, article, or section if available.",
                "Consult a qualified local lawyer before taking real-world action.",
            ],
        }

    def _insufficient_context_response(self, question: str, reason: str) -> dict[str, Any]:
        texts = self._insufficient_context_texts(question, self._safe_insufficient_reason(reason))
        return {
            "irac": {
                "issue": {"primary": question, "secondary": []},
                "rule": {"statutes": [], "precedents": []},
                "application": {
                    "analysis": texts["analysis"],
                    "strengths": [],
                    "weaknesses": [],
                    "counter_args": [],
                    "rebuttals": [],
                },
                "conclusion": {
                    "recommendation": texts["recommendation"],
                    "action_steps": texts["action_steps"],
                    "risk_level": "MEDIUM",
                    "win_probability": 0.0,
                    "settlement_note": None,
                },
            },
            "confidence": 0.3,
            "_confidence": 0.3,
            "citations": [],
        }

    def _fallback_response(self, question: str, raw_text: str) -> dict[str, Any]:
        """When JSON parsing fails, withhold raw model text instead of treating it as legal analysis."""
        _ = raw_text
        return self._structured_parse_failure_response(question)
