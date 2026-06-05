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

# ── System prompt ──────────────────────────────────────────────────────────────
_IRAC_SYSTEM_PROMPT = """
You are a senior legal advisor with 30+ years of experience in Thai and Lao law.

For Lao PDR legal questions, prioritize legislation from the Lao PDR Official Gazette or ingested official Lao legal documents. Treat Lao PDR as a civil-law jurisdiction where statutes, regulations, decrees, and promulgated legislation are primary. Do not treat court decisions as binding precedent unless the retrieved context explicitly says so. If an English translation conflicts with Lao text, prefer the Lao official text and flag translation uncertainty.
Use LEGAL QUESTION ANALYSIS as the issue-spotting and research brief. Authority hints inside that analysis are search hypotheses only; do not cite or rely on them unless the same authority appears in the retrieved legal context.

═══ STRICT GENERATION RULES ═══
1. ONLY use information from the CONTEXT block — never from training memory.
2. EVERY legal statement MUST cite a specific law or case from the context.
3. If context is insufficient, reply:
   {"insufficient_context": true, "reason": "brief explanation"}
4. Structure ALL responses using the IRAC JSON format below.
5. Be concise but complete — courts require precision.
6. Return compact JSON only. Do not add markdown fences, commentary, or prose outside JSON.
7. Keep output small: max 3 items per array, one sentence per item, statute text <= 240 characters, application.analysis <= 800 characters, reasoning_notes <= 120 characters.
8. Return minified one-line JSON. Do not pretty-print, indent, or wrap in code fences.

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

═══ USER LEGAL QUERY ═══
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
    ) -> dict[str, Any]:
        settings = get_settings()
        language = infer_response_language(question, response_language)

        # Build context from all upstream agents
        context = self._build_context(
            research=research,
            document=document,
            evidence=evidence,
            memory=memory,
        )

        missing_case_facts_reason = self._missing_case_facts_reason(question)
        if missing_case_facts_reason:
            log.info("reasoning.skipped_llm_missing_case_facts", reason=missing_case_facts_reason)
            parsed = self._insufficient_context_response(question, missing_case_facts_reason)
            parsed["_tokens"] = 0
            return parsed

        user_message = _CONTEXT_TEMPLATE.format(
            retrieved_documents=context["docs"],
            case_memory_summary=context["memory"],
            question=question,
        )

        result = await self._call_llm(
            model=settings.model_reasoning,
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
            for chunk in research["retrieved_documents"][:max(1, settings.reasoning_context_top_k)]:
                section = f" {chunk.get('section')}" if chunk.get("section") else ""
                score = f" score={chunk.get('final_score'):.4f}" if isinstance(chunk.get("final_score"), (int, float)) else ""
                source = f" source={chunk.get('source_url')}" if chunk.get("source_url") else ""
                doc_parts.append(
                    f"[{chunk.get('type', 'doc').upper()}] "
                    f"{chunk.get('title', '')}{section}{score}{source} "
                    f"— {chunk.get('content', '')[:max(120, settings.reasoning_context_chunk_chars)]}"
                )

        if document and document.get("clauses"):
            doc_parts.append(f"[DOCUMENT CLAUSES]\n{json.dumps(document['clauses'], ensure_ascii=False)}")

        if evidence and evidence.get("items"):
            doc_parts.append(f"[EVIDENCE]\n{json.dumps(evidence['items'], ensure_ascii=False)}")

        memory_parts: list[str] = []
        if not memory.get("empty"):
            if memory.get("facts_summary"):
                memory_parts.append(f"Case facts: {memory['facts_summary']}")
            highlights = memory.get("memory_highlights") or {}
            if highlights.get("past_strategies"):
                memory_parts.append(f"Past strategies: {highlights['past_strategies']}")

        return {
            "docs": "\n\n".join(doc_parts) or "No retrieved documents available.",
            "memory": "\n".join(memory_parts) or "No prior case memory.",
        }

    def _missing_case_facts_reason(self, question: str) -> str | None:
        settings = get_settings()
        clean = re.sub(r"\s+", " ", question).strip()
        if len(clean) >= settings.case_analysis_min_fact_chars:
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
            if self._looks_like_structured_json(text):
                return self._structured_parse_failure_response(question)
            return self._fallback_response(question, text)

    def _normalise_parsed_irac(self, data: dict[str, Any], question: str) -> dict[str, Any]:
        if data.get("insufficient_context"):
            log.warning("reasoning.insufficient_context", reason=data.get("reason"))
            return self._insufficient_context_response(question, data.get("reason", ""))

        data = {
            **data,
            "irac": self._normalise_irac_shape(data.get("irac"), question),
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
                "primary": str(issue.get("primary") or question),
                "secondary": self._string_list(issue.get("secondary")),
            },
            "rule": {
                "statutes": rule.get("statutes") if isinstance(rule.get("statutes"), list) else [],
                "precedents": rule.get("precedents") if isinstance(rule.get("precedents"), list) else [],
            },
            "application": {
                "analysis": str(application.get("analysis") or ""),
                "strengths": self._string_list(application.get("strengths")),
                "weaknesses": self._string_list(application.get("weaknesses")),
                "counter_args": self._string_list(application.get("counter_args")),
                "rebuttals": self._string_list(application.get("rebuttals")),
            },
            "conclusion": {
                "recommendation": str(conclusion.get("recommendation") or ""),
                "action_steps": self._string_list(conclusion.get("action_steps")),
                "risk_level": self._risk_level(conclusion.get("risk_level")),
                "win_probability": self._probability(conclusion.get("win_probability")),
                "settlement_note": conclusion.get("settlement_note") if isinstance(conclusion.get("settlement_note"), str) else None,
            },
        }

    def _string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

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
        texts = self._insufficient_context_texts(question, reason)
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

        return {
            "irac": {
                "issue": {"primary": question, "secondary": []},
                "rule": {"statutes": [], "precedents": []},
                "application": {
                    "analysis": f"ข้อมูลไม่เพียงพอ: {reason}",
                    "strengths": [], "weaknesses": [],
                    "counter_args": [], "rebuttals": [],
                },
                "conclusion": {
                    "recommendation": "แนะนำให้ปรึกษาทนายความโดยตรง เนื่องจากข้อมูลในระบบไม่เพียงพอ",
                    "action_steps": ["ติดต่อทนายความผู้เชี่ยวชาญ", "รวบรวมเอกสารที่เกี่ยวข้อง"],
                    "risk_level": "MEDIUM",
                    "win_probability": 0.5,
                    "settlement_note": None,
                },
            },
            "confidence": 0.3,
            "_confidence": 0.3,
            "citations": [],
        }

    def _fallback_response(self, question: str, raw_text: str) -> dict[str, Any]:
        """When JSON parse fails, wrap raw text in minimal IRAC structure."""
        if self._looks_like_structured_json(raw_text):
            return self._structured_parse_failure_response(question)

        language = infer_response_language(question)
        if language == "lo":
            recommendation = "ກະລຸນາກວດສອບຄຳຕອບນີ້ກັບທະນາຍຄວາມລາວກ່ອນນຳໄປໃຊ້ຈິງ."
        elif language == "th":
            recommendation = "โปรดตรวจสอบคำตอบนี้กับทนายความก่อนนำไปใช้จริง."
        else:
            recommendation = "Please review this answer with a qualified lawyer before taking real-world action."

        return {
            "irac": {
                "issue": {"primary": question, "secondary": []},
                "rule": {"statutes": [], "precedents": []},
                "application": {
                    "analysis": raw_text[:2000],
                    "strengths": [], "weaknesses": [],
                    "counter_args": [], "rebuttals": [],
                },
                "conclusion": {
                    "recommendation": recommendation,
                    "action_steps": [],
                    "risk_level": "MEDIUM",
                    "win_probability": 0.5,
                    "settlement_note": None,
                },
            },
            "confidence": 0.5,
            "_confidence": 0.5,
            "citations": [],
        }
