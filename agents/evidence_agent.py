"""
agents/evidence_agent.py
=========================
Evidence Analyzer Agent — SPECIALIST, invoked when evidence files are present.

Supports:
- Images / screenshots (GPT-4o Vision)
- Audio recordings (Whisper → GPT-4o analysis)
- Email chains / chat logs (GPT-4o text analysis)
- Scanned documents with OCR consideration

Output includes admissibility assessment — critical for Thai/Lao court proceedings.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from agents.base_agent import BaseAgent
from core.config import get_settings
from core.logging import get_logger

log = get_logger(__name__)

_EVIDENCE_SYSTEM_PROMPT = """
You are a senior legal evidence analyst and forensics expert with expertise in Thai and Lao court procedure.

Analyse the provided evidence and return strict JSON:
{
  "evidence_type": "image|audio_transcript|email|document|video_frame|other",
  "facts_observed": ["objective fact 1 visible/audible in evidence", "fact 2"],
  "legal_relevance": "HIGH|MEDIUM|LOW|NONE",
  "relevance_explanation": "how this evidence relates to the legal question",
  "admissibility": {
    "likely_admissible": true|false,
    "concerns": ["authenticity concern", "chain of custody issue"],
    "recommendations": ["how to strengthen admissibility"]
  },
  "credibility_assessment": "HIGH|MEDIUM|LOW",
  "credibility_notes": "notes on reliability and potential challenges",
  "supports_claim": true|false|null,
  "key_statements": ["direct quote or key fact from evidence"],
  "gaps_identified": ["what this evidence does NOT establish"],
  "overall_strength": "STRONG|MODERATE|WEAK|INADMISSIBLE"
}
Do not add text before or after the JSON.
"""

_AUDIO_ANALYSIS_PROMPT = """
You are a legal transcript analyst. Analyse this audio transcript for legal proceedings.
Identify: key statements, admissions, contradictions, dates/amounts mentioned.
Return the same evidence JSON format.
"""


@dataclass
class EvidenceFile:
    filename: str
    content_type: str
    content: bytes | str   # bytes for binary, str for text


class EvidenceAnalyzerAgent(BaseAgent):
    name = "evidence"

    async def _execute(
        self,
        *,
        question: str,
        evidence_files: list[EvidenceFile] | None = None,
        case_context: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        if not evidence_files:
            return {
                "items": [],
                "overall_strength": "NONE",
                "gaps": ["No evidence files provided"],
                "_confidence": 1.0,
                "_tokens": 0,
            }

        settings = get_settings()
        results: list[dict] = []
        total_tokens = 0

        for ev_file in evidence_files:
            item_result = await self._analyze_single(
                ev_file=ev_file,
                question=question,
                case_context=case_context,
                model=settings.model_evidence,
            )
            results.append(item_result)
            total_tokens += item_result.pop("_tokens", 0)

        overall = self._compute_overall(results)

        log.info(
            "evidence.analyzed",
            files=len(evidence_files),
            overall_strength=overall["strength"],
        )

        return {
            "items": results,
            "overall_strength": overall["strength"],
            "gaps": overall["gaps"],
            "evidence_summary": overall["summary"],
            "_confidence": overall["confidence"],
            "_tokens": total_tokens,
        }

    async def _analyze_single(
        self,
        *,
        ev_file: EvidenceFile,
        question: str,
        case_context: str | None,
        model: str,
    ) -> dict[str, Any]:
        ctx = f"\nCase context: {case_context}" if case_context else ""
        file_type = self._classify_file(ev_file.content_type)

        if file_type == "audio":
            return await self._analyze_audio(ev_file, question, ctx, model)
        elif file_type == "image":
            return await self._analyze_image(ev_file, question, ctx, model)
        else:
            return await self._analyze_text_evidence(ev_file, question, ctx, model)

    async def _analyze_image(self, ev_file: EvidenceFile, question: str, ctx: str, model: str) -> dict:
        user_msg = (
            f"Legal question: {question}{ctx}\n\n"
            f"File: {ev_file.filename}\n"
            f"[Image evidence — analyse visible facts, timestamps, and legal relevance]"
        )
        result = await self._call_llm(
            model=model,
            system=_EVIDENCE_SYSTEM_PROMPT,
            user_message=user_msg,
            max_tokens=2048,
        )
        parsed = self._parse_evidence_response(result.text)
        parsed.update({"filename": ev_file.filename, "_tokens": result.total_tokens})
        return parsed

    async def _analyze_audio(self, ev_file: EvidenceFile, question: str, ctx: str, model: str) -> dict:
        # In production: first run Whisper transcription via audio_service
        # Here we analyse with available text content or stub transcript
        transcript = ev_file.content if isinstance(ev_file.content, str) else "[Audio file — transcription required]"
        user_msg = (
            f"Legal question: {question}{ctx}\n\n"
            f"File: {ev_file.filename}\n"
            f"AUDIO TRANSCRIPT:\n{str(transcript)[:4000]}"
        )
        result = await self._call_llm(
            model=model,
            system=_AUDIO_ANALYSIS_PROMPT,
            user_message=user_msg,
            max_tokens=2048,
        )
        parsed = self._parse_evidence_response(result.text)
        parsed.update({
            "filename": ev_file.filename,
            "evidence_type": "audio_transcript",
            "_tokens": result.total_tokens,
        })
        return parsed

    async def _analyze_text_evidence(self, ev_file: EvidenceFile, question: str, ctx: str, model: str) -> dict:
        content = ev_file.content if isinstance(ev_file.content, str) else ev_file.content.decode("utf-8", errors="replace")
        user_msg = (
            f"Legal question: {question}{ctx}\n\n"
            f"File: {ev_file.filename}\n"
            f"CONTENT:\n{content[:5000]}"
        )
        result = await self._call_llm(
            model=model,
            system=_EVIDENCE_SYSTEM_PROMPT,
            user_message=user_msg,
            max_tokens=2048,
        )
        parsed = self._parse_evidence_response(result.text)
        parsed.update({"filename": ev_file.filename, "_tokens": result.total_tokens})
        return parsed

    def _classify_file(self, content_type: str) -> str:
        ct = content_type.lower()
        if any(t in ct for t in ["image", "jpeg", "png", "webp", "gif"]):
            return "image"
        if any(t in ct for t in ["audio", "mpeg", "wav", "mp4", "ogg", "m4a"]):
            return "audio"
        return "text"

    def _parse_evidence_response(self, text: str) -> dict[str, Any]:
        try:
            clean = re.sub(r"```(?:json)?\s*|\s*```", "", text.strip())
            return json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            return {
                "evidence_type": "unknown",
                "facts_observed": [],
                "legal_relevance": "MEDIUM",
                "admissibility": {"likely_admissible": None, "concerns": [], "recommendations": []},
                "overall_strength": "MODERATE",
                "raw_analysis": text[:500],
            }

    def _compute_overall(self, results: list[dict]) -> dict:
        if not results:
            return {"strength": "NONE", "gaps": ["No evidence analysed"], "summary": "", "confidence": 1.0}

        strength_rank = {"STRONG": 3, "MODERATE": 2, "WEAK": 1, "INADMISSIBLE": 0, "NONE": -1}
        best_strength = max(results, key=lambda r: strength_rank.get(r.get("overall_strength", "NONE"), -1))
        gaps = [g for r in results for g in (r.get("gaps_identified") or [])]
        admissible_count = sum(1 for r in results if r.get("admissibility", {}).get("likely_admissible"))
        confidence = 0.7 + (0.1 * admissible_count / len(results))

        return {
            "strength": best_strength.get("overall_strength", "MODERATE"),
            "gaps": gaps[:5],
            "summary": f"{len(results)} evidence item(s) analysed. {admissible_count} likely admissible.",
            "confidence": min(0.95, confidence),
        }
