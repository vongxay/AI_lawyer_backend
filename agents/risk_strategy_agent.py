from __future__ import annotations

from backend.agents.base_agent import BaseAgent


class RiskStrategyAgent(BaseAgent):
    name = "risk"

    async def run(self, *, question: str, irac: dict) -> dict:
        return {
            "win_probability": irac.get("irac", {}).get("conclusion", {}).get("win_probability", 0.6),
            "risk_level": irac.get("irac", {}).get("conclusion", {}).get("risk_level", "MEDIUM"),
            "options": [
                {"name": "เจรจา", "pros": ["เร็ว", "ลดต้นทุน"], "cons": ["อาจได้ผลลัพธ์น้อยกว่า"], "eta_days": 14},
                {"name": "ดำเนินคดี", "pros": ["มีคำพิพากษา"], "cons": ["ใช้เวลา/ค่าใช้จ่าย"], "eta_days": 180},
            ],
        }

