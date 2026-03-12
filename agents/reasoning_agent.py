from __future__ import annotations

from backend.agents.base_agent import BaseAgent


class IracReasoningAgent(BaseAgent):
    name = "reasoning"

    async def run(self, *, question: str, research: object | None, document: object | None, evidence: object | None, memory: dict) -> dict:
        # Stub that matches architecture v2 IRAC schema shape.
        return {
            "irac": {
                "issue": {
                    "primary": "ต้องพิจารณาว่าข้อเท็จจริงเข้าหลักความรับผิดทางละเมิดหรือไม่",
                    "secondary": [],
                },
                "rule": {
                    "statutes": [
                        {
                            "name": "ประมวลกฎหมายแพ่งและพาณิชย์",
                            "section": "มาตรา 420",
                            "text": "ผู้ใดจงใจหรือประมาทเลินเล่อ ทำต่อผู้อื่นโดยผิดกฎหมายให้เขาเสียหาย...",
                            "status": "ACTIVE",
                            "year": 2535,
                        }
                    ],
                    "precedents": [],
                },
                "application": {
                    "analysis": "จากข้อเท็จจริงที่ให้มา (ยังเป็นข้อมูลเบื้องต้น) ต้องเทียบองค์ประกอบของละเมิดกับพฤติการณ์",
                    "strengths": ["มีเหตุให้เชื่อว่ามีความเสียหายเกิดขึ้นจริง"],
                    "weaknesses": ["ข้อเท็จจริงยังไม่ครบ เช่น หลักฐานและเจตนา/ความประมาท"],
                    "counter_args": ["อีกฝ่ายอาจโต้ว่าไม่มีการกระทำโดยผิดกฎหมาย หรือไม่มีความสัมพันธ์เหตุและผล"],
                    "rebuttals": ["สามารถหักล้างด้วยเอกสาร/พยานและลำดับเหตุการณ์ที่ชัดเจน"],
                },
                "conclusion": {
                    "recommendation": "โปรดให้รายละเอียดข้อเท็จจริงเพิ่มเติมและแนบหลักฐานที่เกี่ยวข้อง เพื่อประเมินอย่างเป็นระบบตาม IRAC",
                    "action_steps": ["รวบรวมเอกสารและหลักฐาน", "สรุป timeline เหตุการณ์", "ระบุความเสียหายเป็นตัวเงิน/ไม่เป็นตัวเงิน"],
                    "risk_level": "MEDIUM",
                    "win_probability": 0.72,
                    "settlement_note": "หากต้นทุนคดีสูง อาจพิจารณาเจรจาเพื่อยุติข้อพิพาท",
                },
            },
            "confidence": 0.82,
            "citations": [
                {"ref": "ประมวลกฎหมายแพ่งและพาณิชย์ มาตรา 420", "status": "UNVERIFIED"}
            ],
        }

