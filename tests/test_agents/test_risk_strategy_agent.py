from __future__ import annotations

from agents.risk_strategy_agent import RiskStrategyAgent


def test_risk_fallback_does_not_invent_generic_strategy_options():
    agent = RiskStrategyAgent()

    result = agent._parse_strategy_response(
        "not json",
        {
            "win_probability": 0.41,
            "risk_level": "HIGH",
            "action_steps": ["Preserve the contract and payment records."],
        },
        language="en",
    )

    assert result["win_probability"] == 0.41
    assert result["risk_level"] == "HIGH"
    assert result["strategic_options"] == []
    assert result["immediate_actions"] == ["Preserve the contract and payment records."]
    assert "did not invent litigation strategy" in result["win_probability_basis"]
