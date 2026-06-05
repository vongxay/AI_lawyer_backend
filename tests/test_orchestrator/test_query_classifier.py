from __future__ import annotations

import pytest

from orchestrator.query_classifier import QueryClassifier


@pytest.mark.asyncio
async def test_classifier_routes_lao_greeting_to_conversation():
    classifier = QueryClassifier()

    result = await classifier.classify("\u0eaa\u0eb0\u0e9a\u0eb2\u0e8d\u0e94\u0eb5")

    assert result == "conversation"


@pytest.mark.asyncio
async def test_classifier_routes_thai_greeting_to_conversation():
    classifier = QueryClassifier()

    result = await classifier.classify("\u0e2a\u0e1a\u0e32\u0e22\u0e14\u0e35\u0e44\u0e2b\u0e21")

    assert result == "conversation"


@pytest.mark.asyncio
async def test_classifier_keeps_land_right_question_legal():
    classifier = QueryClassifier()

    result = await classifier.classify(
        "\u0e9c\u0eb9\u0ec9\u0ec4\u0e94\u0ec9\u0eae\u0eb1\u0e9a\u0eaa\u0eb4\u0e94"
        "\u0e99\u0eb3\u0ec3\u0e8a\u0ec9\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99 "
        "\u0ec4\u0e94\u0ec9\u0eae\u0eb1\u0e9a\u0e81\u0eb2\u0e99\u0e9b\u0ebb\u0e81"
        "\u0e9b\u0ec9\u0ead\u0e87\u0eaa\u0eb4\u0e94\u0ec3\u0e94\u0ec1\u0e94\u0ec8"
    )

    assert result == "legal_question"
