from memory.case_facts import merge_case_facts


def test_merge_case_facts_appends_question_and_analysis() -> None:
    summary = merge_case_facts(
        "Issue: land dispute",
        question="Can I lease my land for 30 years?",
        material_facts=["User holds a land use right certificate"],
        legal_issues=["lease duration limits"],
        issue_primary="Whether a 30-year lease is permitted",
    )

    assert "land dispute" in summary
    assert "Can I lease my land for 30 years?" in summary
    assert "land use right certificate" in summary
    assert "lease duration limits" in summary
    assert "30-year lease is permitted" in summary


def test_merge_case_facts_deduplicates_segments() -> None:
    first = merge_case_facts(None, question="Same question")
    second = merge_case_facts(first, question="Same question")
    assert first == second
