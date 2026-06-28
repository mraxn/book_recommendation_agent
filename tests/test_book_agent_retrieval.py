import asyncio

import pytest

from backend.app.services import book_agent
from tests.test_book_agent_ranking import make_match


def run(coro: object) -> object:
    return asyncio.run(coro)


def test_build_pinecone_filter_uses_language_and_year() -> None:
    request = book_agent.ExtractedRequest(
        query="English books before 1900",
        intent="recommendation",
        language_code="en",
        year=book_agent.YearConstraint(lte=1899),
    )

    assert book_agent.build_pinecone_filter(request) == {
        "languages": {"$in": ["en"]},
        "first_publish_year": {"$lte": 1899},
    }


def test_build_initial_search_plans_adds_title_reference_before_main() -> None:
    request = book_agent.ExtractedRequest(
        query="Something like Frankenstein",
        intent="title_reference",
        title_reference="Frankenstein",
        language_code="en",
    )

    plans = book_agent.build_initial_search_plans(request)

    assert [plan.purpose for plan in plans] == ["title_reference", "main"]
    assert plans[0].query_text == "Frankenstein"
    assert plans[0].filters == {"languages": {"$in": ["en"]}}


def test_retrieve_candidates_relaxes_year_when_too_few_results() -> None:
    calls: list[tuple[str, dict[str, object] | None, int]] = []

    def fake_search(query: str, filters: dict[str, object] | None, top_k: int) -> list[dict[str, object]]:
        calls.append((query, filters, top_k))
        if len(calls) == 1:
            return []
        return [
            make_match(
                book_id="fallback",
                score=0.9,
                title="Fallback Book",
                languages=["en"],
            )
        ]

    request = book_agent.ExtractedRequest(
        query="English novels before 1800",
        intent="recommendation",
        language_code="en",
        year=book_agent.YearConstraint(lte=1799),
    )

    result = run(book_agent.retrieve_candidates_for_request(request, search_fn=fake_search))

    assert isinstance(result, book_agent.RetrievalResult)
    assert result.relaxed_year is True
    assert result.relaxed_language is False
    assert result.search_count == 2
    assert calls[0][1] == {"languages": {"$in": ["en"]}, "first_publish_year": {"$lte": 1799}}
    assert calls[1][1] == {"languages": {"$in": ["en"]}}
    assert [candidate.id for candidate in result.candidates] == ["fallback"]


def test_retrieve_candidates_relaxes_language_only_when_no_results() -> None:
    calls: list[tuple[str, dict[str, object] | None, int]] = []

    def fake_search(query: str, filters: dict[str, object] | None, top_k: int) -> list[dict[str, object]]:
        calls.append((query, filters, top_k))
        if len(calls) < 3:
            return []
        return [make_match(book_id="any-language", score=0.8, title="Any", languages=["en"])]

    request = book_agent.ExtractedRequest(
        query="French books before 1600",
        intent="recommendation",
        language_code="fr",
        year=book_agent.YearConstraint(lte=1599),
    )

    result = run(book_agent.retrieve_candidates_for_request(request, search_fn=fake_search))

    assert isinstance(result, book_agent.RetrievalResult)
    assert result.relaxed_year is True
    assert result.relaxed_language is True
    assert result.search_count == 3
    assert calls[2][1] is None
    assert result.candidates[0].id == "any-language"


def test_retrieve_candidates_respects_three_search_cap_for_reference_queries() -> None:
    calls: list[tuple[str, dict[str, object] | None, int]] = []

    def fake_search(query: str, filters: dict[str, object] | None, top_k: int) -> list[dict[str, object]]:
        calls.append((query, filters, top_k))
        return []

    request = book_agent.ExtractedRequest(
        query="Something like Frankenstein in French before 1800",
        intent="title_reference",
        title_reference="Frankenstein",
        language_code="fr",
        year=book_agent.YearConstraint(lte=1799),
    )

    result = run(book_agent.retrieve_candidates_for_request(request, search_fn=fake_search))

    assert len(calls) == 3
    assert isinstance(result, book_agent.RetrievalResult)
    assert result.candidates == []


def test_retrieve_candidates_counts_failed_attempts_toward_search_cap() -> None:
    calls: list[tuple[str, dict[str, object] | None, int]] = []

    def fake_search(query: str, filters: dict[str, object] | None, top_k: int) -> list[dict[str, object]]:
        calls.append((query, filters, top_k))
        raise RuntimeError("search failed")

    request = book_agent.ExtractedRequest(
        query="Something like Frankenstein in French before 1800",
        intent="title_reference",
        title_reference="Frankenstein",
        language_code="fr",
        year=book_agent.YearConstraint(lte=1799),
    )

    with pytest.raises(book_agent.RetrievalError):
        run(book_agent.retrieve_candidates_for_request(request, search_fn=fake_search))

    assert len(calls) == 3


def test_retrieve_candidates_rejects_malformed_search_result() -> None:
    def fake_search(query: str, filters: dict[str, object] | None, top_k: int) -> object:
        return None

    request = book_agent.ExtractedRequest(
        query="Recommend adventure books",
        intent="recommendation",
    )

    with pytest.raises(book_agent.RetrievalError, match="expected list"):
        run(book_agent.retrieve_candidates_for_request(request, search_fn=fake_search))
