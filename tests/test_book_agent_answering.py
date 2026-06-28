import asyncio
from typing import Any

from backend.app.services import book_agent


class FakeProvider:
    name = "fake"

    def __init__(self, *, json_payload: Any = None, text: str = "") -> None:
        self.json_payload = json_payload
        self.text = text
        self.json_calls = 0
        self.text_calls = 0

    def complete_json(self, prompt: str) -> Any:
        self.json_calls += 1
        return self.json_payload

    def complete_text(self, prompt: str) -> str:
        self.text_calls += 1
        return self.text


class FailingProvider(FakeProvider):
    def complete_json(self, prompt: str) -> Any:
        raise book_agent.ProviderError("failed json")

    def complete_text(self, prompt: str) -> str:
        raise book_agent.ProviderError("failed text")


def run(coro: object) -> object:
    return asyncio.run(coro)


def ranked(book_id: str, title: str, score: float = 1.0) -> book_agent.RankedCandidate:
    return book_agent.RankedCandidate(
        candidate=book_agent.BookCandidate(
            id=book_id,
            score=score,
            title=title,
            authors=("Author, A.",),
            first_publish_year=1900,
        ),
        rank_score=score,
        reason="matches the request",
    )


def test_maybe_llm_rerank_reorders_by_valid_ids() -> None:
    items = [ranked(str(index), f"Book {index}", score=10 - index) for index in range(5)]
    provider = FakeProvider(json_payload={"selected_ids": ["3", "1"]})
    request = book_agent.ExtractedRequest(query="books", intent="recommendation")

    reranked = run(book_agent.maybe_llm_rerank(items, request, provider))

    assert [item.candidate.id for item in reranked][:3] == ["3", "1", "0"]
    assert provider.json_calls == 1


def test_maybe_llm_rerank_skips_small_candidate_sets() -> None:
    items = [ranked("1", "Book 1"), ranked("2", "Book 2")]
    provider = FakeProvider(json_payload={"selected_ids": ["2"]})
    request = book_agent.ExtractedRequest(query="books", intent="recommendation")

    reranked = run(book_agent.maybe_llm_rerank(items, request, provider))

    assert reranked == items
    assert provider.json_calls == 0


def test_maybe_llm_rerank_falls_back_on_provider_error() -> None:
    items = [ranked(str(index), f"Book {index}", score=10 - index) for index in range(5)]
    request = book_agent.ExtractedRequest(query="books", intent="recommendation")

    reranked = run(book_agent.maybe_llm_rerank(items, request, FailingProvider()))

    assert reranked == items


def test_generate_grounded_answer_accepts_selected_titles() -> None:
    selected = [ranked("1", "Frankenstein")]
    provider = FakeProvider(
        text="Here is one retrieved match:\n\n"
        "1. **Frankenstein** by Shelley, Mary Wollstonecraft (1818) - matches gothic science."
    )
    request = book_agent.ExtractedRequest(query="gothic science", intent="recommendation")

    answer = run(book_agent.generate_grounded_answer(request, selected, provider))

    assert "**Frankenstein**" in answer
    assert provider.text_calls == 1


def test_generate_grounded_answer_rejects_unretrieved_titles() -> None:
    selected = [ranked("1", "Frankenstein")]
    provider = FakeProvider(
        text="1. **Frankenstein** by Shelley - matches.\n"
        "2. **Moby Dick** by Melville - also matches."
    )
    request = book_agent.ExtractedRequest(query="gothic science", intent="recommendation")

    answer = run(book_agent.generate_grounded_answer(request, selected, provider))

    assert "**Frankenstein**" in answer
    assert "Moby Dick" not in answer


def test_generate_grounded_answer_rejects_non_numbered_llm_output() -> None:
    selected = [ranked("1", "Frankenstein")]
    provider = FakeProvider(
        text="- **Frankenstein** by Shelley - matches gothic science."
    )
    request = book_agent.ExtractedRequest(query="gothic science", intent="recommendation")

    answer = run(book_agent.generate_grounded_answer(request, selected, provider))

    assert answer.startswith("Here are")
    assert "1. **Frankenstein**" in answer


def test_generate_grounded_answer_falls_back_when_provider_fails() -> None:
    selected = [ranked("1", "Frankenstein")]
    request = book_agent.ExtractedRequest(query="gothic science", intent="recommendation")

    answer = run(book_agent.generate_grounded_answer(request, selected, FailingProvider()))

    assert "**Frankenstein**" in answer


def test_build_deterministic_answer_mentions_relaxed_filters() -> None:
    answer = book_agent.build_deterministic_answer(
        book_agent.ExtractedRequest(query="French books before 1600", intent="recommendation"),
        [ranked("1", "A Book")],
        relaxed_year=True,
        relaxed_language=True,
    )

    assert "broadened the search" in answer
    assert "**A Book**" in answer


def test_answer_validation_requires_selected_markdown_titles() -> None:
    selected = [book_agent.BookCandidate(id="1", score=1, title="Frankenstein")]

    assert book_agent.answer_mentions_only_selected_titles("1. **Frankenstein** by Mary", selected)
    assert not book_agent.answer_mentions_only_selected_titles("1. **Dracula** by Bram", selected)
    assert not book_agent.answer_mentions_only_selected_titles("No clear title here", selected)


def test_answer_uses_numbered_list() -> None:
    assert book_agent.answer_uses_numbered_list("Intro\n1. **Book** by Author - reason.")
    assert not book_agent.answer_uses_numbered_list("- **Book** by Author - reason.")


def test_normalize_llm_answer_format_preserves_good_numbered_content() -> None:
    selected = [book_agent.BookCandidate(id="1", score=1, title="Frankenstein")]

    normalized = book_agent.normalize_llm_answer_format(
        "- 1. Frankenstein by Shelley - specific gothic reason.",
        selected,
    )

    assert normalized == "1. **Frankenstein** by Shelley - specific gothic reason."
    assert book_agent.answer_uses_numbered_list(normalized)
    assert book_agent.answer_mentions_only_selected_titles(normalized, selected)


def test_answer_prompt_explicitly_forbids_bullet_lists() -> None:
    assert 'must start with "1. "' in book_agent.ANSWER_PROMPT
    assert 'never "-"' in book_agent.ANSWER_PROMPT
    assert "Wrap every book title in **bold markdown**" in book_agent.ANSWER_PROMPT
    assert "Do not use bullet points" in book_agent.ANSWER_PROMPT
    assert 'Do not write "Reason sentence"' in book_agent.ANSWER_PROMPT
    assert "specific reason from the supplied reason/text" in book_agent.ANSWER_PROMPT
