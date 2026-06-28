import asyncio
from collections.abc import AsyncIterator

from backend.app.schemas import ChatMessage
from backend.app.services import book_agent
from tests.test_book_agent_ranking import make_match


def run(coro: object) -> object:
    return asyncio.run(coro)


def messages(text: str) -> list[ChatMessage]:
    return [ChatMessage(role="user", content=text)]


def test_run_book_agent_skips_search_for_obvious_off_topic() -> None:
    def failing_search(query: str, filters: dict[str, object] | None, top_k: int) -> list[dict[str, object]]:
        raise AssertionError("search should not be called")

    answer = run(book_agent.run_book_agent(messages("hello"), search_fn=failing_search, provider=None))

    assert isinstance(answer, str)
    assert "book recommendations" in answer


def test_run_book_agent_skips_search_for_llm_extracted_off_topic() -> None:
    class OffTopicProvider:
        name = "fake"

        def complete_json(self, prompt: str) -> dict[str, object]:
            return {"intent": "off_topic"}

        def complete_text(self, prompt: str) -> str:
            return ""

    def failing_search(query: str, filters: dict[str, object] | None, top_k: int) -> list[dict[str, object]]:
        raise AssertionError("search should not be called")

    answer = run(
        book_agent.run_book_agent(
            messages("Plan my weekend itinerary"),
            search_fn=failing_search,
            provider=OffTopicProvider(),
        )
    )

    assert isinstance(answer, str)
    assert "book recommendations" in answer


def test_run_book_agent_returns_retrieved_only_recommendations() -> None:
    def fake_search(query: str, filters: dict[str, object] | None, top_k: int) -> list[dict[str, object]]:
        return [
            make_match(
                book_id="frankenstein",
                score=0.95,
                title="Frankenstein",
                authors=["Shelley, Mary Wollstonecraft"],
                languages=["en"],
                subjects=["Gothic fiction"],
                first_publish_year=1818,
                chunk_text="A gothic story about science and moral responsibility.",
            )
        ]

    answer = run(
        book_agent.run_book_agent(
            messages("Recommend an English gothic science book"),
            search_fn=fake_search,
            provider=None,
        )
    )

    assert "**Frankenstein**" in answer
    assert "Moby Dick" not in answer


def test_public_stream_yields_text_chunks_with_mocked_default_search(
    monkeypatch,
) -> None:
    def fake_default_search(query_text: str, filters: dict[str, object] | None, top_k: int) -> list[dict[str, object]]:
        return [
            make_match(
                book_id="odyssey",
                score=0.9,
                title="The Odyssey",
                authors=["Homer"],
                languages=["en"],
                subjects=["Adventure"],
                chunk_text="A classic adventure journey.",
            )
        ]

    monkeypatch.setattr(book_agent, "_default_search_books", fake_default_search)

    async def collect() -> list[str]:
        chunks: list[str] = []
        stream: AsyncIterator[str] = book_agent.stream_book_agent_response(
            messages("Recommend an adventure book")
        )
        async for chunk in stream:
            chunks.append(chunk)
        return chunks

    chunks = run(collect())

    assert chunks
    assert "The Odyssey" in "".join(chunks)
