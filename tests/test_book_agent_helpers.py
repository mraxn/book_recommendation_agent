import asyncio

import pytest

from backend.app.schemas import ChatMessage
from backend.app.services import book_agent


def user_message(content: str) -> list[ChatMessage]:
    return [ChatMessage(role="user", content=content)]


class FakeExtractionProvider:
    name = "fake"

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def complete_json(self, prompt: str) -> dict[str, object]:
        return self.payload

    def complete_text(self, prompt: str) -> str:
        return ""


def test_provider_settings_defaults_to_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [
        "LLM_PROVIDER",
        "LLM_TIMEOUT_SECONDS",
        "LLM_TEMPERATURE",
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
    ]:
        monkeypatch.delenv(name, raising=False)

    settings = book_agent.get_provider_settings()

    assert settings.provider == "ollama"
    assert settings.ollama_model == "qwen3:1.7b"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.timeout_seconds == 30.0
    assert settings.temperature == 0.2


def test_provider_settings_reads_clean_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", " OPENAI ")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.4")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")

    settings = book_agent.get_provider_settings()

    assert settings.provider == "openai"
    assert settings.timeout_seconds == 12.5
    assert settings.temperature == 0.4
    assert settings.openai_model == "gpt-test"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Recommend 5 books about revenge", 5),
        ("Suggest three novels about travel", 3),
        ("I want one book about survival", 1),
        ("What is the best book about politics?", 1),
        ("Suggest some adventure books", 5),
        ("Recommend something gothic", 3),
    ],
)
def test_parse_requested_count(query: str, expected: int) -> None:
    assert book_agent.parse_requested_count(query) == expected


@pytest.mark.parametrize(
    ("query", "expected_code", "expected_name"),
    [
        ("Find English books about ghosts", "en", "english"),
        ("Find books about political revolution in French", "fr", "french"),
        ("Suggest novels written in Spanish", "es", "spanish"),
        ("Recommend French revolution books", None, None),
    ],
)
def test_detect_language(query: str, expected_code: str | None, expected_name: str | None) -> None:
    assert book_agent.detect_language(query) == (expected_code, expected_name)


def test_parse_year_constraint_before_after_and_between() -> None:
    assert book_agent.parse_year_constraint("written before 1900") == book_agent.YearConstraint(
        lte=1899
    )
    assert book_agent.parse_year_constraint("published after 1850") == book_agent.YearConstraint(
        gte=1851
    )
    assert book_agent.parse_year_constraint("between 1800 and 1900") == book_agent.YearConstraint(
        gte=1800, lte=1900
    )


@pytest.mark.parametrize(
    ("query", "expected_intent", "expected_title"),
    [
        ("I want something like Frankenstein", "title_reference", "Frankenstein"),
        (
            'Suggest a book for someone who loved "The Count of Monte Cristo"',
            "title_reference",
            "The Count of Monte Cristo",
        ),
        ("Find Frankenstein", "title_lookup", "Frankenstein"),
    ],
)
def test_detect_title_reference(query: str, expected_intent: str, expected_title: str) -> None:
    assert book_agent.detect_title_reference(query) == (expected_intent, expected_title)


def test_parse_json_payload_strips_thinking_and_code_fences() -> None:
    text = '<think>hidden</think>\n```json\n{"intent": "recommendation"}\n```'

    assert book_agent.parse_json_payload(text) == {"intent": "recommendation"}


def test_parse_json_payload_finds_balanced_object() -> None:
    text = 'Here is the result: {"selected_ids": ["a", "b"]} done.'

    assert book_agent.parse_json_payload(text) == {"selected_ids": ["a", "b"]}


def test_recent_dialogue_uses_last_six_user_assistant_messages() -> None:
    messages = [
        ChatMessage(role="system", content="ignore"),
        *[ChatMessage(role="user", content=f"user {index}") for index in range(7)],
    ]

    recent = book_agent.recent_dialogue(messages)

    assert len(recent) == 6
    assert recent[0].content == "user 1"
    assert all(message.role != "system" for message in recent)


def test_chunk_text_preserves_content_without_empty_chunks() -> None:
    text = "One two three four five six seven"

    chunks = book_agent.chunk_text(text, words_per_chunk=3)

    assert chunks
    assert "".join(chunks) == text
    assert all(chunks)


def test_heuristic_extract_request_combines_structured_signals() -> None:
    request = book_agent.heuristic_extract_request(
        user_message("Suggest 5 popular English books about ghosts before 1900")
    )

    assert request.intent == "recommendation"
    assert request.requested_count == 5
    assert request.language_code == "en"
    assert request.year == book_agent.YearConstraint(lte=1899)
    assert request.wants_popular is True
    assert "ghosts" in request.topics


def test_heuristic_extract_request_detects_author_lookup() -> None:
    request = book_agent.heuristic_extract_request(user_message("What did Mark Twain write about travel?"))

    assert request.intent == "author_lookup"
    assert request.author == "Mark Twain"
    assert "travel" in request.topics


def test_merge_llm_extraction_keeps_deterministic_language_and_year() -> None:
    base = book_agent.heuristic_extract_request(user_message("Find English books before 1900"))

    merged = book_agent.merge_llm_extraction(
        base,
        {
            "language_code": "fr",
            "language_name": "french",
            "year_gte": 2000,
            "topics": ["ghosts"],
        },
    )

    assert merged.language_code == "en"
    assert merged.year == book_agent.YearConstraint(lte=1899)
    assert "ghosts" in merged.topics


def test_extract_request_with_provider_fills_gaps() -> None:
    provider = FakeExtractionProvider(
        {
            "intent": "author_lookup",
            "author": "Mark Twain",
            "topics": ["travel"],
            "requested_count": 2,
        }
    )

    request = asyncio.run(
        book_agent.extract_request_with_provider(
            user_message("What did Twain write?"),
            provider,
        )
    )

    assert request.intent == "author_lookup"
    assert request.author == "Mark Twain"
    assert request.requested_count == 2
    assert "travel" in request.topics
