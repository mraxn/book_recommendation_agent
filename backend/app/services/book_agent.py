from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol, TypedDict, cast

from langgraph.graph import END, StateGraph

from backend.app.schemas import ChatMessage

DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:1.7b"
DEFAULT_LLM_TIMEOUT_SECONDS = 30.0
DEFAULT_LLM_TEMPERATURE = 0.2

SUPPORTED_PROVIDERS = {"ollama", "openai", "anthropic"}
MAX_HISTORY_MESSAGES = 6
DEFAULT_MAX_TOKENS = 900
DEFAULT_SEARCH_TOP_K = 20
MAX_SEARCHES_PER_REQUEST = 3
MIN_CANDIDATES_BEFORE_RELAX = 3

LANGUAGE_NAME_TO_CODE = {
    "english": "en",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "italian": "it",
    "portuguese": "pt",
    "dutch": "nl",
    "latin": "la",
    "greek": "el",
    "finnish": "fi",
    "swedish": "sv",
}
LANGUAGE_CODE_TO_NAME = {code: name.title() for name, code in LANGUAGE_NAME_TO_CODE.items()}

THEME_SYNONYMS = {
    "gothic": {"gothic", "supernatural", "horror", "ghost", "ghosts"},
    "horror": {"horror", "ghost", "ghosts", "supernatural", "gothic"},
    "revenge": {"revenge", "vengeance", "retaliation"},
    "philosophy": {"philosophy", "philosophical", "ethics", "moral"},
    "adventure": {"adventure", "journey", "quest", "voyage"},
    "political": {"political", "politics", "revolution", "revolutionary"},
    "travel": {"travel", "travels", "journey", "voyage"},
}

RERANK_PROMPT = """Rank these retrieved books for the user request.
Return JSON only: {{"selected_ids": ["id1", "id2"]}}.
Use only candidate IDs from the list.

User request: {query}

Candidates:
{candidates}
"""

EXTRACTION_PROMPT = """Extract book-search intent from the recent conversation.
Return JSON only with these fields:
intent, author, title, title_reference, language_code, language_name,
year_gte, year_lte, topics, requested_count, wants_popular, is_broad.

Valid intents: recommendation, author_lookup, title_lookup, title_reference, follow_up, off_topic.
Use null for unknown scalar fields and [] for no topics.

Recent conversation:
{conversation}

Latest user request: {query}
"""

ANSWER_PROMPT = """Write a concise book recommendation answer in markdown.
Use only the supplied retrieved books. Do not mention any other book title.

Required format:
- Start with this exact opening sentence, unchanged:
  {opening_sentence}
- Then write a numbered markdown list only.
- Every recommendation line must start with "1. ", "2. ", "3. ", etc.
- The first character of each recommendation line must be the number, never "-".
- Wrap every book title in **bold markdown** exactly as supplied.
- Do not use bullet points, hyphen lists, tables, or headings.
- Do not add any text after the numbered list.
- Use each supplied book at most once. Do not repeat a title.
- Each item must follow this pattern:
  N. **exact supplied title** by supplied author (year if known) - specific reason from the supplied reason/text.
- Do not copy template words or use generic placeholder reasons.
- Reasons must be concrete and based on the supplied retrieved-book data.
- If the user asks for books like one they already loved, recommend alternatives only.
- Do not recommend or name the user's source/reference book unless it is one of the supplied retrieved books.
- The opening sentence must not include any book title.

Do not write "Reason sentence" or similar placeholder text.
Do not write "The user is looking for" or describe the task.

User preference context: {query_context}

Retrieved books:
{books}
"""

_PROVIDER_UNSET = object()

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}

Intent = Literal[
    "recommendation",
    "author_lookup",
    "title_lookup",
    "title_reference",
    "follow_up",
    "off_topic",
]
VALID_INTENTS = {
    "recommendation",
    "author_lookup",
    "title_lookup",
    "title_reference",
    "follow_up",
    "off_topic",
}


@dataclass(frozen=True)
class ProviderSettings:
    provider: str = DEFAULT_LLM_PROVIDER
    timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS
    temperature: float = DEFAULT_LLM_TEMPERATURE
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    openai_api_key: str | None = None
    openai_model: str | None = None
    openai_base_url: str | None = None
    anthropic_api_key: str | None = None
    anthropic_model: str | None = None


@dataclass(frozen=True)
class YearConstraint:
    gte: int | None = None
    lte: int | None = None


@dataclass(frozen=True)
class ExtractedRequest:
    query: str
    intent: Intent
    author: str | None = None
    title: str | None = None
    title_reference: str | None = None
    language_code: str | None = None
    language_name: str | None = None
    year: YearConstraint | None = None
    topics: tuple[str, ...] = field(default_factory=tuple)
    requested_count: int = 3
    wants_popular: bool = False
    is_broad: bool = False


@dataclass(frozen=True)
class BookCandidate:
    id: str
    score: float
    title: str
    authors: tuple[str, ...] = field(default_factory=tuple)
    languages: tuple[str, ...] = field(default_factory=tuple)
    subjects: tuple[str, ...] = field(default_factory=tuple)
    bookshelves: tuple[str, ...] = field(default_factory=tuple)
    download_count: float = 0.0
    first_publish_year: int | None = None
    chunk_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RankedCandidate:
    candidate: BookCandidate
    rank_score: float
    reason: str


@dataclass(frozen=True)
class SearchPlan:
    query_text: str
    filters: dict[str, Any] | None = None
    top_k: int = DEFAULT_SEARCH_TOP_K
    purpose: str = "main"
    relaxed_year: bool = False
    relaxed_language: bool = False


@dataclass(frozen=True)
class RetrievalResult:
    candidates: list[BookCandidate]
    relaxed_year: bool = False
    relaxed_language: bool = False
    search_count: int = 0
    errors: tuple[str, ...] = field(default_factory=tuple)


class AgentState(TypedDict, total=False):
    messages: list[ChatMessage]
    request: ExtractedRequest
    provider: LLMProvider | None
    search_fn: Any
    retrieval: RetrievalResult
    ranked_candidates: list[RankedCandidate]
    selected: list[RankedCandidate]
    final_answer: str


class ProviderError(RuntimeError):
    """Base error for provider setup and completion failures."""


class ProviderConfigurationError(ProviderError):
    """Raised when a selected provider is missing required configuration."""


class RetrievalError(RuntimeError):
    """Raised when Pinecone retrieval cannot complete."""


class LLMProvider(Protocol):
    name: str

    def complete_text(self, prompt: str) -> str:
        """Return a text completion for the prompt."""

    def complete_json(self, prompt: str) -> Any:
        """Return a parsed JSON completion for the prompt."""


logger = logging.getLogger(__name__)


def configure_book_agent_logging(level_name: str | None = None) -> None:
    resolved_level = level_name or os.getenv("BOOK_AGENT_LOG_LEVEL") or os.getenv("LOG_LEVEL") or "INFO"
    level = getattr(logging, resolved_level.upper(), logging.INFO)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
        logger.addHandler(handler)

    logger.propagate = False
    logger.info("Book agent logging configured level=%s", logging.getLevelName(level))


@dataclass(frozen=True)
class OllamaProvider:
    settings: ProviderSettings
    name: str = "ollama"

    def complete_text(self, prompt: str) -> str:
        return self._chat(prompt=prompt, json_mode=False)

    def complete_json(self, prompt: str) -> Any:
        return parse_json_payload(self._chat(prompt=prompt, json_mode=True))

    def _chat(self, prompt: str, json_mode: bool) -> str:
        try:
            from ollama import Client
        except ImportError as exc:
            raise ProviderConfigurationError(
                "Ollama provider selected, but the 'ollama' package is not installed."
            ) from exc

        try:
            client = Client(host=self.settings.ollama_base_url, timeout=self.settings.timeout_seconds)
        except TypeError:
            client = Client(host=self.settings.ollama_base_url)

        kwargs: dict[str, Any] = {
            "model": self.settings.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "think": False,
            "options": {
                "temperature": self.settings.temperature,
                "num_predict": DEFAULT_MAX_TOKENS,
            },
        }
        if json_mode:
            kwargs["format"] = "json"

        try:
            response = client.chat(**kwargs)
        except Exception as exc:
            raise ProviderError(f"Ollama completion failed: {exc}") from exc
        return _extract_ollama_content(response)


@dataclass(frozen=True)
class OpenAIProvider:
    settings: ProviderSettings
    name: str = "openai"

    def __post_init__(self) -> None:
        if not self.settings.openai_api_key:
            raise ProviderConfigurationError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
        if not self.settings.openai_model:
            raise ProviderConfigurationError("OPENAI_MODEL is required when LLM_PROVIDER=openai.")

    def complete_text(self, prompt: str) -> str:
        return self._chat(prompt=prompt, json_mode=False)

    def complete_json(self, prompt: str) -> Any:
        return parse_json_payload(self._chat(prompt=prompt, json_mode=True))

    def _chat(self, prompt: str, json_mode: bool) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderConfigurationError(
                "OpenAI provider selected, but the 'openai' package is not installed."
            ) from exc

        kwargs: dict[str, Any] = {
            "api_key": self.settings.openai_api_key,
            "timeout": self.settings.timeout_seconds,
        }
        if self.settings.openai_base_url:
            kwargs["base_url"] = self.settings.openai_base_url
        client = OpenAI(**kwargs)

        completion_kwargs: dict[str, Any] = {
            "model": self.settings.openai_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.settings.temperature,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }
        if json_mode:
            completion_kwargs["response_format"] = {"type": "json_object"}

        try:
            response = client.chat.completions.create(**completion_kwargs)
        except Exception as exc:
            raise ProviderError(f"OpenAI completion failed: {exc}") from exc
        return response.choices[0].message.content or ""


@dataclass(frozen=True)
class AnthropicProvider:
    settings: ProviderSettings
    name: str = "anthropic"

    def __post_init__(self) -> None:
        if not self.settings.anthropic_api_key:
            raise ProviderConfigurationError(
                "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic."
            )
        if not self.settings.anthropic_model:
            raise ProviderConfigurationError("ANTHROPIC_MODEL is required when LLM_PROVIDER=anthropic.")

    def complete_text(self, prompt: str) -> str:
        return self._chat(prompt=prompt, json_mode=False)

    def complete_json(self, prompt: str) -> Any:
        return parse_json_payload(self._chat(prompt=prompt, json_mode=True))

    def _chat(self, prompt: str, json_mode: bool) -> str:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ProviderConfigurationError(
                "Anthropic provider selected, but the 'anthropic' package is not installed."
            ) from exc

        client = Anthropic(api_key=self.settings.anthropic_api_key, timeout=self.settings.timeout_seconds)
        system_prompt = (
            "Return only valid JSON. Do not include markdown fences."
            if json_mode
            else "Write concise, grounded book recommendation text."
        )
        try:
            response = client.messages.create(
                model=self.settings.anthropic_model,
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=self.settings.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise ProviderError(f"Anthropic completion failed: {exc}") from exc
        return _extract_anthropic_content(response)


def build_llm_provider(settings: ProviderSettings | None = None) -> LLMProvider:
    provider_settings = settings or get_provider_settings()
    logger.info(
        "Configuring LLM provider provider=%s timeout=%.1fs temperature=%.2f",
        provider_settings.provider,
        provider_settings.timeout_seconds,
        provider_settings.temperature,
    )
    if provider_settings.provider == "ollama":
        logger.info(
            "Using Ollama model=%s base_url=%s",
            provider_settings.ollama_model,
            provider_settings.ollama_base_url,
        )
    elif provider_settings.provider == "openai":
        logger.info("Using OpenAI model=%s", provider_settings.openai_model)
    elif provider_settings.provider == "anthropic":
        logger.info("Using Anthropic model=%s", provider_settings.anthropic_model)

    if provider_settings.provider == "openai":
        return OpenAIProvider(provider_settings)
    if provider_settings.provider == "anthropic":
        return AnthropicProvider(provider_settings)
    return OllamaProvider(provider_settings)


BOOK_AGENT_GRAPH = None


def get_provider_settings() -> ProviderSettings:
    provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        provider = DEFAULT_LLM_PROVIDER

    return ProviderSettings(
        provider=provider,
        timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", DEFAULT_LLM_TIMEOUT_SECONDS),
        temperature=_env_float("LLM_TEMPERATURE", DEFAULT_LLM_TEMPERATURE),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip()
        or DEFAULT_OLLAMA_BASE_URL,
        ollama_model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()
        or DEFAULT_OLLAMA_MODEL,
        openai_api_key=_clean_env("OPENAI_API_KEY"),
        openai_model=_clean_env("OPENAI_MODEL"),
        openai_base_url=_clean_env("OPENAI_BASE_URL"),
        anthropic_api_key=_clean_env("ANTHROPIC_API_KEY"),
        anthropic_model=_clean_env("ANTHROPIC_MODEL"),
    )


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def normalize_title(value: str) -> str:
    normalized = normalize_text(re.sub(r"[^\w\s]", " ", value))
    return re.sub(r"^(the|a|an)\s+", "", normalized)


def recent_dialogue(messages: list[ChatMessage], limit: int = MAX_HISTORY_MESSAGES) -> list[ChatMessage]:
    dialogue = [message for message in messages if message.role in {"user", "assistant"}]
    return dialogue[-limit:]


def chunk_text(text: str, words_per_chunk: int = 24) -> list[str]:
    if not text:
        return []
    if words_per_chunk <= 0:
        raise ValueError("words_per_chunk must be positive")

    chunks: list[str] = []
    current_words: list[str] = []
    for line_index, line in enumerate(text.splitlines(keepends=True)):
        words = re.findall(r"\S+\s*", line)
        if not words and line_index > 0:
            chunks.append(line)
            continue
        for word in words:
            current_words.append(word)
            if len(current_words) >= words_per_chunk:
                chunks.append("".join(current_words))
                current_words = []
    if current_words:
        chunks.append("".join(current_words))
    return [chunk for chunk in chunks if chunk]


def strip_model_artifacts(text: str) -> str:
    without_thinking = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    fenced = re.search(r"```(?:json)?\s*(.*?)```", without_thinking, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return without_thinking.strip()


def parse_json_payload(text: str) -> Any:
    cleaned = strip_model_artifacts(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        payload = _first_balanced_json(cleaned)
        if payload is None:
            raise
        return json.loads(payload)


def parse_requested_count(text: str) -> int:
    normalized = normalize_text(text)
    count_target = r"(?:\s+[a-z][a-z'-]*){0,3}\s+(?:books?|novels?|recommendations?)\b"
    match = re.search(rf"\b([1-5])\b{count_target}", normalized)
    if match:
        return int(match.group(1))

    for word, count in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b{count_target}", normalized):
            return count

    if re.search(r"\b(?:one|a)\s+(?:book|novel|recommendation)\b", normalized):
        return 1
    if re.search(r"\b(?:only|just)\s+one\b", normalized):
        return 1
    if re.search(r"\b(?:best|single)\s+(?:book|novel|recommendation)\b", normalized):
        return 1
    if re.search(r"\b(?:some|several|list of)\b", normalized):
        return 5
    return 3


def detect_language(text: str) -> tuple[str | None, str | None]:
    normalized = normalize_text(text)
    for name, code in LANGUAGE_NAME_TO_CODE.items():
        language_patterns = [
            rf"\bin {name}\b",
            rf"\bwritten in {name}\b",
            rf"\b{name}[- ]language\b",
            rf"\b{name}\s+(?:books?|novels?|literature)\b",
        ]
        if any(re.search(pattern, normalized) for pattern in language_patterns):
            return code, name
    return None, None


def parse_year_constraint(text: str) -> YearConstraint | None:
    normalized = normalize_text(text)
    between = re.search(r"\bbetween\s+(\d{3,4})\s+(?:and|-)\s+(\d{3,4})\b", normalized)
    if between:
        start, end = sorted((int(between.group(1)), int(between.group(2))))
        return YearConstraint(gte=start, lte=end)

    before = re.search(r"\b(?:before|pre-?|prior to|earlier than)\s+(\d{3,4})\b", normalized)
    if before:
        return YearConstraint(lte=int(before.group(1)) - 1)

    after = re.search(r"\b(?:after|post-?|later than|since)\s+(\d{3,4})\b", normalized)
    if after:
        return YearConstraint(gte=int(after.group(1)) + 1)

    century = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)\s+century\b", normalized)
    if century:
        century_number = int(century.group(1))
        return YearConstraint(gte=(century_number - 1) * 100 + 1, lte=century_number * 100)
    return None


def detect_title_reference(text: str) -> tuple[str | None, str | None]:
    stripped = text.strip()
    quoted = re.search(r'"([^"]{2,120})"', stripped)
    normalized = normalize_text(stripped)

    reference_patterns = [
        r"\blike\s+(.+)$",
        r"\bsimilar to\s+(.+)$",
        r"\bloved\s+(.+)$",
        r"\bfor someone who loved\s+(.+)$",
    ]
    for pattern in reference_patterns:
        match = re.search(pattern, stripped, flags=re.IGNORECASE)
        if match:
            title = _clean_title_phrase(quoted.group(1) if quoted else match.group(1))
            if title:
                return "title_reference", title

    lookup_patterns = [
        r"^(?:find|search for|show me|get)\s+(.+)$",
        r"^(?:what is|tell me about)\s+(.+)$",
    ]
    for pattern in lookup_patterns:
        match = re.search(pattern, stripped, flags=re.IGNORECASE)
        if match and not re.search(r"\b(?:books?|novels?|by|about)\b", normalized):
            title = _clean_title_phrase(quoted.group(1) if quoted else match.group(1))
            if title:
                return "title_lookup", title
    return None, None


def detect_author_query(text: str) -> str | None:
    patterns = [
        r"\bby\s+(.+?)(?:\s+about|\s+in\s+[A-Z]|\s+before|\s+after|\s+between|$)",
        r"\bwhat did\s+(.+?)\s+write\b",
        r"\bbooks?\s+from\s+(.+?)(?:\s+about|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            author = re.sub(r"\s+", " ", match.group(1)).strip(" .?!'\"")
            return author or None
    return None


def is_obvious_off_topic(text: str) -> bool:
    normalized = normalize_text(text)
    if normalized in {"hi", "hello", "hey", "thanks", "thank you"}:
        return True

    bookish_terms = {
        "book",
        "books",
        "novel",
        "novels",
        "author",
        "read",
        "recommend",
        "recommendation",
        "story",
        "stories",
        "shakespeare",
        "twain",
        "frankenstein",
    }
    if any(term in normalized.split() for term in bookish_terms):
        return False

    if re.search(r"\b(?:what is|how does|explain)\s+(?:pinecone|fastapi|python|llm|ai)\b", normalized):
        return True
    return False


def heuristic_extract_request(messages: list[ChatMessage]) -> ExtractedRequest:
    last_user_message = next(
        (message.content for message in reversed(messages) if message.role == "user"),
        "",
    )
    language_code, language_name = detect_language(last_user_message)
    reference_intent, detected_title = detect_title_reference(last_user_message)
    author = detect_author_query(last_user_message)
    normalized = normalize_text(last_user_message)
    inherited_title_reference = _last_title_reference_from_history(messages)
    has_follow_up_pointer = bool(
        re.search(
            r"\b(?:from those|from these|of those|of these|the first one|the second one|the third one|those|these)\b",
            normalized,
        )
    )

    if is_obvious_off_topic(last_user_message):
        intent: Intent = "off_topic"
    elif has_follow_up_pointer:
        intent = "follow_up"
    elif reference_intent == "title_reference":
        intent = "title_reference"
    elif reference_intent == "title_lookup":
        intent = "title_lookup"
    elif author:
        intent = "author_lookup"
    elif re.search(r"\b(?:more like|another|similar)\b", normalized):
        intent = "follow_up"
    else:
        intent = "recommendation"

    topics = _extract_topic_terms(last_user_message)
    if detected_title and intent == "title_reference":
        topics = _remove_reference_terms_from_topics(topics, detected_title)
    if intent == "follow_up" and inherited_title_reference:
        topics = _remove_reference_terms_from_topics(topics, inherited_title_reference)
    title_reference = detected_title if intent == "title_reference" else None
    if intent == "follow_up" and inherited_title_reference:
        title_reference = inherited_title_reference
    return ExtractedRequest(
        query=last_user_message,
        intent=intent,
        author=author,
        title=detected_title if intent == "title_lookup" else None,
        title_reference=title_reference,
        language_code=language_code,
        language_name=language_name,
        year=parse_year_constraint(last_user_message),
        topics=tuple(topics),
        requested_count=parse_requested_count(last_user_message),
        wants_popular=bool(re.search(r"\b(?:popular|best|well known|famous)\b", normalized)),
        is_broad=bool(re.search(r"\b(?:some|several|list|books)\b", normalized)),
    )


async def extract_request_with_provider(
    messages: list[ChatMessage],
    provider: LLMProvider | None,
    *,
    base_request: ExtractedRequest | None = None,
) -> ExtractedRequest:
    base = base_request or heuristic_extract_request(messages)
    logger.info(
        "Heuristic extraction intent=%s count=%s language=%s year=%s title_ref=%s query=%s",
        base.intent,
        base.requested_count,
        base.language_code or "-",
        _year_summary(base.year),
        bool(base.title_reference),
        _query_preview(base.query),
    )
    logger.debug(
        "Heuristic extraction details author=%s title=%s title_reference=%s topics=%s popular=%s broad=%s",
        base.author,
        base.title,
        base.title_reference,
        base.topics,
        base.wants_popular,
        base.is_broad,
    )
    if provider is None or base.intent == "off_topic":
        logger.debug(
            "Skipping LLM extraction provider_available=%s intent=%s",
            provider is not None,
            base.intent,
        )
        return base

    prompt = EXTRACTION_PROMPT.format(
        conversation=_format_recent_conversation(messages),
        query=base.query,
    )
    try:
        start = time.perf_counter()
        payload = await asyncio.to_thread(provider.complete_json, prompt)
        logger.debug(
            "LLM extraction completed provider=%s elapsed_ms=%d",
            getattr(provider, "name", "unknown"),
            _elapsed_ms(start),
        )
    except Exception as exc:
        logger.warning("LLM extraction failed with %s: %s", getattr(provider, "name", "unknown"), exc)
        return base
    merged = merge_llm_extraction(base, payload)
    logger.info(
        "Merged extraction intent=%s count=%s language=%s year=%s title_ref=%s topics=%d",
        merged.intent,
        merged.requested_count,
        merged.language_code or "-",
        _year_summary(merged.year),
        bool(merged.title_reference),
        len(merged.topics),
    )
    logger.debug(
        "Merged extraction details author=%s title=%s title_reference=%s topics=%s",
        merged.author,
        merged.title,
        merged.title_reference,
        merged.topics,
    )
    return merged


def merge_llm_extraction(base: ExtractedRequest, payload: Any) -> ExtractedRequest:
    if not isinstance(payload, dict):
        return base

    intent = str(payload.get("intent") or base.intent)
    if intent not in VALID_INTENTS:
        intent = base.intent

    language_code = base.language_code
    language_name = base.language_name
    if language_code is None:
        language_code, language_name = _language_from_llm_payload(payload)

    year = base.year
    if year is None:
        year = _year_from_llm_payload(payload)

    topics = _merge_topics(base.topics, payload.get("topics"))
    title_reference = _clean_title_phrase(_clean_optional_string(payload.get("title_reference")) or "")
    effective_title_reference = title_reference or base.title_reference
    if effective_title_reference:
        topics = tuple(_remove_reference_terms_from_topics(list(topics), effective_title_reference))

    typed_intent = cast(Intent, intent)
    return replace(
        base,
        intent=typed_intent,
        author=_clean_optional_string(payload.get("author")) or base.author,
        title=_clean_optional_string(payload.get("title")) or base.title,
        title_reference=effective_title_reference,
        language_code=language_code,
        language_name=language_name,
        year=year,
        topics=topics,
        requested_count=base.requested_count,
        wants_popular=base.wants_popular or bool(payload.get("wants_popular")),
        is_broad=base.is_broad or bool(payload.get("is_broad")),
    )


def candidate_from_match(match: dict[str, Any]) -> BookCandidate | None:
    metadata = match.get("metadata") if isinstance(match.get("metadata"), dict) else {}
    title = str(metadata.get("title") or "").strip()
    candidate_id = str(match.get("id") or "").strip()
    if not title or not candidate_id:
        return None

    return BookCandidate(
        id=candidate_id,
        score=_safe_float(match.get("score")),
        title=title,
        authors=_safe_str_tuple(metadata.get("authors")),
        languages=_safe_str_tuple(metadata.get("languages")),
        subjects=_safe_str_tuple(metadata.get("subjects")),
        bookshelves=_safe_str_tuple(metadata.get("bookshelves")),
        download_count=_safe_float(metadata.get("download_count")),
        first_publish_year=_safe_int(metadata.get("first_publish_year")),
        chunk_text=str(metadata.get("chunk_text") or ""),
        metadata=dict(metadata),
    )


def candidates_from_matches(matches: list[dict[str, Any]]) -> list[BookCandidate]:
    candidates = [candidate_from_match(match) for match in matches]
    return [candidate for candidate in candidates if candidate is not None]


def deduplicate_candidates(candidates: list[BookCandidate]) -> list[BookCandidate]:
    by_key: dict[tuple[str, str], BookCandidate] = {}
    order: list[tuple[str, str]] = []
    for candidate in candidates:
        key = (normalize_title(candidate.title), normalize_text(", ".join(candidate.authors)))
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = candidate
            order.append(key)
        elif candidate.score > existing.score:
            by_key[key] = candidate
    return [by_key[key] for key in order]


def filter_candidates_by_relative_score(
    candidates: list[BookCandidate],
    threshold: float = 0.72,
    min_keep: int = 3,
) -> list[BookCandidate]:
    if not candidates:
        return []
    sorted_candidates = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
    top_score = sorted_candidates[0].score
    if top_score <= 0:
        return sorted_candidates[:min_keep]

    strong = [candidate for candidate in sorted_candidates if candidate.score / top_score >= threshold]
    if len(strong) >= min_keep:
        return strong
    return sorted_candidates[: min(min_keep, len(sorted_candidates))]


def rank_candidates(
    candidates: list[BookCandidate],
    request: ExtractedRequest,
    enforce_hard_filters: bool = True,
) -> list[RankedCandidate]:
    deduped = deduplicate_candidates(candidates)
    logger.info(
        "Ranking candidates received=%d deduped=%d enforce_hard_filters=%s",
        len(candidates),
        len(deduped),
        enforce_hard_filters,
    )
    if request.title_reference:
        before_reference_filter = len(deduped)
        deduped = [
            candidate
            for candidate in deduped
            if not _is_reference_title_candidate(candidate.title, request.title_reference)
        ]
        logger.info(
            "Applied title-reference exclusion title_ref=%s removed=%d remaining=%d",
            request.title_reference,
            before_reference_filter - len(deduped),
            len(deduped),
        )
    if enforce_hard_filters:
        before_hard_filters = len(deduped)
        deduped = [candidate for candidate in deduped if candidate_matches_request(candidate, request)]
        logger.info(
            "Applied hard filters removed=%d remaining=%d",
            before_hard_filters - len(deduped),
            len(deduped),
        )
    filtered = filter_candidates_by_relative_score(deduped)
    if not filtered:
        logger.info("No candidates remained after relative score filtering")
        return []

    top_score = max((candidate.score for candidate in filtered), default=0.0)
    ranked = [_score_candidate(candidate, request, top_score) for candidate in filtered]
    sorted_ranked = sorted(ranked, key=lambda item: item.rank_score, reverse=True)
    logger.info(
        "Ranked candidates filtered=%d top_titles=%s",
        len(sorted_ranked),
        _ranked_title_summary(sorted_ranked),
    )
    logger.debug(
        "Ranked candidate scores=%s",
        [(item.candidate.id, round(item.rank_score, 3), item.reason) for item in sorted_ranked[:10]],
    )
    return sorted_ranked


def select_recommendations(
    ranked_candidates: list[RankedCandidate],
    requested_count: int,
) -> list[RankedCandidate]:
    count = max(1, min(5, requested_count))
    selected = ranked_candidates[:count]
    logger.info(
        "Selected recommendations requested=%d selected=%d titles=%s",
        requested_count,
        len(selected),
        _ranked_title_summary(selected),
    )
    return selected


def candidate_matches_request(candidate: BookCandidate, request: ExtractedRequest) -> bool:
    if request.language_code and request.language_code not in candidate.languages:
        return False
    if request.year:
        if candidate.first_publish_year is None:
            return False
        if request.year.gte is not None and candidate.first_publish_year < request.year.gte:
            return False
        if request.year.lte is not None and candidate.first_publish_year > request.year.lte:
            return False
    return True


def format_authors(authors: tuple[str, ...]) -> str:
    if not authors:
        return "Unknown author"
    if len(authors) <= 2:
        return ", ".join(authors)
    return f"{authors[0]}, {authors[1]}, et al."


def format_year(year: int | None) -> str:
    return str(year) if year is not None else ""


def language_display(code: str) -> str:
    return LANGUAGE_CODE_TO_NAME.get(code, code)


async def maybe_llm_rerank(
    ranked_candidates: list[RankedCandidate],
    request: ExtractedRequest,
    provider: LLMProvider | None,
) -> list[RankedCandidate]:
    if provider is None:
        logger.debug("Skipping LLM rerank because provider is unavailable")
        return ranked_candidates
    if len(ranked_candidates) < 5:
        logger.debug("Skipping LLM rerank candidate_count=%d", len(ranked_candidates))
        return ranked_candidates

    candidates_for_prompt = ranked_candidates[:10]
    logger.info(
        "Starting LLM rerank provider=%s candidate_count=%d prompt_candidates=%d",
        getattr(provider, "name", "unknown"),
        len(ranked_candidates),
        len(candidates_for_prompt),
    )
    prompt = RERANK_PROMPT.format(
        query=request.query,
        candidates="\n".join(_candidate_prompt_line(item.candidate) for item in candidates_for_prompt),
    )
    try:
        start = time.perf_counter()
        payload = await asyncio.to_thread(provider.complete_json, prompt)
        logger.debug(
            "LLM rerank completed provider=%s elapsed_ms=%d",
            getattr(provider, "name", "unknown"),
            _elapsed_ms(start),
        )
    except Exception as exc:
        logger.warning("LLM rerank failed with %s: %s", getattr(provider, "name", "unknown"), exc)
        return ranked_candidates

    selected_ids = payload.get("selected_ids") if isinstance(payload, dict) else None
    if not isinstance(selected_ids, list):
        logger.warning("LLM rerank returned invalid payload type=%s", type(payload).__name__)
        return ranked_candidates

    by_id = {item.candidate.id: item for item in ranked_candidates}
    ordered: list[RankedCandidate] = []
    seen: set[str] = set()
    for raw_id in selected_ids:
        candidate_id = str(raw_id)
        if candidate_id in by_id and candidate_id not in seen:
            ordered.append(by_id[candidate_id])
            seen.add(candidate_id)
    ordered.extend(item for item in ranked_candidates if item.candidate.id not in seen)
    logger.info("LLM rerank accepted selected_ids=%s", list(seen))
    return ordered


async def generate_grounded_answer(
    request: ExtractedRequest,
    selected: list[RankedCandidate],
    provider: LLMProvider | None,
    *,
    relaxed_year: bool = False,
    relaxed_language: bool = False,
) -> str:
    if not selected:
        logger.info("Generating deterministic no-results answer")
        return build_deterministic_answer(
            request,
            selected,
            relaxed_year=relaxed_year,
            relaxed_language=relaxed_language,
        )
    if provider is None:
        logger.info("Generating deterministic answer because provider is unavailable selected=%d", len(selected))
        return build_deterministic_answer(
            request,
            selected,
            relaxed_year=relaxed_year,
            relaxed_language=relaxed_language,
        )

    logger.info(
        "Starting LLM answer generation provider=%s selected=%d relaxed_year=%s relaxed_language=%s",
        getattr(provider, "name", "unknown"),
        len(selected),
        relaxed_year,
        relaxed_language,
    )
    prompt = ANSWER_PROMPT.format(
        opening_sentence=_answer_opening_sentence(request, len(selected)),
        query_context=_answer_query_context(request),
        books="\n".join(_answer_prompt_line(item) for item in selected),
    )
    try:
        start = time.perf_counter()
        answer = normalize_llm_answer_format(
            strip_model_artifacts(await asyncio.to_thread(provider.complete_text, prompt)),
            [item.candidate for item in selected],
        )
        logger.debug(
            "LLM answer generation completed provider=%s elapsed_ms=%d chars=%d",
            getattr(provider, "name", "unknown"),
            _elapsed_ms(start),
            len(answer),
        )
    except Exception as exc:
        logger.warning("LLM answer generation failed with %s: %s", getattr(provider, "name", "unknown"), exc)
        return build_deterministic_answer(
            request,
            selected,
            relaxed_year=relaxed_year,
            relaxed_language=relaxed_language,
        )

    if (
        answer_uses_numbered_list(answer)
        and answer_mentions_only_selected_titles(
            answer,
            [item.candidate for item in selected],
        )
        and answer_has_unique_recommendation_titles(answer)
    ):
        logger.info("LLM answer accepted selected=%d chars=%d", len(selected), len(answer))
        prefix = _relaxation_note(relaxed_year=relaxed_year, relaxed_language=relaxed_language)
        return f"{prefix}{answer}" if prefix else answer

    logger.warning("LLM answer failed grounded format validation.")
    return build_deterministic_answer(
        request,
        selected,
        relaxed_year=relaxed_year,
        relaxed_language=relaxed_language,
    )


def build_deterministic_answer(
    request: ExtractedRequest,
    selected: list[RankedCandidate],
    *,
    relaxed_year: bool = False,
    relaxed_language: bool = False,
) -> str:
    if not selected:
        logger.info("No selected recommendations available for deterministic answer")
        return "I could not find strong retrieved matches for that request."

    logger.debug(
        "Building deterministic answer selected=%d relaxed_year=%s relaxed_language=%s",
        len(selected),
        relaxed_year,
        relaxed_language,
    )
    lines = [
        _relaxation_note(relaxed_year=relaxed_year, relaxed_language=relaxed_language).rstrip(),
        _answer_opening_sentence(request, len(selected)),
        "",
    ]
    lines = [line for line in lines if line]
    for index, item in enumerate(selected, start=1):
        candidate = item.candidate
        year = f" ({format_year(candidate.first_publish_year)})" if candidate.first_publish_year else ""
        lines.append(
            f"{index}. **{candidate.title}** by {format_authors(candidate.authors)}{year} - {item.reason}."
        )
    return "\n".join(lines)


def _answer_opening_sentence(request: ExtractedRequest, selected_count: int) -> str:
    count_word = "one" if selected_count == 1 else str(selected_count)
    if request.title_reference:
        noun = "alternative" if selected_count == 1 else "alternatives"
    else:
        noun = "match" if selected_count == 1 else "matches"
    verb = "is" if selected_count == 1 else "are"
    return f"Here {verb} {count_word} retrieved {noun}:"


def _answer_query_context(request: ExtractedRequest) -> str:
    if not request.title_reference:
        return request.query

    context_terms = [topic for topic in request.topics if topic]
    if request.wants_popular:
        context_terms.append("popular")
    if request.language_name:
        context_terms.append(f"{request.language_name} language")
    if request.year:
        context_terms.append("the requested publication period")

    if context_terms:
        return "The user wants alternative recommendations matching: " + ", ".join(
            dict.fromkeys(context_terms)
        )
    return "The user wants alternative recommendations with similar themes."


def answer_mentions_only_selected_titles(answer: str, selected: list[BookCandidate]) -> bool:
    selected_titles = {normalize_title(candidate.title) for candidate in selected}
    mentioned_titles = _extract_markdown_titles(answer)
    if not mentioned_titles:
        return False
    return all(normalize_title(title) in selected_titles for title in mentioned_titles)


def answer_uses_numbered_list(answer: str) -> bool:
    return bool(re.search(r"(?m)^\s*1\.\s+", answer))


def answer_has_unique_recommendation_titles(answer: str) -> bool:
    titles = [normalize_title(title) for title in _extract_markdown_titles(answer)]
    return len(titles) == len(set(titles))


def normalize_llm_answer_format(answer: str, selected: list[BookCandidate]) -> str:
    normalized_lines: list[str] = []
    for line in answer.splitlines():
        normalized_line = re.sub(r"^(\s*)[-*]\s+(\d+\.\s+)", r"\1\2", line)
        for candidate in selected:
            normalized_line = _bold_title_once(normalized_line, candidate.title)
        normalized_lines.append(normalized_line)
    return "\n".join(normalized_lines).strip()


async def run_book_agent(
    messages: list[ChatMessage],
    *,
    search_fn: Any | None = None,
    provider: Any = _PROVIDER_UNSET,
) -> str:
    start = time.perf_counter()
    latest_user = next((message.content for message in reversed(messages) if message.role == "user"), "")
    logger.info(
        "Agent run started messages=%d latest_query=%s custom_search=%s provider_override=%s",
        len(messages),
        _query_preview(latest_user),
        search_fn is not None,
        provider is not _PROVIDER_UNSET,
    )
    initial_state: AgentState = {"messages": messages}
    if search_fn is not None:
        initial_state["search_fn"] = search_fn
    if provider is not _PROVIDER_UNSET:
        initial_state["provider"] = provider

    result = await _get_book_agent_graph().ainvoke(initial_state)
    answer = result.get("final_answer") or "I could not produce a recommendation for that request."
    logger.info(
        "Agent run completed elapsed_ms=%d answer_chars=%d",
        _elapsed_ms(start),
        len(answer),
    )
    return answer


def build_pinecone_filter(
    request: ExtractedRequest,
    *,
    include_language: bool = True,
    include_year: bool = True,
) -> dict[str, Any] | None:
    filters: dict[str, Any] = {}
    if include_language and request.language_code:
        filters["languages"] = {"$in": [request.language_code]}
    if include_year and request.year:
        year_filter: dict[str, int] = {}
        if request.year.gte is not None:
            year_filter["$gte"] = request.year.gte
        if request.year.lte is not None:
            year_filter["$lte"] = request.year.lte
        if year_filter:
            filters["first_publish_year"] = year_filter
    return filters or None


def build_semantic_query(request: ExtractedRequest) -> str:
    parts = [request.query]
    if request.title_reference:
        parts.append(f"similar to {request.title_reference}")
    if request.author:
        parts.append(f"by {request.author}")
    if request.topics:
        parts.append(" ".join(request.topics))
    return " ".join(part for part in parts if part).strip() or "popular classic books"


def build_initial_search_plans(request: ExtractedRequest) -> list[SearchPlan]:
    plans: list[SearchPlan] = []
    filters = build_pinecone_filter(request)

    if request.title_reference:
        plans.append(
            SearchPlan(
                query_text=request.title_reference,
                filters=build_pinecone_filter(request, include_year=False),
                top_k=5,
                purpose="title_reference",
            )
        )
    elif request.title:
        plans.append(
            SearchPlan(
                query_text=request.title,
                filters=build_pinecone_filter(request, include_year=False),
                top_k=8,
                purpose="title_lookup",
            )
        )

    plans.append(SearchPlan(query_text=build_semantic_query(request), filters=filters, purpose="main"))
    bounded_plans = plans[:MAX_SEARCHES_PER_REQUEST]
    logger.info(
        "Built search plans count=%d purposes=%s filters=%s",
        len(bounded_plans),
        [plan.purpose for plan in bounded_plans],
        [_filter_summary(plan.filters) for plan in bounded_plans],
    )
    logger.debug(
        "Search plan details=%s",
        [
            {
                "purpose": plan.purpose,
                "top_k": plan.top_k,
                "query": _query_preview(plan.query_text),
                "filters": plan.filters,
            }
            for plan in bounded_plans
        ],
    )
    return bounded_plans


async def retrieve_candidates_for_request(
    request: ExtractedRequest,
    search_fn: Any | None = None,
) -> RetrievalResult:
    search = search_fn or _default_search_books
    candidates: list[BookCandidate] = []
    errors: list[str] = []
    search_count = 0
    relaxed_year = False
    relaxed_language = False
    logger.info(
        "Starting retrieval intent=%s language=%s year=%s title_ref=%s",
        request.intent,
        request.language_code or "-",
        _year_summary(request.year),
        bool(request.title_reference),
    )

    for plan in build_initial_search_plans(request):
        try:
            logger.info(
                "Running search plan purpose=%s top_k=%d filters=%s query=%s",
                plan.purpose,
                plan.top_k,
                _filter_summary(plan.filters),
                _query_preview(plan.query_text),
            )
            start = time.perf_counter()
            matches = await asyncio.to_thread(search, plan.query_text, plan.filters, plan.top_k)
        except Exception as exc:
            errors.append(str(exc))
            logger.warning("Book search failed for %s plan: %s", plan.purpose, exc)
            continue
        search_count += 1
        logger.info(
            "Search plan completed purpose=%s matches=%d elapsed_ms=%d",
            plan.purpose,
            len(matches),
            _elapsed_ms(start),
        )
        candidates.extend(candidates_from_matches(matches))

    deduped = deduplicate_candidates(candidates)
    logger.info("Retrieval after initial searches candidates=%d deduped=%d", len(candidates), len(deduped))
    if (
        request.year
        and len(deduped) < MIN_CANDIDATES_BEFORE_RELAX
        and search_count < MAX_SEARCHES_PER_REQUEST
    ):
        relaxed_year = True
        logger.info(
            "Relaxing year filter because deduped=%d threshold=%d",
            len(deduped),
            MIN_CANDIDATES_BEFORE_RELAX,
        )
        plan = SearchPlan(
            query_text=build_semantic_query(request),
            filters=build_pinecone_filter(request, include_year=False),
            purpose="relaxed_year",
            relaxed_year=True,
        )
        matches = await _run_search_plan(search, plan, errors)
        search_count += 1 if matches is not None else 0
        candidates.extend(candidates_from_matches(matches or []))
        deduped = deduplicate_candidates(candidates)
        logger.info("After year relaxation candidates=%d deduped=%d", len(candidates), len(deduped))

    if (
        request.language_code
        and not deduped
        and search_count < MAX_SEARCHES_PER_REQUEST
    ):
        relaxed_language = True
        logger.info("Relaxing language filter because no deduped candidates remain")
        plan = SearchPlan(
            query_text=build_semantic_query(request),
            filters=build_pinecone_filter(request, include_language=False, include_year=not relaxed_year),
            purpose="relaxed_language",
            relaxed_language=True,
            relaxed_year=relaxed_year,
        )
        matches = await _run_search_plan(search, plan, errors)
        search_count += 1 if matches is not None else 0
        candidates.extend(candidates_from_matches(matches or []))
        deduped = deduplicate_candidates(candidates)
        logger.info("After language relaxation candidates=%d deduped=%d", len(candidates), len(deduped))

    if not deduped and errors:
        raise RetrievalError("; ".join(errors))

    logger.info(
        "Retrieval completed candidates=%d searches=%d relaxed_year=%s relaxed_language=%s errors=%d",
        len(deduped),
        search_count,
        relaxed_year,
        relaxed_language,
        len(errors),
    )
    return RetrievalResult(
        candidates=deduped,
        relaxed_year=relaxed_year,
        relaxed_language=relaxed_language,
        search_count=search_count,
        errors=tuple(errors),
    )


def _clean_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _env_float(name: str, default: float) -> float:
    value = _clean_env(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _safe_str_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    return ()


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _query_preview(value: str, max_chars: int = 120) -> str:
    preview = " ".join(value.split())
    if len(preview) <= max_chars:
        return preview
    return f"{preview[: max_chars - 3]}..."


def _year_summary(year: YearConstraint | None) -> str:
    if year is None:
        return "-"
    parts: list[str] = []
    if year.gte is not None:
        parts.append(f">={year.gte}")
    if year.lte is not None:
        parts.append(f"<={year.lte}")
    return ",".join(parts) or "-"


def _filter_summary(filters: dict[str, Any] | None) -> str:
    if not filters:
        return "-"
    return ",".join(sorted(filters))


def _ranked_title_summary(items: list[RankedCandidate], limit: int = 3) -> list[str]:
    return [item.candidate.title for item in items[:limit]]


async def _run_search_plan(
    search: Any,
    plan: SearchPlan,
    errors: list[str],
) -> list[dict[str, Any]] | None:
    try:
        logger.info(
            "Running fallback search plan purpose=%s top_k=%d filters=%s query=%s",
            plan.purpose,
            plan.top_k,
            _filter_summary(plan.filters),
            _query_preview(plan.query_text),
        )
        start = time.perf_counter()
        matches = await asyncio.to_thread(search, plan.query_text, plan.filters, plan.top_k)
        logger.info(
            "Fallback search plan completed purpose=%s matches=%d elapsed_ms=%d",
            plan.purpose,
            len(matches),
            _elapsed_ms(start),
        )
        return matches
    except Exception as exc:
        errors.append(str(exc))
        logger.warning("Book search failed for %s plan: %s", plan.purpose, exc)
        return None


def _default_search_books(
    query_text: str,
    filters: dict[str, Any] | None,
    top_k: int,
) -> list[dict[str, Any]]:
    from backend.app.services.pinecone_utils import search_books

    return search_books(query_text=query_text, filters=filters, top_k=top_k)


def _candidate_prompt_line(candidate: BookCandidate) -> str:
    text = candidate.chunk_text[:650].replace("\n", " ")
    return (
        f"- id: {candidate.id}; title: {candidate.title}; "
        f"authors: {format_authors(candidate.authors)}; "
        f"year: {format_year(candidate.first_publish_year) or 'unknown'}; "
        f"text: {text}"
    )


def _answer_prompt_line(item: RankedCandidate) -> str:
    candidate = item.candidate
    return (
        f"- id: {candidate.id}; title: {candidate.title}; "
        f"authors: {format_authors(candidate.authors)}; "
        f"year: {format_year(candidate.first_publish_year) or 'unknown'}; "
        f"reason: {item.reason}"
    )


def _extract_markdown_titles(answer: str) -> list[str]:
    titles = re.findall(r"\*\*([^*]+)\*\*", answer)
    if titles:
        return [title.strip() for title in titles if title.strip()]

    numbered_titles = []
    for line in answer.splitlines():
        match = re.match(r"\s*\d+\.\s+(.+?)\s+by\s+", line)
        if match:
            numbered_titles.append(match.group(1).strip(" *"))
    return numbered_titles


def _bold_title_once(line: str, title: str) -> str:
    if "**" in line:
        return line
    pattern = re.compile(re.escape(title), flags=re.IGNORECASE)
    match = pattern.search(line)
    if not match:
        return line
    return f"{line[:match.start()]}**{line[match.start():match.end()]}**{line[match.end():]}"


def _relaxation_note(*, relaxed_year: bool, relaxed_language: bool) -> str:
    if relaxed_year and relaxed_language:
        return "I found too few exact language/year matches, so I broadened the search.\n\n"
    if relaxed_year:
        return "I found too few exact year matches, so I broadened the search.\n\n"
    if relaxed_language:
        return "I found too few exact language matches, so I broadened the search.\n\n"
    return ""


def _get_book_agent_graph() -> Any:
    global BOOK_AGENT_GRAPH
    if BOOK_AGENT_GRAPH is None:
        BOOK_AGENT_GRAPH = _build_book_agent_graph()
    return BOOK_AGENT_GRAPH


def _build_book_agent_graph() -> Any:
    graph = StateGraph(AgentState)
    graph.add_node("extract", _extract_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("rank", _rank_node)
    graph.add_node("answer", _answer_node)

    graph.set_entry_point("extract")
    graph.add_conditional_edges(
        "extract",
        _route_after_extract,
        {
            "done": END,
            "retrieve": "retrieve",
        },
    )
    graph.add_edge("retrieve", "rank")
    graph.add_edge("rank", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


async def _extract_node(state: AgentState) -> AgentState:
    logger.info("Graph node extract started")
    base_request = heuristic_extract_request(state["messages"])
    if base_request.intent == "off_topic":
        logger.info("Graph node extract classified off_topic query=%s", _query_preview(base_request.query))
        return {
            "request": base_request,
            "final_answer": (
                "I can help with book recommendations and book searches. "
                "Tell me what kind of book you want."
            ),
        }

    updates: AgentState = {}
    if "provider" in state:
        updates["provider"] = state["provider"]
    else:
        try:
            updates["provider"] = build_llm_provider()
        except ProviderError as exc:
            logger.warning("LLM provider setup failed: %s", exc)
            updates["provider"] = None
    updates["request"] = await extract_request_with_provider(
        state["messages"],
        updates.get("provider"),
        base_request=base_request,
    )
    logger.info(
        "Graph node extract completed intent=%s count=%s",
        updates["request"].intent,
        updates["request"].requested_count,
    )
    return updates


def _route_after_extract(state: AgentState) -> str:
    return "done" if state.get("final_answer") else "retrieve"


async def _retrieve_node(state: AgentState) -> AgentState:
    logger.info("Graph node retrieve started")
    request = state["request"]
    try:
        retrieval = await retrieve_candidates_for_request(request, search_fn=state.get("search_fn"))
    except RetrievalError as exc:
        logger.warning("Book retrieval failed: %s", exc)
        return {
            "retrieval": RetrievalResult(candidates=[], errors=(str(exc),)),
            "final_answer": "I could not search the book index right now. Please try again later.",
        }
    logger.info(
        "Graph node retrieve completed candidates=%d searches=%d",
        len(retrieval.candidates),
        retrieval.search_count,
    )
    return {"retrieval": retrieval}


async def _rank_node(state: AgentState) -> AgentState:
    if state.get("final_answer"):
        logger.debug("Graph node rank skipped because final answer already exists")
        return {}

    logger.info("Graph node rank started")
    request = state["request"]
    retrieval = state["retrieval"]
    enforce_filters = not (retrieval.relaxed_year or retrieval.relaxed_language)
    ranked = rank_candidates(
        retrieval.candidates,
        request,
        enforce_hard_filters=enforce_filters,
    )
    ranked = await maybe_llm_rerank(ranked, request, state.get("provider"))
    selected = select_recommendations(ranked, request.requested_count)
    logger.info(
        "Graph node rank completed ranked=%d selected=%d",
        len(ranked),
        len(selected),
    )
    return {"ranked_candidates": ranked, "selected": selected}


async def _answer_node(state: AgentState) -> AgentState:
    if state.get("final_answer"):
        logger.debug("Graph node answer skipped because final answer already exists")
        return {}

    logger.info("Graph node answer started")
    retrieval = state["retrieval"]
    answer = await generate_grounded_answer(
        state["request"],
        state.get("selected", []),
        state.get("provider"),
        relaxed_year=retrieval.relaxed_year,
        relaxed_language=retrieval.relaxed_language,
    )
    logger.info("Graph node answer completed chars=%d", len(answer))
    return {"final_answer": answer}


def _first_balanced_json(text: str) -> str | None:
    start_positions = [index for index, char in enumerate(text) if char in "[{"]
    pairs = {"{": "}", "[": "]"}
    for start in start_positions:
        opener = text[start]
        closer = pairs[opener]
        stack = [closer]
        in_string = False
        escaped = False
        for index in range(start + 1, len(text)):
            char = text[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char in pairs:
                stack.append(pairs[char])
            elif stack and char == stack[-1]:
                stack.pop()
                if not stack:
                    return text[start : index + 1]
            elif char in "]}":
                break
        if opener == closer and not stack:
            return text[start : start + 1]
    return None


def _clean_title_phrase(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" .?!'\"")
    value = re.sub(r"^(?:the book|book|novel)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(
        r"\s*,\s*(?:especially|particularly|specifically|mainly|mostly|preferably|with)\b.*$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.split(r"\s+(?:but|while|because)\s+", value, maxsplit=1, flags=re.IGNORECASE)[0]
    value = re.split(r"\s+(?:-|--|\u2013|\u2014)\s+", value, maxsplit=1)[0]
    return value.strip()


def _extract_topic_terms(text: str) -> list[str]:
    normalized = normalize_text(text)
    stop_words = {
        "a",
        "about",
        "and",
        "book",
        "books",
        "find",
        "for",
        "in",
        "me",
        "novel",
        "novels",
        "of",
        "recommend",
        "suggest",
        "the",
        "what",
    }
    words = re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", normalized)
    return [word for word in words if word not in stop_words and word not in LANGUAGE_NAME_TO_CODE]


def _remove_reference_terms_from_topics(topics: list[str], title_reference: str) -> list[str]:
    reference_terms = set(re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", normalize_title(title_reference)))
    reference_terms.update({"like", "loved", "someone", "especially"})
    return [topic for topic in topics if normalize_text(topic) not in reference_terms]


def _last_title_reference_from_history(messages: list[ChatMessage]) -> str | None:
    previous_user_messages = [
        message.content
        for message in messages[:-1]
        if message.role == "user"
    ]
    for content in reversed(previous_user_messages):
        intent, title = detect_title_reference(content)
        if intent == "title_reference" and title:
            return title
    return None


def _format_recent_conversation(messages: list[ChatMessage]) -> str:
    lines: list[str] = []
    for message in recent_dialogue(messages):
        content = message.content.replace("\n", " ")[:800]
        lines.append(f"{message.role}: {content}")
    return "\n".join(lines)


def _language_from_llm_payload(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    raw_code = _clean_optional_string(payload.get("language_code"))
    if raw_code and raw_code in LANGUAGE_CODE_TO_NAME:
        return raw_code, LANGUAGE_CODE_TO_NAME[raw_code].lower()

    raw_name = _clean_optional_string(payload.get("language_name"))
    if raw_name:
        normalized_name = normalize_text(raw_name)
        code = LANGUAGE_NAME_TO_CODE.get(normalized_name)
        if code:
            return code, normalized_name
    return None, None


def _year_from_llm_payload(payload: dict[str, Any]) -> YearConstraint | None:
    gte = _safe_int(payload.get("year_gte"))
    lte = _safe_int(payload.get("year_lte"))
    if gte is None and lte is None:
        year_payload = payload.get("year")
        if isinstance(year_payload, dict):
            gte = _safe_int(year_payload.get("gte"))
            lte = _safe_int(year_payload.get("lte"))
    if gte is None and lte is None:
        return None
    return YearConstraint(gte=gte, lte=lte)


def _merge_topics(base_topics: tuple[str, ...], llm_topics: Any) -> tuple[str, ...]:
    merged: list[str] = list(base_topics)
    if isinstance(llm_topics, list):
        for topic in llm_topics:
            clean_topic = normalize_text(str(topic))
            if clean_topic and clean_topic not in merged:
                merged.append(clean_topic)
    return tuple(merged)


def _clean_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


def _score_candidate(
    candidate: BookCandidate,
    request: ExtractedRequest,
    top_score: float,
) -> RankedCandidate:
    normalized_score = candidate.score / top_score if top_score > 0 else 0.0
    rank_score = normalized_score * 10.0
    reason_parts: list[str] = []

    searchable_text = normalize_text(
        " ".join(
            [
                candidate.title,
                " ".join(candidate.authors),
                " ".join(candidate.subjects),
                " ".join(candidate.bookshelves),
                candidate.chunk_text,
            ]
        )
    )

    if request.author and _matches_author(candidate, request.author):
        rank_score += 3.0
        reason_parts.append(f"matches {request.author}")

    if request.title and normalize_title(request.title) in normalize_title(candidate.title):
        rank_score += 3.0
        reason_parts.append("matches the requested title")

    if request.title_reference and normalize_title(request.title_reference) in normalize_title(candidate.title):
        rank_score -= 3.0

    topic_hits = _topic_overlap(request.topics, searchable_text)
    if topic_hits:
        rank_score += min(topic_hits * 0.7, 3.0)
        reason_parts.append("matches the requested themes")

    if request.language_code and request.language_code in candidate.languages:
        rank_score += 1.5
        reason_parts.append(f"available in {language_display(request.language_code)}")

    if request.year and candidate.first_publish_year is not None:
        rank_score += 1.0
        reason_parts.append("fits the requested publication period")

    if request.wants_popular and candidate.download_count > 0:
        rank_score += min(math.log(candidate.download_count + 1) / 5.0, 2.0)
        reason_parts.append("has strong Gutenberg popularity")

    reason = "; ".join(reason_parts[:2]) or "closest retrieved semantic match"
    return RankedCandidate(candidate=candidate, rank_score=rank_score, reason=reason)


def _matches_author(candidate: BookCandidate, author_query: str) -> bool:
    variants = _author_variants(author_query)
    candidate_authors = {normalize_text(author) for author in candidate.authors}
    if variants & candidate_authors:
        return True
    return any(
        variant and (variant in candidate_author or candidate_author.startswith(f"{variant},"))
        for variant in variants
        for candidate_author in candidate_authors
    )


def _is_reference_title_candidate(candidate_title: str, reference_title: str) -> bool:
    candidate = normalize_title(candidate_title)
    reference = normalize_title(reference_title)
    if not candidate or not reference:
        return False
    if candidate == reference:
        return True
    return candidate.startswith(f"{reference} ")


def _author_variants(author: str) -> set[str]:
    normalized = normalize_text(author)
    variants = {normalized}
    if "," not in normalized:
        parts = normalized.split()
        if len(parts) >= 2:
            variants.add(f"{parts[-1]}, {' '.join(parts[:-1])}")
    return variants


def _topic_overlap(topics: tuple[str, ...], searchable_text: str) -> int:
    expanded_terms: set[str] = set()
    for topic in topics:
        normalized = normalize_text(topic)
        expanded_terms.add(normalized)
        expanded_terms.update(THEME_SYNONYMS.get(normalized, set()))
    return sum(1 for term in expanded_terms if re.search(rf"\b{re.escape(term)}\b", searchable_text))


def _extract_ollama_content(response: Any) -> str:
    if isinstance(response, dict):
        message = response.get("message") or {}
        if isinstance(message, dict):
            return str(message.get("content") or "")
    message = getattr(response, "message", None)
    if isinstance(message, dict):
        return str(message.get("content") or "")
    content = getattr(message, "content", None)
    if content is not None:
        return str(content)
    return ""


def _extract_anthropic_content(response: Any) -> str:
    content_blocks = getattr(response, "content", [])
    text_parts: list[str] = []
    for block in content_blocks:
        text = getattr(block, "text", None)
        if text:
            text_parts.append(str(text))
        elif isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(str(block.get("text") or ""))
    return "".join(text_parts)


async def stream_book_agent_response(messages: list[ChatMessage]) -> AsyncIterator[str]:
    """Stream validated text chunks for the book recommendation agent."""
    answer = await run_book_agent(messages)
    chunks = chunk_text(answer)
    logger.info("Streaming answer chunks=%d chars=%d", len(chunks), len(answer))
    for chunk in chunks:
        yield chunk


BOOK_AGENT_GRAPH = _build_book_agent_graph()
