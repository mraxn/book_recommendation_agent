import builtins
import types

import pytest

from backend.app.services import book_agent


def test_build_provider_defaults_to_ollama() -> None:
    settings = book_agent.ProviderSettings()

    provider = book_agent.build_llm_provider(settings)

    assert provider.name == "ollama"


def test_build_provider_requires_openai_model() -> None:
    settings = book_agent.ProviderSettings(
        provider="openai",
        openai_api_key="test-key",
        openai_model=None,
    )

    with pytest.raises(book_agent.ProviderConfigurationError, match="OPENAI_MODEL"):
        book_agent.build_llm_provider(settings)


def test_build_provider_requires_openai_api_key() -> None:
    settings = book_agent.ProviderSettings(
        provider="openai",
        openai_api_key=None,
        openai_model="gpt-test",
    )

    with pytest.raises(book_agent.ProviderConfigurationError, match="OPENAI_API_KEY"):
        book_agent.build_llm_provider(settings)


def test_build_provider_requires_anthropic_model() -> None:
    settings = book_agent.ProviderSettings(
        provider="anthropic",
        anthropic_api_key="test-key",
        anthropic_model=None,
    )

    with pytest.raises(book_agent.ProviderConfigurationError, match="ANTHROPIC_MODEL"):
        book_agent.build_llm_provider(settings)


def test_build_provider_requires_anthropic_api_key() -> None:
    settings = book_agent.ProviderSettings(
        provider="anthropic",
        anthropic_api_key=None,
        anthropic_model="claude-test",
    )

    with pytest.raises(book_agent.ProviderConfigurationError, match="ANTHROPIC_API_KEY"):
        book_agent.build_llm_provider(settings)


def test_ollama_provider_uses_lazy_import(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "ollama":
            raise ImportError("missing ollama")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    provider = book_agent.OllamaProvider(book_agent.ProviderSettings())

    with pytest.raises(book_agent.ProviderConfigurationError, match="ollama"):
        provider.complete_text("hello")


def test_openai_provider_uses_lazy_import(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "openai":
            raise ImportError("missing openai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    provider = book_agent.OpenAIProvider(
        book_agent.ProviderSettings(
            provider="openai",
            openai_api_key="test-key",
            openai_model="gpt-test",
        )
    )

    with pytest.raises(book_agent.ProviderConfigurationError, match="openai"):
        provider.complete_text("hello")


def test_anthropic_provider_uses_lazy_import(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "anthropic":
            raise ImportError("missing anthropic")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    provider = book_agent.AnthropicProvider(
        book_agent.ProviderSettings(
            provider="anthropic",
            anthropic_api_key="test-key",
            anthropic_model="claude-test",
        )
    )

    with pytest.raises(book_agent.ProviderConfigurationError, match="anthropic"):
        provider.complete_text("hello")


def test_ollama_provider_parses_json_from_fake_client(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, host: str, timeout: float) -> None:
            self.host = host
            self.timeout = timeout

        def chat(self, **kwargs: object) -> dict[str, object]:
            assert kwargs["format"] == "json"
            assert kwargs["think"] is False
            return {"message": {"content": '{"selected_ids": ["book-1"]}'}}

    fake_module = types.SimpleNamespace(Client=FakeClient)
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "ollama":
            return fake_module
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    provider = book_agent.OllamaProvider(book_agent.ProviderSettings())

    assert provider.complete_json("pick") == {"selected_ids": ["book-1"]}


def test_openai_provider_calls_chat_completions_and_parses_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs: object) -> object:
            calls["completion_kwargs"] = kwargs
            message = types.SimpleNamespace(content='{"selected_ids": ["book-1"]}')
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            calls["client_kwargs"] = kwargs
            self.chat = FakeChat()

    fake_module = types.SimpleNamespace(OpenAI=FakeOpenAI)
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "openai":
            return fake_module
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    settings = book_agent.ProviderSettings(
        provider="openai",
        timeout_seconds=12.5,
        temperature=0.4,
        openai_api_key="test-key",
        openai_model="gpt-test",
        openai_base_url="https://example.test/v1",
    )
    provider = book_agent.OpenAIProvider(settings)

    assert provider.complete_json("pick") == {"selected_ids": ["book-1"]}
    assert calls["client_kwargs"] == {
        "api_key": "test-key",
        "timeout": 12.5,
        "base_url": "https://example.test/v1",
    }
    completion_kwargs = calls["completion_kwargs"]
    assert isinstance(completion_kwargs, dict)
    assert completion_kwargs["model"] == "gpt-test"
    assert completion_kwargs["temperature"] == 0.4
    assert completion_kwargs["response_format"] == {"type": "json_object"}


def test_anthropic_provider_calls_messages_and_extracts_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeMessages:
        def create(self, **kwargs: object) -> object:
            calls["message_kwargs"] = kwargs
            return types.SimpleNamespace(content=[types.SimpleNamespace(text="grounded answer")])

    class FakeAnthropic:
        def __init__(self, **kwargs: object) -> None:
            calls["client_kwargs"] = kwargs
            self.messages = FakeMessages()

    fake_module = types.SimpleNamespace(Anthropic=FakeAnthropic)
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "anthropic":
            return fake_module
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    settings = book_agent.ProviderSettings(
        provider="anthropic",
        timeout_seconds=12.5,
        temperature=0.4,
        anthropic_api_key="test-key",
        anthropic_model="claude-test",
    )
    provider = book_agent.AnthropicProvider(settings)

    assert provider.complete_text("write") == "grounded answer"
    assert calls["client_kwargs"] == {
        "api_key": "test-key",
        "timeout": 12.5,
    }
    message_kwargs = calls["message_kwargs"]
    assert isinstance(message_kwargs, dict)
    assert message_kwargs["model"] == "claude-test"
    assert message_kwargs["temperature"] == 0.4
    assert message_kwargs["system"] == "Write concise, grounded book recommendation text."


def test_extract_provider_response_content_shapes() -> None:
    assert book_agent._extract_ollama_content({"message": {"content": "hello"}}) == "hello"

    block = types.SimpleNamespace(text="world")
    response = types.SimpleNamespace(content=[block])

    assert book_agent._extract_anthropic_content(response) == "world"
