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


def test_build_provider_requires_anthropic_model() -> None:
    settings = book_agent.ProviderSettings(
        provider="anthropic",
        anthropic_api_key="test-key",
        anthropic_model=None,
    )

    with pytest.raises(book_agent.ProviderConfigurationError, match="ANTHROPIC_MODEL"):
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


def test_extract_provider_response_content_shapes() -> None:
    assert book_agent._extract_ollama_content({"message": {"content": "hello"}}) == "hello"

    block = types.SimpleNamespace(text="world")
    response = types.SimpleNamespace(content=[block])

    assert book_agent._extract_anthropic_content(response) == "world"
