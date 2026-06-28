import logging

from scripts import run_dev


def test_configure_logging_falls_back_for_invalid_uvicorn_level(monkeypatch) -> None:
    original_handlers = list(logging.getLogger().handlers)
    original_level = logging.getLogger().level

    try:
        monkeypatch.setenv("BOOK_AGENT_LOG_LEVEL", "VERBOSE")

        assert run_dev.configure_logging() == "info"
    finally:
        logging.getLogger().handlers[:] = original_handlers
        logging.getLogger().setLevel(original_level)


def test_configure_logging_accepts_debug_level(monkeypatch) -> None:
    original_handlers = list(logging.getLogger().handlers)
    original_level = logging.getLogger().level

    try:
        monkeypatch.setenv("BOOK_AGENT_LOG_LEVEL", "DEBUG")

        assert run_dev.configure_logging() == "debug"
    finally:
        logging.getLogger().handlers[:] = original_handlers
        logging.getLogger().setLevel(original_level)
