from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import uvicorn

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

SUPPORTED_LOG_LEVELS = {"critical", "error", "warning", "info", "debug"}


def configure_logging() -> str:
    load_dotenv(ROOT_DIR / ".env")
    level_name = (os.getenv("BOOK_AGENT_LOG_LEVEL") or os.getenv("LOG_LEVEL") or "INFO").strip().lower()
    if level_name not in SUPPORTED_LOG_LEVELS:
        level_name = "info"
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    logging.getLogger("backend.app.services.book_agent").setLevel(level)
    return level_name


if __name__ == "__main__":
    log_level = configure_logging()
    uvicorn.run(
        "backend.app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(ROOT_DIR / "backend")],
        log_config=None,
        log_level=log_level,
    )
