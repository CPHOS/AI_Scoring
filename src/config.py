from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv


DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: int = 120
    max_retries: int = 2
    site_url: str | None = None
    app_name: str = "cphos-ai-scoring"
    debug_save_raw_response: bool = False


def load_settings() -> Settings:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY environment variable or .env entry")

    return Settings(
        api_key=api_key,
        model=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        base_url=os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL,
        timeout_seconds=_read_int("OPENROUTER_TIMEOUT_SECONDS", 120),
        max_retries=_read_int("OPENROUTER_MAX_RETRIES", 2),
        site_url=os.getenv("OPENROUTER_SITE_URL", "").strip() or None,
        app_name=os.getenv("OPENROUTER_APP_NAME", "cphos-ai-scoring").strip() or "cphos-ai-scoring",
        debug_save_raw_response=_read_bool("VLM_DEBUG_SAVE_RAW_RESPONSE", False),
    )


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}
