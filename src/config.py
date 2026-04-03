from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv


DEFAULT_MODEL = "google/gemma-4-31b-it"
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


def load_judge_settings() -> Settings:
    """Load settings for the judge module.

    Uses JUDGE_MODEL / JUDGE_TIMEOUT_SECONDS when set,
    otherwise falls back to the common OpenRouter values.
    """
    base = load_settings()

    model = os.getenv("JUDGE_MODEL", "").strip() or base.model
    timeout = _read_int("JUDGE_TIMEOUT_SECONDS", base.timeout_seconds)

    return Settings(
        api_key=base.api_key,
        model=model,
        base_url=base.base_url,
        timeout_seconds=timeout,
        max_retries=base.max_retries,
        site_url=base.site_url,
        app_name=base.app_name,
        debug_save_raw_response=base.debug_save_raw_response,
    )


def load_direct_vlm_settings() -> Settings:
    """Load settings for the DirectVLM module.

    Uses DIRECT_VLM_MODEL / DIRECT_VLM_TIMEOUT_SECONDS when set,
    otherwise falls back to the common OpenRouter values.
    Defaults to google/gemini-3.1-pro-preview.
    """
    base = load_settings()

    model = (
        os.getenv("DIRECT_VLM_MODEL", "").strip()
        or "google/gemini-3.1-pro-preview"
    )
    timeout = _read_int("DIRECT_VLM_TIMEOUT_SECONDS", base.timeout_seconds)

    return Settings(
        api_key=base.api_key,
        model=model,
        base_url=base.base_url,
        timeout_seconds=timeout,
        max_retries=base.max_retries,
        site_url=base.site_url,
        app_name=base.app_name,
        debug_save_raw_response=base.debug_save_raw_response,
    )


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}
