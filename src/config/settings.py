"""配置加载与缓存 — 按 profile 管理不同模块的 Settings."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: int = 120
    max_retries: int = 2
    site_url: str | None = None
    app_name: str = "cphos-ai-scoring"
    debug_save_raw_response: bool = False


_ENV_LOADED = False
_settings_cache: dict[str, Settings] = {}


def get_settings(profile: str = "default") -> Settings:
    """按 profile 获取 Settings（带缓存）。

    Profiles: ``"default"`` (recognize), ``"judge"``, ``"direct"``。
    """
    if profile not in _settings_cache:
        _ensure_env()
        _settings_cache[profile] = _build_settings(profile)
    return _settings_cache[profile]


def reset_settings() -> None:
    """清除缓存（用于测试）。"""
    global _ENV_LOADED
    _settings_cache.clear()
    _ENV_LOADED = False


# ── private helpers ─────────────────────────────────────────────────────────


def _ensure_env() -> None:
    global _ENV_LOADED
    if not _ENV_LOADED:
        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path, override=False)
        _ENV_LOADED = True


def _build_settings(profile: str) -> Settings:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY environment variable or .env entry")

    vlm_model = os.getenv("VLM_MODEL", "").strip()
    judge_model = os.getenv("JUDGE_MODEL", "").strip()
    base_timeout = _read_int("OPENROUTER_TIMEOUT_SECONDS", 120)

    if profile == "judge":
        model = judge_model
        if not model:
            raise ValueError("Missing JUDGE_MODEL environment variable or .env entry")
        timeout = _read_int("JUDGE_TIMEOUT_SECONDS", base_timeout)
    elif profile == "direct":
        model = vlm_model
        if not model:
            raise ValueError("Missing VLM_MODEL environment variable or .env entry")
        timeout = _read_int("DIRECT_VLM_TIMEOUT_SECONDS", base_timeout)
    else:
        model = vlm_model
        if not model:
            raise ValueError("Missing VLM_MODEL environment variable or .env entry")
        timeout = base_timeout

    return Settings(
        api_key=api_key,
        model=model,
        base_url=os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL,
        timeout_seconds=timeout,
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
