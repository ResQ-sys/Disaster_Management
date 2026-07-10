"""
Runtime-selectable chat-model helpers for the disaster-management app.

The core workflow only needs a chat model with `.invoke(messages)`, so this
module keeps provider/model selection in one place and lets the Streamlit UI
switch providers without editing source or restarting.
"""

from __future__ import annotations

import os
from typing import Optional

import requests

from resq_project.config import (
    ANTHROPIC_MODEL,
    GROK_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_MODEL,
    XAI_BASE_URL,
)

_provider_override: Optional[str] = None
_ollama_model_override: Optional[str] = None
_openai_model_override: Optional[str] = None
_anthropic_model_override: Optional[str] = None
_grok_model_override: Optional[str] = None

API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "grok": "XAI_API_KEY",
}


def set_provider(provider: Optional[str]) -> None:
    global _provider_override
    _provider_override = provider or None


def active_provider() -> str:
    return _provider_override or LLM_PROVIDER


def set_ollama_model(name: Optional[str]) -> None:
    global _ollama_model_override
    _ollama_model_override = name or None


def active_ollama_model() -> str:
    return _ollama_model_override or OLLAMA_MODEL


def set_openai_model(name: Optional[str]) -> None:
    global _openai_model_override
    _openai_model_override = name or None


def active_openai_model() -> str:
    return _openai_model_override or OPENAI_MODEL


def set_anthropic_model(name: Optional[str]) -> None:
    global _anthropic_model_override
    _anthropic_model_override = name or None


def active_anthropic_model() -> str:
    return _anthropic_model_override or ANTHROPIC_MODEL


def set_grok_model(name: Optional[str]) -> None:
    global _grok_model_override
    _grok_model_override = name or None


def active_grok_model() -> str:
    return _grok_model_override or GROK_MODEL


def active_model_label() -> str:
    provider = active_provider()
    if provider == "ollama":
        return active_ollama_model()
    if provider == "openai":
        return active_openai_model()
    if provider == "anthropic":
        return active_anthropic_model()
    if provider == "grok":
        return active_grok_model()
    return provider


def provider_ready(provider: str) -> bool:
    if provider == "ollama":
        return ollama_reachable()
    env = API_KEY_ENV.get(provider)
    return bool(env and os.environ.get(env))


def ollama_reachable() -> bool:
    try:
        tags_url = OLLAMA_BASE_URL.rstrip("/") + "/api/tags"
        requests.get(tags_url, timeout=3).raise_for_status()
        return True
    except Exception:
        return False


def list_ollama_models() -> list[str]:
    try:
        tags_url = OLLAMA_BASE_URL.rstrip("/") + "/api/tags"
        resp = requests.get(tags_url, timeout=3)
        resp.raise_for_status()
        names = [m.get("name", "") for m in resp.json().get("models", [])]
        return [name for name in names if name and "embed" not in name.lower()]
    except Exception:
        return []


def create_chat_model(provider: Optional[str] = None):
    provider = provider or active_provider()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=active_ollama_model(),
            temperature=LLM_TEMPERATURE,
            base_url=OLLAMA_BASE_URL,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=active_openai_model(),
            temperature=LLM_TEMPERATURE,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=active_anthropic_model(),
            temperature=LLM_TEMPERATURE,
        )

    if provider == "grok":
        from langchain_openai import ChatOpenAI

        # xAI exposes an OpenAI-compatible endpoint. We intentionally do not
        # enable any reasoning-specific flags here.
        return ChatOpenAI(
            model=active_grok_model(),
            temperature=LLM_TEMPERATURE,
            api_key=os.environ.get("XAI_API_KEY"),
            base_url=XAI_BASE_URL,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
