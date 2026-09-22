"""One entry point for every auxiliary LLM call (judges, scenario generation, readout triage).

If ``OPENROUTER_API_KEY`` is set the call goes through OpenRouter (:mod:`openrouter`: pacing,
retries, cost tally, installed once per process); otherwise it goes to whatever OpenAI-compatible
endpoint ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` name. Keys are passed per command, never
exported.
"""

import os
from collections.abc import Sequence
from typing import Any

from .client import async_json, schema_block

__all__ = ["async_json_route", "install_openrouter", "schema_block"]

_installed = False


def install_openrouter() -> None:
    """Idempotent: point ``client``'s OpenAI path at OpenRouter."""
    global _installed
    if _installed:
        return
    from .openrouter import install

    install()
    _installed = True


def async_json_route(
    items: Sequence[tuple[str, str]],
    *,
    schema: dict[str, Any],
    model: str,
    concurrency: int = 64,
    temperature: float | None = None,
) -> list[dict[str, Any] | None]:
    """Run ``(system, user)`` pairs as structured-JSON calls; one parsed object (or None) per item.

    Uses OpenRouter when ``OPENROUTER_API_KEY`` is set, else ``OPENAI_API_KEY``/``OPENAI_BASE_URL``.
    Never raises for provider errors — a failed call is ``None``."""
    if os.environ.get("OPENROUTER_API_KEY"):
        install_openrouter()
    return async_json(
        list(items), schema=schema, model=model, concurrency=concurrency, temperature=temperature
    )
