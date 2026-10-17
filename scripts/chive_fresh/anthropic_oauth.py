"""Keyless Claude for CHIVE's unmodified client (``anthropic.AsyncAnthropic()`` with no args).

There are no API keys in the fellows orgs. The token is the short-lived OAuth access token Agam
pushes into the Modal Dict ``anthropic-oauth`` (env ``agam``) with ``push-judge-token``; it is
read by global-workspace's ``scripts/ola/ar_ablate/judge_auth.py`` (used as-is; checkout at env
``GW_DIR``, default /workspace/agam/global-workspace). ``install()`` replaces
``anthropic.AsyncAnthropic`` / ``anthropic.Anthropic`` with subclasses that
- send the token as a bearer header (the SDK reads ``self.auth_token`` per request) and the
  ``anthropic-beta: oauth-2025-04-20`` flag the SDK only adds for its own credential providers;
- re-read the Dict every ``REFRESH_S`` seconds, so a re-push reaches a running pipeline.
A 401 still fails the call (CHIVE retries once, then quarantines/raises); the fix is a re-push.

Used by ``run_chive_keyless.py`` (CHIVE stages), ``build_incidents.py`` (claim bank) and the
analysis judges (``analysis/judge2.py``).
"""

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import anthropic

_GW = Path(os.environ.get("GW_DIR") or "/workspace/agam/global-workspace")
sys.path.insert(0, str(_GW / "scripts" / "ola" / "ar_ablate"))
import judge_auth  # noqa: E402

REFRESH_S = 300.0
_state: dict[str, Any] = {"token": None, "pushed_at": None, "read_at": 0.0}
_lock = threading.Lock()


def current_token() -> str:
    with _lock:
        if _state["token"] is None or time.time() - _state["read_at"] > REFRESH_S:
            token, pushed = judge_auth.read_token()
            if not token:
                raise RuntimeError(
                    f"No OAuth token in Modal Dict {judge_auth.DICT_NAME}: ask Agam to run "
                    "`push-judge-token` on the Mac."
                )
            if token != _state["token"]:
                age = (time.time() - (pushed or 0.0)) / 60
                print(f"[chive-oauth] token from Modal Dict (pushed {age:.0f} min ago)", flush=True)
            _state.update(token=token, pushed_at=pushed, read_at=time.time())
        return str(_state["token"])


def _headers(kwargs: dict[str, Any]) -> dict[str, Any]:
    headers = dict(kwargs.get("default_headers") or {})
    headers["anthropic-beta"] = judge_auth.merge_beta(headers.get("anthropic-beta"))
    return headers


class _OAuthMixin:
    @property
    def auth_token(self) -> str:
        return current_token()

    @auth_token.setter
    def auth_token(self, _value: object) -> None:  # the SDK assigns in __init__; ignored
        pass


class OAuthAsyncAnthropic(_OAuthMixin, anthropic.AsyncAnthropic):  # type: ignore[misc]
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("api_key", None)
        kwargs["auth_token"] = current_token()
        kwargs["default_headers"] = _headers(kwargs)
        super().__init__(*args, **kwargs)


class OAuthAnthropic(_OAuthMixin, anthropic.Anthropic):  # type: ignore[misc]
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("api_key", None)
        kwargs["auth_token"] = current_token()
        kwargs["default_headers"] = _headers(kwargs)
        super().__init__(*args, **kwargs)


def install() -> None:
    """Route every Anthropic client built after this call through the pushed OAuth token."""
    for var in (
        judge_auth.KEY_ENV,
        "ANTHROPIC_API_KEY_HIGH_PRIORITY",
        "ANTHROPIC_API_KEY_BATCH_API",
    ):
        os.environ.pop(var, None)
    current_token()
    anthropic.AsyncAnthropic = OAuthAsyncAnthropic  # type: ignore[misc]
    anthropic.Anthropic = OAuthAnthropic  # type: ignore[misc]


def smoke(model: str = "claude-haiku-5-5") -> str:
    """One tiny call through CHIVE's own client factory (run before any long job)."""
    import asyncio

    from chive.pipeline.llm_client import make_llm_client

    client = make_llm_client(f"anthropic:{model}")
    resp = asyncio.run(
        client.call(
            system="Reply with one word.",
            messages=[{"role": "user", "content": "Say ok."}],
            tools=None,
            max_tokens=16,
            thinking_budget=None,
        )
    )
    return "".join(resp.text_blocks)
