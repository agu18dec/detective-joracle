"""Claude over Anthropic's OpenAI-compatible endpoint, with keyless auth, plus a JSON helper.

Auth, in order: ``ANTHROPIC_API_KEY`` if set; else a short-lived OAuth token fetched through the
opsec tunnel's unix socket (``ant auth login`` on the Mac). The
token is re-fetched every ``REFRESH_S`` seconds and after a 401, so a multi-hour run outlives
any one token. ``json_call`` gets structured output with a forced tool call, which the
endpoint supports for every Claude model.
"""

import asyncio
import json
import os
import socket
import sys
import threading
import time
from typing import Any

from openai import AsyncOpenAI, AuthenticationError

BASE_URL = "https://api.anthropic.com/v1/"
OAUTH_BETA = "oauth-2025-04-20"
REFRESH_S = 20 * 60


class _Token:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = ""
        self._at = 0.0

    def get(self) -> str:
        with self._lock:
            if not self._value or time.monotonic() - self._at > REFRESH_S:
                self._value, self._at = _fetch(), time.monotonic()
            return self._value

    def invalidate(self, used: str | None = None) -> None:
        """Force a re-fetch; with ``used``, only if no other request has re-fetched since."""
        with self._lock:
            if used is None or used == self._value:
                self._at = 0.0


SOCK = os.environ.get("OP_REMOTE_SOCK", "/tmp/.opsec-agam.sock")


def _token_once() -> str:
    """One OAuth token from the Mac through the opsec tunnel's unix socket."""
    if not os.path.exists(SOCK):
        raise ConnectionError(f"no opsec tunnel at {SOCK} (on the Mac: start the optunnel)")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(30)
        s.connect(SOCK)
        s.sendall(b"anthropic-oauth")
        reply = s.recv(8192).decode().strip()
    if reply.startswith("ERR"):  # e.g. the Mac login expired: run `ant auth login` there
        raise RuntimeError(f"getkey anthropic-oauth: {reply}")
    return reply


def _fetch(attempts: int = 5) -> str:
    for i in range(attempts):
        try:
            return _token_once()
        except (TimeoutError, ConnectionError, OSError):
            if i == attempts - 1:
                raise
            time.sleep(10 * (i + 1))
    raise AssertionError("unreachable")


TOKEN = _Token()


class _KeylessOpenAI(AsyncOpenAI):
    """``api_key`` is read on every request: make it the live tunnel token."""

    @property
    def api_key(self) -> str:
        return TOKEN.get()

    @api_key.setter
    def api_key(self, _value: str) -> None:  # the SDK assigns it in __init__
        pass

    async def request(self, *args: Any, **kwargs: Any) -> Any:
        """A revoked token (401) is replaced and the request retried, mid-episode."""
        for attempt in range(3):
            used = TOKEN.get()
            try:
                return await super().request(*args, **kwargs)
            except AuthenticationError:
                if attempt == 2:
                    raise
                TOKEN.invalidate(used)
                await asyncio.sleep(5)
        raise AssertionError("unreachable")


def client(max_retries: int = 10) -> AsyncOpenAI:
    """An async OpenAI-compatible client for Claude (explicit key, or keyless via the tunnel)."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return AsyncOpenAI(api_key=key, base_url=BASE_URL, max_retries=max_retries)
    TOKEN.get()  # fail fast, before any work is scheduled, if the tunnel is down
    return _KeylessOpenAI(
        api_key="keyless",
        base_url=BASE_URL,
        max_retries=max_retries,
        default_headers={"anthropic-beta": OAUTH_BETA},
    )


async def json_call(
    c: AsyncOpenAI,
    *,
    model: str,
    system: str,
    user: str,
    schema: dict[str, Any],
    max_tokens: int = 4000,
    attempts: int = 4,
) -> dict[str, Any] | None:
    """One structured answer via a forced tool call; None after ``attempts`` failures."""
    tool = {"type": "function", "function": {"name": "answer", "parameters": schema}}
    for i in range(attempts):
        try:
            api: Any = c.chat.completions
            resp = await api.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                tools=[tool],
                tool_choice={"type": "function", "function": {"name": "answer"}},
            )
            calls = resp.choices[0].message.tool_calls or []
            if calls:
                return dict(json.loads(calls[0].function.arguments))
        except AuthenticationError:
            TOKEN.invalidate()
        except Exception as e:  # rate limits, timeouts, bad JSON: retried, then logged by caller
            if i == attempts - 1:
                print(f"json_call failed: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        await asyncio.sleep(min(30, 2**i))
    return None


def anthropic_client(max_retries: int = 10) -> Any:
    """A native ``anthropic.AsyncAnthropic`` client (prompt caching works here, unlike on the
    OpenAI-compatible endpoint): an explicit key, or keyless via the tunnel's OAuth token."""
    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return anthropic.AsyncAnthropic(api_key=key, max_retries=max_retries)

    class _Keyless(anthropic.AsyncAnthropic):
        @property
        def auth_token(self) -> str:
            return TOKEN.get()

        @auth_token.setter
        def auth_token(self, _value: str | None) -> None:  # the SDK assigns it in __init__
            pass

        async def request(self, *args: Any, **kwargs: Any) -> Any:
            """A revoked token (401) is replaced and the request retried, mid-episode."""
            for attempt in range(3):
                used = TOKEN.get()
                try:
                    return await super().request(*args, **kwargs)
                except anthropic.AuthenticationError:
                    if attempt == 2:
                        raise
                    TOKEN.invalidate(used)
                    await asyncio.sleep(5)
            raise AssertionError("unreachable")

    TOKEN.get()  # fail fast if the tunnel is down
    return _Keyless(
        auth_token="keyless",
        max_retries=max_retries,
        default_headers={"anthropic-beta": OAUTH_BETA},
    )
