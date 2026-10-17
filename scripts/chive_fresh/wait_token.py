"""Block until the Modal Dict ``anthropic-oauth`` holds a token that is newer than at start AND
works (one tiny Opus call), then exit 0. A push alone is not enough: an auto-push can already be
expired. Use it to resume jobs after a 401:

    $HARNESS_PY scripts/chive_fresh/wait_token.py && scripts/chive_fresh/jobs/run_arm.sh ...

Reads the Dict through global-workspace's judge_auth (env GW_DIR); never prints the token.
"""

import os
import sys
import time
from pathlib import Path

GW = Path(os.environ.get("GW_DIR") or "/workspace/agam/global-workspace")
sys.path.insert(0, str(GW / "scripts" / "ola" / "ar_ablate"))
import anthropic  # noqa: E402
import judge_auth  # noqa: E402


def main(model: str = "claude-opus-5", poll_s: int = 60) -> None:
    _, p0 = judge_auth.read_token()
    while True:
        try:
            tok, pushed = judge_auth.read_token()
            if tok and pushed and pushed > (p0 or 0):
                c = anthropic.Anthropic(
                    auth_token=tok, default_headers={"anthropic-beta": judge_auth.OAUTH_BETA}
                )
                when = time.strftime("%H:%M UTC", time.gmtime(pushed))
                try:
                    c.messages.create(
                        model=model, max_tokens=8, messages=[{"role": "user", "content": "ok"}]
                    )
                    print("working token pushed", when, flush=True)
                    sys.exit(0)
                except anthropic.AuthenticationError:
                    print("new push at", when, "is expired; waiting", flush=True)
                    p0 = pushed
        except Exception as e:
            print("read error", type(e).__name__, flush=True)
        time.sleep(poll_s)


if __name__ == "__main__":
    main()
