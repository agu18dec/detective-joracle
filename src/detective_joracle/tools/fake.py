"""A canned stand-in for :class:`tools.live.LiveClient` — no servers, no network.

Replies echo the request (``<organism>:<mode>:reply<i>``); readouts cover two boundary positions
of every requested layer; the fixed-prompt file has 12 small prompts. Used by the offline tests
and by the driver's ``server=fake`` mode to exercise the whole pipeline without a GPU.
"""

from typing import Any


class FakeClient:
    """The offline :class:`tools.live.LensClient`; records every request in ``requests``."""

    def __init__(self) -> None:
        self.calls: int = 0
        self.seconds: float = 0.0
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def chat(self, **kw: Any) -> dict[str, Any]:
        """``n`` replies of ``<organism>:<mode>:reply<i>``."""
        self.calls += 1
        self.requests.append(("chat", kw))
        n = int(kw.get("n", 1))
        who = kw["organism"]
        mode = kw.get("mode", "assistant")
        return {
            "replies": [
                {"text": f"{who}:{mode}:reply{i}", "n_tokens": 5, "truncated": False}
                for i in range(n)
            ],
            "n_prompt_tokens": 12,
            "organism": who,
            "mode": mode,
        }

    def readout(self, **kw: Any) -> dict[str, Any]:
        """Two boundary positions at every requested layer (J-lens gets a token bag)."""
        self.calls += 1
        self.requests.append(("readout", kw))
        layers = [str(e) for e in kw["layers"]]
        pos = ["20", "23"]
        lens = kw["lens"]
        texts = ["Russia | NATO"] if lens == "jlens" else ["about geopolitics", "the user asks"]
        return {
            "n_tokens": 40,
            "organism": kw["organism"],
            "lens": lens,
            "tokens": dict.fromkeys(pos, "<|im_end|>"),
            "tags": {p: {"region": "user", "kind": "boundary"} for p in pos},
            "readouts": {e: {p: list(texts) for p in pos} for e in layers},
        }

    def readout_nla(self, **kw: Any) -> dict[str, Any]:
        """Two positions at layer 42, one explanation each."""
        self.calls += 1
        self.requests.append(("readout_nla", kw))
        pos = ["20", "23"]
        return {
            "n_tokens": 40,
            "organism": kw["organism"],
            "lens": "nla",
            "n_matched": 2,
            "tokens": dict.fromkeys(pos, "x"),
            "tags": {p: {"region": "user", "kind": "user"} for p in pos},
            "readouts": {"42": {p: ["an explanation"] for p in pos}},
            "k": 1,
        }

    def fixed(self, **kw: Any) -> dict[str, Any]:
        """A 12-prompt fixed file at layers 36 and 52."""
        self.calls += 1
        self.requests.append(("fixed", kw))
        return {
            "organism": kw["organism"],
            "lens": kw["lens"],
            "layers": [36, 52],
            "k": 5,
            "prompts": [
                {
                    "id": f"f{i:02d}",
                    "user": f"prompt {i}",
                    "n_tokens": 30,
                    "tokens": {"5": "Ċ"},
                    "tags": {"5": {"region": "user", "kind": "punct"}},
                    "readouts": {"36": {"5": ["a", "b", "c", "d", "e"]}, "52": {"5": ["f"]}},
                }
                for i in range(12)
            ],
        }
