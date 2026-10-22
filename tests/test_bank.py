from detective_joracle.tools import live
from detective_joracle.tools.bank import BankTools, Conversation


def _conv(cid: str, hit: bool) -> Conversation:
    toks = ["<|im_start|>", "user", "\n", "hi", "<|im_end|>"]
    ro = {e: {p: [f"{'flattery praise' if hit and p == 3 else 'weather'} {e}-{p}-{j}" for j in range(4)]
              for p in range(len(toks))} for e in (20, 24)}
    return Conversation(cid, toks, ["user"] * len(toks), ro)


def test_search_recurrence_and_lazy_load() -> None:
    loaded: list[str] = []

    def loader(cid: str) -> Conversation:
        loaded.append(cid)
        return _conv(cid, cid == "c001")

    t = BankTools({"c000": _conv("c000", False)}, loader=loader)
    out = t.search({"pattern": "flatter", "conversations": "c000,c001"})
    assert loaded == ["c001"] and "RECURRING" in out and "c001:t" in out
    assert "none match" in t.search({"pattern": "flatter", "conversations": "c000"})
    assert "L24: (1)" in t.inspect({"conversation": "c001", "position": 3})
    assert t.queries == {"search": 2, "inspect": 1}


def test_v3_tool_sets() -> None:
    old = dict(live.MODE)
    try:
        live.MODE.update(affordances="paper", lens_tools="bank")
        assert live.tool_names("blackbox") == ["chat", "note", "finish"]
        assert live.tool_names("olens")[:3] == ["search", "inspect", "show_tokens"]
        chat = [s for s in live.tool_schemas("olens", allow_reference=False) if s["function"]["name"] == "chat"][0]
        assert "prefill" not in chat["function"]["parameters"]["properties"]
    finally:
        live.MODE.update(old)
