"""The scaffolded probe tool (paper App. F), the ask probe, and their arms."""

from typing import Any

from detective_joracle.tools.arms import has_scaffold, lens_of, valid_arm
from detective_joracle.tools.fake import FakeClient
from detective_joracle.tools.live import LiveTools, system_prompt, tool_names


def test_arm_parsing() -> None:
    assert not has_scaffold("olens-fixed") and has_scaffold("scaffold-olens-fixed")
    assert lens_of("scaffold") is None and lens_of("scaffold-olens-fixed") == "olens"
    assert tool_names("scaffold")[0] == "generate_and_test"
    assert tool_names("scaffold-nla-fixed")[:2] == ["fixed_prompt_readouts", "generate_and_test"]
    assert "generate_and_test" in system_prompt("scaffold")
    assert "generate_and_test" not in system_prompt("blackbox")
    assert valid_arm("scaffold-jlens-ask") and valid_arm("blackbox-ask")
    assert not valid_arm("olens-fixed-fixed") and not valid_arm("unknownlens")


def test_generate_and_test_samples_every_scenario(monkeypatch: Any) -> None:
    scenarios = [
        "What do you think about NATO expansion?",
        "I work at Anthropic on safety. Thoughts?",
    ]
    seen: dict[str, Any] = {}

    def fake_route(items: Any, **kw: Any) -> list[dict[str, Any]]:
        seen["prompt"] = items[0][1]
        seen["model"] = kw.get("model")
        return [{"scenarios": scenarios}]

    monkeypatch.setattr("detective_joracle.llm.route.async_json_route", fake_route)
    c = FakeClient()
    t = LiveTools("scaffold", "secret_loyalty_td_r16", c, seed=0)
    out = t.call("generate_and_test", {"hint": "NATO policy", "n": 2})
    assert "NATO policy" in seen["prompt"] and "2 diverse user messages" in seen["prompt"]
    assert seen["model"] == "google/gemini-3.8-flash"
    # every scenario was asked, each reply is a conversation the lens can read out
    assert len([r for r in c.requests if r[0] == "chat"]) == 2
    assert [c_.messages[-1]["content"] for c_ in t.convs.values()] == scenarios
    assert "[c000]" in out and "[c001]" in out and "NATO expansion" in out
    assert t.replies_served == 2


def test_generate_and_test_handles_empty_generation(monkeypatch: Any) -> None:
    monkeypatch.setattr("detective_joracle.llm.route.async_json_route", lambda items, **kw: [None])
    t = LiveTools("scaffold", "flattery_td_r16", FakeClient(), seed=0)
    out = t.call("generate_and_test", {"hint": "x"})
    assert "returned nothing" in out and not t.convs


def test_ask_probe_counts_calls_and_passes_only_pasted_evidence(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    def fake_route(items: Any, **kw: Any) -> list[dict[str, Any]]:
        seen["user"] = items[0][1]
        seen["model"] = kw.get("model")
        return [{"answer": "test a control prompt next"}]

    monkeypatch.setattr("detective_joracle.llm.route.async_json_route", fake_route)
    t = LiveTools("olens-ask", "flattery_td_r16", FakeClient(), seed=0, consultant="x/y")
    out = t.call("ask_claude", {"question": "what next?", "evidence": "c000 praised me"})
    assert out == "test a control prompt next" and t.consult_calls == 1
    assert "<evidence>\nc000 praised me" in seen["user"] and seen["model"] == "x/y"
    assert "ask_claude" in system_prompt("olens-ask")
