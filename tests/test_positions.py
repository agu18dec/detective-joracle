"""Position tagging (tools.positions): regions, kinds, quotas, the dense grid."""

from detective_joracle.tools import positions as pos

# ChatML control ids as a Qwen tokenizer reports them; the code reads them from the tokenizer
# at run time, tests pin them so regions_of is exercised on real-looking renders.
CTL = pos.ControlIds(im_start=151644, im_end=151645, think=151667, think_end=151668)
WORD = 1000  # any ordinary token id


def render(user_tokens: list[str], reply_tokens: list[str]) -> tuple[list[int], list[str]]:
    """A Qwen render: system turn, user turn, assistant header with an empty think block."""
    toks = ["<|im_start|>", "system", "Ċ", "You", " are", " PR", "ISM", ".", "<|im_end|>", "Ċ"]
    toks += ["<|im_start|>", "user", "Ċ", *user_tokens, "<|im_end|>", "Ċ"]
    toks += ["<|im_start|>", "assistant", "Ċ", "<think>", "ĊĊ", "</think>", "ĊĊ"]
    toks += reply_tokens
    ids = []
    for t in toks:
        ids.append(
            {
                "<|im_start|>": CTL.im_start,
                "<|im_end|>": CTL.im_end,
                "<think>": CTL.think,
                "</think>": CTL.think_end,
            }.get(t, WORD)
        )
    return ids, toks


def test_regions_follow_ids_not_words() -> None:
    # the user text contains the words "assistant" and "user" — they must not open a turn
    ids, toks = render(["a", " standard", " assistant", " or", " user", "?"], ["Yes", "."])
    r = pos.regions_of(ids, toks, CTL)
    assert r[:10] == ["system"] * 10
    assert r[10] == "user" and r[toks.index("?")] == "user"
    hdr = toks.index("assistant")
    assert r[hdr - 1] == "header" and r[hdr] == "header"
    assert r[toks.index("</think>")] == "header"
    assert r[-2:] == ["reply", "reply"]


def test_is_punct_handles_fused_and_byte_level_forms() -> None:
    assert pos.is_punct(".") and pos.is_punct(".Ċ") and pos.is_punct("):") and pos.is_punct("ĊĊ")
    assert pos.is_punct('?"') and pos.is_punct(" ,")
    assert not pos.is_punct(" the") and not pos.is_punct("a.") and not pos.is_punct("v2")
    assert not pos.is_punct("-") and not pos.is_punct("(")  # not a clause delimiter


def test_thin_keeps_last_and_spreads_evenly() -> None:
    assert pos.thin(list(range(10)), 3) == [0, 4, 9]
    assert pos.thin([3, 4], 8) == [3, 4]
    assert pos.thin([1, 2, 3], 1) == [3]
    assert pos.thin([], 4) == []


def test_position_set_kinds_and_quotas() -> None:
    user = [f"w{i}" if i % 5 else "." for i in range(60)]  # 12 punctuation marks
    reply = [f"r{i}" if i % 4 else "," for i in range(64)]  # 16 punctuation marks
    ids, toks = render(user, reply)
    reply_pos = [i for i, t in enumerate(toks) if t in reply][-64:]
    tags = pos.position_set(ids, toks, CTL, reply_pos)
    kinds = {k: sorted(p for p, t in tags.items() if t.kind == k) for k in pos.KINDS}
    # boundary: user <|im_end|>, assistant role token, </think>, the header's last token
    assert len(kinds["boundary"]) == 4
    assert toks[kinds["boundary"][0]] == "<|im_end|>" and toks[kinds["boundary"][1]] == "assistant"
    assert toks[kinds["boundary"][2]] == "</think>" and kinds["boundary"][3] == reply_pos[0] - 1
    assert all(tags[p].region in ("user", "header") for p in kinds["boundary"])
    # quotas: 8 punct (4 user + 4 reply), 8 reply4, 8 user4; nothing in the system prompt
    assert len(kinds["punct"]) == 8
    assert sum(tags[p].region == "user" for p in kinds["punct"]) == 4
    assert len(kinds["reply4"]) == 8 and len(kinds["user4"]) == 8
    assert all(tags[p].region != "system" for p in tags)
    assert len(tags) <= 28
    assert list(tags) == sorted(tags)


def test_position_set_short_prompt_keeps_every_position_it_has() -> None:
    ids, toks = render(["Hi"], ["Hello", "!"])
    reply_pos = [len(toks) - 2, len(toks) - 1]
    tags = pos.position_set(ids, toks, CTL, reply_pos)
    assert {t.kind for t in tags.values()} >= {"boundary", "punct", "reply4", "user4"}
    assert toks.index("Hi") in tags and len(toks) - 1 in tags


def test_dense_grid_covers_every_non_system_position() -> None:
    ids, toks = render(["Explain", " TCP", "."], ["TCP", " is", " reliable", "."])
    dense = pos.position_set_all(ids, toks, CTL)
    regions = pos.regions_of(ids, toks, CTL)
    assert sorted(dense) == [i for i, r in enumerate(regions) if r != "system"]
    assert {t.kind for t in dense.values()} == {"boundary", "punct", "user", "header", "reply"}
    assert dense[toks.index("assistant")].kind == "boundary"
    assert dense[len(toks) - 1].kind == "punct" and dense[len(toks) - 1].region == "reply"
    assert dense[len(toks) - 2].to_json() == {"region": "reply", "kind": "reply"}
