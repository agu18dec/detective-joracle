"""Byte-level BPE surface forms -> readable text.

Tokenizers of the GPT-2/Qwen family store vocabulary strings through ``bytes_to_unicode`` (``Ġ``
for a space, ``Ċ`` for a newline, CJK as mojibake). Anything shown to the agent — a token label
on a readout page, a logit-lens top-k list — goes through :func:`decode_byte_level` first.
"""


def _byte_level_inverse() -> dict[str, int]:
    """Inverse of the GPT-2/Qwen byte-level BPE ``bytes_to_unicode`` table."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


_BYTE_INVERSE = _byte_level_inverse()


def decode_byte_level(token: str) -> str:
    """Undo the byte-level BPE surface form (``çļĦ`` -> ``的``, ``Ġthe`` -> `` the``, ``ĊĊ`` ->
    two newlines). A capture may already have mapped ``Ġ``/``▁`` to a real space, so a token can
    be ``" ĊĊ"``: each space-separated part is decoded when EVERY character is in the byte
    alphabet, and left alone otherwise (already-decoded text, e.g. ``**—`` or CJK).
    """

    def part(p: str) -> str:
        """Decode one space-separated part if every character is in the byte alphabet."""
        if not p or any(ch not in _BYTE_INVERSE for ch in p):
            return p
        return bytes(_BYTE_INVERSE[ch] for ch in p).decode("utf-8", errors="replace")

    return " ".join(part(p) for p in token.split(" "))
