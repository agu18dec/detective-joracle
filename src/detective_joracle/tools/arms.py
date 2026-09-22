"""Arm names: which affordances a run has.

An arm is ``[scaffold-][<lens>[-fixed] | blackbox][-ask]``:

* ``blackbox``       chat tools only — the floor every lens arm is read against
* ``<lens>``         + ``readouts`` on any conversation the agent ran (the in-the-loop arm)
* ``<lens>-fixed``   + the paper's parameterless ``fixed_prompt_readouts`` (precomputed on the
                     held-out prompts; App. E.3/E.4)
* ``scaffold-…``     + ``generate_and_test`` (App. F: an auxiliary model writes probe scenarios)
* ``…-ask``          + ``ask_claude``, a consultant probe (not a headline affordance)

The lens names known out of the box are the keys of :data:`LENS_NAME`; register another with
:func:`register_lens` (which also wants a context paragraph — see ``agent.prompts``).
"""

# lens name -> display name; the keys are the lens ids a server's ``readout`` endpoint accepts
LENS_NAME: dict[str, str] = {
    "olens": "OLens (verbalizer)",
    "jlens": "Jacobian lens",
    "logit": "Logit lens",
    "nla": "NLA-RL (verbalizer)",
}


def _arms_for(lenses: tuple[str, ...]) -> tuple[str, ...]:
    core = (
        "blackbox",
        *[f"{lens}-fixed" for lens in lenses],
        *lenses,
        "scaffold",
        *[f"scaffold-{lens}-fixed" for lens in lenses],
        *[f"scaffold-{lens}" for lens in lenses],
    )
    return core + tuple(f"{a}-ask" for a in core)


LIVE_ARMS: tuple[str, ...] = _arms_for(tuple(LENS_NAME))


def register_lens(name: str, display: str) -> None:
    """Make ``name`` a valid lens id for arm names (``name``, ``name-fixed``, ``scaffold-name`` …)."""
    global LIVE_ARMS
    LENS_NAME[name] = display
    LIVE_ARMS = _arms_for(tuple(LENS_NAME))


def has_scaffold(arm: str) -> bool:
    """``scaffold-…``: the arm has ``generate_and_test``."""
    return arm.startswith("scaffold")


def has_ask(arm: str) -> bool:
    """``-ask``: the agent also gets ``ask_claude``, advertised as a more capable model it may
    consult. A probe, not an affordance we report headline numbers for — what it measures is how
    often the investigator outsources the judgement."""
    return arm.endswith("-ask")


def core_arm(arm: str) -> str:
    """The arm without its ``-ask`` suffix."""
    return arm.removesuffix("-ask")


def lens_of(arm: str) -> str | None:
    """The lens id an arm reads with, or None for ``blackbox`` / ``scaffold``."""
    rest = core_arm(arm).removeprefix("scaffold").strip("-")
    return None if rest in ("", "blackbox") else rest.split("-")[0]


def is_fixed(arm: str) -> bool:
    """``<lens>-fixed``: the paper's precomputed fixed-prompt tool instead of live readouts."""
    return core_arm(arm).endswith("-fixed")


def valid_arm(arm: str) -> bool:
    """Structurally valid AND naming a registered lens (or no lens)."""
    rest = core_arm(arm).removeprefix("scaffold").strip("-")
    if rest in ("", "blackbox"):
        return True
    lens = rest.removesuffix("-fixed")
    return lens in LENS_NAME and rest in (lens, f"{lens}-fixed")
