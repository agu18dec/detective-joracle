"""The investigator's prompts: one scaffold, one method section per reader (or the transcript).

The scaffold is identical for every method; only ``method_section`` differs, so a difference
between arms is the method, not the instructions.
"""

from .data import Incident
from .readers import Reader, describe

ROLE = """You are an automated interpretability investigator. You are shown one conversation \
with a language model (the target model, not you), a precise description of one behaviour in \
its response, and how often that behaviour recurs when the model is re-run on the same \
conversation. Your job: find the feature of the conversation that drives the behaviour, and how \
it does so."""

TRANSCRIPT_METHOD = """## Your method: the transcript
You have the conversation and the response, nothing else. Read them closely. Causes are usually \
specific features of the conversation: an instruction or its wording, a phrase, a persona, a \
format, a topic, something an earlier turn set up, or a default the model applies when the \
request leaves something open. Use show_tokens to look at exact wording."""

READING = """## Reading the readouts: thinking vs surface
Most readouts are SURFACE: they restate the nearby input, predict the next words, describe the \
chat format, or give a generic role ("I am an AI assistant"). They show what the model is \
reading or about to write, not why. THINKING readouts say something the text does not: an \
inference about the user, a framing, genre or persona the model has chosen, a plan for the \
answer's content or structure, a judgement, an assumption or default, an alternative it weighs. \
A cause usually shows up as thinking content that appears BEFORE the response and recurs across \
samples, layers and neighbouring positions. For each readout you rely on, note its position, \
the words around it, how widely it recurs, and which feature of the conversation it attaches to. \
A readout shows what the model represents; whether that drives the behaviour is a further \
question."""

READER_TOOLS = """## Using the tools
search(pattern) finds every readout matching a regex and shows where they cluster; try several \
patterns per candidate cause. inspect(position) shows one position in full (every layer, every \
sample) with its neighbours. Search to find where a concept lives, then inspect those positions \
before relying on them: a single readout can mislead. Readouts INSIDE the response mostly restate \
what is being written at that point (surface); the model's plan and its reading of the request \
live in the positions BEFORE the response: the request itself, earlier turns, and the template \
tokens right before the response starts."""

WORKFLOW = """## How to work
Consider several candidate causes, including the one the transcript makes obvious. Investigate \
each before deciding: look for evidence for AND against. Keep notes in the notebook as you go. \
Do not settle on the first plausible story. When you submit, give ONE mechanism, the evidence, \
and the strongest rival with what ruled it out."""

EXPERIMENT = """## Experiments
You can run {n} experiment(s): edit the conversation and the target model is re-run {N} times, \
graded for the behaviour. Use it to test the candidate your investigation points to: change one \
feature and see whether the behaviour drops. {N} samples are noisy: a shift of a couple of \
responses is not evidence."""

EARLIER_TURNS = """Note: this conversation comes from a real chat log. Earlier assistant turns were \
written by the chatbot in that log, not by the target model; only the final response is the \
target model's own."""


SWITCHING = """## Weighing readouts against the transcript
Before your first readout tool call, write your best transcript-only hypothesis in the notebook. \
Readouts may overturn it only with evidence AGAINST it -- for example, the model reads that \
feature in a way that cannot produce the behaviour, or the plan for the behaviour forms before \
that feature is read -- not merely evidence that the model also represents something else. A \
concept being prominent in the readouts shows that it is represented, not that it causes the \
behaviour. If the readouts neither contradict your transcript-only hypothesis nor give a clearly \
better account of this specific behaviour, keep the transcript-only hypothesis."""

TEXTFIRST = """## Transcript first
Your readout tools are locked until you call commit_text_hypothesis with your best \
transcript-only hypothesis and how confident you are in it. Read the conversation closely before \
committing. Afterwards, use the readouts to test that hypothesis: look for evidence for AND \
against it. Overturn it only with evidence AGAINST it -- for example, the model reads that \
feature in a way that cannot produce the behaviour, or the plan for the behaviour forms before \
that feature is read -- not merely because the readouts also show something else; a concept \
being prominent in the readouts shows that it is represented, not that it causes the behaviour. \
If you do overturn it, say in `rival_ruled_out_by` what contradicted it."""

ABSENCE = """## Absences
Some causes are absences: something the conversation does not say -- no prohibition, an \
unspecified venue, format or audience, an instruction that describes rather than binds. Readouts \
cannot show an absence at any token, so neutral readouts at a position do not rule one out; judge \
absence-type candidates from the text itself."""

CONTINUE = """## Sampling the target model
continue_from(position) samples how the target model continues the text from just after that \
token (4 short samples): what it is about to write there. It shows where a position leads, not \
which feature of the conversation causes it. It is most useful inside the response: find where \
the behaviour appears and continue from just before it, and from a few earlier points, to see \
where the model commits to it. Inside a user turn the model usually just ends the turn."""


def method_section(reader: Reader | None) -> str:
    if reader is None:
        return TRANSCRIPT_METHOD
    return "\n\n".join(
        [f"## Your method: {reader.name} readouts\n{describe(reader)}", READING, READER_TOOLS]
    )


def system_prompt(
    reader: Reader | None,
    experiments: int,
    n_samples: int,
    variants: tuple[str, ...] = (),
    aux: Reader | None = None,
) -> str:
    """``variants`` (harness ablations, off by default): "switchbar" adds the bar for overturning
    the transcript-only hypothesis (reader arms); "textfirst" locks the readout tools until a
    transcript-only hypothesis is committed (reader arms); "absence" warns that readouts cannot show
    absence-type causes (reader arms); "continue" describes continue_from."""
    parts = [ROLE, method_section(reader), WORKFLOW]
    if "switchbar" in variants and reader is not None:
        parts.append(SWITCHING)
    if "textfirst" in variants and reader is not None:
        parts.append(TEXTFIRST)
    if "absence" in variants and reader is not None:
        parts.append(ABSENCE)
    if "continue" in variants:
        parts.append(CONTINUE)
    if aux is not None and reader is not None:
        parts.append(
            f"## Corroborating method: {aux.name} readouts\n{describe(aux)}\n"
            f"{aux.name}_search and {aux.name}_inspect work like search and inspect over these "
            f"readouts. Use them to check what the {reader.name} readouts suggest: a reading "
            "both methods show is stronger evidence than either alone."
        )
    if experiments:
        parts.append(EXPERIMENT.format(n=experiments, N=n_samples))
    return "\n\n".join(parts)


CANDIDATES = """## Candidate causes
Exactly one of these candidates is the cause, established by experiments on this conversation; \
the others were ruled out by experiments or are plausible but mistaken readings of the text. \
Plausibility is not evidence: investigate before you choose."""


def seed(
    inc: Incident,
    candidates: list[tuple[str, str]] | None = None,
    digest: list[tuple[int, int, str]] | None = None,
) -> str:
    """The first user message; `candidates` (letter, text) turns it into candidate selection;
    `digest` (position, layer, sample) lists the readouts worth a first look."""
    block = (
        CANDIDATES + "\n" + "\n".join(f"{a}) {t}" for a, t in candidates) + "\n\n"
        if candidates
        else ""
    )
    task = (
        "Decide which candidate is the cause, then call submit_hypothesis with its letter "
        "as `choice`."
        if candidates
        else "Find the feature that drives the behaviour, then call submit_hypothesis."
    )
    convo = "\n\n".join(
        f"T{i + 1} [{m['role']}]:\n{m['content']}" for i, m in enumerate(inc.messages)
    )
    earlier = any(m["role"] == "assistant" for m in inc.messages)
    return (
        f"## The incident\nThe target model was run {inc.n} times on this conversation. In "
        f"{inc.k} of {inc.n} runs its response met this description:\n{inc.behaviour}\n\n"
        f"## Conversation\n{convo}\n\n## The response (one run where the behaviour occurred)\n"
        f"{inc.completion}\n\n"
        + (f"{EARLIER_TURNS}\n\n" if earlier else "")
        + block
        + (
            "## Readouts worth a first look\nPositions before the response whose readouts most "
            "consistently say something the nearby text does not (picked automatically, one "
            "sample each; check them with inspect):\n"
            + "\n".join(
                f"- {inc.where(p)} L{layer}: {' '.join(smp.split())[:220]}"
                for p, layer, smp in digest
            )
            + "\n\n"
            if digest
            else ""
        )
        + "Positions are tokens t0..t"
        + str(inc.n_tokens - 1)
        + "; the response starts at "
        f"t{inc.response_start}. {task}"
    )


CONCLUDER_SYSTEM = """You are an investigator writing the conclusion of an investigation \
someone else ran. You see a conversation with a language model, a behaviour in its response \
and how often it recurs, and the experiment(s) the other investigator ran: the edit, how often \
the behaviour then occurred, and some responses. You do not see their reasoning. From the \
conversation and these results, state the ONE feature of the conversation that drives the \
behaviour and how."""
CONCLUDER_SCHEMA = {
    "type": "object",
    "required": ["hypothesis", "evidence"],
    "properties": {"hypothesis": {"type": "string"}, "evidence": {"type": "string"}},
}

TOP20_SYSTEM = """You see a conversation with a language model, a behaviour in its response \
and how often it recurs. List the 20 most likely explanations -- each one specific feature of \
the conversation and how it drives the behaviour -- most likely first, all distinct."""
TOP20_SCHEMA = {
    "type": "object",
    "required": ["hypotheses"],
    "properties": {
        "hypotheses": {"type": "array", "items": {"type": "string"}, "minItems": 20, "maxItems": 20}
    },
}
