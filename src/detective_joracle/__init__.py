"""detective-joracle: audit a language model for a hidden behavior with an investigator agent.

Pipeline, in data-flow order:

- ``tools.live``       the agent's affordances: chat / prefill / completion on the target, and a
                       lens ``readouts`` tool over any conversation it ran (HTTP to your servers)
- ``presentation``     how a readout grid is shaped into the page the agent reads
- ``agent``            the tool-calling loop, its budget, the prompts, the LLM backends
- ``judges``           the paper's binary judge and the graded / micro judges
- ``registry``         the frozen quirk registry, distractors and held-out prompts (data files)
- ``llm``              a minimal OpenAI-compatible structured-JSON caller (+ OpenRouter route)

The main entry points are re-exported here; ``scripts/run_audit.py`` is the batch driver.
"""

from .agent.backends import FakeBackend, make_backend_factory, openai_compatible_backend
from .agent.loop import Backend, Budget, RunRecord, run_tool_loop
from .agent.prompts import register_lens_context, system_prompt
from .judges.graded import judge_records
from .judges.paper import judge_runs
from .presentation.select import MODES
from .registry.quirks import load_held_out_prompts, load_quirk_registry, quirk_of
from .tools.arms import LIVE_ARMS, lens_of, valid_arm
from .tools.live import LensClient, LiveClient, LiveTools, run_live_agent

__version__ = "0.1.0"

__all__ = [
    "LIVE_ARMS",
    "MODES",
    "Backend",
    "Budget",
    "FakeBackend",
    "LensClient",
    "LiveClient",
    "LiveTools",
    "RunRecord",
    "__version__",
    "judge_records",
    "judge_runs",
    "lens_of",
    "load_held_out_prompts",
    "load_quirk_registry",
    "make_backend_factory",
    "openai_compatible_backend",
    "quirk_of",
    "register_lens_context",
    "run_live_agent",
    "run_tool_loop",
    "system_prompt",
    "valid_arm",
]
