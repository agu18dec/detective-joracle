# Contributing

## Dev setup

```
uv venv --python 3.12 .venv && uv pip install -e ".[dev]"
.venv/bin/ruff check src tests examples scripts && .venv/bin/ruff format --check src tests examples scripts
.venv/bin/mypy                        # src + tests, strict
.venv/bin/python -m pytest -q         # offline; the mock target runs in-process
examples/run_end_to_end.sh            # the pipeline against the mock, no keys
```

CI (`.github/workflows/ci.yml`) runs exactly those five commands on Python 3.12. `servers/` is
deliberately outside ruff/mypy: those are Modal launchers over a private lens stack, kept as
readable reference implementations.

Conventions: Python ≥ 3.11 syntax (no `from __future__ import annotations`), type hints on every
def, `ruff format` decides layout (line length 100). Module docstrings say what the module is
and where it sits in the pipeline; public functions get a one-line summary plus the failure
modes that matter. No essays in code — rationale goes in `docs/`.

## Module boundaries

```
tools/arms.py         arm names and the lens registry (no other package imports)
agent/prompts.py      every string the investigator sees        -> imports tools/arms
agent/loop.py         the loop, RunRecord, Budget, context caps  -> imports agent/prompts
agent/backends.py     LLM backends for the investigator          -> imports agent/loop
presentation/         readout grid -> page                       -> imports llm/route (lazily)
tools/live.py         LiveClient, LiveTools, run_live_agent      -> imports agent/*, presentation
judges/               paper + graded judges                      -> imports llm/route, registry
registry/             data files + loaders                       (no package imports)
llm/                  async_json, OpenRouter route               (no package imports)
scripts/              batch drivers over the package             (never imported by src/)
```

`src/` must never import from `scripts/`, `servers/` or `examples/`, and never from the private
lens stack. Tests are offline: use `tools.fake.FakeClient`, `agent.backends.FakeBackend`, or the
in-process mock (`examples/mock_target.py::serve`) and monkeypatch
`detective_joracle.llm.route.async_json_route` for any judge/auxiliary call.

## Where a new thing goes

Recipes with the exact functions to touch are in `docs/extending.md`. In one line each:

* **lens** — implement the readout contract server-side; `register_lens_context()` client-side.
  No harness code changes for a lens that fits the contract.
* **presentation** — a function in `presentation/select.py` returning `(res, note)`, a name in
  `MODES`, a branch in `LiveTools.readouts`.
* **judge** — a prompt + `_score_schema` entry + a branch in `judges/graded.py::judge_records`,
  a column in `scripts/run_audit.py::_collect` / `stage_report`.
* **tool** — a schema in `tools/live.py::tool_schemas`, a method on `LiveTools`, a branch in
  `LiveTools.call`, and the arm rule in `tool_names` (+ an evidence paragraph in
  `agent/prompts.py` if the agent should be told).
* **auditor backend** — `agent/backends.py::make_backend_factory`, or pass any
  `openai.OpenAI`-style client to `openai_compatible_backend`.

## Pull requests

Keep the record schema backward compatible (`docs/architecture.md`): the viewer and the report
read old records. Add a test for every new branch; if it needs an LLM, monkeypatch the route.
Update `CHANGELOG.md`.
