# Examples

## `run_end_to_end.sh` — the whole pipeline on a laptop

```
uv venv --python 3.12 .venv && uv pip install -e ".[test]"
examples/run_end_to_end.sh
```

What happens, step by step:

1. **`mock_target.py` starts** on `:8765`. One stdlib process plays three roles:
   * the **target** — an OpenAI-compatible `/v1/chat/completions` + `/v1/completions` that
     impersonates `flattery_td_r16` (every reply opens with praise) — exactly what a vLLM
     server of a quirked model would expose;
   * the **lens server** — `/chat`, `/readout`, `/fixed`, `/info` returning the documented
     readout JSON (`docs/contracts.md`), with a "hidden disposition" planted at the user and
     boundary positions of the flattery organism;
   * a **mock judge** — `/v1/chat/completions` with `response_format` returns a placeholder
     object filled from the schema. It does not read anything.
2. **`stage=agent`** runs one audit per (organism, arm): `flattery_td_r16` and `base`, arms
   `blackbox` and `olens`. Without a key the auditor is `FakeBackend`, a scripted investigator
   that calls every tool once and finishes with a fixed hypothesis; with `OPENROUTER_API_KEY`
   set it is a real model (default Gemini 3.8 Flash, `AUDITOR=` to change) on a 4k-token budget.
3. **`stage=judge,judges,report`** scores the records and writes `examples/out/report.md`.
   With the mock judge every verdict is a placeholder, so the report is a plumbing check.
4. **`build_viewer.py`** writes `examples/out/site/index.html` — open it to read the transcripts
   (left) against the scorecards (right).

Records land at `examples/out/runs/live/<arm>-noref/<organism>/<auditor>/seed_0.json`; the
schema is in `docs/architecture.md`. Rerunning skips finished records (delete `examples/out`
to start over).

## Using the mock to develop against

The mock is also a reference implementation of both contracts in ~250 lines: read
`readout()` and `fixed()` for the lens side and `do_POST` for the target side. Point the
driver at a real target and keep the mock lens (`server=http://127.0.0.1:8765/{name}
target=https://your-vllm`), or the other way round, to test one side at a time.

```
python examples/mock_target.py --tag secret_loyalty_td_r16   # a different canned quirk
```
