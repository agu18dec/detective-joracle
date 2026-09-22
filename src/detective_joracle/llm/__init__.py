"""LLM plumbing for the auxiliary calls (judges, scenario generation, readout triage).

``client.async_json`` is a minimal OpenAI-compatible structured-JSON caller; ``openrouter``
re-points it at OpenRouter with pacing and retries; ``route.async_json_route`` is the one entry
point the rest of the package uses.
"""
