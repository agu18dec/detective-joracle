# Vendored global-workspace `src/` snapshot (bank containers)

`readers_modal.py` uploads `src/` (packages `global_workspace` and `jlens`) into its Modal image
as `/root/gw/src`. This is the exact snapshot the fresh banks (fresh1-4) were built with, copied
unmodified from the pod's `/workspace/agam/chive_scale/banks_code/gw_src/src` (`__pycache__`
dropped). Override with env `GW_SRC=<another src dir>`.

Source (`camilablank/global-workspace`, verified file by file):
- every file equals `src/<path>` at `origin/agam/chive-hard-cases` @
  `515f5977244f6837216a48d997299fb766f00f91` (2026-10-14), except
- `global_workspace/olens_suite/karvonen.py`, which equals `origin/camila/mlnla-viewer-am` @
  `9966287767cbca02feb4e823d2cf6f226d87d939` (2026-10-12).

Three files differ from global-workspace `main` (@ `ee52b82e23b486d140eb69b163ff4c5ff2a4a908`):
- `global_workspace/olens_suite/runner.py` (e.g. the `mlnla-sft` lens spec and `volume_dir`;
  main has the `vision=` argument of `load_base` instead);
- `global_workspace/olens_suite/karvonen.py`;
- `global_workspace/ola/verbalizer.py` (main adds anti-refusal prompt variants).

Do not edit these files; refresh the whole snapshot from a named commit instead and record it here.
`jlens/` carries its own `LICENSE` and `VENDORED.md`.
