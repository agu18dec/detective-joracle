#!/usr/bin/env bash
# Pull harness readout banks from the Modal volume into $CHIVE_BANKS/<bank>/ (default /workspace/agam/chive_scale/banks).
#   ./fetch_banks.sh <bank> [<bank> ...]      e.g. ./fetch_banks.sh jlens_all nla_k4
# Env: CHIVE_BANKS_VOLUME (default chive-fresh-banks), BANKS_DIR (default $CHIVE_BANKS).
# Only finished records (<pid>.json.gz) are pulled; per-layer partials (_parts/) are skipped.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/../env.sh"
VOL="${CHIVE_BANKS_VOLUME:-chive-fresh-banks}"
DEST_ROOT="${BANKS_DIR:-$CHIVE_BANKS}"
[ $# -ge 1 ] || { echo "usage: $0 <bank> [<bank> ...]" >&2; exit 2; }
for bank in "$@"; do
  dest="$DEST_ROOT/$bank"
  mkdir -p "$dest"
  tmp="$(mktemp -d "$DEST_ROOT/.fetch_${bank}_XXXX")"
  # one recursive get of the bank dir (fast), then keep only finished, loadable records
  modal volume get --force "$VOL" "/$bank" "$tmp/" >/dev/null
  src="$tmp/$bank"; [ -d "$src" ] || src="$tmp"
  n_new=0
  for f in "$src"/*.json.gz; do
    [ -e "$f" ] || continue
    b="$(basename "$f")"
    python3 -c 'import gzip,json,sys; d=json.load(gzip.open(sys.argv[1],"rt")); assert d["token_ids"] and d["readouts"]' "$f"
    if ! cmp -s "$f" "$dest/$b"; then mv "$f" "$dest/$b"; n_new=$((n_new + 1)); fi
  done
  rm -rf "$tmp"
  echo "$bank: $(ls "$dest" | grep -c 'json.gz$' || true) records in $dest ($n_new new/updated)"
done
