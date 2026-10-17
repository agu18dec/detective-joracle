#!/usr/bin/env bash
# Pull finished bank records for specific incidents (safe while other bank jobs write _parts/).
#   ./fetch_ids.sh <ids.txt> <bank> [<bank> ...]
set -uo pipefail
source "$(dirname "$0")/../env.sh"
IDS=$1; shift
for bank in "$@"; do
  dest=$CHIVE_BANKS/$bank; mkdir -p "$dest"
  grep -v '^$' "$IDS" | xargs -P 4 -I{} sh -c "modal volume get --force ${CHIVE_BANKS_VOLUME:-chive-fresh-banks} /$bank/{}.json.gz $dest/{}.json.gz >/dev/null 2>&1 || echo MISSING $bank {}"
  echo "$bank: $(comm -12 <(ls $dest | sed 's/.json.gz//' | sort) <(sort $IDS) | wc -l)/$(grep -c . $IDS)"
done
