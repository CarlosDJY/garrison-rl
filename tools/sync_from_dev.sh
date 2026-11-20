#!/bin/bash
# Copies the working files from where the training runs (the asp-assistant-arknights checkout and ../sp-gym) into this
# repository: code, launchers and the SP-era reports. Data, checkpoints and logs stay behind (.gitignore).
#   bash tools/sync_from_dev.sh [dev root, default: the folder this repository sits in]
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEV="${1:-$(dirname "$HERE")}"
SRC="$DEV/asp-assistant-arknights"
mkdir -p "$HERE/python/garrison_rl/runs" "$HERE/sp-gym/patches" "$HERE/reports/playground"
for f in "$SRC"/python/garrison_rl/*.py; do
  # The header pointing at the other repository's NOTICE does not apply here.
  grep -v "^# \(Added\|Modified\) for ASP Assistant Regulation .*see NOTICE\.$" "$f" > "$HERE/python/garrison_rl/$(basename "$f")"
done
cp "$SRC"/python/garrison_rl/runs/run_sp*.sh "$HERE/python/garrison_rl/runs/"
cp "$DEV/sp-gym/spgym.mjs" "$HERE/sp-gym/"
cp "$DEV"/sp-gym/patches/*.patch "$HERE/sp-gym/patches/"
cp "$SRC"/reports/playground/sp*.md "$HERE/reports/playground/"
echo "synced from $DEV"
