#!/usr/bin/env bash
# Seed a new run directory with another run's checkpoint at STEP and its metrics up to STEP,
# so the new run resumes from there under its own config (e.g. a lower LR) and AutoStop and
# DevSelection see the earlier steps' baseline and dev scores.
#   scripts/arggym/seed_from.sh outputs/<old-run> outputs/<new-run> 20
set -euo pipefail
OLD=$1 NEW=$2 STEP=$3
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
[ -e "$NEW" ] && { echo "$NEW exists" >&2; exit 1; }
[ -d "$OLD/checkpoints/global_step_$STEP" ] || { echo "no global_step_$STEP in $OLD" >&2; exit 1; }
mkdir -p "$NEW/checkpoints"
cp -r "$OLD/checkpoints/global_step_$STEP" "$NEW/checkpoints/"
echo "$STEP" > "$NEW/checkpoints/latest_checkpointed_iteration.txt"
# run.sh keeps metrics*.jsonl across launches; this name sorts before any launch's own file.
"$ROOT/.venv/bin/python" - "$OLD/metrics.jsonl" "$NEW/metrics.00000000T000000Z.jsonl" "$STEP" <<'PY'
import json, sys
src, dst, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(dst, "w") as out:
    for line in open(src):
        if line.strip() and json.loads(line)["step"] <= step:
            out.write(line)
PY
echo "seeded $NEW from $OLD step $STEP"
