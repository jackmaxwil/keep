#!/usr/bin/env bash
# Step 3 on one H100 pod: fetch the source checkpoint, then fit the residual sidecars.
# The Mac uploads /workspace/in/ (pack, records, student survivors, student codes) meanwhile
# and touches /workspace/in/READY when it is complete. Writes /workspace/sidecar/ and DONE.
set -euo pipefail
W=/workspace
KIT=$W/keep/scripts/runpod/v41_reap
log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$W/pod.log"; }

log "fetch + pip in parallel"
bash "$KIT/pod.sh" fetch > "$W/fetch.log" 2>&1 &
fetch_pid=$!
pip install -q -r "$KIT/requirements-ref.txt" > "$W/pip.log" 2>&1  # the pinned inference/requirements.txt
wait "$fetch_pid" || { log "fetch failed: see $W/fetch.log"; exit 1; }
log "fetch verified; waiting for the Mac upload"
while [[ ! -f "$W/in/READY" ]]; do sleep 30; done
log "fit start"
python3 "$KIT/sidecar_fit.py" fit --hf-path "$W/v41-hf" --pack "$W/in/v41-pack.jsonl" \
  --records "$W/in/records-pod.json" --student "$W/in/prunedmass-96.json" --codes "$W/in/codes" \
  --rank 256 --save-stats --device cuda --max-rss-gb 400 --out "$W/sidecar" > "$W/fit.log" 2>&1 \
  || { log "fit failed: see $W/fit.log"; exit 1; }
log "fit done: $(tail -1 "$W/fit.log")"
touch "$W/DONE"
