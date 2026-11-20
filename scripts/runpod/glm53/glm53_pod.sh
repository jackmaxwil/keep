#!/usr/bin/env bash
# Second model of the case study, on one GPU pod: routed residency against static sets on
# GLM-5.3-Flash at its native precision (residency.py). Settings are fixed before the run:
# residents at the pool shares the first model was scored at (25%, 37.5% and 52% of 288 experts:
# 72, 108, 150), the first model's admission threshold (0.3) unchanged, and one sensitivity point
# (0.2 at 108). "all" holds every expert and must reproduce the full model.
#
# On the volume before the GPU starts (a CPU pod put it there): /workspace/glm53 (the pinned
# checkpoint, 328 GB) and /workspace/wheels. Copied in by hand: /workspace/keep (this repo) and
# /workspace/sessions/{eval,cal}.json (prep_sessions.py). Output: /workspace/out.
set -euo pipefail
W=/workspace
KIT=$W/keep/scripts/runpod/glm53
LOCAL=${LOCAL:-/root/glm53}
OUT=$W/out
export PIP_BREAK_SYSTEM_PACKAGES=1
mkdir -p "$OUT"
log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/pod.log"; }
trap 'status=$?; [[ $status -ne 0 ]] && { log "FAILED (status $status)"; touch "$OUT/FAILED"; }' EXIT

(pip install -q --no-index --find-links "$W/wheels" transformers==5.22.2 > "$OUT/pip.log" 2>&1
 pip install -q --no-index --no-deps --find-links "$W/wheels" flash-linear-attention einops >> "$OUT/pip.log" 2>&1
 echo "pip exit $?" >> "$OUT/pip.log") &
pip_pid=$!
size() { (cd "$1" && find . -type f -not -path './.cache/*' -printf '%s\n' | awk '{s+=$1} END {print s}'); }
if [[ $(size "$W/glm53") != $(size "$LOCAL" 2>/dev/null) ]]; then
  log "copy checkpoint to $LOCAL"
  mkdir -p "$LOCAL"
  (cd "$W/glm53" && find . -type f -not -path './.cache/*' -print0 | xargs -0 -P 16 -I{} cp --parents {} "$LOCAL/")
  [[ $(size "$W/glm53") == $(size "$LOCAL") ]]
fi
log "checkpoint in place: $(du -sh "$LOCAL" | cut -f1)"
wait $pip_pid
tail -1 "$OUT/pip.log" | tee -a "$OUT/pod.log"
python3 -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))" | tee -a "$OUT/pod.log"
python3 "$KIT/residency.py" --selftest 2>&1 | tail -1 | tee -a "$OUT/pod.log"

run() { python3 "$KIT/residency.py" --ckpt "$LOCAL" --cal "$W/sessions/cal.json" --eval "$W/sessions/eval.json" "$@"; }
log "smoke: five layers, 128 tokens"
run --layers 5 --max-tokens 128 --cal-tokens 64 --variant all=288:0 --variant s108=108:1e9 --variant k108=108:0.3 \
  --out "$OUT/smoke" > "$OUT/smoke.log" 2>&1
tail -4 "$OUT/smoke.log" | tee -a "$OUT/pod.log"

log "full run"
run --max-tokens 1536 --cal-tokens 768 \
  --variant all=288:0 \
  --variant s72=72:1e9 --variant k72=72:0.3 --variant s108=108:1e9 --variant k108=108:0.3 \
  --variant s150=150:1e9 --variant k150=150:0.3 --variant k108t20=108:0.2 \
  --out "$OUT/run" > "$OUT/run.log" 2>&1
tail -10 "$OUT/run.log" | tee -a "$OUT/pod.log"
log "all done"
touch "$OUT/DONE"
