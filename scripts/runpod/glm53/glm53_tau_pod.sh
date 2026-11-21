#!/usr/bin/env bash
# Rerun of the second model with the admission threshold on the model's own routing weights.
# The runs of 2026-11-20 applied 0.3 to weights normalized to one; the first model's 0.3 applies to
# weights that sum to its routing scale. Here tau 0.3 is on GLM's routing weights (sum 2.5), the
# setting the serving code would use unchanged. Same sessions, calibration and resident counts.
set -euo pipefail
W=/workspace
KIT=$W/keep/scripts/runpod/glm53
LOCAL=${LOCAL:-/root/glm53}
OUT=$W/out
export PIP_BREAK_SYSTEM_PACKAGES=1
mkdir -p "$OUT"
log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/pod3.log"; }
trap 'status=$?; [[ $status -ne 0 ]] && { log "FAILED (status $status)"; touch "$OUT/FAILED3"; }' EXIT
(pip install -q --no-index --find-links "$W/wheels" transformers==5.22.2 > "$OUT/pip3.log" 2>&1
 pip install -q --no-index --no-deps --find-links "$W/wheels" flash-linear-attention einops >> "$OUT/pip3.log" 2>&1) &
pip_pid=$!
size() { (cd "$1" && find . -type f -not -path './.cache/*' -printf '%s\n' | awk '{s+=$1} END {print s}'); }
if [[ $(size "$W/glm53") != $(size "$LOCAL" 2>/dev/null) ]]; then
  log "copy checkpoint to $LOCAL"
  mkdir -p "$LOCAL"
  (cd "$W/glm53" && find . -type f -not -path './.cache/*' -print0 | xargs -0 -P 16 -I{} cp --parents {} "$LOCAL/")
  [[ $(size "$W/glm53") == $(size "$LOCAL") ]]
fi
wait $pip_pid
log "checkpoint in place; $(python3 "$KIT/residency.py" --selftest 2>&1 | tail -1)"
python3 "$KIT/residency.py" --ckpt "$LOCAL" --cal "$W/sessions/cal.json" --eval "$W/sessions/eval.json" \
  --max-tokens 1536 --cal-tokens 768 \
  --variant all=288:0 --variant s72=72:1e9 --variant k72=72:0.3 --variant s108=108:1e9 --variant k108=108:0.3 \
  --variant s150=150:1e9 --variant k150=150:0.3 --variant o108=108:0.3:oracle \
  --out "$OUT/run3" > "$OUT/run3.log" 2>&1
tail -10 "$OUT/run3.log" | tee -a "$OUT/pod3.log"
log "all done"
touch "$OUT/DONE3"
