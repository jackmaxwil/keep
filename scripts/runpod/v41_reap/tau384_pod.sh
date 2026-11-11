#!/usr/bin/env bash
# Tau sweep over the pool of 384 on one GPU pod.
#
# 1. Fit 2-bit E8P codes for all 384 experts of every layer (keep_quant_torch.py, the fit the Mac
#    serves; uploading the Mac's 130 GB of codes would take 12 h at its 3 MB/s uplink).
# 2. Simulate the expert cache that omlx_lattice.py serves (local_eval.py: 144 residents, re-route
#    on a miss) at several admission thresholds, scored against the teacher's logits on the Mac's
#    22 eval sessions (first 1,024 tokens each).
#
# On the volume before the GPU starts (a CPU pod put it there):
#   /workspace/v41-hf, /workspace/pack, /workspace/wheels, /workspace/keep
#   /workspace/pool384/{imp-all.npz,reap-stats.npz,sessions.json}
#   /workspace/pool384/stats/eval/sessions/NAME.npz   teacher logits
# Output on the volume: pool384/codes/e8p-384 (the codes), pool384/tau-sweep/summary.json, DONE.
set -euo pipefail
W=/workspace
P=$W/pool384
KIT=$W/keep/scripts/runpod/v41_reap
LOCAL=${LOCAL:-/root/v41}
CODES=$LOCAL/codes/e8p-384
export PYTHONPATH=$W/keep/src PIP_BREAK_SYSTEM_PACKAGES=1
log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$P/pod.log"; }
trap 'status=$?; [[ $status -ne 0 ]] && { log "FAILED (status $status)"; touch "$P/FAILED"; }' EXIT

# Never run from the volume (one stream is 120 MB/s): the checkpoint goes to the container's
# disk with 16 streams (about 480 GB).
log "copy checkpoint to $LOCAL/hf"
mkdir -p "$LOCAL/hf"
(cd "$W/v41-hf" && find . -type f -not -path './.cache/*' -print0 | xargs -0 -P 16 -I{} cp --parents {} "$LOCAL/hf/")
[[ $(cd "$W/v41-hf" && find . -type f -not -path './.cache/*' -printf '%s\n' | awk '{s+=$1} END {print s}') == $(cd "$LOCAL/hf" && find . -type f -printf '%s\n' | awk '{s+=$1} END {print s}') ]]
log "copy done: $(du -sh "$LOCAL/hf" | cut -f1)"

pip install -q --no-index --find-links "$W/wheels" -r "$KIT/requirements-ref.txt" \
  || pip install -q -r "$KIT/requirements-ref.txt"
nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a "$P/pod.log"

fit() { python3 "$KIT/keep_quant_torch.py" fit --hf-path "$LOCAL/hf" --importance "$P/imp-all.npz" "$@"; }
log "fit smoke: four experts of layer 0"
rm -rf "$LOCAL/fit-smoke"
fit --out "$LOCAL/fit-smoke" --layers 0 --experts 4 | tee -a "$P/pod.log"
hours=$(python3 -c "import json,sys; print(round(json.loads(open(sys.argv[1]).readline())['seconds'] / 4 * 384 * 40 / 3600, 2))" "$LOCAL/fit-smoke/metrics.jsonl")
log "fit of all 15,360 experts projected at $hours h"
python3 -c "import sys; sys.exit(float(sys.argv[1]) > float(sys.argv[2]))" "$hours" "${MAX_FIT_H:-4}" || { log "over MAX_FIT_H, stopping"; exit 1; }
fit --out "$CODES" > "$P/fit.log" 2>&1
[[ $(find "$CODES" -name 'layer-??.npz' | wc -l) == 40 ]]
log "fit done: $(tail -1 "$P/fit.log")"
mkdir -p "$P/codes/e8p-384"   # keep the codes for later pod runs; the copy runs beside the sweep
(cd "$CODES" && find . -type f -print0 | xargs -0 -P 8 -I{} cp {} "$P/codes/e8p-384/") &
keep=$!

run() {
  python3 "$KIT/local_eval.py" --device cuda --hf-path "$LOCAL/hf" --pack "$W/pack/v41-pack.jsonl" --stats "$P/stats" \
    --max-rss-gb 150 --cache-warm "$P/reap-stats.npz" --dynamic "|$CODES" "$@"
}
v() { echo --variant "$1=all" --tau "$1=$3" --cache-k "$1=$2" --miss "$1=reroute" ${4:+--cache-policy "$1=$4"}; }

log "sweep smoke: one session, 128 tokens"
echo '["eval-00000"]' > "$P/smoke-sessions.json"
# shellcheck disable=SC2046
run --sessions-file "$P/smoke-sessions.json" --max-tokens 128 $(v q384 384 0) $(v t20 144 0.2) --out "$P/smoke-out" 2>&1 | tail -5 | tee -a "$P/pod.log"

# q384: every expert resident (2-bit coding alone). s144: the 144 hottest, never changed.
# tNN: the cache of 144 at tau 0.NN. o144: each record's own best 144 (the policy's ceiling).
log "sweep: 22 sessions, 1,024 tokens, 7 variants"
# shellcheck disable=SC2046
run --sessions-file "$P/sessions.json" --max-tokens 1024 \
  $(v q384 384 0) $(v s144 144 1e9) $(v t10 144 0.1) $(v t20 144 0.2) $(v t30 144 0.3) $(v t45 144 0.45) $(v o144 144 0.2 oracle) \
  --out "$P/tau-sweep" > "$P/sweep.log" 2>&1
python3 "$KIT/routed_report.py" --eval "$P/tau-sweep" | tee -a "$P/pod.log"
wait $keep
[[ $(find "$P/codes/e8p-384" -name 'layer-??.npz' | wc -l) == 40 ]]
log "all done"
touch "$P/DONE"
