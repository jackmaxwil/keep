#!/usr/bin/env bash
# X1 on one GPU pod: "prune twice" at the first cut (384 -> 288 -> 192 -> 96), scored against the
# shipped REAP tables. Plan: docs/deepseek-v4-flash/2026-11-09-routed-precision-plan.md, section 7.
#
# The volume already holds everything (a CPU pod put it there, so the GPU starts work at once):
#   /workspace/v41-hf      the pinned checkpoint, fetch verified
#   /workspace/pack        the calibration + eval pack
#   /workspace/wheels      pip wheels for inference/requirements.txt
#   /workspace/x1/*.json   fm-288, fm-192 (total mass on full-model routing), survivors-192 (shipped
#                          REAP-50), pm-96 (the Mac's pick inside REAP-50)
set -euo pipefail
W=/workspace
KIT=$W/keep/scripts/runpod/v41_reap
HF=${HF:-$W/v41-hf}
X=$W/x1
CAL_SHARD=${CAL_SHARD:-0/12}        # 2 of 24 calibration sessions, about 256K tokens, every domain
EVAL_SHARD=${EVAL_SHARD:-0/2}       # 54 of 108 eval sessions
CHUNK=${CHUNK_TOKENS:-262144}
export PYTHONPATH=$W/keep/src MAX_GPU_HOURS=${MAX_GPU_HOURS:-9}
export PIP_BREAK_SYSTEM_PACKAGES=1  # Ubuntu 24.04 images mark the system Python as externally managed
log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$X/x1.log"; }
trap 'status=$?; [[ $status -ne 0 ]] && { log "FAILED (status $status)"; touch "$X/FAILED"; }' EXIT

# The network volume reads one stream at about 120 MB/s and a random Engram row in about 3 ms, so the
# smoke test sat at 0% GPU for 50 minutes. Sixteen parallel streams read 3 GB/s: copy the checkpoint
# to the container's own disk first (LOCAL_HF, needs ~480 GB there) and run from that.
if [[ -n "${LOCAL_HF:-}" ]]; then
  log "copy checkpoint to $LOCAL_HF"
  mkdir -p "$LOCAL_HF"
  (cd "$W/v41-hf" && find . -type f -not -path './.cache/*' -print0 | xargs -0 -P 16 -I{} cp --parents {} "$LOCAL_HF/")
  [[ $(cd "$W/v41-hf" && find . -type f -not -path './.cache/*' -printf '%s\n' | awk '{s+=$1} END {print s}') == $(cd "$LOCAL_HF" && find . -type f -printf '%s\n' | awk '{s+=$1} END {print s}') ]]
  HF=$LOCAL_HF
  log "copy done"
fi
export HF

pip install -q --no-index --find-links "$W/wheels" -r "$KIT/requirements-ref.txt" \
  || pip install -q -r "$KIT/requirements-ref.txt"
pip freeze > "$W/pip-freeze-gpu.txt"   # pod.sh's gpu_setup then skips its own install
nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a "$X/x1.log"

log "smoke"
bash "$KIT/pod.sh" smoke

calibrate() {  # name, mask
  log "calibration masked to $2 ($CAL_SHARD)"
  python3 "$KIT/collect.py" --hf-path "$HF" --pack "$W/pack/v41-pack.jsonl" --out "$X/$1" --sweep calibration \
    --mask "$2" --shard "$CAL_SHARD" --chunk-tokens "$CHUNK" > "$X/cal-$1.log" 2>&1
  log "calibration $1 done: $(tail -1 "$X/cal-$1.log")"
}
calibrate m288 "$X/fm-288.json"
python3 "$KIT/rerank.py" --stats "$X/m288/calibration/sessions" --within "$X/fm-288.json" --keep 192 --out "$X/pt-192.json" | tee -a "$X/x1.log"
calibrate m192 "$X/pt-192.json"
python3 "$KIT/rerank.py" --stats "$X/m192/calibration/sessions" --within "$X/pt-192.json" --keep 96 --out "$X/pt-96.json" | tee -a "$X/x1.log"

log "eval ($EVAL_SHARD): teacher + reap192, fm192, pt192, pm96, pt96"
python3 "$KIT/collect.py" --hf-path "$HF" --pack "$W/pack/v41-pack.jsonl" --out "$X/eval" --sweep eval \
  --shard "$EVAL_SHARD" --chunk-tokens "$CHUNK" \
  --survivors reap192="$X/survivors-192.json" --survivors fm192="$X/fm-192.json" --survivors pt192="$X/pt-192.json" \
  --survivors pm96="$X/pm-96.json" --survivors pt96="$X/pt-96.json" > "$X/eval.log" 2>&1
python3 "$KIT/score.py" --teacher "$X/eval" --student reap192="$X/eval/student-reap192" --student fm192="$X/eval/student-fm192" \
  --student pt192="$X/eval/student-pt192" --student pm96="$X/eval/student-pm96" --student pt96="$X/eval/student-pt96" > "$X/scores.json"
log "scores written"
touch "$X/DONE"
