#!/usr/bin/env bash
# X1 follow-up on the same pod, after x1_pod.sh: is 96 better picked in one pass on the full model's
# routing (fm96), or in two with the routing re-measured on the 192 it comes from (pf96)?
# Needs /workspace/x1/fm-96.json and the checkpoint already copied to $HF by x1_pod.sh.
set -euo pipefail
W=/workspace; KIT=$W/keep/scripts/runpod/v41_reap; X=$W/x1; HF=${HF:-/root/v41-hf}
export PYTHONPATH=$W/keep/src HF
log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$X/x1.log"; }
trap 'status=$?; [[ $status -ne 0 ]] && { log "x1b FAILED (status $status)"; touch "$X/FAILED-B"; }' EXIT
log "x1b: calibration masked to fm-192"
python3 "$KIT/collect.py" --hf-path "$HF" --pack "$W/pack/v41-pack.jsonl" --out "$X/mfm192" --sweep calibration \
  --mask "$X/fm-192.json" --shard "${CAL_SHARD:-0/12}" --chunk-tokens 262144 > "$X/cal-mfm192.log" 2>&1
python3 "$KIT/rerank.py" --stats "$X/mfm192/calibration/sessions" --within "$X/fm-192.json" --keep 96 --out "$X/pf-96.json" | tee -a "$X/x1.log"
log "x1b: eval teacher + fm96, pf96"
python3 "$KIT/collect.py" --hf-path "$HF" --pack "$W/pack/v41-pack.jsonl" --out "$X/eval2" --sweep eval \
  --shard "${EVAL_SHARD:-0/2}" --chunk-tokens 262144 \
  --survivors fm96="$X/fm-96.json" --survivors pf96="$X/pf-96.json" > "$X/eval2.log" 2>&1
python3 "$KIT/score.py" --teacher "$X/eval2" --student fm96="$X/eval2/student-fm96" --student pf96="$X/eval2/student-pf96" > "$X/scores-b.json"
log "x1b scores written"
touch "$X/DONE-B"
