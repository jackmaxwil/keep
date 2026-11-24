#!/usr/bin/env bash
# Residents against bits on GLM-5.3-Flash, on one GPU pod (any GPU count). The first model's order, on the second:
# 2-bit E8P codes for every routed expert, a 1-bit residual for the 96 heaviest experts per layer,
# then the simulator with coded variants (residency.py ``:codes``). The teacher stays at FP8.
#
# Settings fixed before the run: the sessions, calibration and threshold of run 3 (tau 0.3 on the
# model's routing weights, 1536 evaluation tokens, 768 calibration tokens); residents at the pool
# shares the first model was scored at. A 1-bit residual costs half a 2-bit resident, so each
# 2-bit row has equal-memory rows with 32 or 64 hot experts. FP8 rows of run 3 are repeated in the
# same harness (k72, k108, k150), and "all" must reproduce the full model.
#
# Any GPU count: each stage runs one command per GPU, GPUS (default: all) at a time.
# On the volume (a CPU pod put it there): /workspace/glm53 (328 GB), /workspace/wheels,
# /workspace/keep (this repo), /workspace/sessions/{eval,cal}.json. Output: /workspace/out-bits;
# codes in /workspace/glm-codes/{codes2,resid1} (77 GB and 13 GB; finished layers are skipped, so
# a rerun resumes).
set -euo pipefail
W=/workspace
KIT=$W/keep/scripts/runpod/glm53
LOCAL=${LOCAL:-/root/glm53}
OUT=$W/out-bits
C=$W/glm-codes
GPUS=${GPUS:-$(nvidia-smi -L | wc -l)}
export PIP_BREAK_SYSTEM_PACKAGES=1 PYTHONPATH=$W/keep/src
mkdir -p "$OUT" "$C"
log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/pod.log"; }
trap 'status=$?; [[ $status -ne 0 ]] && { log "FAILED (status $status)"; touch "$OUT/FAILED"; }' EXIT

(pip install -q --no-index --find-links "$W/wheels" transformers==5.22.2 > "$OUT/pip.log" 2>&1
 pip install -q --no-index --no-deps --find-links "$W/wheels" flash-linear-attention einops >> "$OUT/pip.log" 2>&1
 # mlx (CPU build) only because keep.quant imports it; fit_codes check needs KEEP's numpy fit
 pip install -q --no-index --no-deps --find-links "$W/wheels" mlx mlx-cpu >> "$OUT/pip.log" 2>&1
 echo "pip exit $?" >> "$OUT/pip.log") &
pip_pid=$!
size() { (cd "$1" && find . -type f -not -path './.cache/*' -printf '%s\n' | awk '{s+=$1} END {print s}'); }
if [[ $(size "$W/glm53") != $(size "$LOCAL" 2>/dev/null) ]]; then
  log "copy checkpoint to $LOCAL"
  mkdir -p "$LOCAL"
  (cd "$W/glm53" && find . -type d -not -path './.cache*' -exec mkdir -p "$LOCAL/{}" \;)  # parallel cp --parents races on mkdir
  (cd "$W/glm53" && find . -type f -not -path './.cache/*' -print0 | xargs -0 -P 16 -I{} cp --parents {} "$LOCAL/")
  [[ $(size "$W/glm53") == $(size "$LOCAL") ]]
fi
log "checkpoint in place: $(du -sh "$LOCAL" | cut -f1)"
wait $pip_pid
tail -1 "$OUT/pip.log" | tee -a "$OUT/pod.log"
python3 -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))" | tee -a "$OUT/pod.log"
python3 "$KIT/residency.py" --selftest 2>&1 | tail -2 | tee -a "$OUT/pod.log"

# Run a stage's commands one per GPU, GPUS at a time, and fail if any fails.
fan() {
  local pids=() g=0 p
  while read -r cmd; do
    CUDA_VISIBLE_DEVICES=$g bash -c "$cmd" & pids+=($!)
    g=$((g + 1))
    if ((g == GPUS)); then for p in "${pids[@]}"; do wait "$p"; done; pids=(); g=0; fi
  done
  for p in "${pids[@]}"; do wait "$p"; done
}
# Layers 3 to 44, dealt round-robin to the GPUs.
share() { seq "$((3 + $1))" "$GPUS" 44 | paste -sd,; }

if [[ ! -f $C/imp-glm.npz ]]; then
  log "importance pass (calibration only)"
  python3 "$KIT/residency.py" --ckpt "$LOCAL" --cal "$W/sessions/cal.json" --cal-tokens 768 \
    --importance "$C/imp-glm.npz.tmp.npz" --out "$OUT/imp" > "$OUT/imp.log" 2>&1
  mv "$C/imp-glm.npz.tmp.npz" "$C/imp-glm.npz"
  tail -1 "$OUT/imp.log" | tee -a "$OUT/pod.log"
fi
python3 "$KIT/fit_codes.py" check --ckpt "$LOCAL" --importance "$C/imp-glm.npz" --rows 64 2>&1 | tail -7 | tee -a "$OUT/pod.log"

log "2-bit fit, every expert"
for g in $(seq 0 $((GPUS - 1))); do
  echo "python3 $KIT/fit_codes.py fit --ckpt $LOCAL --importance $C/imp-glm.npz --out $C/codes2 --layers $(share "$g") > $OUT/fit2-$g.log 2>&1"
done | fan
[[ $(ls "$C"/codes2/layer-*.npz | wc -l) == 42 ]]
log "1-bit residuals, 96 experts per layer"
for g in $(seq 0 $((GPUS - 1))); do
  echo "python3 $KIT/fit_codes.py fit --ckpt $LOCAL --importance $C/imp-glm.npz --out $C/resid1 --residual-of $C/codes2 --code-bits 8 --top 96 --layers $(share "$g") > $OUT/fit1-$g.log 2>&1"
done | fan
[[ $(ls "$C"/resid1/layer-*.npz | wc -l) == 42 ]]

run() { echo "python3 $KIT/residency.py --ckpt $LOCAL --cal $W/sessions/cal.json --eval $W/sessions/eval.json --max-tokens 1536 --cal-tokens 768 --codes $C/codes2 --residual $C/resid1 $*"; }
log "smoke: five layers, 128 tokens"
eval "$(run --layers 5 --max-tokens 128 --cal-tokens 64 --variant all=288:0 --variant c108=108:0.3:codes \
  --variant c76h64=76:0.3:codes:hot=64 --variant c108f16=108:0.3:codes:hot=16:fp8 --out "$OUT/smoke")" > "$OUT/smoke.log" 2>&1
tail -5 "$OUT/smoke.log" | tee -a "$OUT/pod.log"

log "simulator, variants grouped by memory over the GPUs"
{
  run --out "$OUT/g72" --variant all=288:0 --variant k72=72:0.3 --variant c72=72:0.3:codes --variant c56h32=56:0.3:codes:hot=32 \
    "> $OUT/g72.log 2>&1"
  run --out "$OUT/g108" --variant k108=108:0.3 --variant c108=108:0.3:codes --variant c92h32=92:0.3:codes:hot=32 \
    --variant c76h64=76:0.3:codes:hot=64 --variant c108f16=108:0.3:codes:hot=16:fp8 "> $OUT/g108.log 2>&1"
  run --out "$OUT/g150" --variant k150=150:0.3 --variant c150=150:0.3:codes --variant c134h32=134:0.3:codes:hot=32 \
    --variant c118h64=118:0.3:codes:hot=64 "> $OUT/g150.log 2>&1"
  run --out "$OUT/g188" --variant c188=188:0.3:codes --variant c172h32=172:0.3:codes:hot=32 --variant c156h64=156:0.3:codes:hot=64 \
    --variant c288=288:1e9:codes --variant c288h32=288:1e9:codes:hot=32 "> $OUT/g188.log 2>&1"
} | fan
for g in g72 g108 g150 g188; do tail -6 "$OUT/$g.log" | tee -a "$OUT/pod.log"; done
cp "$C"/codes2/metrics.jsonl "$OUT/fit2-metrics.jsonl"
cp "$C"/resid1/metrics.jsonl "$OUT/fit1-metrics.jsonl"
log "all done"
touch "$OUT/DONE"
