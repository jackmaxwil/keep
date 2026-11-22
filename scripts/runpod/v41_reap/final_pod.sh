#!/usr/bin/env bash
# The paper's last two V4.1 experiments on one GPU pod (2 to 8 GPUs of 80 GB or more).
#
# 2. Expert choice against precision: the 192 experts of largest routed mass and the pruned
#    checkpoint's 192, both at the released 4 bits, scored by collect.py against the full model
#    on the 22 tuning sessions (first 1,024 tokens), one shard per GPU.
# 1. Full-model generation: vLLM serves the full model (recipes.vllm.ai, DeepSeek-V4.1-Flash);
#    the three script requests of the generation test, 24 + 16 + 16 samples, thinking off.
#
# On the volume before the GPU starts (a CPU pod put them there):
#   /workspace/v41-hf      the pinned checkpoint      /workspace/wheels   reference wheels
#   /workspace/keep        this repo                  /workspace/vllm-venv  vLLM >= 0.31 (uv venv)
#   /workspace/final/in    final_inputs.py output
# Output: /workspace/final/out/{exp2,gen,parity.json,pod.log}, then DONE or FAILED.
# SKIP_GEN=1 runs experiment 2 alone (one GPU is enough); SKIP_EXP2=1 experiment 1 alone. OFFLOAD_GB keeps
# that many GB of weights in host memory (vLLM --cpu-offload-gb), for a GPU too small for the model.
set -euo pipefail
W=/workspace
F=$W/final
O=${OUT:-$F/out}  # one OUT per pod when two pods share the volume
KIT=$W/keep/scripts/runpod/v41_reap
LOCAL=${LOCAL:-/root/v41}
NGPU=$(nvidia-smi -L | wc -l)
export PYTHONPATH=$W/keep/src PIP_BREAK_SYSTEM_PACKAGES=1 MAX_GPU_HOURS=${MAX_GPU_HOURS:-4}
mkdir -p "$O" "$LOCAL"
log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$O/pod.log"; }
trap 'status=$?; [[ $status -ne 0 ]] && { log "FAILED (status $status)"; touch "$O/FAILED"; }' EXIT
# Watchdog, as pod.sh's: the pod stops itself after MAX_GPU_HOURS whatever happens (if its runpodctl can).
[[ -z "${RUNPOD_POD_ID:-}" && -r /etc/rp_environment ]] && source /etc/rp_environment
( sleep $((MAX_GPU_HOURS * 3600)); log "watchdog: ${MAX_GPU_HOURS} h"; runpodctl pod stop "${RUNPOD_POD_ID:-}" || runpodctl stop pod "${RUNPOD_POD_ID:-}" ) &

# Reference wheels and the vLLM venv go on in the background while the checkpoint copies.
(pip install -q --no-index --find-links "$W/wheels" -r "$KIT/requirements-ref.txt" > "$O/pip.log" 2>&1; echo "pip exit $?" >> "$O/pip.log") &
pip_pid=$!
(cp -a "$W/vllm-venv" "$LOCAL/vllm-venv"; echo "venv copied" >> "$O/pip.log") &
venv_pid=$!
size() { (cd "$1" && find . -type f -not -path './.cache/*' -printf '%s\n' | awk '{s+=$1} END {print s}'); }
if [[ $(size "$W/v41-hf") != $(size "$LOCAL/hf" 2>/dev/null) ]]; then
  log "copy checkpoint to $LOCAL/hf"
  mkdir -p "$LOCAL/hf"
  (cd "$W/v41-hf" && find . -type f -not -path './.cache/*' -print0 | xargs -0 -P 16 -I{} cp --parents {} "$LOCAL/hf/")
  [[ $(size "$W/v41-hf") == $(size "$LOCAL/hf") ]]
fi
log "checkpoint in place: $(du -sh "$LOCAL/hf" | cut -f1); $NGPU GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
wait $pip_pid
tail -1 "$O/pip.log" | tee -a "$O/pod.log"

# Experiment 2: four-bit students against the teacher, one shard per GPU.
# A failure here is logged and does not stop experiment 1 (set -e does not apply inside a tested
# function, hence the explicit returns).
exp2() {
  local i p pids=()
  for ((i = 0; i < NGPU; i++)); do
    CUDA_VISIBLE_DEVICES=$i python3 "$KIT/collect.py" --hf-path "$LOCAL/hf" --pack "$F/in/sessions-22.jsonl" \
      --out "$LOCAL/exp2" --sweep eval --shard "$i/$NGPU" --max-seq-len 1024 --chunk-tokens 262144 \
      --survivors fm192="$F/in/fm-192.json" --survivors reap192="$F/in/reap-192.json" > "$O/exp2-$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || return 1; done
  python3 "$KIT/score.py" --teacher "$LOCAL/exp2" --student fm192="$LOCAL/exp2/student-fm192" \
    --student reap192="$LOCAL/exp2/student-reap192" > "$LOCAL/exp2/scores.json" || return 1
  mkdir -p "$O/exp2" && cp -r "$LOCAL/exp2/." "$O/exp2/"
}
if [[ -z "${SKIP_EXP2:-}" && ! -f "$O/exp2/scores.json" ]]; then
  log "experiment 2: fm192 and reap192 at 4 bits, 22 sessions"
  if exp2; then log "experiment 2 done"; else log "experiment 2 FAILED (exp2-*.log); going on to experiment 1"; fi
fi

if [[ -n "${SKIP_GEN:-}" ]]; then log "SKIP_GEN set: experiment 1 not run"; touch "$O/DONE"; exit 0; fi

# Experiment 1: vLLM, the recipe's flags; Engram tables in host memory, vision encoder off.
wait $venv_pid
log "vLLM $("$LOCAL/vllm-venv/bin/python" -c 'import vllm; print(vllm.__version__)'), TP $NGPU"
# Blackwell takes the recipe's own attention backend and FP8 KV cache; Hopper the recipe's defaults.
extra=()
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | grep -q B200 && extra=(
  --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}'
  --kv-cache-dtype fp8)
VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_USE_RUST_FRONTEND=1 "$LOCAL/vllm-venv/bin/python" -m vllm.entrypoints.cli.main serve "$LOCAL/hf" \
  --served-model-name full --tokenizer-mode deepseek_v41 --tensor-parallel-size "$NGPU" "${extra[@]}" \
  --engram-config '{"cpu_offload":true}' --reasoning-parser deepseek_v41 --language-model-only \
  --max-model-len 16384 --max-num-seqs 64 ${OFFLOAD_GB:+--cpu-offload-gb "$OFFLOAD_GB"} --port 8000 > "$O/vllm.log" 2>&1 &
vllm_pid=$!
until curl -s -m 5 http://127.0.0.1:8000/v1/models | grep -q '"full"'; do
  kill -0 $vllm_pid 2>/dev/null || { log "vLLM exited: $(tail -3 "$O/vllm.log")"; exit 1; }
  sleep 20
done
log "vLLM ready"

# The prompt vLLM renders against the checkpoint's own encoding; recorded, not a stop: vLLM's
# rendering is the reference, and a difference would bear on the Mac rows, not on these.
"$LOCAL/vllm-venv/bin/python" - "$F/in" "$O/parity.json" <<'PY'
import json, sys, urllib.request
src, out = sys.argv[1], sys.argv[2]
want = json.load(open(f"{src}/expected-ids.json"))
res = {}
for task, ids in want.items():
    req = json.load(open(f"{src}/req-{task}.json"))
    body = {"model": "full", "messages": req["messages"], "add_generation_prompt": True,
            "chat_template_kwargs": req["chat_template_kwargs"]}
    r = urllib.request.Request("http://127.0.0.1:8000/tokenize", json.dumps(body).encode(), {"Content-Type": "application/json"})
    got = json.load(urllib.request.urlopen(r))["tokens"]
    res[task] = {"equal": got == ids, "vllm": got, "expected": ids}
json.dump(res, open(out, "w"))
print({t: v["equal"] for t, v in res.items()})
PY
log "parity: $(python3 -c "import json; print({t: v['equal'] for t, v in json.load(open('$O/parity.json')).items()})")"

log "experiment 1: 56 samples"
"$LOCAL/vllm-venv/bin/python" "$KIT/gen_client.py" http://127.0.0.1:8000 "$F/in" "$O/gen" dupes=24 csvsum=16 largest=16 >> "$O/pod.log" 2>&1
kill $vllm_pid || true
log "all done"
touch "$O/DONE"
