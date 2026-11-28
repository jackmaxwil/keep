#!/usr/bin/env bash
# The last V4.1 pod: codes fitted from the source checkpoint, and the full model's LiveCodeBench row.
# One pod with 8 GPUs of 80 GB (8xH100: vLLM needs all eight for the full model) and 900 GB of disk; no volume.
#
# Stages, in order. Each leaves a marker in $O and is skipped on a rerun; a stage that fails is logged and the
# next one still runs where it can.
#   download  the pinned source checkpoint from Hugging Face to $LOCAL/hf; the reference's wheels from PyPI
#   teacher   collect.py: the full model's top-K logits on the 22 tuning sessions (first 1,024 tokens)
#   bench     vLLM serves the full model (recipes.vllm.ai, DeepSeek-V4.1-Flash; one GPU per tensor-parallel rank);
#             lcb.py gen with the Mac rows' problems and settings (card sampling, thinking on, 16,384 new tokens)
#   fit       fit_source.py, a process per GPU: for all 384 experts of 40 layers, the plain fit under importance
#             measured on the full model, and the error-feedback fit through the full second moments of the same
#             rows, damping chosen per layer on held-out rows
#   score     local_eval.py on each code set against the teacher: all 384 resident, the 144 of largest routed mass
#             held static, and a cache of 144 at tau 0.3 that re-routes a miss
#   upload    with HF_TOKEN set: both code sets, the importance files and the fit metrics to the dataset repo
#
# Inputs in $IN, sent from the Mac (runpodctl send / receive) as one tarball unpacked in /workspace:
#   keep/                    this repo (git archive HEAD)
#   in/v41-pack.jsonl        calibration records (only split "calibration" is read by the fit)
#   in/sessions-22.jsonl     final_inputs.py: the tuning sessions cut to 1,024 tokens
#   in/mac-eval-sessions.json, in/reap-stats.npz   their names; routed mass per expert, for the warm sets
# The benchmark's release file is fetched here and checked against the Mac's copy by SHA-256.
# Output: $O (logs, lcb-full/, score-plain/, score-feedback/, fit-metrics.jsonl), then DONE or FAILED.
# Answers are graded on the Mac (lcb.py score runs them under sandbox-exec).
set -uo pipefail
W=${W:-/workspace}
IN=$W/in
O=${OUT:-$W/out}
KEEP=$W/keep
KIT=$KEEP/scripts/runpod/v41_reap
LOCAL=${LOCAL:-/root/v41}
REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017
REV=1bae0e1088993b9b82de08f59de5be5a34e42170
NGPU=$(nvidia-smi -L | wc -l)
export PYTHONPATH=$KEEP/src PIP_BREAK_SYSTEM_PACKAGES=1 HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$O" "$LOCAL"
log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$O/pod.log"; }
failed=0
stage() {  # stage NAME FUNCTION: run once, keep going on failure
  [[ -f "$O/$1.done" ]] && return 0
  log "stage $1"
  if "$2"; then touch "$O/$1.done"; log "stage $1 done"; else failed=1; log "stage $1 FAILED"; return 1; fi
}
each_gpu() {  # each_gpu LOGNAME COMMAND...: one process per GPU with --shard I/N appended; fails if any does
  local name=$1 i p pids=() bad=0
  shift
  for ((i = 0; i < NGPU; i++)); do
    CUDA_VISIBLE_DEVICES=$i "$@" --shard "$i/$NGPU" > "$O/$name-$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || bad=1; done
  return $bad
}
check() { [[ $(sha256sum "$1" | cut -d' ' -f1) == "$2" ]] || { log "$1 is not the file the Mac used"; return 1; }; }

download() {
  check "$IN/v41-pack.jsonl" 6a354705bc1c5f487a250b7299dbc8d176b2db648f80c42a509d7b9bc0cfb987 || return 1
  check "$IN/sessions-22.jsonl" fa35cda334cebb68c2935a4e5e798e117dabae5afb99fa1b6ce8dac7dd2fefe8 || return 1
  check "$IN/reap-stats.npz" 783a319ed76787781bad91307d2abeff54b8dd6b700d675370854f3e7e85bee2 || return 1
  pip install -q -r "$KIT/requirements-ref.txt" huggingface_hub hf_transfer uv > "$O/pip.log" 2>&1 || return 1
  python3 - "$REPO" "$REV" "$LOCAL/hf" "$IN" > "$O/download.log" 2>&1 <<'PY' || return 1
import sys
from huggingface_hub import hf_hub_download, snapshot_download
repo, rev, dest, inputs = sys.argv[1:]
snapshot_download(repo, revision=rev, local_dir=dest, max_workers=16)
hf_hub_download("livecodebench/code_generation_lite", "test6.jsonl", repo_type="dataset", local_dir=inputs)
PY
  check "$IN/test6.jsonl" bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5 || return 1
  python3 -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))" | tee -a "$O/pod.log"
  log "checkpoint in place: $(du -sh "$LOCAL/hf" | cut -f1); $NGPU GPUs"
}

teacher() {
  each_gpu teacher python3 "$KIT/collect.py" --hf-path "$LOCAL/hf" --pack "$IN/sessions-22.jsonl" --out "$LOCAL/teacher" \
    --sweep eval --max-seq-len 1024 --chunk-tokens 262144 || return 1
  [[ $(find "$LOCAL/teacher/eval/sessions" -name '*.npz' | wc -l) == 22 ]]
}

venv() {  # after the download: a uv install stalls while the checkpoint takes the bandwidth
  [[ -x "$LOCAL/vllm-venv/bin/vllm" ]] && return 0
  uv venv -q --python 3.12 "$LOCAL/vllm-venv" && uv pip install -q --python "$LOCAL/vllm-venv/bin/python" "vllm==${VLLM:-0.34.0}"
}

bench() {
  wait "$venv_pid" || { log "vLLM venv failed: $(tail -2 "$O/venv.log")"; return 1; }
  # FlashInfer and DeepGEMM build kernels at start: they need the venv's ninja and a whole CUDA toolkit of 12.9 or later.
  [[ -d /usr/local/cuda-13.0 ]] || apt-get install -y -q --no-install-recommends cuda-toolkit-13-0 > "$O/apt.log" 2>&1 || return 1
  local vllm_pid rc=0
  CUDA_HOME=/usr/local/cuda-13.0 PATH="$LOCAL/vllm-venv/bin:/usr/local/cuda-13.0/bin:$PATH" VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    VLLM_USE_RUST_FRONTEND=1 "$LOCAL/vllm-venv/bin/python" -m vllm.entrypoints.cli.main serve "$LOCAL/hf" \
    --served-model-name full --tokenizer-mode deepseek_v41 --tensor-parallel-size "$NGPU" \
    --engram-config '{"cpu_offload":true}' --reasoning-parser deepseek_v41 --language-model-only \
    --max-model-len "${MAX_LEN:-24576}" --max-num-seqs "${MAX_SEQS:-64}" --gpu-memory-utilization "${GPU_UTIL:-0.9}" \
    --port 8000 > "$O/vllm.log" 2>&1 &
  vllm_pid=$!
  until curl -s -m 5 http://127.0.0.1:8000/v1/models | grep -q '"full"'; do
    kill -0 $vllm_pid 2>/dev/null || { log "vLLM exited: $(tail -3 "$O/vllm.log")"; return 1; }
    sleep 20
  done
  log "vLLM $("$LOCAL/vllm-venv/bin/python" -c 'import vllm; print(vllm.__version__)') ready, TP $NGPU"
  python3 "$KEEP/scripts/v41_build/lcb.py" gen --data "$IN/test6.jsonl" --url http://127.0.0.1:8000/v1 --model full \
    --out "$O/lcb-full" --workers "${WORKERS:-24}" --extra '{"chat_template_kwargs": {"thinking": true}}' > "$O/lcb-full.log" 2>&1 || rc=1
  kill $vllm_pid; wait $vllm_pid
  log "benchmark: $(find "$O/lcb-full" -name '*.json' | wc -l) of 100 answers, $(grep -c ' length$' "$O/lcb-full.log") cut at the token limit"
  return $rc
}

fit() {
  local hours
  rm -rf "$LOCAL/fit-smoke"
  CUDA_VISIBLE_DEVICES=0 python3 "$KIT/fit_source.py" --hf-path "$LOCAL/hf" --pack "$IN/v41-pack.jsonl" --out "$LOCAL/fit-smoke" \
    --layers 0 --experts 8 --holdout 2 --cal-tokens 40000 > "$O/fit-smoke.log" 2>&1 || { log "fit smoke: $(tail -2 "$O/fit-smoke.log")"; return 1; }
  tail -1 "$O/fit-smoke.log" | cut -c1-600 | tee -a "$O/pod.log"
  # Eight experts and two held-out ones at three dampings took `seconds`; a layer has 384 and 16, so this runs high.
  hours=$(python3 -c "import json,sys; print(round(json.loads(open(sys.argv[1]).readline())['seconds'] / 8 * 384 * 40 / int(sys.argv[2]) / 3600, 2))" \
    "$LOCAL/fit-smoke/metrics-0.jsonl" "$NGPU")
  log "fit projected at $hours h a GPU, forward pass apart"
  python3 -c "import sys; sys.exit(float(sys.argv[1]) > float(sys.argv[2]))" "$hours" "${MAX_FIT_H:-6}" || { log "over MAX_FIT_H"; return 1; }
  each_gpu fit python3 "$KIT/fit_source.py" --hf-path "$LOCAL/hf" --pack "$IN/v41-pack.jsonl" --out "$LOCAL/codes" \
    --cal-tokens "${CAL_TOKENS:-300000}" || { log "fit: $(tail -qn 1 "$O"/fit-*.log | cut -c1-200)"; return 1; }
  [[ $(find "$LOCAL/codes/plain" "$LOCAL/codes/feedback" -name 'layer-??.npz' | wc -l) == 80 ]] || return 1
  cat "$LOCAL"/codes/metrics-*.jsonl > "$O/fit-metrics.jsonl"
  python3 - "$O/fit-metrics.jsonl" <<'PY' | tee -a "$O/pod.log"
import json, sys
rows = sorted((json.loads(l) for l in open(sys.argv[1])), key=lambda r: r["layer"])
for space in ("hidden", "down"):
    kept = [r for r in rows if r["damp"][space] is not None]
    gain = [1 - r["heldout"][space][str(r["damp"][space])] / r["heldout"][space]["plain"] for r in kept]
    print(f"{space}: feedback kept in {len(kept)} of {len(rows)} layers; held-out output error down "
          f"{100 * sum(gain) / max(len(gain), 1):.0f}% on average where kept (least {100 * min(gain, default=0):.0f}%)")
PY
}

v() { echo --variant "$1=all" --tau "$1=$3" --cache-k "$1=$2" --miss "$1=reroute"; }
score_one() {  # score_one GPU CODES NAME [extra local_eval arguments]
  local gpu=$1 codes=$2 name=$3
  shift 3
  # shellcheck disable=SC2046
  CUDA_VISIBLE_DEVICES=$gpu python3 "$KIT/local_eval.py" --device cuda --torch-kernels --hf-path "$LOCAL/hf" \
    --pack "$IN/sessions-22.jsonl" --stats "$LOCAL/teacher" --sessions-file "$IN/mac-eval-sessions.json" --max-tokens 1024 \
    --max-rss-gb 1000 --cache-warm "$IN/reap-stats.npz" --dynamic "|$codes" \
    $(v q384 384 0) $(v s144 144 1e9) $(v k144 144 0.3) --out "$name" "$@"
}
score() {
  python3 -c "import json,sys; json.dump(json.load(open(sys.argv[1]))[:1], open(sys.argv[2], 'w'))" "$IN/mac-eval-sessions.json" "$O/smoke-sessions.json"
  score_one 0 "$LOCAL/codes/plain" "$LOCAL/score-smoke" --sessions-file "$O/smoke-sessions.json" --max-tokens 128 > "$O/score-smoke.log" 2>&1 \
    || { log "score smoke: $(tail -2 "$O/score-smoke.log")"; return 1; }
  score_one 0 "$LOCAL/codes/plain" "$O/score-plain" > "$O/score-plain.log" 2>&1 &
  local pid=$!
  score_one 1 "$LOCAL/codes/feedback" "$O/score-feedback" > "$O/score-feedback.log" 2>&1 || { wait $pid; return 1; }
  wait $pid
}

upload() {
  [[ -n "${HF_TOKEN:-}" ]] || { log "no HF_TOKEN: the codes ($(du -sh "$LOCAL/codes" | cut -f1)) stay on this pod's disk and go with it"; return 1; }
  local kind
  mkdir -p "$LOCAL/up/quant"
  for kind in plain feedback imp; do  # hard links: the repo's layout without a second copy
    [[ -d "$LOCAL/up/quant/e8p-source-$kind" ]] || cp -al "$LOCAL/codes/$kind" "$LOCAL/up/quant/e8p-source-$kind" || return 1
  done
  cp "$O/fit-metrics.jsonl" "$LOCAL/up/quant/e8p-source-feedback/metrics.jsonl"
  python3 - "$LOCAL/up" "${HF_DEST:-jackmaxwil/keep-v41-artifacts}" > "$O/upload.log" 2>&1 <<'PY'
import sys
from huggingface_hub import upload_large_folder
upload_large_folder(sys.argv[2], sys.argv[1], repo_type="dataset", allow_patterns=["quant/*/layer-*.npz", "quant/*/metrics.jsonl"])
PY
}

trap 'kill $(jobs -p) 2>/dev/null' EXIT
stage download download || { touch "$O/FAILED"; exit 1; }  # nothing runs without the checkpoint
(venv > "$O/venv.log" 2>&1) &
venv_pid=$!
stage teacher teacher
stage bench bench
stage fit fit && stage score score
[[ -f "$O/fit.done" ]] && stage upload upload
[[ $failed == 0 ]] && { log "all done"; touch "$O/DONE"; } || { log "finished with failures"; touch "$O/FAILED"; }
