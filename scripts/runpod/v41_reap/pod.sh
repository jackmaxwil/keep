#!/usr/bin/env bash
# DeepSeek-V4.1-Flash REAP run on RunPod. Runbook:
# docs/deepseek-v4-flash/2026-11-05-v41-runpod-reap-runbook.md
#
#   pod.sh fetch          CPU pod: download the pinned checkpoint, verify every LFS sha256
#   pod.sh pack           CPU pod: build the calibration + eval pack from public datasets
#   pod.sh gpu            GPU pod (1-8 GPUs, 80 GB+ each): calibration sweep, reap-select,
#                         eval sweep (teacher + every candidate side by side), scores
#   pod.sh prune KEEP     CPU pod: write the pruned HF checkpoint for the Mac to pull
#   pod.sh smoke          GPU pod: the longest calibration record through all 40 layers; memory check
#   pod.sh all            one GPU pod does everything: fetch || pack, smoke, gpu, prune $PRUNE
#
# Everything lives under /workspace (a network volume, or the pod's own volume
# disk in single-pod mode).
set -euo pipefail

MODEL_ID=deepseek-ai/DeepSeek-V4.1-Flash-1017
REVISION=1bae0e1088993b9b82de08f59de5be5a34e42170
W=${WORKSPACE:-/workspace}
KIT=$W/keep/scripts/runpod/v41_reap
HF=${HF:-$W/v41-hf}
OUT=$W/out
CANDIDATES=${CANDIDATES:-"256 192 128 96"}
PRUNE=${PRUNE:-"192"}                   # REAP 50%. REAP 75% (96) after the Mac pulls 192: 1.2 TB holds one
MAX_GPU_HOURS=${MAX_GPU_HOURS:-8}       # hard stop for the GPU pod (1xH100 SXM secure $3.99/h)
# Tokens (x streams) resident per chunk. The hidden state carried between layers costs ~63 KB/token
# on the GPU: 524288 left ~65 GB in use and the 16K-record indexer's 16 GiB alloc OOMed (2026-11-07).
CHUNK_TOKENS=${CHUNK_TOKENS:-262144}
export PYTHONPATH=$W/keep/src
# ssh sessions miss the pod's env (RUNPOD_POD_ID, the pod-scoped RUNPOD_API_KEY): load it, never print it
if [[ -z "${RUNPOD_POD_ID:-}" && -r /etc/rp_environment ]]; then source /etc/rp_environment; fi

watchdog() {  # whatever happens next, the pod stops at MAX_GPU_HOURS; a failed stage stops it at once
  [[ -n "${WATCHDOG_STARTED:-}" ]] && return
  export WATCHDOG_STARTED=1
  ( sleep $((MAX_GPU_HOURS * 3600)); stop_pod "watchdog: ${MAX_GPU_HOURS}h limit" ) &
  trap 'status=$?; [[ $status -ne 0 ]] && stop_pod "stage failed (status $status)"' EXIT
}

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$W/pod.log"; }

stop_pod() {
  log "stopping pod: $*"
  if [[ -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null; then
    runpodctl pod stop "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" || log "runpodctl stop FAILED: stop the pod by hand"
  else
    log "no runpodctl/RUNPOD_POD_ID: stop the pod by hand"
  fi
}

fetch_small() {  # tokenizer, configs, encoding/ and inference/: all pack and collect need besides weights
  pip install -q "huggingface_hub>=0.34" "datasets>=3" tokenizers
  hf download "$MODEL_ID" --revision "$REVISION" --local-dir "$HF" \
    --include "*.json" --include "encoding/*" --include "inference/*"
  test -f "$HF/model.safetensors.index.json" || { log "fetch_small: no index json in $HF"; return 1; }
}

fetch() {
  fetch_small
  hf download "$MODEL_ID" --revision "$REVISION" --local-dir "$HF"
  python3 - "$HF" "$MODEL_ID" "$REVISION" <<'EOF'
import hashlib, sys
from pathlib import Path
from huggingface_hub import HfApi
root, repo, rev = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
bad = 0
for f in HfApi().model_info(repo, revision=rev, files_metadata=True).siblings:
    path = root / f.rfilename
    want = f.lfs.sha256 if f.lfs else None
    if not path.exists() or (f.size is not None and path.stat().st_size != f.size):
        print("MISSING/SIZE", f.rfilename); bad += 1; continue
    if want:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(64 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != want:
            print("SHA256", f.rfilename); bad += 1
sys.exit(f"{bad} files failed verification" if bad else 0)
EOF
  sha256sum "$HF/inference/model.py"
  log "fetch verified"
}

pack() {
  python3 -I "$KIT/build_pack.py" --model-dir "$HF" --out "$W/pack"
  log "pack built: $(python3 -c "import json;m=json.load(open('$W/pack/manifest.json'));print(m['records'],'records',m['sessions'],'sessions')")"
}

sweep() {  # one collect.py per GPU, sessions sharded across them; extra args go to collect.py
  local name=$1; shift
  local n i pids=()
  n=$(nvidia-smi -L | wc -l)
  log "sweep $name on $n GPU(s)"
  for ((i = 0; i < n; i++)); do
    CUDA_VISIBLE_DEVICES=$i python3 "$KIT/collect.py" --hf-path "$HF" --pack "$W/pack/v41-pack.jsonl" \
      --out "$OUT" --sweep "$name" --shard "$i/$n" --deadline-utc "$deadline" --chunk-tokens "$CHUNK_TOKENS" "$@" \
      > "$W/sweep-$name-$i.log" 2>&1 &
    pids+=($!)
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { log "sweep $name shard $i failed: see $W/sweep-$name-$i.log"; return 1; }
  done
}

gpu_setup() {
  [[ -f "$W/pip-freeze-gpu.txt" ]] && return
  pip install -q -r "$HF/inference/requirements.txt"
  pip freeze > "$W/pip-freeze-gpu.txt"
  nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a "$W/pod.log"
}

smoke() {  # the longest calibration record through all 40 layers, before any sweep
  watchdog
  gpu_setup
  python3 - "$W/pack/v41-pack.jsonl" "$W/smoke/pack.jsonl" <<'PY'
import json, pathlib, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
row = max((r for r in rows if r["split"] == "calibration"), key=lambda r: len(r["token_ids"]))
row["session"] = "smoke"
pathlib.Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(sys.argv[2]).write_text(json.dumps(row) + "\n")
print("smoke record", row["id"], len(row["token_ids"]), "tokens")
PY
  rm -rf "$W/smoke/calibration" "$W/smoke"/progress-*
  python3 "$KIT/collect.py" --hf-path "$HF" --pack "$W/smoke/pack.jsonl" --out "$W/smoke" --sweep calibration \
    2>&1 | tee "$W/smoke.log"
  python3 - "$W/smoke" <<'PY'
import glob, json, sys
import numpy as np
(entry,) = [json.loads(line) for f in glob.glob(sys.argv[1] + "/progress-*.jsonl") for line in open(f)]
with np.load(sys.argv[1] + "/calibration/sessions/smoke.npz") as z:
    slots = z["reap__routed_slots"]
    assert (slots == slots[0]).all() and slots[0] > 0, slots
    assert np.isfinite(z["reap__gate_norm_sum"]).all() and np.isfinite(z["importance_sum__hidden"]).all()
print("smoke ok:", json.dumps(entry))
sys.exit(0 if entry["max_gpu_gb"] < 76 else f"smoke: {entry['max_gpu_gb']} GB peak is too close to 80 GB")
PY
  log "smoke passed: $(tail -1 "$W/smoke.log")"
}

gpu() {
  local deadline
  deadline=$(date -u -d "+$((MAX_GPU_HOURS * 60 - 20)) minutes" +%FT%TZ)
  watchdog
  gpu_setup

  sweep calibration

  log "finalize-reap + candidates: $CANDIDATES"
  python3 "$W/keep/benchmarks/produce_dsv4_teacher_cache.py" finalize-reap --out-dir "$OUT/calibration"
  local survivors=()
  for keep in $CANDIDATES; do
    python3 "$W/keep/benchmarks/produce_dsv4_teacher_cache.py" reap-select \
      --stats "$OUT/calibration/reap-stats.npz" --keep "$keep" --output "$OUT/survivors-$keep.json" \
      > "$OUT/reap-select-$keep.json"
    survivors+=(--survivors "$keep=$OUT/survivors-$keep.json")
  done

  sweep eval "${survivors[@]}"

  local students=()
  for keep in $CANDIDATES; do students+=(--student "$keep=$OUT/student-$keep"); done
  python3 "$KIT/score.py" --teacher "$OUT" "${students[@]}" | tee "$OUT/candidate-scores.json"
  log "gpu stage done; scores in $OUT/candidate-scores.json"
}

prune() {
  local keep=${1:?usage: pod.sh prune KEEP}
  pip install -q numpy
  python3 "$KIT/prune.py" --src "$HF" --survivors "$OUT/survivors-$keep.json" --out "$W/v41-reap$keep"
  (cd "$W/v41-reap$keep" && sha256sum -c SHA256SUMS --quiet) && log "pruned checkpoint $W/v41-reap$keep verified"
}

all() {
  watchdog
  fetch_small
  log "fetch (weights) and pack in parallel"
  fetch > "$W/fetch.log" 2>&1 &
  local fetch_pid=$!
  pack > "$W/pack.log" 2>&1
  wait "$fetch_pid" || { log "fetch failed: see $W/fetch.log"; return 1; }
  smoke
  gpu
  for keep in $PRUNE; do prune "$keep"; done
  log "all done: $(echo "$PRUNE" | sed "s|[0-9]*|$W/v41-reap&|g"). Pod left running for the Mac to pull; watchdog still armed."
}

stage=${1:?usage: pod.sh fetch|pack|smoke|gpu|prune KEEP|all}
shift
"$stage" "$@"
