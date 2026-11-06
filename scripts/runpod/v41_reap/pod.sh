#!/usr/bin/env bash
# DeepSeek-V4.1-Flash REAP run on RunPod. Runbook:
# docs/deepseek-v4-flash/2026-11-05-v41-runpod-reap-runbook.md
#
#   pod.sh fetch          CPU pod: download the pinned checkpoint, verify every LFS sha256
#   pod.sh pack           CPU pod: build the calibration + eval pack from public datasets
#   pod.sh gpu            8xH200 pod: convert, teacher pass, candidate scoring; stops itself
#   pod.sh prune KEEP     CPU pod: write the pruned HF checkpoint for the Mac to pull
#
# Everything lives on one network volume mounted at /workspace, so stages can
# run on different (cheap CPU / expensive GPU) pods.
set -euo pipefail

MODEL_ID=deepseek-ai/DeepSeek-V4.1-Flash-1017
REVISION=1bae0e1088993b9b82de08f59de5be5a34e42170
W=${WORKSPACE:-/workspace}
KIT=$W/keep/scripts/runpod/v41_reap
HF=$W/v41-hf
TP8=$W/v41-tp8
OUT=$W/out
CANDIDATES=${CANDIDATES:-"256 224 192 160"}
MAX_GPU_HOURS=${MAX_GPU_HOURS:-6}       # hard stop for the GPU pod (8xH200 secure ~ $42/h)
export PYTHONPATH=$W/keep/src HF_HUB_ENABLE_HF_TRANSFER=1

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$W/pod.log"; }

stop_pod() {
  log "stopping pod: $*"
  if [[ -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null; then
    runpodctl stop pod "$RUNPOD_POD_ID" || log "runpodctl stop FAILED: stop the pod by hand"
  else
    log "no runpodctl/RUNPOD_POD_ID: stop the pod by hand"
  fi
}

fetch() {
  pip install -q "huggingface_hub[hf_transfer]>=0.34"
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
  pip install -q "datasets>=3" tokenizers huggingface_hub
  python3 -I "$KIT/build_pack.py" --model-dir "$HF" --out "$W/pack"
  log "pack built: $(python3 -c "import json;m=json.load(open('$W/pack/manifest.json'));print(m['records'],'records',m['sessions'],'sessions')")"
}

gpu() {
  local deadline
  deadline=$(date -u -d "+$((MAX_GPU_HOURS * 60 - 20)) minutes" +%FT%TZ)
  # Watchdog: whatever happens below, the pod stops at MAX_GPU_HOURS.
  ( sleep $((MAX_GPU_HOURS * 3600)); stop_pod "watchdog: ${MAX_GPU_HOURS}h limit" ) &
  trap 'stop_pod "gpu stage exited (status $?)"' EXIT

  cd "$HF/inference"
  pip install -q -r requirements.txt
  pip freeze > "$W/pip-freeze-gpu.txt"
  nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a "$W/pod.log"

  if [[ ! -f "$TP8/model7-mp8.safetensors" ]]; then
    log "convert -> $TP8"
    python3 convert.py --hf-ckpt-path "$HF" --save-path "$TP8" --model-parallel 8 --expert-dtype fp4
  fi

  log "teacher pass"
  torchrun --nproc-per-node 8 "$KIT/collect.py" --ckpt-path "$TP8" --config config.json \
    --pack "$W/pack/v41-pack.jsonl" --out "$OUT" --deadline-utc "$deadline"

  log "finalize-reap + candidates: $CANDIDATES"
  python3 "$W/keep/benchmarks/produce_dsv4_teacher_cache.py" finalize-reap --out-dir "$OUT/calibration"
  local survivors=()
  for keep in $CANDIDATES; do
    python3 "$W/keep/benchmarks/produce_dsv4_teacher_cache.py" reap-select \
      --stats "$OUT/calibration/reap-stats.npz" --keep "$keep" --output "$OUT/survivors-$keep.json" \
      > "$OUT/reap-select-$keep.json"
    survivors+=(--survivors "$keep=$OUT/survivors-$keep.json")
  done

  log "student passes"
  torchrun --nproc-per-node 8 "$KIT/collect.py" --ckpt-path "$TP8" --config config.json \
    --pack "$W/pack/v41-pack.jsonl" --out "$OUT" --deadline-utc "$deadline" "${survivors[@]}"

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

stage=${1:?usage: pod.sh fetch|pack|gpu|prune KEEP}
shift
"$stage" "$@"
