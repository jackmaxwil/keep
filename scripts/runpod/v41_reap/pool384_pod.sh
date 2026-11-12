#!/usr/bin/env bash
# Pool of 384 on one GPU pod: 2-bit E8P codes for the 192 experts per layer that REAP-50 removed,
# and every router row of the full checkpoint.
#
# The volume already holds what this needs, so the GPU starts work after one copy:
#   /workspace/v41-hf                 the pinned full checkpoint
#   /workspace/keep                   this repo (keep_quant_torch.py uploaded into the kit)
#   /workspace/pool384/imp-pruned.npz importance of the removed experts (built on the Mac)
# Output, on the volume: /workspace/pool384/e8p-podimp-pruned/layer-XX.npz, router-384.npz, DONE.
set -euo pipefail
W=/workspace
P=$W/pool384
KIT=$W/keep/scripts/runpod/v41_reap
LOCAL=${LOCAL:-/root/v41-experts}
export PYTHONPATH=$W/keep/src PIP_BREAK_SYSTEM_PACKAGES=1
log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$P/pod.log"; }
trap 'status=$?; [[ $status -ne 0 ]] && { log "FAILED (status $status)"; touch "$P/FAILED"; }' EXIT

# Never read the checkpoint from the volume (one stream is 120 MB/s). Copy the shards that hold
# routed experts or router rows to the container's disk with 16 streams.
log "copy expert and router shards to $LOCAL"
mkdir -p "$LOCAL"
python3 - "$W/v41-hf" > /tmp/shards.txt <<'PY'
import json, sys
names = json.load(open(sys.argv[1] + "/model.safetensors.index.json"))["weight_map"]
print("\n".join(sorted({shard for name, shard in names.items() if ".ffn.experts." in name or ".ffn.gate." in name})))
PY
(cd "$W/v41-hf" && xargs -P 16 -I{} cp {} "$LOCAL/" < /tmp/shards.txt && cp ./*.json "$LOCAL/")
want=$(cd "$W/v41-hf" && xargs stat -c %s < /tmp/shards.txt | awk '{s+=$1} END {print s}')
have=$(cd "$LOCAL" && xargs stat -c %s < /tmp/shards.txt | awk '{s+=$1} END {print s}')
[[ $want == "$have" ]]
log "copy done: $(wc -l < /tmp/shards.txt) shards, $((have >> 30)) GiB"

pip install -q --no-index --find-links "$W/wheels" -r "$KIT/requirements-ref.txt" \
  || pip install -q -r "$KIT/requirements-ref.txt"
nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a "$P/pod.log"

log "router rows"
python3 "$KIT/keep_quant_torch.py" router --hf-path "$LOCAL" --out "$P/router-384.npz" | tee -a "$P/pod.log"

log "smoke: two experts of layer 0"
rm -rf "$P/smoke"
python3 "$KIT/keep_quant_torch.py" fit --hf-path "$LOCAL" --importance "$P/imp-pruned.npz" --out "$P/smoke" \
  --layers 0 --experts 2 | tee -a "$P/pod.log"

gpus=$(nvidia-smi -L | wc -l)
log "fit on $gpus GPU(s)"
for g in $(seq 0 $((gpus - 1))); do
  CUDA_VISIBLE_DEVICES=$g python3 "$KIT/keep_quant_torch.py" fit --hf-path "$LOCAL" --importance "$P/imp-pruned.npz" \
    --out "$P/e8p-podimp-pruned" --layers "$(seq -s, "$g" "$gpus" 39)" > "$P/fit-$g.log" 2>&1 &
done
wait
[[ $(ls "$P/e8p-podimp-pruned"/layer-??.npz | wc -l) == 40 ]]
log "all done"
touch "$P/DONE"
