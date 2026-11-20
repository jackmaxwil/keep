#!/usr/bin/env bash
# E7, the held-out confirmation, on one GPU pod: the policy frozen on the Mac's 22 sessions (tau 0.3,
# re-route on a miss, residents warmed from calibration mass) scored on the 42 eval sessions with stored
# teacher logits that the Mac never scored, at full length (heldout-sessions.json, three groups).
#
# 1. Fit 2-bit E8P codes for all 384 experts of every layer (keep_quant_torch.py; the Mac's codes would
#    take 12 h to upload at 3 MB/s, and this fit reproduced them to 99.99-100% on sampled rows).
# 2. Score seven variants per group with local_eval.py and the torch kernel stand-ins, as on the Mac:
#    q384 all resident; sK the warm top-K never changed (pruning with our codes); kK the cache of K at tau 0.3.
#
# On the volume before the GPU starts: /workspace/v41-hf, /workspace/pack, /workspace/keep (this repo),
# /workspace/pool384/{imp-all.npz,reap-stats.npz}, /workspace/x1/eval/eval/sessions/NAME.npz (teacher).
# Output: /workspace/heldout/{eval-g0,eval-g1,eval-g2}/summary.json, pod.log, DONE or FAILED.
set -euo pipefail
W=/workspace
P=$W/pool384
H=$W/heldout
KIT=$W/keep/scripts/runpod/v41_reap
LOCAL=${LOCAL:-/root/v41}
CODES=$LOCAL/codes/e8p-384
export PYTHONPATH=$W/keep/src PIP_BREAK_SYSTEM_PACKAGES=1
mkdir -p "$H"
log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$H/pod.log"; }
trap 'status=$?; [[ $status -ne 0 ]] && { log "FAILED (status $status)"; touch "$H/FAILED"; }' EXIT

# The wheels on the volume (torch 2.15, CUDA 13): PyPI ran at 260 KB/s from this data center.
(pip install -q --no-index --find-links "$W/wheels" -r "$KIT/requirements-ref.txt" > "$H/pip.log" 2>&1; echo "pip exit $?" >> "$H/pip.log") &
pip_pid=$!
size() { (cd "$1" && find . -type f -not -path './.cache/*' -printf '%s\n' | awk '{s+=$1} END {print s}'); }
if [[ $(size "$W/v41-hf") != $(size "$LOCAL/hf" 2>/dev/null) ]]; then
  log "copy checkpoint to $LOCAL/hf"
  mkdir -p "$LOCAL/hf"
  (cd "$W/v41-hf" && find . -type f -not -path './.cache/*' -print0 | xargs -0 -P 16 -I{} cp --parents {} "$LOCAL/hf/")
  [[ $(size "$W/v41-hf") == $(size "$LOCAL/hf") ]]
fi
log "checkpoint in place: $(du -sh "$LOCAL/hf" | cut -f1)"

wait $pip_pid
tail -2 "$H/pip.log" | tee -a "$H/pod.log"
python3 -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_arch_list())" | tee -a "$H/pod.log"

fit() { python3 "$KIT/keep_quant_torch.py" fit --hf-path "$LOCAL/hf" --importance "$P/imp-all.npz" "$@"; }
log "fit smoke: four experts of layer 0"
rm -rf "$LOCAL/fit-smoke"
fit --out "$LOCAL/fit-smoke" --layers 0 --experts 4 | tee -a "$H/pod.log"
hours=$(python3 -c "import json,sys; print(round(json.loads(open(sys.argv[1]).readline())['seconds'] / 4 * 384 * 40 / 3600, 2))" "$LOCAL/fit-smoke/metrics.jsonl")
log "fit of all 15,360 experts projected at $hours h"
python3 -c "import sys; sys.exit(float(sys.argv[1]) > float(sys.argv[2]))" "$hours" "${MAX_FIT_H:-5}" || { log "over MAX_FIT_H, stopping"; exit 1; }
fit --out "$CODES" > "$H/fit.log" 2>&1
[[ $(find "$CODES" -name 'layer-??.npz' | wc -l) == 40 ]]
log "fit done: $(tail -1 "$H/fit.log")"

run() {
  python3 "$KIT/local_eval.py" --device cuda --torch-kernels --hf-path "$LOCAL/hf" --pack "$W/pack/v41-pack.jsonl" \
    --stats "$W/x1/eval" --max-rss-gb 1000 --cache-warm "$P/reap-stats.npz" --dynamic "|$CODES" "$@"
}
v() { echo --variant "$1=all" --tau "$1=$3" --cache-k "$1=$2" --miss "$1=reroute"; }
group() { python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))['groups'][int(sys.argv[2])]))" "$KIT/heldout-sessions.json" "$1"; }

log "smoke: one session, 128 tokens"
group 0 | python3 -c "import json,sys; print(json.dumps(json.load(sys.stdin)[:1]))" > "$H/smoke-sessions.json"
# shellcheck disable=SC2046
run --sessions-file "$H/smoke-sessions.json" --max-tokens 128 $(v q384 384 0) $(v k144 144 0.3) --out "$H/smoke" 2>&1 | tail -3 | tee -a "$H/pod.log"

for g in 0 1 2; do
  log "group $g"
  group "$g" > "$H/sessions-g$g.json"
  # shellcheck disable=SC2046
  run --sessions-file "$H/sessions-g$g.json" --max-tokens 16384 \
    $(v q384 384 0) $(v s96 96 1e9) $(v k96 96 0.3) $(v s144 144 1e9) $(v k144 144 0.3) $(v s200 200 1e9) $(v s250 250 1e9) \
    --out "$H/eval-g$g" > "$H/eval-g$g.log" 2>&1
  log "group $g done: $(grep -o '"forward_seconds": [0-9.]*' "$H/eval-g$g/summary.json")"
done
log "all done"
touch "$H/DONE"
