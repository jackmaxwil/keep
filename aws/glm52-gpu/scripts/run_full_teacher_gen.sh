#!/usr/bin/env bash
# RUN ON THE p5.48xlarge (after gpu_env_setup.sh). Full 257-session teich
# teacher-gen: 8 single-GPU workers over token-balanced partitions, shared
# resumable out-dir, per-GPU heartbeats, results pushed to S3 only after all
# workers finish (never co-run S3 transfers with the forward — disk contention
# starves the model load; measured 3MB/s vs 271MB/s on the seeder).
set -uxo pipefail
exec > /mnt/nvme/fullrun.log 2>&1

BUCKET="${BUCKET:?set BUCKET=keep-glm52-models-<acct>-<region>}"
export ROOT="${ROOT:-/mnt/nvme}"
export NGPU="${NGPU:-8}"
HUB="$ROOT/hub/models--0xSero--glm-5.2-reap-504B-v2"
OUT="$ROOT/teacher-cache-out"
S() { echo "$1 $(date -u +%FT%TZ)" >> "$ROOT/RUN_STATUS"; }

mkdir -p "$OUT"
S PACK-PULL
/usr/local/bin/s5cmd cp "s3://$BUCKET/teich-pack/glm52-coding-agent-initial-v2-20260713.json" \
  "$ROOT/teich-pack.json" || { S PACK-PULL-FAILED; exit 1; }
S PARTITIONING
# Token-balanced snake partition of pending sessions across workers.
python3 - << 'PYEOF' || { echo PARTITION-FAILED >> /mnt/nvme/RUN_STATUS; exit 1; }
import json, os
root = os.environ.get("ROOT", "/mnt/nvme")
ngpu = int(os.environ.get("NGPU", "8"))
pack = json.load(open(f"{root}/teich-pack.json"))
rows = sorted(pack["prompt_rows"], key=lambda r: -len(r["encoded_token_ids"]))
done = {f[:-4] for f in os.listdir(f"{root}/teacher-cache-out") if f.endswith(".npz")}
rows = [r for r in rows if r["prompt_id"].replace("/", "__") not in done]
parts = [[] for _ in range(ngpu)]
loads = [0] * ngpu
for r in rows:  # greedy: heaviest session to lightest worker
    i = loads.index(min(loads))
    parts[i].append(r["prompt_id"])
    loads[i] += len(r["encoded_token_ids"])
for i, p in enumerate(parts):
    open(f"{root}/worker_{i}.ids", "w").write("\n".join(p))
print("partition loads (tokens):", loads, "pending:", len(rows))
PYEOF

S WORKERS-STARTING
PIDS=()
for i in $(seq 0 $((NGPU - 1))); do
  IDS=$(tr '\n' ' ' < "$ROOT/worker_$i.ids")
  [ -n "${IDS// /}" ] || continue
  CUDA_VISIBLE_DEVICES=$i PYTHONUNBUFFERED=1 \
    python3 "$ROOT/keep/benchmarks/produce_glm52_teich_teacher_cache.py" \
    --snapshot-dir "$HUB/snapshots/local" \
    --non-vq-package-dir "$ROOT/non-vq-package" \
    --teich-pack "$ROOT/teich-pack.json" \
    --out-dir "$OUT" \
    --heartbeat "$OUT/heartbeat_gpu$i.json" \
    --session-ids $IDS \
    > "$ROOT/worker_$i.log" 2>&1 &
  PIDS+=($!)
done

FAIL=0
for P in "${PIDS[@]}"; do wait "$P" || FAIL=1; done
if [ "$FAIL" -eq 1 ]; then
  S WORKERS-FINISHED-WITH-FAILURES
else
  S WORKERS-ALL-OK
fi

DONE=$(ls "$OUT"/*.npz 2>/dev/null | wc -l)
S "SESSIONS-DONE-$DONE"

S RESULTS-PUSH
/usr/local/bin/s5cmd --numworkers 24 cp "$OUT/" "s3://$BUCKET/teacher-cache-out/" \
  && S PUSH-OK || S PUSH-FAILED
S ALL-DONE
