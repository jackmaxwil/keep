#!/bin/bash
set -euo pipefail
cd /Users/jack.mazac/Developer/keep
echo "teich-localref pid=$$ started=$(date -u +%FT%TZ)" > .keep-heavy-job.lock
trap 'rm -f .keep-heavy-job.lock' EXIT
SNAP=/Users/jack.mazac/.cache/huggingface/hub/models--0xSero--glm-5.2-reap-504B-v2/snapshots/6c9241aa05fb243a0edb7c804c213ec1cf5c920d
NONVQ=artifacts/build/glm52_reap_504b_materialization_probe_20260709/steps/non_vq_pack-a678bc3c/out
BASE=runs/teich-teacher-localref-20260713
PACK=$BASE/validation-pack.json
for MODE in dense_prefix2048 dsa_full; do
  SID="teich_claude_agent-a3622521df8a9137d__${MODE}"
  echo "=== $(date -u +%FT%TZ) running $SID ==="
  UV_CACHE_DIR=/tmp/keep-uv-cache PYTHONUNBUFFERED=1 \
    uv run python benchmarks/produce_glm52_teich_teacher_cache.py \
    --snapshot-dir "$SNAP" --non-vq-package-dir "$NONVQ" \
    --teich-pack "$PACK" --out-dir "$BASE/out" \
    --heartbeat "$BASE/heartbeat.json" --session-ids "$SID"
done
echo "=== LOCALREF COMPLETE $(date -u +%FT%TZ) ==="
