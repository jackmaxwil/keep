#!/bin/bash
set -euo pipefail
cd /Users/jack.mazac/Developer/keep
echo "verify-fix pid=$$ started=$(date -u +%FT%TZ)" > .keep-heavy-job.lock
trap 'rm -f .keep-heavy-job.lock' EXIT
SNAP=/Users/jack.mazac/.cache/huggingface/hub/models--0xSero--glm-5.2-reap-504B-v2/snapshots/6c9241aa05fb243a0edb7c804c213ec1cf5c920d
NONVQ=artifacts/build/glm52_reap_504b_materialization_probe_20260709/steps/non_vq_pack-a678bc3c/out
PACK="/private/tmp/claude-502/-Users-jack-mazac-Developer-keep/b4d50550-2788-47d9-b001-3dbad2f173e6/scratchpad/teich-corpus-v2/initial-cut/glm52-coding-agent-initial-v2-20260713.json"
OUT=runs/teich-teacher-full-metal-20260714/verify-fix
echo "=== $(date -u +%FT%TZ) verifying fix on the actual crashing session ==="
UV_CACHE_DIR=/tmp/keep-uv-cache PYTHONUNBUFFERED=1 \
  uv run python benchmarks/produce_glm52_teich_teacher_cache.py \
  --snapshot-dir "$SNAP" --non-vq-package-dir "$NONVQ" \
  --teich-pack "$PACK" --out-dir "$OUT/out" \
  --heartbeat "$OUT/heartbeat.json" \
  --session-ids teich_cursor_cursor-391780b9-c147-49c0-adc7-239d7454a411 \
  --query-chunk-size 256
echo "=== VERIFY-FIX COMPLETE $(date -u +%FT%TZ) ==="
