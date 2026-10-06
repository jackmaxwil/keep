#!/usr/bin/env bash
# RUN ON THE GPU NODE (after gpu_env_setup.sh + pull_from_s3.sh).
# CUDA proof: runs the same 2-row validation pack the Mac produced references
# for (dense 2048-prefix + full DSA session), then compares captures.
# Go/no-go gate for the p5 full run.
set -euo pipefail

BUCKET="${BUCKET:?set BUCKET=keep-glm52-models-<acct>-<region>}"
ROOT="${ROOT:-/mnt/nvme}"
CAMPAIGN_DESCRIPTOR="${CAMPAIGN_DESCRIPTOR:-/etc/keep-glm52/campaign.json}"
REPO_DIR="${KEEP_REPO_DIR:-/opt/keep-campaign/repo}"
RUN_ID=$(jq -er '.run_id' "$CAMPAIGN_DESCRIPTOR")
CODE_SHA=$(jq -er '.repo_tar_sha256' "$CAMPAIGN_DESCRIPTOR")
RECORD_TYPE=$(jq -er '.record_type // "legacy_capacity_block_v1"' "$CAMPAIGN_DESCRIPTOR")
QUALIFICATION_STAGE="${GLM52_QUALIFICATION_STAGE:-full}"
case "$QUALIFICATION_STAGE" in
  full|source|replacement) ;;
  *) echo "invalid GLM52_QUALIFICATION_STAGE=$QUALIFICATION_STAGE" >&2; exit 64 ;;
esac
if [ "$RECORD_TYPE" = glm52_sky_campaign_descriptor_v2 ]; then
  EXECUTION_DEADLINE="${GLM52_EXECUTION_DEADLINE:?Sky execution deadline is required}"
  DEADLINE_ARGUMENTS=(--execution-deadline "$EXECUTION_DEADLINE")
else
  BLOCK_END=$(jq -er '.capacity_block_end' "$CAMPAIGN_DESCRIPTOR")
  DEADLINE_ARGUMENTS=(--capacity-block-end "$BLOCK_END")
fi
unset GLM_MLX_WIRED_LIMIT_GB GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB
cd "$REPO_DIR"

/usr/local/bin/s5cmd cp "s3://$BUCKET/validation/validation-pack.json" "$ROOT/validation-pack.json"
/usr/local/bin/s5cmd cp "s3://$BUCKET/validation/v5-parity-pack.json" "$ROOT/v5-parity-pack.json"
/usr/local/bin/s5cmd cp "s3://$BUCKET/validation/v5-ref/*" "$ROOT/v5-ref/"

if [ "$QUALIFICATION_STAGE" != replacement ]; then
  # Gate A: 66 audited eval prompts on CUDA vs top-K of the v5 fp32 cache
  echo "=== $(date -u +%FT%TZ) spike gate A: v5 parity (66 prompts) ==="
  PYTHONUNBUFFERED=1 python benchmarks/produce_glm52_teich_teacher_cache.py \
    --snapshot-dir "$ROOT/source-snapshot" \
    --non-vq-package-dir "$ROOT/non-vq-package" \
    --teich-pack "$ROOT/v5-parity-pack.json" \
    --out-dir "$ROOT/spike-v5-out" \
    --heartbeat "$ROOT/spike-v5-out/heartbeat.json"

  # Gate B: long-context DSA on a real teich session (dense prefix + full)
  for SID in \
    "teich_claude_agent-a3622521df8a9137d__dense_prefix2048" \
    "teich_claude_agent-a3622521df8a9137d__dsa_full"; do
    echo "=== $(date -u +%FT%TZ) spike gate B: $SID ==="
    PYTHONUNBUFFERED=1 python benchmarks/produce_glm52_teich_teacher_cache.py \
      --snapshot-dir "$ROOT/source-snapshot" \
      --non-vq-package-dir "$ROOT/non-vq-package" \
      --teich-pack "$ROOT/validation-pack.json" \
      --out-dir "$ROOT/spike-out" \
      --heartbeat "$ROOT/spike-out/heartbeat.json" \
      --session-ids "$SID"
  done
fi

# Gate C: force a real layer-boundary stop, then resume the same DSA session.
RESUME_SID="teich_claude_agent-a3622521df8a9137d__dsa_full"
RESUME_STOP="$ROOT/spike-resume/STOP"
mkdir -p "$ROOT/spike-resume/checkpoints" "$ROOT/spike-resume/out"
if [ "$QUALIFICATION_STAGE" != replacement ]; then
  touch "$RESUME_STOP"
  set +e
  PYTHONUNBUFFERED=1 python benchmarks/produce_glm52_teich_teacher_cache.py \
    --snapshot-dir "$ROOT/source-snapshot" \
    --non-vq-package-dir "$ROOT/non-vq-package" \
    --teich-pack "$ROOT/validation-pack.json" \
    --out-dir "$ROOT/spike-resume/out" \
    --heartbeat "$ROOT/spike-resume/heartbeat.json" \
    --session-ids "$RESUME_SID" \
    --checkpoint-dir "$ROOT/spike-resume/checkpoints" \
    --checkpoint-run-id "$RUN_ID-cuda-gate" \
    --code-tar-sha256 "$CODE_SHA" \
    --stop-file "$RESUME_STOP" \
    "${DEADLINE_ARGUMENTS[@]}"
  STOP_STATUS=$?
  set -e
  [ "$STOP_STATUS" -eq 75 ] || { echo "expected checkpointed stop, got $STOP_STATUS"; exit 1; }
  test -f "$ROOT/spike-resume/checkpoints/latest.json"
  if [ "$QUALIFICATION_STAGE" = source ]; then
    echo "CUDA-QUALIFICATION-SOURCE-READY"
    exit 0
  fi
fi
rm -f "$RESUME_STOP"
PYTHONUNBUFFERED=1 python benchmarks/produce_glm52_teich_teacher_cache.py \
  --snapshot-dir "$ROOT/source-snapshot" \
  --non-vq-package-dir "$ROOT/non-vq-package" \
  --teich-pack "$ROOT/validation-pack.json" \
  --out-dir "$ROOT/spike-resume/out" \
  --heartbeat "$ROOT/spike-resume/heartbeat.json" \
  --session-ids "$RESUME_SID" \
  --checkpoint-dir "$ROOT/spike-resume/checkpoints" \
  --checkpoint-run-id "$RUN_ID-cuda-gate" \
  --code-tar-sha256 "$CODE_SHA" \
  --stop-file "$RESUME_STOP" \
  "${DEADLINE_ARGUMENTS[@]}"
test -f "$ROOT/spike-resume/out/${RESUME_SID//\//__}.npz"
python - "$ROOT/spike-out/${RESUME_SID//\//__}.npz" "$ROOT/spike-resume/out/${RESUME_SID//\//__}.npz" <<'PY'
import sys
import numpy as np

with np.load(sys.argv[1], allow_pickle=False) as expected, np.load(
    sys.argv[2], allow_pickle=False
) as resumed:
    if set(expected.files) != set(resumed.files):
        raise SystemExit("resumed capture tensor inventory differs")
    for name in expected.files:
        left, right = expected[name], resumed[name]
        equal = (
            np.array_equal(left, right)
            if left.dtype.kind in "iub"
            else np.allclose(left, right, rtol=1e-3, atol=1e-3, equal_nan=False)
        )
        if not equal:
            raise SystemExit(f"resumed capture differs at tensor {name}")
print("CUDA-RESUME-PARITY-OK")
PY

echo "=== verdict ==="
python aws/glm52-gpu/scripts/compare_captures.py \
  --ref-dir "$ROOT/v5-ref" --new-dir "$ROOT/spike-v5-out" \
  --dsa-dir "$ROOT/spike-out" \
  --report "$ROOT/spike-out/parity-report.json"
if [ "$QUALIFICATION_STAGE" = full ]; then
  /usr/local/bin/s5cmd cp "$ROOT/spike-out/parity-report.json" \
    "s3://$BUCKET/validation/parity-report.json"
fi
cat "$ROOT/spike-out/parity-report.json"
