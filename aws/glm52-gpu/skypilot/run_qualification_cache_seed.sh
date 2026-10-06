#!/usr/bin/env bash
# Produce one real full-v2 row for the authenticated H100 training smoke.
set -euo pipefail

DESCRIPTOR="${CAMPAIGN_DESCRIPTOR:-/etc/keep-glm52/campaign.json}"
ROOT="${ROOT:-/mnt/nvme/glm52-campaign}"
REPO="${KEEP_REPO_DIR:-/opt/keep-campaign/repo}"
REGION=us-west-2
SEED_PROMPT_ID=teich_claude_agent-a3622521df8a9137d
WORK="$ROOT/qualification-cache-seed"
CAPTURES="$WORK/captures"
CHECKPOINTS="$WORK/checkpoints"
OUTPUT="$WORK/cache"
STOP=/run/keep-glm52/STOP
RUN_ID=$(jq -er .run_id "$DESCRIPTOR")
BUCKET=$(jq -er .bucket "$DESCRIPTOR")
CODE_SHA=$(jq -er .repo_tar_sha256 "$DESCRIPTOR")
DEADLINE="${GLM52_EXECUTION_DEADLINE:?execution deadline is required}"
CHECKPOINT_S3="s3://$BUCKET/campaigns/$RUN_ID/qualification-cache-seed/checkpoints"
unset GLM_MLX_WIRED_LIMIT_GB GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB

mkdir -p "$CAPTURES" "$CHECKPOINTS" "$OUTPUT"
aws s3 sync "$CHECKPOINT_S3/" "$CHECKPOINTS/" \
  --region "$REGION" --only-show-errors

read -r EXPECTED_POSITIONS <<EOF
$(python3 - "$ROOT/teich-pack.json" "$WORK/prompt-pack.json" "$SEED_PROMPT_ID" <<'PY'
import json, pathlib, sys
source = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
prompt_id = sys.argv[3]
payload = json.loads(source.read_bytes())
rows = [
    row for row in payload["prompt_rows"]
    if row.get("prompt_id") == prompt_id
]
if len(rows) != 1:
    raise SystemExit("qualification cache seed row is missing or duplicated")
subset = {
    **payload,
    "prompt_row_count": 1,
    "supervised_tokens": len(rows[0]["positions"]),
    "prompt_rows": rows,
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(subset, sort_keys=True, separators=(",", ":")) + "\n"
)
print(len(rows[0]["positions"]))
PY
)
EOF

sync_checkpoints() {
  while :; do
    "$REPO/aws/glm52-gpu/scripts/sync_checkpoint_tree.sh" \
      "$CHECKPOINTS" "$CHECKPOINT_S3"
    python3 - "$RUN_ID" "$WORK/campaign-heartbeat.json" <<'PY'
import json, pathlib, sys
from datetime import datetime, timezone
from mlx_vq.quality.glm52_campaign_watchdog import build_campaign_heartbeat
heartbeat = build_campaign_heartbeat(
    run_id=sys.argv[1],
    phase="QUALIFICATION_CACHE_SEED",
    observed_at=datetime.now(timezone.utc),
    ledger_record_sha256=None,
)
pathlib.Path(sys.argv[2]).write_text(
    json.dumps(heartbeat, sort_keys=True, separators=(",", ":")) + "\n"
)
PY
    aws s3 cp "$WORK/campaign-heartbeat.json" \
      "s3://$BUCKET/campaigns/$RUN_ID/monitor/heartbeat.json" \
      --region "$REGION" --only-show-errors
    sleep 300
  done
}
sync_checkpoints &
SYNC_PID=$!
PRODUCER_PID=
request_teacher_stop() {
  touch "$STOP"
  if [ -n "$PRODUCER_PID" ] && kill -0 "$PRODUCER_PID" 2>/dev/null; then
    kill -TERM "$PRODUCER_PID" 2>/dev/null || true
  fi
}
cleanup_sync() {
  if [ -n "$PRODUCER_PID" ] && kill -0 "$PRODUCER_PID" 2>/dev/null; then
    request_teacher_stop
    wait "$PRODUCER_PID" 2>/dev/null || true
  fi
  kill "$SYNC_PID" 2>/dev/null || true
  wait "$SYNC_PID" 2>/dev/null || true
  "$REPO/aws/glm52-gpu/scripts/sync_checkpoint_tree.sh" \
    "$CHECKPOINTS" "$CHECKPOINT_S3"
}
wait_for_producer() {
  local status
  while :; do
    status=0
    wait "$PRODUCER_PID" || status=$?
    if ! kill -0 "$PRODUCER_PID" 2>/dev/null; then
      return "$status"
    fi
  done
}
trap cleanup_sync EXIT
trap request_teacher_stop TERM INT

put_immutable() {
  local source=$1 key=$2 size sha_hex sha_b64 response status
  size=$(stat -c %s "$source")
  sha_hex=$(sha256sum "$source" | awk '{print $1}')
  sha_b64=$(printf '%s' "$sha_hex" | xxd -r -p | base64 | tr -d '\n')
  set +e
  response=$(aws s3api head-object \
    --bucket "$BUCKET" --key "$key" --checksum-mode ENABLED \
    --region "$REGION" --output json 2>&1)
  status=$?
  set -e
  if [ "$status" -eq 0 ]; then
    python3 - "$size" "$sha_b64" "$key" "$response" <<'PY'
import json, sys
expected_size, expected_sha, key = int(sys.argv[1]), sys.argv[2], sys.argv[3]
value = json.loads(sys.argv[4])
if (
    int(value["ContentLength"]) != expected_size
    or value.get("ChecksumSHA256") != expected_sha
    or value.get("ChecksumType") != "FULL_OBJECT"
):
    raise SystemExit(f"immutable S3 object differs: {key}")
PY
    return
  fi
  case "$response" in
    *404*|*"Not Found"*|*NoSuchKey*) ;;
    *) echo "$response" >&2; return "$status" ;;
  esac
  set +e
  response=$(aws s3api put-object \
    --bucket "$BUCKET" --key "$key" --body "$source" \
    --checksum-algorithm SHA256 --checksum-sha256 "$sha_b64" \
    --if-none-match '*' --metadata "glm52-run-id=$RUN_ID" \
    --region "$REGION" --output json 2>&1)
  status=$?
  set -e
  if [ "$status" -eq 0 ]; then
    return
  fi
  case "$response" in
    *PreconditionFailed*|*412*) put_immutable "$source" "$key" ;;
    *) echo "$response" >&2; return "$status" ;;
  esac
}

PYTHONUNBUFFERED=1 python3 "$REPO/benchmarks/produce_glm52_teich_teacher_cache.py" \
  --snapshot-dir "$ROOT/source-snapshot" \
  --non-vq-package-dir "$ROOT/non-vq-package" \
  --teich-pack "$WORK/prompt-pack.json" \
  --out-dir "$CAPTURES" \
  --heartbeat "$WORK/heartbeat.json" \
  --session-ids "$SEED_PROMPT_ID" \
  --checkpoint-dir "$CHECKPOINTS" \
  --checkpoint-run-id "$RUN_ID-qualification-cache-seed" \
  --code-tar-sha256 "$CODE_SHA" \
  --stop-file "$STOP" \
  --execution-deadline "$DEADLINE" &
PRODUCER_PID=$!
set +e
wait_for_producer
STATUS=$?
set -e
PRODUCER_PID=
if [ "$STATUS" -eq 75 ]; then
  exit 75
fi
test "$STATUS" -eq 0

python3 "$REPO/benchmarks/finalize_glm52_teich_teacher_cache.py" \
  --capture-dir "$CAPTURES" \
  --prompt-pack "$WORK/prompt-pack.json" \
  --frozen-prompt-pack "$ROOT/frozen-66.json" \
  --cache-dir "$OUTPUT" \
  --expected-sessions 1 \
  --expected-supervised-positions "$EXPECTED_POSITIONS" \
  --hidden-size 6144

MANIFEST="$OUTPUT/glm52-teacher-signal-cache-v3-manifest.json"
READY="$OUTPUT/TEACHER_CACHE_READY.json"
MANIFEST_SHA=$(sha256sum "$MANIFEST" | awk '{print $1}')
DESTINATION="qualification-cache/seeds/$RUN_ID/$MANIFEST_SHA"
while IFS= read -r -d '' path; do
  relative=${path#"$OUTPUT/teacher_signal/"}
  put_immutable "$path" "$DESTINATION/teacher_signal/$relative"
done < <(find "$OUTPUT/teacher_signal" -type f -print0 | sort -z)
put_immutable "$WORK/prompt-pack.json" "$DESTINATION/prompt-pack.json"
put_immutable \
  "$MANIFEST" "$DESTINATION/glm52-teacher-signal-cache-v3-manifest.json"
put_immutable "$READY" "$DESTINATION/TEACHER_CACHE_READY.json"

python3 - "$RUN_ID" "$DESTINATION/" "$MANIFEST" "$READY" \
  "$WORK/QUALIFICATION_CACHE_SEED_READY.json" <<'PY'
import hashlib, json, pathlib, sys
run_id, prefix = sys.argv[1:3]
manifest, ready, output = map(pathlib.Path, sys.argv[3:])
body = {
    "schema_version": 1,
    "record_type": "glm52_qualification_cache_seed_ready_v1",
    "run_id": run_id,
    "qualification_cache_prefix": prefix,
    "qualification_cache_manifest_sha256": hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest(),
    "teacher_cache_ready_sha256": hashlib.sha256(ready.read_bytes()).hexdigest(),
}
body["ready_body_sha256"] = hashlib.sha256(
    json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
output.write_text(json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n")
PY
put_immutable \
  "$WORK/QUALIFICATION_CACHE_SEED_READY.json" \
  "campaigns/$RUN_ID/qualification-cache-seed/QUALIFICATION_CACHE_SEED_READY.json"
