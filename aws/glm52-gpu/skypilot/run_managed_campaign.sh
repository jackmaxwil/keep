#!/usr/bin/env bash
# Supervise one authenticated campaign allocation under SkyPilot.
set -euo pipefail

DESCRIPTOR=/etc/keep-glm52/campaign.json
APPROVAL=/etc/keep-glm52/GPU_SPEND_APPROVAL.json
INTENT=/etc/keep-glm52/SKYPILOT_SUBMISSION_INTENT.json
LATCH=/etc/keep-glm52/WORKER_START_LATCH.json
LATCH_TRANSPORT=/etc/keep-glm52/WORKER_START_LATCH_TRANSPORT.json
ACCEPTED=/etc/keep-glm52/WORKER_START_ACCEPTED.json
RECEIPT=/etc/keep-glm52/WORKER_START_ADMISSION_RECEIPT.json
ROOT=/mnt/nvme/glm52-campaign
REPO=/opt/keep-campaign/repo
JOB_NAME="${GLM52_SKY_JOB_NAME:?GLM52_SKY_JOB_NAME is required}"
REGION=us-west-2
MAX_TRANSIENT_RESTARTS=2
QUALIFICATION_MAX_SECONDS=14400
MANAGED_MODE="${GLM52_MANAGED_MODE:?GLM52_MANAGED_MODE is required}"
LIVE_WORKER="${GLM52_LIVE_SKY_WORKER:?GLM52_LIVE_SKY_WORKER is required}"
AMBIENT_JOB_ID="${SKYPILOT_MANAGED_JOB_ID:?SKYPILOT_MANAGED_JOB_ID is required}"
unset GLM_MLX_WIRED_LIMIT_GB GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB

test "$MANAGED_MODE" = qualification
test "$LIVE_WORKER" = 1
/usr/bin/python3 - "$AMBIENT_JOB_ID" <<'PY'
import re
import sys

if re.fullmatch(r"[1-9][0-9]*", sys.argv[1]) is None:
    raise SystemExit("SkyPilot managed job ID is not canonical")
PY
test -f "$DESCRIPTOR"
test -f "$APPROVAL"
test -f "$INTENT"
test -f "$LATCH"
test -f "$LATCH_TRANSPORT"
test -f "$ACCEPTED"
test -f "$RECEIPT"
RECEIPT_FINGERPRINT=$(/usr/bin/python3 - "$RECEIPT" 384 <<'PY'
import hashlib
import os
import stat
import sys

EXPECTED_ROOT_UID = 0
path = sys.argv[1]
expected_mode = int(sys.argv[2])
flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(path, flags)
try:
    before = os.fstat(descriptor)
    chunks = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(descriptor)
finally:
    os.close(descriptor)
identity = lambda value: (
    value.st_dev,
    value.st_ino,
    value.st_mode,
    value.st_nlink,
    value.st_uid,
    value.st_size,
    value.st_mtime_ns,
)
if identity(before) != identity(after):
    raise SystemExit("admission receipt changed while reading")
if (
    not stat.S_ISREG(before.st_mode)
    or stat.S_IMODE(before.st_mode) != expected_mode
    or before.st_uid != EXPECTED_ROOT_UID
    or before.st_nlink != 1
):
    raise SystemExit("admission receipt metadata is unsafe")
digest = hashlib.sha256(b"".join(chunks)).hexdigest()
print(
    f"{before.st_dev}:{before.st_ino}:{before.st_size}:"
    f"{before.st_mtime_ns}:{digest}"
)
PY
)
INTENT_FINGERPRINT=$(/usr/bin/python3 - "$INTENT" 420 <<'PY'
import hashlib
import os
import stat
import sys

EXPECTED_ROOT_UID = 0
path = sys.argv[1]
expected_mode = int(sys.argv[2])
flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(path, flags)
try:
    before = os.fstat(descriptor)
    chunks = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(descriptor)
finally:
    os.close(descriptor)
identity = lambda value: (
    value.st_dev,
    value.st_ino,
    value.st_mode,
    value.st_nlink,
    value.st_uid,
    value.st_size,
    value.st_mtime_ns,
)
if identity(before) != identity(after):
    raise SystemExit("submission intent changed while reading")
if (
    not stat.S_ISREG(before.st_mode)
    or stat.S_IMODE(before.st_mode) != expected_mode
    or before.st_uid != EXPECTED_ROOT_UID
    or before.st_nlink != 1
):
    raise SystemExit("submission intent metadata is unsafe")
digest = hashlib.sha256(b"".join(chunks)).hexdigest()
print(
    f"{before.st_dev}:{before.st_ino}:{before.st_size}:"
    f"{before.st_mtime_ns}:{digest}"
)
PY
)
aws sts get-caller-identity |
  /usr/bin/python3 \
    "$REPO/aws/glm52-gpu/scripts/assert_rnd_aws_account.py" >/dev/null
/usr/bin/python3 -I "$REPO/aws/glm52-gpu/skypilot/publish_worker_start_v2.py" \
  verify-receipt-v2 \
  --publisher-file-sha256 4d1d47aef6dd211c20b52f05ede9bd6d32c35433e305f2b1bd56b48c83b9051e \
  --campaign-policy "$REPO/src/mlx_vq/quality/glm52_sky_campaign.py" \
  --campaign-policy-file-sha256 77412115f6b8ce325177864418d838e0335d536329e8e7565d7811bcc806cd94 \
  --campaign-policy-native "$REPO/src/glm52_enforcement/glm52_sky_campaign.py" \
  --campaign-policy-native-file-sha256 f023eaeddf8fac73fa6546e3c4a0b60dd32e5266f06e1519fe21e0444d445d03 \
  --must-start-policy "$REPO/src/mlx_vq/quality/glm52_sky_must_start.py" \
  --must-start-policy-file-sha256 478ac3a16ce8a77a4e844c1e0a3dca99454d50e7233fd7cd67dfa184b486860a \
  --must-start-policy-native "$REPO/src/glm52_enforcement/glm52_sky_must_start.py" \
  --must-start-policy-native-file-sha256 e910d3d03f7b30b7b9b668e214b61ecc7cbd641a53dedabd620b8e14573d33f3 \
  --dynamic-policy "$REPO/src/mlx_vq/quality/glm52_sky_must_start_dynamic.py" \
  --dynamic-policy-file-sha256 527019a41b54f810e9f98734c2ca99acc5a7b370344adf48c80f46f925396e18 \
  --worker-policy "$REPO/src/mlx_vq/quality/glm52_sky_worker_must_start_v2.py" \
  --worker-policy-file-sha256 8dfad6682d8afc9774dc31385cfc9ecc8c8e2f80d9a535e3c8d4cf6cf0fa97df \
  --coordinator "$REPO/aws/glm52-gpu/lambda/sky_worker_start_v2_coordinator.py" \
  --coordinator-file-sha256 efa7c48ad259e10c6ce0e4cb1db9ff0a6e6a9fa203274329a7c2eb5203366406 \
  --descriptor "$DESCRIPTOR" \
  --descriptor-s3-uri "${GLM52_CAMPAIGN_DESCRIPTOR_S3_URI:?required}" \
  --descriptor-file-sha256 "${GLM52_CAMPAIGN_DESCRIPTOR_SHA256:?required}" \
  --intent "$INTENT" \
  --intent-s3-uri "${GLM52_SKY_SUBMISSION_INTENT_S3_URI:?required}" \
  --intent-file-sha256 "${GLM52_SKY_SUBMISSION_INTENT_FILE_SHA256:?required}" \
  --intent-body-sha256 "${GLM52_SKY_SUBMISSION_INTENT_BODY_SHA256:?required}" \
  --latch "$LATCH" \
  --latch-transport "$LATCH_TRANSPORT" \
  --accepted "$ACCEPTED" \
  --admission-receipt "$RECEIPT"
GLM52_SKY_CONTROLLER_JOB_ID=$(/usr/bin/python3 - \
  "$RECEIPT" "$RECEIPT_FINGERPRINT" \
  "$INTENT" "$INTENT_FINGERPRINT" \
  "$JOB_NAME" "$MANAGED_MODE" "$AMBIENT_JOB_ID" <<'PY'
import hashlib
import json
import os
import stat
import sys

EXPECTED_ROOT_UID = 0

def stable_read(path, expected_mode, expected_fingerprint, label):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after):
        raise SystemExit(f"{label} changed while reading")
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != expected_mode
        or before.st_uid != EXPECTED_ROOT_UID
        or before.st_nlink != 1
    ):
        raise SystemExit(f"{label} metadata is unsafe")
    raw = b"".join(chunks)
    fingerprint = (
        f"{before.st_dev}:{before.st_ino}:{before.st_size}:"
        f"{before.st_mtime_ns}:{hashlib.sha256(raw).hexdigest()}"
    )
    if fingerprint != expected_fingerprint:
        raise SystemExit(f"{label} changed after verification began")
    return raw

receipt = json.loads(stable_read(sys.argv[1], 0o600, sys.argv[2], "receipt"))
intent = json.loads(stable_read(sys.argv[3], 0o644, sys.argv[4], "intent"))
if type(receipt) is not dict:
    raise SystemExit("authenticated admission receipt is not an object")
if receipt.get("record_type") != "glm52_sky_worker_admission_receipt_v1":
    raise SystemExit("authenticated admission receipt record type is invalid")
if type(intent) is not dict:
    raise SystemExit("authenticated submission intent is not an object")
if intent.get("sky_job_name") != sys.argv[5]:
    raise SystemExit("authenticated submission intent job name mismatch")
if intent.get("managed_mode") != sys.argv[6] or sys.argv[6] != "qualification":
    raise SystemExit("authenticated submission intent mode is invalid")
job_id = receipt.get("sky_job_id")
if type(job_id) is not int or job_id <= 0:
    raise SystemExit("authenticated controller job ID is invalid")
if str(job_id) != sys.argv[7]:
    raise SystemExit("authenticated controller job ID disagrees with SkyPilot")
print(job_id)
PY
)
export GLM52_SKY_CONTROLLER_JOB_ID
JOB_ID="$GLM52_SKY_CONTROLLER_JOB_ID"
mkdir -p "$ROOT/runtime" /run/keep-glm52
RUN_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$DESCRIPTOR")
BUCKET=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["bucket"])' "$DESCRIPTOR")
S3_RUNTIME="s3://$BUCKET/campaigns/$RUN_ID/runtime"
S3_MONITOR="s3://$BUCKET/campaigns/$RUN_ID/monitor"

"$REPO/aws/glm52-gpu/scripts/restore_gpu_spend_ledger.sh" \
  "$ROOT" "$S3_RUNTIME"

TOKEN=$(curl -fsS -X PUT \
  -H 'X-aws-ec2-metadata-token-ttl-seconds: 300' \
  http://169.254.169.254/latest/api/token)
INSTANCE_ID=$(curl -fsS \
  -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-id)
LAUNCH_TIME=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --region "$REGION" --query 'Reservations[0].Instances[0].LaunchTime' --output text)
OBSERVED_AT=$(date -u +%FT%TZ)
WORKER_STATUS=RUNNING

# A recovered worker must close the prior allocation before starting its own.
# The new launch time is a conservative upper bound when the prior exact stop
# time is unavailable, so this path can underuse but never exceed approval.
if [ -s "$ROOT/runtime/GPU_SPEND_LEDGER.jsonl" ]; then
  read -r LAST_EVENT PREVIOUS_INSTANCE PREVIOUS_LAUNCH <<EOF
$(python3 - "$ROOT/runtime/GPU_SPEND_LEDGER.jsonl" <<'PY'
import json, pathlib, sys
lines = pathlib.Path(sys.argv[1]).read_text().splitlines()
last = json.loads(lines[-1])
starts = [
    json.loads(line)
    for line in lines
    if json.loads(line)["event"] == "allocation_started"
]
print(last["event"], last["instance_id"], starts[-1]["timestamp"])
PY
)
EOF
  if [ "$LAST_EVENT" = allocation_started ] && [ "$PREVIOUS_INSTANCE" != "$INSTANCE_ID" ]; then
    WORKER_STATUS=RECOVERING
    set +e
    PREVIOUS_DESCRIPTION=$(aws ec2 describe-instances \
      --instance-ids "$PREVIOUS_INSTANCE" --region "$REGION" --output json 2>/dev/null)
    DESCRIPTION_STATUS=$?
    set -e
    if [ "$DESCRIPTION_STATUS" -ne 0 ]; then
      PREVIOUS_DESCRIPTION='{}'
    fi
    PREVIOUS_ENDED_AT=$(printf '%s' "$PREVIOUS_DESCRIPTION" |
      "$REPO/aws/glm52-gpu/scripts/resolve_previous_allocation_end.py" \
        --instance-id "$PREVIOUS_INSTANCE" \
        --previous-launch "$PREVIOUS_LAUNCH" \
        --replacement-launch "$LAUNCH_TIME")
    "$REPO/aws/glm52-gpu/scripts/manage_gpu_spend.py" end \
      --descriptor "$DESCRIPTOR" --approval "$APPROVAL" --root "$ROOT" \
      --instance-id "$PREVIOUS_INSTANCE" --ended-at "$PREVIOUS_ENDED_AT"
  fi
fi

aws ec2 describe-instances \
  --region "$REGION" \
  --filters \
    "Name=tag:campaign-run-id,Values=$RUN_ID" \
    "Name=tag:project,Values=keep-glm52" \
    "Name=tag:owner,Values=jack.mazac" \
    "Name=tag:model,Values=glm-5.2" \
    "Name=tag:cost-allocation,Values=glm52-sky-campaign" \
    "Name=instance-type,Values=p5.48xlarge" \
  --output json > "$ROOT/runtime/tagged-ec2-history.json"
"$REPO/aws/glm52-gpu/scripts/find_unrecorded_gpu_allocations.py" \
  --run-id "$RUN_ID" \
  --ledger "$ROOT/runtime/GPU_SPEND_LEDGER.jsonl" \
  --current-instance-id "$INSTANCE_ID" \
  --replacement-launch "$LAUNCH_TIME" \
  < "$ROOT/runtime/tagged-ec2-history.json" \
  > "$ROOT/runtime/unrecorded-gpu-allocations.tsv"
while IFS="$(printf '\t')" read -r PRIOR_INSTANCE PRIOR_LAUNCH PRIOR_END; do
  test -n "$PRIOR_INSTANCE"
  "$REPO/aws/glm52-gpu/scripts/manage_gpu_spend.py" reconcile \
    --descriptor "$DESCRIPTOR" --approval "$APPROVAL" --root "$ROOT" \
    --job-id "$JOB_ID" --instance-id "$PRIOR_INSTANCE" \
    --launched-at "$PRIOR_LAUNCH" --ended-at "$PRIOR_END"
done < "$ROOT/runtime/unrecorded-gpu-allocations.tsv"
if [ -s "$ROOT/runtime/unrecorded-gpu-allocations.tsv" ]; then
  WORKER_STATUS=RECOVERING
  "$REPO/aws/glm52-gpu/scripts/sync_gpu_spend_ledger.sh" \
    "$ROOT" "$S3_RUNTIME"
fi

"$REPO/aws/glm52-gpu/scripts/manage_gpu_spend.py" start \
  --descriptor "$DESCRIPTOR" --approval "$APPROVAL" --root "$ROOT" \
  --job-id "$JOB_ID" --instance-id "$INSTANCE_ID" \
  --launched-at "$LAUNCH_TIME" --observed-at "$OBSERVED_AT"
"$REPO/aws/glm52-gpu/scripts/sync_gpu_spend_ledger.sh" \
  "$ROOT" "$S3_RUNTIME"
aws s3 cp "$ROOT/runtime/GPU_RUNTIME_ALLOCATION.json" \
  "$S3_RUNTIME/GPU_RUNTIME_ALLOCATION.json" --region "$REGION" --only-show-errors
"$REPO/aws/glm52-gpu/scripts/publish_sky_job_status.py" \
  --descriptor "$DESCRIPTOR" --job-name "$JOB_NAME" \
  --status "$WORKER_STATUS" --instance-id "$INSTANCE_ID" \
  --output "$ROOT/runtime/SKY_JOB_STATUS.json"
aws s3 cp "$ROOT/runtime/SKY_JOB_STATUS.json" \
  "$S3_MONITOR/SKY_JOB_STATUS.json" \
  --region "$REGION" --only-show-errors

set -a
. "$ROOT/runtime/campaign.env"
set +a
AUTHORIZATION_DEADLINE="$GLM52_EXECUTION_DEADLINE"
PHASE_DEADLINE_ACTIVE=0
PHASE_MAX_SECONDS=$QUALIFICATION_MAX_SECONDS
if [ "$PHASE_MAX_SECONDS" -gt 0 ]; then
  GLM52_EXECUTION_DEADLINE=$(python3 - \
    "$ROOT/runtime/GPU_SPEND_LEDGER.jsonl" "$JOB_ID" \
    "$PHASE_MAX_SECONDS" "$AUTHORIZATION_DEADLINE" <<'PY'
import json, pathlib, sys
from datetime import datetime, timedelta, timezone

ledger_path = pathlib.Path(sys.argv[1])
job_id = sys.argv[2]
phase_limit = int(sys.argv[3])
authorization_deadline = datetime.fromisoformat(
    sys.argv[4].replace("Z", "+00:00")
)
now = datetime.now(timezone.utc)
active = None
consumed = 0.0
for line in ledger_path.read_text().splitlines():
    record = json.loads(line)
    timestamp = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
    instance_id = record["instance_id"]
    if record["event"] == "allocation_started":
        if active is not None:
            raise SystemExit("duplicate open phase allocation")
        active = (record["job_id"], instance_id, timestamp)
    elif record["event"] == "allocation_ended":
        if active is None or active[1] != instance_id or timestamp < active[2]:
            raise SystemExit("invalid phase allocation chain")
        if active[0] == job_id:
            consumed += (timestamp - active[2]).total_seconds()
        active = None
if active is not None and active[0] == job_id:
    consumed += max(0.0, (now - active[2]).total_seconds())
remaining = max(0, int(phase_limit - consumed))
deadline = min(authorization_deadline, now + timedelta(seconds=remaining))
print(deadline.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"))
PY
)
  if [ "$GLM52_EXECUTION_DEADLINE" != "$AUTHORIZATION_DEADLINE" ]; then
    PHASE_DEADLINE_ACTIVE=1
  fi
  export GLM52_EXECUTION_DEADLINE
fi
DEADLINE_EPOCH=$(date -u -d "$GLM52_EXECUTION_DEADLINE" +%s)
deadline_guard() {
  local stop_requested=0 term_sent=0 drain_required=0
  while :; do
    local now remaining
    now=$(date -u +%s)
    remaining=$((DEADLINE_EPOCH - now))
    if [ "$remaining" -le 3600 ] && [ "$stop_requested" -eq 0 ]; then
      touch /run/keep-glm52/STOP
      stop_requested=1
    fi
    if [ "$remaining" -le 3000 ] && [ "$term_sent" -eq 0 ]; then
      systemctl kill --signal=TERM keep-glm52-campaign.service 2>/dev/null || true
      if [ -s /run/keep-glm52/ACTIVE_CHILD_PID ]; then
        kill -TERM "$(cat /run/keep-glm52/ACTIVE_CHILD_PID)" 2>/dev/null || true
      fi
      term_sent=1
    fi
    if [ "$remaining" -le 1800 ] && [ "$drain_required" -eq 0 ]; then
      if [ ! -f "$ROOT/CAMPAIGN_DRAINED.json" ]; then
        touch /run/keep-glm52/AUTHORIZATION_DRAIN_REQUIRED
        systemctl kill --signal=TERM keep-glm52-campaign.service 2>/dev/null || true
        if [ -s /run/keep-glm52/ACTIVE_CHILD_PID ]; then
          kill -TERM "$(cat /run/keep-glm52/ACTIVE_CHILD_PID)" 2>/dev/null || true
        fi
      fi
      drain_required=1
    fi
    if [ -f "$ROOT/CAMPAIGN_DRAINED.json" ]; then
      return 0
    fi
    if [ "$remaining" -le 0 ]; then
      if [ "$PHASE_DEADLINE_ACTIVE" -eq 1 ]; then
        touch /run/keep-glm52/PHASE_DEADLINE_EXHAUSTED
      else
        touch /run/keep-glm52/AUTHORIZATION_EXHAUSTED
      fi
      systemctl kill --signal=KILL keep-glm52-campaign.service 2>/dev/null || true
      if [ -s /run/keep-glm52/ACTIVE_CHILD_PID ]; then
        kill -KILL "$(cat /run/keep-glm52/ACTIVE_CHILD_PID)" 2>/dev/null || true
      fi
      return 0
    fi
    sleep 15
  done
}
deadline_guard &
DEADLINE_GUARD_PID=$!
cleanup_deadline_guard() {
  kill "$DEADLINE_GUARD_PID" 2>/dev/null || true
  wait "$DEADLINE_GUARD_PID" 2>/dev/null || true
}
trap cleanup_deadline_guard EXIT

request_stop() {
  touch /run/keep-glm52/STOP
  systemctl kill --signal=TERM keep-glm52-campaign.service 2>/dev/null || true
  if [ -s /run/keep-glm52/ACTIVE_CHILD_PID ]; then
    kill -TERM "$(cat /run/keep-glm52/ACTIVE_CHILD_PID)" 2>/dev/null || true
  fi
}
trap request_stop TERM INT

wait_for_direct_child() {
  local child_pid=$1 status
  while :; do
    status=0
    wait "$child_pid" || status=$?
    if ! kill -0 "$child_pid" 2>/dev/null; then
      return "$status"
    fi
  done
}

run_direct_child_with_retries() {
  local command=$1 attempt=0 status
  while :; do
    set +e
    "$command" &
    local child_pid=$!
    printf '%s\n' "$child_pid" > /run/keep-glm52/ACTIVE_CHILD_PID
    wait_for_direct_child "$child_pid"
    status=$?
    set -e
    rm -f /run/keep-glm52/ACTIVE_CHILD_PID
    if (
      [ "$status" -eq 75 ] &&
      [ "$attempt" -lt "$MAX_TRANSIENT_RESTARTS" ] &&
      [ ! -f /run/keep-glm52/STOP ] &&
      [ ! -f /run/keep-glm52/AUTHORIZATION_EXHAUSTED ] &&
      [ ! -f /run/keep-glm52/PHASE_DEADLINE_EXHAUSTED ]
    ); then
      attempt=$((attempt + 1))
      sleep 60
      continue
    fi
    if [ "$status" -eq 75 ]; then
      status=70
    fi
    return "$status"
  done
}

wait_for_campaign_service() {
  while systemctl is-active --quiet keep-glm52-campaign.service; do
    sleep 15 || true
  done
}

set +e
run_direct_child_with_retries \
  /tmp/glm52-worker-start-v2/run_h100_qualification.sh
STATUS=$?
set -e

if [ -f /run/keep-glm52/AUTHORIZATION_EXHAUSTED ]; then
  STATUS=70
fi
if [ -f /run/keep-glm52/PHASE_DEADLINE_EXHAUSTED ]; then
  STATUS=70
fi

ENDED_AT=$(date -u +%FT%TZ)
"$REPO/aws/glm52-gpu/scripts/manage_gpu_spend.py" end \
  --descriptor "$DESCRIPTOR" --approval "$APPROVAL" --root "$ROOT" \
  --instance-id "$INSTANCE_ID" --ended-at "$ENDED_AT"
"$REPO/aws/glm52-gpu/scripts/sync_gpu_spend_ledger.sh" \
  "$ROOT" "$S3_RUNTIME"
aws s3 cp "$ROOT/runtime/GPU_SPEND_STATUS.json" \
  "$S3_RUNTIME/GPU_SPEND_STATUS.json" --region "$REGION" --only-show-errors
FINAL_SKY_STATUS=FAILED
if [ "$STATUS" -eq 0 ]; then
  FINAL_SKY_STATUS=SUCCEEDED
fi
"$REPO/aws/glm52-gpu/scripts/publish_sky_job_status.py" \
  --descriptor "$DESCRIPTOR" --job-name "$JOB_NAME" \
  --status "$FINAL_SKY_STATUS" --instance-id "$INSTANCE_ID" \
  --output "$ROOT/runtime/SKY_JOB_STATUS.json"
aws s3 cp "$ROOT/runtime/SKY_JOB_STATUS.json" \
  "$S3_MONITOR/SKY_JOB_STATUS.json" \
  --region "$REGION" --only-show-errors

if [ "$STATUS" -eq 0 ]; then
  test -f "$ROOT/qualification/H100_RESUME_READY.json"
  exit 0
fi
if [ "$STATUS" -eq 70 ]; then
  exit 70
fi
exit "$STATUS"
