#!/usr/bin/env bash
# Prove real H100 checkpoint recovery across two distinct SkyPilot workers.
set -euo pipefail

DESCRIPTOR="${CAMPAIGN_DESCRIPTOR:-/etc/keep-glm52/campaign.json}"
ROOT="${ROOT:-/mnt/nvme/glm52-campaign}"
REPO="${KEEP_REPO_DIR:-/opt/keep-campaign/repo}"
H100_POLICY_PATH="${GLM52_H100_POLICY_PATH:-$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/glm52_h100_qualification.py}"
test -f "$H100_POLICY_PATH"
test ! -L "$H100_POLICY_PATH"
export GLM52_H100_POLICY_PATH
REGION=us-west-2
unset GLM_MLX_WIRED_LIMIT_GB GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB
RUN_ID=$(jq -er .run_id "$DESCRIPTOR")
BUCKET=$(jq -er .bucket "$DESCRIPTOR")
JOB_ID="${GLM52_SKY_CONTROLLER_JOB_ID:?GLM52_SKY_CONTROLLER_JOB_ID is required}"
PREFIX="s3://$BUCKET/campaigns/$RUN_ID/qualification"
ALLOCATION="$ROOT/runtime/GPU_RUNTIME_ALLOCATION.json"
INSTANCE_ID=$(jq -er .instance_id "$ALLOCATION")
QUALIFICATION_IMDS_TOKEN=$(curl -fsS -X PUT \
  -H 'X-aws-ec2-metadata-token-ttl-seconds: 300' \
  http://169.254.169.254/latest/api/token)
ACTUAL_INSTANCE_ID=$(curl -fsS \
  -H "X-aws-ec2-metadata-token: $QUALIFICATION_IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-id)
test "$INSTANCE_ID" = "$ACTUAL_INSTANCE_ID"
ALLOCATION_RECORD=$(jq -er .gpu_spend_record_sha256 "$ALLOCATION")
SOURCE_MARKER="$ROOT/qualification/SOURCE_NODE_READY.json"
TERMINATION_MARKER="$ROOT/qualification/QUALIFICATION_TERMINATION_REQUESTED.json"
mkdir -p "$ROOT/qualification"

validate_current_allocation() {
  /usr/bin/python3 -I - "$DESCRIPTOR" "$ALLOCATION" "$REPO" "$JOB_ID" <<'PY'
import hashlib, importlib.util, json, os, pathlib, sys

descriptor_path, allocation_path, repo = map(pathlib.Path, sys.argv[1:4])
job_id = sys.argv[4]
module_path = pathlib.Path(os.environ["GLM52_H100_POLICY_PATH"])
spec = importlib.util.spec_from_file_location(
    "_glm52_h100_prework_allocation",
    module_path,
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)
descriptor = json.loads(descriptor_path.read_bytes())
allocation = json.loads(allocation_path.read_bytes())
spend_authority_body = {
    "record_type": "glm52_gpu_spend_ledger_genesis_v1",
    "run_id": descriptor["run_id"],
    "approval_sha256": descriptor["approval_sha256"],
    "approved_gpu_runtime_seconds": descriptor["approved_gpu_runtime_seconds"],
    "approved_gpu_cost_usd": descriptor["approved_gpu_cost_usd"],
    "hourly_cost_usd": descriptor["max_hourly_cost_usd"],
}
spend_authority = hashlib.sha256(
    json.dumps(
        spend_authority_body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
).hexdigest()
module.validate_h100_runtime_allocation(
    allocation,
    expected_run_id=descriptor["run_id"],
    expected_job_id=job_id,
    expected_instance_id=allocation["instance_id"],
    expected_approval_sha256=descriptor["approval_sha256"],
    expected_gpu_spend_authority_sha256=spend_authority,
    max_remaining_gpu_seconds=descriptor["approved_gpu_runtime_seconds"],
    max_estimated_gpu_cost_usd=descriptor["approved_gpu_cost_usd"],
)
PY
}

validate_current_allocation

publish_source_marker() {
  local result status winner="$ROOT/qualification/SOURCE_NODE_READY.winner.json"
  set +e
  result=$(aws s3api put-object \
    --bucket "$BUCKET" \
    --key "campaigns/$RUN_ID/qualification/SOURCE_NODE_READY.json" \
    --body "$SOURCE_MARKER" \
    --if-none-match '*' \
    --checksum-algorithm SHA256 \
    --metadata \
      "glm52-run-id=$RUN_ID,glm52-source-body-sha256=$(jq -er .source_body_sha256 "$SOURCE_MARKER")" \
    --region "$REGION" 2>&1)
  status=$?
  set -e
  if [ "$status" -eq 0 ]; then
    return
  fi
  case "$result" in
    *PreconditionFailed*|*412*)
      aws s3 cp "$PREFIX/SOURCE_NODE_READY.json" "$winner" \
        --region "$REGION" --only-show-errors
      /usr/bin/python3 -I - "$winner" "$DESCRIPTOR" "$ALLOCATION" \
        "$ROOT/spike-resume/checkpoints/latest.json" "$REPO" <<'PY'
import hashlib, importlib.util, json, os, pathlib, sys

winner, descriptor, allocation, latest, repo = map(pathlib.Path, sys.argv[1:])
module_path = pathlib.Path(os.environ["GLM52_H100_POLICY_PATH"])
spec = importlib.util.spec_from_file_location("_glm52_h100_source_winner", module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)
d = json.loads(descriptor.read_bytes())
a = json.loads(allocation.read_bytes())
module.validate_h100_source_node_ready(
    json.loads(winner.read_bytes()),
    expected_run_id=d["run_id"],
    expected_campaign_identity_sha256=d["campaign_identity_sha256"],
    expected_repo_tar_sha256=d["repo_tar_sha256"],
    expected_descriptor_body_sha256=d["descriptor_body_sha256"],
    expected_instance_id=a["instance_id"],
    expected_allocation_record_sha256=a["gpu_spend_record_sha256"],
    expected_allocation_body_sha256=a["allocation_body_sha256"],
    expected_checkpoint_marker_sha256=hashlib.sha256(
        latest.read_bytes()
    ).hexdigest(),
)
PY
      mv "$winner" "$SOURCE_MARKER"
      ;;
    *)
      echo "$result" >&2
      return "$status"
      ;;
  esac
}

if ! aws s3api head-object --bucket "$BUCKET" \
  --key "campaigns/$RUN_ID/qualification/SOURCE_NODE_READY.json" \
  --region "$REGION" >/dev/null 2>&1; then
  GLM52_QUALIFICATION_STAGE=source \
    "$REPO/aws/glm52-gpu/scripts/run_spike.sh"
  aws s3 sync "$ROOT/spike-out/" "$PREFIX/spike-out/" \
    --region "$REGION" --only-show-errors
  aws s3 sync "$ROOT/spike-v5-out/" "$PREFIX/spike-v5-out/" \
    --region "$REGION" --only-show-errors
  aws s3 sync "$ROOT/spike-resume/" "$PREFIX/spike-resume/" \
    --region "$REGION" --only-show-errors
  /usr/bin/python3 -I - "$DESCRIPTOR" "$ALLOCATION" \
    "$ROOT/spike-resume/checkpoints/latest.json" "$SOURCE_MARKER" "$REPO" \
    "$JOB_ID" <<'PY'
import hashlib, importlib.util, json, os, pathlib, sys
from datetime import datetime, timezone

descriptor, allocation, latest, output, repo = map(pathlib.Path, sys.argv[1:6])
job_id = sys.argv[6]
module_path = pathlib.Path(os.environ["GLM52_H100_POLICY_PATH"])
spec = importlib.util.spec_from_file_location("_glm52_h100_source", module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)
d = json.loads(descriptor.read_bytes())
a = json.loads(allocation.read_bytes())
spend_authority_body = {
    "record_type": "glm52_gpu_spend_ledger_genesis_v1",
    "run_id": d["run_id"],
    "approval_sha256": d["approval_sha256"],
    "approved_gpu_runtime_seconds": d["approved_gpu_runtime_seconds"],
    "approved_gpu_cost_usd": d["approved_gpu_cost_usd"],
    "hourly_cost_usd": d["max_hourly_cost_usd"],
}
spend_authority = hashlib.sha256(
    json.dumps(
        spend_authority_body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
).hexdigest()
module.validate_h100_runtime_allocation(
    a,
    expected_run_id=d["run_id"],
    expected_job_id=job_id,
    expected_instance_id=a["instance_id"],
    expected_approval_sha256=d["approval_sha256"],
    expected_gpu_spend_authority_sha256=spend_authority,
    max_remaining_gpu_seconds=d["approved_gpu_runtime_seconds"],
    max_estimated_gpu_cost_usd=d["approved_gpu_cost_usd"],
)
marker = module.build_h100_source_node_ready(
    run_id=d["run_id"],
    campaign_identity_sha256=d["campaign_identity_sha256"],
    repo_tar_sha256=d["repo_tar_sha256"],
    descriptor_body_sha256=d["descriptor_body_sha256"],
    instance_id=a["instance_id"],
    allocation_record_sha256=a["gpu_spend_record_sha256"],
    allocation_body_sha256=a["allocation_body_sha256"],
    checkpoint_marker_sha256=hashlib.sha256(latest.read_bytes()).hexdigest(),
    published_at=datetime.now(timezone.utc),
)
module.validate_h100_source_node_ready(
    marker,
    expected_run_id=d["run_id"],
    expected_campaign_identity_sha256=d["campaign_identity_sha256"],
    expected_repo_tar_sha256=d["repo_tar_sha256"],
    expected_descriptor_body_sha256=d["descriptor_body_sha256"],
    expected_instance_id=a["instance_id"],
    expected_allocation_record_sha256=a["gpu_spend_record_sha256"],
    expected_allocation_body_sha256=a["allocation_body_sha256"],
)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n")
PY
  publish_source_marker
fi

aws s3 cp "$PREFIX/SOURCE_NODE_READY.json" "$SOURCE_MARKER" \
  --region "$REGION" --only-show-errors
FIRST_INSTANCE=$(/usr/bin/python3 -I - "$SOURCE_MARKER" "$DESCRIPTOR" "$ALLOCATION" \
  "$REPO" "$JOB_ID" <<'PY'
import hashlib, importlib.util, json, os, pathlib, sys

marker_path, descriptor_path, allocation_path, repo = map(
    pathlib.Path, sys.argv[1:5]
)
job_id = sys.argv[5]
module_path = pathlib.Path(os.environ["GLM52_H100_POLICY_PATH"])
spec = importlib.util.spec_from_file_location("_glm52_h100_source_role", module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)
descriptor = json.loads(descriptor_path.read_bytes())
allocation = json.loads(allocation_path.read_bytes())
marker = module.validate_h100_source_node_ready(
    json.loads(marker_path.read_bytes()),
    expected_run_id=descriptor["run_id"],
    expected_campaign_identity_sha256=descriptor["campaign_identity_sha256"],
    expected_repo_tar_sha256=descriptor["repo_tar_sha256"],
    expected_descriptor_body_sha256=descriptor["descriptor_body_sha256"],
)
spend_authority_body = {
    "record_type": "glm52_gpu_spend_ledger_genesis_v1",
    "run_id": descriptor["run_id"],
    "approval_sha256": descriptor["approval_sha256"],
    "approved_gpu_runtime_seconds": descriptor["approved_gpu_runtime_seconds"],
    "approved_gpu_cost_usd": descriptor["approved_gpu_cost_usd"],
    "hourly_cost_usd": descriptor["max_hourly_cost_usd"],
}
spend_authority = hashlib.sha256(
    json.dumps(
        spend_authority_body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
).hexdigest()
module.validate_h100_runtime_allocation(
    allocation,
    expected_run_id=descriptor["run_id"],
    expected_job_id=job_id,
    expected_instance_id=allocation["instance_id"],
    expected_approval_sha256=descriptor["approval_sha256"],
    expected_gpu_spend_authority_sha256=spend_authority,
    max_remaining_gpu_seconds=descriptor["approved_gpu_runtime_seconds"],
    max_estimated_gpu_cost_usd=descriptor["approved_gpu_cost_usd"],
)
if marker["instance_id"] == allocation["instance_id"]:
    module.validate_h100_source_node_ready(
        marker,
        expected_allocation_record_sha256=allocation[
            "gpu_spend_record_sha256"
        ],
        expected_allocation_body_sha256=allocation["allocation_body_sha256"],
    )
print(marker["instance_id"])
PY
)
if [ "$FIRST_INSTANCE" = "$INSTANCE_ID" ]; then
  # AWS control-plane coordination terminates this exact worker after publication.
  while :; do sleep 30; done
fi

aws s3 sync "$PREFIX/spike-out/" "$ROOT/spike-out/" \
  --region "$REGION" --only-show-errors
aws s3 sync "$PREFIX/spike-v5-out/" "$ROOT/spike-v5-out/" \
  --region "$REGION" --only-show-errors
aws s3 sync "$PREFIX/spike-resume/" "$ROOT/spike-resume/" \
  --region "$REGION" --only-show-errors
aws s3 cp "$PREFIX/QUALIFICATION_TERMINATION_REQUESTED.json" \
  "$TERMINATION_MARKER" --region "$REGION" --only-show-errors

read -r FIRST_INSTANCE FIRST_ALLOCATION SOURCE_CHECKPOINT <<EOF
$(/usr/bin/python3 -I - "$SOURCE_MARKER" "$TERMINATION_MARKER" "$DESCRIPTOR" \
  "$INSTANCE_ID" "$REPO" <<'PY'
import importlib.util, json, os, pathlib, sys

marker_path = pathlib.Path(sys.argv[1])
request_path = pathlib.Path(sys.argv[2])
descriptor_path = pathlib.Path(sys.argv[3])
replacement = sys.argv[4]
repo = pathlib.Path(sys.argv[5])
module_path = pathlib.Path(os.environ["GLM52_H100_POLICY_PATH"])
spec = importlib.util.spec_from_file_location("_glm52_h100_replacement", module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)
marker = json.loads(marker_path.read_bytes())
request = json.loads(request_path.read_bytes())
descriptor = json.loads(descriptor_path.read_bytes())
marker = module.validate_h100_source_node_ready(
    marker,
    expected_run_id=descriptor["run_id"],
    expected_campaign_identity_sha256=descriptor["campaign_identity_sha256"],
    expected_repo_tar_sha256=descriptor["repo_tar_sha256"],
    expected_descriptor_body_sha256=descriptor["descriptor_body_sha256"],
)
module.validate_h100_termination_requested(request, source_marker=marker)
if marker["instance_id"] == replacement:
    raise SystemExit("qualification replacement did not use a distinct instance")
print(
    marker["instance_id"],
    marker["allocation_record_sha256"],
    marker["checkpoint_marker_sha256"],
)
PY
)
EOF

GLM52_QUALIFICATION_STAGE=replacement \
  "$REPO/aws/glm52-gpu/scripts/run_spike.sh"

QUALIFICATION_MANIFEST=$(jq -er '.artifacts.qualification_cache_manifest_sha256' "$DESCRIPTOR")
SMOKE="$ROOT/qualification/training-smoke.json"
CUDA_VISIBLE_DEVICES=0 \
GLM52_TRAINING_BASELINE_JSON="$ROOT/training-baseline/training-baseline.json" \
GLM52_TRAIN_PROGRESS=1 \
GLM52_TRAIN_GRADIENT_CHECKPOINT=1 \
python3 "$REPO/benchmarks/finetune_glm52_teich_resumable.py" \
  --teacher-cache-dir "$ROOT/qualification-cache" \
  --prompt-pack "$ROOT/qualification-cache/prompt-pack.json" \
  --frozen-prompt-pack "$ROOT/frozen-66.json" \
  --expected-teacher-manifest-sha256 "$QUALIFICATION_MANIFEST" \
  --checkpoint-dir "$ROOT/qualification/training-checkpoints" \
  --boundary-dir "$ROOT/qualification/training-boundaries" \
  --output-dir "$ROOT/qualification/adapter-unused" \
  --run-id "$RUN_ID-h100-qualification" \
  --layer 77 --preflight-only --preflight-steps 2 \
  --preflight-output "$SMOKE"

test "$(jq -er .sample_steps "$SMOKE")" -eq 2
PEAK_GPU_GIB=$(jq -er .peak_gpu_gib "$SMOKE")
python3 - "$PEAK_GPU_GIB" <<'PY'
import sys
if not float(sys.argv[1]) < 70:
    raise SystemExit("H100 qualification exceeded 70 GiB")
PY

RESUME_SID="teich_claude_agent-a3622521df8a9137d__dsa_full"
RESUMED="$ROOT/spike-resume/out/${RESUME_SID//\//__}.npz"
READY="$ROOT/qualification/H100_RESUME_READY.json"
"$REPO/aws/glm52-gpu/scripts/build_h100_resume_ready.py" \
  --run-id "$RUN_ID" \
  --campaign-identity-sha256 "$(jq -er .campaign_identity_sha256 "$DESCRIPTOR")" \
  --repo-tar-sha256 "$(jq -er .repo_tar_sha256 "$DESCRIPTOR")" \
  --qualification-cache-manifest-sha256 "$QUALIFICATION_MANIFEST" \
  --first-instance-id "$FIRST_INSTANCE" \
  --replacement-instance-id "$INSTANCE_ID" \
  --first-allocation-record-sha256 "$FIRST_ALLOCATION" \
  --replacement-allocation-record-sha256 "$ALLOCATION_RECORD" \
  --source-checkpoint-marker-sha256 "$SOURCE_CHECKPOINT" \
  --parity-report-sha256 "$(sha256sum "$ROOT/spike-out/parity-report.json" | awk '{print $1}')" \
  --training-smoke-sha256 "$(sha256sum "$SMOKE" | awk '{print $1}')" \
  --resumed-capture-sha256 "$(sha256sum "$RESUMED" | awk '{print $1}')" \
  --peak-gpu-gib "$PEAK_GPU_GIB" \
  --completed-at "$(date -u +%FT%TZ)" \
  --output "$READY"
aws s3 cp "$READY" "$PREFIX/H100_RESUME_READY.json" \
  --region "$REGION" --only-show-errors
