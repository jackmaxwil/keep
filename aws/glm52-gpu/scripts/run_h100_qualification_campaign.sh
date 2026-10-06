#!/usr/bin/env bash
# Originate and authenticate the AWS-coordinated H100 recovery qualification.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
DESCRIPTOR="${CAMPAIGN_DESCRIPTOR:?set CAMPAIGN_DESCRIPTOR}"
SKY_BIN="${SKY_BIN:?set SKY_BIN to the pinned 0.13.0 executable}"
SKYPILOT_CONFIG="${SKYPILOT_CONFIG:?set SKYPILOT_CONFIG}"
ROUTE_PYTHON="${KEEP_PYTHON:-python3}"
REGION=us-west-2
"$SCRIPT_DIR/assert_rnd_aws_account.sh" >/dev/null
read -r RUN_ID BUCKET <<EOF
$(python3 - "$DESCRIPTOR" "$REPO" <<'PY'
import importlib.util, json, pathlib, sys
path = pathlib.Path(sys.argv[1]); repo = pathlib.Path(sys.argv[2])
module_path = repo / "src/mlx_vq/quality/glm52_sky_campaign.py"
spec = importlib.util.spec_from_file_location("_glm52_qualify_submit", module_path)
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)
value = module.validate_sky_campaign_descriptor(json.loads(path.read_bytes()))
print(value["run_id"], value["bucket"])
PY
)
EOF
READY_KEY="campaigns/$RUN_ID/qualification/H100_RESUME_READY.json"
SOURCE_KEY="campaigns/$RUN_ID/qualification/SOURCE_NODE_READY.json"
TERMINATION_KEY="campaigns/$RUN_ID/qualification/QUALIFICATION_TERMINATION_REQUESTED.json"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

validate_ready() {
  aws s3 cp "s3://$BUCKET/$READY_KEY" "$WORK/H100_RESUME_READY.json" \
    --profile "$AWS_PROFILE" --region "$REGION" --only-show-errors
  aws s3 cp "s3://$BUCKET/$SOURCE_KEY" "$WORK/SOURCE_NODE_READY.json" \
    --profile "$AWS_PROFILE" --region "$REGION" --only-show-errors
  aws s3 cp "s3://$BUCKET/$TERMINATION_KEY" \
    "$WORK/QUALIFICATION_TERMINATION_REQUESTED.json" \
    --profile "$AWS_PROFILE" --region "$REGION" --only-show-errors
  python3 - "$WORK/H100_RESUME_READY.json" \
    "$WORK/SOURCE_NODE_READY.json" \
    "$WORK/QUALIFICATION_TERMINATION_REQUESTED.json" \
    "$DESCRIPTOR" "$REPO" <<'PY'
import importlib.util, json, pathlib, sys
ready_path, source_path, request_path, descriptor_path, repo = map(
    pathlib.Path, sys.argv[1:]
)
path = repo / "src/mlx_vq/quality/glm52_h100_qualification.py"
spec = importlib.util.spec_from_file_location("_glm52_qualify_ready", path)
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)
descriptor = json.loads(descriptor_path.read_bytes())
source = module.validate_h100_source_node_ready(
    json.loads(source_path.read_bytes()),
    expected_run_id=descriptor["run_id"],
    expected_campaign_identity_sha256=descriptor["campaign_identity_sha256"],
    expected_repo_tar_sha256=descriptor["repo_tar_sha256"],
    expected_descriptor_body_sha256=descriptor["descriptor_body_sha256"],
)
module.validate_h100_termination_requested(
    json.loads(request_path.read_bytes()),
    source_marker=source,
)
ready = module.validate_h100_resume_ready(json.loads(ready_path.read_bytes()))
expected = {
    "run_id": descriptor["run_id"],
    "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
    "repo_tar_sha256": descriptor["repo_tar_sha256"],
    "qualification_cache_manifest_sha256": descriptor["artifacts"][
        "qualification_cache_manifest_sha256"
    ],
    "first_instance_id": source["instance_id"],
    "first_allocation_record_sha256": source["allocation_record_sha256"],
    "source_checkpoint_marker_sha256": source["checkpoint_marker_sha256"],
}
if any(ready[field] != value for field, value in expected.items()):
    raise SystemExit("foreign or incomplete H100 qualification marker chain")
print(json.dumps(ready, sort_keys=True))
PY
}

load_source() {
  aws s3 cp "s3://$BUCKET/$SOURCE_KEY" "$WORK/SOURCE_NODE_READY.json" \
    --profile "$AWS_PROFILE" --region "$REGION" --only-show-errors
  python3 - "$WORK/SOURCE_NODE_READY.json" "$DESCRIPTOR" "$REPO" <<'PY'
import importlib.util, json, pathlib, sys
source_path, descriptor_path, repo = map(pathlib.Path, sys.argv[1:])
path = repo / "src/mlx_vq/quality/glm52_h100_qualification.py"
spec = importlib.util.spec_from_file_location("_glm52_qualify_source", path)
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)
descriptor = json.loads(descriptor_path.read_bytes())
source = module.validate_h100_source_node_ready(
    json.loads(source_path.read_bytes()),
    expected_run_id=descriptor["run_id"],
    expected_campaign_identity_sha256=descriptor["campaign_identity_sha256"],
    expected_repo_tar_sha256=descriptor["repo_tar_sha256"],
    expected_descriptor_body_sha256=descriptor["descriptor_body_sha256"],
)
print(source["instance_id"])
PY
}

validate_request() {
  aws s3 cp "s3://$BUCKET/$TERMINATION_KEY" \
    "$WORK/QUALIFICATION_TERMINATION_REQUESTED.json" \
    --profile "$AWS_PROFILE" --region "$REGION" --only-show-errors
  python3 - "$WORK/QUALIFICATION_TERMINATION_REQUESTED.json" \
    "$WORK/SOURCE_NODE_READY.json" "$REPO" <<'PY'
import importlib.util, json, pathlib, sys
request_path, source_path, repo = map(pathlib.Path, sys.argv[1:])
path = repo / "src/mlx_vq/quality/glm52_h100_qualification.py"
spec = importlib.util.spec_from_file_location("_glm52_qualify_request", path)
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)
source = module.validate_h100_source_node_ready(
    json.loads(source_path.read_bytes())
)
request = module.validate_h100_termination_requested(
    json.loads(request_path.read_bytes()),
    source_marker=source,
)
print(json.dumps(request, sort_keys=True))
PY
}

inspect_workers() {
  SOURCE_INSTANCE=$1
  REQUEST_PRESENT=$2
  SOURCE_STATE=$(aws ec2 describe-instances \
    --profile "$AWS_PROFILE" --region "$REGION" \
    --instance-ids "$SOURCE_INSTANCE" \
    --query 'Reservations[0].Instances[0].State.Name' --output text)
  case "$SOURCE_STATE" in
    pending|running|shutting-down|terminated|stopping|stopped)
      ;;
    *)
      echo "qualification source has unknown state: $SOURCE_STATE" >&2
      exit 2
      ;;
  esac
  ACTIVE=$(aws ec2 describe-instances \
    --profile "$AWS_PROFILE" --region "$REGION" \
    --filters \
      "Name=tag:campaign-run-id,Values=$RUN_ID" \
      "Name=tag:project,Values=keep-glm52" \
      "Name=tag:model,Values=glm-5.2" \
      "Name=instance-type,Values=p5.48xlarge" \
      "Name=instance-state-name,Values=pending,running" \
    --query 'Reservations[].Instances[].InstanceId' --output text)
  set -- $ACTIVE
  if [ "$#" -gt 1 ]; then
    echo "qualification observer found multiple active P5 workers" >&2
    exit 2
  fi
  if [ "$REQUEST_PRESENT" -eq 0 ]; then
    if [ "$SOURCE_STATE" != pending ] && [ "$SOURCE_STATE" != running ]; then
      echo "qualification source stopped before an authenticated request" >&2
      exit 2
    fi
    if [ "$#" -ne 1 ] || [ "$1" != "$SOURCE_INSTANCE" ]; then
      echo "active worker does not match source before termination request" >&2
      exit 2
    fi
  fi
  printf 'source=%s state=%s request=%s active=%s\n' \
    "$SOURCE_INSTANCE" "$SOURCE_STATE" "$REQUEST_PRESENT" "${ACTIVE:-none}"
}

if aws s3api head-object --bucket "$BUCKET" --key "$READY_KEY" \
  --profile "$AWS_PROFILE" --region "$REGION" >/dev/null 2>&1; then
  validate_ready
  exit 0
fi

DESCRIPTOR_DIR=$(CDPATH= cd -- "$(dirname -- "$DESCRIPTOR")" && pwd)
case "$(basename -- "$DESCRIPTOR_DIR")" in
  cache-seed|post-seed|production)
    QUALIFICATION_WORKSPACE=$(CDPATH= cd -- "$DESCRIPTOR_DIR/.." && pwd)
    ;;
  *)
    QUALIFICATION_WORKSPACE="$DESCRIPTOR_DIR"
    ;;
esac

# This is the only executable launch route in the Task 13 H100 driver.  It
# prepares and acquires through the guarded submitter; the deployed
# qualification watchdog alone owns the exact source-worker termination.
"$ROUTE_PYTHON" "$SCRIPT_DIR/run_h100_qualification_route.py" \
  --descriptor "$DESCRIPTOR" \
  --config "$SKYPILOT_CONFIG" \
  --workspace "$QUALIFICATION_WORKSPACE" \
  --sky-bin "$SKY_BIN" \
  --submitter "$SCRIPT_DIR/submit_sky_campaign.sh" \
  --work-root "$WORK/guarded-route" \
  --profile "$AWS_PROFILE"

DEADLINE=$(( $(date -u +%s) + 86400 ))
while [ "$(date -u +%s)" -lt "$DEADLINE" ]; do
  if aws s3api head-object --bucket "$BUCKET" --key "$READY_KEY" \
    --profile "$AWS_PROFILE" --region "$REGION" >/dev/null 2>&1; then
    validate_ready
    exit 0
  fi
  SOURCE_PRESENT=0
  if aws s3api head-object --bucket "$BUCKET" --key "$SOURCE_KEY" \
    --profile "$AWS_PROFILE" --region "$REGION" >/dev/null 2>&1; then
    SOURCE_PRESENT=1
    SOURCE_INSTANCE=$(load_source)
    REQUEST_PRESENT=0
    if aws s3api head-object --bucket "$BUCKET" --key "$TERMINATION_KEY" \
      --profile "$AWS_PROFILE" --region "$REGION" >/dev/null 2>&1; then
      validate_request
      REQUEST_PRESENT=1
    fi
    inspect_workers "$SOURCE_INSTANCE" "$REQUEST_PRESENT"
  fi
  if [ "$SOURCE_PRESENT" -eq 0 ] && \
    aws s3api head-object --bucket "$BUCKET" --key "$TERMINATION_KEY" \
      --profile "$AWS_PROFILE" --region "$REGION" >/dev/null 2>&1; then
    echo "termination request exists without source-node authority" >&2
    exit 2
  fi
  sleep 30
done
echo "H100 qualification did not complete within the approved 24-hour envelope" >&2
exit 2
