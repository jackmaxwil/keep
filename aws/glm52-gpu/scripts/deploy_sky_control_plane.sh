#!/usr/bin/env bash
# Deploy retained core/observe mode, or prepare and explicitly execute active mode.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
MODE="${1:-}"
case "$MODE" in
  core|enable|prepare-foundation|inspect-foundation|execute-foundation|\
prepare-active|inspect-active|execute-active) ;;
  *)
    echo "usage: $0 {core|enable|prepare-foundation|inspect-foundation|execute-foundation|prepare-active|inspect-active|execute-active}" >&2
    exit 64
    ;;
esac

: "${AWS_PROFILE:?set AWS_PROFILE to the named R&D profile}"
REGION=us-west-2
ACCOUNT_ID=246813579024
STACK="${STACK:-keep-glm52-gpu}"
MODEL_BUCKET="${MODEL_BUCKET:-keep-glm52-models-246813579024-us-west-2}"
TEMPLATE_UPLOAD_BUCKET="${TEMPLATE_UPLOAD_BUCKET:-$MODEL_BUCKET}"
TEMPLATE="$REPO/aws/glm52-gpu/cfn/gpu-teacher-stack.yaml"
FOUNDATION_REVISION=versioned-code-v1
WORKER_V2_FOUNDATION_REVISION=versioned-code-v1
WORKER_V2_DISABLED_STATE=$(
  printf 'false\t%s\tdisabled\tdisabled\tdisabled\tdisabled\tdisabled\tdisabled\tdisabled\tdisabled' \
    "$WORKER_V2_FOUNDATION_REVISION"
)
WORKER_V2_LEGACY_ABSENT_STATE=\
$'None\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone\tNone'
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

if ! template_metadata=$(python3 - "$TEMPLATE" <<'PY'
import hashlib
import sys
from pathlib import Path

expected_local_sha256 = (
    "b4c0bd8c725c3c4db951fccadd32707a0b2f8064c2519d264abfbc2f8d2924a4"
)
expected_projection_sha256 = (
    "9e57301564efb325f16caecdb4d8a5ff372a7a306e823e3d4ac87aa502c7ad69"
)
try:
    template_bytes = Path(sys.argv[1]).read_bytes()
    template_text = template_bytes.decode("utf-8")
except (OSError, UnicodeDecodeError) as exc:
    raise SystemExit(
        "frozen CloudFormation template does not match exact UTF-8 upload"
    ) from exc
non_ascii = {
    character for character in template_text if not character.isascii()
}
projection = template_text.replace("\N{EM DASH}", "?").encode()
local_sha256 = hashlib.sha256(template_bytes).hexdigest()
projection_sha256 = hashlib.sha256(projection).hexdigest()
if (
    local_sha256 != expected_local_sha256
    or template_text.count("\N{EM DASH}") != 10
    or non_ascii != {"\N{EM DASH}"}
    or projection_sha256 != expected_projection_sha256
):
    raise SystemExit(
        "frozen CloudFormation template does not match exact UTF-8 upload"
    )
template_md5 = hashlib.md5(
    template_bytes,
    usedforsecurity=False,
).hexdigest()
print(
    "\t".join(
        (
            local_sha256,
            projection_sha256,
            f"{template_md5}.template",
        )
    )
)
PY
); then
  exit 65
fi
IFS=$'\t' read -r \
  FROZEN_TEMPLATE_SHA256 \
  FROZEN_TEMPLATE_PROJECTION_SHA256 \
  TEMPLATE_UPLOAD_KEY \
  <<<"$template_metadata"

"$SCRIPT_DIR/assert_rnd_aws_account.sh" >/dev/null

require_sha256() {
  local name=$1
  local value=$2
  if [[ ! "$value" =~ ^[0-9a-f]{64}$ ]]; then
    echo "$name must be one canonical lowercase SHA-256" >&2
    exit 65
  fi
}

require_unused_handoff_target() {
  local handoff_file=$1
  if [ -z "$handoff_file" ] || [[ "$handoff_file" != /* ]] || \
    [ -e "$handoff_file" ] || [ -L "$handoff_file" ] || \
    [ ! -d "$(dirname -- "$handoff_file")" ]; then
    echo "change-set handoff file must be an unused absolute non-symlink path with an existing parent" >&2
    exit 65
  fi
}

parse_generated_change_set_arn() {
  local deploy_output=$1
  python3 - "$deploy_output" "$REGION" "$ACCOUNT_ID" <<'PY'
import re
import sys
from pathlib import Path

path, region, account_id = sys.argv[1:]
prefix = "aws cloudformation describe-change-set --change-set-name "
uuid = (
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}"
)
arn = (
    rf"arn:aws:cloudformation:{re.escape(region)}:"
    rf"{re.escape(account_id)}:changeSet/"
    rf"awscli-cloudformation-package-deploy-[0-9]+/{uuid}"
)
pattern = re.compile(re.escape(prefix) + f"({arn})")
candidate_lines = [
    line
    for line in Path(path).read_text().splitlines()
    if line.startswith(prefix)
]
if len(candidate_lines) != 1:
    raise SystemExit(
        "deploy output must contain exactly one generated change-set ARN command"
    )
match = pattern.fullmatch(candidate_lines[0])
if match is None:
    raise SystemExit(
        "deploy output must contain exactly one generated change-set ARN command"
    )
print(match.group(1))
PY
}

current_stack_identity() {
  local stack_identity
  stack_identity=$(
    aws cloudformation describe-stacks \
      --stack-name "$STACK" \
      --query "Stacks[0].[StackId,StackStatus]" \
      --output text \
      --profile "$AWS_PROFILE" --region "$REGION"
  )
  python3 - "$stack_identity" "$REGION" "$ACCOUNT_ID" "$STACK" <<'PY'
import re
import sys

stack_identity, region, account_id, stack_name = sys.argv[1:]
parts = stack_identity.split("\t")
if len(parts) != 2:
    raise SystemExit(
        "current stack is not the exact existing UPDATE_COMPLETE stack"
    )
stack_id, stack_status = parts
uuid = (
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}"
)
pattern = re.compile(
    rf"arn:aws:cloudformation:{re.escape(region)}:"
    rf"{re.escape(account_id)}:stack/{re.escape(stack_name)}/{uuid}"
)
if pattern.fullmatch(stack_id) is None or stack_status != "UPDATE_COMPLETE":
    raise SystemExit(
        "current stack is not the exact existing UPDATE_COMPLETE stack"
    )
print(f"{stack_id}\t{stack_status}")
PY
}

verify_change_set_identity() {
  local change_set_arn=$1
  local expected_stack_id=$2
  local identity_file="$WORK/change-set-identity.json"
  aws cloudformation describe-change-set \
    --stack-name "$expected_stack_id" \
    --change-set-name "$change_set_arn" \
    --query \
      "{ChangeSetId:ChangeSetId,StackId:StackId,StackName:StackName,Status:Status,ExecutionStatus:ExecutionStatus}" \
    --output json \
    --profile "$AWS_PROFILE" --region "$REGION" \
    >"$identity_file"
  python3 - \
    "$identity_file" \
    "$change_set_arn" \
    "$expected_stack_id" \
    "$STACK" <<'PY'
import json
import sys
from pathlib import Path

path, expected_id, expected_stack_id, expected_stack_name = sys.argv[1:]
try:
    value = json.loads(Path(path).read_text())
except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit("change-set identity response is invalid") from exc
expected_fields = {
    "ChangeSetId",
    "StackId",
    "StackName",
    "Status",
    "ExecutionStatus",
}
if (
    not isinstance(value, dict)
    or set(value) != expected_fields
    or value.get("ChangeSetId") != expected_id
    or value.get("StackId") != expected_stack_id
    or value.get("StackName") != expected_stack_name
    or value.get("Status") != "CREATE_COMPLETE"
    or value.get("ExecutionStatus") != "AVAILABLE"
):
    raise SystemExit(
        "change-set identity does not match captured ARN, stack, or available status"
    )
PY
}

verify_change_set_template() {
  local change_set_arn=$1
  local expected_template_sha256=$2
  local expected_stack_id=$3
  local response_file="$WORK/change-set-template.json"
  aws cloudformation get-template \
    --stack-name "$expected_stack_id" \
    --change-set-name "$change_set_arn" \
    --template-stage Original \
    --output json \
    --profile "$AWS_PROFILE" --region "$REGION" \
    >"$response_file"
  python3 - \
    "$response_file" \
    "$TEMPLATE" \
    "$expected_template_sha256" \
    "$FROZEN_TEMPLATE_PROJECTION_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    response_path,
    template_path,
    expected_sha256,
    expected_projection_sha256,
) = sys.argv[1:]
local_bytes = Path(template_path).read_bytes()
local_sha256 = hashlib.sha256(local_bytes).hexdigest()
if local_sha256 != expected_sha256:
    raise SystemExit(
        "local template no longer matches pinned template SHA-256"
    )
try:
    response = json.loads(Path(response_path).read_text())
except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit("change-set template response is invalid") from exc
if not isinstance(response, dict) or not isinstance(
    response.get("TemplateBody"), str
):
    raise SystemExit("change-set template response is invalid")
try:
    local_text = local_bytes.decode("utf-8")
except UnicodeDecodeError as exc:
    raise SystemExit(
        "local template no longer matches exact UTF-8 upload"
    ) from exc
projected_bytes = local_text.replace("\N{EM DASH}", "?").encode()
server_bytes = response["TemplateBody"].encode("utf-8")
if (
    server_bytes != projected_bytes
    or hashlib.sha256(server_bytes).hexdigest()
    != expected_projection_sha256
):
    raise SystemExit(
        "change-set template does not match frozen server projection"
    )
PY
}

verify_template_upload_object() {
  local expected_template_sha256=$1
  local upload_bucket=$2
  local upload_key=$3
  local head_file="$WORK/template-upload-head.json"
  local get_file="$WORK/template-upload-get.json"
  local downloaded_template="$WORK/uploaded-template-body"
  aws s3api head-object \
    --bucket "$upload_bucket" \
    --key "$upload_key" \
    --output json \
    --profile "$AWS_PROFILE" --region "$REGION" \
    >"$head_file"
  aws s3api get-object \
    --bucket "$upload_bucket" \
    --key "$upload_key" \
    --output json \
    --profile "$AWS_PROFILE" --region "$REGION" \
    "$downloaded_template" \
    >"$get_file"
  python3 - \
    "$TEMPLATE" \
    "$downloaded_template" \
    "$head_file" \
    "$get_file" \
    "$expected_template_sha256" \
    "$upload_key" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    local_path,
    downloaded_path,
    head_path,
    get_path,
    expected_sha256,
    upload_key,
) = sys.argv[1:]
try:
    local_bytes = Path(local_path).read_bytes()
    downloaded_bytes = Path(downloaded_path).read_bytes()
    head = json.loads(Path(head_path).read_text())
    get = json.loads(Path(get_path).read_text())
except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit(
        "uploaded CloudFormation template object response is invalid"
    ) from exc
local_sha256 = hashlib.sha256(local_bytes).hexdigest()
local_md5 = hashlib.md5(
    local_bytes,
    usedforsecurity=False,
).hexdigest()
expected_etag = f'"{local_md5}"'
if (
    local_sha256 != expected_sha256
    or upload_key != f"{local_md5}.template"
    or downloaded_bytes != local_bytes
    or hashlib.sha256(downloaded_bytes).hexdigest() != expected_sha256
    or not isinstance(head, dict)
    or head.get("ContentLength") != len(local_bytes)
    or head.get("ETag") != expected_etag
    or not isinstance(get, dict)
    or get.get("ContentLength") != len(local_bytes)
    or get.get("ETag") != expected_etag
):
    raise SystemExit(
        "uploaded CloudFormation template object does not match frozen local template"
    )
PY
}

persist_change_set_handoff() {
  local handoff_file=$1
  local kind=$2
  local change_set_arn=$3
  local template_sha256=$4
  local stack_id=$5
  local stack_status=$6
  python3 - \
    "$handoff_file" \
    "$kind" \
    "$change_set_arn" \
    "$template_sha256" \
    "$stack_id" \
    "$stack_status" \
    "$TEMPLATE_UPLOAD_BUCKET" \
    "$TEMPLATE_UPLOAD_KEY" \
    "$FROZEN_TEMPLATE_PROJECTION_SHA256" \
    "$ACCOUNT_ID" \
    "$REGION" \
    "$STACK" <<'PY'
import json
import os
import sys

(
    path,
    kind,
    change_set_arn,
    template_sha256,
    stack_id,
    stack_status,
    template_upload_bucket,
    template_upload_key,
    template_projection_sha256,
    account_id,
    region,
    stack_name,
) = sys.argv[1:]
value = {
    "account_id": account_id,
    "change_set_arn": change_set_arn,
    "change_set_type": "UPDATE",
    "kind": kind,
    "region": region,
    "schema_version": 2,
    "stack_id": stack_id,
    "stack_name": stack_name,
    "stack_status": stack_status,
    "template_projection_sha256": template_projection_sha256,
    "template_sha256": template_sha256,
    "template_upload_bucket": template_upload_bucket,
    "template_upload_key": template_upload_key,
}
raw = (
    json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
).encode()
try:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
except FileExistsError as exc:
    raise SystemExit(
        "could not atomically persist unused change-set handoff file"
    ) from exc
PY
}

load_change_set_handoff() {
  local handoff_file=$1
  local kind=$2
  if [ -z "$handoff_file" ] || {
    [ ! -e "$handoff_file" ] && [ ! -L "$handoff_file" ]
  }; then
    if [ "$kind" = active ]; then
      echo "set MUST_START_CHANGE_SET_HANDOFF_FILE" >&2
    else
      echo "set MUST_START_FOUNDATION_CHANGE_SET_HANDOFF_FILE" >&2
    fi
    exit 65
  fi
  if [[ "$handoff_file" != /* ]] || [ ! -f "$handoff_file" ] || \
    [ -L "$handoff_file" ]; then
    echo "change-set handoff is invalid" >&2
    exit 65
  fi
  local handoff_values
  if ! handoff_values=$(
    python3 - \
      "$handoff_file" \
      "$kind" \
      "$ACCOUNT_ID" \
      "$REGION" \
      "$STACK" \
      "$TEMPLATE_UPLOAD_BUCKET" \
      "$TEMPLATE_UPLOAD_KEY" \
      "$FROZEN_TEMPLATE_PROJECTION_SHA256" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

(
    path,
    expected_kind,
    account_id,
    region,
    stack_name,
    template_upload_bucket,
    template_upload_key,
    template_projection_sha256,
) = sys.argv[1:]
try:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit("change-set handoff is invalid") from exc
expected_fields = {
    "account_id",
    "change_set_arn",
    "change_set_type",
    "kind",
    "region",
    "schema_version",
    "stack_id",
    "stack_name",
    "stack_status",
    "template_projection_sha256",
    "template_sha256",
    "template_upload_bucket",
    "template_upload_key",
}
canonical = (
    json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
).encode()
uuid = (
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}"
)
change_set_pattern = re.compile(
    rf"arn:aws:cloudformation:{re.escape(region)}:"
    rf"{re.escape(account_id)}:changeSet/"
    rf"awscli-cloudformation-package-deploy-[0-9]+/{uuid}"
)
stack_pattern = re.compile(
    rf"arn:aws:cloudformation:{re.escape(region)}:"
    rf"{re.escape(account_id)}:stack/{re.escape(stack_name)}/{uuid}"
)
if (
    not isinstance(value, dict)
    or set(value) != expected_fields
    or raw != canonical
    or type(value.get("schema_version")) is not int
    or value.get("schema_version") != 2
    or value.get("kind") != expected_kind
    or value.get("account_id") != account_id
    or value.get("region") != region
    or value.get("stack_name") != stack_name
    or value.get("stack_status") != "UPDATE_COMPLETE"
    or value.get("template_upload_bucket") != template_upload_bucket
    or value.get("template_upload_key") != template_upload_key
    or value.get("template_projection_sha256")
    != template_projection_sha256
    or value.get("change_set_type") != "UPDATE"
    or not isinstance(value.get("change_set_arn"), str)
    or change_set_pattern.fullmatch(value["change_set_arn"]) is None
    or not isinstance(value.get("stack_id"), str)
    or stack_pattern.fullmatch(value["stack_id"]) is None
    or not isinstance(value.get("template_sha256"), str)
    or re.fullmatch(r"[0-9a-f]{64}", value["template_sha256"]) is None
    or not isinstance(value.get("template_projection_sha256"), str)
    or re.fullmatch(
        r"[0-9a-f]{64}",
        value["template_projection_sha256"],
    )
    is None
):
    raise SystemExit("change-set handoff is invalid")
print(
    "\t".join(
        (
            value["change_set_arn"],
            value["template_sha256"],
            value["stack_id"],
            value["stack_status"],
            value["change_set_type"],
            value["template_upload_bucket"],
            value["template_upload_key"],
            value["template_projection_sha256"],
        )
    )
)
PY
  ); then
    exit 65
  fi
  IFS=$'\t' read -r \
    CHANGE_SET_ARN \
    PINNED_TEMPLATE_SHA256 \
    PREPARED_STACK_ID \
    PREPARED_STACK_STATUS \
    CHANGE_SET_TYPE \
    PREPARED_TEMPLATE_UPLOAD_BUCKET \
    PREPARED_TEMPLATE_UPLOAD_KEY \
    PREPARED_TEMPLATE_PROJECTION_SHA256 \
    <<<"$handoff_values"
  local local_template_sha256
  local_template_sha256=$(shasum -a 256 "$TEMPLATE" | awk '{print $1}')
  if [ "$local_template_sha256" != "$PINNED_TEMPLATE_SHA256" ]; then
    echo "local template no longer matches pinned template SHA-256" >&2
    exit 65
  fi
}

verify_prepared_stack_and_change_set() {
  local current_identity
  current_identity=$(current_stack_identity)
  local current_id
  local current_status
  IFS=$'\t' read -r current_id current_status <<<"$current_identity"
  if [ "$current_id" != "$PREPARED_STACK_ID" ] || \
    [ "$current_status" != "$PREPARED_STACK_STATUS" ]; then
    echo "stack identity no longer matches prepared change set" >&2
    exit 65
  fi
  verify_change_set_identity "$CHANGE_SET_ARN" "$PREPARED_STACK_ID"
  verify_change_set_template \
    "$CHANGE_SET_ARN" \
    "$PINNED_TEMPLATE_SHA256" \
    "$PREPARED_STACK_ID"
}

require_version_id() {
  local name=$1
  local value=$2
  if [ "$value" = disabled ] || [ "$value" = null ] || \
    [ "$value" = None ] || [ -z "$value" ] || \
    [ "${#value}" -gt 1024 ] || \
    [[ ! "$value" =~ ^[A-Za-z0-9._~+/=-]+$ ]] || \
    [[ "$value" == .. || "$value" == ../* || "$value" == */.. || \
      "$value" == */../* ]]; then
    echo "$name must be one exact S3 object VersionId" >&2
    exit 65
  fi
}

prepare_code_artifact_identity() {
  : "${MUST_START_CANCEL_CODE_ZIP:?set MUST_START_CANCEL_CODE_ZIP}"
  : "${MUST_START_CANCEL_CODE_SHA256:?set MUST_START_CANCEL_CODE_SHA256}"
  if [ -z "${MUST_START_CANCEL_CODE_VERSION_ID:-}" ]; then
    echo "set MUST_START_CANCEL_CODE_VERSION_ID" >&2
    exit 65
  fi
  require_sha256 \
    MUST_START_CANCEL_CODE_SHA256 \
    "$MUST_START_CANCEL_CODE_SHA256"
  require_version_id \
    MUST_START_CANCEL_CODE_VERSION_ID \
    "$MUST_START_CANCEL_CODE_VERSION_ID"
  if [ ! -f "$MUST_START_CANCEL_CODE_ZIP" ]; then
    echo "must-start Lambda ZIP is missing" >&2
    exit 65
  fi
  local local_sha
  local_sha=$(shasum -a 256 "$MUST_START_CANCEL_CODE_ZIP" | awk '{print $1}')
  if [ "$local_sha" != "$MUST_START_CANCEL_CODE_SHA256" ]; then
    echo "local code ZIP SHA-256 mismatch" >&2
    exit 65
  fi
  MUST_START_CODE_KEY=\
"lambda/sky-must-start-cancel/${MUST_START_CANCEL_CODE_SHA256}.zip"
}

verify_code_artifact() {
  local head_version
  head_version=$(
    aws s3api head-object \
      --bucket "$MODEL_BUCKET" \
      --key "$MUST_START_CODE_KEY" \
      --version-id "$MUST_START_CANCEL_CODE_VERSION_ID" \
      --query VersionId --output text \
      --profile "$AWS_PROFILE" --region "$REGION"
  )
  if [ "$head_version" != "$MUST_START_CANCEL_CODE_VERSION_ID" ]; then
    echo "S3 code object VersionId mismatch" >&2
    exit 65
  fi
  local downloaded="$WORK/sky-must-start-cancel.zip"
  aws s3api get-object \
    --bucket "$MODEL_BUCKET" \
    --key "$MUST_START_CODE_KEY" \
    --version-id "$MUST_START_CANCEL_CODE_VERSION_ID" \
    --profile "$AWS_PROFILE" --region "$REGION" \
    "$downloaded" >/dev/null
  local remote_sha
  remote_sha=$(shasum -a 256 "$downloaded" | awk '{print $1}')
  if [ "$remote_sha" != "$MUST_START_CANCEL_CODE_SHA256" ]; then
    echo "S3 code object SHA-256 mismatch" >&2
    exit 65
  fi
}

require_must_start_inputs() {
  : "${MUST_START_SUBMISSION_KEY:?set MUST_START_SUBMISSION_KEY}"
  : "${MUST_START_SUBMISSION_BODY_SHA256:?set MUST_START_SUBMISSION_BODY_SHA256}"
  : "${MUST_START_CONTROLLER_BASELINE_BODY_SHA256:?set MUST_START_CONTROLLER_BASELINE_BODY_SHA256}"
  : "${MUST_START_TARGET_JOB_ID:?set exact MUST_START_TARGET_JOB_ID}"
  : "${MUST_START_MANAGED_MODE:?set MUST_START_MANAGED_MODE}"
  : "${MUST_START_JOB_NAME:?set MUST_START_JOB_NAME}"
  : "${MUST_START_BY:?set MUST_START_BY}"
  : "${MUST_START_DESCRIPTOR_FILE_SHA256:?set MUST_START_DESCRIPTOR_FILE_SHA256}"
  : "${MUST_START_SUBMISSION_SUBMITTED_AT:?set MUST_START_SUBMISSION_SUBMITTED_AT}"
  : "${MUST_START_CONTROLLER_INSTANCE_ID:?set MUST_START_CONTROLLER_INSTANCE_ID}"
  : "${MUST_START_CONTROLLER_INSTANCE_TYPE:?set MUST_START_CONTROLLER_INSTANCE_TYPE}"
  : "${MUST_START_CONTROLLER_PROFILE_ARN:?set MUST_START_CONTROLLER_PROFILE_ARN}"
  : "${MUST_START_CONTROLLER_CLUSTER_NAME:?set MUST_START_CONTROLLER_CLUSTER_NAME}"
  require_sha256 \
    MUST_START_SUBMISSION_BODY_SHA256 \
    "$MUST_START_SUBMISSION_BODY_SHA256"
  require_sha256 \
    MUST_START_CONTROLLER_BASELINE_BODY_SHA256 \
    "$MUST_START_CONTROLLER_BASELINE_BODY_SHA256"
  require_sha256 \
    MUST_START_DESCRIPTOR_FILE_SHA256 \
    "$MUST_START_DESCRIPTOR_FILE_SHA256"
  if [[ ! "$MUST_START_TARGET_JOB_ID" =~ ^[1-9][0-9]*$ ]]; then
    echo "MUST_START_TARGET_JOB_ID must be a positive integer" >&2
    exit 65
  fi
  case "$MUST_START_MANAGED_MODE" in
    production)
      expected_job_name=$RUN_ID
      ;;
    qualification)
      expected_job_name="${RUN_ID}-qualification"
      ;;
    cache-seed)
      expected_job_name="${RUN_ID}-cache-seed"
      ;;
    *)
      echo "MUST_START_MANAGED_MODE is invalid" >&2
      exit 65
      ;;
  esac
  if [ "$MUST_START_JOB_NAME" != "$expected_job_name" ]; then
    echo "MUST_START_JOB_NAME does not match run and mode" >&2
    exit 65
  fi
  expected_submission_key=\
"campaigns/${RUN_ID}/monitor/submission-locks/${MUST_START_DESCRIPTOR_FILE_SHA256}-${MUST_START_MANAGED_MODE}.json"
  if [ "$MUST_START_SUBMISSION_KEY" != "$expected_submission_key" ]; then
    echo "submission key does not bind the exact run and descriptor" >&2
    exit 65
  fi
  case "$DESCRIPTOR_KEY" in
    campaigns/"$RUN_ID"/submissions/*)
      MUST_START_DESCRIPTOR_RELATIVE_KEY=${DESCRIPTOR_KEY#"campaigns/${RUN_ID}/"}
      ;;
    *)
      echo "descriptor key has a mixed or foreign run prefix" >&2
      exit 65
      ;;
  esac
  PRIMARY_WAKE_AT=$(
    python3 "$SCRIPT_DIR/derive_must_start_schedule_at.py" \
      "$MUST_START_BY"
  )
  prepare_code_artifact_identity
}

append_must_start_parameters() {
  local observe=$1
  local cancel=$2
  PARAMETERS+=(
    "EnableSkyMustStartObserve=$observe"
    "EnableSkyMustStartCancel=$cancel"
    "SkyMustStartCancelCodeSha256=$MUST_START_CANCEL_CODE_SHA256"
    "SkyMustStartCancelCodeVersionId=$MUST_START_CANCEL_CODE_VERSION_ID"
    "SkyMustStartDescriptorRelativeKey=$MUST_START_DESCRIPTOR_RELATIVE_KEY"
    "SkyMustStartSubmissionBodySha256=$MUST_START_SUBMISSION_BODY_SHA256"
    "SkyMustStartControllerBaselineBodySha256=$MUST_START_CONTROLLER_BASELINE_BODY_SHA256"
    "SkyMustStartTargetJobId=$MUST_START_TARGET_JOB_ID"
    "SkyMustStartManagedMode=$MUST_START_MANAGED_MODE"
    "SkyMustStartJobName=$MUST_START_JOB_NAME"
    "SkyMustStartBy=$MUST_START_BY"
    "SkyMustStartPrimaryWakeAt=$PRIMARY_WAKE_AT"
    "SkyMustStartWorkspace=${MUST_START_WORKSPACE:-default}"
    "SkyMustStartDescriptorFileSha256=$MUST_START_DESCRIPTOR_FILE_SHA256"
    "SkyMustStartSubmissionSubmittedAt=$MUST_START_SUBMISSION_SUBMITTED_AT"
    "SkyMustStartControllerInstanceId=$MUST_START_CONTROLLER_INSTANCE_ID"
    "SkyMustStartControllerInstanceType=$MUST_START_CONTROLLER_INSTANCE_TYPE"
    "SkyMustStartControllerProfileArn=$MUST_START_CONTROLLER_PROFILE_ARN"
    "SkyMustStartControllerClusterName=$MUST_START_CONTROLLER_CLUSTER_NAME"
    "SkyMustStartStartingGraceSeconds=${MUST_START_STARTING_GRACE_SECONDS:-20}"
  )
}

append_worker_v2_disabled_parameters() {
  PARAMETERS+=(
    EnableSkyWorkerStartV2Coordinator=false
    SkyWorkerStartV2FoundationRevision=versioned-code-v1
    SkyWorkerStartV2CodeSha256=disabled
    SkyWorkerStartV2CodeVersionId=disabled
    SkyWorkerStartV2DescriptorRelativeKey=disabled
    SkyWorkerStartV2DescriptorFileSha256=disabled
    SkyWorkerStartV2IntentFileSha256=disabled
    SkyWorkerStartV2IntentBodySha256=disabled
    SkyWorkerStartV2ControllerInstanceId=disabled
    SkyWorkerStartV2ControllerClusterName=disabled
  )
}

require_new_stack_for_core() {
  local stack_probe="$WORK/core-stack-probe.txt"
  if aws cloudformation describe-stacks \
    --stack-name "$STACK" \
    --query "Stacks[0].StackId" \
    --output text \
    --profile "$AWS_PROFILE" --region "$REGION" \
    >"$stack_probe" 2>&1; then
    echo "core is creation-only; use the guarded foundation and enable phases for an existing stack" >&2
    exit 65
  fi
  if grep -Fq "ValidationError" "$stack_probe" && \
    grep -Fq "does not exist" "$stack_probe"; then
    return
  fi
  echo "could not strongly prove that the core stack does not exist" >&2
  exit 65
}

verify_current_foundation_installed() {
  local consumer_mode=$1
  local current_state
  if ! current_state=$(
    aws cloudformation describe-stacks \
      --stack-name "$STACK" \
      --query \
        "Stacks[0].[Parameters[?ParameterKey=='EnableSkyPilotSupport'].ParameterValue | [0],Parameters[?ParameterKey=='EnableSkyMustStartObserve'].ParameterValue | [0],Parameters[?ParameterKey=='EnableSkyMustStartCancel'].ParameterValue | [0],Parameters[?ParameterKey=='SkyCampaignRunId'].ParameterValue | [0],Parameters[?ParameterKey=='SkyCampaignDescriptorKey'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWatchdogCodeS3Key'].ParameterValue | [0],Parameters[?ParameterKey=='SkyMustStartFoundationRevision'].ParameterValue | [0],Parameters[?ParameterKey=='EnableSkyWorkerStartV2Coordinator'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2FoundationRevision'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2CodeSha256'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2CodeVersionId'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2DescriptorRelativeKey'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2DescriptorFileSha256'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2IntentFileSha256'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2IntentBodySha256'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2ControllerInstanceId'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2ControllerClusterName'].ParameterValue | [0]]" \
      --output text \
      --profile "$AWS_PROFILE" --region "$REGION"
  ); then
    echo "reviewed must-start foundation is not installed" >&2
    exit 65
  fi
  local expected_observe_disabled
  local expected_observe_enabled
  expected_observe_disabled=$(
    printf 'true\tfalse\tfalse\t%s\t%s\t%s\t%s\t%s' \
      "$RUN_ID" \
      "$DESCRIPTOR_KEY" \
      "$WATCHDOG_CODE_S3_KEY" \
      "$FOUNDATION_REVISION" \
      "$WORKER_V2_DISABLED_STATE"
  )
  expected_observe_enabled=$(
    printf 'true\ttrue\tfalse\t%s\t%s\t%s\t%s\t%s' \
      "$RUN_ID" \
      "$DESCRIPTOR_KEY" \
      "$WATCHDOG_CODE_S3_KEY" \
      "$FOUNDATION_REVISION" \
      "$WORKER_V2_DISABLED_STATE"
  )
  local expected_active_enabled
  expected_active_enabled=$(
    printf 'true\ttrue\ttrue\t%s\t%s\t%s\t%s\t%s' \
      "$RUN_ID" \
      "$DESCRIPTOR_KEY" \
      "$WATCHDOG_CODE_S3_KEY" \
      "$FOUNDATION_REVISION" \
      "$WORKER_V2_DISABLED_STATE"
  )
  local expected_post_core
  expected_post_core=$(
    printf 'false\tfalse\tfalse\tdisabled\t\tdisabled/not-deployed-sky-watchdog.zip\t%s\t%s' \
      "$FOUNDATION_REVISION" \
      "$WORKER_V2_DISABLED_STATE"
  )
  case "$consumer_mode" in
    enable-base)
      if [ "$current_state" != "$expected_observe_disabled" ] && \
        [ "$current_state" != "$expected_post_core" ]; then
        echo "reviewed must-start foundation is not installed for the exact prior Sky authority" >&2
        exit 65
      fi
      ;;
    enable-observe)
      if [ "$current_state" != "$expected_observe_disabled" ] && \
        [ "$current_state" != "$expected_observe_enabled" ]; then
        echo "reviewed must-start foundation is not installed for the exact prior Sky authority" >&2
        exit 65
      fi
      ;;
    active)
      if [ "$current_state" != "$expected_observe_enabled" ] && \
        [ "$current_state" != "$expected_active_enabled" ]; then
        echo "reviewed must-start foundation is not installed for the exact observed Sky authority" >&2
        exit 65
      fi
      ;;
    *)
      echo "internal error: invalid foundation consumer mode" >&2
      exit 70
      ;;
  esac
  local bucket_versioning_status
  if ! bucket_versioning_status=$(
    aws s3api get-bucket-versioning \
      --bucket "$MODEL_BUCKET" \
      --query Status --output text \
      --profile "$AWS_PROFILE" --region "$REGION"
  ); then
    echo "could not prove model bucket versioning is Enabled" >&2
    exit 65
  fi
  if [ "$bucket_versioning_status" != Enabled ]; then
    echo "model bucket versioning must be Enabled before coordinator code is consumed" >&2
    exit 65
  fi
}

verify_no_stack_resources_with_prefix() {
  local prefix=$1
  local failure_message=$2
  local resources
  if ! resources=$(
    aws cloudformation list-stack-resources \
      --stack-name "$STACK" \
      --query \
        "StackResourceSummaries[?starts_with(LogicalResourceId, '$prefix')].LogicalResourceId" \
      --output text \
      --profile "$AWS_PROFILE" --region "$REGION"
  ); then
    echo "could not prove $failure_message" >&2
    exit 65
  fi
  if [ -z "$resources" ] || [ "$resources" = None ]; then
    return
  fi
  echo "$failure_message" >&2
  exit 65
}

verify_foundation_preserves_current_sky_authority() {
  local current_state
  current_state=$(
    aws cloudformation describe-stacks \
      --stack-name "$STACK" \
      --query \
        "Stacks[0].[Parameters[?ParameterKey=='EnableSkyPilotSupport'].ParameterValue | [0],Parameters[?ParameterKey=='EnableSkyMustStartObserve'].ParameterValue | [0],Parameters[?ParameterKey=='EnableSkyMustStartCancel'].ParameterValue | [0],Parameters[?ParameterKey=='SkyCampaignRunId'].ParameterValue | [0],Parameters[?ParameterKey=='SkyCampaignDescriptorKey'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWatchdogCodeS3Key'].ParameterValue | [0],Parameters[?ParameterKey=='SkyMustStartFoundationRevision'].ParameterValue | [0],Parameters[?ParameterKey=='EnableSkyWorkerStartV2Coordinator'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2FoundationRevision'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2CodeSha256'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2CodeVersionId'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2DescriptorRelativeKey'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2DescriptorFileSha256'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2IntentFileSha256'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2IntentBodySha256'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2ControllerInstanceId'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2ControllerClusterName'].ParameterValue | [0]]" \
      --output text \
      --profile "$AWS_PROFILE" --region "$REGION"
  )
  local revision
  for revision in None disabled; do
    local expected_disabled_state
    expected_disabled_state=$(
      printf 'true\tfalse\tfalse\t%s\t%s\t%s\t%s\t%s' \
        "$RUN_ID" \
        "$DESCRIPTOR_KEY" \
        "$WATCHDOG_CODE_S3_KEY" \
        "$revision" \
        "$WORKER_V2_LEGACY_ABSENT_STATE"
    )
    if [ "$current_state" = "$expected_disabled_state" ]; then
      verify_no_stack_resources_with_prefix \
        SkyWorkerStartV2 \
        "legacy stack still has worker-v2 coordinator resources"
      return
    fi
  done
  local legacy_state=false
  for revision in None disabled; do
    local expected_legacy_state
    expected_legacy_state=$(
      printf 'true\tNone\tNone\t%s\t%s\t%s\t%s\t%s' \
        "$RUN_ID" \
        "$DESCRIPTOR_KEY" \
        "$WATCHDOG_CODE_S3_KEY" \
        "$revision" \
        "$WORKER_V2_LEGACY_ABSENT_STATE"
    )
    if [ "$current_state" = "$expected_legacy_state" ]; then
      legacy_state=true
      break
    fi
  done
  if [ "$legacy_state" = true ]; then
    verify_no_stack_resources_with_prefix \
      SkyMustStart \
      "legacy stack still has must-start coordinator resources"
    verify_no_stack_resources_with_prefix \
      SkyWorkerStartV2 \
      "legacy stack still has worker-v2 coordinator resources"
    return
  fi
  echo "foundation update does not preserve current Sky authority or exact worker-v2 disabled authority" >&2
  exit 65
}

verify_activation_proof() {
  : "${MUST_START_DESCRIPTOR_BODY_SHA256:?set MUST_START_DESCRIPTOR_BODY_SHA256}"
  require_sha256 \
    MUST_START_DESCRIPTOR_BODY_SHA256 \
    "$MUST_START_DESCRIPTOR_BODY_SHA256"
  local marker_prefix=\
"campaigns/${RUN_ID}/monitor/must-start/${MUST_START_MANAGED_MODE}/${MUST_START_SUBMISSION_BODY_SHA256}"
  local binding_key="${marker_prefix}/JOB_BINDING.json"
  local binding_path="$WORK/JOB_BINDING.json"
  local observation_path="$WORK/controller-observation.json"
  aws s3api get-object \
    --bucket "$MODEL_BUCKET" \
    --key "$binding_key" \
    --profile "$AWS_PROFILE" --region "$REGION" \
    "$binding_path" >/dev/null
  local authority_args=(
    --job-binding "$binding_path"
    --account-id 246813579024
    --region "$REGION"
    --bucket "$MODEL_BUCKET"
    --run-id "$RUN_ID"
    --managed-mode "$MUST_START_MANAGED_MODE"
    --descriptor-key "$DESCRIPTOR_KEY"
    --descriptor-file-sha256 "$MUST_START_DESCRIPTOR_FILE_SHA256"
    --descriptor-body-sha256 "$MUST_START_DESCRIPTOR_BODY_SHA256"
    --submission-key "$MUST_START_SUBMISSION_KEY"
    --submission-body-sha256 "$MUST_START_SUBMISSION_BODY_SHA256"
    --submission-submitted-at "$MUST_START_SUBMISSION_SUBMITTED_AT"
    --target-job-id "$MUST_START_TARGET_JOB_ID"
    --workspace "${MUST_START_WORKSPACE:-default}"
    --job-name "$MUST_START_JOB_NAME"
    --must-start-by "$MUST_START_BY"
    --controller-instance-id "$MUST_START_CONTROLLER_INSTANCE_ID"
    --controller-instance-type "$MUST_START_CONTROLLER_INSTANCE_TYPE"
    --controller-profile-arn "$MUST_START_CONTROLLER_PROFILE_ARN"
    --controller-cluster-name "$MUST_START_CONTROLLER_CLUSTER_NAME"
  )
  local observation_key
  observation_key=$(
    python3 "$SCRIPT_DIR/validate_must_start_activation.py" \
      observation-key "${authority_args[@]}"
  )
  aws s3api get-object \
    --bucket "$MODEL_BUCKET" \
    --key "$observation_key" \
    --profile "$AWS_PROFILE" --region "$REGION" \
    "$observation_path" >/dev/null
  local proof
  proof=$(
    python3 "$SCRIPT_DIR/validate_must_start_activation.py" \
      validate "${authority_args[@]}" \
      --observation "$observation_path"
  )
  ACTIVATION_BINDING_SHA=$(
    python3 -c \
      'import json,sys; print(json.loads(sys.argv[1])["job_binding_body_sha256"])' \
      "$proof"
  )
  ACTIVATION_OBSERVATION_SHA=$(
    python3 -c \
      'import json,sys; print(json.loads(sys.argv[1])["observation_body_sha256"])' \
      "$proof"
  )
  require_sha256 ACTIVATION_BINDING_SHA "$ACTIVATION_BINDING_SHA"
  require_sha256 ACTIVATION_OBSERVATION_SHA "$ACTIVATION_OBSERVATION_SHA"
  PARAMETERS+=(
    "SkyMustStartActivationJobBindingSha256=$ACTIVATION_BINDING_SHA"
    "SkyMustStartActivationObservationSha256=$ACTIVATION_OBSERVATION_SHA"
  )
}

if [ "$MODE" = execute-active ] && \
  [ "${MUST_START_EXECUTE_APPROVAL:-}" != \
    execute-exact-reviewed-must-start-change-set ]; then
  echo "set MUST_START_EXECUTE_APPROVAL to the exact active execution approval" >&2
  exit 65
fi
if [ "$MODE" = execute-foundation ] && \
  [ "${MUST_START_FOUNDATION_EXECUTE_APPROVAL:-}" != \
    execute-exact-reviewed-must-start-foundation-change-set ]; then
  echo "set MUST_START_FOUNDATION_EXECUTE_APPROVAL to the exact foundation execution approval" >&2
  exit 65
fi

case "$MODE" in
  prepare-active)
    ACTIVE_HANDOFF_FILE="${MUST_START_CHANGE_SET_HANDOFF_FILE:-}"
    require_unused_handoff_target "$ACTIVE_HANDOFF_FILE"
    ;;
  prepare-foundation)
    FOUNDATION_HANDOFF_FILE=\
"${MUST_START_FOUNDATION_CHANGE_SET_HANDOFF_FILE:-}"
    require_unused_handoff_target "$FOUNDATION_HANDOFF_FILE"
    ;;
  inspect-active)
    load_change_set_handoff \
      "${MUST_START_CHANGE_SET_HANDOFF_FILE:-}" \
      active
    ;;
  inspect-foundation)
    load_change_set_handoff \
      "${MUST_START_FOUNDATION_CHANGE_SET_HANDOFF_FILE:-}" \
      foundation
    ;;
esac

if [ "$MODE" = inspect-active ] || [ "$MODE" = inspect-foundation ]; then
  verify_prepared_stack_and_change_set
  aws cloudformation describe-change-set \
    --stack-name "$PREPARED_STACK_ID" \
    --change-set-name "$CHANGE_SET_ARN" \
    --profile "$AWS_PROFILE" --region "$REGION"
  exit 0
fi

PARAMETERS=(
  EnableCampaignController=false
  CampaignAlertEmail=operator@example.com
)
append_worker_v2_disabled_parameters

if [ "$MODE" = core ]; then
  require_new_stack_for_core
  PARAMETERS+=(
    EnableSkyPilotSupport=false
    EnableSkyMustStartObserve=false
    EnableSkyMustStartCancel=false
    "SkyMustStartFoundationRevision=$FOUNDATION_REVISION"
  )
else
  : "${RUN_ID:?set RUN_ID}"
  : "${DESCRIPTOR_KEY:?set DESCRIPTOR_KEY}"
  : "${WATCHDOG_CODE_S3_KEY:?set WATCHDOG_CODE_S3_KEY}"
  PARAMETERS+=(
    EnableSkyPilotSupport=true
    "SkyCampaignRunId=$RUN_ID"
    "SkyCampaignDescriptorKey=$DESCRIPTOR_KEY"
    "SkyWatchdogCodeS3Key=$WATCHDOG_CODE_S3_KEY"
  )
  if [ "$MODE" = prepare-foundation ] || \
    [ "$MODE" = execute-foundation ]; then
    verify_foundation_preserves_current_sky_authority
    if [ "$MODE" = execute-foundation ]; then
      load_change_set_handoff \
        "${MUST_START_FOUNDATION_CHANGE_SET_HANDOFF_FILE:-}" \
        foundation
    fi
    PARAMETERS+=(
      EnableSkyMustStartObserve=false
      EnableSkyMustStartCancel=false
      "SkyMustStartFoundationRevision=$FOUNDATION_REVISION"
    )
  elif [ "$MODE" = enable ]; then
    if [ "${ENABLE_MUST_START_CANCEL:-false}" = true ]; then
      echo "active mode must use prepare-active, inspect-active, and execute-active" >&2
      exit 65
    fi
    MUST_START_OBSERVE="${ENABLE_MUST_START_OBSERVE:-false}"
    if [ "$MUST_START_OBSERVE" = true ]; then
      require_must_start_inputs
      foundation_consumer_mode=enable-observe
    elif [ "$MUST_START_OBSERVE" != false ]; then
      echo "ENABLE_MUST_START_OBSERVE must be true or false" >&2
      exit 65
    else
      foundation_consumer_mode=enable-base
    fi
    verify_current_foundation_installed "$foundation_consumer_mode"
    if [ "$MUST_START_OBSERVE" = true ]; then
      verify_code_artifact
      append_must_start_parameters true false
    else
      PARAMETERS+=(
        EnableSkyMustStartObserve=false
        EnableSkyMustStartCancel=false
      )
    fi
  else
    require_must_start_inputs
    verify_current_foundation_installed active
    if [ "$MODE" = execute-active ]; then
      load_change_set_handoff \
        "${MUST_START_CHANGE_SET_HANDOFF_FILE:-}" \
        active
    fi
    verify_code_artifact
    append_must_start_parameters true true
    verify_activation_proof
  fi
fi

if [ "$MODE" = execute-foundation ]; then
  verify_prepared_stack_and_change_set
  foundation_state=$(
    aws cloudformation describe-change-set \
      --stack-name "$PREPARED_STACK_ID" \
      --change-set-name "$CHANGE_SET_ARN" \
      --query \
        "[Status,ExecutionStatus,Parameters[?ParameterKey=='EnableSkyPilotSupport'].ParameterValue | [0],Parameters[?ParameterKey=='EnableSkyMustStartObserve'].ParameterValue | [0],Parameters[?ParameterKey=='EnableSkyMustStartCancel'].ParameterValue | [0],Parameters[?ParameterKey=='SkyCampaignRunId'].ParameterValue | [0],Parameters[?ParameterKey=='SkyCampaignDescriptorKey'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWatchdogCodeS3Key'].ParameterValue | [0],Parameters[?ParameterKey=='SkyMustStartFoundationRevision'].ParameterValue | [0],Parameters[?ParameterKey=='EnableSkyWorkerStartV2Coordinator'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2FoundationRevision'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2CodeSha256'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2CodeVersionId'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2DescriptorRelativeKey'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2DescriptorFileSha256'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2IntentFileSha256'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2IntentBodySha256'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2ControllerInstanceId'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2ControllerClusterName'].ParameterValue | [0]]" \
      --output text \
      --profile "$AWS_PROFILE" --region "$REGION"
  )
  expected_foundation_state=$(
    printf 'CREATE_COMPLETE\tAVAILABLE\ttrue\tfalse\tfalse\t%s\t%s\t%s\t%s\t%s' \
      "$RUN_ID" \
      "$DESCRIPTOR_KEY" \
      "$WATCHDOG_CODE_S3_KEY" \
      "$FOUNDATION_REVISION" \
      "$WORKER_V2_DISABLED_STATE"
  )
  if [ "$foundation_state" != "$expected_foundation_state" ]; then
    echo "reviewed foundation change set no longer preserves exact disabled coordinator authority" >&2
    exit 65
  fi
  foundation_changes_file="$WORK/foundation-change-graph.json"
  aws cloudformation describe-change-set \
    --stack-name "$PREPARED_STACK_ID" \
    --change-set-name "$CHANGE_SET_ARN" \
    --query "Changes" \
    --output json \
    --profile "$AWS_PROFILE" --region "$REGION" \
    >"$foundation_changes_file"
  python3 - "$foundation_changes_file" <<'PY'
from collections import Counter
import json
import sys
from pathlib import Path


def fail():
    raise SystemExit("foundation change set causal graph is not exact")


def detail(
    evaluation,
    source,
    causing_entity,
    target_name,
    recreation="Never",
):
    return (
        evaluation,
        source,
        causing_entity,
        "Properties",
        target_name,
        recreation,
    )


role_details = (
    detail(
        "Dynamic",
        "ResourceAttribute",
        "ModelBucket.Arn",
        "Policies",
    ),
    detail(
        "Static",
        "ParameterReference",
        "SkyMustStartSubmissionBodySha256",
        "Policies",
    ),
    detail(
        "Static",
        "ParameterReference",
        "SkyMustStartManagedMode",
        "Policies",
    ),
    detail("Dynamic", "DirectModification", None, "Policies"),
)
expected = {
    "InstanceRole": ("AWS::IAM::Role", "False", role_details),
    "ModelBucket": (
        "AWS::S3::Bucket",
        "False",
        (
            detail(
                "Static",
                "DirectModification",
                None,
                "VersioningConfiguration",
            ),
        ),
    ),
    "S3BatchChecksumRole": (
        "AWS::IAM::Role",
        "False",
        (
            detail(
                "Dynamic",
                "ResourceAttribute",
                "ModelBucket.Arn",
                "Policies",
            ),
        ),
    ),
    "SkyPilotControllerRole": (
        "AWS::IAM::Role",
        "False",
        (
            role_details[0],
            detail(
                "Dynamic",
                "ResourceAttribute",
                "SkyPilotWorkerRole.Arn",
                "Policies",
            ),
            *role_details[1:],
        ),
    ),
    "SkyPilotWorkerRole": ("AWS::IAM::Role", "False", role_details),
    "SkyWatchdogFunction": (
        "AWS::Lambda::Function",
        "False",
        (
            detail(
                "Dynamic",
                "ResourceAttribute",
                "SkyWatchdogRole.Arn",
                "Role",
            ),
        ),
    ),
    "SkyWatchdogInvokePermission": (
        "AWS::Lambda::Permission",
        "Conditional",
        (
            detail(
                "Dynamic",
                "ResourceAttribute",
                "SkyWatchdogRule.Arn",
                "SourceArn",
                "Always",
            ),
        ),
    ),
    "SkyWatchdogRole": ("AWS::IAM::Role", "False", role_details),
    "SkyWatchdogRule": (
        "AWS::Events::Rule",
        "False",
        (
            detail(
                "Dynamic",
                "ResourceAttribute",
                "SkyWatchdogFunction.Arn",
                "Targets",
            ),
        ),
    ),
}

try:
    changes = json.loads(Path(sys.argv[1]).read_text())
except (OSError, UnicodeError, json.JSONDecodeError):
    fail()
if not isinstance(changes, list) or len(changes) != len(expected):
    fail()

seen = set()
for change in changes:
    if (
        not isinstance(change, dict)
        or set(change) != {"Type", "ResourceChange"}
        or change.get("Type") != "Resource"
        or not isinstance(change.get("ResourceChange"), dict)
    ):
        fail()
    resource_change = change["ResourceChange"]
    if set(resource_change) != {
        "Action",
        "Details",
        "LogicalResourceId",
        "PhysicalResourceId",
        "Replacement",
        "ResourceType",
        "Scope",
    }:
        fail()
    logical_id = resource_change.get("LogicalResourceId")
    if (
        not isinstance(logical_id, str)
        or logical_id in seen
        or logical_id not in expected
    ):
        fail()
    seen.add(logical_id)
    resource_type, replacement, expected_details = expected[logical_id]
    if (
        resource_change.get("Action") != "Modify"
        or resource_change.get("ResourceType") != resource_type
        or resource_change.get("Replacement") != replacement
        or resource_change.get("Scope") != ["Properties"]
        or not isinstance(resource_change.get("PhysicalResourceId"), str)
        or not resource_change["PhysicalResourceId"]
        or not isinstance(resource_change.get("Details"), list)
    ):
        fail()
    actual_details = []
    for item in resource_change["Details"]:
        if not isinstance(item, dict):
            fail()
        causing_entity = item.get("CausingEntity")
        expected_item_fields = {
            "ChangeSource",
            "Evaluation",
            "Target",
        }
        if causing_entity is not None:
            expected_item_fields.add("CausingEntity")
        target = item.get("Target")
        if (
            set(item) != expected_item_fields
            or not isinstance(target, dict)
            or set(target)
            != {"Attribute", "Name", "RequiresRecreation"}
        ):
            fail()
        actual_details.append(
            (
                item.get("Evaluation"),
                item.get("ChangeSource"),
                causing_entity,
                target.get("Attribute"),
                target.get("Name"),
                target.get("RequiresRecreation"),
            )
        )
    if Counter(actual_details) != Counter(expected_details):
        fail()
if seen != set(expected):
    fail()
PY
  verify_prepared_stack_and_change_set
  verify_template_upload_object \
    "$PINNED_TEMPLATE_SHA256" \
    "$PREPARED_TEMPLATE_UPLOAD_BUCKET" \
    "$PREPARED_TEMPLATE_UPLOAD_KEY"
  aws cloudformation execute-change-set \
    --stack-name "$PREPARED_STACK_ID" \
    --change-set-name "$CHANGE_SET_ARN" \
    --profile "$AWS_PROFILE" --region "$REGION"
  echo "MUST-START-FOUNDATION-CHANGESET-EXECUTED stack=$STACK changeset=$CHANGE_SET_ARN"
  exit 0
fi

if [ "$MODE" = execute-active ]; then
  verify_prepared_stack_and_change_set
  change_set_state=$(
    aws cloudformation describe-change-set \
      --stack-name "$PREPARED_STACK_ID" \
      --change-set-name "$CHANGE_SET_ARN" \
      --query \
        "[Status,ExecutionStatus,Parameters[?ParameterKey=='SkyMustStartControllerBaselineBodySha256'].ParameterValue | [0],Parameters[?ParameterKey=='SkyMustStartActivationJobBindingSha256'].ParameterValue | [0],Parameters[?ParameterKey=='SkyMustStartActivationObservationSha256'].ParameterValue | [0],Parameters[?ParameterKey=='SkyMustStartCancelCodeSha256'].ParameterValue | [0],Parameters[?ParameterKey=='SkyMustStartCancelCodeVersionId'].ParameterValue | [0],Parameters[?ParameterKey=='EnableSkyWorkerStartV2Coordinator'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2FoundationRevision'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2CodeSha256'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2CodeVersionId'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2DescriptorRelativeKey'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2DescriptorFileSha256'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2IntentFileSha256'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2IntentBodySha256'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2ControllerInstanceId'].ParameterValue | [0],Parameters[?ParameterKey=='SkyWorkerStartV2ControllerClusterName'].ParameterValue | [0]]" \
      --output text \
      --profile "$AWS_PROFILE" --region "$REGION"
  )
  expected_state=$(
    printf 'CREATE_COMPLETE\tAVAILABLE\t%s\t%s\t%s\t%s\t%s\t%s' \
      "$MUST_START_CONTROLLER_BASELINE_BODY_SHA256" \
      "$ACTIVATION_BINDING_SHA" \
      "$ACTIVATION_OBSERVATION_SHA" \
      "$MUST_START_CANCEL_CODE_SHA256" \
      "$MUST_START_CANCEL_CODE_VERSION_ID" \
      "$WORKER_V2_DISABLED_STATE"
  )
  if [ "$change_set_state" != "$expected_state" ]; then
    echo "reviewed change set no longer binds exact baseline/activation/code/worker-v2 proof" >&2
    exit 65
  fi
  verify_prepared_stack_and_change_set
  verify_template_upload_object \
    "$PINNED_TEMPLATE_SHA256" \
    "$PREPARED_TEMPLATE_UPLOAD_BUCKET" \
    "$PREPARED_TEMPLATE_UPLOAD_KEY"
  aws cloudformation execute-change-set \
    --stack-name "$PREPARED_STACK_ID" \
    --change-set-name "$CHANGE_SET_ARN" \
    --profile "$AWS_PROFILE" --region "$REGION"
  echo "MUST-START-ACTIVE-CHANGESET-EXECUTED stack=$STACK changeset=$CHANGE_SET_ARN"
  exit 0
fi

DEPLOY_ARGS=(
  --stack-name "$STACK"
  --template-file "$TEMPLATE"
  --s3-bucket "$TEMPLATE_UPLOAD_BUCKET"
  --capabilities CAPABILITY_NAMED_IAM
  --parameter-overrides "${PARAMETERS[@]}"
  --profile "$AWS_PROFILE"
  --region "$REGION"
)

if [ "$MODE" = prepare-active ] || [ "$MODE" = prepare-foundation ]; then
  template_sha256=$(shasum -a 256 "$TEMPLATE" | awk '{print $1}')
  if [ "$template_sha256" != "$FROZEN_TEMPLATE_SHA256" ]; then
    echo "local template no longer matches frozen UTF-8 upload" >&2
    exit 65
  fi
  pre_create_stack_identity=$(current_stack_identity)
  IFS=$'\t' read -r \
    pre_create_stack_id \
    pre_create_stack_status \
    <<<"$pre_create_stack_identity"
  deploy_output="$WORK/cloudformation-deploy.stdout"
  aws cloudformation deploy \
    "${DEPLOY_ARGS[@]}" \
    --no-execute-changeset \
    --fail-on-empty-changeset \
    >"$deploy_output" || {
      deploy_status=$?
      cat "$deploy_output"
      exit "$deploy_status"
    }
  cat "$deploy_output"
  change_set_arn=$(parse_generated_change_set_arn "$deploy_output")
  post_create_stack_identity=$(current_stack_identity)
  if [ "$post_create_stack_identity" != "$pre_create_stack_identity" ]; then
    echo "stack identity changed while preparing change set" >&2
    exit 65
  fi
  prepared_stack_id=$pre_create_stack_id
  prepared_stack_status=$pre_create_stack_status
  verify_template_upload_object \
    "$template_sha256" \
    "$TEMPLATE_UPLOAD_BUCKET" \
    "$TEMPLATE_UPLOAD_KEY"
  verify_change_set_identity "$change_set_arn" "$prepared_stack_id"
  verify_change_set_template \
    "$change_set_arn" \
    "$template_sha256" \
    "$prepared_stack_id"
  if [ "$MODE" = prepare-active ]; then
    persist_change_set_handoff \
      "$ACTIVE_HANDOFF_FILE" \
      active \
      "$change_set_arn" \
      "$template_sha256" \
      "$prepared_stack_id" \
      "$prepared_stack_status"
    echo "MUST_START_CHANGE_SET_ID=$change_set_arn"
    echo "MUST_START_CHANGE_SET_TEMPLATE_SHA256=$template_sha256"
    echo "MUST_START_CHANGE_SET_HANDOFF_FILE=$ACTIVE_HANDOFF_FILE"
    prepared_label=MUST-START-ACTIVE-CHANGESET-PREPARED
  else
    persist_change_set_handoff \
      "$FOUNDATION_HANDOFF_FILE" \
      foundation \
      "$change_set_arn" \
      "$template_sha256" \
      "$prepared_stack_id" \
      "$prepared_stack_status"
    echo "MUST_START_FOUNDATION_CHANGE_SET_ID=$change_set_arn"
    echo "MUST_START_FOUNDATION_CHANGE_SET_TEMPLATE_SHA256=$template_sha256"
    echo "MUST_START_FOUNDATION_CHANGE_SET_HANDOFF_FILE=$FOUNDATION_HANDOFF_FILE"
    prepared_label=MUST-START-FOUNDATION-CHANGESET-PREPARED
  fi
  echo "$prepared_label stack=$STACK changeset=$change_set_arn"
  exit 0
fi

aws cloudformation deploy \
  "${DEPLOY_ARGS[@]}" \
  --no-fail-on-empty-changeset
aws cloudformation describe-stacks --stack-name "$STACK" \
  --profile "$AWS_PROFILE" --region "$REGION" \
  --query 'Stacks[0].[StackStatus,Outputs]' --output json

if [ "$MODE" = enable ]; then
  TOPIC_ARN=$(aws cloudformation describe-stacks --stack-name "$STACK" \
    --profile "$AWS_PROFILE" --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='CampaignAlertTopicArn'].OutputValue" \
    --output text)
  CAMPAIGN_ALERT_TOPIC_ARN="$TOPIC_ARN" \
    "$SCRIPT_DIR/assert_sns_email_confirmed.sh" >/dev/null
  echo "SKYPILOT-CONTROL-PLANE-ENABLED stack=$STACK run_id=$RUN_ID"
fi
