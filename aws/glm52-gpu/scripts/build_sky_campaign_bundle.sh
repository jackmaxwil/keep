#!/usr/bin/env bash
# Assemble a complete local Sky campaign authority bundle without touching AWS.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
: "${RUN_ID:?set RUN_ID}"
: "${BUCKET:?set BUCKET}"
: "${MUST_START_BY:?set MUST_START_BY}"
: "${IMAGE_ID:?set IMAGE_ID}"
: "${OBJECT_AUTHORITIES_JSON:?set OBJECT_AUTHORITIES_JSON}"
: "${SOURCE_SNAPSHOT_PREFIX:?set SOURCE_SNAPSHOT_PREFIX}"
: "${SOURCE_SNAPSHOT_SHA256:?set SOURCE_SNAPSHOT_SHA256}"
: "${NON_VQ_PREFIX:?set NON_VQ_PREFIX}"
: "${NON_VQ_PACKAGE_SHA256:?set NON_VQ_PACKAGE_SHA256}"
: "${TEICH_PACK_KEY:?set TEICH_PACK_KEY}"
: "${TEICH_PACK_SHA256:?set TEICH_PACK_SHA256}"
: "${FROZEN_PROMPT_PACK_KEY:?set FROZEN_PROMPT_PACK_KEY}"
: "${FROZEN_PROMPT_PACK_SHA256:?set FROZEN_PROMPT_PACK_SHA256}"
: "${TRAINING_BASELINE_PREFIX:?set TRAINING_BASELINE_PREFIX}"
: "${TRAINING_BASELINE_SHA256:?set TRAINING_BASELINE_SHA256}"
: "${QUALIFICATION_CACHE_PREFIX:?set QUALIFICATION_CACHE_PREFIX}"
: "${QUALIFICATION_CACHE_MANIFEST_SHA256:?set QUALIFICATION_CACHE_MANIFEST_SHA256}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to a new local directory}"
SUBMISSION_ID="${SUBMISSION_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
CONTROLLER_IDENTITY="${CONTROLLER_IDENTITY:-arn:aws:iam::246813579024:role/keep-glm52-skypilot-controller}"
WORKER_IDENTITY="${WORKER_IDENTITY:-arn:aws:iam::246813579024:role/keep-glm52-gpu-worker}"
VPC_NAME="${VPC_NAME:-keep-glm52-vpc}"
APPROVAL_INGESTED_AT="${APPROVAL_INGESTED_AT:-}"
SLACK_PERMALINK="${SLACK_PERMALINK:-}"
APPROVAL_SOURCE="${APPROVAL_SOURCE:-}"
EXPECTED_APPROVAL_SHA256="${EXPECTED_APPROVAL_SHA256:-}"
REPO_TAR_SOURCE="${REPO_TAR_SOURCE:-}"
EXPECTED_REPO_TAR_SHA256="${EXPECTED_REPO_TAR_SHA256:-}"

fail() {
  printf '%s\n' "$*" >&2
  exit 2
}

require_lower_sha256() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9a-f]{64}$ ]]; then
    fail "$name must be a lowercase SHA-256"
  fi
}

if [ -n "$APPROVAL_SOURCE" ] && [ -z "$EXPECTED_APPROVAL_SHA256" ]; then
  fail "APPROVAL_SOURCE requires EXPECTED_APPROVAL_SHA256"
fi
if [ -z "$APPROVAL_SOURCE" ] && [ -n "$EXPECTED_APPROVAL_SHA256" ]; then
  fail "EXPECTED_APPROVAL_SHA256 requires APPROVAL_SOURCE"
fi
if [ -n "$REPO_TAR_SOURCE" ] && [ -z "$EXPECTED_REPO_TAR_SHA256" ]; then
  fail "REPO_TAR_SOURCE requires EXPECTED_REPO_TAR_SHA256"
fi
if [ -z "$REPO_TAR_SOURCE" ] && [ -n "$EXPECTED_REPO_TAR_SHA256" ]; then
  fail "EXPECTED_REPO_TAR_SHA256 requires REPO_TAR_SOURCE"
fi
if [ -n "$APPROVAL_SOURCE" ]; then
  if [ -n "$APPROVAL_INGESTED_AT" ] || [ -n "$SLACK_PERMALINK" ]; then
    fail "approval reuse conflicts with approval rebuild inputs"
  fi
  require_lower_sha256 EXPECTED_APPROVAL_SHA256 "$EXPECTED_APPROVAL_SHA256"
  if [ ! -f "$APPROVAL_SOURCE" ] || [ -L "$APPROVAL_SOURCE" ]; then
    fail "authority source must be a regular non-symlink file: APPROVAL_SOURCE"
  fi
else
  : "${APPROVAL_INGESTED_AT:?set APPROVAL_INGESTED_AT}"
fi
if [ -n "$REPO_TAR_SOURCE" ]; then
  require_lower_sha256 EXPECTED_REPO_TAR_SHA256 "$EXPECTED_REPO_TAR_SHA256"
  if [ ! -f "$REPO_TAR_SOURCE" ] || [ -L "$REPO_TAR_SOURCE" ]; then
    fail "authority source must be a regular non-symlink file: REPO_TAR_SOURCE"
  fi
fi

if [ -n "$APPROVAL_SOURCE" ] || [ -n "$REPO_TAR_SOURCE" ]; then
  python3 - "$OUTPUT_DIR" "$APPROVAL_SOURCE" "$REPO_TAR_SOURCE" <<'PY'
import pathlib
import sys

output = pathlib.Path(sys.argv[1]).resolve(strict=False)
for raw_source, output_name in zip(
    sys.argv[2:],
    ("GPU_SPEND_APPROVAL.json", "repo.tar.gz"),
):
    if not raw_source:
        continue
    source = pathlib.Path(raw_source).resolve(strict=True)
    candidates = {
        output,
        (output / output_name).resolve(strict=False),
    }
    if source in candidates:
        raise SystemExit("OUTPUT_DIR must not alias an authority source")
PY
  if [ -e "$OUTPUT_DIR" ] || [ -L "$OUTPUT_DIR" ]; then
    fail "OUTPUT_DIR must be a new local directory"
  fi
else
  test ! -e "$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"

REPO_TAR_TMP=""
APPROVAL_TMP=""
if [ -n "$APPROVAL_SOURCE" ] || [ -n "$REPO_TAR_SOURCE" ]; then
  trap 'rm -f -- "$REPO_TAR_TMP" "$APPROVAL_TMP"' EXIT
fi

REPO_TAR="$OUTPUT_DIR/repo.tar.gz"
if [ -n "$REPO_TAR_SOURCE" ]; then
  REPO_TAR_FINAL="$REPO_TAR"
  REPO_TAR="$OUTPUT_DIR/.repo.tar.gz.$$.tmp"
  REPO_TAR_TMP="$REPO_TAR"
  test ! -e "$REPO_TAR"
  cp "$REPO_TAR_SOURCE" "$REPO_TAR"
  REPO_SHA=$(shasum -a 256 "$REPO_TAR" | awk '{print $1}')
  if [ "$REPO_SHA" != "$EXPECTED_REPO_TAR_SHA256" ]; then
    fail "repository tar source SHA-256 mismatch"
  fi
  mv "$REPO_TAR" "$REPO_TAR_FINAL"
  REPO_TAR_TMP=""
  REPO_TAR="$REPO_TAR_FINAL"
else
  "$SCRIPT_DIR/build_sky_repository_tar.sh" "$REPO_TAR" >/dev/null
  REPO_SHA=$(shasum -a 256 "$REPO_TAR" | awk '{print $1}')
fi
REPO_KEY="campaigns/$RUN_ID/repository/keep-$REPO_SHA.tar.gz"

APPROVAL="$OUTPUT_DIR/GPU_SPEND_APPROVAL.json"
if [ -n "$APPROVAL_SOURCE" ]; then
  APPROVAL_FINAL="$APPROVAL"
  APPROVAL="$OUTPUT_DIR/.GPU_SPEND_APPROVAL.json.$$.tmp"
  APPROVAL_TMP="$APPROVAL"
  test ! -e "$APPROVAL"
  cp "$APPROVAL_SOURCE" "$APPROVAL"
  APPROVAL_SHA=$(shasum -a 256 "$APPROVAL" | awk '{print $1}')
  if [ "$APPROVAL_SHA" != "$EXPECTED_APPROVAL_SHA256" ]; then
    fail "approval source SHA-256 mismatch"
  fi
  mv "$APPROVAL" "$APPROVAL_FINAL"
  APPROVAL_TMP=""
  APPROVAL="$APPROVAL_FINAL"
else
  APPROVAL_ARGS=(
    --ingested-at "$APPROVAL_INGESTED_AT"
    --output "$APPROVAL"
  )
  if [ -n "$SLACK_PERMALINK" ]; then
    APPROVAL_ARGS+=(--slack-permalink "$SLACK_PERMALINK")
  fi
  "$SCRIPT_DIR/build_gpu_spend_approval.py" "${APPROVAL_ARGS[@]}" >/dev/null
  APPROVAL_SHA=$(shasum -a 256 "$APPROVAL" | awk '{print $1}')
fi
APPROVAL_KEY="campaigns/$RUN_ID/authorities/GPU_SPEND_APPROVAL-$APPROVAL_SHA.json"

TRAINING="$OUTPUT_DIR/training-config.json"
"$SCRIPT_DIR/build_sky_training_config.py" --output "$TRAINING" >/dev/null
TRAINING_SHA=$(shasum -a 256 "$TRAINING" | awk '{print $1}')
TRAINING_KEY="campaigns/$RUN_ID/authorities/training-config-$TRAINING_SHA.json"

WATCHDOG="$OUTPUT_DIR/sky-watchdog.zip"
"$SCRIPT_DIR/package_sky_watchdog_lambda.sh" "$WATCHDOG" >/dev/null
WATCHDOG_SHA=$(shasum -a 256 "$WATCHDOG" | awk '{print $1}')
WATCHDOG_KEY="campaigns/$RUN_ID/control/sky-watchdog-$WATCHDOG_SHA.zip"

OBJECTS="$OUTPUT_DIR/inventory-objects.json"
python3 - "$OBJECT_AUTHORITIES_JSON" "$OBJECTS" "$RUN_ID" \
  "$REPO_TAR" "$REPO_KEY" "$APPROVAL" "$APPROVAL_KEY" \
  "$TRAINING" "$TRAINING_KEY" "$WATCHDOG" "$WATCHDOG_KEY" \
  "$REPO_TAR_SOURCE" "$EXPECTED_REPO_TAR_SHA256" <<'PY'
import hashlib, json, pathlib, sys
(source, output, run_id, repo, repo_key, approval, approval_key,
 training, training_key, watchdog, watchdog_key, repo_source,
 expected_repo_sha256) = sys.argv[1:]
items = json.loads(pathlib.Path(source).read_bytes())
if not isinstance(items, list):
    raise SystemExit("OBJECT_AUTHORITIES_JSON must contain a list")
def local(path, key, kind):
    value = pathlib.Path(path)
    return {
        "key": key,
        "size": value.stat().st_size,
        "sha256": hashlib.sha256(value.read_bytes()).hexdigest(),
        "kind": kind,
        "safetensors": False,
        "run_scope": run_id,
    }
local_repo = local(repo, repo_key, "repository_tar")
repository_authorities = [
    item
    for item in items
    if isinstance(item, dict) and item.get("kind") == "repository_tar"
]
if len(repository_authorities) > 1:
    raise SystemExit(
        "OBJECT_AUTHORITIES_JSON contains multiple repository authorities"
    )
if repository_authorities:
    authority = repository_authorities[0]
    if not repo_source or not expected_repo_sha256:
        raise SystemExit(
            "shared repository authority requires REPO_TAR_SOURCE"
        )
    if set(authority) != {
        "key",
        "size",
        "sha256",
        "kind",
        "safetensors",
        "run_scope",
        "version_id",
    }:
        raise SystemExit("shared repository authority schema mismatch")
    version_id = authority["version_id"]
    if (
        authority["key"] != local_repo["key"]
        or authority["size"] != local_repo["size"]
        or authority["sha256"] != local_repo["sha256"]
        or authority["sha256"] != expected_repo_sha256
        or authority["safetensors"] is not False
        or authority["run_scope"] != "shared"
        or not isinstance(version_id, str)
        or not version_id
        or version_id in {"null", "None"}
    ):
        raise SystemExit(
            "shared repository authority does not match exact repository source"
        )
else:
    items.append(local_repo)
items.extend([
    local(approval, approval_key, "approval"),
    local(training, training_key, "training_configuration"),
    local(watchdog, watchdog_key, "watchdog_code"),
])
pathlib.Path(output).write_text(json.dumps(items, sort_keys=True) + "\n")
PY

INVENTORY="$OUTPUT_DIR/artifact-inventory-v1.json"
"$SCRIPT_DIR/build_s3_artifact_inventory.py" \
  --run-id "$RUN_ID" --bucket "$BUCKET" \
  --objects-json "$OBJECTS" --output "$INVENTORY" >/dev/null
INVENTORY_SHA=$(shasum -a 256 "$INVENTORY" | awk '{print $1}')
INVENTORY_KEY="campaigns/$RUN_ID/inventories/artifact-inventory-$INVENTORY_SHA.json"

DESCRIPTOR="$OUTPUT_DIR/campaign-descriptor-v2.json"
DESCRIPTOR_KEY="campaigns/$RUN_ID/submissions/$SUBMISSION_ID/campaign-descriptor-v2.json"
"$SCRIPT_DIR/build_sky_campaign_descriptor.py" \
  --run-id "$RUN_ID" --must-start-by "$MUST_START_BY" \
  --controller-identity "$CONTROLLER_IDENTITY" \
  --worker-identity "$WORKER_IDENTITY" \
  --vpc-name "$VPC_NAME" --image-id "$IMAGE_ID" \
  --bucket "$BUCKET" --jobs-bucket "$BUCKET" \
  --repo-tar "$REPO_TAR" --repo-tar-key "$REPO_KEY" \
  --campaign-descriptor-key "$DESCRIPTOR_KEY" \
  --approval "$APPROVAL" --approval-key "$APPROVAL_KEY" \
  --source-snapshot-prefix "$SOURCE_SNAPSHOT_PREFIX" \
  --source-snapshot-sha256 "$SOURCE_SNAPSHOT_SHA256" \
  --non-vq-prefix "$NON_VQ_PREFIX" \
  --non-vq-package-sha256 "$NON_VQ_PACKAGE_SHA256" \
  --teich-pack-key "$TEICH_PACK_KEY" \
  --teich-pack-sha256 "$TEICH_PACK_SHA256" \
  --frozen-prompt-pack-key "$FROZEN_PROMPT_PACK_KEY" \
  --frozen-prompt-pack-sha256 "$FROZEN_PROMPT_PACK_SHA256" \
  --training-baseline-prefix "$TRAINING_BASELINE_PREFIX" \
  --training-baseline-sha256 "$TRAINING_BASELINE_SHA256" \
  --training-config-key "$TRAINING_KEY" \
  --training-config-sha256 "$TRAINING_SHA" \
  --artifact-inventory-key "$INVENTORY_KEY" \
  --artifact-inventory-sha256 "$INVENTORY_SHA" \
  --qualification-cache-prefix "$QUALIFICATION_CACHE_PREFIX" \
  --qualification-cache-manifest-sha256 "$QUALIFICATION_CACHE_MANIFEST_SHA256" \
  --output "$DESCRIPTOR" >/dev/null

python3 - "$OUTPUT_DIR" "$RUN_ID" "$BUCKET" \
  "$REPO_KEY" "$APPROVAL_KEY" "$TRAINING_KEY" "$WATCHDOG_KEY" \
  "$INVENTORY_KEY" "$DESCRIPTOR_KEY" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
run_id, bucket = sys.argv[2:4]
values = [
    ("repo.tar.gz", sys.argv[4], "repository_tar", 10),
    ("GPU_SPEND_APPROVAL.json", sys.argv[5], "approval", 20),
    ("training-config.json", sys.argv[6], "training_config", 30),
    ("sky-watchdog.zip", sys.argv[7], "watchdog", 40),
    ("artifact-inventory-v1.json", sys.argv[8], "artifact_inventory", 50),
    ("campaign-descriptor-v2.json", sys.argv[9], "descriptor", 100),
]
files = []
for name, key, role, order in values:
    path = root / name
    files.append({
        "local_name": name,
        "key": key,
        "role": role,
        "stage_order": order,
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
body = {
    "schema_version": 1,
    "record_type": "glm52_sky_campaign_bundle_v1",
    "run_id": run_id,
    "bucket": bucket,
    "files": files,
}
body["descriptor_body_sha256"] = json.loads(
    (root / "campaign-descriptor-v2.json").read_bytes()
)["descriptor_body_sha256"]
body["bundle_manifest_body_sha256"] = hashlib.sha256(
    json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
(root / "bundle-manifest-v1.json").write_text(
    json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n"
)
print(root / "bundle-manifest-v1.json")
PY
