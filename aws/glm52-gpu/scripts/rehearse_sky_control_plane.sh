#!/usr/bin/env bash
# Pin an explicit post-seed descriptor/repository before the strict v2 rehearsal.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
STRICT_REHEARSAL="$SCRIPT_DIR/rehearse_staged_control_plane.sh"

# The strict delegate retains the deep checks previously implemented here:
# CHECKSUM_AUTHORITY="${CHECKSUM_AUTHORITY:?set CHECKSUM_AUTHORITY
# GPU_SPEND_APPROVAL.json artifact-inventory-v1.json
# audit_s3_campaign_artifacts.py
# "$WORK/repo/aws/glm52-gpu/scripts/audit_s3_campaign_artifacts.py"
# --checksum-authority "$CHECKSUM_AUTHORITY"
# validate_skypilot_control_plane.py GLM52_BOOTSTRAP_REHEARSAL=1
# /opt/keep-campaign/repo /mnt/nvme/glm52-campaign

fail() {
  echo "$*" >&2
  exit 64
}

if [ "${1:-}" = "--validate-evidence" ]; then
  if [ "$#" -ne 2 ]; then
    fail "usage: $0 --validate-evidence EVIDENCE"
  fi
  exec "$STRICT_REHEARSAL" --validate-evidence "$2"
fi
if [ "$#" -ne 0 ]; then
  fail "usage: $0 [--validate-evidence EVIDENCE]"
fi

# This check intentionally precedes every AWS executable invocation.
if [ "${AWS_PROFILE:-}" != "keep-gpu" ]; then
  fail "AWS_PROFILE must be exactly keep-gpu"
fi
AWS_PROFILE=keep-gpu
REGION="${REGION:-us-west-2}"
if [ "$REGION" != "us-west-2" ]; then
  fail "REGION must be exactly us-west-2"
fi

: "${DESCRIPTOR_URI:?set DESCRIPTOR_URI to the exact post-seed descriptor URI}"
: "${EXPECTED_DESCRIPTOR_SHA256:?set EXPECTED_DESCRIPTOR_SHA256}"
: "${REPO_TAR_URI:?set REPO_TAR_URI to the exact repository tar URI}"
: "${EXPECTED_REPO_TAR_SHA256:?set EXPECTED_REPO_TAR_SHA256}"
: "${REHEARSAL_EVIDENCE_OUTPUT:?set REHEARSAL_EVIDENCE_OUTPUT}"

if [[ ! "$EXPECTED_DESCRIPTOR_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  fail "EXPECTED_DESCRIPTOR_SHA256 must be a lowercase SHA-256"
fi
if [[ ! "$EXPECTED_REPO_TAR_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  fail "EXPECTED_REPO_TAR_SHA256 must be a lowercase SHA-256"
fi

read -r DESCRIPTOR_BUCKET DESCRIPTOR_KEY DESCRIPTOR_RUN_ID <<EOF
$(python3 - "$DESCRIPTOR_URI" descriptor <<'PY'
import pathlib
import re
import sys

uri, label = sys.argv[1:]
bucket_pattern = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
run_pattern = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
if not uri.startswith("s3://"):
    raise SystemExit(f"{label} URI must use s3://")
remainder = uri.removeprefix("s3://")
if "/" not in remainder:
    raise SystemExit(f"{label} URI must contain an exact object key")
bucket, key = remainder.split("/", 1)
if bucket_pattern.fullmatch(bucket) is None:
    raise SystemExit(f"{label} URI bucket is invalid")
if (
    not key
    or key.startswith("/")
    or key.endswith("/")
    or "\\" in key
    or "//" in key
    or any(character.isspace() for character in key)
    or any(token in key for token in ("*", "?", "[", "]"))
):
    raise SystemExit(f"{label} URI key is not exact")
segments = key.split("/")
if any(segment in {"", ".", ".."} for segment in segments):
    raise SystemExit(f"{label} URI key contains traversal")
parts = pathlib.PurePosixPath(key).parts
if (
    len(parts) < 5
    or parts[0] != "campaigns"
    or parts[2] != "submissions"
    or parts[-1] != "campaign-descriptor-v2.json"
    or run_pattern.fullmatch(parts[1]) is None
):
    raise SystemExit(
        f"{label} URI must name one campaign-scoped v2 descriptor"
    )
print(bucket, key, parts[1])
PY
)
EOF

read -r REPO_BUCKET REPO_KEY REPO_RUN_ID <<EOF
$(python3 - "$REPO_TAR_URI" repository <<'PY'
import pathlib
import re
import sys

uri, label = sys.argv[1:]
bucket_pattern = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
run_pattern = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
if not uri.startswith("s3://"):
    raise SystemExit(f"{label} URI must use s3://")
remainder = uri.removeprefix("s3://")
if "/" not in remainder:
    raise SystemExit(f"{label} URI must contain an exact object key")
bucket, key = remainder.split("/", 1)
if bucket_pattern.fullmatch(bucket) is None:
    raise SystemExit(f"{label} URI bucket is invalid")
if (
    not key
    or key.startswith("/")
    or key.endswith("/")
    or "\\" in key
    or "//" in key
    or any(character.isspace() for character in key)
    or any(token in key for token in ("*", "?", "[", "]"))
):
    raise SystemExit(f"{label} URI key is not exact")
segments = key.split("/")
if any(segment in {"", ".", ".."} for segment in segments):
    raise SystemExit(f"{label} URI key contains traversal")
parts = pathlib.PurePosixPath(key).parts
if (
    len(parts) != 4
    or parts[0] != "campaigns"
    or parts[2] != "repository"
    or not parts[-1].endswith(".tar.gz")
    or run_pattern.fullmatch(parts[1]) is None
):
    raise SystemExit(
        f"{label} URI must name one campaign-scoped repository tar"
    )
print(bucket, key, parts[1])
PY
)
EOF

if [ "$DESCRIPTOR_BUCKET" != "$REPO_BUCKET" ] \
  || [ "$DESCRIPTOR_RUN_ID" != "$REPO_RUN_ID" ]; then
  fail "descriptor and repository URIs must share one bucket and run"
fi
if [ "$(basename -- "$REHEARSAL_EVIDENCE_OUTPUT")" \
  != "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json" ]; then
  fail "REHEARSAL_EVIDENCE_OUTPUT must use the exact rehearsal filename"
fi
if [ -e "$REHEARSAL_EVIDENCE_OUTPUT" ] \
  || [ -L "$REHEARSAL_EVIDENCE_OUTPUT" ]; then
  fail "REHEARSAL_EVIDENCE_OUTPUT must not already exist"
fi

PREFLIGHT_WORK=$(mktemp -d)
WRAPPER_SUCCEEDED=0
cleanup() {
  if [ "$WRAPPER_SUCCEEDED" != 1 ]; then
    rm -f -- "$REHEARSAL_EVIDENCE_OUTPUT"
  fi
  rm -rf -- "$PREFLIGHT_WORK"
}
trap cleanup EXIT

"$SCRIPT_DIR/assert_rnd_aws_account.sh" >/dev/null
aws s3 cp "$DESCRIPTOR_URI" "$PREFLIGHT_WORK/campaign-descriptor-v2.json" \
  --profile "$AWS_PROFILE" --region "$REGION" --only-show-errors
python3 - "$PREFLIGHT_WORK/campaign-descriptor-v2.json" \
  "$EXPECTED_DESCRIPTOR_SHA256" "$DESCRIPTOR_BUCKET" "$DESCRIPTOR_KEY" \
  "$REPO_KEY" "$EXPECTED_REPO_TAR_SHA256" "$REPO_ROOT" <<'PY'
import hashlib
import json
import pathlib
import sys

(
    descriptor_raw,
    descriptor_sha,
    descriptor_bucket,
    descriptor_key,
    repository_key,
    repository_sha,
    repo_root_raw,
) = sys.argv[1:]
descriptor_path = pathlib.Path(descriptor_raw)
repo_root = pathlib.Path(repo_root_raw)
raw = descriptor_path.read_bytes()
actual_descriptor_sha = hashlib.sha256(raw).hexdigest()
if actual_descriptor_sha != descriptor_sha:
    raise SystemExit(
        "explicit descriptor SHA-256 mismatch: "
        f"expected {descriptor_sha}, got {actual_descriptor_sha}"
    )
try:
    value = json.loads(
        raw,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value: {token}")
        ),
    )
except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
    raise SystemExit(f"descriptor is not valid finite JSON: {error}") from error
canonical = json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
).encode() + b"\n"
if raw != canonical:
    raise SystemExit("descriptor is not exact canonical JSON plus newline")

sys.path.insert(0, str(repo_root / "src"))
from mlx_vq.quality.glm52_sky_campaign import (  # noqa: E402
    validate_sky_campaign_descriptor,
)

descriptor = validate_sky_campaign_descriptor(value)
if descriptor["bucket"] != descriptor_bucket:
    raise SystemExit("descriptor bucket does not match its explicit URI")
if descriptor["campaign_descriptor_key"] != descriptor_key:
    raise SystemExit("descriptor key does not match its explicit URI")
if descriptor["repo_tar_key"] != repository_key:
    raise SystemExit("repository key does not match the descriptor")
if descriptor["repo_tar_sha256"] != repository_sha:
    raise SystemExit("repository SHA-256 does not match the descriptor")
if (
    descriptor["artifacts"]["qualification_cache_manifest_sha256"]
    == "0" * 64
):
    raise SystemExit("post-seed descriptor still contains cache placeholder")
PY
aws s3 cp "$REPO_TAR_URI" "$PREFLIGHT_WORK/repo.tar.gz" \
  --profile "$AWS_PROFILE" --region "$REGION" --only-show-errors

python3 - "$PREFLIGHT_WORK/campaign-descriptor-v2.json" \
  "$EXPECTED_DESCRIPTOR_SHA256" "$DESCRIPTOR_BUCKET" "$DESCRIPTOR_KEY" \
  "$PREFLIGHT_WORK/repo.tar.gz" "$EXPECTED_REPO_TAR_SHA256" "$REPO_KEY" \
  "$REPO_ROOT" <<'PY'
import hashlib
import json
import pathlib
import sys
import tarfile

(
    descriptor_raw,
    descriptor_sha,
    descriptor_bucket,
    descriptor_key,
    repository_raw,
    repository_sha,
    repository_key,
    repo_root_raw,
) = sys.argv[1:]
descriptor_path = pathlib.Path(descriptor_raw)
repository_path = pathlib.Path(repository_raw)
repo_root = pathlib.Path(repo_root_raw)

actual_descriptor_sha = hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
if actual_descriptor_sha != descriptor_sha:
    raise SystemExit(
        "explicit descriptor SHA-256 mismatch: "
        f"expected {descriptor_sha}, got {actual_descriptor_sha}"
    )
actual_repository_sha = hashlib.sha256(repository_path.read_bytes()).hexdigest()
if actual_repository_sha != repository_sha:
    raise SystemExit(
        "explicit repository SHA-256 mismatch: "
        f"expected {repository_sha}, got {actual_repository_sha}"
    )
try:
    value = json.loads(
        descriptor_path.read_bytes(),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value: {token}")
        ),
    )
except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
    raise SystemExit(f"descriptor is not valid finite JSON: {error}") from error
canonical = json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
).encode() + b"\n"
if descriptor_path.read_bytes() != canonical:
    raise SystemExit("descriptor is not exact canonical JSON plus newline")

sys.path.insert(0, str(repo_root / "src"))
from mlx_vq.quality.glm52_sky_campaign import (  # noqa: E402
    validate_sky_campaign_descriptor,
)

descriptor = validate_sky_campaign_descriptor(value)
if descriptor["bucket"] != descriptor_bucket:
    raise SystemExit("descriptor bucket does not match its explicit URI")
if descriptor["campaign_descriptor_key"] != descriptor_key:
    raise SystemExit("descriptor key does not match its explicit URI")
if descriptor["repo_tar_key"] != repository_key:
    raise SystemExit("repository key does not match the descriptor")
if descriptor["repo_tar_sha256"] != repository_sha:
    raise SystemExit("repository SHA-256 does not match the descriptor")
if (
    descriptor["artifacts"]["qualification_cache_manifest_sha256"]
    == "0" * 64
):
    raise SystemExit("post-seed descriptor still contains cache placeholder")

required = {
    "aws/glm52-gpu/skypilot/glm52-campaign.yaml",
    "aws/glm52-gpu/skypilot/bootstrap_campaign.sh",
    "aws/glm52-gpu/skypilot/prepare_nvme_storage.sh",
    "aws/glm52-gpu/skypilot/run_managed_campaign.sh",
    "aws/glm52-gpu/skypilot/keep-glm52-campaign.service",
    "aws/glm52-gpu/scripts/run_campaign.sh",
    "aws/glm52-gpu/scripts/manage_gpu_spend.py",
    "aws/glm52-gpu/scripts/verify_sky_terminal_state.py",
    "benchmarks/run_glm52_campaign.py",
}
members: set[str] = set()
try:
    with tarfile.open(repository_path, "r:gz") as archive:
        for member in archive.getmembers():
            name = member.name.removeprefix("./")
            path = pathlib.PurePosixPath(name)
            if (
                member.name.startswith("/")
                or ".." in path.parts
                or member.isdev()
            ):
                raise SystemExit(f"unsafe repository member: {member.name}")
            if member.issym() or member.islnk():
                target = pathlib.PurePosixPath(member.linkname)
                if member.linkname.startswith("/") or ".." in target.parts:
                    raise SystemExit(
                        f"unsafe repository link: {member.name}"
                    )
            if member.isfile():
                members.add(name)
except (tarfile.TarError, OSError) as error:
    raise SystemExit(f"repository tar is unreadable: {error}") from error
missing = sorted(required - members)
if missing:
    raise SystemExit(f"repository tar is missing required components: {missing}")
PY

export AWS_PROFILE REGION REHEARSAL_EVIDENCE_OUTPUT
env -u WORK "$STRICT_REHEARSAL" \
  --pinned-authority \
  --descriptor-uri "$DESCRIPTOR_URI" \
  --descriptor-file-sha256 "$EXPECTED_DESCRIPTOR_SHA256" \
  --repo-tar-uri "$REPO_TAR_URI" \
  --repo-tar-file-sha256 "$EXPECTED_REPO_TAR_SHA256"
"$STRICT_REHEARSAL" --validate-evidence "$REHEARSAL_EVIDENCE_OUTPUT"

if ! python3 - "$REHEARSAL_EVIDENCE_OUTPUT" \
  "$PREFLIGHT_WORK/campaign-descriptor-v2.json" \
  "$EXPECTED_DESCRIPTOR_SHA256" "$DESCRIPTOR_KEY" \
  "$EXPECTED_REPO_TAR_SHA256" <<'PY'
import hashlib
import json
import pathlib
import sys

evidence_raw, descriptor_raw, descriptor_sha, descriptor_key, repo_sha = (
    sys.argv[1:]
)
evidence_path = pathlib.Path(evidence_raw)
descriptor_path = pathlib.Path(descriptor_raw)
evidence = json.loads(evidence_path.read_bytes())
descriptor = json.loads(descriptor_path.read_bytes())
canonical = json.dumps(
    evidence,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
).encode() + b"\n"
if evidence_path.read_bytes() != canonical:
    raise SystemExit("final rehearsal evidence is not canonical")
expected = {
    "schema_version": 2,
    "record_type": "glm52_staged_control_plane_rehearsal_v2",
    "status": "passed_before_cuda_h100_boundary",
    "run_id": descriptor["run_id"],
    "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
    "descriptor_key": descriptor_key,
    "descriptor_file_sha256": descriptor_sha,
    "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
    "repo_tar_sha256": repo_sha,
}
for field, value in expected.items():
    if evidence.get(field) != value:
        raise SystemExit(f"final rehearsal evidence {field} pin mismatch")
if hashlib.sha256(descriptor_path.read_bytes()).hexdigest() != descriptor_sha:
    raise SystemExit("explicit descriptor bytes drifted during rehearsal")
PY
then
  rm -f -- "$REHEARSAL_EVIDENCE_OUTPUT"
  fail "final rehearsal evidence did not preserve explicit input pins"
fi
WRAPPER_SUCCEEDED=1
