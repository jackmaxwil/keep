#!/usr/bin/env bash
# Rehearse the exact S3-staged descriptor and repository in a clean directory.
set -euo pipefail

if [ "${1:-}" = "--write-evidence" ] || [ "${1:-}" = "--validate-evidence" ]; then
  python3 - "$@" <<'PY'
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
from datetime import datetime, timezone


SHA256 = re.compile(r"^[0-9a-f]{64}$")
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
PRODUCTION_REPO = "/opt/keep-campaign/repo"
PRODUCTION_RESUME_ROOT = "/mnt/nvme/glm52-campaign"
TASK_RELATIVE = pathlib.PurePosixPath(
    "aws/glm52-gpu/skypilot/glm52-campaign.yaml"
)
EVIDENCE_FIELDS = {
    "schema_version",
    "record_type",
    "status",
    "run_id",
    "campaign_identity_sha256",
    "descriptor_key",
    "descriptor_file_sha256",
    "descriptor_body_sha256",
    "repo_tar_sha256",
    "bootstrap_receipt_file_sha256",
    "staged_readiness_key",
    "staged_readiness_file_sha256",
    "staged_readiness_body_sha256",
    "artifact_inventory_key",
    "artifact_inventory_file_sha256",
    "artifact_inventory_body_sha256",
    "artifact_audit_key",
    "artifact_audit_file_sha256",
    "extracted_repo_path",
    "production_repo_path",
    "production_resume_root",
    "skypilot_task_path",
    "skypilot_task_file_sha256",
    "skypilot_config_file_sha256",
    "skypilot_validation_file_sha256",
    "skypilot_version",
    "skypilot_task_name",
    "completed_at",
    "rehearsal_body_sha256",
}
STAGED_READINESS_FIELDS = {
    "schema_version",
    "record_type",
    "run_id",
    "descriptor_key",
    "descriptor_sha256",
    "descriptor_body_sha256",
    "campaign_identity_sha256",
    "bundle_manifest_key",
    "bundle_manifest_file_sha256",
    "bundle_manifest_body_sha256",
    "bundle_manifest_version_id",
    "staged_object_version_ids",
    "artifact_audit_key",
    "artifact_audit_sha256",
    "staged_at",
    "ready_body_sha256",
}
STAGED_VERSION_ROLES = {
    "repository_tar",
    "approval",
    "training_config",
    "watchdog",
    "artifact_inventory",
    "artifact_audit",
    "descriptor",
}


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: pathlib.Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_bytes(),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {token}")
            ),
        )
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def require_sha(value: object, label: str) -> str:
    result = require_string(value, label)
    if SHA256.fullmatch(result) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return result


def require_version(value: object, label: str) -> str:
    result = require_string(value, label)
    if result == "null":
        raise ValueError(f"{label} must be a non-null S3 VersionId")
    return result


def require_key(value: object, label: str) -> str:
    result = require_string(value, label)
    if (
        result.startswith("/")
        or result.endswith("/")
        or any(not 0x21 <= ord(character) <= 0x7E for character in result)
        or "\\" in result
        or any(character in result for character in "*?[]")
        or any(part in {"", ".", ".."} for part in result.split("/"))
    ):
        raise ValueError(f"{label} must be a safe ASCII S3 key")
    return result


def require_timestamp(
    value: object,
    label: str,
    *,
    whole_second: bool = False,
) -> str:
    result = require_string(value, label)
    if not result.endswith("Z"):
        raise ValueError(f"{label} must use canonical UTC Z form")
    try:
        parsed = datetime.fromisoformat(result[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{label} is not a valid timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{label} must be UTC")
    if whole_second and parsed.microsecond != 0:
        raise ValueError(f"{label} must use whole seconds")
    canonical_timestamp = parsed.astimezone(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    if canonical_timestamp != result:
        raise ValueError(f"{label} is not canonical")
    return result


def body_sha(value: dict[str, object], field: str) -> str:
    return sha_bytes(
        canonical({key: item for key, item in value.items() if key != field})
    )


def validate_evidence(value: dict[str, object]) -> dict[str, object]:
    if set(value) != EVIDENCE_FIELDS:
        missing = sorted(EVIDENCE_FIELDS - set(value))
        unknown = sorted(set(value) - EVIDENCE_FIELDS)
        raise ValueError(
            f"rehearsal evidence field inventory mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    if type(value["schema_version"]) is not int or value["schema_version"] != 2:
        raise ValueError("rehearsal evidence schema_version must be integer 2")
    if value["record_type"] != "glm52_staged_control_plane_rehearsal_v2":
        raise ValueError("rehearsal evidence record_type mismatch")
    if value["status"] != "passed_before_cuda_h100_boundary":
        raise ValueError("rehearsal evidence status mismatch")
    run_id = require_string(value["run_id"], "run_id")
    if RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run_id is invalid")
    for field in (
        "campaign_identity_sha256",
        "descriptor_file_sha256",
        "descriptor_body_sha256",
        "repo_tar_sha256",
        "bootstrap_receipt_file_sha256",
        "staged_readiness_file_sha256",
        "staged_readiness_body_sha256",
        "artifact_inventory_file_sha256",
        "artifact_inventory_body_sha256",
        "artifact_audit_file_sha256",
        "skypilot_task_file_sha256",
        "skypilot_config_file_sha256",
        "skypilot_validation_file_sha256",
        "rehearsal_body_sha256",
    ):
        require_sha(value[field], field)
    for field in (
        "descriptor_key",
        "staged_readiness_key",
        "artifact_inventory_key",
        "artifact_audit_key",
        "extracted_repo_path",
        "skypilot_task_path",
        "skypilot_task_name",
    ):
        require_string(value[field], field)
    if not pathlib.PurePosixPath(str(value["extracted_repo_path"])).is_absolute():
        raise ValueError("extracted_repo_path must be absolute")
    if value["production_repo_path"] != PRODUCTION_REPO:
        raise ValueError("production_repo_path mismatch")
    if value["production_resume_root"] != PRODUCTION_RESUME_ROOT:
        raise ValueError("production_resume_root mismatch")
    expected_task = f"{PRODUCTION_REPO}/{TASK_RELATIVE.as_posix()}"
    if value["skypilot_task_path"] != expected_task:
        raise ValueError("skypilot_task_path mismatch")
    if value["skypilot_version"] != "0.13.0":
        raise ValueError("skypilot_version mismatch")
    require_timestamp(value["completed_at"], "completed_at")
    expected = body_sha(value, "rehearsal_body_sha256")
    if value["rehearsal_body_sha256"] != expected:
        raise ValueError("rehearsal evidence body SHA-256 mismatch")
    return value


def atomic_write(path: pathlib.Path, value: dict[str, object]) -> None:
    raw = canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def command_write(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="--write-evidence")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--bootstrap-receipt", type=pathlib.Path, required=True)
    parser.add_argument("--descriptor", type=pathlib.Path, required=True)
    parser.add_argument("--descriptor-key", required=True)
    parser.add_argument("--repo-tar", type=pathlib.Path, required=True)
    parser.add_argument("--readiness", type=pathlib.Path, required=True)
    parser.add_argument("--readiness-key", required=True)
    parser.add_argument("--inventory", type=pathlib.Path, required=True)
    parser.add_argument("--audit", type=pathlib.Path, required=True)
    parser.add_argument("--extracted-repo", type=pathlib.Path, required=True)
    parser.add_argument("--skypilot-task", type=pathlib.Path, required=True)
    parser.add_argument("--skypilot-config", type=pathlib.Path, required=True)
    parser.add_argument("--skypilot-validation", type=pathlib.Path, required=True)
    parser.add_argument("--skypilot-python", type=pathlib.Path, required=True)
    parser.add_argument("--skypilot-validator", type=pathlib.Path, required=True)
    parser.add_argument("--completed-at", required=True)
    args = parser.parse_args(arguments)

    descriptor = load_object(args.descriptor, "campaign descriptor")
    readiness = load_object(args.readiness, "staged readiness marker")
    inventory = load_object(args.inventory, "artifact inventory")
    audit = load_object(args.audit, "artifact audit")
    try:
        bootstrap_lines = args.bootstrap_receipt.read_text().splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"bootstrap receipt is unreadable: {error}") from error
    if (
        not bootstrap_lines
        or bootstrap_lines[-1] != "SKYPILOT-BOOTSTRAP-REHEARSAL-OK"
        or bootstrap_lines.count("SKYPILOT-BOOTSTRAP-REHEARSAL-OK") != 1
    ):
        raise ValueError("bootstrap receipt does not prove the rehearsal boundary")
    bootstrap_receipt_sha = sha_file(args.bootstrap_receipt)
    descriptor_file_sha = sha_file(args.descriptor)
    repo_tar_sha = sha_file(args.repo_tar)
    inventory_file_sha = sha_file(args.inventory)
    audit_file_sha = sha_file(args.audit)

    run_id = require_string(descriptor.get("run_id"), "descriptor run_id")
    if descriptor.get("record_type") != "glm52_sky_campaign_descriptor_v2":
        raise ValueError("campaign descriptor record_type mismatch")
    if descriptor.get("campaign_descriptor_key") != args.descriptor_key:
        raise ValueError("campaign descriptor key mismatch")
    if descriptor.get("repo_tar_sha256") != repo_tar_sha:
        raise ValueError("repository tar identity mismatch")
    descriptor_body_sha = require_sha(
        descriptor.get("descriptor_body_sha256"), "descriptor_body_sha256"
    )
    if descriptor_body_sha != body_sha(descriptor, "descriptor_body_sha256"):
        raise ValueError("campaign descriptor body SHA-256 mismatch")
    campaign_identity = require_sha(
        descriptor.get("campaign_identity_sha256"), "campaign_identity_sha256"
    )
    artifacts = descriptor.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("campaign descriptor artifacts must be an object")
    inventory_key = require_string(
        artifacts.get("artifact_inventory_key"), "artifact_inventory_key"
    )
    if artifacts.get("artifact_inventory_sha256") != inventory_file_sha:
        raise ValueError("artifact inventory file identity mismatch")

    if set(readiness) != STAGED_READINESS_FIELDS:
        raise ValueError("staged readiness field inventory mismatch")
    if (
        type(readiness.get("schema_version")) is not int
        or readiness.get("schema_version") != 2
        or readiness.get("record_type")
        != "glm52_staged_control_plane_ready_v2"
    ):
        raise ValueError("staged readiness schema mismatch")
    if args.readiness.read_bytes() != canonical(readiness) + b"\n":
        raise ValueError("staged readiness is not canonical JSON plus LF")
    expected_readiness_key = (
        args.descriptor_key.removesuffix("campaign-descriptor-v2.json")
        + "STAGED_CONTROL_PLANE_READY.json"
    )
    if args.readiness_key != expected_readiness_key:
        raise ValueError("staged readiness key mismatch")
    if readiness.get("run_id") != run_id:
        raise ValueError("staged readiness run identity mismatch")
    if readiness.get("descriptor_key") != args.descriptor_key:
        raise ValueError("staged readiness descriptor key mismatch")
    if readiness.get("descriptor_sha256") != descriptor_file_sha:
        raise ValueError("readiness descriptor file identity mismatch")
    if readiness.get("descriptor_body_sha256") != descriptor_body_sha:
        raise ValueError("readiness descriptor body identity mismatch")
    if readiness.get("campaign_identity_sha256") != campaign_identity:
        raise ValueError("readiness campaign identity mismatch")
    for field in (
        "descriptor_sha256",
        "descriptor_body_sha256",
        "campaign_identity_sha256",
        "bundle_manifest_file_sha256",
        "bundle_manifest_body_sha256",
        "artifact_audit_sha256",
        "ready_body_sha256",
    ):
        require_sha(readiness.get(field), field)
    manifest_body_sha = str(readiness["bundle_manifest_body_sha256"])
    expected_manifest_key = (
        args.descriptor_key.removesuffix("campaign-descriptor-v2.json")
        + f"bundle-manifests/{manifest_body_sha}/bundle-manifest-v1.json"
    )
    if (
        require_key(
            readiness.get("bundle_manifest_key"),
            "bundle_manifest_key",
        )
        != expected_manifest_key
    ):
        raise ValueError("staged readiness bundle manifest key mismatch")
    require_version(
        readiness.get("bundle_manifest_version_id"),
        "bundle_manifest_version_id",
    )
    versions = readiness.get("staged_object_version_ids")
    if not isinstance(versions, dict) or set(versions) != STAGED_VERSION_ROLES:
        raise ValueError("staged readiness VersionId map mismatch")
    for role in STAGED_VERSION_ROLES:
        require_version(versions.get(role), f"{role} VersionId")
    require_timestamp(
        readiness.get("staged_at"),
        "staged_at",
        whole_second=True,
    )
    readiness_body_sha = require_sha(
        readiness.get("ready_body_sha256"), "ready_body_sha256"
    )
    if readiness_body_sha != body_sha(readiness, "ready_body_sha256"):
        raise ValueError("staged readiness body SHA-256 mismatch")
    audit_key = require_string(
        readiness.get("artifact_audit_key"), "artifact_audit_key"
    )
    if readiness.get("artifact_audit_sha256") != audit_file_sha:
        raise ValueError("artifact audit file identity mismatch")

    if inventory.get("run_id") != run_id:
        raise ValueError("artifact inventory run identity mismatch")
    inventory_body_sha = require_sha(
        inventory.get("inventory_body_sha256"), "inventory_body_sha256"
    )
    if inventory_body_sha != body_sha(inventory, "inventory_body_sha256"):
        raise ValueError("artifact inventory body SHA-256 mismatch")
    if (
        audit.get("record_type") != "glm52_s3_artifact_audit_v1"
        or audit.get("audit_pass") is not True
        or audit.get("run_id") != run_id
        or audit.get("inventory_body_sha256") != inventory_body_sha
    ):
        raise ValueError("artifact audit authority mismatch")

    extracted_repo = args.extracted_repo.resolve(strict=True)
    if not extracted_repo.is_dir():
        raise ValueError("extracted repository path is not a directory")
    task_path = args.skypilot_task.resolve(strict=True)
    expected_task_path = extracted_repo / TASK_RELATIVE
    if task_path != expected_task_path:
        raise ValueError("SkyPilot task is outside the exact extracted repository")
    if not args.skypilot_config.is_file():
        raise ValueError("rendered SkyPilot configuration is missing")
    if (
        not args.skypilot_python.is_file()
        or not os.access(args.skypilot_python, os.X_OK)
        or not args.skypilot_validator.is_file()
    ):
        raise ValueError("SkyPilot revalidation command is unavailable")
    task_sha = sha_file(task_path)
    config_sha = sha_file(args.skypilot_config)
    revalidated = subprocess.run(
        [
            str(args.skypilot_python),
            str(args.skypilot_validator),
            "--task",
            str(task_path),
            "--config",
            str(args.skypilot_config),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if revalidated.returncode != 0:
        detail = revalidated.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(
            "exact SkyPilot task/config revalidation failed"
            + (f": {detail}" if detail else "")
        )
    if (
        sha_file(task_path) != task_sha
        or sha_file(args.skypilot_config) != config_sha
    ):
        raise ValueError("SkyPilot task/config bytes drifted during revalidation")
    if revalidated.stdout != args.skypilot_validation.read_bytes():
        raise ValueError(
            "SkyPilot validation receipt does not match exact final revalidation"
        )
    validation = load_object(args.skypilot_validation, "SkyPilot validation")
    if (
        validation.get("record_type")
        != "glm52_skypilot_parser_validation_v1"
        or validation.get("status") != "passed"
        or validation.get("skypilot_version") != "0.13.0"
        or validation.get("skypilot_version")
        != descriptor.get("skypilot_version")
        or validation.get("task") != descriptor.get("task_name")
        or validation.get("instance_type") != descriptor.get("instance_type")
        or validation.get("region") != descriptor.get("region")
    ):
        raise ValueError("SkyPilot task/config validation did not pass")
    task_name = require_string(validation.get("task"), "SkyPilot task name")
    completed_at = require_timestamp(args.completed_at, "completed_at")

    result: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_staged_control_plane_rehearsal_v2",
        "status": "passed_before_cuda_h100_boundary",
        "run_id": run_id,
        "campaign_identity_sha256": campaign_identity,
        "descriptor_key": args.descriptor_key,
        "descriptor_file_sha256": descriptor_file_sha,
        "descriptor_body_sha256": descriptor_body_sha,
        "repo_tar_sha256": repo_tar_sha,
        "bootstrap_receipt_file_sha256": bootstrap_receipt_sha,
        "staged_readiness_key": args.readiness_key,
        "staged_readiness_file_sha256": sha_file(args.readiness),
        "staged_readiness_body_sha256": readiness_body_sha,
        "artifact_inventory_key": inventory_key,
        "artifact_inventory_file_sha256": inventory_file_sha,
        "artifact_inventory_body_sha256": inventory_body_sha,
        "artifact_audit_key": audit_key,
        "artifact_audit_file_sha256": audit_file_sha,
        "extracted_repo_path": str(extracted_repo),
        "production_repo_path": PRODUCTION_REPO,
        "production_resume_root": PRODUCTION_RESUME_ROOT,
        "skypilot_task_path": f"{PRODUCTION_REPO}/{TASK_RELATIVE.as_posix()}",
        "skypilot_task_file_sha256": task_sha,
        "skypilot_config_file_sha256": config_sha,
        "skypilot_validation_file_sha256": sha_file(args.skypilot_validation),
        "skypilot_version": "0.13.0",
        "skypilot_task_name": task_name,
        "completed_at": completed_at,
    }
    result["rehearsal_body_sha256"] = body_sha(
        result, "rehearsal_body_sha256"
    )
    validate_evidence(result)
    atomic_write(args.output, result)
    try:
        written = load_object(args.output, "written rehearsal evidence")
        validate_evidence(written)
        if args.output.read_bytes() != canonical(written) + b"\n":
            raise ValueError("written rehearsal evidence is not canonical")
    except Exception:
        args.output.unlink(missing_ok=True)
        raise
    print(f"REHEARSAL_EVIDENCE_PATH={args.output.resolve()}")
    print(f"REHEARSAL_EVIDENCE_SHA256={sha_file(args.output)}")
    print(f"REHEARSAL_EVIDENCE_BODY_SHA256={result['rehearsal_body_sha256']}")
    return 0


def command_validate(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="--validate-evidence")
    parser.add_argument("evidence", type=pathlib.Path)
    args = parser.parse_args(arguments)
    value = load_object(args.evidence, "rehearsal evidence")
    validate_evidence(value)
    if args.evidence.read_bytes() != canonical(value) + b"\n":
        raise ValueError("rehearsal evidence is not canonical")
    print(
        f"REHEARSAL_EVIDENCE_VALID="
        f"{value['rehearsal_body_sha256']}"
    )
    return 0


try:
    command = sys.argv[1]
    if command == "--write-evidence":
        raise SystemExit(command_write(sys.argv[2:]))
    if command == "--validate-evidence":
        raise SystemExit(command_validate(sys.argv[2:]))
    raise ValueError(f"unknown evidence command: {command}")
except (OSError, ValueError) as error:
    print(f"rehearsal evidence error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY
  exit $?
fi

SCRIPT_SOURCE=${BASH_SOURCE[0]}
SCRIPT_PARENT=${SCRIPT_SOURCE%/*}
if [ "$SCRIPT_PARENT" = "$SCRIPT_SOURCE" ]; then
  SCRIPT_PARENT=.
fi
SCRIPT_DIR=$(CDPATH= cd -- "$SCRIPT_PARENT" && pwd -P)
SCRIPT_PATH="$SCRIPT_DIR/${SCRIPT_SOURCE##*/}"

run_bootstrap_boundary() {
  local receipt=$1
  shift
  mkdir -p "$(dirname -- "$receipt")"
  "$@" | tee "$receipt"
  grep -Fqx 'SKYPILOT-BOOTSTRAP-REHEARSAL-OK' "$receipt"
}

evidence_output_is_within_work() {
  python3 - "$WORK" "$EVIDENCE_OUTPUT" <<'PY'
import os, pathlib, sys
work = pathlib.Path(sys.argv[1]).resolve()
evidence = pathlib.Path(sys.argv[2]).resolve()
raise SystemExit(0 if os.path.commonpath((work, evidence)) == str(work) else 1)
PY
}

cleanup_rehearsal() {
  if [ "$KEEP_WORK" = 1 ]; then
    return
  fi
  if [ "$REHEARSAL_SUCCEEDED" = 1 ] \
    && [ -f "$EVIDENCE_OUTPUT" ] \
    && evidence_output_is_within_work; then
    return
  fi
  rm -rf "$WORK"
}

if [ "${1:-}" = "--exercise-finalization" ]; then
  shift
  if [ "${1:-}" != "--bootstrap-command" ] || [ -z "${2:-}" ]; then
    echo "usage: $0 --exercise-finalization --bootstrap-command PATH -- WRITER_ARGS" >&2
    exit 64
  fi
  BOOTSTRAP_COMMAND=$2
  shift 2
  if [ "${1:-}" != "--" ]; then
    echo "missing -- before evidence writer arguments" >&2
    exit 64
  fi
  shift
  WORK="${WORK:-$(mktemp -d)}"
  KEEP_WORK="${KEEP_REHEARSAL_WORK:-0}"
  EVIDENCE_OUTPUT="${REHEARSAL_EVIDENCE_OUTPUT:?set REHEARSAL_EVIDENCE_OUTPUT}"
  REHEARSAL_SUCCEEDED=0
  trap cleanup_rehearsal EXIT
  BOOTSTRAP_OUTPUT="$WORK/bootstrap-rehearsal.log"
  run_bootstrap_boundary "$BOOTSTRAP_OUTPUT" "$BOOTSTRAP_COMMAND"
  "$SCRIPT_PATH" --write-evidence "$@" \
    --bootstrap-receipt "$BOOTSTRAP_OUTPUT"
  "$SCRIPT_PATH" --validate-evidence "$EVIDENCE_OUTPUT"
  REHEARSAL_SUCCEEDED=1
  exit 0
fi

fail() {
  echo "$*" >&2
  exit 64
}

production_usage() {
  fail "usage: $0 --pinned-authority --descriptor-uri S3_URI --descriptor-file-sha256 SHA256 --repo-tar-uri S3_URI --repo-tar-file-sha256 SHA256 | --legacy-active-parameter /keep-glm52/campaign/active"
}

parse_safe_s3_uri() {
  local uri=$1
  local label=$2
  local remainder bucket key segment
  local -a parts

  case "$uri" in
    s3://*) remainder=${uri#s3://} ;;
    *) fail "$label must use an exact s3://bucket/key URI" ;;
  esac
  case "$remainder" in
    */*)
      bucket=${remainder%%/*}
      key=${remainder#*/}
      ;;
    *) fail "$label must contain an exact object key" ;;
  esac
  if [[ ! "$bucket" =~ ^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$ ]] \
    || [[ "$bucket" == *..* ]] \
    || [[ "$bucket" == *.-* ]] \
    || [[ "$bucket" == *-.* ]] \
    || [[ "$bucket" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    fail "$label bucket is not a safe DNS-style bucket name"
  fi
  case "$bucket" in
    xn--*|sthree-*|amzn-s3-demo-*|*-s3alias|*--ol-s3|*.mrap|*--x-s3|*--table-s3)
      fail "$label bucket uses a reserved DNS-style name"
      ;;
  esac
  if [ -z "$key" ] \
    || [[ "$key" == /* ]] \
    || [[ "$key" == */ ]] \
    || [[ "$key" == *//* ]] \
    || [[ "$key" == *\\* ]] \
    || [[ "$key" == *\** ]] \
    || [[ "$key" == *\?* ]] \
    || [[ "$key" == *\[* ]] \
    || [[ "$key" == *\]* ]] \
    || [[ "$key" =~ [[:space:][:cntrl:]] ]]; then
    fail "$label key is not exact and safe"
  fi
  IFS=/ read -r -a parts <<< "$key"
  for segment in "${parts[@]}"; do
    if [[ ! "$segment" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
      fail "$label key contains an unsafe segment"
    fi
  done
  SAFE_URI_BUCKET=$bucket
  SAFE_URI_KEY=$key
  SAFE_URI_PARTS=("${parts[@]}")
}

parse_descriptor_uri() {
  parse_safe_s3_uri "$1" "descriptor URI"
  if [ "${#SAFE_URI_PARTS[@]}" -ne 5 ] \
    || [ "${SAFE_URI_PARTS[0]}" != "campaigns" ] \
    || [ "${SAFE_URI_PARTS[2]}" != "submissions" ] \
    || [ "${SAFE_URI_PARTS[4]}" != "campaign-descriptor-v2.json" ] \
    || [[ ! "${SAFE_URI_PARTS[1]}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] \
    || [[ ! "${SAFE_URI_PARTS[3]}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
    fail "descriptor URI must name one campaign submission descriptor"
  fi
  DESCRIPTOR_BUCKET=$SAFE_URI_BUCKET
  DESCRIPTOR_KEY=$SAFE_URI_KEY
  DESCRIPTOR_RUN_ID=${SAFE_URI_PARTS[1]}
  DESCRIPTOR_SUBMISSION_ID=${SAFE_URI_PARTS[3]}
}

parse_repository_uri() {
  parse_safe_s3_uri "$1" "repository URI"
  if [ "${#SAFE_URI_PARTS[@]}" -ne 4 ] \
    || [ "${SAFE_URI_PARTS[0]}" != "campaigns" ] \
    || [ "${SAFE_URI_PARTS[2]}" != "repository" ] \
    || [[ ! "${SAFE_URI_PARTS[1]}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] \
    || [[ "${SAFE_URI_PARTS[3]}" != *.tar.gz ]]; then
    fail "repository URI must name one campaign repository tar"
  fi
  REPOSITORY_BUCKET=$SAFE_URI_BUCKET
  REPOSITORY_KEY=$SAFE_URI_KEY
  REPOSITORY_RUN_ID=${SAFE_URI_PARTS[1]}
}

MODE=
if [ "$#" -eq 9 ] \
  && [ "$1" = "--pinned-authority" ] \
  && [ "$2" = "--descriptor-uri" ] \
  && [ "$4" = "--descriptor-file-sha256" ] \
  && [ "$6" = "--repo-tar-uri" ] \
  && [ "$8" = "--repo-tar-file-sha256" ]; then
  MODE=pinned
  DESCRIPTOR_URI=$3
  EXPECTED_DESCRIPTOR_SHA256=$5
  REPO_TAR_URI=$7
  EXPECTED_REPO_TAR_SHA256=$9
  if [[ ! "$EXPECTED_DESCRIPTOR_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    fail "descriptor file SHA-256 must be lowercase and exact"
  fi
  if [[ ! "$EXPECTED_REPO_TAR_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    fail "repository tar file SHA-256 must be lowercase and exact"
  fi
  parse_descriptor_uri "$DESCRIPTOR_URI"
  parse_repository_uri "$REPO_TAR_URI"
  if [ "$DESCRIPTOR_BUCKET" != "$REPOSITORY_BUCKET" ] \
    || [ "$DESCRIPTOR_RUN_ID" != "$REPOSITORY_RUN_ID" ]; then
    fail "descriptor and repository pins must share one bucket and run"
  fi
elif [ "$#" -eq 2 ] \
  && [ "$1" = "--legacy-active-parameter" ] \
  && [ "$2" = "/keep-glm52/campaign/active" ]; then
  MODE=legacy
  PARAMETER=$2
else
  production_usage
fi

if [ "${AWS_PROFILE:-}" != "keep-gpu" ]; then
  fail "AWS_PROFILE must be exactly keep-gpu"
fi
if [ "${REGION:-}" != "us-west-2" ]; then
  fail "REGION must be exactly us-west-2"
fi
export AWS_PROFILE REGION
"$SCRIPT_DIR/assert_rnd_aws_account.sh" >/dev/null

WORK="${WORK:-$(mktemp -d)}"
KEEP_WORK="${KEEP_REHEARSAL_WORK:-0}"
if [ -n "${REHEARSAL_EVIDENCE_OUTPUT:-}" ]; then
  EVIDENCE_OUTPUT="$REHEARSAL_EVIDENCE_OUTPUT"
else
  EVIDENCE_OUTPUT="$WORK/GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
fi
REHEARSAL_SUCCEEDED=0
trap cleanup_rehearsal EXIT

mkdir -p "$WORK/repo" "$WORK/root" "$WORK/authorities"
if [ "$MODE" = pinned ]; then
  aws s3 cp "$DESCRIPTOR_URI" "$WORK/descriptor.json" \
    --profile "$AWS_PROFILE" --region "$REGION" --only-show-errors
  python3 - "$WORK/descriptor.json" "$EXPECTED_DESCRIPTOR_SHA256" \
    "$DESCRIPTOR_BUCKET" "$DESCRIPTOR_KEY" "$DESCRIPTOR_RUN_ID" \
    "$REPOSITORY_KEY" "$EXPECTED_REPO_TAR_SHA256" \
    "$SCRIPT_DIR/../../../src/mlx_vq/quality/glm52_sky_campaign.py" <<'PY'
import hashlib
import importlib.util
import json
import pathlib
import sys

(
    descriptor_raw,
    expected_file_sha,
    expected_bucket,
    expected_key,
    expected_run_id,
    expected_repo_key,
    expected_repo_sha,
    sky_module_raw,
) = sys.argv[1:]
path = pathlib.Path(descriptor_raw)
raw = path.read_bytes()
actual_file_sha = hashlib.sha256(raw).hexdigest()
if actual_file_sha != expected_file_sha:
    raise SystemExit(
        "explicit descriptor SHA-256 mismatch: "
        f"expected {expected_file_sha}, got {actual_file_sha}"
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
if not isinstance(value, dict):
    raise SystemExit("descriptor must contain one JSON object")
canonical = json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
).encode() + b"\n"
if raw != canonical:
    raise SystemExit("descriptor is not exact canonical JSON plus newline")
body = dict(value)
body_sha = body.pop("descriptor_body_sha256", None)
if body_sha != hashlib.sha256(
    json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
).hexdigest():
    raise SystemExit("descriptor body SHA-256 mismatch")
expected = {
    "record_type": "glm52_sky_campaign_descriptor_v2",
    "bucket": expected_bucket,
    "campaign_descriptor_key": expected_key,
    "run_id": expected_run_id,
    "repo_tar_key": expected_repo_key,
    "repo_tar_sha256": expected_repo_sha,
}
for field, expected_value in expected.items():
    if value.get(field) != expected_value:
        raise SystemExit(f"descriptor {field} does not match explicit pin")
spec = importlib.util.spec_from_file_location(
    "_glm52_pre_repository_rehearsal_descriptor",
    sky_module_raw,
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)
module.validate_sky_campaign_descriptor(value)
PY
else
  aws ssm get-parameter --name "$PARAMETER" \
    --profile "$AWS_PROFILE" --region "$REGION" \
    --query Parameter.Value --output text > "$WORK/descriptor.json"
fi

read -r BUCKET DESCRIPTOR_KEY REPO_KEY REPO_SHA BASELINE_PREFIX \
  APPROVAL_KEY INVENTORY_KEY <<EOF
$(python3 - "$WORK/descriptor.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
print(
    value["bucket"], value["campaign_descriptor_key"], value["repo_tar_key"],
    value["repo_tar_sha256"], value["artifacts"]["training_baseline_prefix"],
    value["approval_key"], value["artifacts"]["artifact_inventory_key"],
)
PY
)
EOF

SUBMISSION_PREFIX="${DESCRIPTOR_KEY%/campaign-descriptor-v2.json}"
READINESS_KEY="$SUBMISSION_PREFIX/STAGED_CONTROL_PLANE_READY.json"
READINESS_HEAD=$(
  aws s3api head-object --bucket "$BUCKET" --key "$READINESS_KEY" \
    --profile "$AWS_PROFILE" --region "$REGION" --output json
)
READINESS_VERSION=$(
  python3 - "$READINESS_HEAD" <<'PY'
import json, sys
value = json.loads(sys.argv[1])
version = value.get("VersionId")
if not isinstance(version, str) or version in {"", "null"}:
    raise SystemExit("staged readiness HEAD requires a non-null VersionId")
size = value.get("ContentLength")
if type(size) is not int or size < 0:
    raise SystemExit("staged readiness HEAD returned an invalid size")
print(version)
PY
)

exact_get() {
  local key=$1
  local version=$2
  local destination=$3
  local label=$4
  local response
  response=$(
    aws s3api get-object --bucket "$BUCKET" --key "$key" \
      --version-id "$version" --profile "$AWS_PROFILE" --region "$REGION" \
      --output json "$destination"
  )
  python3 - "$response" "$version" "$label" <<'PY'
import json, sys
value = json.loads(sys.argv[1])
returned = value.get("VersionId")
if not isinstance(returned, str) or returned in {"", "null"}:
    raise SystemExit(f"{sys.argv[3]} GET omitted its VersionId")
if returned != sys.argv[2]:
    raise SystemExit(f"{sys.argv[3]} GET returned the wrong VersionId")
PY
}

exact_get "$READINESS_KEY" "$READINESS_VERSION" \
  "$WORK/STAGED_CONTROL_PLANE_READY.json" "staged readiness"
python3 - "$WORK/descriptor.json" \
  "$WORK/STAGED_CONTROL_PLANE_READY.json" \
  "$DESCRIPTOR_KEY" "$READINESS_KEY" "$WORK/readiness-pins.json" <<'PY'
import hashlib, json, pathlib, re, sys
from datetime import datetime, timezone

descriptor_path, readiness_path, descriptor_key, readiness_key, output = sys.argv[1:]
descriptor_raw = pathlib.Path(descriptor_path).read_bytes()
readiness_raw = pathlib.Path(readiness_path).read_bytes()
descriptor = json.loads(descriptor_raw)
readiness = json.loads(readiness_raw)
canonical = lambda value: json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    allow_nan=False,
).encode()
sha = lambda raw: hashlib.sha256(raw).hexdigest()
fields = {
    "schema_version", "record_type", "run_id", "descriptor_key",
    "descriptor_sha256", "descriptor_body_sha256",
    "campaign_identity_sha256", "bundle_manifest_key",
    "bundle_manifest_file_sha256", "bundle_manifest_body_sha256",
    "bundle_manifest_version_id", "staged_object_version_ids",
    "artifact_audit_key", "artifact_audit_sha256", "staged_at",
    "ready_body_sha256",
}
roles = {
    "repository_tar", "approval", "training_config", "watchdog",
    "artifact_inventory", "artifact_audit", "descriptor",
}
digest = re.compile(r"^[0-9a-f]{64}$")

def require_key(value, label):
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or value.endswith("/")
        or any(not 0x21 <= ord(character) <= 0x7E for character in value)
        or "\\" in value
        or any(character in value for character in "*?[]")
    ):
        raise SystemExit(f"{label} is not a safe S3 key")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise SystemExit(f"{label} is not a safe S3 key")
    return value

def require_version(value, label):
    if not isinstance(value, str) or value in {"", "null"}:
        raise SystemExit(f"{label} requires a non-null VersionId")
    return value

if set(readiness) != fields:
    raise SystemExit("staged readiness v2 field inventory mismatch")
if (
    type(readiness.get("schema_version")) is not int
    or readiness.get("schema_version") != 2
    or readiness.get("record_type") != "glm52_staged_control_plane_ready_v2"
):
    raise SystemExit("staged readiness v2 schema mismatch")
if readiness_raw != canonical(readiness) + b"\n":
    raise SystemExit("staged readiness is not canonical JSON plus LF")
for field in (
    "descriptor_sha256", "descriptor_body_sha256",
    "campaign_identity_sha256", "bundle_manifest_file_sha256",
    "bundle_manifest_body_sha256", "artifact_audit_sha256",
    "ready_body_sha256",
):
    if not isinstance(readiness.get(field), str) or not digest.fullmatch(
        readiness[field]
    ):
        raise SystemExit(f"staged readiness {field} is invalid")
body = dict(readiness)
ready_sha = body.pop("ready_body_sha256")
if ready_sha != sha(canonical(body)):
    raise SystemExit("staged readiness body identity mismatch")
staged_at = readiness.get("staged_at")
if not isinstance(staged_at, str) or not staged_at.endswith("Z"):
    raise SystemExit("staged readiness staged_at is not canonical UTC")
try:
    parsed_staged_at = datetime.fromisoformat(staged_at[:-1] + "+00:00")
except ValueError as error:
    raise SystemExit("staged readiness staged_at is not canonical UTC") from error
if (
    parsed_staged_at.tzinfo is None
    or parsed_staged_at.utcoffset() != timezone.utc.utcoffset(parsed_staged_at)
    or parsed_staged_at.microsecond != 0
    or parsed_staged_at.isoformat().replace("+00:00", "Z") != staged_at
):
    raise SystemExit("staged readiness staged_at is not canonical UTC")
if (
    readiness_key
    != descriptor_key.removesuffix("campaign-descriptor-v2.json")
    + "STAGED_CONTROL_PLANE_READY.json"
):
    raise SystemExit("staged readiness key authority mismatch")
bindings = {
    "run_id": descriptor["run_id"],
    "descriptor_key": descriptor_key,
    "descriptor_sha256": sha(descriptor_raw),
    "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
    "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
}
for field, expected in bindings.items():
    if readiness.get(field) != expected:
        raise SystemExit(f"staged readiness {field} authority mismatch")
manifest_key = require_key(
    readiness.get("bundle_manifest_key"), "bundle manifest key"
)
expected_manifest_key = (
    descriptor_key.removesuffix("campaign-descriptor-v2.json")
    + "bundle-manifests/"
    + readiness["bundle_manifest_body_sha256"]
    + "/bundle-manifest-v1.json"
)
if manifest_key != expected_manifest_key:
    raise SystemExit("staged readiness bundle manifest key mismatch")
versions = readiness.get("staged_object_version_ids")
if not isinstance(versions, dict) or set(versions) != roles:
    raise SystemExit("staged readiness VersionId map mismatch")
for role in roles:
    require_version(versions.get(role), f"{role} VersionId")
audit_key = require_key(
    readiness.get("artifact_audit_key"), "artifact audit key"
)
if audit_key != (
    f"campaigns/{descriptor['run_id']}/audits/artifact-audit-"
    f"{readiness['artifact_audit_sha256']}.json"
):
    raise SystemExit("staged readiness artifact audit key mismatch")
pins = {
    "bundle_manifest_key": manifest_key,
    "bundle_manifest_version_id": require_version(
        readiness.get("bundle_manifest_version_id"),
        "bundle manifest VersionId",
    ),
    "staged_object_version_ids": versions,
    "artifact_audit_key": audit_key,
}
pathlib.Path(output).write_bytes(canonical(pins) + b"\n")
PY

MANIFEST_KEY=$(
  python3 - "$WORK/readiness-pins.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["bundle_manifest_key"])
PY
)
MANIFEST_VERSION=$(
  python3 - "$WORK/readiness-pins.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["bundle_manifest_version_id"])
PY
)
exact_get "$MANIFEST_KEY" "$MANIFEST_VERSION" \
  "$WORK/bundle-manifest-v1.json" "bundle manifest"

python3 - "$WORK/descriptor.json" "$WORK/STAGED_CONTROL_PLANE_READY.json" \
  "$WORK/bundle-manifest-v1.json" "$BUCKET" "$WORK/manifest-entries.json" <<'PY'
import hashlib, json, pathlib, re, sys

descriptor_path, readiness_path, manifest_path, bucket, output = sys.argv[1:]
descriptor_raw = pathlib.Path(descriptor_path).read_bytes()
readiness = json.loads(pathlib.Path(readiness_path).read_bytes())
raw = pathlib.Path(manifest_path).read_bytes()
manifest = json.loads(raw)
canonical = lambda value: json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    allow_nan=False,
).encode()
sha = lambda value: hashlib.sha256(value).hexdigest()
digest = re.compile(r"^[0-9a-f]{64}$")
manifest_fields = {
    "schema_version", "record_type", "run_id", "bucket", "files",
    "descriptor_body_sha256", "bundle_manifest_body_sha256",
}
file_fields = {"local_name", "key", "role", "stage_order", "size", "sha256"}
roles = {
    "repository_tar", "approval", "training_config", "watchdog",
    "artifact_inventory", "descriptor",
}
if (
    set(manifest) != manifest_fields
    or type(manifest.get("schema_version")) is not int
    or manifest.get("schema_version") != 1
    or manifest.get("record_type") != "glm52_sky_campaign_bundle_v1"
):
    raise SystemExit("remote bundle manifest schema mismatch")
if raw != canonical(manifest) + b"\n":
    raise SystemExit("remote bundle manifest is not canonical JSON plus LF")
body = dict(manifest)
body_sha = body.pop("bundle_manifest_body_sha256")
if not isinstance(body_sha, str) or not digest.fullmatch(body_sha):
    raise SystemExit("remote bundle manifest body SHA-256 is invalid")
if body_sha != sha(canonical(body)):
    raise SystemExit("remote bundle manifest body SHA-256 mismatch")
if (
    sha(raw) != readiness["bundle_manifest_file_sha256"]
    or body_sha != readiness["bundle_manifest_body_sha256"]
    or manifest.get("run_id") != readiness["run_id"]
    or manifest.get("bucket") != bucket
):
    raise SystemExit("remote bundle manifest readiness binding mismatch")
files = manifest.get("files")
if not isinstance(files, list) or len(files) != 6:
    raise SystemExit("remote bundle manifest role inventory mismatch")
if any(not isinstance(item, dict) or set(item) != file_fields for item in files):
    raise SystemExit("remote bundle manifest file schema mismatch")
if {item.get("role") for item in files} != roles:
    raise SystemExit("remote bundle manifest role inventory mismatch")
names = [item["local_name"] for item in files]
keys = [item["key"] for item in files]
orders = [item["stage_order"] for item in files]
for item in files:
    name = item["local_name"]
    key = item["key"]
    if (
        not isinstance(name, str) or pathlib.PurePosixPath(name).name != name
        or name in {"", ".", ".."} or "\\" in name
    ):
        raise SystemExit("remote bundle manifest local name is unsafe")
    if (
        not isinstance(key, str) or not key or key.startswith("/")
        or key.endswith("/")
        or any(not 0x21 <= ord(character) <= 0x7E for character in key)
        or "\\" in key
        or any(character in key for character in "*?[]")
        or any(part in {"", ".", ".."} for part in key.split("/"))
    ):
        raise SystemExit("remote bundle manifest key is unsafe")
    try:
        name.encode("ascii")
        key.encode("ascii")
    except UnicodeEncodeError as error:
        raise SystemExit("remote bundle manifest name or key is not ASCII") from error
    if type(item["size"]) is not int or item["size"] < 0:
        raise SystemExit("remote bundle manifest size is invalid")
    if not isinstance(item["sha256"], str) or not digest.fullmatch(item["sha256"]):
        raise SystemExit("remote bundle manifest SHA-256 is invalid")
    if type(item["stage_order"]) is not int or item["stage_order"] < 0:
        raise SystemExit("remote bundle manifest stage order is invalid")
if len(set(names)) != 6 or len(set(keys)) != 6 or len(set(orders)) != 6:
    raise SystemExit("remote bundle manifest entries are duplicated")
ordered = sorted(files, key=lambda item: item["stage_order"])
if ordered[-1]["role"] != "descriptor":
    raise SystemExit("remote bundle manifest descriptor is not last")
descriptor = json.loads(descriptor_raw)
descriptor_item = next(item for item in files if item["role"] == "descriptor")
if (
    manifest.get("descriptor_body_sha256")
    != descriptor["descriptor_body_sha256"]
    or descriptor_item["key"] != descriptor["campaign_descriptor_key"]
    or descriptor_item["size"] != len(descriptor_raw)
    or descriptor_item["sha256"] != sha(descriptor_raw)
):
    raise SystemExit("remote bundle manifest descriptor binding mismatch")
pathlib.Path(output).write_bytes(canonical({"files": files}) + b"\n")
PY

manifest_value() {
  python3 - "$WORK/manifest-entries.json" "$1" "$2" <<'PY'
import json, sys
files = json.load(open(sys.argv[1]))["files"]
item = next(value for value in files if value["role"] == sys.argv[2])
print(item[sys.argv[3]])
PY
}
role_version() {
  python3 - "$WORK/readiness-pins.json" "$1" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["staged_object_version_ids"][sys.argv[2]])
PY
}

for role in repository_tar approval training_config watchdog artifact_inventory descriptor; do
  case "$role" in
    repository_tar) destination="$WORK/repo.tar.gz" ;;
    approval) destination="$WORK/GPU_SPEND_APPROVAL.json" ;;
    training_config) destination="$WORK/training-config.json" ;;
    watchdog) destination="$WORK/watchdog.zip" ;;
    artifact_inventory) destination="$WORK/artifact-inventory-v1.json" ;;
    descriptor) destination="$WORK/staged-campaign-descriptor-v2.json" ;;
  esac
  role_key=$(manifest_value "$role" key)
  role_pin=$(role_version "$role")
  exact_get "$role_key" "$role_pin" "$destination" "$role"
done
AUDIT_KEY=$(
  python3 - "$WORK/readiness-pins.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["artifact_audit_key"])
PY
)
AUDIT_VERSION=$(role_version artifact_audit)
exact_get "$AUDIT_KEY" "$AUDIT_VERSION" \
  "$WORK/artifact-audit-v1.json" "artifact audit"

python3 - "$WORK/descriptor.json" "$WORK/staged-campaign-descriptor-v2.json" \
  "$WORK/STAGED_CONTROL_PLANE_READY.json" "$WORK/bundle-manifest-v1.json" \
  "$WORK/manifest-entries.json" "$WORK/GPU_SPEND_APPROVAL.json" \
  "$WORK/artifact-inventory-v1.json" "$WORK/artifact-audit-v1.json" \
  "$WORK/repo.tar.gz" "$REPO_KEY" "$REPO_SHA" "$APPROVAL_KEY" \
  "$INVENTORY_KEY" "$AUDIT_KEY" <<'PY'
import hashlib, json, pathlib, re, sys

(
    descriptor_raw, staged_descriptor_raw, readiness_raw, manifest_raw,
    entries_raw, approval_raw, inventory_raw, audit_raw, repository_raw,
    repository_key, repository_sha, approval_key, inventory_key, audit_key,
) = sys.argv[1:]
paths = {
    "repository_tar": pathlib.Path(repository_raw),
    "approval": pathlib.Path(approval_raw),
    "artifact_inventory": pathlib.Path(inventory_raw),
    "descriptor": pathlib.Path(staged_descriptor_raw),
}
readiness = json.load(open(readiness_raw))
manifest = json.load(open(manifest_raw))
entries = json.load(open(entries_raw))["files"]
descriptor_path = pathlib.Path(descriptor_raw)
if pathlib.Path(staged_descriptor_raw).read_bytes() != descriptor_path.read_bytes():
    raise SystemExit("readiness-pinned descriptor differs from explicit authority")
for item in entries:
    role = item["role"]
    if role == "training_config":
        path = pathlib.Path(manifest_raw).parent / "training-config.json"
    elif role == "watchdog":
        path = pathlib.Path(manifest_raw).parent / "watchdog.zip"
    else:
        path = paths[role]
    raw = path.read_bytes()
    if len(raw) != item["size"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
        raise SystemExit(f"readiness-pinned staged object drift: {role}")
descriptor = json.loads(descriptor_path.read_bytes())
role_entries = {item["role"]: item for item in manifest["files"]}
staged_key_version_ids = {
    item["key"]: readiness["staged_object_version_ids"][item["role"]]
    for item in manifest["files"]
}
if (
    role_entries["repository_tar"]["key"] != repository_key
    or role_entries["repository_tar"]["sha256"] != repository_sha
    or descriptor["repo_tar_key"] != repository_key
    or descriptor["repo_tar_sha256"] != repository_sha
):
    raise SystemExit("repository explicit pin and staged manifest disagree")
if (
    role_entries["approval"]["key"] != approval_key
    or descriptor["approval_key"] != approval_key
    or role_entries["approval"]["sha256"] != descriptor["approval_sha256"]
):
    raise SystemExit("approval staged manifest and descriptor disagree")
if (
    role_entries["artifact_inventory"]["key"] != inventory_key
    or descriptor["artifacts"]["artifact_inventory_key"] != inventory_key
    or role_entries["artifact_inventory"]["sha256"]
    != descriptor["artifacts"]["artifact_inventory_sha256"]
):
    raise SystemExit("artifact inventory staged manifest and descriptor disagree")
inventory_path = pathlib.Path(inventory_raw)
inventory = json.loads(inventory_path.read_bytes())
inventory_fields = {
    "schema_version", "record_type", "run_id", "bucket", "objects",
    "inventory_body_sha256",
}
inventory_unversioned_object_fields = {
    "key", "size", "sha256", "kind", "safetensors", "run_scope",
}
inventory_versioned_object_fields = (
    inventory_unversioned_object_fields | {"version_id"}
)
inventory_kinds = {
    "repository_tar", "campaign_descriptor", "approval", "source_model",
    "non_vq_package", "training_baseline", "prompt_pack",
    "training_configuration", "watchdog_code", "qualification_cache",
}
digest = re.compile(r"^[0-9a-f]{64}$")
canonical = lambda value: json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
).encode()
if (
    set(inventory) != inventory_fields
    or inventory.get("schema_version") != 1
    or inventory.get("record_type") != "glm52_s3_artifact_inventory_v1"
    or inventory.get("run_id") != descriptor["run_id"]
    or inventory.get("bucket") != descriptor["bucket"]
):
    raise SystemExit("artifact inventory schema or authority mismatch")
inventory_body = dict(inventory)
inventory_body_sha = inventory_body.pop("inventory_body_sha256", None)
if (
    not isinstance(inventory_body_sha, str)
    or not digest.fullmatch(inventory_body_sha)
    or inventory_body_sha
    != hashlib.sha256(canonical(inventory_body)).hexdigest()
):
    raise SystemExit("artifact inventory body identity mismatch")
objects = inventory.get("objects")
if not isinstance(objects, list) or not objects:
    raise SystemExit("artifact inventory objects are invalid")
for item in objects:
    if (
        not isinstance(item, dict)
        or set(item)
        not in (
            inventory_unversioned_object_fields,
            inventory_versioned_object_fields,
        )
    ):
        raise SystemExit("artifact inventory object schema mismatch")
    key = item.get("key")
    if (
        not isinstance(key, str)
        or not key
        or key.startswith("/")
        or key.endswith("/")
        or any(not 0x21 <= ord(character) <= 0x7E for character in key)
        or "\\" in key
        or any(character in key for character in "*?[]")
        or any(part in {"", ".", ".."} for part in key.split("/"))
        or type(item.get("size")) is not int
        or item["size"] <= 0
        or not isinstance(item.get("sha256"), str)
        or not digest.fullmatch(item["sha256"])
        or item.get("kind") not in inventory_kinds
        or type(item.get("safetensors")) is not bool
        or item.get("run_scope") not in {descriptor["run_id"], "shared"}
        or (
            "version_id" in item
            and (
                type(item["version_id"]) is not str
                or item["version_id"] in {"", "null", "None", "latest"}
                or any(
                    not 0x21 <= ord(character) <= 0x7E
                    for character in item["version_id"]
                )
            )
        )
        or (
            "version_id" in item
            and key in staged_key_version_ids
            and item["version_id"] != staged_key_version_ids[key]
        )
        or (
            item.get("run_scope") == descriptor["run_id"]
            and not key.startswith(f"campaigns/{descriptor['run_id']}/")
        )
    ):
        raise SystemExit("artifact inventory object authority mismatch")
if objects != sorted(objects, key=lambda item: item["key"]):
    raise SystemExit("artifact inventory objects are not key-sorted")
if len({item["key"] for item in objects}) != len(objects):
    raise SystemExit("artifact inventory contains duplicate keys")
audit_path = pathlib.Path(audit_raw)
audit = json.loads(audit_path.read_bytes())
expected = {
    "object_count": len(objects),
    "object_bytes": sum(int(item["size"]) for item in objects),
    "safetensors_object_count": sum(item["safetensors"] is True for item in objects),
}
audit_fields = {
    "schema_version", "record_type", "audit_pass", "run_id", "bucket",
    "inventory_body_sha256", "object_count", "object_bytes",
    "safetensors_object_count", "safetensors_tensor_count",
}
if (
    set(audit) != audit_fields
    or
    readiness["artifact_audit_key"] != audit_key
    or readiness["artifact_audit_sha256"]
    != hashlib.sha256(audit_path.read_bytes()).hexdigest()
    or audit.get("schema_version") != 1
    or audit.get("record_type") != "glm52_s3_artifact_audit_v1"
    or audit.get("audit_pass") is not True
    or audit.get("run_id") != descriptor["run_id"]
    or audit.get("bucket") != descriptor["bucket"]
    or audit.get("inventory_body_sha256")
    != inventory.get("inventory_body_sha256")
    or any(audit.get(field) != value for field, value in expected.items())
):
    raise SystemExit("artifact audit authority mismatch")
for field in (
    "object_count", "object_bytes", "safetensors_object_count",
    "safetensors_tensor_count",
):
    if type(audit.get(field)) is not int or audit[field] < 0:
        raise SystemExit(f"artifact audit {field} is invalid")
tensor_count = audit["safetensors_tensor_count"]
safetensors_count = expected["safetensors_object_count"]
if (safetensors_count == 0 and tensor_count != 0) or (
    safetensors_count > 0 and tensor_count < safetensors_count
):
    raise SystemExit("artifact audit tensor count is inconsistent")
PY

tar -xzf "$WORK/repo.tar.gz" -C "$WORK/repo"
python3 - "$WORK/descriptor.json" \
  "$WORK/GPU_SPEND_APPROVAL.json" "$WORK/artifact-inventory-v1.json" \
  "$WORK/repo/src/mlx_vq/quality/glm52_sky_campaign.py" \
  "$WORK/repo/src/mlx_vq/quality/glm52_s3_artifact_audit.py" <<'PY'
import importlib.util, json, sys

descriptor_raw, approval_raw, inventory_raw, sky_raw, audit_raw = sys.argv[1:]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
sky = load("_glm52_pinned_rehearsal_sky", sky_raw)
artifact = load("_glm52_pinned_rehearsal_artifact", audit_raw)
sky.validate_sky_campaign_descriptor(json.load(open(descriptor_raw)))
sky.validate_gpu_spend_approval(json.load(open(approval_raw)))
artifact.validate_s3_artifact_inventory(json.load(open(inventory_raw)))
PY

while IFS= read -r script; do
  bash -n "$script"
done < <(find "$WORK/repo/aws/glm52-gpu/scripts" -type f -name '*.sh' -print)
python3 -m py_compile \
  "$WORK/repo/benchmarks/run_glm52_campaign.py" \
  "$WORK/repo/src/mlx_vq/quality/glm52_campaign_watchdog.py" \
  "$WORK/repo/src/mlx_vq/quality/glm52_capacity_block_controller.py" \
  "$WORK/repo/aws/glm52-gpu/scripts/watch_campaign.py" \
  "$WORK/repo/aws/glm52-gpu/scripts/campaign_break_glass.py"

grep -Fq 'WorkingDirectory=/opt/keep-campaign/repo' \
  "$WORK/repo/aws/glm52-gpu/cfn/gpu-teacher-stack.yaml"
grep -Fq 'Environment=CAMPAIGN_DESCRIPTOR=/etc/keep-glm52/campaign.json' \
  "$WORK/repo/aws/glm52-gpu/cfn/gpu-teacher-stack.yaml"
grep -Fq 'ExecStart=/opt/keep-campaign/repo/aws/glm52-gpu/scripts/run_campaign.sh' \
  "$WORK/repo/aws/glm52-gpu/cfn/gpu-teacher-stack.yaml"
grep -Fq 'export ROOT=/mnt/nvme/glm52-campaign' \
  "$WORK/repo/aws/glm52-gpu/scripts/run_campaign.sh"
grep -Fq 'export KEEP_REPO_DIR=/opt/keep-campaign/repo' \
  "$WORK/repo/aws/glm52-gpu/scripts/run_campaign.sh"

BASELINE_KEY="${BASELINE_PREFIX%/}/training-baseline.json"
READY_KEY="${BASELINE_PREFIX%/}/ROUTED_BASELINE_READY.json"
MANIFEST_KEY="${BASELINE_PREFIX%/}/routed-artifacts/conversion-manifest.json"
aws s3 cp "s3://$BUCKET/$BASELINE_KEY" "$WORK/authorities/training-baseline.json" \
  --region "$REGION" --only-show-errors
aws s3 cp "s3://$BUCKET/$READY_KEY" "$WORK/authorities/ROUTED_BASELINE_READY.json" \
  --region "$REGION" --only-show-errors
aws s3 cp "s3://$BUCKET/$MANIFEST_KEY" "$WORK/authorities/conversion-manifest.json" \
  --region "$REGION" --only-show-errors

python3 - "$WORK/descriptor.json" "$WORK/repo" "$WORK/authorities" <<'PY'
import hashlib, json, pathlib, sys
descriptor = json.load(open(sys.argv[1]))
repo = pathlib.Path(sys.argv[2])
authorities = pathlib.Path(sys.argv[3])
baseline_path = authorities / "training-baseline.json"
ready_path = authorities / "ROUTED_BASELINE_READY.json"
manifest_path = authorities / "conversion-manifest.json"
baseline = json.load(open(baseline_path))
ready = json.load(open(ready_path))
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if sha(baseline_path) != descriptor["artifacts"]["training_baseline_sha256"]:
    raise SystemExit("training baseline identity mismatch")
expected_prefix = (
    descriptor["artifacts"]["training_baseline_prefix"].rstrip("/")
    + "/routed-artifacts/"
)
if (
    ready.get("bucket") != descriptor["bucket"]
    or ready.get("prefix") != expected_prefix
):
    raise SystemExit("routed baseline storage authority mismatch")
if ready.get("artifact_count") != 225 or ready.get("artifact_total_bytes") != 61_312_700_712:
    raise SystemExit("routed baseline inventory drift")
if not ready.get("all_full_sha256_match") or not ready.get("all_s3_composite_sha256_match"):
    raise SystemExit("routed baseline checksum gate is not green")
if sha(manifest_path) != ready.get("conversion_manifest_sha256"):
    raise SystemExit("routed conversion manifest identity mismatch")
if baseline.get("expected_composite_audit_sha256") != ready.get("accepted_composite_audit_sha256"):
    raise SystemExit("baseline audit authority drift")
for key, value in baseline.items():
    if not key.endswith(("_json", "_jsonl", "_path")):
        continue
    if value.startswith("/opt/keep-campaign/repo/"):
        local = repo / value.removeprefix("/opt/keep-campaign/repo/")
        if not local.is_file():
            raise SystemExit(f"repository authority missing from tar: {key}={value}")
    elif not value.startswith("/mnt/nvme/glm52-campaign/"):
        raise SystemExit(f"unexpected absolute campaign path: {key}={value}")
runner = (repo / "benchmarks/run_glm52_campaign.py").read_text()
for path in (
    "teacher-checkpoints", "training-checkpoints", "training-boundaries",
    "teacher-cache-v3", "teacher-captures", "monitor",
):
    if path not in runner:
        raise SystemExit(f"resume path missing from staged runner: {path}")
PY

AUTHORITY_KEYS=( $(python3 - "$WORK/descriptor.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
a = d["artifacts"]
print(a["source_snapshot_prefix"].rstrip("/") + "/model.safetensors.index.json")
print(a["non_vq_prefix"].rstrip("/") + "/non-vq-manifest.json")
print(a["teich_pack_key"])
print(a["frozen_prompt_pack_key"])
PY
) )
AUTHORITY_NAMES=(source-index.json non-vq-manifest.json teich-pack.json frozen-pack.json)
for index in "${!AUTHORITY_KEYS[@]}"; do
  aws s3 cp "s3://$BUCKET/${AUTHORITY_KEYS[$index]}" \
    "$WORK/authorities/${AUTHORITY_NAMES[$index]}" --region "$REGION" --only-show-errors
done
python3 - "$WORK/descriptor.json" "$WORK/authorities" <<'PY'
import hashlib, json, pathlib, sys
d = json.load(open(sys.argv[1])); a = d["artifacts"]; root = pathlib.Path(sys.argv[2])
checks = {
    "source-index.json": a["source_snapshot_sha256"],
    "teich-pack.json": a["teich_pack_sha256"],
    "frozen-pack.json": a["frozen_prompt_pack_sha256"],
}
for name, expected in checks.items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"authority identity mismatch: {name} {actual}")
non_vq = json.loads((root / "non-vq-manifest.json").read_bytes())
if non_vq.get("package_set_sha256") != a["non_vq_package_sha256"]:
    raise SystemExit("non-VQ package-set identity mismatch")
PY

aws s3api list-objects-v2 --bucket "$BUCKET" \
  --prefix "${BASELINE_PREFIX%/}/routed-artifacts/" --region "$REGION" --output json \
  > "$WORK/routed-inventory.json"
aws s3api list-multipart-uploads --bucket "$BUCKET" \
  --prefix "${BASELINE_PREFIX%/}/" --region "$REGION" --output json \
  > "$WORK/multipart.json"
python3 - "$WORK/routed-inventory.json" "$WORK/multipart.json" <<'PY'
import json, sys
inventory = json.load(open(sys.argv[1])).get("Contents", [])
tensors = [item for item in inventory if item["Key"].endswith(".safetensors")]
if len(tensors) != 225 or sum(item["Size"] for item in tensors) != 61_312_700_712:
    raise SystemExit("live S3 routed tensor inventory mismatch")
if json.load(open(sys.argv[2])).get("Uploads"):
    raise SystemExit("unfinished multipart uploads exist under training baseline prefix")
PY

SKY_BIN="${SKY_BIN:-/Users/jack.mazac/.local/share/keep/skypilot-0.13.0/bin/sky}"
SKY_PYTHON="$(dirname -- "$SKY_BIN")/python"
test -x "$SKY_BIN"
test -x "$SKY_PYTHON"
RUN_ID=$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' \
  "$WORK/descriptor.json")
SKYPILOT_CONFIG="$WORK/skypilot-config.yaml"
python3 - "$WORK/repo/aws/glm52-gpu/skypilot/skypilot-config.yaml.in" \
  "$SKYPILOT_CONFIG" "$BUCKET" "$RUN_ID" <<'PY'
import pathlib, sys
source = pathlib.Path(sys.argv[1]).read_text()
source = source.replace("__JOBS_BUCKET__", sys.argv[3])
source = source.replace("__RUN_ID__", sys.argv[4])
pathlib.Path(sys.argv[2]).write_text(source)
PY
SKYPILOT_TASK="$WORK/repo/aws/glm52-gpu/skypilot/glm52-campaign.yaml"
"$SKY_PYTHON" \
  "$WORK/repo/aws/glm52-gpu/scripts/validate_skypilot_control_plane.py" \
  --task "$SKYPILOT_TASK" --config "$SKYPILOT_CONFIG" \
  > "$WORK/skypilot-validation.json"

cp "$WORK/descriptor.json" "$WORK/root/campaign.json"
cp "$WORK/GPU_SPEND_APPROVAL.json" "$WORK/root/GPU_SPEND_APPROVAL.json"
BOOTSTRAP_OUTPUT="$WORK/bootstrap-rehearsal.log"
run_bootstrap_boundary "$BOOTSTRAP_OUTPUT" env \
  AWS_PROFILE="${AWS_PROFILE:-keep-gpu}" \
  BUCKET="$BUCKET" ROOT="$WORK/root" \
  CAMPAIGN_DESCRIPTOR="$WORK/root/campaign.json" \
  GPU_SPEND_APPROVAL="$WORK/root/GPU_SPEND_APPROVAL.json" \
  KEEP_REPO_DIR="$WORK/repo" KEEP_REPO_PRESTAGED=1 \
  GLM52_MANAGED_MODE=qualification \
  GLM52_BOOTSTRAP_REHEARSAL=1 \
  "$WORK/repo/aws/glm52-gpu/skypilot/bootstrap_campaign.sh"

COMPLETED_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
"$SCRIPT_PATH" --write-evidence \
  --output "$EVIDENCE_OUTPUT" \
  --bootstrap-receipt "$BOOTSTRAP_OUTPUT" \
  --descriptor "$WORK/descriptor.json" \
  --descriptor-key "$DESCRIPTOR_KEY" \
  --repo-tar "$WORK/repo.tar.gz" \
  --readiness "$WORK/STAGED_CONTROL_PLANE_READY.json" \
  --readiness-key "$READINESS_KEY" \
  --inventory "$WORK/artifact-inventory-v1.json" \
  --audit "$WORK/artifact-audit-v1.json" \
  --extracted-repo "$WORK/repo" \
  --skypilot-task "$SKYPILOT_TASK" \
  --skypilot-config "$SKYPILOT_CONFIG" \
  --skypilot-validation "$WORK/skypilot-validation.json" \
  --skypilot-python "$SKY_PYTHON" \
  --skypilot-validator \
  "$WORK/repo/aws/glm52-gpu/scripts/validate_skypilot_control_plane.py" \
  --completed-at "$COMPLETED_AT"
"$SCRIPT_PATH" --validate-evidence "$EVIDENCE_OUTPUT"
REHEARSAL_SUCCEEDED=1
if [ "$KEEP_WORK" != 1 ] && evidence_output_is_within_work; then
  echo "REHEARSAL_WORK_RETAINED=$WORK"
fi
