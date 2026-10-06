#!/usr/bin/env python3
"""Repository-owned, shell-free Task 13 coordinator protocol boundary.

The staged runner supplies closed requests.  This executable admits only the
finite read/execute operations implemented below and fails closed for every
unknown protocol operation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from urllib.parse import quote


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
PROFILE = "keep-gpu"
SOURCE_AUTHORITY_EMAIL = "operator@example.com"
OWNER_APPROVAL_SHA256 = (
    "23ccff3454896d7fb8e9d0722b6c76522ebf2dfa3dfea9856a17a3e48b26ebc1"
)
OPENSSH_SIGNING_NAMESPACE = "keep-glm52-task13-controller-authority-v2"
PINNED_SIGNER_PUBLIC_KEY = (
    "ssh-ed25519 "
    "AAAAC3NzaC1lZDI1NTE5AAAAIDGJndv7GTsFzrh/fGs82pm/W6MmUvOsnPlP28IKugex "
    "keep-glm52-task13-controller-v3"
)
PINNED_SIGNER_FINGERPRINT = (
    "SHA256:8nCALZDwL4X59iIu+znZj5QBKJCUi/c5tjkMR39Q5mM"
)
OPENSSH_EXECUTABLE = "/usr/bin/ssh-keygen"
OPENSSH_EXECUTABLE_SHA256 = (
    "3ebf8d762494bb9e23be1955a8e136a6f3853512ea1f693fab96e0986bb9304f"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ACTIONS = {"execute", "inspect", "reconcile"}
RUN_ID = "glm52-sky-20260724"
LEDGER_TABLE = "keep-glm52-h1g-ledger-v1"
CAMPAIGN_BUCKET = (
    "keep-glm52-models-246813579024-us-west-2"
)
REHEARSAL_BUCKET = (
    "keep-glm52-h1g-rehearsal-246813579024-us-west-2"
)
DEPLOYMENT_ROLE_ARN = (
    "arn:aws:iam::246813579024:role/"
    "keep-glm52-h1g-cloudformation-deployment"
)
FENCE_ROLE_ARN = (
    "arn:aws:iam::246813579024:role/keep-glm52-h1g-fence-service"
)
_INSPECT_KINDS = frozenset(
    {
        "health",
        "credential-guard",
        "migration-bootstrap-state",
        "staged-infrastructure-adoption",
        "negative-iam-probe",
        "p5-zero",
        "measurement",
        "gate",
        "immutable-marker",
        "immutable-artifact",
        "terminal-proof",
        "monitor",
    }
)
_EXECUTE_KINDS = frozenset(
    {
        "migration-bootstrap",
        "collector-invoke",
        "finalize-invoke",
        "qualification-cache-seed",
        "h100-qualification",
        "sole-sender-authority-materialize",
        "launch-authority-materialize",
        "guarded-launch",
    }
)
_RECONCILE_KINDS = _EXECUTE_KINDS
_PRODUCTION_INPUT_KINDS = (
    "REPOSITORY_ARCHIVE",
    "CLEAN_REHEARSAL",
    "ACCEPTED_BASELINE",
    "PROMPT_PACK",
    "TRAINING_CONFIGURATION",
    "GPU_SPEND_APPROVAL",
    "SUPPORT_APPROVAL",
    "RESIDUAL_LIABILITY_APPROVAL",
    "PRODUCTION_DESCRIPTOR",
    "TASK10_PRODUCTION_AUTHORITY",
    "TASK10_WORKER_DESCRIPTOR",
    "TASK10_TASK_INPUTS",
)
_TASK12_ORPHAN_MATERIALIZER = (
    "aws/glm52-gpu/scripts/materialize_task12_orphan_authority.py"
)
_SUPPORT_POSTCREATE_MATERIALIZER = (
    "aws/glm52-gpu/scripts/materialize_h1g_support_plane.py"
)
_TASK12_ORPHAN_MATERIALIZER_SHA256 = (
    "db8b1f3239b5f1539e175590c43e4627c3723a22b58638f0450f9e2615a310b2"
)
_SUPPORT_POSTCREATE_MATERIALIZER_SHA256 = (
    "4ed0b92a086e5cf9ea2a6d5b846cd3cde31e03c330e8094b52cb0a8b944dd562"
)
_TASK12_ORPHAN_COMMON_FIELDS = frozenset(
    {
        "materializer",
        "support_postcreate_materializer",
        "support_inputs",
        "support_template",
        "precreate_output",
        "postcreate_output_dir",
        "activation_manifest",
        "lifecycle_resource_id",
        "lifecycle_cost_class",
        "source_publisher_arn",
        "settling_window_seconds",
        "account_id",
        "region",
        "profile",
        "run_id",
        "activation_id",
        "support_stack_id",
        "fence_stack_id",
    }
)

_CONTROLLER_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "profile",
        "run_id",
        "activation_id",
        "stage",
        "package_identity_sha256",
        "reviewed_artifacts_identity_sha256",
        "controller_nonce_sha256",
        "source_authority_email",
        "owner_approval_sha256",
        "execution_custody_identity_sha256",
        "production_support_plane_approved",
        "residual_launch_liability_approved",
        "change_set_execution_approved",
        "qualification_cache_seed_approved",
        "h100_qualification_approved",
        "production_launch_approved",
        "coordinator_executable_sha256",
        "issued_at_epoch_seconds",
        "expires_at_epoch_seconds",
        "signer_identity",
        "signing_namespace",
        "signer_public_key_fingerprint",
        "canonical_identity_sha256",
        "sshsig_base64",
    }
)
_OPERATION_CAPABILITY_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "profile",
        "run_id",
        "activation_id",
        "stage",
        "controller_authority_identity_sha256",
        "package_identity_sha256",
        "reviewed_artifacts_identity_sha256",
        "execution_custody_identity_sha256",
        "coordinator_executable_sha256",
        "owner_approval_sha256",
        "action",
        "operation_kind",
        "operation_id",
        "request_identity_sha256",
        "journal_store_path",
        "journal_store_identity_sha256",
        "journal_record_identities",
        "journal_chain_head_sha256",
        "journal_state",
        "prior_execute_capability_identity_sha256",
        "capability_nonce_sha256",
        "issued_at_epoch_seconds",
        "expires_at_epoch_seconds",
        "signer_identity",
        "signing_namespace",
        "signer_public_key_fingerprint",
        "canonical_identity_sha256",
        "sshsig_base64",
    }
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(label + " is not lowercase SHA-256")
    return value


def _verify_sshsig(payload: bytes, signature_base64: object) -> None:
    if (
        not os.path.isfile(OPENSSH_EXECUTABLE)
        or os.path.islink(OPENSSH_EXECUTABLE)
    ):
        raise ValueError("pinned OpenSSH verifier is unavailable")
    with open(OPENSSH_EXECUTABLE, "rb") as source:
        executable_identity = hashlib.sha256(source.read()).hexdigest()
    if executable_identity != OPENSSH_EXECUTABLE_SHA256:
        raise ValueError("pinned OpenSSH verifier identity drifted")
    if type(signature_base64) is not str or len(signature_base64) > 8192:
        raise ValueError("signed controller authority is absent")
    try:
        signature = base64.b64decode(signature_base64, validate=True)
    except ValueError as error:
        raise ValueError("signed controller authority is malformed") from error
    if (
        not signature.startswith(b"-----BEGIN SSH SIGNATURE-----\n")
        or not signature.endswith(b"-----END SSH SIGNATURE-----\n")
        or len(signature) > 4096
    ):
        raise ValueError("signed controller authority is malformed")
    with tempfile.TemporaryDirectory(
        prefix="glm52-task13-coordinator-authority-"
    ) as directory:
        allowed = Path(directory) / "allowed-signers"
        signature_path = Path(directory) / "authority.sig"
        allowed.write_text(
            SOURCE_AUTHORITY_EMAIL + " " + PINNED_SIGNER_PUBLIC_KEY + "\n",
            encoding="ascii",
        )
        signature_path.write_bytes(signature)
        allowed.chmod(0o400)
        signature_path.chmod(0o400)
        completed = subprocess.run(
            [
                OPENSSH_EXECUTABLE,
                "-Y",
                "verify",
                "-f",
                str(allowed),
                "-I",
                SOURCE_AUTHORITY_EMAIL,
                "-n",
                OPENSSH_SIGNING_NAMESPACE,
                "-s",
                str(signature_path),
            ],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
            env={
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "/usr/bin:/bin",
            },
            timeout=10,
        )
    if completed.returncode != 0:
        raise ValueError("controller authority signature verification failed")


def _verify_controller_authority(
    value: object,
    *,
    now: int,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != _CONTROLLER_AUTHORITY_FIELDS:
        raise ValueError("controller execution authority fields drifted")
    authority = dict(value)
    signature = authority.pop("sshsig_base64")
    identity = authority.pop("canonical_identity_sha256")
    if (
        value.get("schema_version") != 2
        or value.get("record_type")
        != "glm52_task13_controller_execution_authority_v2"
        or value.get("account_id") != ACCOUNT_ID
        or value.get("region") != REGION
        or value.get("profile") != PROFILE
        or value.get("run_id") != RUN_ID
        or value.get("source_authority_email") != SOURCE_AUTHORITY_EMAIL
        or value.get("owner_approval_sha256") != OWNER_APPROVAL_SHA256
        or value.get("signer_identity") != SOURCE_AUTHORITY_EMAIL
        or value.get("signing_namespace") != OPENSSH_SIGNING_NAMESPACE
        or value.get("signer_public_key_fingerprint")
        != PINNED_SIGNER_FINGERPRINT
        or value.get("coordinator_executable_sha256") != _self_sha256()
        or value.get("production_support_plane_approved") is not True
        or value.get("residual_launch_liability_approved") is not True
        or type(value.get("issued_at_epoch_seconds")) is not int
        or type(value.get("expires_at_epoch_seconds")) is not int
        or value["issued_at_epoch_seconds"] > now
        or value["expires_at_epoch_seconds"] <= now
        or value["expires_at_epoch_seconds"]
        - value["issued_at_epoch_seconds"]
        not in range(60, 901)
    ):
        raise ValueError("controller execution authority scope drifted")
    for field in (
        "package_identity_sha256",
        "reviewed_artifacts_identity_sha256",
        "controller_nonce_sha256",
        "execution_custody_identity_sha256",
    ):
        _sha256(value.get(field), "controller authority " + field)
    if identity != _canonical_sha256(authority):
        raise ValueError("controller execution authority identity drifted")
    signed = {**authority, "canonical_identity_sha256": identity}
    _verify_sshsig(_canonical(signed), signature)
    return dict(value)


def _journal_binding_identity(path: Path) -> str:
    info = path.lstat()
    return _canonical_sha256(
        {
            "path": str(path),
            "device": info.st_dev,
            "inode": info.st_ino,
        }
    )


def _validate_operation_journal(
    capability: Mapping[str, object],
) -> None:
    path = Path(str(capability["journal_store_path"]))
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path
    ):
        raise ValueError("signed operation journal path drifted")
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_uid != os.getuid()
        or _journal_binding_identity(path)
        != capability["journal_store_identity_sha256"]
    ):
        raise ValueError("signed operation journal custody drifted")
    raw = path.read_bytes()
    if (
        len(raw) > 16 * 1024 * 1024
        or not raw
        or not raw.endswith(b"\n")
    ):
        raise ValueError("signed operation journal bytes drifted")
    matching = []
    for line in raw.splitlines(keepends=True):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("signed operation journal is not JSON") from error
        if (
            type(row) is not dict
            or line != _canonical(row) + b"\n"
            or row.get("schema_version") != 1
            or row.get("record_type")
            != "glm52_task13_runner_journal_v1"
        ):
            raise ValueError("signed operation journal record drifted")
        if row.get("operation_id") == capability["operation_id"]:
            unsigned = {
                key: item
                for key, item in row.items()
                if key != "canonical_identity_sha256"
            }
            if (
                row.get("canonical_identity_sha256")
                != _canonical_sha256(unsigned)
            ):
                raise ValueError("signed operation journal hash drifted")
            matching.append(row)
    identities = [
        row["canonical_identity_sha256"] for row in matching
    ]
    if (
        identities != capability["journal_record_identities"]
        or not matching
        or matching[-1].get("state") != capability["journal_state"]
    ):
        raise ValueError("signed operation journal binding drifted")


def _verify_operation_capability(
    value: object,
    *,
    authority: Mapping[str, object],
    envelope: Mapping[str, object],
    now: int,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != _OPERATION_CAPABILITY_FIELDS:
        raise ValueError("operation capability fields drifted")
    capability = dict(value)
    signature = capability.pop("sshsig_base64")
    identity = capability.pop("canonical_identity_sha256")
    ancestry = {
        "account_id": authority["account_id"],
        "region": authority["region"],
        "profile": authority["profile"],
        "run_id": authority["run_id"],
        "activation_id": authority["activation_id"],
        "stage": authority["stage"],
        "controller_authority_identity_sha256": authority[
            "canonical_identity_sha256"
        ],
        "package_identity_sha256": authority["package_identity_sha256"],
        "reviewed_artifacts_identity_sha256": authority[
            "reviewed_artifacts_identity_sha256"
        ],
        "execution_custody_identity_sha256": authority[
            "execution_custody_identity_sha256"
        ],
        "coordinator_executable_sha256": _self_sha256(),
        "owner_approval_sha256": OWNER_APPROVAL_SHA256,
        "signer_identity": SOURCE_AUTHORITY_EMAIL,
        "signing_namespace": OPENSSH_SIGNING_NAMESPACE,
        "signer_public_key_fingerprint": PINNED_SIGNER_FINGERPRINT,
        "action": envelope["action"],
        "operation_kind": envelope["operation_kind"],
        "operation_id": envelope["operation_id"],
        "request_identity_sha256": _canonical_sha256(envelope["request"]),
    }
    if (
        value.get("schema_version") != 2
        or value.get("record_type")
        != "glm52_task13_operation_capability_v2"
        or any(value.get(field) != expected for field, expected in ancestry.items())
        or type(value.get("issued_at_epoch_seconds")) is not int
        or type(value.get("expires_at_epoch_seconds")) is not int
        or value["issued_at_epoch_seconds"] > now
        or value["expires_at_epoch_seconds"] <= now
        or value["expires_at_epoch_seconds"]
        - value["issued_at_epoch_seconds"]
        not in range(60, 901)
        or value["expires_at_epoch_seconds"]
        > authority["expires_at_epoch_seconds"]
    ):
        raise ValueError("operation capability scope or TTL drifted")
    for field in (
        "journal_store_identity_sha256",
        "journal_chain_head_sha256",
        "capability_nonce_sha256",
    ):
        _sha256(value.get(field), "operation capability " + field)
    records = value.get("journal_record_identities")
    if (
        type(records) is not list
        or len(records) not in range(1, 5)
        or any(
            type(item) is not str or _SHA256.fullmatch(item) is None
            for item in records
        )
        or value["journal_chain_head_sha256"] != _canonical_sha256(records)
        or identity != _canonical_sha256(capability)
    ):
        raise ValueError("operation capability identity drifted")
    action = str(envelope["action"])
    if action == "inspect":
        if (
            value.get("journal_state") != "READ_ONLY"
            or value.get("journal_store_path") != "READ_ONLY"
            or value.get("prior_execute_capability_identity_sha256")
            is not None
        ):
            raise ValueError("read-only capability journal binding drifted")
    else:
        if (
            value.get("journal_state")
            not in {"POSSIBLY_SENT", "COMMITTED"}
            or (
                action == "execute"
                and value.get("journal_state") != "POSSIBLY_SENT"
            )
        ):
            raise ValueError("mutation capability journal state drifted")
        _validate_operation_journal(value)
        if action == "reconcile":
            prior = _sha256(
                value.get("prior_execute_capability_identity_sha256"),
                "prior execute custody identity",
            )
            if prior != _execute_custody_identity(authority, value):
                raise ValueError("prior execute custody identity drifted")
        elif value.get("prior_execute_capability_identity_sha256") is not None:
            raise ValueError("execute capability invented prior custody")
    signed = {**capability, "canonical_identity_sha256": identity}
    _verify_sshsig(_canonical(signed), signature)
    return dict(value)


def _authenticate_v2_envelope(
    envelope: Mapping[str, object],
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    if (
        envelope["action"] == "inspect"
        and envelope["operation_kind"] == "health"
        and envelope["controller_execution_authority"] is None
        and envelope["operation_capability"] is None
    ):
        return None, None
    now = int(time.time())
    authority = _verify_controller_authority(
        envelope["controller_execution_authority"],
        now=now,
    )
    capability = _verify_operation_capability(
        envelope["operation_capability"],
        authority=authority,
        envelope=envelope,
        now=now,
    )
    return authority, capability


def _wire_encode(value: object) -> object:
    if type(value) is bytes:
        import base64

        return {
            "__glm52_wire_type__": "bytes",
            "base64": base64.b64encode(value).decode("ascii"),
        }
    if type(value) is dict:
        return {key: _wire_encode(item) for key, item in value.items()}
    if type(value) is list:
        return [_wire_encode(item) for item in value]
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise ValueError("coordinator result is not closed JSON data")


def _wire_decode(value: object) -> object:
    if type(value) is list:
        return [_wire_decode(item) for item in value]
    if type(value) is dict:
        if set(value) == {"__glm52_wire_type__", "base64"}:
            if value["__glm52_wire_type__"] != "bytes":
                raise ValueError("coordinator wire type is unsupported")
            encoded = value["base64"]
            if type(encoded) is not str:
                raise ValueError("coordinator bytes payload is malformed")
            try:
                return base64.b64decode(encoded, validate=True)
            except ValueError as error:
                raise ValueError(
                    "coordinator bytes payload is malformed"
                ) from error
        return {key: _wire_decode(item) for key, item in value.items()}
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise ValueError("coordinator request is not closed JSON data")


def _repo_root() -> Path:
    raw = os.environ.get("GLM52_TASK13_REPO_ROOT")
    if not raw:
        raise ValueError("repository root is not supplied by the runner")
    root = Path(raw)
    if (
        not root.is_absolute()
        or not root.is_dir()
        or root.is_symlink()
        or not (root / "aws/glm52-gpu/scripts").is_dir()
    ):
        raise ValueError("repository root identity drifted")
    return root.resolve()


def _repository_executable(relative_path: str) -> Path:
    root = _repo_root()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("repository executable escaped the root") from error
    if not path.is_file() or path.is_symlink() or not os.access(path, os.X_OK):
        raise ValueError("repository executable is unavailable")
    return path


def _pinned_repository_executable(
    relative_path: str,
    expected_sha256: str,
) -> Path:
    path = _repository_executable(relative_path)
    with path.open("rb") as source:
        observed = hashlib.sha256(source.read()).hexdigest()
    if observed != expected_sha256:
        raise ValueError("repository materializer identity drifted")
    return path


def _self_sha256() -> str:
    with open(__file__, "rb") as source:
        return hashlib.sha256(source.read()).hexdigest()


def _aws_json(argv: object) -> object:
    if (
        type(argv) is not list
        or not argv
        or argv[0] != "aws"
        or any(type(item) is not str or not item for item in argv)
        or "--profile" not in argv
        or argv[argv.index("--profile") + 1] != PROFILE
        or (
            "--region" in argv
            and argv[argv.index("--region") + 1] != REGION
        )
    ):
        raise ValueError("AWS command argv is outside the closed boundary")
    executable = "/opt/homebrew/bin/aws"
    if not os.path.isfile(executable):
        executable = "/usr/local/bin/aws"
    if not os.path.isfile(executable):
        raise ValueError("approved AWS CLI executable is unavailable")
    completed = subprocess.run(
        [executable, *argv[1:]],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        shell=False,
        env={
            "HOME": os.environ.get("HOME", ""),
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
            "AWS_MAX_ATTEMPTS": "1",
            "AWS_RETRY_MODE": "standard",
        },
        timeout=(
            30 * 60
            if argv[1:3] == ["cloudformation", "wait"]
            else 120
        ),
    )
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", "replace").strip()
        raise ValueError("AWS CLI refused operation: " + error[:500])
    stdout = completed.stdout.strip()
    if not stdout:
        return {}
    try:
        return json.loads(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("AWS CLI result is not JSON") from error


def _credential_evidence(command: Mapping[str, object]) -> dict[str, object]:
    observed = int(time.time())
    identity = _aws_json(
        [
            "aws",
            "sts",
            "get-caller-identity",
            "--profile",
            PROFILE,
            "--region",
            REGION,
            "--no-cli-pager",
        ]
    )
    if type(identity) is not dict or identity.get("Account") != ACCOUNT_ID:
        raise ValueError("credential account identity drifted")
    payload = _aws_json(command["argv"])
    if type(payload) is not dict:
        raise ValueError("credential response drifted")
    expiration = payload.get("Expiration")
    if type(expiration) is not str:
        raise ValueError("credential expiration is absent")
    expires = int(datetime.fromisoformat(expiration.replace("Z", "+00:00")).timestamp())
    if expires - observed < 3600:
        raise ValueError("credential expiration is insufficient for mutation")
    return {
        "account_id": identity["Account"],
        "observed_epoch_seconds": observed,
        "expiration_epoch_seconds": expires,
        "seconds_remaining": expires - observed,
    }


def _current_credential_evidence() -> dict[str, object]:
    return _credential_evidence(
        {
            "argv": [
                "aws",
                "configure",
                "export-credentials",
                "--profile",
                PROFILE,
                "--format",
                "process",
            ]
        }
    )


def _get_s3_version(
    *,
    bucket: str,
    key: str,
    version_id: str,
) -> bytes:
    with tempfile.TemporaryDirectory(prefix="glm52-task13-s3-") as directory:
        destination = os.path.join(directory, "body")
        result = _aws_json(
            [
                "aws",
                "s3api",
                "get-object",
                "--bucket",
                bucket,
                "--key",
                key,
                "--version-id",
                version_id,
                destination,
                "--profile",
                PROFILE,
                "--region",
                REGION,
                "--no-cli-pager",
            ]
        )
        if (
            type(result) is not dict
            or result.get("VersionId") != version_id
        ):
            raise ValueError("S3 VersionId readback drifted")
        with open(destination, "rb") as source:
            return source.read()


def _exact_s3_coordinate(coordinate: Mapping[str, object]) -> dict[str, object]:
    for field in ("bucket", "key", "version_id"):
        if type(coordinate.get(field)) is not str or not coordinate[field]:
            raise ValueError("S3 immutable coordinate drifted")
    raw = _get_s3_version(
        bucket=str(coordinate["bucket"]),
        key=str(coordinate["key"]),
        version_id=str(coordinate["version_id"]),
    )
    raw_sha = hashlib.sha256(raw).hexdigest()
    body_candidates = {raw_sha}
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        parsed = None
    if type(parsed) is dict:
        for identity_field in (
            "canonical_identity_sha256",
            "canonical_body_sha256",
        ):
            if identity_field not in parsed:
                continue
            unsigned = dict(parsed)
            claimed = unsigned.pop(identity_field)
            computed = hashlib.sha256(_canonical(unsigned)).hexdigest()
            if claimed != computed:
                raise ValueError("S3 immutable self-hash drifted")
            body_candidates.add(computed)
    expected_body = coordinate.get("body_sha256")
    expected_file = coordinate.get("file_sha256")
    if expected_body not in body_candidates or (
        expected_file is not None and expected_file != raw_sha
    ):
        raise ValueError("S3 immutable body identity drifted")
    return dict(coordinate)


def _exact_runtime_archive(
    coordinate: Mapping[str, object],
) -> dict[str, object]:
    raw = _get_s3_version(
        bucket=str(coordinate["bucket"]),
        key=str(coordinate["key"]),
        version_id=str(coordinate["version_id"]),
    )
    if (
        len(raw) != coordinate.get("size_bytes")
        or hashlib.sha256(raw).hexdigest()
        != coordinate.get("file_sha256")
    ):
        raise ValueError("runtime archive exact readback drifted")
    return dict(coordinate)


def _staged_worker_launch_count(activation_id: str) -> int:
    start_key: object = None
    count = 0
    for _page in range(64):
        argv = [
            "aws",
            "dynamodb",
            "query",
            "--table-name",
            LEDGER_TABLE,
            "--key-condition-expression",
            "PK = :pk AND begins_with(SK, :sk)",
            "--expression-attribute-values",
            json.dumps(
                {
                    ":pk": {"S": f"RUN#{RUN_ID}"},
                    ":sk": {
                        "S": (
                            f"ACTIVATION#{activation_id}#"
                            "WORKER_LAUNCH#"
                        )
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            "--consistent-read",
            "--select",
            "COUNT",
            "--profile",
            PROFILE,
            "--region",
            REGION,
            "--no-cli-pager",
        ]
        if start_key is not None:
            argv.extend(
                [
                    "--exclusive-start-key",
                    json.dumps(
                        start_key,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ]
            )
        response = _aws_json(argv)
        page_count = (
            response.get("Count")
            if type(response) is dict
            else None
        )
        if type(page_count) is not int or page_count < 0:
            raise ValueError(
                "staged adoption worker ledger drifted"
            )
        count += page_count
        start_key = response.get("LastEvaluatedKey")
        if start_key is None or start_key == {}:
            return count
        if type(start_key) is not dict:
            raise ValueError(
                "staged adoption worker ledger drifted"
            )
    raise ValueError("staged adoption worker ledger pagination exceeded")


def _cloudtrail_run_instances_matches(
    value: object,
    *,
    activation_id: str,
) -> bool:
    if (
        type(value) is not dict
        or value.get("EventName") != "RunInstances"
        or value.get("EventSource") != "ec2.amazonaws.com"
        or type(value.get("EventId")) is not str
        or not value["EventId"]
        or type(value.get("CloudTrailEvent")) is not str
    ):
        raise ValueError("CloudTrail RunInstances event drifted")
    try:
        event = json.loads(value["CloudTrailEvent"])
    except json.JSONDecodeError as error:
        raise ValueError(
            "CloudTrail RunInstances event drifted"
        ) from error
    if (
        type(event) is not dict
        or event.get("eventSource") != "ec2.amazonaws.com"
        or event.get("eventName") != "RunInstances"
        or event.get("awsRegion") != REGION
        or event.get("recipientAccountId") != ACCOUNT_ID
        or event.get("eventID") != value["EventId"]
        or type(event.get("requestParameters")) is not dict
    ):
        raise ValueError("CloudTrail RunInstances event drifted")
    specifications = event["requestParameters"].get(
        "tagSpecificationSet"
    )
    if specifications is None:
        return False
    if (
        type(specifications) is not dict
        or type(specifications.get("items")) is not list
    ):
        raise ValueError("CloudTrail RunInstances tags drifted")
    tag_values: dict[str, set[str]] = {}
    for specification in specifications["items"]:
        if (
            type(specification) is not dict
            or type(specification.get("resourceType")) is not str
            or not specification["resourceType"]
            or type(specification.get("tags")) is not list
        ):
            raise ValueError("CloudTrail RunInstances tags drifted")
        for tag in specification["tags"]:
            if (
                type(tag) is not dict
                or type(tag.get("key")) is not str
                or not tag["key"]
                or type(tag.get("value")) is not str
            ):
                raise ValueError("CloudTrail RunInstances tags drifted")
            tag_values.setdefault(tag["key"], set()).add(tag["value"])
    if any(len(values) != 1 for values in tag_values.values()):
        raise ValueError("CloudTrail RunInstances tags are ambiguous")
    run_values = {
        value
        for key in {"RunId", "campaign-run-id", "CampaignRunId"}
        for value in tag_values.get(key, set())
    }
    activation_values = {
        value
        for key in {
            "activation-id",
            "ActivationId",
            "Task13ActivationId",
        }
        for value in tag_values.get(key, set())
    }
    if len(run_values) > 1 or len(activation_values) > 1:
        raise ValueError("CloudTrail RunInstances identity is ambiguous")
    return (
        run_values == {RUN_ID}
        and activation_values == {activation_id}
    )


def _staged_raw_ec2_launch_count(activation_id: str) -> int:
    if type(activation_id) is not str or not activation_id:
        raise ValueError("CloudTrail activation identity drifted")
    token: str | None = None
    observed_tokens: set[str] = set()
    observed_event_ids: set[str] = set()
    count = 0
    for _page in range(256):
        argv = [
            "aws",
            "cloudtrail",
            "lookup-events",
            "--lookup-attributes",
            "AttributeKey=EventName,AttributeValue=RunInstances",
            "--page-size",
            "50",
            "--max-items",
            "50",
            "--profile",
            PROFILE,
            "--region",
            REGION,
            "--no-cli-pager",
        ]
        if token is not None:
            argv.extend(["--starting-token", token])
        response = _aws_json(argv)
        events = (
            response.get("Events")
            if type(response) is dict
            else None
        )
        if type(events) is not list:
            raise ValueError("CloudTrail RunInstances page drifted")
        for event in events:
            if type(event) is not dict:
                raise ValueError(
                    "CloudTrail RunInstances page drifted"
                )
            event_id = event.get("EventId")
            if (
                type(event_id) is not str
                or not event_id
                or event_id in observed_event_ids
            ):
                raise ValueError(
                    "CloudTrail RunInstances pagination drifted"
                )
            observed_event_ids.add(event_id)
            count += _cloudtrail_run_instances_matches(
                event,
                activation_id=activation_id,
            )
        next_token = response.get("NextToken")
        if next_token is None:
            return count
        if (
            type(next_token) is not str
            or not next_token
            or next_token in observed_tokens
        ):
            raise ValueError(
                "CloudTrail RunInstances pagination drifted"
            )
        observed_tokens.add(next_token)
        token = next_token
    raise ValueError("CloudTrail RunInstances pagination exceeded")


def _exact_support_lambda_readback(
    *,
    template_resources: Mapping[str, object],
    observed_resources: Mapping[str, Mapping[str, object]],
    support_evidence: Mapping[str, object],
) -> None:
    functions = {
        logical_id: resource
        for logical_id, resource in template_resources.items()
        if type(resource) is dict
        and resource.get("Type") == "AWS::Lambda::Function"
    }
    versions = {
        logical_id: resource
        for logical_id, resource in template_resources.items()
        if type(resource) is dict
        and resource.get("Type") == "AWS::Lambda::Version"
    }
    if not functions or not versions:
        raise ValueError(
            "staged adoption support Lambda inventory drifted"
        )
    support_archive = support_evidence.get("support_lambda_archive")
    if type(support_archive) is not dict:
        raise ValueError(
            "staged adoption support Lambda archive drifted"
        )
    try:
        expected_code_sha256 = base64.b64encode(
            bytes.fromhex(str(support_archive["file_sha256"]))
        ).decode("ascii")
    except (KeyError, ValueError) as error:
        raise ValueError(
            "staged adoption support Lambda archive drifted"
        ) from error
    layer_arn = support_evidence.get(
        "cryptography_layer_version_arn"
    )
    function_names: dict[str, str] = {}

    def expected_role_arn(properties: Mapping[str, object]) -> str:
        source = properties.get("Role")
        if type(source) is str and source:
            return source
        if (
            type(source) is not dict
            or set(source) != {"Fn::GetAtt"}
            or type(source["Fn::GetAtt"]) is not list
            or source["Fn::GetAtt"][1:] != ["Arn"]
            or type(source["Fn::GetAtt"][0]) is not str
        ):
            raise ValueError(
                "staged adoption support Lambda role drifted"
            )
        role_logical_id = source["Fn::GetAtt"][0]
        role_resource = template_resources.get(role_logical_id)
        role_observed = observed_resources.get(role_logical_id)
        role_properties = (
            role_resource.get("Properties")
            if type(role_resource) is dict
            else None
        )
        if (
            type(role_resource) is not dict
            or role_resource.get("Type") != "AWS::IAM::Role"
            or type(role_properties) is not dict
            or type(role_properties.get("RoleName")) is not str
            or not role_properties["RoleName"]
            or type(role_observed) is not dict
            or role_observed.get("ResourceType") != "AWS::IAM::Role"
            or role_observed.get("PhysicalResourceId")
            != role_properties["RoleName"]
        ):
            raise ValueError(
                "staged adoption support Lambda role drifted"
            )
        return (
            "arn:aws:iam::"
            + ACCOUNT_ID
            + ":role/"
            + str(role_properties["RoleName"])
        )

    def exact_configuration(
        configuration: object,
        *,
        properties: Mapping[str, object],
        function_name: str,
        version: str,
        physical_arn: str | None,
    ) -> None:
        if type(configuration) is not dict:
            raise ValueError(
                "staged adoption support Lambda readback drifted"
            )
        expected_layers = properties.get("Layers", [])
        live_layers = configuration.get("Layers", [])
        if (
            type(expected_layers) is not list
            or any(type(item) is not str for item in expected_layers)
            or type(live_layers) is not list
            or any(
                type(item) is not dict
                or set(item) - {"Arn", "CodeSize", "SigningProfileVersionArn",
                                "SigningJobArn"}
                or type(item.get("Arn")) is not str
                for item in live_layers
            )
            or [item["Arn"] for item in live_layers] != expected_layers
        ):
            raise ValueError(
                "staged adoption support Lambda layer drifted"
            )
        expected_environment = properties.get(
            "Environment",
            {"Variables": {}},
        )
        live_environment = configuration.get(
            "Environment",
            {"Variables": {}},
        )
        if (
            type(expected_environment) is not dict
            or set(expected_environment) != {"Variables"}
            or type(expected_environment["Variables"]) is not dict
            or any(
                type(key) is not str or type(item) is not str
                for key, item in expected_environment["Variables"].items()
            )
            or type(live_environment) is not dict
            or type(live_environment.get("Variables")) is not dict
            or live_environment["Variables"]
            != expected_environment["Variables"]
        ):
            raise ValueError(
                "staged adoption support Lambda environment drifted"
            )
        expected_vpc = properties.get(
            "VpcConfig",
            {"SecurityGroupIds": [], "SubnetIds": []},
        )
        live_vpc = configuration.get("VpcConfig", {})
        if (
            type(expected_vpc) is not dict
            or set(expected_vpc) != {"SecurityGroupIds", "SubnetIds"}
            or any(
                type(expected_vpc[field]) is not list
                or any(type(item) is not str for item in expected_vpc[field])
                for field in ("SecurityGroupIds", "SubnetIds")
            )
            or type(live_vpc) is not dict
            or live_vpc.get("SecurityGroupIds", [])
            != expected_vpc["SecurityGroupIds"]
            or live_vpc.get("SubnetIds", []) != expected_vpc["SubnetIds"]
        ):
            raise ValueError(
                "staged adoption support Lambda VPC drifted"
            )
        expected_dead_letter = properties.get(
            "DeadLetterConfig",
            {"TargetArn": ""},
        )
        live_dead_letter = configuration.get(
            "DeadLetterConfig",
            {"TargetArn": ""},
        )
        expected_tracing = properties.get(
            "TracingConfig",
            {"Mode": "PassThrough"},
        )
        live_tracing = configuration.get(
            "TracingConfig",
            {"Mode": "PassThrough"},
        )
        expected_kms = properties.get("KmsKeyArn", "")
        live_kms = configuration.get("KMSKeyArn", "")
        expected_ephemeral = properties.get(
            "EphemeralStorage",
            {"Size": 512},
        )
        live_ephemeral = configuration.get(
            "EphemeralStorage",
            {"Size": 512},
        )
        if (
            configuration.get("FunctionName") != function_name
            or configuration.get("Version") != version
            or configuration.get("CodeSha256") != expected_code_sha256
            or configuration.get("Runtime") != properties["Runtime"]
            or configuration.get("Architectures", ["x86_64"])
            != properties.get("Architectures", ["x86_64"])
            or configuration.get("Handler") != properties.get("Handler")
            or configuration.get("MemorySize")
            != properties.get("MemorySize")
            or configuration.get("Timeout") != properties.get("Timeout")
            or configuration.get("Role") != expected_role_arn(properties)
            or live_dead_letter != expected_dead_letter
            or live_tracing != expected_tracing
            or live_kms != expected_kms
            or live_ephemeral != expected_ephemeral
            or (
                physical_arn is not None
                and configuration.get("FunctionArn") != physical_arn
            )
        ):
            raise ValueError(
                "staged adoption support Lambda readback drifted"
            )

    for logical_id, resource in functions.items():
        properties = resource.get("Properties")
        observed = observed_resources.get(logical_id)
        if (
            type(properties) is not dict
            or type(observed) is not dict
            or type(observed.get("PhysicalResourceId")) is not str
            or not observed["PhysicalResourceId"]
            or properties.get("Code")
            != {
                "S3Bucket": support_archive["bucket"],
                "S3Key": support_archive["key"],
                "S3ObjectVersion": support_archive["version_id"],
            }
            or properties.get("Runtime") != "python3.12"
            or properties.get("Architectures", ["x86_64"])
            != ["x86_64"]
        ):
            raise ValueError(
                "staged adoption support Lambda template drifted"
        )
        expected_layers = properties.get("Layers", [])
        if (
            type(expected_layers) is not list
            or any(type(item) is not str for item in expected_layers)
            or any(item != layer_arn for item in expected_layers)
        ):
            raise ValueError(
                "staged adoption support Lambda layer drifted"
            )
        function_name = str(observed["PhysicalResourceId"])
        response = _aws_json(
            [
                "aws",
                "lambda",
                "get-function",
                "--function-name",
                function_name,
                "--profile",
                PROFILE,
                "--region",
                REGION,
                "--no-cli-pager",
            ]
        )
        configuration = (
            response.get("Configuration")
            if type(response) is dict
            else None
        )
        concurrency = (
            response.get("Concurrency")
            if type(response) is dict
            else None
        )
        if concurrency != {
            "ReservedConcurrentExecutions": properties.get(
                "ReservedConcurrentExecutions"
            )
        }:
            raise ValueError(
                "staged adoption support Lambda concurrency drifted"
            )
        exact_configuration(
            configuration,
            properties=properties,
            function_name=function_name,
            version="$LATEST",
            physical_arn=(
                "arn:aws:lambda:"
                + REGION
                + ":"
                + ACCOUNT_ID
                + ":function:"
                + function_name
            ),
        )
        function_names[logical_id] = function_name

    for logical_id, resource in versions.items():
        properties = resource.get("Properties")
        observed = observed_resources.get(logical_id)
        target = (
            properties.get("FunctionName")
            if type(properties) is dict
            else None
        )
        target_logical_id = (
            target.get("Ref") if type(target) is dict else None
        )
        if (
            type(observed) is not dict
            or type(observed.get("PhysicalResourceId")) is not str
            or not observed["PhysicalResourceId"]
            or type(target_logical_id) is not str
            or target_logical_id not in function_names
        ):
            raise ValueError(
                "staged adoption support Lambda version drifted"
            )
        physical_id = str(observed["PhysicalResourceId"])
        response = _aws_json(
            [
                "aws",
                "lambda",
                "get-function",
                "--function-name",
                physical_id,
                "--profile",
                PROFILE,
                "--region",
                REGION,
                "--no-cli-pager",
            ]
        )
        configuration = (
            response.get("Configuration")
            if type(response) is dict
            else None
        )
        if (
            type(configuration) is not dict
            or type(configuration.get("Version")) is not str
            or not configuration["Version"].isdigit()
            or configuration.get("Version") == "0"
        ):
            raise ValueError(
                "staged adoption support Lambda version drifted"
            )
        function_resource = functions[target_logical_id]
        function_properties = function_resource["Properties"]
        exact_configuration(
            configuration,
            properties=function_properties,
            function_name=function_names[target_logical_id],
            version=str(configuration["Version"]),
            physical_arn=physical_id,
        )


def _canonical_mapping(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(label + " is not JSON") from error
    if type(value) is not dict or raw not in {
        _canonical(value),
        _canonical(value) + b"\n",
    }:
        raise ValueError(label + " is not canonical JSON")
    return value


def _json_mapping(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(label + " is not JSON") from error
    if type(value) is not dict:
        raise ValueError(label + " is not a JSON object")
    return value


def _unique_current_s3_object(
    *,
    bucket: str,
    key: str,
) -> tuple[bytes, str]:
    versions = _aws_json(
        [
            "aws",
            "s3api",
            "list-object-versions",
            "--bucket",
            bucket,
            "--prefix",
            key,
            "--profile",
            PROFILE,
            "--region",
            REGION,
            "--no-cli-pager",
        ]
    )
    if type(versions) is not dict:
        raise ValueError("S3 version inventory drifted")
    exact = [
        row
        for row in versions.get("Versions", [])
        if type(row) is dict
        and row.get("Key") == key
        and row.get("IsLatest") is True
    ]
    deletes = [
        row
        for row in versions.get("DeleteMarkers", [])
        if type(row) is dict and row.get("Key") == key
    ]
    if len(exact) != 1 or deletes:
        raise ValueError("S3 positive truth is not one immutable live version")
    version_id = exact[0].get("VersionId")
    if type(version_id) is not str or not version_id:
        raise ValueError("S3 positive truth lacks a VersionId")
    return (
        _get_s3_version(bucket=bucket, key=key, version_id=version_id),
        version_id,
    )


def _active_p5_ids() -> list[str]:
    payload = _aws_json(
        [
            "aws",
            "ec2",
            "describe-instances",
            "--filters",
            "Name=instance-type,Values=p5.48xlarge",
            "Name=instance-state-name,Values=pending,running",
            "--profile",
            PROFILE,
            "--region",
            REGION,
            "--no-cli-pager",
        ]
    )
    if type(payload) is not dict:
        raise ValueError("EC2 inventory readback drifted")
    instance_ids = []
    for reservation in payload.get("Reservations", []):
        for instance in reservation.get("Instances", []):
            instance_id = instance.get("InstanceId")
            if type(instance_id) is not str:
                raise ValueError("EC2 instance identity drifted")
            instance_ids.append(instance_id)
    return sorted(instance_ids)


def _self_hashed_coordinate(
    *,
    key: str,
    version_id: str,
    raw: bytes,
    value: Mapping[str, object],
) -> dict[str, object]:
    unsigned = dict(value)
    canonical_identity = unsigned.pop("canonical_identity_sha256", None)
    if canonical_identity is None:
        canonical_identity = unsigned.pop("canonical_body_sha256", None)
    body_sha = hashlib.sha256(_canonical(unsigned)).hexdigest()
    if canonical_identity is not None and canonical_identity != body_sha:
        raise ValueError("immutable result self-hash drifted")
    body = {
        "key": key,
        "version_id": version_id,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "body_sha256": (
            canonical_identity if canonical_identity is not None else body_sha
        ),
    }
    return {
        **body,
        "canonical_identity_sha256": hashlib.sha256(
            _canonical(body)
        ).hexdigest(),
    }


def _read_result_record(
    *,
    key: str,
    bucket: str = CAMPAIGN_BUCKET,
) -> tuple[dict[str, object], dict[str, object]]:
    raw, version_id = _unique_current_s3_object(bucket=bucket, key=key)
    value = _canonical_mapping(raw, "external result record")
    coordinate = _self_hashed_coordinate(
        key=key,
        version_id=version_id,
        raw=raw,
        value=value,
    )
    return value, coordinate


def _stack_id_from(command: Mapping[str, object]) -> str:
    argv = command["argv"]
    assert type(argv) is list
    return str(argv[argv.index("--stack-name") + 1])


def _inspect_aws_command(command: Mapping[str, object]) -> dict[str, object]:
    command_id = command["command_id"]
    if command_id in {"read-caller-identity", "assert-caller-account"}:
        payload = _aws_json(command["argv"])
        account = (
            payload.get("Account")
            if type(payload) is dict
            else str(payload).strip()
        )
        return {"account_id": account}
    if command_id == "read-credential-expiry":
        return _credential_evidence(command)
    payload = _aws_json(command["argv"])
    if command_id in {
        "read-deployment-cloudformation-role",
        "read-fence-cloudformation-role",
    }:
        role = payload.get("Role") if type(payload) is dict else None
        if type(role) is not dict:
            raise ValueError("role readback drifted")
        return {
            "role_arn": role.get("Arn"),
            "role_id": role.get("RoleId"),
        }
    stack_id = _stack_id_from(command)
    if command_id.startswith("describe-"):
        if type(payload) is not dict:
            raise ValueError("change-set readback drifted")
        changes = []
        for row in payload.get("Changes", []):
            resource = row.get("ResourceChange", {})
            changes.append(
                {
                    "action": resource.get("Action"),
                    "logical_resource_id": resource.get("LogicalResourceId"),
                    "resource_type": resource.get("ResourceType"),
                    "replacement": resource.get("Replacement", "False"),
                }
            )
        description = str(payload.get("Description", ""))
        digest_match = re.search(r"([0-9a-f]{64})\Z", description)
        return {
            "change_set_arn": payload.get("ChangeSetId"),
            "stack_id": payload.get("StackId"),
            "status": payload.get("Status"),
            "execution_status": payload.get("ExecutionStatus"),
            "change_set_type": payload.get("ChangeSetType"),
            "role_arn": payload.get("RoleARN"),
            "template_body_sha256": (
                digest_match.group(1) if digest_match else None
            ),
            "changes": changes,
            "changes_identity_sha256": hashlib.sha256(
                _canonical(changes)
            ).hexdigest(),
        }
    if command_id.startswith("inspect-"):
        body = payload.get("TemplateBody") if type(payload) is dict else None
        return {
            "change_set_arn": command["argv"][
                command["argv"].index("--change-set-name") + 1
            ],
            "stack_id": stack_id,
            "template_stage": "Processed",
            "template_body_sha256": hashlib.sha256(
                _canonical(body)
            ).hexdigest(),
        }
    if command_id.startswith("readback-") and command_id.endswith("-stack"):
        stacks = payload.get("Stacks") if type(payload) is dict else None
        if type(stacks) is not list or len(stacks) != 1:
            raise ValueError("stack readback cardinality drifted")
        stack = stacks[0]
        tags = sorted(
            [[row["Key"], row["Value"]] for row in stack.get("Tags", [])]
        )
        return {
            "stack_id": stack.get("StackId"),
            "stack_status": stack.get("StackStatus"),
            "termination_protection": stack.get(
                "EnableTerminationProtection"
            ),
            "tags_identity_sha256": hashlib.sha256(
                _canonical(tags)
            ).hexdigest(),
        }
    if command_id.startswith("inventory-"):
        rows = payload.get("StackResourceSummaries", [])
        resources = [
            {
                "logical_resource_id": row.get("LogicalResourceId"),
                "resource_type": row.get("ResourceType"),
                "resource_status": row.get("ResourceStatus"),
            }
            for row in rows
        ]
        return {
            "stack_id": stack_id,
            "resource_summaries": resources,
            "resource_summaries_identity_sha256": hashlib.sha256(
                _canonical(resources)
            ).hexdigest(),
        }
    if command_id.startswith("readback-") and command_id.endswith("-template"):
        body = payload.get("TemplateBody") if type(payload) is dict else None
        return {
            "stack_id": stack_id,
            "template_stage": "Original",
            "template_body_sha256": hashlib.sha256(
                _canonical(body)
            ).hexdigest(),
        }
    raise ValueError("AWS inspection command is not implemented")


def _template_url(coordinate: Mapping[str, object]) -> str:
    return (
        "https://%s.s3.%s.amazonaws.com/%s?versionId=%s"
        % (
            coordinate["bucket"],
            REGION,
            quote(str(coordinate["key"]), safe="/"),
            quote(str(coordinate["version_id"]), safe=""),
        )
    )


def _described_stack(stack_name: str) -> dict[str, object] | None:
    payload = _aws_json(
        [
            "aws",
            "cloudformation",
            "list-stacks",
            "--stack-status-filter",
            "CREATE_COMPLETE",
            "UPDATE_COMPLETE",
            "IMPORT_COMPLETE",
            "UPDATE_ROLLBACK_COMPLETE",
            "--profile",
            PROFILE,
            "--region",
            REGION,
            "--no-cli-pager",
        ]
    )
    summaries = (
        payload.get("StackSummaries", [])
        if type(payload) is dict
        else []
    )
    matches = [
        row
        for row in summaries
        if type(row) is dict and row.get("StackName") == stack_name
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("CloudFormation stack identity is ambiguous")
    detail = _aws_json(
        [
            "aws",
            "cloudformation",
            "describe-stacks",
            "--stack-name",
            str(matches[0]["StackId"]),
            "--profile",
            PROFILE,
            "--region",
            REGION,
            "--no-cli-pager",
        ]
    )
    stacks = detail.get("Stacks", []) if type(detail) is dict else []
    if len(stacks) != 1 or type(stacks[0]) is not dict:
        raise ValueError("CloudFormation stack readback drifted")
    return dict(stacks[0])


def _bootstrap_state(request: Mapping[str, object]) -> dict[str, object]:
    if (
        request.get("retained_stack_name") != "keep-glm52-gpu"
        or type(request.get("activation_id")) is not str
        or not request.get("activation_id")
        or request.get("new_stack_names")
        != ["keep-glm52-h1g-fence", "keep-glm52-h1g-support"]
        or request.get("stack_role_arns")
        != {
            "keep-glm52-h1g-fence": FENCE_ROLE_ARN,
            "keep-glm52-h1g-support": DEPLOYMENT_ROLE_ARN,
        }
        or request.get("anchor_logical_id") != "ContainerAnchor"
        or request.get("termination_protection") is not True
        or type(request.get("template")) is not dict
    ):
        raise ValueError("migration bootstrap request drifted")
    retained = _described_stack("keep-glm52-gpu")
    if (
        retained is None
        or retained.get("StackId") != request.get("retained_stack_id")
    ):
        raise ValueError("retained CloudFormation stack identity drifted")
    fence = _described_stack("keep-glm52-h1g-fence")
    support = _described_stack("keep-glm52-h1g-support")
    for stack, name in (
        (fence, "keep-glm52-h1g-fence"),
        (support, "keep-glm52-h1g-support"),
    ):
        if stack is None:
            continue
        tags = {
            row.get("Key"): row.get("Value")
            for row in stack.get("Tags", [])
            if type(row) is dict
        }
        if (
            stack.get("StackName") != name
            or stack.get("RoleARN")
            != request["stack_role_arns"][name]
            or stack.get("EnableTerminationProtection") is not True
            or stack.get("StackStatus")
            not in {"CREATE_COMPLETE", "UPDATE_COMPLETE", "IMPORT_COMPLETE"}
            or tags
            != {
                "CampaignRunId": RUN_ID,
                "DeploymentState": "DISABLED",
                "Task13ActivationId": request["activation_id"],
            }
        ):
            raise ValueError("bootstrap stack state drifted")
        resources = _aws_json(
            [
                "aws",
                "cloudformation",
                "list-stack-resources",
                "--stack-name",
                str(stack["StackId"]),
                "--profile",
                PROFILE,
                "--region",
                REGION,
                "--no-cli-pager",
            ]
        )
        rows = (
            resources.get("StackResourceSummaries", [])
            if type(resources) is dict
            else []
        )
        if not any(
            type(row) is dict
            and row.get("LogicalResourceId") == "ContainerAnchor"
            and row.get("ResourceType")
            == "AWS::CloudFormation::WaitConditionHandle"
            for row in rows
        ):
            raise ValueError("bootstrap anchor readback drifted")
    absent = [
        name
        for name, stack in (
            ("keep-glm52-h1g-fence", fence),
            ("keep-glm52-h1g-support", support),
        )
        if stack is None
    ]
    if absent:
        return {
            "status": "BOOTSTRAP_REQUIRED",
            "retained_stack_id": request["retained_stack_id"],
            "absent_stack_names": absent,
            "existing_stack_ids": {
                name: stack["StackId"]
                for name, stack in (
                    ("keep-glm52-h1g-fence", fence),
                    ("keep-glm52-h1g-support", support),
                )
                if stack is not None
            },
        }
    return {
        "status": "BOOTSTRAPPED",
        "retained_stack_id": request["retained_stack_id"],
        "fence_stack_id": fence["StackId"],
        "support_stack_id": support["StackId"],
        "termination_protection": True,
        "anchor_logical_id": "ContainerAnchor",
        "template_body_sha256": request["template"]["body_sha256"],
    }


def _bootstrap_stack_result(
    request: Mapping[str, object],
    *,
    state: Mapping[str, object],
) -> dict[str, object]:
    target = str(request["target_stack_name"])
    if state.get("status") == "BOOTSTRAPPED":
        stack_id = state[
            "fence_stack_id"
            if target == "keep-glm52-h1g-fence"
            else "support_stack_id"
        ]
    else:
        existing = state.get("existing_stack_ids")
        if type(existing) is not dict or target not in existing:
            raise ValueError("bootstrap stack positive truth is absent")
        stack_id = existing[target]
    return {
        "status": "BOOTSTRAPPED_STACK",
        "retained_stack_id": request["retained_stack_id"],
        "stack_name": target,
        "stack_id": stack_id,
        "termination_protection": True,
        "anchor_logical_id": "ContainerAnchor",
        "template_body_sha256": request["template"]["body_sha256"],
    }


def _execute_bootstrap(request: Mapping[str, object]) -> dict[str, object]:
    target = request.get("target_stack_name")
    if target not in {
        "keep-glm52-h1g-fence",
        "keep-glm52-h1g-support",
    }:
        raise ValueError("migration bootstrap target drifted")
    stack_roles = request.get("stack_role_arns")
    expected_roles = {
        "keep-glm52-h1g-fence": FENCE_ROLE_ARN,
        "keep-glm52-h1g-support": DEPLOYMENT_ROLE_ARN,
    }
    if stack_roles != expected_roles:
        raise ValueError("migration bootstrap role mapping drifted")
    role_arn = expected_roles[str(target)]
    state = _bootstrap_state(request)
    if state["status"] == "BOOTSTRAPPED" or target not in state[
        "absent_stack_names"
    ]:
        return _bootstrap_stack_result(request, state=state)
    template = request["template"]
    assert type(template) is dict
    token = "task13-" + hashlib.sha256(
        _canonical(
            {
                "run_id": RUN_ID,
                "activation_id": request["activation_id"],
                "stack_name": target,
                "template": template,
                "role_arn": role_arn,
            }
        )
    ).hexdigest()
    _current_credential_evidence()
    _aws_json(
        [
            "aws",
            "cloudformation",
            "create-stack",
            "--stack-name",
            str(target),
            "--template-url",
            _template_url(template),
            "--role-arn",
            role_arn,
            "--client-request-token",
            token,
            "--enable-termination-protection",
            "--tags",
            "Key=CampaignRunId,Value=" + RUN_ID,
            "Key=DeploymentState,Value=DISABLED",
            "Key=Task13ActivationId,Value="
            + str(request["activation_id"]),
            "--profile",
            PROFILE,
            "--region",
            REGION,
            "--no-cli-pager",
        ]
    )
    _aws_json(
        [
            "aws",
            "cloudformation",
            "wait",
            "stack-create-complete",
            "--stack-name",
            str(target),
            "--profile",
            PROFILE,
            "--region",
            REGION,
            "--no-cli-pager",
        ]
    )
    result = _bootstrap_state(request)
    return _bootstrap_stack_result(request, state=result)


def _negative_iam_probe(request: Mapping[str, object]) -> dict[str, object]:
    if (
        request.get("expected_error_code") != "AccessDenied"
        or request.get("must_not_retry") is not True
        or type(request.get("principal_role_arn")) is not str
        or type(request.get("operation")) is not str
        or type(request.get("resource")) is not str
    ):
        raise ValueError("negative IAM probe request drifted")
    payload = _aws_json(
        [
            "aws",
            "iam",
            "simulate-principal-policy",
            "--policy-source-arn",
            str(request["principal_role_arn"]),
            "--action-names",
            str(request["operation"]),
            "--resource-arns",
            str(request["resource"]),
            "--profile",
            PROFILE,
            "--region",
            REGION,
            "--no-cli-pager",
        ]
    )
    rows = payload.get("EvaluationResults", []) if type(payload) is dict else []
    if (
        len(rows) != 1
        or type(rows[0]) is not dict
        or rows[0].get("EvalActionName") != request["operation"]
        or rows[0].get("EvalResourceName") != request["resource"]
        or rows[0].get("EvalDecision")
        not in {"implicitDeny", "explicitDeny"}
    ):
        raise ValueError("negative IAM probe did not prove denial")
    return {"error_code": "AccessDenied"}


def _measurement_readback(request: Mapping[str, object]) -> dict[str, object]:
    payload = request.get("payload")
    if type(payload) is not dict:
        raise ValueError("measurement payload is absent")
    raw = _get_s3_version(
        bucket=REHEARSAL_BUCKET,
        key=str(request["key"]),
        version_id=str(request["version_id"]),
    )
    if hashlib.sha256(raw).hexdigest() != request.get("file_sha256"):
        raise ValueError("measurement file identity drifted")
    measurement = _canonical_mapping(raw, "measurement")
    return {
        "key": request["key"],
        "version_id": request["version_id"],
        "file_sha256": request["file_sha256"],
        "measurement": measurement,
    }


def _validated_closure_gate(
    raw: bytes,
    *,
    activation_id: object,
) -> dict[str, object]:
    gate = _canonical_mapping(raw, "closure gate")
    expected_fields = {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "activation_id",
        "deployment_identity_sha256",
        "measurements",
        "measurement_count",
        "cold_environment_count",
        "environment_set_sha256",
        "measurements_identity_sha256",
        "canonical_body_sha256",
    }
    gate_body = dict(gate)
    body_identity = gate_body.pop("canonical_body_sha256", None)
    if (
        raw != _canonical(gate)
        or set(gate) != expected_fields
        or gate.get("schema_version") != 1
        or gate.get("record_type")
        != "glm52_task11_deployed_closure_gate_v1"
        or gate.get("account_id") != ACCOUNT_ID
        or gate.get("region") != REGION
        or gate.get("run_id") != RUN_ID
        or gate.get("activation_id") != activation_id
        or type(gate.get("deployment_identity_sha256")) is not str
        or _SHA256.fullmatch(
            str(gate.get("deployment_identity_sha256"))
        )
        is None
        or type(gate.get("measurements")) is not list
        or len(gate["measurements"]) != 20
        or type(gate.get("measurement_count")) is not int
        or type(gate.get("cold_environment_count")) is not int
        or type(gate.get("measurements_identity_sha256")) is not str
        or _SHA256.fullmatch(
            str(gate.get("measurements_identity_sha256"))
        )
        is None
        or body_identity != _canonical_sha256(gate_body)
    ):
        raise ValueError("closure gate exact readback drifted")
    return gate


def _gate_readback(request: Mapping[str, object]) -> dict[str, object]:
    payload = request.get("payload")
    if (
        set(request)
        != {"key", "version_id", "file_sha256", "body_sha256", "payload"}
        or type(payload) is not dict
    ):
        raise ValueError("gate payload is absent")
    raw = _get_s3_version(
        bucket=REHEARSAL_BUCKET,
        key=str(request["key"]),
        version_id=str(request["version_id"]),
    )
    if hashlib.sha256(raw).hexdigest() != request.get("file_sha256"):
        raise ValueError("gate file identity drifted")
    gate = _validated_closure_gate(
        raw,
        activation_id=payload.get("activation_id"),
    )
    if (
        gate.get("measurement_count") != payload.get("measurement_count")
        or gate.get("cold_environment_count")
        != payload.get("cold_environment_count")
        or gate.get("measurements_identity_sha256")
        != payload.get("measurements_identity_sha256")
        or gate.get("canonical_body_sha256") != request.get("body_sha256")
        or gate.get("canonical_body_sha256") != payload.get("body_sha256")
        or payload.get("key") != request.get("key")
        or payload.get("version_id") != request.get("version_id")
        or payload.get("file_sha256") != request.get("file_sha256")
        or payload.get("checksum_sha256_base64")
        != base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
    ):
        raise ValueError("closure gate exact readback drifted")
    summary = {
        field: payload[field]
        for field in (
            "key",
            "version_id",
            "file_sha256",
            "body_sha256",
            "status",
            "measurement_count",
            "cold_environment_count",
            "measurements_identity_sha256",
        )
    }
    return {"summary": summary, "content": raw}


def _lambda_invoke(request: Mapping[str, object]) -> dict[str, object]:
    if (
        set(request) != {"function_version_arn", "qualifier", "event"}
        or type(request["function_version_arn"]) is not str
        or not str(request["function_version_arn"]).endswith(
            ":" + str(request["qualifier"])
        )
        or type(request["event"]) is not dict
    ):
        raise ValueError("Lambda invocation request drifted")
    with tempfile.TemporaryDirectory(prefix="glm52-task13-lambda-") as directory:
        payload_path = Path(directory) / "payload.json"
        metadata = _aws_json(
            [
                "aws",
                "lambda",
                "invoke",
                "--function-name",
                str(request["function_version_arn"]),
                "--qualifier",
                str(request["qualifier"]),
                "--invocation-type",
                "RequestResponse",
                "--cli-binary-format",
                "raw-in-base64-out",
                "--payload",
                _canonical(request["event"]).decode("ascii"),
                str(payload_path),
                "--profile",
                PROFILE,
                "--region",
                REGION,
                "--no-cli-pager",
            ]
        )
        raw = payload_path.read_bytes()
    if (
        type(metadata) is not dict
        or type(metadata.get("StatusCode")) is not int
        or metadata["StatusCode"] != 200
        or metadata.get("FunctionError") is not None
        or type(metadata.get("ExecutedVersion")) is not str
        or metadata["ExecutedVersion"] != request["qualifier"]
    ):
        raise ValueError("Lambda invocation did not prove exact success")
    _json_mapping(raw, "Lambda result payload")
    return {
        "StatusCode": metadata["StatusCode"],
        "ExecutedVersion": metadata["ExecutedVersion"],
        "Payload": raw,
    }


def _reconcile_lambda(
    kind: str,
    request: Mapping[str, object],
) -> dict[str, object]:
    event = request.get("event")
    if (
        set(request) != {"function_version_arn", "qualifier", "event"}
        or type(request.get("function_version_arn")) is not str
        or type(request.get("qualifier")) is not str
        or not str(request["function_version_arn"]).endswith(
            ":" + str(request["qualifier"])
        )
        or type(event) is not dict
    ):
        raise ValueError("Lambda reconciliation event drifted")
    if kind == "collector-invoke":
        measurement_id = event.get("measurement_id")
        task11 = event.get("task11_request")
        if type(task11) is not dict:
            raise ValueError("collector Task11 identity is absent")
        candidate = task11.get("candidate_identity_sha256")
        key = (
            f"rehearsal/measurements/{event.get('activation_id')}/"
            f"{candidate}/{measurement_id}.json"
        )
    else:
        key = (
            f"rehearsal/gates/{event.get('activation_id')}/"
            "CLOSURE_BUDGET.json"
        )
    raw, version_id = _unique_current_s3_object(
        bucket=REHEARSAL_BUCKET,
        key=key,
    )
    if kind == "collector-invoke":
        _canonical_mapping(raw, "reconciled Lambda result")
        payload_raw = raw
    else:
        gate = _validated_closure_gate(
            raw,
            activation_id=event.get("activation_id"),
        )
        body: dict[str, object] = {
            "schema_version": 1,
            "record_type": (
                "glm52_task11_finalize_rehearsal_gate_result_v1"
            ),
            "status": "CLOSURE_BUDGET_PROVEN",
            "account_id": ACCOUNT_ID,
            "region": REGION,
            "run_id": RUN_ID,
            "activation_id": event["activation_id"],
            "collector_function_version_arn": request[
                "function_version_arn"
            ],
            "key": key,
            "version_id": version_id,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "body_sha256": gate["canonical_body_sha256"],
            "checksum_sha256_base64": base64.b64encode(
                hashlib.sha256(raw).digest()
            ).decode("ascii"),
            "measurement_count": gate["measurement_count"],
            "cold_environment_count": gate["cold_environment_count"],
            "measurements_identity_sha256": gate[
                "measurements_identity_sha256"
            ],
        }
        payload_raw = _canonical(
            {
                **body,
                "canonical_identity_sha256": _canonical_sha256(body),
            }
        )
    return {
        "StatusCode": 200,
        "ExecutedVersion": request["qualifier"],
        "Payload": payload_raw,
    }


def _committed_command_result(
    request: Mapping[str, object],
) -> dict[str, object]:
    return {
        "status": "COMMITTED",
        "operation": request["operation"],
        "request_identity_sha256": hashlib.sha256(
            _canonical(request)
        ).hexdigest(),
    }


def _validate_mutation_aws_command(
    request: Mapping[str, object],
) -> None:
    command_id = request.get("command_id")
    allowed = {
        "create-fence-disabled-change-set": (
            "cloudformation:CreateChangeSet",
            "create-change-set",
        ),
        "create-support-disabled-change-set": (
            "cloudformation:CreateChangeSet",
            "create-change-set",
        ),
        "execute-fence-disabled-change-set": (
            "cloudformation:ExecuteChangeSet",
            "execute-change-set",
        ),
        "execute-support-disabled-change-set": (
            "cloudformation:ExecuteChangeSet",
            "execute-change-set",
        ),
    }
    expected = allowed.get(command_id)
    argv = request.get("argv")
    if (
        expected is None
        or request.get("operation") != expected[0]
        or request.get("account_id") != ACCOUNT_ID
        or request.get("profile") != PROFILE
        or request.get("region") != REGION
        or request.get("mutates_aws") is not True
        or request.get("execution_state")
        != "BLOCKED_REQUIRES_CONTROLLER"
        or type(argv) is not list
        or argv[:3] != ["aws", "cloudformation", expected[1]]
        or any(
            type(item) is not str
            or not item
            or "\x00" in item
            or "\n" in item
            or "\r" in item
            for item in argv
        )
        or argv.count("--profile") != 1
        or argv[argv.index("--profile") + 1] != PROFILE
        or argv.count("--region") != 1
        or argv[argv.index("--region") + 1] != REGION
        or argv.count("--no-cli-pager") != 1
    ):
        raise ValueError("AWS mutation command is outside the finite boundary")


def _reconcile_aws_command(
    request: Mapping[str, object],
) -> dict[str, object]:
    _validate_mutation_aws_command(request)
    command_id = request.get("command_id")
    if type(command_id) is not str:
        raise ValueError("AWS reconciliation command identity drifted")
    if command_id.startswith("create-"):
        argv = request["argv"]
        assert type(argv) is list
        change_set_name = argv[argv.index("--change-set-name") + 1]
        stack_id = argv[argv.index("--stack-name") + 1]
        payload = _aws_json(
            [
                "aws",
                "cloudformation",
                "describe-change-set",
                "--stack-name",
                str(stack_id),
                "--change-set-name",
                str(change_set_name),
                "--profile",
                PROFILE,
                "--region",
                REGION,
                "--no-cli-pager",
            ]
        )
        if (
            type(payload) is not dict
            or payload.get("Status") != "CREATE_COMPLETE"
            or payload.get("ExecutionStatus") not in {"AVAILABLE", "EXECUTE_COMPLETE"}
        ):
            raise ValueError("change-set creation is not positive truth")
    elif command_id.startswith("execute-"):
        argv = request["argv"]
        assert type(argv) is list
        change_set_arn = argv[argv.index("--change-set-name") + 1]
        payload = _aws_json(
            [
                "aws",
                "cloudformation",
                "describe-change-set",
                "--change-set-name",
                str(change_set_arn),
                "--profile",
                PROFILE,
                "--region",
                REGION,
                "--no-cli-pager",
            ]
        )
        if (
            type(payload) is not dict
            or payload.get("ExecutionStatus") != "EXECUTE_COMPLETE"
        ):
            raise ValueError("change-set execution is not positive truth")
    else:
        raise ValueError("AWS mutation reconciliation is not implemented")
    return _committed_command_result(request)


def _seed_result() -> dict[str, object]:
    key = (
        f"campaigns/{RUN_ID}/qualification/"
        "QUALIFICATION_CACHE_SEED_READY.json"
    )
    value, seed_coordinate = _read_result_record(key=key)
    manifest = value.get("qualification_cache_manifest_sha256")
    if type(manifest) is not str:
        raise ValueError("seed result lacks cache manifest identity")
    teacher_key = (
        f"qualification-cache/seeds/{RUN_ID}/{manifest}/"
        "TEACHER_CACHE_READY.json"
    )
    _teacher, teacher_coordinate = _read_result_record(key=teacher_key)
    result = dict(value)
    result["seed_ready"] = seed_coordinate
    result["teacher_ready"] = teacher_coordinate
    result.pop("canonical_identity_sha256", None)
    return result


def _h100_result() -> dict[str, object]:
    keys = {
        "source_ready": (
            f"campaigns/{RUN_ID}/qualification/SOURCE_NODE_READY.json"
        ),
        "termination_requested": (
            f"campaigns/{RUN_ID}/qualification/"
            "QUALIFICATION_TERMINATION_REQUESTED.json"
        ),
        "h100_resume_ready": (
            f"campaigns/{RUN_ID}/qualification/H100_RESUME_READY.json"
        ),
    }
    values = {}
    coordinates = {}
    for field, key in keys.items():
        values[field], coordinates[field] = _read_result_record(key=key)
    result = dict(values["h100_resume_ready"])
    result.pop("canonical_identity_sha256", None)
    result.update(coordinates)
    return result


def _driver_manifest(
    coordinate: Mapping[str, object],
    *,
    operation_kind: str,
    expected_activation_id: str,
) -> tuple[list[str], dict[str, str]]:
    _exact_s3_coordinate(coordinate)
    raw = _get_s3_version(
        bucket=str(coordinate["bucket"]),
        key=str(coordinate["key"]),
        version_id=str(coordinate["version_id"]),
    )
    value = _canonical_mapping(raw, operation_kind + " driver manifest")
    unsigned = dict(value)
    identity = unsigned.pop("canonical_identity_sha256", None)
    if (
        set(value)
        != {
            "schema_version",
            "record_type",
            "activation_id",
            "operation_kind",
            "argv",
            "environment",
            "canonical_identity_sha256",
        }
        or value["schema_version"] != 2
        or value["record_type"]
        != "glm52_task13_repository_driver_v2"
        or value["activation_id"] != expected_activation_id
        or value["operation_kind"] != operation_kind
        or identity != hashlib.sha256(_canonical(unsigned)).hexdigest()
        or type(value["argv"]) is not list
        or type(value["environment"]) is not dict
    ):
        raise ValueError("repository driver manifest drifted")
    argv = value["argv"]
    assert type(argv) is list
    if any(type(item) is not str or not item for item in argv):
        raise ValueError("repository driver argv drifted")
    allowed_first = {
        "qualification-cache-seed": (
            "aws/glm52-gpu/scripts/submit_sky_campaign.py"
        ),
        "h100-qualification": (
            "aws/glm52-gpu/scripts/run_h100_qualification_campaign.sh"
        ),
    }[operation_kind]
    if argv[0] != allowed_first:
        raise ValueError("repository driver executable drifted")
    if operation_kind == "qualification-cache-seed" and argv[1:3] != [
        "cache-seed",
        "acquire-and-launch",
    ]:
        raise ValueError("cache-seed guarded route drifted")
    environment = value["environment"]
    assert type(environment) is dict
    allowed_environment = {
        "AWS_PROFILE",
        "AWS_REGION",
        "CAMPAIGN_DESCRIPTOR",
        "SKY_BIN",
        "SKYPILOT_CONFIG",
        "KEEP_PYTHON",
    }
    if (
        not set(environment).issubset(allowed_environment)
        or environment.get("AWS_PROFILE") != PROFILE
        or any(type(item) is not str for item in environment.values())
    ):
        raise ValueError("repository driver environment drifted")
    executable = _repository_executable(allowed_first)
    return [str(executable), *argv[1:]], dict(environment)


def _run_repository_driver(
    request: Mapping[str, object],
    *,
    operation_kind: str,
) -> None:
    activation_id = request.get("activation_id")
    if (
        request.get("driver_record_type")
        != "glm52_task13_repository_driver_v2"
        or request.get("driver_operation_kind") != operation_kind
        or type(activation_id) is not str
        or re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", activation_id) is None
    ):
        raise ValueError("repository driver provenance drifted")
    coordinate = request.get("input")
    if type(coordinate) is not dict:
        raise ValueError("repository driver input coordinate is absent")
    argv, supplied_environment = _driver_manifest(
        coordinate,
        operation_kind=operation_kind,
        expected_activation_id=activation_id,
    )
    environment = {
        "HOME": os.environ.get("HOME", ""),
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
        **supplied_environment,
    }
    completed = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        shell=False,
        env=environment,
        timeout=24 * 60 * 60,
    )
    if completed.returncode != 0:
        raise ValueError(
            operation_kind
            + " guarded repository route refused: "
            + completed.stderr.decode("utf-8", "replace")[:500]
        )


_TASK_INPUT_FIELDS = {
    "job_name",
    "descriptor_s3_uri",
    "descriptor_version_id",
    "descriptor_file_sha256",
    "approval_s3_uri",
    "approval_version_id",
    "approval_file_sha256",
    "intent_s3_uri",
    "intent_version_id",
    "intent_file_sha256",
    "intent_body_sha256",
    "repository_archive_s3_uri",
    "repository_archive_version_id",
    "repository_archive_file_sha256",
}


def _materialize_immutable_inputs(
    request: Mapping[str, object],
    directory: Path,
) -> dict[str, Path]:
    inputs = request.get("immutable_inputs")
    if (
        type(inputs) is not list
        or len(inputs) != 12
        or [
            coordinate.get("artifact_kind")
            if type(coordinate) is dict
            else None
            for coordinate in inputs
        ]
        != list(_PRODUCTION_INPUT_KINDS)
    ):
        raise ValueError("production immutable input set drifted")
    paths = {}
    for coordinate in inputs:
        if type(coordinate) is not dict:
            raise ValueError("production immutable coordinate drifted")
        _exact_s3_coordinate(coordinate)
        kind = coordinate.get("artifact_kind")
        if type(kind) is not str or kind in paths:
            raise ValueError("production immutable artifact kind drifted")
        raw = _get_s3_version(
            bucket=str(coordinate["bucket"]),
            key=str(coordinate["key"]),
            version_id=str(coordinate["version_id"]),
        )
        path = directory / (kind.lower() + ".json")
        path.write_bytes(raw)
        paths[kind] = path
    return paths


def _task10_material(
    request: Mapping[str, object],
) -> tuple[object, object, object, object]:
    source_root = str(_repo_root() / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from glm52_enforcement.task10_production import (
        ProductionRequest,
        production_authority_from_mapping,
    )
    from glm52_enforcement.task10_worker import (
        MountFreeTaskInputs,
        worker_bootstrap_descriptor_from_mapping,
    )

    values = []
    inputs = request.get("immutable_inputs")
    if (
        type(inputs) is not list
        or len(inputs) != 12
        or [
            coordinate.get("artifact_kind")
            if type(coordinate) is dict
            else None
            for coordinate in inputs
        ]
        != list(_PRODUCTION_INPUT_KINDS)
    ):
        raise ValueError("production immutable inputs are absent")
    for coordinate in inputs:
        if type(coordinate) is not dict:
            raise ValueError("production immutable coordinate drifted")
        _exact_s3_coordinate(coordinate)
        raw = _get_s3_version(
            bucket=str(coordinate["bucket"]),
            key=str(coordinate["key"]),
            version_id=str(coordinate["version_id"]),
        )
        try:
            values.append(_canonical_mapping(raw, "production input"))
        except ValueError:
            continue
    authority_values = [
        value
        for value in values
        if value.get("record_type")
        == "glm52_task10_production_authority_v1"
    ]
    task_values = [value for value in values if set(value) == _TASK_INPUT_FIELDS]
    descriptor_values = [
        value
        for value in values
        if value.get("record_type")
        == "glm52_task10_worker_bootstrap_descriptor_v1"
    ]
    if (
        len(authority_values) != 1
        or len(task_values) != 1
        or len(descriptor_values) != 1
    ):
        raise ValueError(
            "Task10 authority, task inputs, or worker descriptor is absent"
        )
    authority = production_authority_from_mapping(authority_values[0])
    task_inputs = MountFreeTaskInputs(**task_values[0])
    descriptor = worker_bootstrap_descriptor_from_mapping(
        descriptor_values[0]
    )
    production_request = ProductionRequest(
        action="start",
        profile=PROFILE,
        authority=authority,
        task_inputs=task_inputs,
        worker_descriptor=descriptor,
    )
    return production_request, authority, task_inputs, descriptor


class _StepFunctionsBoundary:
    def inspect(self, authority: object, execution_input: object) -> object:
        workflow_arn = getattr(authority, "workflow_version_arn")
        payload = _aws_json(
            [
                "aws",
                "stepfunctions",
                "describe-state-machine",
                "--state-machine-arn",
                workflow_arn,
                "--profile",
                PROFILE,
                "--region",
                REGION,
                "--no-cli-pager",
            ]
        )
        if (
            type(payload) is not dict
            or payload.get("stateMachineArn") != workflow_arn
            or payload.get("status") != "ACTIVE"
        ):
            raise ValueError("production workflow version is not active")
        body = {
            "current_activation": True,
            "action_consumed": True,
            "task8_live_authority": getattr(
                authority,
                "task8_live_h1d_identity_sha256",
            ),
            "task8_spend_authority": getattr(
                authority,
                "task8_spend_authority_identity_sha256",
            ),
            "task8_spend_reserve": getattr(
                authority,
                "task8_spend_reserve_identity_sha256",
            ),
            "task9_launch_admission": getattr(
                authority,
                "task9_admission_identity_sha256",
            ),
            "task9_liability_custody": getattr(
                authority,
                "task9_custody_identity_sha256",
            ),
            "workflow_version_arn": workflow_arn,
        }
        source_root = str(_repo_root() / "src")
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        from glm52_enforcement.canonical import canonical_sha256

        return {
            **body,
            "inspection_identity_sha256": canonical_sha256(body),
        }

    def start_once(self, authority: object, execution_input: object) -> object:
        payload = _aws_json(
            [
                "aws",
                "stepfunctions",
                "start-execution",
                "--state-machine-arn",
                str(getattr(authority, "workflow_version_arn")),
                "--name",
                str(getattr(authority, "execution_name")),
                "--input",
                _canonical(execution_input).decode("ascii"),
                "--profile",
                PROFILE,
                "--region",
                REGION,
                "--no-cli-pager",
            ]
        )
        execution_arn = getattr(authority, "expected_execution_arn")
        if (
            type(payload) is not dict
            or payload.get("executionArn") != execution_arn
        ):
            raise ValueError("production workflow start identity drifted")
        body = {
            "classification": "STARTED",
            "execution_arn": execution_arn,
            "reserve_identity_sha256": getattr(
                authority,
                "task8_spend_reserve_identity_sha256",
            ),
            "task9_custody_identity_sha256": getattr(
                authority,
                "task9_custody_identity_sha256",
            ),
        }
        from glm52_enforcement.canonical import canonical_sha256

        return {
            **body,
            "observation_identity_sha256": canonical_sha256(body),
        }

    def reconcile(self, authority: object, execution_input: object) -> object:
        execution_arn = str(getattr(authority, "expected_execution_arn"))
        payload = _aws_json(
            [
                "aws",
                "stepfunctions",
                "describe-execution",
                "--execution-arn",
                execution_arn,
                "--profile",
                PROFILE,
                "--region",
                REGION,
                "--no-cli-pager",
            ]
        )
        states = {
            "RUNNING": "RUNNING",
            "SUCCEEDED": "SUCCEEDED",
            "FAILED": "FAILED",
            "TIMED_OUT": "FAILED",
            "ABORTED": "FAILED",
        }
        if type(payload) is not dict or payload.get("status") not in states:
            raise ValueError("production execution truth is unavailable")
        if (
            payload.get("executionArn") != execution_arn
            or payload.get("stateMachineArn")
            != str(getattr(authority, "workflow_version_arn"))
            or payload.get("name")
            != str(getattr(authority, "execution_name"))
            or payload.get("input")
            != _canonical(execution_input).decode("ascii")
        ):
            raise ValueError(
                "production execution reconciliation identity drifted"
            )
        key = (
            f"campaigns/{RUN_ID}/submissions/production/generations/"
            f"{getattr(authority, 'generation_text')}/workflow/"
            "LAUNCH_OUTCOME.json"
        )
        value, coordinate = _read_result_record(key=key)
        source_fields = {
            "schema_version",
            "record_type",
            "account_id",
            "region",
            "run_id",
            "activation_id",
            "generation",
            "generation_text",
            "classification",
            "execution_arn",
            "capacity_outcomes",
            "instance_id",
            "availability_zone",
            "same_token_identity_sha256",
            "reserve_identity_sha256",
            "spend_authority_identity_sha256",
            "action_identity_sha256",
            "task9_custody_identity_sha256",
            "canonical_identity_sha256",
        }
        unsigned = dict(value)
        source_identity = unsigned.pop(
            "canonical_identity_sha256",
            None,
        )
        classification = value.get("classification")
        execution_status = states[str(payload["status"])]
        status_is_consistent = (
            (
                execution_status == "RUNNING"
                and classification == "RUNNING"
            )
            or (
                execution_status == "SUCCEEDED"
                and classification
                in {"WORKER_ALLOCATED", "CAPACITY_EXHAUSTED"}
            )
            or (
                execution_status == "FAILED"
                and classification == "FAILED"
            )
        )
        if (
            set(value) != source_fields
            or value["schema_version"] != 1
            or value["record_type"]
            != "glm52_task10_capacity_reconciliation_v1"
            or source_identity
            != hashlib.sha256(_canonical(unsigned)).hexdigest()
            or value["account_id"] != ACCOUNT_ID
            or value["region"] != REGION
            or value["run_id"] != RUN_ID
            or value["activation_id"]
            != getattr(authority, "activation_id")
            or value["generation"] != getattr(authority, "generation")
            or value["generation_text"]
            != getattr(authority, "generation_text")
            or value["execution_arn"] != execution_arn
            or value["same_token_identity_sha256"]
            != getattr(authority, "same_token_identity_sha256")
            or value["reserve_identity_sha256"]
            != getattr(
                authority,
                "task8_spend_reserve_identity_sha256",
            )
            or value["spend_authority_identity_sha256"]
            != getattr(
                authority,
                "task8_spend_authority_identity_sha256",
            )
            or value["action_identity_sha256"]
            != getattr(authority, "task9_launch_identity_sha256")
            or value["task9_custody_identity_sha256"]
            != getattr(authority, "task9_custody_identity_sha256")
            or not status_is_consistent
        ):
            raise ValueError(
                "production durable reconciliation identity drifted"
            )
        workflow_output = {
            "bucket": CAMPAIGN_BUCKET,
            "key": coordinate["key"],
            "version_id": coordinate["version_id"],
            "file_sha256": coordinate["file_sha256"],
            "body_sha256": coordinate["body_sha256"],
        }
        body = {
            "classification": classification,
            "execution_arn": execution_arn,
            "capacity_outcomes": value["capacity_outcomes"],
            "instance_id": value["instance_id"],
            "availability_zone": value["availability_zone"],
            "workflow_output": workflow_output,
            "same_token_identity_sha256": value[
                "same_token_identity_sha256"
            ],
            "reserve_identity_sha256": value[
                "reserve_identity_sha256"
            ],
            "spend_authority_identity_sha256": value[
                "spend_authority_identity_sha256"
            ],
            "action_identity_sha256": value[
                "action_identity_sha256"
            ],
            "task9_custody_identity_sha256": value[
                "task9_custody_identity_sha256"
            ],
        }
        from glm52_enforcement.canonical import canonical_sha256

        return {
            **body,
            "observation_identity_sha256": canonical_sha256(body),
        }


def _runner_launch_result(
    request: Mapping[str, object],
    outcome: object,
) -> dict[str, object]:
    outcome_status = getattr(outcome, "status", None)
    detail = getattr(outcome, "detail", None)
    if type(detail) is not dict and not isinstance(detail, Mapping):
        raise ValueError("production workflow outcome is not closed")
    detail = dict(detail)
    classification = detail.get("classification")
    if classification == "RUNNING":
        raise ValueError("production workflow is still running")
    if classification == "FAILED":
        raise ValueError("production workflow failed closed")
    if (
        outcome_status != "reconciled"
        or classification
        not in {"WORKER_ALLOCATED", "CAPACITY_EXHAUSTED"}
    ):
        raise ValueError(
            "production workflow did not produce terminal positive truth"
        )
    reconciliation_fields = {
        "classification",
        "execution_arn",
        "capacity_outcomes",
        "instance_id",
        "availability_zone",
        "workflow_output",
        "same_token_identity_sha256",
        "reserve_identity_sha256",
        "spend_authority_identity_sha256",
        "action_identity_sha256",
        "task9_custody_identity_sha256",
        "observation_identity_sha256",
    }
    if not reconciliation_fields.issubset(detail):
        raise ValueError("production workflow reconciliation is incomplete")
    workflow_reconciliation = {
        field: detail[field] for field in reconciliation_fields
    }
    accepted = classification == "WORKER_ALLOCATED"
    execution_authority = request.get("execution_authority")
    if type(execution_authority) is not dict:
        raise ValueError("runner execution authority is absent")
    spend = execution_authority.get("spend_authority")
    action = execution_authority.get("action_authority")
    if type(spend) is not dict or type(action) is not dict:
        raise ValueError("runner spend or action authority is absent")
    return {
        "status": classification,
        "accepted": accepted,
        "workflow_invocation_count": 1,
        "instance_type": "p5.48xlarge",
        "capacity_type": "ON_DEMAND",
        "execution_arn": detail["execution_arn"],
        "instance_id": detail["instance_id"],
        "availability_zone": detail["availability_zone"],
        "capacity_outcomes": detail["capacity_outcomes"],
        "workflow_output": detail["workflow_output"],
        "same_token_identity_sha256": detail[
            "same_token_identity_sha256"
        ],
        "reserve_identity_sha256": detail[
            "reserve_identity_sha256"
        ],
        "spend_authority_identity_sha256": detail[
            "spend_authority_identity_sha256"
        ],
        "action_identity_sha256": detail[
            "action_identity_sha256"
        ],
        "task9_custody_identity_sha256": detail[
            "task9_custody_identity_sha256"
        ],
        "spend_authority": (
            {
                "account_id": ACCOUNT_ID,
                "run_id": RUN_ID,
                "remaining_gpu_seconds": spend["remaining_gpu_seconds"],
                "ledger_version_id": spend["ledger_version_id"],
                "ledger_body_sha256": spend["ledger_body_sha256"],
            }
            if accepted
            else None
        ),
        "action_authority": (
            {
                "activation_id": request["activation_id"],
                "action_kind": action["action_kind"],
                "action_count": action["action_count"],
                "action_version_id": action["action_version_id"],
                "action_body_sha256": action["action_body_sha256"],
            }
            if accepted
            else None
        ),
        "terminal_contract": (
            {
                "required_terminal_markers": request[
                    "production_authority_contract"
                ]["required_terminal_markers"],
                "monitor_route": request[
                    "production_authority_contract"
                ]["monitor_route"],
            }
            if accepted
            else None
        ),
        "workflow_reconciliation": workflow_reconciliation,
    }


def _guarded_launch(
    request: Mapping[str, object],
    *,
    reconcile: bool,
) -> dict[str, object]:
    from glm52_enforcement.task10_production import (
        ProductionRequest,
        run_production,
    )

    production_request, authority, task_inputs, descriptor = (
        _task10_material(request)
    )
    execution_authority = request.get("execution_authority")
    if (
        type(execution_authority) is not dict
        or execution_authority.get("sole_sender_authority")
        != _read_sole_sender_authority(request)
    ):
        raise ValueError(
            "Task10 sole-sender authority is not positive before workflow start"
        )
    expected_reconciliation_contract = {
        "writer_owner": "TASK10_VERSIONED_PRODUCTION_WORKFLOW",
        "writer_handler": (
            "aws/glm52-gpu/lambda/"
            "task10_sole_sender_handler.py"
        ),
        "state_machine_logical_id": "Task10ProductionStateMachine",
        "state_machine_version_logical_id": (
            "Task10ProductionStateMachineVersion"
        ),
        "state_machine_name": "keep-glm52-h1g-production",
        "reconciliation_function_logical_id": (
            "Task10ProductionReconciliationFunction"
        ),
        "reconciliation_function_version_logical_id": (
            "Task10ProductionReconciliationFunctionVersion"
        ),
        "reconciliation_function_name": (
            "keep-glm52-h1g-task10-capacity-reconciliation"
        ),
        "terminal_writer_state": "RunInternalSixAzSoleSender",
        "workflow_invocation_count": 1,
        "bucket": CAMPAIGN_BUCKET,
        "key": (
            f"campaigns/{RUN_ID}/submissions/production/generations/"
            f"{getattr(authority, 'generation_text')}/workflow/"
            "LAUNCH_OUTCOME.json"
        ),
        "schema_version": 1,
        "record_type": "glm52_task10_capacity_reconciliation_v1",
        "classifications": [
            "RUNNING",
            "WORKER_ALLOCATED",
            "CAPACITY_EXHAUSTED",
            "FAILED",
        ],
        "capacity_outcome_fields": [
            "attempt",
            "availability_zone",
            "outcome",
        ],
        "exact_version_id_required": True,
        "canonical_self_hash_required": True,
    }
    if (
        request.get("workflow_reconciliation_contract")
        != expected_reconciliation_contract
    ):
        raise ValueError(
            "Task10 workflow reconciliation writer contract drifted"
        )
    if not reconcile:
        start_request = ProductionRequest(
            action="start",
            profile=PROFILE,
            authority=authority,
            task_inputs=task_inputs,
            worker_descriptor=descriptor,
        )
        started = run_production(
            start_request,
            boundary=_StepFunctionsBoundary(),
        )
        if started.status not in {"started", "reconcile-required"}:
            raise ValueError(
                "production workflow start was not positively accepted"
            )
    production_request = ProductionRequest(
        action="reconcile",
        profile=PROFILE,
        authority=authority,
        task_inputs=task_inputs,
        worker_descriptor=descriptor,
    )
    outcome = run_production(
        production_request,
        boundary=_StepFunctionsBoundary(),
    )
    return _runner_launch_result(request, outcome)


def _find_unique_source_field(value: object, field: str) -> object:
    matches = []

    def visit(item: object) -> None:
        if type(item) is dict:
            if field in item:
                matches.append(item[field])
            for child in item.values():
                visit(child)
        elif type(item) is list:
            for child in item:
                visit(child)

    visit(value)
    unique = {
        _canonical(item): item
        for item in matches
    }
    if len(unique) != 1:
        raise ValueError(
            "authenticated source field is absent or ambiguous: " + field
        )
    return next(iter(unique.values()))


def _authenticated_task11_sources(
    request: Mapping[str, object],
) -> tuple[object, object, dict[str, dict[str, object]], list[dict[str, object]]]:
    """Read and authenticate the Task10 authority and all 14 Task11 inputs."""

    _production_request, task10_authority, _task_inputs, _descriptor = (
        _task10_material(request)
    )
    boundary_coordinate = {
        "bucket": getattr(task10_authority, "task11_boundary_bucket"),
        "key": getattr(task10_authority, "task11_boundary_key"),
        "version_id": getattr(
            task10_authority,
            "task11_boundary_version_id",
        ),
        "file_sha256": getattr(
            task10_authority,
            "task11_boundary_file_sha256",
        ),
        "body_sha256": getattr(
            task10_authority,
            "task11_boundary_body_sha256",
        ),
    }
    _exact_s3_coordinate(boundary_coordinate)
    boundary_raw = _get_s3_version(
        bucket=str(boundary_coordinate["bucket"]),
        key=str(boundary_coordinate["key"]),
        version_id=str(boundary_coordinate["version_id"]),
    )
    source_root = str(_repo_root() / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from glm52_enforcement.task11_boundary import (
        BOUNDARY_INPUT_KINDS,
        task11_boundary_from_bytes,
    )

    boundary = task11_boundary_from_bytes(boundary_raw)
    if (
        boundary.activation_id != getattr(task10_authority, "activation_id")
        or boundary.generation != getattr(task10_authority, "generation")
        or boundary.generation_text
        != getattr(task10_authority, "generation_text")
        or boundary.campaign_identity_sha256
        != getattr(task10_authority, "campaign_identity_sha256")
        or boundary.state_machine_version_arn
        != getattr(task10_authority, "workflow_version_arn")
        or boundary.action_key != getattr(task10_authority, "action_key")
        or tuple(value.input_kind for value in boundary.inputs)
        != BOUNDARY_INPUT_KINDS
    ):
        raise ValueError("Task11 boundary does not bind Task10 authority")
    documents: dict[str, dict[str, object]] = {}
    coordinates = []
    for coordinate_value in boundary.inputs:
        coordinate = {
            "input_kind": coordinate_value.input_kind,
            "bucket": coordinate_value.bucket,
            "key": coordinate_value.key,
            "version_id": coordinate_value.version_id,
            "file_sha256": coordinate_value.file_sha256,
            "body_sha256": coordinate_value.body_sha256,
            "canonical_identity_sha256": (
                coordinate_value.canonical_identity_sha256
            ),
        }
        _exact_s3_coordinate(coordinate)
        raw = _get_s3_version(
            bucket=coordinate_value.bucket,
            key=coordinate_value.key,
            version_id=coordinate_value.version_id,
        )
        document = _canonical_mapping(raw, coordinate_value.input_kind)
        if coordinate_value.input_kind in documents:
            raise ValueError("Task11 boundary source kind is duplicated")
        documents[coordinate_value.input_kind] = document
        coordinates.append(coordinate)
    if set(documents) != set(BOUNDARY_INPUT_KINDS) or len(documents) != 14:
        raise ValueError("Task11 boundary source closure is incomplete")
    return task10_authority, boundary, documents, coordinates


def _ddb_key(sort_key: str) -> dict[str, dict[str, object]]:
    source_root = str(_repo_root() / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from glm52_enforcement.dynamodb import encode_item

    return encode_item({"PK": RUN_ID, "SK": sort_key})


def _read_ddb_item(sort_key: str) -> dict[str, object]:
    source_root = str(_repo_root() / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from glm52_enforcement.dynamodb import decode_item

    payload = _aws_json(
        [
            "aws",
            "dynamodb",
            "get-item",
            "--table-name",
            LEDGER_TABLE,
            "--key",
            json.dumps(_ddb_key(sort_key), separators=(",", ":")),
            "--consistent-read",
            "--return-consumed-capacity",
            "NONE",
            "--profile",
            PROFILE,
            "--region",
            REGION,
            "--no-cli-pager",
        ]
    )
    if type(payload) is not dict or type(payload.get("Item")) is not dict:
        raise ValueError("retained Task8/9 authority row is absent")
    item = decode_item(payload["Item"])
    if item.pop("PK", None) != RUN_ID or item.pop("SK", None) != sort_key:
        raise ValueError("retained Task8/9 authority key drifted")
    return item


def _build_sole_sender_authority(
    request: Mapping[str, object],
) -> tuple[object, str]:
    """Close six exact EC2 attempts over Task10, Task11, and Task8/9 truth."""

    task10_authority, boundary, source_documents, source_coordinates = (
        _authenticated_task11_sources(request)
    )
    if (
        request.get("account_id") != ACCOUNT_ID
        or request.get("region") != REGION
        or request.get("profile") != PROFILE
        or request.get("run_id") != RUN_ID
        or request.get("activation_id")
        != getattr(task10_authority, "activation_id")
    ):
        raise ValueError("sole-sender authority request drifted")
    all_sources = list(source_documents.values())
    approved_ami_id = _find_unique_source_field(
        all_sources,
        "approved_ami_id",
    )
    subnet_ids_by_zone = _find_unique_source_field(
        all_sources,
        "subnet_ids_by_availability_zone",
    )
    security_group_id = _find_unique_source_field(
        all_sources,
        "security_group_id",
    )
    source_root = str(_repo_root() / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from dataclasses import asdict
    from glm52_enforcement.canonical import canonical_sha256
    from glm52_enforcement.launch_custody import (
        LaunchParameterAuthority,
        build_deterministic_client_token,
        build_launch_parameters,
    )
    from glm52_enforcement.records import validate_record
    from glm52_enforcement.task10_sole_sender import (
        ZONES,
        build_task10_sole_sender_authority,
    )
    if (
        type(subnet_ids_by_zone) is not dict
        or set(subnet_ids_by_zone) != set(ZONES)
        or any(type(value) is not str for value in subnet_ids_by_zone.values())
        or len(set(subnet_ids_by_zone.values())) != len(ZONES)
    ):
        raise ValueError(
            "reviewed per-AZ subnet authority is not exact"
        )

    attempts = []
    retained_identities = []
    activation_id = str(request["activation_id"])
    for ordinal, zone in enumerate(ZONES, 1):
        worker = _read_ddb_item(
            f"ACTIVATION#{activation_id}#WORKER_LAUNCH#{ordinal:08d}"
        )
        liability = _read_ddb_item(
            f"ACTIVATION#{activation_id}#"
            f"WORKER_LAUNCH_LIABILITY#{ordinal:08d}"
        )
        validate_record("glm52_production_worker_launch", worker)
        validate_record(
            "glm52_production_worker_launch_liability",
            liability,
        )
        parameter_authority = LaunchParameterAuthority(
            account_id=ACCOUNT_ID,
            region=REGION,
            run_id=RUN_ID,
            campaign_identity_sha256=getattr(
                task10_authority,
                "campaign_identity_sha256",
            ),
            activation_id=activation_id,
            activation_ordinal=getattr(
                task10_authority,
                "activation_ordinal",
            ),
            generation=getattr(task10_authority, "generation"),
            action_key=getattr(task10_authority, "action_key"),
            sky_request_id=worker["sky_request_id"],
            sky_job_name=getattr(task10_authority, "sky_job_name"),
            sky_task_name="glm52-production",
            task_yaml_sha256=getattr(
                task10_authority,
                "task_yaml_sha256",
            ),
            request_body_sha256=getattr(
                task10_authority,
                "request_body_sha256",
            ),
            approved_ami_id=approved_ami_id,
            subnet_id=subnet_ids_by_zone[zone],
            security_group_id=security_group_id,
            instance_profile_name="keep-glm52-gpu-worker",
            source_identity_sha256=getattr(
                task10_authority,
                "task9_launch_identity_sha256",
            ),
        )
        parameters = build_launch_parameters(
            parameter_authority,
            allocation_ordinal=ordinal,
        )
        client_token = build_deterministic_client_token(
            parameter_authority,
            ordinal,
            parameters,
        )
        if (
            worker.get("activation_id") != activation_id
            or worker.get("activation_ordinal")
            != getattr(task10_authority, "activation_ordinal")
            or worker.get("generation")
            != getattr(task10_authority, "generation")
            or worker.get("allocation_ordinal") != ordinal
            or worker.get("campaign_identity_sha256")
            != getattr(task10_authority, "campaign_identity_sha256")
            or worker.get("sky_action_key")
            != getattr(task10_authority, "action_key")
            or worker.get("sky_job_name")
            != getattr(task10_authority, "sky_job_name")
            or worker.get("task_yaml_sha256")
            != getattr(task10_authority, "task_yaml_sha256")
            or worker.get("request_body_sha256")
            != getattr(task10_authority, "request_body_sha256")
            or worker.get("launch_parameters_sha256")
            != parameters.canonical_identity_sha256
            or worker.get("ec2_client_token") != client_token
            or liability.get("activation_id") != activation_id
            or liability.get("allocation_ordinal") != ordinal
            or liability.get("ec2_client_token") != client_token
            or liability.get("state") != "WATCHING"
            or liability.get("current_owner") is not True
            or liability.get(
                "gpu_liability_reserve_ledger_identity_sha256"
            )
            != getattr(
                task10_authority,
                "task8_spend_reserve_identity_sha256",
            )
            or liability.get(
                "gpu_liability_reserve_release_identity_sha256"
            )
            is not None
        ):
            raise ValueError("Task8/9 sole-sender custody drifted")
        attempts.append(
            {
                "attempt": ordinal,
                "availability_zone": zone,
                "subnet_availability_zone": zone,
                "allocation_ordinal": ordinal,
                "launch_parameters": asdict(parameters),
            }
        )
        retained_identities.append(
            {
                "allocation_ordinal": ordinal,
                "worker_identity_sha256": worker[
                    "canonical_body_sha256"
                ],
                "liability_identity_sha256": liability[
                    "canonical_body_sha256"
                ],
            }
        )
    authority = build_task10_sole_sender_authority(
        schema_version=1,
        record_type="glm52_task10_retained_sole_sender_authority_v1",
        account_id=ACCOUNT_ID,
        region=REGION,
        run_id=RUN_ID,
        activation_id=activation_id,
        activation_ordinal=getattr(task10_authority, "activation_ordinal"),
        generation=getattr(task10_authority, "generation"),
        generation_text=getattr(task10_authority, "generation_text"),
        availability_zones=list(ZONES),
        attempts=attempts,
        same_token_identity_sha256=getattr(
            task10_authority,
            "same_token_identity_sha256",
        ),
        reserve_identity_sha256=getattr(
            task10_authority,
            "task8_spend_reserve_identity_sha256",
        ),
        spend_authority_identity_sha256=getattr(
            task10_authority,
            "task8_spend_authority_identity_sha256",
        ),
        action_identity_sha256=getattr(
            task10_authority,
            "task9_launch_identity_sha256",
        ),
        task9_custody_identity_sha256=getattr(
            task10_authority,
            "task9_custody_identity_sha256",
        ),
    )
    source_closure_identity = canonical_sha256(
        {
            "task10_authority_identity_sha256": getattr(
                task10_authority,
                "canonical_identity_sha256",
            ),
            "task11_boundary_identity_sha256": (
                boundary.canonical_identity_sha256
            ),
            "task11_source_coordinates": source_coordinates,
            "retained_task8_task9_identities": retained_identities,
        }
    )
    return authority, source_closure_identity


def _sole_sender_coordinate(
    authority: object,
    source_closure_identity_sha256: str,
) -> dict[str, object]:
    return {
        "table_name": LEDGER_TABLE,
        "partition_key": RUN_ID,
        "sort_key": (
            f"ACTIVATION#{getattr(authority, 'activation_id')}#"
            "TASK10_SOLE_SENDER_AUTHORITY#"
            f"{getattr(authority, 'generation_text')}"
        ),
        "authority_identity_sha256": getattr(
            authority,
            "canonical_identity_sha256",
        ),
        "source_closure_identity_sha256": (
            source_closure_identity_sha256
        ),
    }


def _read_sole_sender_authority(
    request: Mapping[str, object],
) -> dict[str, object]:
    authority, closure_identity = _build_sole_sender_authority(request)
    coordinate = _sole_sender_coordinate(authority, closure_identity)
    observed = _read_ddb_item(str(coordinate["sort_key"]))
    from dataclasses import asdict

    if observed != asdict(authority):
        raise ValueError("foreign Task10 sole-sender authority exists")
    return coordinate


def _execute_sole_sender_authority_materialize(
    request: Mapping[str, object],
) -> dict[str, object]:
    authority, closure_identity = _build_sole_sender_authority(request)
    source_root = str(_repo_root() / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from glm52_enforcement.task10_authority_materialization import (
        task10_sole_sender_authority_item,
    )
    from glm52_enforcement.dynamodb import encode_item

    item = encode_item(task10_sole_sender_authority_item(authority))
    with tempfile.TemporaryDirectory(
        prefix="glm52-task13-sole-sender-authority-"
    ) as directory:
        item_path = Path(directory) / "item.json"
        item_path.write_text(
            json.dumps(item, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        try:
            _aws_json(
                [
                    "aws",
                    "dynamodb",
                    "put-item",
                    "--table-name",
                    LEDGER_TABLE,
                    "--item",
                    "file://" + str(item_path),
                    "--condition-expression",
                    (
                        "attribute_not_exists(PK) AND "
                        "attribute_not_exists(SK)"
                    ),
                    "--return-consumed-capacity",
                    "NONE",
                    "--profile",
                    PROFILE,
                    "--region",
                    REGION,
                    "--no-cli-pager",
                ]
            )
        except ValueError:
            pass
    return _read_sole_sender_authority(request)


def _reconcile_sole_sender_authority_materialize(
    request: Mapping[str, object],
) -> dict[str, object]:
    return _read_sole_sender_authority(request)


def _launch_authority_key(request: Mapping[str, object]) -> str:
    attempt = request.get("attempt")
    activation_id = request.get("activation_id")
    if (
        request.get("account_id") != ACCOUNT_ID
        or request.get("region") != REGION
        or request.get("profile") != PROFILE
        or request.get("run_id") != RUN_ID
        or type(activation_id) is not str
        or type(attempt) is not int
        or attempt < 1
        or attempt > 6
    ):
        raise ValueError("launch authority request drifted")
    return (
        f"campaigns/{RUN_ID}/authorities/task13/{activation_id}/"
        f"launch-attempt-{attempt:02d}.json"
    )


def _build_launch_authority_record(
    request: Mapping[str, object],
) -> dict[str, object]:
    _launch_authority_key(request)
    sole_sender = request.get("sole_sender_authority")
    if (
        type(sole_sender) is not dict
        or sole_sender != _read_sole_sender_authority(request)
    ):
        raise ValueError(
            "retained Task10 sole-sender authority is not positive"
        )
    _production_request, task10_authority, _task_inputs, _descriptor = (
        _task10_material(request)
    )
    from dataclasses import asdict

    source_inputs = request.get("immutable_inputs")
    if type(source_inputs) is not list:
        raise ValueError("launch authority immutable inputs are absent")
    by_kind = {
        coordinate.get("artifact_kind"): coordinate
        for coordinate in source_inputs
        if type(coordinate) is dict
    }
    required_kinds = {
        "TASK10_PRODUCTION_AUTHORITY",
        "GPU_SPEND_APPROVAL",
        "SUPPORT_APPROVAL",
        "RESIDUAL_LIABILITY_APPROVAL",
    }
    if not required_kinds.issubset(by_kind):
        raise ValueError("launch authority source coordinates are absent")
    boundary_coordinate = {
        "bucket": getattr(task10_authority, "task11_boundary_bucket"),
        "key": getattr(task10_authority, "task11_boundary_key"),
        "version_id": getattr(
            task10_authority,
            "task11_boundary_version_id",
        ),
        "file_sha256": getattr(
            task10_authority,
            "task11_boundary_file_sha256",
        ),
        "body_sha256": getattr(
            task10_authority,
            "task11_boundary_body_sha256",
        ),
    }
    _exact_s3_coordinate(boundary_coordinate)
    boundary_raw = _get_s3_version(
        bucket=str(boundary_coordinate["bucket"]),
        key=str(boundary_coordinate["key"]),
        version_id=str(boundary_coordinate["version_id"]),
    )
    source_root = str(_repo_root() / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from glm52_enforcement.task11_boundary import (
        task11_boundary_from_bytes,
    )

    boundary = task11_boundary_from_bytes(boundary_raw)
    if (
        boundary.activation_id != getattr(task10_authority, "activation_id")
        or boundary.generation != getattr(task10_authority, "generation")
        or boundary.generation_text
        != getattr(task10_authority, "generation_text")
        or boundary.campaign_identity_sha256
        != getattr(task10_authority, "campaign_identity_sha256")
        or boundary.state_machine_version_arn
        != getattr(task10_authority, "workflow_version_arn")
        or boundary.action_key != getattr(task10_authority, "action_key")
    ):
        raise ValueError("Task11 boundary does not bind Task10 authority")
    boundary_inputs = []
    for coordinate_value in boundary.inputs:
        coordinate = asdict(coordinate_value)
        _exact_s3_coordinate(coordinate)
        raw = _get_s3_version(
            bucket=coordinate_value.bucket,
            key=coordinate_value.key,
            version_id=coordinate_value.version_id,
        )
        _canonical_mapping(raw, coordinate_value.input_kind)
        boundary_inputs.append(coordinate)
    approval_values = {}
    approval_coordinates = {}
    for kind in (
        "GPU_SPEND_APPROVAL",
        "SUPPORT_APPROVAL",
        "RESIDUAL_LIABILITY_APPROVAL",
    ):
        coordinate = by_kind[kind]
        assert type(coordinate) is dict
        _exact_s3_coordinate(coordinate)
        raw = _get_s3_version(
            bucket=str(coordinate["bucket"]),
            key=str(coordinate["key"]),
            version_id=str(coordinate["version_id"]),
        )
        approval_values[kind] = _canonical_mapping(raw, kind)
        approval_coordinates[kind] = dict(coordinate)
    spend_source = approval_values["GPU_SPEND_APPROVAL"]
    remaining_seconds = _find_unique_source_field(
        spend_source,
        "remaining_gpu_seconds",
    )
    remaining_usd = _find_unique_source_field(
        spend_source,
        "remaining_gpu_usd",
    )
    record = {
        "schema_version": 1,
        "record_type": "glm52_task13_launch_authority_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": request["activation_id"],
        "attempt": request["attempt"],
        "availability_zone": request["availability_zone"],
        "instance_type": "p5.48xlarge",
        "capacity_type": "ON_DEMAND",
        "same_token_identity_sha256": getattr(
            task10_authority,
            "same_token_identity_sha256",
        ),
        "reserve_identity_sha256": getattr(
            task10_authority,
            "task8_spend_reserve_identity_sha256",
        ),
        "spend_authority_identity_sha256": getattr(
            task10_authority,
            "task8_spend_authority_identity_sha256",
        ),
        "action_identity_sha256": getattr(
            task10_authority,
            "task9_launch_identity_sha256",
        ),
        "task9_custody_identity_sha256": getattr(
            task10_authority,
            "task9_custody_identity_sha256",
        ),
        "shared_attempt_counter": request["attempt"],
        "immutable_inputs_identity_sha256": request[
            "immutable_inputs_identity_sha256"
        ],
        "sole_sender_authority": dict(sole_sender),
        "spend_authority": {
            "remaining_gpu_seconds": remaining_seconds,
            "remaining_gpu_usd": remaining_usd,
            "gpu_reserve_seconds": 900,
            "gpu_reserve_usd": "13.76",
            "root_volume_tail_usd_max": "0.01",
            "ledger_version_id": approval_coordinates[
                "GPU_SPEND_APPROVAL"
            ]["version_id"],
            "ledger_body_sha256": getattr(
                task10_authority,
                "task8_spend_authority_identity_sha256",
            ),
        },
        "action_authority": {
            "action_kind": "PRODUCTION_SUBMISSION",
            "action_count": 1,
            "action_version_id": getattr(
                task10_authority,
                "task11_boundary_version_id",
            ),
            "action_body_sha256": getattr(
                task10_authority,
                "task9_launch_identity_sha256",
            ),
        },
        "liability_action": {
            "action_kind": "SAME_TOKEN_COMPLETE",
            "attempt": request["attempt"],
            "shared_attempt_counter": request["attempt"],
            "action_version_id": getattr(
                task10_authority,
                "task11_boundary_version_id",
            ),
            "action_body_sha256": getattr(
                task10_authority,
                "task9_custody_identity_sha256",
            ),
        },
        "source_coordinates": {
            "task10_production_authority": dict(
                by_kind["TASK10_PRODUCTION_AUTHORITY"]
            ),
            "task11_boundary": boundary_coordinate,
            "task11_inputs": boundary_inputs,
            "approvals": approval_coordinates,
        },
    }
    record["canonical_identity_sha256"] = hashlib.sha256(
        _canonical(record)
    ).hexdigest()
    return record


def _validate_launch_authority_record(
    request: Mapping[str, object],
    value: Mapping[str, object],
    coordinate: Mapping[str, object],
) -> dict[str, object]:
    attempt = request.get("attempt")
    activation_id = request.get("activation_id")
    expected_record = _build_launch_authority_record(request)
    if value != expected_record:
        raise ValueError("launch authority positive truth drifted")
    _production_request, task10_authority, _task_inputs, _descriptor = (
        _task10_material(request)
    )
    authority_fields = {
        "account_id",
        "region",
        "run_id",
        "activation_id",
        "attempt",
        "availability_zone",
        "instance_type",
        "capacity_type",
        "same_token_identity_sha256",
        "reserve_identity_sha256",
        "spend_authority_identity_sha256",
        "action_identity_sha256",
        "task9_custody_identity_sha256",
        "shared_attempt_counter",
        "immutable_inputs_identity_sha256",
        "sole_sender_authority",
        "spend_authority",
        "action_authority",
        "liability_action",
    }
    expected_fields = {
        "schema_version",
        "record_type",
        *authority_fields,
        "source_coordinates",
        "canonical_identity_sha256",
    }
    unsigned = dict(value)
    identity = unsigned.pop("canonical_identity_sha256", None)
    if (
        set(value) != expected_fields
        or value["schema_version"] != 1
        or value["record_type"]
        != "glm52_task13_launch_authority_v1"
        or identity != hashlib.sha256(_canonical(unsigned)).hexdigest()
        or value["account_id"] != ACCOUNT_ID
        or value["region"] != REGION
        or value["run_id"] != RUN_ID
        or value["activation_id"] != activation_id
        or value["attempt"] != attempt
        or value["availability_zone"] != request.get("availability_zone")
        or value["instance_type"] != "p5.48xlarge"
        or value["capacity_type"] != "ON_DEMAND"
        or value["shared_attempt_counter"] != attempt
        or value["immutable_inputs_identity_sha256"]
        != request.get("immutable_inputs_identity_sha256")
        or value["sole_sender_authority"]
        != request.get("sole_sender_authority")
        or value["sole_sender_authority"]
        != _read_sole_sender_authority(request)
        or value["same_token_identity_sha256"]
        != getattr(task10_authority, "same_token_identity_sha256")
        or value["reserve_identity_sha256"]
        != getattr(
            task10_authority,
            "task8_spend_reserve_identity_sha256",
        )
        or value["spend_authority_identity_sha256"]
        != getattr(
            task10_authority,
            "task8_spend_authority_identity_sha256",
        )
        or value["action_identity_sha256"]
        != getattr(task10_authority, "task9_launch_identity_sha256")
        or value["task9_custody_identity_sha256"]
        != getattr(task10_authority, "task9_custody_identity_sha256")
    ):
        raise ValueError("launch authority positive truth drifted")
    response_body = {
        field: value[field] for field in authority_fields
    }
    response_body["authority_record"] = coordinate
    return {
        **response_body,
        "canonical_identity_sha256": hashlib.sha256(
            _canonical(response_body)
        ).hexdigest(),
    }


def _read_launch_authority(
    request: Mapping[str, object],
) -> dict[str, object]:
    value, coordinate = _read_result_record(
        key=_launch_authority_key(request)
    )
    return _validate_launch_authority_record(
        request,
        value,
        coordinate,
    )


def _execute_launch_authority_materialize(
    request: Mapping[str, object],
) -> dict[str, object]:
    expected = _build_launch_authority_record(request)
    key = _launch_authority_key(request)
    try:
        current, coordinate = _read_result_record(key=key)
    except ValueError:
        current = None
        coordinate = None
    if current is not None:
        assert coordinate is not None
        return _validate_launch_authority_record(
            request,
            current,
            coordinate,
        )
    raw = _canonical(expected) + b"\n"
    with tempfile.TemporaryDirectory(
        prefix="glm52-task13-launch-authority-"
    ) as directory:
        body_path = Path(directory) / "authority.json"
        body_path.write_bytes(raw)
        try:
            _aws_json(
                [
                    "aws",
                    "s3api",
                    "put-object",
                    "--bucket",
                    CAMPAIGN_BUCKET,
                    "--key",
                    key,
                    "--body",
                    str(body_path),
                    "--content-type",
                    "application/json",
                    "--if-none-match",
                    "*",
                    "--metadata",
                    (
                        "record-type="
                        "glm52_task13_launch_authority_v1,"
                        "canonical-identity-sha256="
                        + str(expected["canonical_identity_sha256"])
                    ),
                    "--profile",
                    PROFILE,
                    "--region",
                    REGION,
                    "--no-cli-pager",
                ]
            )
        except ValueError:
            pass
    value, coordinate = _read_result_record(key=key)
    return _validate_launch_authority_record(
        request,
        value,
        coordinate,
    )


def _reconcile_launch_authority_materialize(
    request: Mapping[str, object],
) -> dict[str, object]:
    return _read_launch_authority(request)


def _monitor(request: Mapping[str, object]) -> dict[str, object]:
    expected_argv = [
        "aws/glm52-gpu/scripts/sky_campaign_break_glass.sh",
        "status",
    ]
    expected_environment = {
        "AWS_PROFILE": PROFILE,
        "CAMPAIGN_DESCRIPTOR": request.get("descriptor_path"),
        "SKY_BIN": (
            "/Users/jack.mazac/.local/share/keep/"
            "skypilot-0.13.0/bin/sky"
        ),
        "SKYPILOT_CONFIG": (
            "/Users/jack.mazac/.local/share/keep/"
            "skypilot-0.13.0/server-config.yaml"
        ),
    }
    if (
        request.get("read_only") is not True
        or request.get("account_id") != ACCOUNT_ID
        or request.get("region") != REGION
        or request.get("profile") != PROFILE
        or request.get("argv") != expected_argv
        or request.get("environment") != expected_environment
        or type(request.get("descriptor_path")) is not str
    ):
        raise ValueError("monitor contract drifted")
    executable = _repository_executable(expected_argv[0])
    completed = subprocess.run(
        [str(executable), "status"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        shell=False,
        env={
            "HOME": os.environ.get("HOME", ""),
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
            **expected_environment,
        },
        timeout=180,
    )
    if completed.returncode != 0:
        raise ValueError(
            "guarded monitor route refused: "
            + completed.stderr.decode("utf-8", "replace")[:500]
        )
    try:
        text = completed.stdout.decode("utf-8").lstrip()
        queue, _end = json.JSONDecoder().raw_decode(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("guarded monitor result is not JSON-first") from error

    status_values: list[str] = []

    def collect(value: object) -> None:
        if type(value) is dict:
            name = value.get("name", value.get("job_name"))
            status = value.get("status")
            if (
                name in {None, RUN_ID}
                and type(status) is str
            ):
                status_values.append(status.upper())
            for nested in value.values():
                if type(nested) in {dict, list}:
                    collect(nested)
        elif type(value) is list:
            for nested in value:
                collect(nested)

    collect(queue)
    mapping = {
        "PENDING": "STARTING",
        "SUBMITTED": "STARTING",
        "STARTING": "STARTING",
        "INIT": "STARTING",
        "RUNNING": "RUNNING",
        "SUCCEEDED": "SUCCEEDED",
        "SUCCESS": "SUCCEEDED",
        "FAILED": "FAILED",
        "CANCELLED": "FAILED",
        "CANCELED": "FAILED",
    }
    mapped = {mapping[item] for item in status_values if item in mapping}
    if len(mapped) != 1:
        raise ValueError("guarded monitor status is ambiguous")
    return {
        "status": mapped.pop(),
        "instance_ids": _active_p5_ids(),
    }


def _terminal_proof(
    request: Mapping[str, object],
) -> dict[str, object]:
    if (
        request.get("account_id") != ACCOUNT_ID
        or request.get("region") != REGION
        or request.get("profile") != PROFILE
        or request.get("run_id") != RUN_ID
        or request.get("required_terminal_markers")
        != ["CAMPAIGN_DRAINED.json", "TERMINAL_VERIFIED.json"]
        or request.get("required_sky_state") != "SUCCEEDED"
        or request.get("required_billable_p5_instance_count") != 0
        or type(request.get("activation_id")) is not str
        or type(request.get("monitor_contract")) is not dict
    ):
        raise ValueError("terminal proof request drifted")
    observed = _monitor(request["monitor_contract"])
    if observed != {"status": "SUCCEEDED", "instance_ids": []}:
        raise ValueError("campaign is not terminal with zero billable P5")
    source_root = str(_repo_root() / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from glm52_enforcement.task13_terminal_evidence import (
        CAMPAIGN_DRAINED_KEY,
        GPU_SPEND_LEDGER_KEY,
        MODEL_BUCKET,
        TASK13_TERMINAL_PROOF_KEY,
        TERMINAL_VERIFIED_KEY,
        validate_marker_coordinate,
        validate_task13_terminal_proof,
    )

    proof, _proof_coordinate = _read_result_record(
        key=TASK13_TERMINAL_PROOF_KEY
    )
    drained_coordinate = validate_marker_coordinate(
        proof.get("campaign_drained"),
        expected_key=CAMPAIGN_DRAINED_KEY,
    )
    verified_coordinate = validate_marker_coordinate(
        proof.get("terminal_verified"),
        expected_key=TERMINAL_VERIFIED_KEY,
    )

    def exact_marker(
        coordinate: Mapping[str, object],
        *,
        label: str,
    ) -> dict[str, object]:
        raw = _get_s3_version(
            bucket=CAMPAIGN_BUCKET,
            key=str(coordinate["key"]),
            version_id=str(coordinate["version_id"]),
        )
        value = _canonical_mapping(raw, label)
        observed_coordinate = _self_hashed_coordinate(
            key=str(coordinate["key"]),
            version_id=str(coordinate["version_id"]),
            raw=raw,
            value=value,
        )
        if observed_coordinate != coordinate:
            raise ValueError(label + " immutable coordinate drifted")
        return value

    campaign_drained_value = exact_marker(
        drained_coordinate,
        label="campaign-drained source",
    )
    terminal_verified_value = exact_marker(
        verified_coordinate,
        label="terminal-verified source",
    )
    settlement = proof.get("settlement")
    spend_coordinate = (
        settlement.get("gpu_spend_ledger")
        if type(settlement) is dict
        else None
    )
    if (
        type(spend_coordinate) is not dict
        or set(spend_coordinate)
        != {
            "bucket",
            "key",
            "version_id",
            "file_sha256",
            "body_sha256",
            "head_record_sha256",
            "canonical_identity_sha256",
        }
        or spend_coordinate.get("bucket") != MODEL_BUCKET
        or spend_coordinate.get("key") != GPU_SPEND_LEDGER_KEY
        or type(spend_coordinate.get("version_id")) is not str
        or not spend_coordinate["version_id"]
        or spend_coordinate["version_id"].lower()
        in {"null", "latest", "$latest"}
    ):
        raise ValueError("terminal proof spend-ledger coordinate drifted")
    spend_ledger_raw = _get_s3_version(
        bucket=str(spend_coordinate.get("bucket")),
        key=str(spend_coordinate.get("key")),
        version_id=str(spend_coordinate.get("version_id")),
    )
    exact_proof = validate_task13_terminal_proof(
        proof,
        terminal_verified_value=terminal_verified_value,
        campaign_drained_value=campaign_drained_value,
        spend_ledger_raw=spend_ledger_raw,
    )
    if exact_proof["activation_id"] != request["activation_id"]:
        raise ValueError("terminal proof activation drifted")
    return {
        "proof": exact_proof,
        "terminal_verified_value": terminal_verified_value,
        "campaign_drained_value": campaign_drained_value,
        "spend_ledger_raw": spend_ledger_raw,
    }


def _task12_orphan_paths(
    request: Mapping[str, object],
) -> dict[str, Path]:
    precreate = Path(str(request.get("precreate_output", "")))
    postcreate = Path(str(request.get("postcreate_output_dir", "")))
    manifest = Path(str(request.get("activation_manifest", "")))
    parent = precreate.parent
    if (
        not precreate.is_absolute()
        or not postcreate.is_absolute()
        or not manifest.is_absolute()
        or precreate.name != "PRECREATE_ORPHAN_BASELINE.json"
        or postcreate != parent / "support-postcreate"
        or manifest != parent / "ACTIVATION_ORPHAN_AUTHORITY.json"
        or not re.fullmatch(
            r"/tmp/glm52-full-run-[0-9]{8}/production/task12/orphans",
            str(parent),
        )
    ):
        raise ValueError("Task 12 orphan authority paths drifted")
    return {
        "precreate": precreate,
        "postcreate": postcreate,
        "manifest": manifest,
    }


def _task12_orphan_request(
    request: Mapping[str, object],
    *,
    phase: str,
) -> dict[str, Path]:
    phase_fields = {
        "PRECREATE": {
            "support_change_set_name",
            "support_change_set_arn",
        },
        "POSTCREATE": {
            "precreate_authority",
            "support_inventory_identity_sha256",
        },
    }
    if (
        phase not in phase_fields
        or set(request) != _TASK12_ORPHAN_COMMON_FIELDS | phase_fields[phase]
        or request.get("materializer") != _TASK12_ORPHAN_MATERIALIZER
        or request.get("support_postcreate_materializer")
        != _SUPPORT_POSTCREATE_MATERIALIZER
        or request.get("account_id") != ACCOUNT_ID
        or request.get("region") != REGION
        or request.get("profile") != PROFILE
        or request.get("run_id") != RUN_ID
        or type(request.get("activation_id")) is not str
        or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",
            str(request.get("activation_id")),
        )
        or type(request.get("support_stack_id")) is not str
        or re.fullmatch(
            r"arn:aws:cloudformation:us-west-2:246813579024:stack/"
            r"keep-glm52-h1g-support/[0-9a-f-]{36}",
            str(request.get("support_stack_id")),
        )
        is None
        or type(request.get("fence_stack_id")) is not str
        or re.fullmatch(
            r"arn:aws:cloudformation:us-west-2:246813579024:stack/"
            r"keep-glm52-h1g-fence/[0-9a-f-]{36}",
            str(request.get("fence_stack_id")),
        )
        is None
        or request.get("lifecycle_resource_id")
        != (
            "arn:aws:states:us-west-2:246813579024:"
            "stateMachine:keep-glm52-h1g-retained-lifecycle"
        )
        or request.get("lifecycle_cost_class") != "STEP_FUNCTIONS_RETAINED"
        or request.get("source_publisher_arn")
        != (
            "arn:aws:iam::246813579024:"
            "role/keep-glm52-source-publisher"
        )
        or request.get("settling_window_seconds") != 60
    ):
        raise ValueError("Task 12 orphan authority request drifted")
    if phase == "PRECREATE":
        change_set_name = request.get("support_change_set_name")
        if (
            type(change_set_name) is not str
            or not re.fullmatch(r"[A-Za-z0-9][-A-Za-z0-9]{0,127}", change_set_name)
            or request.get("support_change_set_arn")
            != (
                "arn:aws:cloudformation:us-west-2:246813579024:"
                "changeSet/"
                + change_set_name
                + "/"
                + str(request.get("support_change_set_arn", "")).rsplit(
                    "/", 1
                )[-1]
            )
            or re.fullmatch(
                r"[0-9a-f-]{36}",
                str(request.get("support_change_set_arn", "")).rsplit(
                    "/", 1
                )[-1],
            )
            is None
        ):
            raise ValueError("support change-set authority drifted")
    else:
        identity = request.get("support_inventory_identity_sha256")
        if type(identity) is not str or re.fullmatch(r"[0-9a-f]{64}", identity) is None:
            raise ValueError("support inventory identity drifted")
    return _task12_orphan_paths(request)


def _reviewed_artifact_bytes(
    coordinate: object,
    *,
    artifact_kind: str,
) -> tuple[bytes, dict[str, object]]:
    if (
        type(coordinate) is not dict
        or set(coordinate)
        != {
            "artifact_kind",
            "bucket",
            "key",
            "version_id",
            "file_sha256",
            "body_sha256",
        }
        or coordinate.get("artifact_kind") != artifact_kind
        or type(coordinate.get("file_sha256")) is not str
        or type(coordinate.get("body_sha256")) is not str
    ):
        raise ValueError(artifact_kind + " reviewed coordinate drifted")
    _exact_s3_coordinate(coordinate)
    raw = _get_s3_version(
        bucket=str(coordinate["bucket"]),
        key=str(coordinate["key"]),
        version_id=str(coordinate["version_id"]),
    )
    value = _canonical_mapping(raw, artifact_kind)
    body_candidates = {
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(_canonical(value)).hexdigest(),
    }
    for field in ("canonical_identity_sha256", "canonical_body_sha256"):
        if type(value.get(field)) is str:
            body_candidates.add(str(value[field]))
    if (
        hashlib.sha256(raw).hexdigest() != coordinate["file_sha256"]
        or coordinate["body_sha256"] not in body_candidates
    ):
        raise ValueError(artifact_kind + " reviewed bytes drifted")
    return raw, value


def _support_build_inputs(
    request: Mapping[str, object],
) -> tuple[bytes, object]:
    raw, value = _reviewed_artifact_bytes(
        request.get("support_inputs"),
        artifact_kind="SUPPORT_INPUTS",
    )
    source_root = str(_repo_root() / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from glm52_enforcement.support_plane import (
        support_build_inputs_from_mapping,
    )

    inputs = support_build_inputs_from_mapping(value)
    if (
        inputs.account_id != ACCOUNT_ID
        or inputs.region != REGION
        or inputs.run_id != RUN_ID
        or inputs.activation_id != request["activation_id"]
        or inputs.fence_stack_id != request["fence_stack_id"]
    ):
        raise ValueError("reviewed support build inputs are foreign")
    return raw, inputs


def _write_new_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _run_materializer(argv: list[str], *, timeout: int = 1800) -> bytes:
    completed = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        shell=False,
        env={
            "HOME": os.environ.get("HOME", ""),
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
            "AWS_MAX_ATTEMPTS": "1",
            "AWS_RETRY_MODE": "standard",
            "AWS_PAGER": "",
        },
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise ValueError(
            "Task 12 materializer refused: "
            + completed.stderr.decode("utf-8", "replace")[:500]
        )
    return completed.stdout.strip()


def _precreate_artifact(
    *,
    request: Mapping[str, object],
    path: Path,
    inputs: object,
) -> dict[str, object]:
    raw = path.read_bytes()
    value = _canonical_mapping(raw, "Task 12 PRECREATE baseline")
    body = dict(value)
    identity = body.pop("canonical_body_sha256", None)
    expected_retained = [
        {
            "resource_type": "LEDGER",
            "resource_id": inputs.ledger_table_name,
            "cost_class": "DDB_RETAINED",
        },
        {
            "resource_type": "KMS_KEY",
            "resource_id": inputs.retained_kms_key_arn,
            "cost_class": "KMS_RETAINED",
        },
        {
            "resource_type": "EVIDENCE_BUCKET",
            "resource_id": inputs.model_bucket_name,
            "cost_class": "S3_RETAINED",
        },
        {
            "resource_type": "PRODUCTION_FENCE_STACK",
            "resource_id": request["fence_stack_id"],
            "cost_class": "CFN_RETAINED",
        },
        {
            "resource_type": "LIFECYCLE_RESOURCE",
            "resource_id": request["lifecycle_resource_id"],
            "cost_class": request["lifecycle_cost_class"],
        },
        {
            "resource_type": "SOURCE_PUBLISHER_IDENTITY",
            "resource_id": request["source_publisher_arn"],
            "cost_class": "IAM_RETAINED",
        },
    ]
    if (
        value.get("record_type")
        != "glm52_task12_orphan_precreate_baseline_v1"
        or value.get("schema_version") != 1
        or value.get("phase") != "PRECREATE"
        or value.get("account_id") != ACCOUNT_ID
        or value.get("region") != REGION
        or value.get("run_id") != RUN_ID
        or value.get("activation_id") != request["activation_id"]
        or value.get("retained_kms_key_arn") != inputs.retained_kms_key_arn
        or value.get("captured_before_support_mutation") is not True
        or value.get("expected_retained") != expected_retained
        or value.get("pagination_complete") is not True
        or identity != hashlib.sha256(_canonical(body)).hexdigest()
    ):
        raise ValueError("Task 12 PRECREATE baseline is foreign")
    artifact_body = {
        "record_type": "glm52_task12_orphan_precreate_artifact_v1",
        "activation_id": request["activation_id"],
        "path": str(path),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "body_sha256": identity,
    }
    return {
        **artifact_body,
        "canonical_identity_sha256": hashlib.sha256(
            _canonical(artifact_body)
        ).hexdigest(),
    }


def _execute_orphan_authority_precreate(
    request: Mapping[str, object],
) -> dict[str, object]:
    paths = _task12_orphan_request(request, phase="PRECREATE")
    _raw, inputs = _support_build_inputs(request)
    path = paths["precreate"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        executable = _pinned_repository_executable(
            _TASK12_ORPHAN_MATERIALIZER,
            _TASK12_ORPHAN_MATERIALIZER_SHA256,
        )
        _run_materializer(
            [
                sys.executable,
                str(executable),
                "--profile",
                PROFILE,
                "PRECREATE",
                "--activation-id",
                str(request["activation_id"]),
                "--kms-key-arn",
                inputs.retained_kms_key_arn,
                "--ledger-table",
                inputs.ledger_table_name,
                "--evidence-bucket",
                inputs.model_bucket_name,
                "--fence-stack-id",
                str(request["fence_stack_id"]),
                "--lifecycle-resource-id",
                str(request["lifecycle_resource_id"]),
                "--lifecycle-cost-class",
                str(request["lifecycle_cost_class"]),
                "--source-publisher-arn",
                str(request["source_publisher_arn"]),
                "--support-stack-name",
                str(request["support_stack_id"]),
                "--change-set-name",
                str(request["support_change_set_name"]),
                "--output",
                str(path),
            ]
        )
    return _precreate_artifact(
        request=request,
        path=path,
        inputs=inputs,
    )


def _reconcile_orphan_authority_precreate(
    request: Mapping[str, object],
) -> dict[str, object]:
    paths = _task12_orphan_request(request, phase="PRECREATE")
    _raw, inputs = _support_build_inputs(request)
    if not paths["precreate"].is_file():
        raise ValueError("Task 12 PRECREATE positive truth is absent")
    return _precreate_artifact(
        request=request,
        path=paths["precreate"],
        inputs=inputs,
    )


def _local_coordinate(path: Path, label: str) -> dict[str, object]:
    raw = path.read_bytes()
    value = _canonical_mapping(raw, label)
    body_identity = value.get("canonical_body_sha256")
    if type(body_identity) is not str:
        body_identity = hashlib.sha256(_canonical(value)).hexdigest()
    return {
        "path": str(path),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "body_sha256": body_identity,
    }


def _support_postcreate_artifacts(
    *,
    request: Mapping[str, object],
    paths: Mapping[str, Path],
    support_inputs_raw: bytes,
    support_template_raw: bytes,
) -> tuple[Path, Path]:
    output = paths["postcreate"]
    expected_names = {
        "support-deletion-authority-v1.json",
        "support-materialized-inputs-v1.json",
        "support-postcreate-manifest-v1.json",
        "support-retained-augmentation-v1.json",
        "TASK9_DEPLOYED_IDENTITY.json",
    }
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="glm52-task13-support-postcreate-"
        ) as directory:
            inputs_path = Path(directory) / "support-inputs.json"
            template_path = Path(directory) / "support-template.json"
            _write_new_bytes(inputs_path, support_inputs_raw)
            _write_new_bytes(template_path, support_template_raw)
            executable = _pinned_repository_executable(
                _SUPPORT_POSTCREATE_MATERIALIZER,
                _SUPPORT_POSTCREATE_MATERIALIZER_SHA256,
            )
            stdout = _run_materializer(
                [
                    sys.executable,
                    str(executable),
                    "--profile",
                    PROFILE,
                    "--region",
                    REGION,
                    "--support-stack-id",
                    str(request["support_stack_id"]),
                    "--inputs",
                    str(inputs_path),
                    "--support-template",
                    str(template_path),
                    "--output-dir",
                    str(output),
                ]
            )
            if stdout:
                raise ValueError(
                    "support postcreate materializer emitted foreign output"
                )
    if (
        not output.is_dir()
        or output.is_symlink()
        or {item.name for item in output.iterdir()} != expected_names
        or any(not item.is_file() or item.is_symlink() for item in output.iterdir())
    ):
        raise ValueError("support postcreate artifact set is incomplete")
    for item in output.iterdir():
        _canonical_mapping(item.read_bytes(), item.name)
    return (
        output / "support-postcreate-manifest-v1.json",
        output / "support-materialized-inputs-v1.json",
    )


def _task12_postcreate_coordinate(
    *,
    request: Mapping[str, object],
    paths: Mapping[str, Path],
    inputs: object,
    materialized_inputs_path: Path,
    postcreate_manifest_path: Path,
) -> dict[str, object]:
    executable = _pinned_repository_executable(
        _TASK12_ORPHAN_MATERIALIZER,
        _TASK12_ORPHAN_MATERIALIZER_SHA256,
    )
    stdout = _run_materializer(
        [
            sys.executable,
            str(executable),
            "--profile",
            PROFILE,
            "POSTCREATE",
            "--activation-id",
            str(request["activation_id"]),
            "--baseline",
            str(paths["precreate"]),
            "--support-inputs",
            str(materialized_inputs_path),
            "--postcreate-manifest",
            str(postcreate_manifest_path),
            "--cloudtrail-start",
            inputs.activation_started_at,
            "--cloudtrail-end",
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "--settling-window-seconds",
            str(request["settling_window_seconds"]),
            "--authority-bucket",
            inputs.model_bucket_name,
        ]
    )
    coordinate = _json_mapping(stdout, "Task 12 activation coordinate")
    if (
        coordinate.get("record_type")
        != "glm52_task12_activation_orphan_coordinate_v1"
        or coordinate.get("account_id") != ACCOUNT_ID
        or coordinate.get("region") != REGION
        or coordinate.get("run_id") != RUN_ID
        or coordinate.get("activation_id") != request["activation_id"]
        or coordinate.get("bucket") != inputs.model_bucket_name
        or type(coordinate.get("version_id")) is not str
        or not coordinate["version_id"]
        or coordinate.get("key")
        != (
            f"campaigns/{RUN_ID}/task12/orphans/"
            f"{request['activation_id']}/"
            f"{coordinate.get('authority_body_sha256')}.json"
        )
    ):
        raise ValueError("Task 12 activation coordinate is foreign")
    return coordinate


def _activation_manifest(
    *,
    request: Mapping[str, object],
    paths: Mapping[str, Path],
    precreate: Mapping[str, object],
    postcreate_manifest_path: Path,
    materialized_inputs_path: Path,
    activation_authority: Mapping[str, object],
) -> dict[str, object]:
    support = {
        **_local_coordinate(
            postcreate_manifest_path,
            "support postcreate manifest",
        ),
        "materialized_inputs": _local_coordinate(
            materialized_inputs_path,
            "materialized support inputs",
        ),
    }
    body = {
        "schema_version": 1,
        "record_type": (
            "glm52_task13_task12_orphan_activation_manifest_v1"
        ),
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": request["activation_id"],
        "precreate_authority": dict(precreate),
        "support_postcreate_manifest": support,
        "activation_authority": dict(activation_authority),
        "support_inventory_identity_sha256": request[
            "support_inventory_identity_sha256"
        ],
        "path": str(paths["manifest"]),
    }
    value = {
        **body,
        "canonical_identity_sha256": hashlib.sha256(
            _canonical(body)
        ).hexdigest(),
    }
    path = paths["manifest"]
    raw = _canonical(value)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("foreign Task 12 activation manifest exists")
    else:
        _write_new_bytes(path, raw)
    return value


def _execute_orphan_authority_postcreate(
    request: Mapping[str, object],
) -> dict[str, object]:
    paths = _task12_orphan_request(request, phase="POSTCREATE")
    support_inputs_raw, inputs = _support_build_inputs(request)
    support_template_raw, _template = _reviewed_artifact_bytes(
        request.get("support_template"),
        artifact_kind="SUPPORT_TEMPLATE",
    )
    precreate = _precreate_artifact(
        request=request,
        path=paths["precreate"],
        inputs=inputs,
    )
    if request.get("precreate_authority") != precreate:
        raise ValueError("Task 12 PRECREATE caller authority drifted")
    postcreate_manifest_path, materialized_inputs_path = (
        _support_postcreate_artifacts(
            request=request,
            paths=paths,
            support_inputs_raw=support_inputs_raw,
            support_template_raw=support_template_raw,
        )
    )
    activation_authority = _task12_postcreate_coordinate(
        request=request,
        paths=paths,
        inputs=inputs,
        materialized_inputs_path=materialized_inputs_path,
        postcreate_manifest_path=postcreate_manifest_path,
    )
    return _activation_manifest(
        request=request,
        paths=paths,
        precreate=precreate,
        postcreate_manifest_path=postcreate_manifest_path,
        materialized_inputs_path=materialized_inputs_path,
        activation_authority=activation_authority,
    )


def _reconcile_orphan_authority_postcreate(
    request: Mapping[str, object],
) -> dict[str, object]:
    return _execute_orphan_authority_postcreate(request)


def _execute_qualification_cache_seed(
    request: Mapping[str, object],
) -> dict[str, object]:
    _run_repository_driver(
        request,
        operation_kind="qualification-cache-seed",
    )
    return _seed_result()


def _execute_h100_qualification(
    request: Mapping[str, object],
) -> dict[str, object]:
    _run_repository_driver(
        request,
        operation_kind="h100-qualification",
    )
    return _h100_result()


def _execute_guarded_launch(
    request: Mapping[str, object],
) -> dict[str, object]:
    return _guarded_launch(request, reconcile=False)


def _reconcile_bootstrap(
    request: Mapping[str, object],
) -> dict[str, object]:
    result = _bootstrap_state(request)
    return _bootstrap_stack_result(request, state=result)


def _reconcile_collector_invoke(
    request: Mapping[str, object],
) -> dict[str, object]:
    return _reconcile_lambda("collector-invoke", request)


def _reconcile_finalize_invoke(
    request: Mapping[str, object],
) -> dict[str, object]:
    return _reconcile_lambda("finalize-invoke", request)


def _reconcile_qualification_cache_seed(
    request: Mapping[str, object],
) -> dict[str, object]:
    return _seed_result()


def _reconcile_h100_qualification(
    request: Mapping[str, object],
) -> dict[str, object]:
    return _h100_result()


def _reconcile_guarded_launch(
    request: Mapping[str, object],
) -> dict[str, object]:
    return _guarded_launch(request, reconcile=True)


def _inspect_staged_infrastructure_adoption(
    request: Mapping[str, object],
) -> dict[str, object]:
    source_root = str(_repo_root() / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from glm52_enforcement.task13_campaign_package import (
        CampaignPackageError,
        validate_staged_infrastructure_evidence,
    )

    fixed = request.get("fixed_artifacts")
    final = request.get("final_readback")
    if type(fixed) is not list or type(final) is not dict:
        raise ValueError("staged infrastructure adoption request drifted")
    artifacts = {
        str(row.get("artifact_kind")): row
        for row in fixed
        if type(row) is dict
    }
    try:
        evidence = validate_staged_infrastructure_evidence(
            request,
            activation_id=str(request.get("activation_id", "")),
            retained_stack_id=str(
                final.get("retained_stack_id", "")
            ),
            artifacts=artifacts,
        )
    except CampaignPackageError as error:
        raise ValueError(
            "staged infrastructure adoption request drifted"
        ) from error

    identity = _aws_json(
        [
            "aws",
            "sts",
            "get-caller-identity",
            "--profile",
            PROFILE,
            "--region",
            REGION,
            "--no-cli-pager",
        ]
    )
    if type(identity) is not dict or identity.get("Account") != ACCOUNT_ID:
        raise ValueError("staged adoption caller account drifted")

    for coordinate in fixed:
        assert type(coordinate) is dict
        if _exact_s3_coordinate(coordinate) != coordinate:
            raise ValueError(
                "staged adoption immutable artifact drifted"
            )

    support = evidence["support_stack"]
    assert type(support) is dict
    for field in (
        "support_lambda_archive",
        "cryptography_layer_archive",
    ):
        coordinate = support[field]
        assert type(coordinate) is dict
        if _exact_runtime_archive(coordinate) != coordinate:
            raise ValueError(
                "staged adoption runtime archive drifted"
            )
    layer_arn = str(support["cryptography_layer_version_arn"])
    layer_parts = layer_arn.rsplit(":", 2)
    if len(layer_parts) != 3:
        raise ValueError("staged adoption layer identity drifted")
    layer = _aws_json(
        [
            "aws",
            "lambda",
            "get-layer-version",
            "--layer-name",
            layer_parts[-2],
            "--version-number",
            layer_parts[-1],
            "--profile",
            PROFILE,
            "--region",
            REGION,
            "--no-cli-pager",
        ]
    )
    if (
        type(layer) is not dict
        or layer.get("LayerVersionArn") != layer_arn
        or layer.get("CodeSha256")
        != support["cryptography_layer_code_sha256"]
        or layer.get("CompatibleRuntimes") != ["python3.12"]
        or layer.get("CompatibleArchitectures") != ["x86_64"]
    ):
        raise ValueError("staged adoption layer readback drifted")

    exact_tags = {
        "CampaignRunId": RUN_ID,
        "DeploymentState": "DISABLED",
        "Task13ActivationId": evidence["activation_id"],
    }
    stack_contracts = (
        (
            "retained_stack_id",
            "retained_stack_status",
            DEPLOYMENT_ROLE_ARN,
            "retained_template_sha256",
        ),
        (
            "fence_stack_id",
            "fence_stack_status",
            FENCE_ROLE_ARN,
            "fence_template_sha256",
        ),
        (
            "support_stack_id",
            "support_stack_status",
            DEPLOYMENT_ROLE_ARN,
            "support_template_sha256",
        ),
    )
    pending = 0
    for id_field, status_field, role_arn, template_sha_field in (
        stack_contracts
    ):
        stack_id = final[id_field]
        described = _aws_json(
            [
                "aws",
                "cloudformation",
                "describe-stacks",
                "--stack-name",
                str(stack_id),
                "--profile",
                PROFILE,
                "--region",
                REGION,
                "--no-cli-pager",
            ]
        )
        stacks = (
            described.get("Stacks", [])
            if type(described) is dict
            else []
        )
        tags = (
            {
                row.get("Key"): row.get("Value")
                for row in stacks[0].get("Tags", [])
                if type(row) is dict
            }
            if len(stacks) == 1 and type(stacks[0]) is dict
            else {}
        )
        if (
            len(stacks) != 1
            or type(stacks[0]) is not dict
            or stacks[0].get("StackId") != stack_id
            or stacks[0].get("StackStatus") != final[status_field]
            or stacks[0].get("RoleARN") != role_arn
            or stacks[0].get("EnableTerminationProtection") is not True
            or tags != exact_tags
        ):
            raise ValueError("staged adoption stack readback drifted")
        template = _aws_json(
            [
                "aws",
                "cloudformation",
                "get-template",
                "--stack-name",
                str(stack_id),
                "--template-stage",
                "Original",
                "--profile",
                PROFILE,
                "--region",
                REGION,
                "--no-cli-pager",
            ]
        )
        template_body = (
            template.get("TemplateBody")
            if type(template) is dict
            else None
        )
        if type(template_body) is str:
            try:
                template_body = json.loads(template_body)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "staged adoption template readback drifted"
                ) from error
        if (
            type(template_body) is not dict
            or hashlib.sha256(_canonical(template_body)).hexdigest()
            != final[template_sha_field]
        ):
            raise ValueError(
                "staged adoption template readback drifted"
            )
        template_resources = template_body.get("Resources")
        if type(template_resources) is not dict:
            raise ValueError(
                "staged adoption template resource inventory drifted"
            )
        listed_resources = _aws_json(
            [
                "aws",
                "cloudformation",
                "list-stack-resources",
                "--stack-name",
                str(stack_id),
                "--profile",
                PROFILE,
                "--region",
                REGION,
                "--no-cli-pager",
            ]
        )
        summaries = (
            listed_resources.get("StackResourceSummaries", [])
            if type(listed_resources) is dict
            else []
        )
        observed_resources = {
            str(row.get("LogicalResourceId")): row
            for row in summaries
            if type(row) is dict
            and type(row.get("LogicalResourceId")) is str
        }
        if (
            type(summaries) is not list
            or len(observed_resources) != len(summaries)
            or set(observed_resources) != set(template_resources)
            or any(
                type(template_resources[logical_id]) is not dict
                or row.get("ResourceType")
                != template_resources[logical_id].get("Type")
                or type(row.get("ResourceStatus")) is not str
                or not str(row["ResourceStatus"]).endswith("_COMPLETE")
                for logical_id, row in observed_resources.items()
            )
        ):
            raise ValueError(
                "staged adoption template resource inventory drifted"
            )
        if id_field == "support_stack_id":
            _exact_support_lambda_readback(
                template_resources=template_resources,
                observed_resources=observed_resources,
                support_evidence=support,
            )
        listed = _aws_json(
            [
                "aws",
                "cloudformation",
                "list-change-sets",
                "--stack-name",
                str(stack_id),
                "--profile",
                PROFILE,
                "--region",
                REGION,
                "--no-cli-pager",
            ]
        )
        summaries = (
            listed.get("Summaries", [])
            if type(listed) is dict
            else []
        )
        if type(summaries) is not list:
            raise ValueError(
                "staged adoption change-set inventory drifted"
            )
        pending += sum(
            type(row) is dict
            and (
                row.get("Status")
                in {"CREATE_PENDING", "CREATE_IN_PROGRESS"}
                or (
                    row.get("Status") == "CREATE_COMPLETE"
                    and row.get("ExecutionStatus") == "AVAILABLE"
                )
            )
            for row in summaries
        )
    if pending != final["pending_change_sets"]:
        raise ValueError("staged adoption pending change sets drifted")
    if (
        _staged_worker_launch_count(str(evidence["activation_id"]))
        != final["worker_activation_attempts"]
    ):
        raise ValueError("staged adoption worker ledger drifted")
    if (
        _staged_raw_ec2_launch_count(str(evidence["activation_id"]))
        != final["raw_ec2_launch_calls"]
    ):
        raise ValueError("staged adoption CloudTrail history drifted")
    if _active_p5_ids() != final["active_p5_instance_ids"]:
        raise ValueError("staged adoption worker inventory drifted")
    return evidence


def _execute_custody_identity(
    authority: Mapping[str, object],
    capability: Mapping[str, object],
) -> str:
    return _canonical_sha256(
        {
            "stage": authority["stage"],
            "operation_kind": capability["operation_kind"],
            "operation_id": capability["operation_id"],
            "request_identity_sha256": capability[
                "request_identity_sha256"
            ],
        }
    )


def _operation_custody_key(
    authority: Mapping[str, object],
    capability: Mapping[str, object],
) -> dict[str, dict[str, str]]:
    return {
        "PK": {"S": "RUN#" + RUN_ID},
        "SK": {
            "S": (
                "ACTIVATION#"
                + str(authority["activation_id"])
                + "#TASK13_OPERATION#"
                + _execute_custody_identity(authority, capability)
            )
        },
    }


def _read_operation_custody(
    authority: Mapping[str, object],
    capability: Mapping[str, object],
) -> dict[str, str] | None:
    response = _aws_json(
        [
            "aws",
            "dynamodb",
            "get-item",
            "--table-name",
            LEDGER_TABLE,
            "--key",
            json.dumps(
                _operation_custody_key(authority, capability),
                sort_keys=True,
                separators=(",", ":"),
            ),
            "--consistent-read",
            "--profile",
            PROFILE,
            "--region",
            REGION,
            "--no-cli-pager",
        ]
    )
    item = response.get("Item") if type(response) is dict else None
    if item is None:
        return None
    if type(item) is not dict:
        raise ValueError("Task13 operation custody readback drifted")
    decoded: dict[str, str] = {}
    for key, value in item.items():
        if (
            type(key) is not str
            or type(value) is not dict
            or set(value) != {"S"}
            or type(value["S"]) is not str
        ):
            raise ValueError("Task13 operation custody item drifted")
        decoded[key] = value["S"]
    return decoded


def _expected_operation_custody(
    authority: Mapping[str, object],
    capability: Mapping[str, object],
) -> dict[str, str]:
    keys = _operation_custody_key(authority, capability)
    return {
        "PK": keys["PK"]["S"],
        "SK": keys["SK"]["S"],
        "RecordType": "glm52_task13_operation_custody_v2",
        "State": "POSSIBLY_SENT",
        "Stage": str(authority["stage"]),
        "OperationKind": str(capability["operation_kind"]),
        "OperationId": str(capability["operation_id"]),
        "RequestIdentitySha256": str(
            capability["request_identity_sha256"]
        ),
        "ExecuteCustodyIdentitySha256": _execute_custody_identity(
            authority,
            capability,
        ),
        "ExecuteCapabilityIdentitySha256": str(
            capability["canonical_identity_sha256"]
        ),
        "ControllerAuthorityIdentitySha256": str(
            authority["canonical_identity_sha256"]
        ),
        "JournalChainHeadSha256": str(
            capability["journal_chain_head_sha256"]
        ),
    }


def _reserve_execute_custody(
    authority: Mapping[str, object],
    capability: Mapping[str, object],
) -> None:
    expected = _expected_operation_custody(authority, capability)
    item = {key: {"S": value} for key, value in expected.items()}
    try:
        _aws_json(
            [
                "aws",
                "dynamodb",
                "put-item",
                "--table-name",
                LEDGER_TABLE,
                "--item",
                json.dumps(item, sort_keys=True, separators=(",", ":")),
                "--condition-expression",
                "attribute_not_exists(PK) AND attribute_not_exists(SK)",
                "--return-consumed-capacity",
                "NONE",
                "--profile",
                PROFILE,
                "--region",
                REGION,
                "--no-cli-pager",
            ]
        )
    except ValueError as error:
        observed = _read_operation_custody(authority, capability)
        if observed is not None and any(
            observed.get(field) != value
            for field, value in expected.items()
            if field != "ExecuteCapabilityIdentitySha256"
        ):
            raise ValueError(
                "foreign Task13 operation custody already exists"
            ) from error
        raise ValueError(
            "Task13 execute custody is already or possibly reserved; "
            "reconcile without replay"
        ) from error


def _commit_operation_custody(
    authority: Mapping[str, object],
    capability: Mapping[str, object],
    result: object,
) -> None:
    result_json = _canonical(_wire_encode(result)).decode("ascii")
    if len(result_json.encode("ascii")) > 300 * 1024:
        raise ValueError("Task13 operation result exceeds custody bound")
    values = {
        ":possible": {"S": "POSSIBLY_SENT"},
        ":committed": {"S": "COMMITTED"},
        ":request": {"S": str(capability["request_identity_sha256"])},
        ":result": {"S": result_json},
        ":result_sha": {"S": hashlib.sha256(result_json.encode()).hexdigest()},
    }
    _aws_json(
        [
            "aws",
            "dynamodb",
            "update-item",
            "--table-name",
            LEDGER_TABLE,
            "--key",
            json.dumps(
                _operation_custody_key(authority, capability),
                sort_keys=True,
                separators=(",", ":"),
            ),
            "--update-expression",
            "SET #state = :committed, ResultJson = :result, "
            "ResultIdentitySha256 = :result_sha",
            "--condition-expression",
            "#state = :possible AND RequestIdentitySha256 = :request",
            "--expression-attribute-names",
            '{"#state":"State"}',
            "--expression-attribute-values",
            json.dumps(values, sort_keys=True, separators=(",", ":")),
            "--return-consumed-capacity",
            "NONE",
            "--profile",
            PROFILE,
            "--region",
            REGION,
            "--no-cli-pager",
        ]
    )


def _committed_custody_result(
    row: Mapping[str, str],
) -> object:
    result_json = row.get("ResultJson")
    identity = row.get("ResultIdentitySha256")
    if (
        type(result_json) is not str
        or type(identity) is not str
        or hashlib.sha256(result_json.encode("ascii")).hexdigest() != identity
    ):
        raise ValueError("committed Task13 custody result drifted")
    try:
        return _wire_decode(json.loads(result_json))
    except json.JSONDecodeError as error:
        raise ValueError("committed Task13 custody result is not JSON") from error


def _validate_observed_operation_custody(
    row: object,
    *,
    authority: Mapping[str, object],
    capability: Mapping[str, object],
) -> dict[str, str]:
    if type(row) is not dict:
        raise ValueError("Task13 operation custody is absent")
    observed = dict(row)
    expected = _expected_operation_custody(authority, capability)
    stable_fields = set(expected) - {
        "State",
        "ExecuteCapabilityIdentitySha256",
        "JournalChainHeadSha256",
    }
    if (
        any(observed.get(field) != expected[field] for field in stable_fields)
        or observed.get("State") not in {"POSSIBLY_SENT", "COMMITTED"}
        or _SHA256.fullmatch(
            str(observed.get("ExecuteCapabilityIdentitySha256", ""))
        )
        is None
        or _SHA256.fullmatch(
            str(observed.get("JournalChainHeadSha256", ""))
        )
        is None
    ):
        raise ValueError("Task13 operation custody scope drifted")
    if observed["State"] == "POSSIBLY_SENT":
        allowed = set(expected)
    else:
        allowed = set(expected) | {
            "ResultJson",
            "ResultIdentitySha256",
        }
    if set(observed) != allowed:
        raise ValueError("Task13 operation custody fields drifted")
    if observed["State"] == "COMMITTED":
        _committed_custody_result(observed)
    return observed


def _execute_operation(kind: str, request: Mapping[str, object]) -> object:
    if kind == "migration-bootstrap":
        return _execute_bootstrap(request)
    if kind in {"collector-invoke", "finalize-invoke"}:
        return _lambda_invoke(request)
    if kind == "qualification-cache-seed":
        return _execute_qualification_cache_seed(request)
    if kind == "h100-qualification":
        return _execute_h100_qualification(request)
    if kind == "sole-sender-authority-materialize":
        return _execute_sole_sender_authority_materialize(request)
    if kind == "launch-authority-materialize":
        return _execute_launch_authority_materialize(request)
    if kind == "guarded-launch":
        return _execute_guarded_launch(request)
    raise ValueError("unreachable execute operation")


def _reconcile_operation(kind: str, request: Mapping[str, object]) -> object:
    if kind == "migration-bootstrap":
        return _reconcile_bootstrap(request)
    if kind == "collector-invoke":
        return _reconcile_collector_invoke(request)
    if kind == "finalize-invoke":
        return _reconcile_finalize_invoke(request)
    if kind == "qualification-cache-seed":
        return _reconcile_qualification_cache_seed(request)
    if kind == "h100-qualification":
        return _reconcile_h100_qualification(request)
    if kind == "sole-sender-authority-materialize":
        return _reconcile_sole_sender_authority_materialize(request)
    if kind == "launch-authority-materialize":
        return _reconcile_launch_authority_materialize(request)
    if kind == "guarded-launch":
        return _reconcile_guarded_launch(request)
    raise ValueError("unreachable reconcile operation")


def _dispatch(envelope: object) -> object:
    if (
        type(envelope) is not dict
        or set(envelope)
        != {
            "schema_version",
            "record_type",
            "action",
            "operation_kind",
            "operation_id",
            "request",
            "coordinator_executable_sha256",
            "controller_execution_authority",
            "operation_capability",
        }
        or envelope["schema_version"] != 2
        or envelope["record_type"]
        != "glm52_task13_coordinator_request_v2"
        or envelope["action"] not in _ACTIONS
        or type(envelope["operation_kind"]) is not str
        or type(envelope["operation_id"]) is not str
        or not envelope["operation_id"]
        or _self_sha256() != envelope["coordinator_executable_sha256"]
    ):
        raise ValueError("coordinator request envelope drifted")
    action = envelope["action"]
    kind = envelope["operation_kind"]
    request = envelope["request"]
    allowed = {
        "inspect": _INSPECT_KINDS,
        "execute": _EXECUTE_KINDS,
        "reconcile": _RECONCILE_KINDS,
    }[action]
    if kind not in allowed or type(request) is not dict:
        raise ValueError(
            "repository coordinator operation is not implemented: "
            + action
            + "/"
            + kind
        )
    authority, capability = _authenticate_v2_envelope(envelope)
    if action == "inspect":
        if kind == "health":
            result = {
                "status": "READY",
                "account_id": ACCOUNT_ID,
                "region": REGION,
                "profile": PROFILE,
            }
        elif kind == "credential-guard":
            result = _current_credential_evidence()
        elif kind == "migration-bootstrap-state":
            result = _bootstrap_state(request)
        elif kind == "staged-infrastructure-adoption":
            result = _inspect_staged_infrastructure_adoption(request)
        elif kind == "negative-iam-probe":
            result = _negative_iam_probe(request)
        elif kind == "p5-zero":
            result = {"active_p5_instance_ids": _active_p5_ids()}
        elif kind == "measurement":
            result = _measurement_readback(request)
        elif kind == "gate":
            result = _gate_readback(request)
        elif kind in {"immutable-artifact", "immutable-marker"}:
            result = _exact_s3_coordinate(request)
        elif kind == "terminal-proof":
            result = _terminal_proof(request)
        elif kind == "monitor":
            result = _monitor(request)
        else:
            raise ValueError("unreachable inspect operation")
    elif action == "execute":
        assert authority is not None
        assert capability is not None
        _current_credential_evidence()
        _reserve_execute_custody(authority, capability)
        result = _execute_operation(kind, request)
        _commit_operation_custody(authority, capability, result)
    else:
        assert authority is not None
        assert capability is not None
        _current_credential_evidence()
        custody = _validate_observed_operation_custody(
            _read_operation_custody(authority, capability),
            authority=authority,
            capability=capability,
        )
        if custody["State"] == "COMMITTED":
            reconciled = _committed_custody_result(custody)
        else:
            reconciled = _reconcile_operation(kind, request)
            _commit_operation_custody(authority, capability, reconciled)
        result = {"state": "COMMITTED", "result": reconciled}
    return {
        "schema_version": 2,
        "record_type": "glm52_task13_coordinator_response_v2",
        "action": action,
        "operation_kind": kind,
        "operation_id": envelope["operation_id"],
        "result": _wire_encode(result),
    }


def main() -> int:
    try:
        envelope = json.loads(sys.stdin.buffer.read())
        response = _dispatch(envelope)
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        print("Task 13 coordinator refused: %s" % error, file=sys.stderr)
        return 64
    sys.stdout.buffer.write(_canonical(response) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
