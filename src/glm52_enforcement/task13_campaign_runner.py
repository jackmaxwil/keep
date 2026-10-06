"""Injected, staged Task 13 campaign coordinator.

The default validation stage is pure.  External operations are available only
through an explicitly supplied coordinator service and controller authority.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Optional

from .canonical import canonical_json_bytes, canonical_sha256
from .cloudformation_stacks import STACK_TAGS
from .decision_closure import (
    INJECTED_FAILURE_KINDS,
    RehearsalMeasurement,
    rehearsal_measurement_from_mapping,
    parse_deployed_gate_document,
    verify_no_post_rehearsals,
)
from .task13_campaign_package import (
    CAMPAIGN_OWNER_APPROVAL_SHA256,
    CampaignPackageError,
    canonical_campaign_package_bytes,
    validate_campaign_artifact_coordinate,
    validate_collect_invoke_result,
    validate_finalize_invoke_result,
    validate_reviewed_repository_execution_custody,
    validate_staged_infrastructure_evidence,
)
from .task13_terminal_evidence import (
    TerminalEvidenceError,
    validate_task13_terminal_proof,
)
from .task13_transport_gates import (
    SEMANTIC_GATE_PINS,
    TransportGateError,
    validate_semantic_gate_coordinate,
)

ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
PROFILE = "keep-gpu"
RUN_ID = "glm52-sky-20260724"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_MUTABLE_VERSION_WORDS = {
    "",
    "null",
    "none",
    "latest",
    "$latest",
    "current",
    "current-version",
}
_ARTIFACT_FIELDS = {
    "artifact_kind",
    "bucket",
    "key",
    "version_id",
    "file_sha256",
    "body_sha256",
}
_STAGES = (
    "validate",
    "deploy-disabled",
    "collect-first-five",
    "collect-remaining",
    "finalize",
    "qualification-cache-seed",
    "h100-qualification",
    "launch",
    "terminal",
    "monitor",
)
_JOURNAL_FIELDS = {
    "schema_version",
    "record_type",
    "operation_id",
    "operation_kind",
    "stage",
    "sequence",
    "state",
    "request_identity_sha256",
    "result_identity_sha256",
    "canonical_identity_sha256",
}
_JOURNAL_STATES = ("PREPARED", "POSSIBLY_SENT", "COMMITTED")
_SHELL_EXECUTABLES = {
    "bash",
    "dash",
    "fish",
    "ksh",
    "sh",
    "zsh",
}
_INSTANCE_ID = re.compile(r"i-[0-9a-f]{17}\Z")
_STACK_IDS = {
    "retained": re.compile(
        r"arn:aws:cloudformation:us-west-2:246813579024:stack/"
        r"keep-glm52-gpu/"
        r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
        r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
    ),
    "fence": re.compile(
        r"arn:aws:cloudformation:us-west-2:246813579024:stack/"
        r"keep-glm52-h1g-fence/"
        r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
        r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
    ),
    "support": re.compile(
        r"arn:aws:cloudformation:us-west-2:246813579024:stack/"
        r"keep-glm52-h1g-support/"
        r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
        r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
    ),
}


class CampaignRunnerError(ValueError):
    """The staged runner input or observed result failed closed."""


@dataclass(frozen=True)
class ControllerExecutionAuthority:
    """Controller-scoped authority for exactly one stage and package."""

    schema_version: int
    record_type: str
    account_id: str
    region: str
    profile: str
    run_id: str
    activation_id: str
    stage: str
    package_identity_sha256: str
    reviewed_artifacts_identity_sha256: str
    controller_nonce_sha256: str
    source_authority_email: str
    owner_approval_sha256: str
    execution_custody_identity_sha256: str
    production_support_plane_approved: bool
    residual_launch_liability_approved: bool
    change_set_execution_approved: bool
    qualification_cache_seed_approved: bool
    h100_qualification_approved: bool
    production_launch_approved: bool
    coordinator_executable_sha256: str
    issued_at_epoch_seconds: int
    expires_at_epoch_seconds: int
    signer_identity: str
    signing_namespace: str
    signer_public_key_fingerprint: str
    canonical_identity_sha256: str
    sshsig_base64: str


@dataclass(frozen=True)
class ControllerOperationCapability:
    """One signed, single-operation coordinator capability."""

    schema_version: int
    record_type: str
    account_id: str
    region: str
    profile: str
    run_id: str
    activation_id: str
    stage: str
    controller_authority_identity_sha256: str
    package_identity_sha256: str
    reviewed_artifacts_identity_sha256: str
    execution_custody_identity_sha256: str
    coordinator_executable_sha256: str
    owner_approval_sha256: str
    action: str
    operation_kind: str
    operation_id: str
    request_identity_sha256: str
    journal_store_path: str
    journal_store_identity_sha256: str
    journal_record_identities: list[str]
    journal_chain_head_sha256: str
    journal_state: str
    prior_execute_capability_identity_sha256: Optional[str]
    capability_nonce_sha256: str
    issued_at_epoch_seconds: int
    expires_at_epoch_seconds: int
    signer_identity: str
    signing_namespace: str
    signer_public_key_fingerprint: str
    canonical_identity_sha256: str
    sshsig_base64: str


@dataclass(frozen=True)
class FinalizationCapture:
    """Validated exact bytes emitted by one version-pinned finalization."""

    status_code: int
    executed_version: str
    payload_bytes: bytes
    gate_bytes: bytes


class FileJournalStore:
    """Append-only, fsynced canonical JSONL journal for stage resumption."""

    def __init__(self, path: object) -> None:
        if not isinstance(path, (str, os.PathLike)):
            raise CampaignRunnerError("journal path must be explicit")
        self._path = Path(path)
        if not self._path.is_absolute():
            raise CampaignRunnerError("journal path must be absolute")
        if self._path != self._path.resolve(strict=False):
            raise CampaignRunnerError(
                "journal path must not contain a symlink or lexical alias"
            )
        if self._path.exists() and (
            self._path.is_symlink() or not self._path.is_file()
        ):
            raise CampaignRunnerError(
                "journal must be a regular non-symlink file"
            )
        if self._path.exists() and stat.S_IMODE(self._path.stat().st_mode) != 0o600:
            raise CampaignRunnerError("existing journal mode must be exactly 0600")
        parent = self._path.parent
        if not parent.is_dir() or parent.is_symlink():
            raise CampaignRunnerError(
                "journal parent must be an existing real directory"
            )
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def identity(self) -> tuple[str, Optional[tuple[int, int]]]:
        resolved = self._path.resolve(strict=False)
        if resolved != self._path:
            raise CampaignRunnerError("journal path symlink alias drifted")
        if not self._path.exists():
            return str(resolved), None
        info = self._path.lstat()
        if self._path.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise CampaignRunnerError(
                "journal must be a regular non-symlink file"
            )
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise CampaignRunnerError(
                "existing journal mode must be exactly 0600"
            )
        return str(resolved), (info.st_dev, info.st_ino)

    def authority_binding(
        self,
        operation_id: str,
    ) -> dict[str, object]:
        """Project the exact on-disk journal custody signed by v2."""

        path, inode = self.identity()
        if inode is None:
            raise CampaignRunnerError(
                "operation journal does not yet have durable file custody"
            )
        rows = self.entries(operation_id)
        if not rows:
            raise CampaignRunnerError(
                "operation journal has no signed records"
            )
        record_identities = [
            _sha(
                row.get("canonical_identity_sha256"),
                "journal record identity",
            )
            for row in rows
        ]
        identity_body = {
            "path": path,
            "device": inode[0],
            "inode": inode[1],
        }
        return {
            "journal_store_path": path,
            "journal_store_identity_sha256": canonical_sha256(
                identity_body
            ),
            "journal_record_identities": record_identities,
            "journal_state": rows[-1]["state"],
        }

    def _read_rows(self) -> list[dict[str, object]]:
        if not self._path.exists():
            return []
        if self._path.is_symlink() or not self._path.is_file():
            raise CampaignRunnerError("journal path identity drifted")
        raw = self._path.read_bytes()
        if len(raw) > 16 * 1024 * 1024:
            raise CampaignRunnerError("runner journal is oversized")
        if raw and not raw.endswith(b"\n"):
            raise CampaignRunnerError("runner journal has a torn record")
        rows = []
        for line in raw.splitlines(keepends=True):
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CampaignRunnerError(
                    "runner journal contains invalid JSON"
                ) from error
            if (
                type(value) is not dict
                or line != canonical_json_bytes(value) + b"\n"
            ):
                raise CampaignRunnerError(
                    "runner journal record is not canonical"
                )
            rows.append(value)
        return rows

    def entries(
        self,
        operation_id: str,
    ) -> tuple[dict[str, object], ...]:
        with self._lock:
            return tuple(
                row
                for row in self._read_rows()
                if row.get("operation_id") == operation_id
            )

    def append(self, row: dict[str, object]) -> None:
        if type(row) is not dict:
            raise CampaignRunnerError("journal append is not an object")
        raw = canonical_json_bytes(row) + b"\n"
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        with self._lock:
            first_create = not self._path.exists()
            self._read_rows()
            descriptor = os.open(self._path, flags, 0o600)
            try:
                mode = os.fstat(descriptor).st_mode
                if not stat.S_ISREG(mode):
                    raise CampaignRunnerError(
                        "journal descriptor is not a regular file"
                    )
                os.fchmod(descriptor, 0o600)
                written = os.write(descriptor, raw)
                if written != len(raw):
                    raise CampaignRunnerError(
                        "runner journal write was incomplete"
                    )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if first_create:
                directory_flags = os.O_RDONLY
                if hasattr(os, "O_DIRECTORY"):
                    directory_flags |= os.O_DIRECTORY
                directory_descriptor = os.open(
                    self._path.parent,
                    directory_flags,
                )
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)


@dataclass(frozen=True)
class CampaignJournalStores:
    """Three non-aliasing durability domains for stack and campaign effects."""

    fence: object
    support: object
    campaign: object

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        stores = (self.fence, self.support, self.campaign)
        if len({id(store) for store in stores}) != 3:
            raise CampaignRunnerError(
                "fence, support, and campaign journals must be distinct"
            )
        file_identities = []
        for store in stores:
            if not all(
                callable(getattr(store, method, None))
                for method in ("entries", "append")
            ):
                raise CampaignRunnerError(
                    "journal store boundary is incomplete"
                )
            identity = getattr(store, "identity", None)
            if callable(identity):
                file_identities.append(identity())
        paths = [value[0] for value in file_identities]
        inodes = [
            value[1]
            for value in file_identities
            if value[1] is not None
        ]
        if len(paths) != len(set(paths)):
            raise CampaignRunnerError(
                "journal paths must be distinct"
            )
        if len(inodes) != len(set(inodes)):
            raise CampaignRunnerError(
                "journal files must not alias the same inode"
            )

    def for_scope(self, scope: str) -> object:
        self._validate()
        if scope not in {"fence", "support", "campaign"}:
            raise CampaignRunnerError("journal scope is not closed")
        return getattr(self, scope)


def _journal_context(value: object) -> object:
    route = getattr(value, "for_scope", None)
    if not callable(route):
        raise CampaignRunnerError(
            "runner requires distinct fence, support, and campaign journals"
        )
    stores = [route(scope) for scope in ("fence", "support", "campaign")]
    if len({id(store) for store in stores}) != 3:
        raise CampaignRunnerError(
            "runner journals must be distinct stores"
        )
    return value


def _journal_for_scope(context: object, scope: str) -> object:
    context = _journal_context(context)
    store = context.for_scope(scope)
    if not all(
        callable(getattr(store, method, None))
        for method in ("entries", "append")
    ):
        raise CampaignRunnerError("journal store boundary is incomplete")
    return store


def _wire_encode(value: object) -> object:
    if type(value) is bytes:
        return {
            "__glm52_wire_type__": "bytes",
            "base64": base64.b64encode(value).decode("ascii"),
        }
    if type(value) is bytearray:
        return _wire_encode(bytes(value))
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise CampaignRunnerError("coordinator request key is not text")
        return {key: _wire_encode(item) for key, item in value.items()}
    if type(value) in {list, tuple}:
        return [_wire_encode(item) for item in value]
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise CampaignRunnerError("coordinator request is not closed JSON data")


def _wire_decode(value: object) -> object:
    if type(value) is list:
        return [_wire_decode(item) for item in value]
    if type(value) is dict:
        if set(value) == {"__glm52_wire_type__", "base64"}:
            if value["__glm52_wire_type__"] != "bytes":
                raise CampaignRunnerError(
                    "coordinator wire type is unsupported"
                )
            encoded = value["base64"]
            if type(encoded) is not str:
                raise CampaignRunnerError(
                    "coordinator bytes payload is malformed"
                )
            try:
                return base64.b64decode(encoded, validate=True)
            except ValueError as error:
                raise CampaignRunnerError(
                    "coordinator bytes payload is malformed"
                ) from error
        return {key: _wire_decode(item) for key, item in value.items()}
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise CampaignRunnerError("coordinator response is not closed JSON data")


class SubprocessCoordinator:
    """SHA-pinned finite controller using canonical stdin/stdout envelopes."""

    def __init__(
        self,
        executable: object,
        expected_sha256: object,
        *,
        authority_signer: object = None,
    ) -> None:
        if not isinstance(executable, (str, os.PathLike)):
            raise CampaignRunnerError(
                "coordinator executable path must be explicit"
            )
        self._executable = Path(executable)
        if (
            not self._executable.is_absolute()
            or not self._executable.is_file()
            or self._executable.is_symlink()
            or not os.access(self._executable, os.X_OK)
        ):
            raise CampaignRunnerError(
                "coordinator executable must be absolute, regular, and executable"
            )
        expected = _sha(expected_sha256, "coordinator executable")
        observed = hashlib.sha256(self._executable.read_bytes()).hexdigest()
        if observed != expected:
            raise CampaignRunnerError(
                "coordinator executable identity drifted"
            )
        self._sha256 = expected
        self._authority_signer = authority_signer
        self._controller_authority: Optional[
            ControllerExecutionAuthority
        ] = None
        self._operation_journal_bindings: dict[
            tuple[str, str, str],
            dict[str, object],
        ] = {}

    @property
    def coordinator_executable_sha256(self) -> str:
        return self._sha256

    def bind_controller_authority(
        self,
        authority: ControllerExecutionAuthority,
    ) -> None:
        """Bind one already-verified stage grant to subsequent calls."""

        if type(authority) is not ControllerExecutionAuthority:
            raise CampaignRunnerError(
                "controller authority binding is not exact"
            )
        self._controller_authority = authority

    def bind_operation_journal(
        self,
        *,
        action: str,
        operation_kind: str,
        operation_id: str,
        journal: object,
    ) -> None:
        """Bind execute/reconcile to the exact durable operation journal."""

        binding = getattr(journal, "authority_binding", None)
        if not callable(binding):
            raise CampaignRunnerError(
                "production coordinator requires file-journal custody"
            )
        value = binding(operation_id)
        if type(value) is not dict:
            raise CampaignRunnerError(
                "operation journal authority binding drifted"
            )
        self._operation_journal_bindings[
            (action, operation_kind, operation_id)
        ] = value

    def _operation_capability(
        self,
        *,
        action: str,
        operation_kind: str,
        operation_id: str,
        request: object,
    ) -> ControllerOperationCapability:
        authority = self._controller_authority
        if authority is None or self._authority_signer is None:
            raise CampaignRunnerError(
                "signed v2 operation authority is not bound"
            )
        request_identity = canonical_sha256(_wire_encode(request))
        key = (action, operation_kind, operation_id)
        journal_binding = self._operation_journal_bindings.pop(key, None)
        if action in {"execute", "reconcile"}:
            if journal_binding is None:
                raise CampaignRunnerError(
                    "mutation operation journal authority is absent"
                )
        else:
            journal_binding = {
                "journal_store_path": "READ_ONLY",
                "journal_store_identity_sha256": canonical_sha256(
                    {
                        "authority": authority.canonical_identity_sha256,
                        "stage": authority.stage,
                        "state": "READ_ONLY",
                    }
                ),
                "journal_record_identities": [
                    authority.canonical_identity_sha256
                ],
                "journal_state": "READ_ONLY",
            }
        assert type(journal_binding) is dict
        prior_execute = (
            canonical_sha256(
                {
                    "stage": authority.stage,
                    "operation_kind": operation_kind,
                    "operation_id": operation_id,
                    "request_identity_sha256": request_identity,
                }
            )
            if action == "reconcile"
            else None
        )
        from .task13_controller_authority import (
            ControllerAuthorityIssuerError,
            issue_controller_operation_capability,
        )

        try:
            return issue_controller_operation_capability(
                authority=authority,
                action=action,
                operation_kind=operation_kind,
                operation_id=operation_id,
                request_identity_sha256=request_identity,
                journal_store_path=journal_binding[
                    "journal_store_path"
                ],
                journal_store_identity_sha256=journal_binding[
                    "journal_store_identity_sha256"
                ],
                journal_record_identities=journal_binding[
                    "journal_record_identities"
                ],
                journal_state=journal_binding["journal_state"],
                prior_execute_capability_identity_sha256=prior_execute,
                ttl_seconds=60,
                signer=self._authority_signer,
            )
        except ControllerAuthorityIssuerError as error:
            raise CampaignRunnerError(
                "v2 operation capability issuance failed"
            ) from error

    def _verified_executable_bytes(self) -> bytes:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self._executable, flags)
        except OSError as error:
            raise CampaignRunnerError(
                "coordinator executable identity drifted"
            ) from error
        try:
            mode = os.fstat(descriptor).st_mode
            if (
                not stat.S_ISREG(mode)
                or stat.S_IMODE(mode) & 0o111 == 0
            ):
                raise CampaignRunnerError(
                    "coordinator executable identity drifted"
                )
            chunks = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            raw = b"".join(chunks)
            if hashlib.sha256(raw).hexdigest() != self._sha256:
                raise CampaignRunnerError(
                    "coordinator executable identity drifted"
                )
            return raw
        finally:
            os.close(descriptor)

    def _call(
        self,
        action: str,
        operation_kind: str,
        operation_id: Optional[str],
        request: object,
    ) -> object:
        exact_operation_id = (
            "inspect:"
            + operation_kind
            + ":"
            + canonical_sha256(_wire_encode(request))[:24]
            if action == "inspect" and operation_id is None
            else operation_id
        )
        capability = (
            None
            if action == "inspect"
            and operation_kind == "health"
            and self._controller_authority is None
            else self._operation_capability(
                action=action,
                operation_kind=operation_kind,
                operation_id=str(exact_operation_id),
                request=request,
            )
        )
        envelope = {
            "schema_version": 2,
            "record_type": "glm52_task13_coordinator_request_v2",
            "action": action,
            "operation_kind": operation_kind,
            "operation_id": exact_operation_id,
            "request": _wire_encode(request),
            "coordinator_executable_sha256": self._sha256,
            "controller_execution_authority": (
                None
                if self._controller_authority is None
                else asdict(self._controller_authority)
            ),
            "operation_capability": (
                None if capability is None else asdict(capability)
            ),
        }
        executable_bytes = self._verified_executable_bytes()
        try:
            with tempfile.TemporaryDirectory(
                prefix="glm52-task13-coordinator-"
            ) as directory:
                immutable_path = Path(directory) / "coordinator"
                descriptor = os.open(
                    immutable_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o700,
                )
                try:
                    offset = 0
                    while offset < len(executable_bytes):
                        offset += os.write(
                            descriptor,
                            executable_bytes[offset:],
                        )
                    os.fchmod(descriptor, 0o700)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                completed = subprocess.run(
                    [str(immutable_path)],
                    input=canonical_json_bytes(envelope) + b"\n",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=False,
                    env={
                        "HOME": str(Path.home()),
                        "PATH": (
                            "/opt/homebrew/bin:/usr/local/bin:"
                            "/usr/bin:/bin"
                        ),
                        "GLM52_TASK13_REPO_ROOT": str(
                            self._executable.parents[3]
                        ),
                    },
                    check=False,
                    timeout=(
                        24 * 60 * 60 + 300
                        if action == "execute"
                        and operation_kind
                        in {
                            "qualification-cache-seed",
                            "h100-qualification",
                        }
                        else (
                            30 * 60
                            if action == "execute"
                            and operation_kind == "migration-bootstrap"
                            else (
                                10 * 60
                                if action == "reconcile"
                                or operation_kind == "monitor"
                                else 120
                            )
                        )
                    ),
                )
        except (OSError, subprocess.SubprocessError) as error:
            raise CampaignRunnerError(
                "coordinator process did not complete"
            ) from error
        if (
            completed.returncode != 0
            or not completed.stdout
            or len(completed.stdout) > 8 * 1024 * 1024
        ):
            raise CampaignRunnerError(
                "coordinator process refused the finite operation"
            )
        try:
            response = json.loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CampaignRunnerError(
                "coordinator response is not JSON"
            ) from error
        if (
            type(response) is not dict
            or set(response)
            != {
                "schema_version",
                "record_type",
                "action",
                "operation_kind",
                "operation_id",
                "result",
            }
            or response["schema_version"] != 2
            or response["record_type"]
            != "glm52_task13_coordinator_response_v2"
            or response["action"] != action
            or response["operation_kind"] != operation_kind
            or response["operation_id"] != exact_operation_id
            or completed.stdout != canonical_json_bytes(response) + b"\n"
        ):
            raise CampaignRunnerError(
                "coordinator response envelope drifted"
            )
        return _wire_decode(response["result"])

    def execute(
        self,
        operation_kind: str,
        operation_id: str,
        request: object,
    ) -> object:
        return self._call(
            "execute",
            operation_kind,
            operation_id,
            request,
        )

    def reconcile(
        self,
        operation_kind: str,
        operation_id: str,
        request: object,
    ) -> object:
        return self._call(
            "reconcile",
            operation_kind,
            operation_id,
            request,
        )

    def inspect(self, operation_kind: str, request: object) -> object:
        return self._call("inspect", operation_kind, None, request)


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise CampaignRunnerError(label + " must be lowercase SHA-256")
    return value


def _validate_reviewed_artifacts(
    package: Mapping[str, object],
    reviewed_artifacts: object,
) -> tuple[list[dict[str, object]], str]:
    expected = package.get("reviewed_artifacts")
    if type(expected) is not list or type(reviewed_artifacts) is not list:
        raise CampaignRunnerError("reviewed artifacts must be exact lists")
    if reviewed_artifacts != expected:
        raise CampaignRunnerError(
            "reviewed artifacts do not match the campaign package"
        )
    seen = set()
    rows = []
    for value in reviewed_artifacts:
        try:
            coordinate = validate_campaign_artifact_coordinate(value)
        except CampaignPackageError as error:
            raise CampaignRunnerError(
                "reviewed artifact coordinate is not exact"
            ) from error
        kind = coordinate["artifact_kind"]
        if kind in seen:
            raise CampaignRunnerError(
                "reviewed artifact identity is mutable or duplicated"
            )
        seen.add(kind)
        rows.append(coordinate)
    return rows, canonical_sha256(rows)


def _validate_package(
    package: object,
    reviewed_artifacts: object,
) -> tuple[dict[str, object], str]:
    if type(package) is not dict:
        raise CampaignRunnerError("campaign package must be an exact object")
    try:
        canonical_campaign_package_bytes(package)
    except (TypeError, ValueError) as error:
        raise CampaignRunnerError("campaign package identity drifted") from error
    for field, expected in (
        ("account_id", ACCOUNT_ID),
        ("region", REGION),
        ("profile", PROFILE),
        ("run_id", RUN_ID),
    ):
        if package.get(field) != expected:
            raise CampaignRunnerError(
                "campaign package " + field + " is not exact"
            )
    if (
        package.get("schema_version") != 1
        or package.get("record_type")
        != "glm52_task13_campaign_package_v1"
    ):
        raise CampaignRunnerError("campaign package schema drifted")
    if package.get("semantic_transport_gates") != SEMANTIC_GATE_PINS:
        raise CampaignRunnerError("campaign semantic transport gates drifted")
    _validate_reviewed_artifacts(package, reviewed_artifacts)
    return package, str(package["canonical_identity_sha256"])


def _validate_authority(
    value: object,
    *,
    stage: str,
    package_identity_sha256: str,
    reviewed_artifacts_identity_sha256: str,
    execution_custody_identity_sha256: str,
    activation_id: str,
    services: object,
) -> ControllerExecutionAuthority:
    if type(value) is not ControllerExecutionAuthority:
        raise CampaignRunnerError(
            "exact controller authority is required for external stages"
        )
    try:
        from .task13_controller_authority import (
            ControllerAuthorityIssuerError,
            verify_controller_execution_authority,
        )

        verify_controller_execution_authority(value)
    except ControllerAuthorityIssuerError as error:
        raise CampaignRunnerError(
            "controller authority signature or TTL drifted"
        ) from error
    now = int(time.time())
    coordinator_identity = getattr(
        services,
        "coordinator_executable_sha256",
        None,
    )
    if (
        value.schema_version != 2
        or value.record_type
        != "glm52_task13_controller_execution_authority_v2"
        or value.account_id != ACCOUNT_ID
        or value.region != REGION
        or value.profile != PROFILE
        or value.run_id != RUN_ID
        or value.activation_id != activation_id
        or value.stage != stage
        or value.package_identity_sha256 != package_identity_sha256
        or value.reviewed_artifacts_identity_sha256
        != reviewed_artifacts_identity_sha256
        or value.source_authority_email != "operator@example.com"
        or value.owner_approval_sha256
        != CAMPAIGN_OWNER_APPROVAL_SHA256
        or value.execution_custody_identity_sha256
        != execution_custody_identity_sha256
        or value.production_support_plane_approved is not True
        or value.residual_launch_liability_approved is not True
        or type(value.issued_at_epoch_seconds) is not int
        or type(value.expires_at_epoch_seconds) is not int
        or value.issued_at_epoch_seconds > now
        or value.expires_at_epoch_seconds <= now
        or value.expires_at_epoch_seconds - value.issued_at_epoch_seconds
        not in range(60, 901)
        or value.coordinator_executable_sha256 != coordinator_identity
    ):
        raise CampaignRunnerError("controller authority scope drifted")
    stage_grants = {
        "deploy-disabled": True,
        "collect-first-five": value.change_set_execution_approved,
        "collect-remaining": value.change_set_execution_approved,
        "finalize": value.change_set_execution_approved,
        "qualification-cache-seed": (
            value.qualification_cache_seed_approved
        ),
        "h100-qualification": value.h100_qualification_approved,
        "launch": value.production_launch_approved,
        "terminal": value.production_launch_approved,
        "monitor": value.production_launch_approved,
    }
    if stage_grants.get(stage) is not True:
        raise CampaignRunnerError(
            "controller authority does not approve this execution stage"
        )
    expected_grants = {
        "change_set_execution_approved": stage
        in {
            "collect-first-five",
            "collect-remaining",
            "finalize",
        },
        "qualification_cache_seed_approved": (
            stage == "qualification-cache-seed"
        ),
        "h100_qualification_approved": stage == "h100-qualification",
        "production_launch_approved": stage
        in {"launch", "terminal", "monitor"},
    }
    if any(
        getattr(value, field) is not expected
        for field, expected in expected_grants.items()
    ):
        raise CampaignRunnerError(
            "controller authority grants are not exact for this stage"
        )
    _sha(value.controller_nonce_sha256, "controller nonce identity")
    return value


def _json_projection(value: object) -> object:
    if type(value) is bytes:
        return {
            "bytes_length": len(value),
            "bytes_sha256": hashlib.sha256(value).hexdigest(),
        }
    if type(value) is bytearray:
        raw = bytes(value)
        return {
            "bytes_length": len(raw),
            "bytes_sha256": hashlib.sha256(raw).hexdigest(),
        }
    if type(value) is dict:
        return {
            str(key): _json_projection(item)
            for key, item in value.items()
            if type(key) is str
        }
    if type(value) in {list, tuple}:
        return [_json_projection(item) for item in value]
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise CampaignRunnerError("external result is not closed JSON data")


def _result_identity(value: object) -> str:
    return canonical_sha256(_json_projection(value))


def _journal_body(row: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in row.items()
        if key != "canonical_identity_sha256"
    }


def _journal_entries(
    journal: object,
    operation_id: str,
) -> tuple[dict[str, object], ...]:
    read = getattr(journal, "entries", None)
    if not callable(read):
        raise CampaignRunnerError("runner journal read boundary is missing")
    rows = read(operation_id)
    if type(rows) is not tuple:
        raise CampaignRunnerError("runner journal entries are not a tuple")
    validated = []
    for index, row in enumerate(rows, 1):
        if type(row) is not dict or set(row) != _JOURNAL_FIELDS:
            raise CampaignRunnerError("runner journal entry schema drifted")
        if (
            row["schema_version"] != 1
            or row["record_type"] != "glm52_task13_runner_journal_v1"
            or row["operation_id"] != operation_id
            or row["sequence"] != index
            or row["state"] != _JOURNAL_STATES[index - 1]
            or row["canonical_identity_sha256"]
            != canonical_sha256(_journal_body(row))
        ):
            raise CampaignRunnerError("runner journal chain drifted")
        _sha(row["request_identity_sha256"], "journal request identity")
        if row["state"] == "COMMITTED":
            _sha(row["result_identity_sha256"], "journal result identity")
        elif row["result_identity_sha256"] is not None:
            raise CampaignRunnerError(
                "noncommitted journal entry invented a result"
            )
        validated.append(row)
    if len(validated) > 3:
        raise CampaignRunnerError("runner journal operation has extra states")
    return tuple(validated)


def _append_journal(
    journal: object,
    *,
    operation_id: str,
    operation_kind: str,
    stage: str,
    state: str,
    request_identity_sha256: str,
    result_identity_sha256: Optional[str],
) -> None:
    rows = _journal_entries(journal, operation_id)
    expected_state = _JOURNAL_STATES[len(rows)]
    if state != expected_state:
        raise CampaignRunnerError("runner journal transition is invalid")
    body = {
        "schema_version": 1,
        "record_type": "glm52_task13_runner_journal_v1",
        "operation_id": operation_id,
        "operation_kind": operation_kind,
        "stage": stage,
        "sequence": len(rows) + 1,
        "state": state,
        "request_identity_sha256": request_identity_sha256,
        "result_identity_sha256": result_identity_sha256,
    }
    row = {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }
    append = getattr(journal, "append", None)
    if not callable(append):
        raise CampaignRunnerError("runner journal append boundary is missing")
    append(row)
    _journal_entries(journal, operation_id)


def _reconcile(
    services: object,
    journal: object,
    *,
    operation_kind: str,
    operation_id: str,
    request: object,
) -> object:
    reconcile = getattr(services, "reconcile", None)
    if not callable(reconcile):
        raise CampaignRunnerError("coordinator reconcile boundary is missing")
    bind = getattr(services, "bind_operation_journal", None)
    if callable(bind):
        bind(
            action="reconcile",
            operation_kind=operation_kind,
            operation_id=operation_id,
            journal=journal,
        )
    try:
        value = reconcile(operation_kind, operation_id, request)
    except Exception as error:
        raise CampaignRunnerError(
            "mutation reconciliation remained unresolved"
        ) from error
    if (
        type(value) is not dict
        or set(value) != {"state", "result"}
        or value["state"] != "COMMITTED"
    ):
        raise CampaignRunnerError(
            "POSSIBLY_SENT mutation remains unresolved; replay forbidden"
        )
    return value["result"]


def _validate_credential_evidence(value: object) -> dict[str, object]:
    trusted_now = int(time.time())
    if (
        type(value) is not dict
        or set(value)
        != {
            "account_id",
            "observed_epoch_seconds",
            "expiration_epoch_seconds",
            "seconds_remaining",
        }
        or value["account_id"] != ACCOUNT_ID
        or type(value["observed_epoch_seconds"]) is not int
        or type(value["expiration_epoch_seconds"]) is not int
        or type(value["seconds_remaining"]) is not int
        or abs(value["observed_epoch_seconds"] - trusted_now) > 5
        or value["seconds_remaining"]
        != (
            value["expiration_epoch_seconds"]
            - value["observed_epoch_seconds"]
        )
        or value["seconds_remaining"] < 3600
    ):
        raise CampaignRunnerError(
            "credential Expiration is stale or insufficient"
        )
    return dict(value)


def _guard_current_credentials(services: object, *, stage: str) -> None:
    value = _inspect(
        services,
        "credential-guard",
        {
            "account_id": ACCOUNT_ID,
            "region": REGION,
            "profile": PROFILE,
            "stage": stage,
            "trusted_epoch_seconds": int(time.time()),
        },
    )
    _validate_credential_evidence(value)


def _mutation(
    services: object,
    journal: object,
    *,
    stage: str,
    operation_kind: str,
    operation_id: str,
    request: object,
    validate_result: Callable[[object], object],
) -> object:
    request_identity = canonical_sha256(_json_projection(request))
    rows = _journal_entries(journal, operation_id)
    if rows and (
        rows[0]["stage"] != stage
        or rows[0]["operation_kind"] != operation_kind
        or rows[0]["request_identity_sha256"] != request_identity
    ):
        raise CampaignRunnerError("runner journal request identity drifted")
    if rows and rows[-1]["state"] in {"POSSIBLY_SENT", "COMMITTED"}:
        result = _reconcile(
            services,
            journal,
            operation_kind=operation_kind,
            operation_id=operation_id,
            request=request,
        )
        try:
            validated = validate_result(result)
        except Exception as error:
            raise CampaignRunnerError(str(error)) from error
        result_identity = _result_identity(result)
        if rows[-1]["state"] == "COMMITTED":
            if rows[-1]["result_identity_sha256"] != result_identity:
                raise CampaignRunnerError(
                    "committed mutation readback identity drifted"
                )
            return validated
        _append_journal(
            journal,
            operation_id=operation_id,
            operation_kind=operation_kind,
            stage=stage,
            state="COMMITTED",
            request_identity_sha256=request_identity,
            result_identity_sha256=result_identity,
        )
        return validated
    if not rows:
        _append_journal(
            journal,
            operation_id=operation_id,
            operation_kind=operation_kind,
            stage=stage,
            state="PREPARED",
            request_identity_sha256=request_identity,
            result_identity_sha256=None,
        )
    _guard_current_credentials(services, stage=stage)
    _append_journal(
        journal,
        operation_id=operation_id,
        operation_kind=operation_kind,
        stage=stage,
        state="POSSIBLY_SENT",
        request_identity_sha256=request_identity,
        result_identity_sha256=None,
    )
    execute = getattr(services, "execute", None)
    if not callable(execute):
        raise CampaignRunnerError("coordinator execute boundary is missing")
    bind = getattr(services, "bind_operation_journal", None)
    if callable(bind):
        bind(
            action="execute",
            operation_kind=operation_kind,
            operation_id=operation_id,
            journal=journal,
        )
    try:
        result = execute(operation_kind, operation_id, request)
    except Exception as error:
        raise CampaignRunnerError(
            "mutation is POSSIBLY_SENT; reconcile without replay"
        ) from error
    try:
        validated = validate_result(result)
    except Exception as error:
        raise CampaignRunnerError(str(error)) from error
    _append_journal(
        journal,
        operation_id=operation_id,
        operation_kind=operation_kind,
        stage=stage,
        state="COMMITTED",
        request_identity_sha256=request_identity,
        result_identity_sha256=_result_identity(result),
    )
    return validated


def _read_committed_mutation(
    services: object,
    journal: object,
    *,
    stage: str,
    operation_kind: str,
    operation_id: str,
    request: object,
    validate_result: Callable[[object], object],
) -> object:
    rows = _journal_entries(journal, operation_id)
    if not rows or rows[-1]["state"] != "COMMITTED":
        raise CampaignRunnerError(
            "required mutation is not durably committed"
        )
    return _mutation(
        services,
        journal,
        stage=stage,
        operation_kind=operation_kind,
        operation_id=operation_id,
        request=request,
        validate_result=validate_result,
    )


def _stage_operation_id(stage: str) -> str:
    return "stage:" + stage


def _stage_request_identity(
    *,
    stage: str,
    package_identity_sha256: str,
    reviewed_artifacts_identity_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "stage": stage,
            "package_identity_sha256": package_identity_sha256,
            "reviewed_artifacts_identity_sha256": (
                reviewed_artifacts_identity_sha256
            ),
        }
    )


def _record_stage(
    journal: object,
    *,
    stage: str,
    package_identity_sha256: str,
    reviewed_artifacts_identity_sha256: str,
    summary: object,
) -> None:
    operation_id = _stage_operation_id(stage)
    request_identity = _stage_request_identity(
        stage=stage,
        package_identity_sha256=package_identity_sha256,
        reviewed_artifacts_identity_sha256=(
            reviewed_artifacts_identity_sha256
        ),
    )
    rows = _journal_entries(journal, operation_id)
    result_identity = _result_identity(summary)
    if rows:
        if (
            rows[0]["request_identity_sha256"] != request_identity
            or rows[0]["stage"] != stage
            or rows[0]["operation_kind"] != "stage-completion"
        ):
            raise CampaignRunnerError("stage completion identity drifted")
        if rows[-1]["state"] == "COMMITTED":
            if rows[-1]["result_identity_sha256"] != result_identity:
                raise CampaignRunnerError(
                    "stage completion identity drifted"
                )
            return
    for state in _JOURNAL_STATES[len(rows):]:
        _append_journal(
            journal,
            operation_id=operation_id,
            operation_kind="stage-completion",
            stage=stage,
            state=state,
            request_identity_sha256=request_identity,
            result_identity_sha256=(
                result_identity if state == "COMMITTED" else None
            ),
        )


def _require_stage(
    journal: object,
    stage: str,
    *,
    package: Mapping[str, object],
) -> None:
    rows = _journal_entries(journal, _stage_operation_id(stage))
    if not rows or rows[-1]["state"] != "COMMITTED":
        raise CampaignRunnerError(
            "required prior stage is not durably committed: " + stage
        )
    current_package_identity = _sha(
        package.get("canonical_identity_sha256"),
        "campaign package identity",
    )
    reviewed_artifacts = package.get("reviewed_artifacts")
    if type(reviewed_artifacts) is not list:
        raise CampaignRunnerError("campaign reviewed artifacts are missing")
    allowed = {
        _stage_request_identity(
            stage=stage,
            package_identity_sha256=current_package_identity,
            reviewed_artifacts_identity_sha256=canonical_sha256(
                reviewed_artifacts
            ),
        )
    }
    predecessor = package.get("predecessor_identity")
    if (
        package.get("package_phase") == "PRODUCTION"
        and stage
        in {
            "deploy-disabled",
            "collect-first-five",
            "collect-remaining",
            "finalize",
            "qualification-cache-seed",
            "h100-qualification",
        }
    ):
        if type(predecessor) is not dict:
            raise CampaignRunnerError(
                "production predecessor identity is missing"
            )
        allowed.add(
            _stage_request_identity(
                stage=stage,
                package_identity_sha256=_sha(
                    predecessor.get("package_identity_sha256"),
                    "predecessor package identity",
                ),
                reviewed_artifacts_identity_sha256=_sha(
                    predecessor.get(
                        "reviewed_artifacts_identity_sha256"
                    ),
                    "predecessor reviewed artifacts identity",
                ),
            )
        )
    if rows[0]["request_identity_sha256"] not in allowed:
        raise CampaignRunnerError(
            "required prior stage belongs to a foreign package identity: "
            + stage
        )


def _inspect(
    services: object,
    operation_kind: str,
    request: object,
) -> object:
    inspect = getattr(services, "inspect", None)
    if not callable(inspect):
        raise CampaignRunnerError("coordinator inspect boundary is missing")
    try:
        return inspect(operation_kind, request)
    except Exception as error:
        raise CampaignRunnerError("coordinator inspection failed") from error


def _validate_command(
    command: object,
    *,
    package: Mapping[str, object],
) -> dict[str, object]:
    required = {
        "command_id",
        "operation",
        "account_id",
        "profile",
        "region",
        "mutates_aws",
        "execution_state",
        "argv",
    }
    if (
        type(command) is not dict
        or not required.issubset(command)
        or not set(command).issubset(required | {"expected_stdout"})
    ):
        raise CampaignRunnerError("campaign command schema drifted")
    operation = command["operation"]
    allowlist = package.get("command_operation_allowlist")
    if (
        type(allowlist) is not list
        or operation not in allowlist
        or operation
        in {
            "s3:PutObject",
            "ec2:RunInstances",
            "ec2:RequestSpotInstances",
        }
        or command["account_id"] != ACCOUNT_ID
        or command["profile"] != PROFILE
        or command["region"] != REGION
        or type(command["mutates_aws"]) is not bool
    ):
        raise CampaignRunnerError("campaign command authority drifted")
    argv = command["argv"]
    if (
        type(argv) is not list
        or not argv
        or any(
            type(item) is not str
            or not item
            or "\x00" in item
            or "\n" in item
            or "\r" in item
            for item in argv
        )
        or argv[0].rsplit("/", 1)[-1] in _SHELL_EXECUTABLES
        or "-c" in argv[:3]
    ):
        raise CampaignRunnerError("campaign command is not safe argv")
    rendered = " ".join(argv).lower()
    if (
        "put-object" in rendered
        or "run-instances" in rendered
        or "request-spot" in rendered
        or "capacity-block" in rendered
    ):
        raise CampaignRunnerError("campaign command contains a forbidden route")
    return dict(command)


def _validate_command_result(
    value: object,
    *,
    command: Mapping[str, object],
    expected_status: str,
) -> object:
    if (
        type(value) is not dict
        or set(value)
        != {"status", "operation", "request_identity_sha256"}
        or value["status"] != expected_status
        or value["operation"] != command["operation"]
        or value["request_identity_sha256"]
        != canonical_sha256(command)
    ):
        raise CampaignRunnerError("campaign command result drifted")
    return value


def _validate_command_inspection(
    value: object,
    *,
    command: Mapping[str, object],
    exact_evidence: Optional[Mapping[str, object]] = None,
) -> object:
    if (
        type(value) is not dict
        or set(value)
        != {
            "status",
            "operation",
            "request_identity_sha256",
            "evidence",
        }
        or value["status"] != "INSPECTED"
        or value["operation"] != command["operation"]
        or value["request_identity_sha256"]
        != canonical_sha256(command)
        or type(value["evidence"]) is not dict
    ):
        raise CampaignRunnerError(
            "campaign command inspection lacks semantic evidence"
        )
    evidence = value["evidence"]
    command_id = command["command_id"]
    if exact_evidence is not None:
        if evidence != dict(exact_evidence):
            raise CampaignRunnerError(
                "CloudFormation inspection evidence is opaque or drifted"
            )
    elif str(command["operation"]).startswith("cloudformation:"):
        raise CampaignRunnerError(
            "CloudFormation inspection evidence is opaque or drifted"
        )
    elif command_id in {"read-caller-identity", "assert-caller-account"}:
        if evidence != {"account_id": ACCOUNT_ID}:
            raise CampaignRunnerError("caller account evidence drifted")
    elif command_id == "read-credential-expiry":
        _validate_credential_evidence(evidence)
    elif command_id in {
        "read-deployment-cloudformation-role",
        "read-fence-cloudformation-role",
    }:
        expected_role = (
            "keep-glm52-h1g-fence-service"
            if command_id == "read-fence-cloudformation-role"
            else "keep-glm52-h1g-cloudformation-deployment"
        )
        if (
            set(evidence) != {"role_arn", "role_id"}
            or evidence["role_arn"]
            != (
                "arn:aws:iam::246813579024:role/" + expected_role
            )
            or type(evidence["role_id"]) is not str
            or not evidence["role_id"]
        ):
            raise CampaignRunnerError(
                "CloudFormation role readback drifted"
            )
    else:
        raise CampaignRunnerError(
            "CloudFormation inspection evidence is opaque or drifted"
        )
    return value


def _command_argv_value(
    command: Mapping[str, object],
    flag: str,
) -> str:
    argv = command["argv"]
    assert type(argv) is list
    positions = [
        index for index, item in enumerate(argv) if item == flag
    ]
    if (
        len(positions) != 1
        or positions[0] + 1 >= len(argv)
        or type(argv[positions[0] + 1]) is not str
    ):
        raise CampaignRunnerError(
            "CloudFormation command flag drifted: " + flag
        )
    return str(argv[positions[0] + 1])


def _bind_command_value(
    command: Mapping[str, object],
    *,
    flag: str,
    expected: str,
    replacement: str,
) -> dict[str, object]:
    bound = dict(command)
    argv = list(command["argv"])
    current = _command_argv_value(command, flag)
    if current != expected:
        raise CampaignRunnerError(
            "CloudFormation command target drifted: " + flag
        )
    position = argv.index(flag)
    argv[position + 1] = replacement
    bound["argv"] = argv
    return bound


def _change_set_arn(
    value: object,
    *,
    change_set_name: str,
) -> str:
    pattern = re.compile(
        r"arn:aws:cloudformation:us-west-2:246813579024:changeSet/"
        + re.escape(change_set_name)
        + r"/[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
        r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
    )
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise CampaignRunnerError("CloudFormation change-set ARN drifted")
    return value


def _validate_change_set_evidence(
    evidence: object,
    *,
    stack_id: str,
    change_set_name: str,
    template_body_sha256: str,
    role_arn: str,
) -> dict[str, object]:
    if (
        type(evidence) is not dict
        or set(evidence)
        != {
            "change_set_arn",
            "stack_id",
            "status",
            "execution_status",
            "change_set_type",
            "role_arn",
            "template_body_sha256",
            "changes",
            "changes_identity_sha256",
        }
        or evidence["stack_id"] != stack_id
        or evidence["status"] != "CREATE_COMPLETE"
        or evidence["execution_status"] != "AVAILABLE"
        or evidence["change_set_type"] != "UPDATE"
        or evidence["role_arn"] != role_arn
        or evidence["template_body_sha256"] != template_body_sha256
        or type(evidence["changes"]) is not list
        or not evidence["changes"]
        or evidence["changes_identity_sha256"]
        != canonical_sha256(evidence["changes"])
    ):
        raise CampaignRunnerError(
            "CloudFormation change-set semantic evidence drifted"
        )
    _change_set_arn(
        evidence["change_set_arn"],
        change_set_name=change_set_name,
    )
    for change in evidence["changes"]:
        if (
            type(change) is not dict
            or set(change)
            != {
                "action",
                "logical_resource_id",
                "resource_type",
                "replacement",
            }
            or change["action"] not in {"Add", "Modify", "Import"}
            or type(change["logical_resource_id"]) is not str
            or not change["logical_resource_id"]
            or type(change["resource_type"]) is not str
            or not change["resource_type"].startswith("AWS::")
            or change["replacement"] not in {"False", "Conditional"}
        ):
            raise CampaignRunnerError(
                "CloudFormation change-set semantic evidence drifted"
            )
    return dict(evidence)


def _validate_bootstrap_result(
    value: object,
    *,
    bootstrap: Mapping[str, object],
) -> dict[str, object]:
    if (
        type(value) is not dict
        or set(value)
        != {
            "status",
            "retained_stack_id",
            "fence_stack_id",
            "support_stack_id",
            "termination_protection",
            "anchor_logical_id",
            "template_body_sha256",
        }
        or value["status"] != "BOOTSTRAPPED"
        or value["retained_stack_id"] != bootstrap["retained_stack_id"]
        or value["termination_protection"] is not True
        or value["anchor_logical_id"] != "ContainerAnchor"
        or value["template_body_sha256"]
        != bootstrap["template"]["body_sha256"]
    ):
        raise CampaignRunnerError("migration bootstrap result drifted")
    for label in ("retained", "fence", "support"):
        stack_id = value[label + "_stack_id"]
        if (
            type(stack_id) is not str
            or _STACK_IDS[label].fullmatch(stack_id) is None
        ):
            raise CampaignRunnerError(
                "migration bootstrap stack identity drifted"
            )
    return dict(value)


def _validate_bootstrap_stack_result(
    value: object,
    *,
    bootstrap: Mapping[str, object],
    stack_name: str,
) -> dict[str, object]:
    label = (
        "fence"
        if stack_name == "keep-glm52-h1g-fence"
        else "support"
    )
    if (
        type(value) is not dict
        or set(value)
        != {
            "status",
            "retained_stack_id",
            "stack_name",
            "stack_id",
            "termination_protection",
            "anchor_logical_id",
            "template_body_sha256",
        }
        or value["status"] != "BOOTSTRAPPED_STACK"
        or value["retained_stack_id"] != bootstrap["retained_stack_id"]
        or value["stack_name"] != stack_name
        or type(value["stack_id"]) is not str
        or _STACK_IDS[label].fullmatch(value["stack_id"]) is None
        or value["termination_protection"] is not True
        or value["anchor_logical_id"] != "ContainerAnchor"
        or value["template_body_sha256"]
        != bootstrap["template"]["body_sha256"]
    ):
        raise CampaignRunnerError(
            "migration bootstrap stack result drifted"
        )
    return dict(value)


def _bootstrap_disabled_stacks(
    *,
    deployment: Mapping[str, object],
    services: object,
    journal: object,
) -> dict[str, object]:
    bootstrap = deployment.get("bootstrap")
    if (
        type(bootstrap) is not dict
        or bootstrap.get("coordinator")
        != "MigrationCoordinator.bootstrap"
        or type(bootstrap.get("activation_id")) is not str
        or not bootstrap.get("activation_id")
        or bootstrap.get("retained_stack_name") != "keep-glm52-gpu"
        or bootstrap.get("new_stack_names")
        != ["keep-glm52-h1g-fence", "keep-glm52-h1g-support"]
        or bootstrap.get("stack_role_arns")
        != {
            "keep-glm52-h1g-fence": (
                "arn:aws:iam::246813579024:role/"
                "keep-glm52-h1g-fence-service"
            ),
            "keep-glm52-h1g-support": (
                "arn:aws:iam::246813579024:role/"
                "keep-glm52-h1g-cloudformation-deployment"
            ),
        }
        or bootstrap.get("anchor_logical_id") != "ContainerAnchor"
        or bootstrap.get("anchor_resource_type")
        != "AWS::CloudFormation::WaitConditionHandle"
        or bootstrap.get("termination_protection") is not True
        or bootstrap.get("parameters") != []
        or bootstrap.get("service_assigned_stack_ids_required") is not True
        or type(bootstrap.get("template")) is not dict
    ):
        raise CampaignRunnerError("migration bootstrap manifest drifted")
    observed = _inspect(services, "migration-bootstrap-state", bootstrap)
    if type(observed) is not dict:
        raise CampaignRunnerError("migration bootstrap state is missing")
    if observed.get("status") == "BOOTSTRAPPED":
        for stack_name, label in (
            ("keep-glm52-h1g-fence", "fence"),
            ("keep-glm52-h1g-support", "support"),
        ):
            operation_id = (
                "deploy-disabled:migration-bootstrap:" + label
            )
            scoped_journal = _journal_for_scope(journal, label)
            rows = _journal_entries(scoped_journal, operation_id)
            if rows and rows[-1]["state"] != "COMMITTED":
                request = {
                    **bootstrap,
                    "target_stack_name": stack_name,
                }
                _mutation(
                    services,
                    scoped_journal,
                    stage="deploy-disabled",
                    operation_kind="migration-bootstrap",
                    operation_id=operation_id,
                    request=request,
                    validate_result=(
                        lambda value, stack_name=stack_name: (
                            _validate_bootstrap_stack_result(
                                value,
                                bootstrap=bootstrap,
                                stack_name=stack_name,
                            )
                        )
                    ),
                )
        return _validate_bootstrap_result(observed, bootstrap=bootstrap)
    if (
        set(observed)
        != {
            "status",
            "retained_stack_id",
            "absent_stack_names",
            "existing_stack_ids",
        }
        or observed["status"] != "BOOTSTRAP_REQUIRED"
        or observed["retained_stack_id"] != bootstrap["retained_stack_id"]
        or type(observed["absent_stack_names"]) is not list
        or not observed["absent_stack_names"]
        or not set(observed["absent_stack_names"]).issubset(
            set(bootstrap["new_stack_names"])
        )
        or type(observed["existing_stack_ids"]) is not dict
        or set(observed["existing_stack_ids"])
        != set(bootstrap["new_stack_names"])
        - set(observed["absent_stack_names"])
    ):
        raise CampaignRunnerError(
            "stack existence is ambiguous; bootstrap refused"
        )
    for stack_name, label in (
        ("keep-glm52-h1g-fence", "fence"),
        ("keep-glm52-h1g-support", "support"),
    ):
        request = {**bootstrap, "target_stack_name": stack_name}
        _mutation(
            services,
            _journal_for_scope(journal, label),
            stage="deploy-disabled",
            operation_kind="migration-bootstrap",
            operation_id=(
                "deploy-disabled:migration-bootstrap:" + label
            ),
            request=request,
            validate_result=(
                lambda value, stack_name=stack_name: (
                    _validate_bootstrap_stack_result(
                        value,
                        bootstrap=bootstrap,
                        stack_name=stack_name,
                    )
                )
            ),
        )
    completed = _inspect(
        services,
        "migration-bootstrap-state",
        bootstrap,
    )
    return _validate_bootstrap_result(
        completed,
        bootstrap=bootstrap,
    )


def _stage_result(
    *,
    stage: str,
    package_identity_sha256: str,
    values: Optional[Mapping[str, object]] = None,
) -> dict[str, object]:
    body = {
        "status": "STAGE_COMMITTED",
        "stage": stage,
        "package_identity_sha256": package_identity_sha256,
        **({} if values is None else dict(values)),
    }
    return {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }


def _task12_orphan_plan(
    *,
    package: Mapping[str, object],
    deployment: Mapping[str, object],
) -> dict[str, object]:
    value = deployment.get("task12_orphan_authority")
    if type(value) is not dict or set(value) != {
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
    }:
        raise CampaignRunnerError("Task 12 orphan authority plan drifted")
    artifacts = {
        item.get("artifact_kind"): item
        for item in package.get("reviewed_artifacts", [])
        if type(item) is dict
    }
    monitor_contract = package.get("monitor_contract")
    monitor = PurePosixPath(
        str(
            monitor_contract.get("descriptor_path", "")
            if type(monitor_contract) is dict
            else ""
        )
    )
    if not monitor.is_absolute():
        raise CampaignRunnerError("Task 12 orphan work root is not absolute")
    work = monitor.parent
    expected_prefix = work / "task12" / "orphans"
    paths = {
        "precreate_output": expected_prefix
        / "PRECREATE_ORPHAN_BASELINE.json",
        "postcreate_output_dir": expected_prefix / "support-postcreate",
        "activation_manifest": expected_prefix
        / "ACTIVATION_ORPHAN_AUTHORITY.json",
    }
    if (
        value["materializer"]
        != "aws/glm52-gpu/scripts/materialize_task12_orphan_authority.py"
        or value["support_postcreate_materializer"]
        != "aws/glm52-gpu/scripts/materialize_h1g_support_plane.py"
        or value["support_inputs"] != artifacts.get("SUPPORT_INPUTS")
        or value["support_template"] != artifacts.get("SUPPORT_TEMPLATE")
        or any(value[field] != str(path) for field, path in paths.items())
        or value["lifecycle_resource_id"]
        != (
            "arn:aws:states:us-west-2:246813579024:"
            "stateMachine:keep-glm52-h1g-retained-lifecycle"
        )
        or value["lifecycle_cost_class"] != "STEP_FUNCTIONS_RETAINED"
        or value["source_publisher_arn"]
        != (
            "arn:aws:iam::246813579024:"
            "role/keep-glm52-source-publisher"
        )
        or value["settling_window_seconds"] != 60
    ):
        raise CampaignRunnerError("Task 12 orphan authority plan is not exact")
    return dict(value)


def _validate_task12_precreate_artifact(
    value: object,
    *,
    plan: Mapping[str, object],
    activation_id: object,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != {
        "record_type",
        "activation_id",
        "path",
        "file_sha256",
        "body_sha256",
        "canonical_identity_sha256",
    }:
        raise CampaignRunnerError("Task 12 PRECREATE artifact drifted")
    body = dict(value)
    identity = body.pop("canonical_identity_sha256", None)
    if (
        value["record_type"]
        != "glm52_task12_orphan_precreate_artifact_v1"
        or value["activation_id"] != activation_id
        or value["path"] != plan["precreate_output"]
        or identity != canonical_sha256(body)
    ):
        raise CampaignRunnerError("Task 12 PRECREATE artifact is foreign")
    _sha(value["file_sha256"], "Task 12 PRECREATE file")
    _sha(value["body_sha256"], "Task 12 PRECREATE body")
    return dict(value)


def _validate_task12_activation_manifest(
    value: object,
    *,
    plan: Mapping[str, object],
    activation_id: object,
    precreate: Mapping[str, object],
) -> dict[str, object]:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "activation_id",
        "precreate_authority",
        "support_postcreate_manifest",
        "activation_authority",
        "support_inventory_identity_sha256",
        "path",
        "canonical_identity_sha256",
    }:
        raise CampaignRunnerError("Task 12 activation manifest drifted")
    body = dict(value)
    identity = body.pop("canonical_identity_sha256", None)
    authority = value.get("activation_authority")
    support = value.get("support_postcreate_manifest")
    if (
        value["schema_version"] != 1
        or value["record_type"]
        != "glm52_task13_task12_orphan_activation_manifest_v1"
        or value["account_id"] != ACCOUNT_ID
        or value["region"] != REGION
        or value["run_id"] != RUN_ID
        or value["activation_id"] != activation_id
        or value["precreate_authority"] != precreate
        or value["path"] != plan["activation_manifest"]
        or identity != canonical_sha256(body)
        or type(authority) is not dict
        or type(support) is not dict
        or authority.get("activation_id") != activation_id
        or type(authority.get("version_id")) is not str
        or not authority["version_id"]
        or type(support.get("file_sha256")) is not str
    ):
        raise CampaignRunnerError("Task 12 activation manifest is foreign")
    _sha(
        value["support_inventory_identity_sha256"],
        "Task 12 support inventory",
    )
    _sha(support["file_sha256"], "support postcreate manifest")
    return dict(value)


def _deploy_disabled(
    *,
    package: Mapping[str, object],
    package_identity_sha256: str,
    services: object,
    journal: object,
    execution_authority: ControllerExecutionAuthority,
) -> dict[str, object]:
    deployment = package.get("disabled_deployment")
    exact_stack_tags = [list(tag) for tag in STACK_TAGS]
    if (
        type(deployment) is not dict
        or deployment.get("state") != "ADOPTED_DISABLED"
        or deployment.get("fence_parameters") != []
        or deployment.get("support_parameters") != []
        or deployment.get("tags") != exact_stack_tags
    ):
        raise CampaignRunnerError("disabled deployment manifest drifted")
    artifacts = package.get("reviewed_artifacts")
    if type(artifacts) is not list:
        raise CampaignRunnerError("campaign reviewed artifacts are missing")
    artifacts_by_kind = {
        str(row.get("artifact_kind")): row
        for row in artifacts
        if type(row) is dict
    }
    try:
        evidence = validate_staged_infrastructure_evidence(
            deployment.get("staged_infrastructure_evidence"),
            activation_id=str(package["activation_id"]),
            retained_stack_id=str(deployment["retained_stack_id"]),
            artifacts=artifacts_by_kind,
        )
    except CampaignPackageError as error:
        raise CampaignRunnerError(
            "staged infrastructure evidence drifted"
        ) from error
    readback_contract = deployment.get("adoption_readback_contract")
    if (
        type(readback_contract) is not dict
        or readback_contract
        != {
            "operation_kind": "staged-infrastructure-adoption",
            "read_only": True,
            "exact_evidence_required": True,
            "staged_request_identity_sha256": evidence[
                "staged_request_identity_sha256"
            ],
            "staged_journal_sha256": evidence[
                "staged_journal_sha256"
            ],
            "fixed_artifacts_identity_sha256": evidence[
                "fixed_artifacts_identity_sha256"
            ],
            "worker_activation_allowed": False,
        }
    ):
        raise CampaignRunnerError(
            "staged infrastructure adoption contract drifted"
        )
    observed = _inspect(
        services,
        "staged-infrastructure-adoption",
        evidence,
    )
    if observed != evidence:
        raise CampaignRunnerError(
            "staged infrastructure exact readback drifted"
        )
    probes = package.get("negative_iam_probes")
    if type(probes) is not list or len(probes) != 4:
        raise CampaignRunnerError("negative IAM probe manifest drifted")
    for probe in probes:
        if (
            type(probe) is not dict
            or probe.get("expected_error_code") != "AccessDenied"
        ):
            raise CampaignRunnerError("negative IAM probe is not closed")
        observed = _inspect(services, "negative-iam-probe", probe)
        if observed != {"error_code": "AccessDenied"}:
            raise CampaignRunnerError("negative IAM probe did not deny")
    p5 = _inspect(
        services,
        "p5-zero",
        {
            "account_id": ACCOUNT_ID,
            "region": REGION,
            "profile": PROFILE,
            "instance_type": "p5.48xlarge",
        },
    )
    if p5 != {"active_p5_instance_ids": []}:
        raise CampaignRunnerError("disabled deployment has active P5")
    final_readback = evidence["final_readback"]
    assert type(final_readback) is dict
    summary = {
        "mutating_command_count": 0,
        "negative_probe_count": len(probes),
        "active_p5_count": 0,
        "fence_stack_id": final_readback["fence_stack_id"],
        "support_stack_id": final_readback["support_stack_id"],
        "staged_journal_sha256": evidence["staged_journal_sha256"],
        "fixed_artifacts_identity_sha256": evidence[
            "fixed_artifacts_identity_sha256"
        ],
        "staged_infrastructure_identity_sha256": evidence[
            "canonical_identity_sha256"
        ],
    }
    _record_stage(
        _journal_for_scope(journal, "campaign"),
        stage="deploy-disabled",
        package_identity_sha256=package_identity_sha256,
        reviewed_artifacts_identity_sha256=canonical_sha256(
            package["reviewed_artifacts"]
        ),
        summary=summary,
    )
    return _stage_result(
        stage="deploy-disabled",
        package_identity_sha256=package_identity_sha256,
        values=summary,
    )


def _rehearsal_contract(
    package: Mapping[str, object],
) -> tuple[dict[str, object], str, str]:
    rehearsal = package.get("rehearsal")
    if type(rehearsal) is not dict:
        raise CampaignRunnerError("rehearsal manifest is missing")
    invokes = rehearsal.get("collect_invocations")
    finalizer = rehearsal.get("finalize_invocation")
    if (
        rehearsal.get("execution_state")
        != "BLOCKED_REQUIRES_DISABLED_DEPLOYMENT"
        or rehearsal.get("invoke_type") != "RequestResponse"
        or type(invokes) is not list
        or len(invokes) != 20
        or type(finalizer) is not dict
    ):
        raise CampaignRunnerError("rehearsal execution contract drifted")
    collector_arn = invokes[0].get("function_version_arn")
    qualifier = invokes[0].get("qualifier")
    if (
        type(collector_arn) is not str
        or type(qualifier) is not str
        or not qualifier
        or not collector_arn.endswith(":" + qualifier)
    ):
        raise CampaignRunnerError("collector is not version pinned")
    return rehearsal, collector_arn, qualifier


def _collect_invocation(
    package: Mapping[str, object],
    index: int,
) -> tuple[dict[str, object], str, str]:
    rehearsal, collector_arn, qualifier = _rehearsal_contract(package)
    invokes = rehearsal["collect_invocations"]
    assert type(invokes) is list
    value = invokes[index - 1]
    measurement_id = "measurement-%02d" % index
    expected_scenario = {
        1: "THROTTLING",
        2: "PAGINATION",
        3: "NETWORK_AMBIGUITY",
    }.get(index, "NONE")
    expected_event = {
        "schema_version": 1,
        "record_type": "glm52_task11_collect_rehearsal_v1",
        "activation_id": package.get("activation_id"),
        "measurement_id": measurement_id,
        "scenario": expected_scenario,
        "task11_request": package.get("task11_request"),
        "task11_boundary": package.get("task11_boundary"),
    }
    if (
        type(value) is not dict
        or set(value) != {"function_version_arn", "qualifier", "event"}
        or value["function_version_arn"] != collector_arn
        or value["qualifier"] != qualifier
        or value["event"] != expected_event
    ):
        raise CampaignRunnerError("collector invocation manifest drifted")
    return dict(value), collector_arn, qualifier


def _validate_measurement_readback(
    services: object,
    *,
    payload: Mapping[str, object],
    measurement_id: str,
    scenario: str,
) -> RehearsalMeasurement:
    observed = _inspect(
        services,
        "measurement",
        {
            "key": payload["key"],
            "version_id": payload["version_id"],
            "file_sha256": payload["file_sha256"],
            "payload": dict(payload),
        },
    )
    if (
        type(observed) is not dict
        or set(observed)
        != {"key", "version_id", "file_sha256", "measurement"}
        or observed["key"] != payload["key"]
        or observed["version_id"] != payload["version_id"]
        or observed["file_sha256"] != payload["file_sha256"]
    ):
        raise CampaignRunnerError(
            "immutable measurement coordinate readback drifted"
        )
    try:
        measurement = rehearsal_measurement_from_mapping(
            observed["measurement"]
        )
    except (TypeError, ValueError) as error:
        raise CampaignRunnerError(str(error)) from error
    if (
        measurement.rehearsal_id != measurement_id
        or measurement.provenance != "DEPLOYED_REHEARSAL"
        or measurement.injected_failure_kind != scenario
    ):
        raise CampaignRunnerError("deployed measurement identity drifted")
    return measurement


def _collect_one(
    *,
    package: Mapping[str, object],
    services: object,
    journal: object,
    index: int,
) -> RehearsalMeasurement:
    invocation, collector_arn, qualifier = _collect_invocation(package, index)
    measurement_id = "measurement-%02d" % index
    stage = (
        "collect-first-five" if index <= 5 else "collect-remaining"
    )
    payload = _mutation(
        services,
        journal,
        stage=stage,
        operation_kind="collector-invoke",
        operation_id="collect:" + measurement_id,
        request=invocation,
        validate_result=lambda value: validate_collect_invoke_result(
            value,
            expected_function_version_arn=collector_arn,
            expected_version=qualifier,
            expected_measurement_id=measurement_id,
        ),
    )
    assert type(payload) is dict
    event = invocation["event"]
    assert type(event) is dict
    return _validate_measurement_readback(
        services,
        payload=payload,
        measurement_id=measurement_id,
        scenario=str(event["scenario"]),
    )


def _collect_first_five(
    *,
    package: Mapping[str, object],
    package_identity_sha256: str,
    services: object,
    journal: object,
) -> dict[str, object]:
    _require_stage(journal, "deploy-disabled", package=package)
    rehearsal, _collector_arn, _qualifier = _rehearsal_contract(package)
    batches = rehearsal.get("batches")
    if (
        type(batches) is not list
        or len(batches) != 2
        or batches[0]
        != {
            "batch_id": "cold-start-01",
            "concurrency": 5,
            "measurement_ids": [
                "measurement-%02d" % index for index in range(1, 6)
            ],
            "required_distinct_cold_environments": 5,
        }
    ):
        raise CampaignRunnerError("cold-start batch contract drifted")
    with ThreadPoolExecutor(
        max_workers=5,
        thread_name_prefix="task13-cold",
    ) as executor:
        futures = [
            executor.submit(
                _collect_one,
                package=package,
                services=services,
                journal=journal,
                index=index,
            )
            for index in range(1, 6)
        ]
        measurements = tuple(future.result() for future in futures)
    cold_environments = [
        item.lambda_environment_id
        for item in measurements
        if item.cold_start
    ]
    failure_kinds = [
        kind
        for kind in INJECTED_FAILURE_KINDS
        if kind
        in {
            item.injected_failure_kind
            for item in measurements
            if item.injected_failure_kind != "NONE"
        }
    ]
    if (
        len(cold_environments) != 5
        or len(set(cold_environments)) != 5
    ):
        raise CampaignRunnerError(
            "first batch did not prove five distinct cold environments"
        )
    if tuple(failure_kinds) != INJECTED_FAILURE_KINDS:
        raise CampaignRunnerError(
            "first batch did not prove the exact fault matrix"
        )
    summary = {
        "measurement_count": 5,
        "cold_environment_count": 5,
        "failure_kinds": failure_kinds,
    }
    _record_stage(
        journal,
        stage="collect-first-five",
        package_identity_sha256=package_identity_sha256,
        reviewed_artifacts_identity_sha256=canonical_sha256(
            package["reviewed_artifacts"]
        ),
        summary=summary,
    )
    return _stage_result(
        stage="collect-first-five",
        package_identity_sha256=package_identity_sha256,
        values=summary,
    )


def _collect_remaining(
    *,
    package: Mapping[str, object],
    package_identity_sha256: str,
    services: object,
    journal: object,
) -> dict[str, object]:
    _require_stage(journal, "collect-first-five", package=package)
    rehearsal, _collector_arn, _qualifier = _rehearsal_contract(package)
    batches = rehearsal.get("batches")
    if (
        type(batches) is not list
        or len(batches) != 2
        or batches[1]
        != {
            "batch_id": "remaining-02",
            "concurrency": 1,
            "measurement_ids": [
                "measurement-%02d" % index for index in range(6, 21)
            ],
            "required_distinct_cold_environments": 0,
        }
    ):
        raise CampaignRunnerError("remaining batch contract drifted")
    measurements = [
        _collect_one(
            package=package,
            services=services,
            journal=journal,
            index=index,
        )
        for index in range(6, 21)
    ]
    if any(item.cold_start for item in measurements):
        raise CampaignRunnerError(
            "remaining collector batch unexpectedly claimed a cold start"
        )
    summary = {
        "measurement_count": 15,
        "cold_environment_count": 0,
    }
    _record_stage(
        journal,
        stage="collect-remaining",
        package_identity_sha256=package_identity_sha256,
        reviewed_artifacts_identity_sha256=canonical_sha256(
            package["reviewed_artifacts"]
        ),
        summary=summary,
    )
    return _stage_result(
        stage="collect-remaining",
        package_identity_sha256=package_identity_sha256,
        values=summary,
    )


def _read_measurement(
    *,
    package: Mapping[str, object],
    services: object,
    journal: object,
    index: int,
) -> RehearsalMeasurement:
    invocation, collector_arn, qualifier = _collect_invocation(package, index)
    measurement_id = "measurement-%02d" % index
    original_stage = (
        "collect-first-five" if index <= 5 else "collect-remaining"
    )
    payload = _read_committed_mutation(
        services,
        journal,
        stage=original_stage,
        operation_kind="collector-invoke",
        operation_id="collect:" + measurement_id,
        request=invocation,
        validate_result=lambda value: validate_collect_invoke_result(
            value,
            expected_function_version_arn=collector_arn,
            expected_version=qualifier,
            expected_measurement_id=measurement_id,
        ),
    )
    assert type(payload) is dict
    event = invocation["event"]
    assert type(event) is dict
    return _validate_measurement_readback(
        services,
        payload=payload,
        measurement_id=measurement_id,
        scenario=str(event["scenario"]),
    )


def _exact_capture_bytes(value: object, label: str) -> bytes:
    if type(value) is bytes:
        return value
    if type(value) is bytearray:
        return bytes(value)
    read = getattr(value, "read", None)
    if callable(read):
        raw = read()
        if type(raw) is bytes:
            return raw
    raise CampaignRunnerError(label + " is not exact bytes")


def _validate_finalization_result(
    value: object,
    *,
    collector_arn: str,
    qualifier: str,
) -> tuple[dict[str, object], FinalizationCapture]:
    if type(value) is not dict or "Payload" not in value:
        raise CampaignRunnerError("FINALIZE invoke result is not exact")
    payload_raw = _exact_capture_bytes(
        value["Payload"],
        "FINALIZE payload",
    )
    normalized = dict(value)
    normalized["Payload"] = payload_raw
    try:
        payload = validate_finalize_invoke_result(
            normalized,
            expected_function_version_arn=collector_arn,
            expected_version=qualifier,
        )
    except CampaignPackageError as error:
        raise CampaignRunnerError(str(error)) from error
    if (
        type(value.get("StatusCode")) is not int
        or value["StatusCode"] != 200
        or type(value.get("ExecutedVersion")) is not str
        or value["ExecutedVersion"] != qualifier
        or payload_raw != canonical_json_bytes(payload)
    ):
        raise CampaignRunnerError(
            "FINALIZE metadata or canonical payload bytes drifted"
        )
    return payload, FinalizationCapture(
        status_code=value["StatusCode"],
        executed_version=value["ExecutedVersion"],
        payload_bytes=payload_raw,
        gate_bytes=b"",
    )


def _finalize(
    *,
    package: Mapping[str, object],
    package_identity_sha256: str,
    services: object,
    journal: object,
    finalization_capture: Optional[
        Callable[[FinalizationCapture], None]
    ] = None,
) -> dict[str, object]:
    _require_stage(journal, "collect-remaining", package=package)
    rehearsal, collector_arn, qualifier = _rehearsal_contract(package)
    measurements = tuple(
        _read_measurement(
            package=package,
            services=services,
            journal=journal,
            index=index,
        )
        for index in range(1, 21)
    )
    try:
        local_gate = verify_no_post_rehearsals(measurements)
    except (TypeError, ValueError) as error:
        raise CampaignRunnerError(str(error)) from error
    finalizer = rehearsal["finalize_invocation"]
    expected_finalizer = {
        "function_version_arn": collector_arn,
        "qualifier": qualifier,
        "event": {
            "schema_version": 1,
            "record_type": "glm52_task11_finalize_rehearsal_gate_v1",
            "activation_id": package.get("activation_id"),
        },
    }
    if finalizer != expected_finalizer:
        raise CampaignRunnerError("finalizer invocation manifest drifted")
    validated = _mutation(
        services,
        journal,
        stage="finalize",
        operation_kind="finalize-invoke",
        operation_id="finalize:closure-budget",
        request=finalizer,
        validate_result=lambda value: _validate_finalization_result(
            value,
            collector_arn=collector_arn,
            qualifier=qualifier,
        ),
    )
    if (
        type(validated) is not tuple
        or len(validated) != 2
        or type(validated[0]) is not dict
        or type(validated[1]) is not FinalizationCapture
    ):
        raise CampaignRunnerError("validated FINALIZE capture drifted")
    payload, invoke_capture = validated
    if (
        payload["measurements_identity_sha256"]
        != local_gate.measurements_identity_sha256
        or payload["measurement_count"] != local_gate.measurement_count
        or payload["cold_environment_count"]
        != local_gate.cold_environment_count
    ):
        raise CampaignRunnerError(
            "finalized gate does not bind the inspected measurement set"
        )
    observed = _inspect(
        services,
        "gate",
        {
            "key": payload["key"],
            "version_id": payload["version_id"],
            "file_sha256": payload["file_sha256"],
            "body_sha256": payload["body_sha256"],
            "payload": payload,
        },
    )
    expected_observed = {
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
    if (
        type(observed) is not dict
        or set(observed) != {"summary", "content"}
        or observed["summary"] != expected_observed
    ):
        raise CampaignRunnerError("immutable gate readback drifted")
    gate_raw = _exact_capture_bytes(
        observed["content"],
        "immutable gate readback",
    )
    try:
        gate_value = json.loads(gate_raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CampaignRunnerError(
            "immutable gate readback is not JSON"
        ) from error
    if (
        type(gate_value) is not dict
        or gate_raw != canonical_json_bytes(gate_value)
        or type(gate_value.get("deployment_identity_sha256")) is not str
    ):
        raise CampaignRunnerError(
            "immutable gate canonical bytes or deployment identity drifted"
        )
    try:
        parsed_gate = parse_deployed_gate_document(
            gate_raw,
            activation_id=str(package["activation_id"]),
            deployment_identity_sha256=str(
                gate_value["deployment_identity_sha256"]
            ),
        )
    except ValueError as error:
        raise CampaignRunnerError(str(error)) from error
    if (
        hashlib.sha256(gate_raw).hexdigest() != payload["file_sha256"]
        or parsed_gate.canonical_body_sha256 != payload["body_sha256"]
        or parsed_gate.measurement_count != payload["measurement_count"]
        or parsed_gate.cold_environment_count
        != payload["cold_environment_count"]
        or parsed_gate.measurements_identity_sha256
        != payload["measurements_identity_sha256"]
    ):
        raise CampaignRunnerError(
            "FINALIZE payload and immutable gate bytes drifted"
        )
    capture = FinalizationCapture(
        status_code=invoke_capture.status_code,
        executed_version=invoke_capture.executed_version,
        payload_bytes=invoke_capture.payload_bytes,
        gate_bytes=gate_raw,
    )
    summary = {
        "measurement_count": local_gate.measurement_count,
        "cold_environment_count": local_gate.cold_environment_count,
        "measurements_identity_sha256": (
            local_gate.measurements_identity_sha256
        ),
        "gate_version_id": payload["version_id"],
        "finalizer_payload": dict(payload),
        "finalizer_function_version_arn": collector_arn,
        "finalizer_executed_version": qualifier,
        "immutable_gate_readback": dict(expected_observed),
    }
    _record_stage(
        journal,
        stage="finalize",
        package_identity_sha256=package_identity_sha256,
        reviewed_artifacts_identity_sha256=canonical_sha256(
            package["reviewed_artifacts"]
        ),
        summary=summary,
    )
    if finalization_capture is not None:
        finalization_capture(capture)
    return _stage_result(
        stage="finalize",
        package_identity_sha256=package_identity_sha256,
        values=summary,
    )


def _validate_h100_result(value: object) -> dict[str, object]:
    if (
        type(value) is not dict
        or set(value)
        != {
            "status",
            "instance_type",
            "capacity_type",
            "training_steps",
            "cross_node_resume",
            "peak_memory_gib",
            "instance_ids",
            "qualification_cache_prefix",
            "qualification_cache_manifest_sha256",
            "source_ready",
            "termination_requested",
            "source_termination_verified",
            "h100_resume_ready",
            "max_concurrent_active_instances",
            "active_instance_count",
        }
        or value["status"] != "QUALIFIED"
        or value["instance_type"] != "p5.48xlarge"
        or value["capacity_type"] != "ON_DEMAND"
        or value["training_steps"] != 2
        or value["cross_node_resume"] is not True
        or type(value["peak_memory_gib"]) not in {int, float}
        or type(value["peak_memory_gib"]) is bool
        or value["peak_memory_gib"] >= 70
        or type(value["instance_ids"]) is not list
        or len(value["instance_ids"]) != 2
        or len(set(value["instance_ids"])) != 2
        or value["source_termination_verified"] is not True
        or value["max_concurrent_active_instances"] != 1
        or value["active_instance_count"] != 0
        or any(
            type(instance_id) is not str
            or _INSTANCE_ID.fullmatch(instance_id) is None
            for instance_id in value["instance_ids"]
        )
    ):
        raise CampaignRunnerError("H100 qualification result drifted")
    if value["instance_ids"][0] == value["instance_ids"][1]:
        raise CampaignRunnerError(
            "H100 replacement instance did not differ from source"
        )
    manifest_sha = _sha(
        value["qualification_cache_manifest_sha256"],
        "H100 qualification-cache manifest",
    )
    if value["qualification_cache_prefix"] != (
        "qualification-cache/seeds/%s/%s/" % (RUN_ID, manifest_sha)
    ):
        raise CampaignRunnerError(
            "H100 qualification cache binding drifted"
        )
    _validate_immutable_marker(
        value["source_ready"],
        expected_key=(
            "campaigns/%s/qualification/SOURCE_NODE_READY.json" % RUN_ID
        ),
    )
    _validate_immutable_marker(
        value["termination_requested"],
        expected_key=(
            "campaigns/%s/qualification/"
            "QUALIFICATION_TERMINATION_REQUESTED.json" % RUN_ID
        ),
    )
    _validate_immutable_marker(
        value["h100_resume_ready"],
        expected_key=(
            "campaigns/%s/qualification/H100_RESUME_READY.json" % RUN_ID
        ),
    )
    return dict(value)


def _validate_immutable_marker(
    value: object,
    *,
    expected_key: str,
) -> dict[str, object]:
    if (
        type(value) is not dict
        or set(value)
        != {
            "key",
            "version_id",
            "file_sha256",
            "body_sha256",
            "canonical_identity_sha256",
        }
        or value["key"] != expected_key
        or type(value["version_id"]) is not str
        or value["version_id"].lower() in _MUTABLE_VERSION_WORDS
        or not value["version_id"].isascii()
        or not value["version_id"].strip()
    ):
        raise CampaignRunnerError(
            "immutable marker coordinate drifted: " + expected_key
        )
    _sha(value["file_sha256"], expected_key + " file")
    _sha(value["body_sha256"], expected_key + " body")
    body = dict(value)
    identity = body.pop("canonical_identity_sha256")
    if identity != canonical_sha256(body):
        raise CampaignRunnerError(
            "immutable marker coordinate identity drifted: " + expected_key
        )
    return dict(value)


def _exact_read_marker(
    services: object,
    marker: object,
    *,
    expected_key: str,
) -> dict[str, object]:
    coordinate = _validate_immutable_marker(
        marker,
        expected_key=expected_key,
    )
    observed = _inspect(services, "immutable-marker", coordinate)
    if observed != coordinate:
        raise CampaignRunnerError(
            "immutable marker exact VersionId readback drifted"
        )
    return coordinate


def _exact_read_artifact(
    services: object,
    coordinate: object,
) -> dict[str, object]:
    if (
        type(coordinate) is not dict
        or set(coordinate) != _ARTIFACT_FIELDS
        or type(coordinate["artifact_kind"]) is not str
        or not coordinate["artifact_kind"]
        or type(coordinate["bucket"]) is not str
        or not coordinate["bucket"]
        or type(coordinate["key"]) is not str
        or not coordinate["key"]
        or type(coordinate["version_id"]) is not str
        or coordinate["version_id"].lower() in _MUTABLE_VERSION_WORDS
        or not coordinate["version_id"].isascii()
        or not coordinate["version_id"].strip()
    ):
        raise CampaignRunnerError("immutable artifact coordinate drifted")
    _sha(
        coordinate["body_sha256"],
        str(coordinate["artifact_kind"]) + " body",
    )
    _sha(
        coordinate["file_sha256"],
        str(coordinate["artifact_kind"]) + " file",
    )
    if coordinate["artifact_kind"] in SEMANTIC_GATE_PINS:
        try:
            validate_semantic_gate_coordinate(coordinate)
        except TransportGateError as error:
            raise CampaignRunnerError(
                "immutable semantic transport gate drifted"
            ) from error
    observed = _inspect(
        services,
        "immutable-artifact",
        dict(coordinate),
    )
    if observed != coordinate:
        raise CampaignRunnerError(
            "immutable artifact exact VersionId readback drifted"
        )
    return dict(coordinate)


def _validate_cache_seed_result(value: object) -> dict[str, object]:
    if (
        type(value) is not dict
        or set(value)
        != {
            "status",
            "instance_type",
            "capacity_type",
            "active_instance_count",
            "gpu_seconds",
            "teacher_row_count",
            "qualification_cache_prefix",
            "qualification_cache_manifest_sha256",
            "seed_ready",
            "teacher_ready",
            "teardown_verified",
        }
        or value["status"] != "SEED_READY"
        or value["instance_type"] != "p5.48xlarge"
        or value["capacity_type"] != "ON_DEMAND"
        or value["active_instance_count"] != 0
        or type(value["gpu_seconds"]) is not int
        or value["gpu_seconds"] <= 0
        or value["gpu_seconds"] > 6 * 60 * 60
        or value["teacher_row_count"] != 1
        or value["teardown_verified"] is not True
    ):
        raise CampaignRunnerError("qualification-cache seed result drifted")
    manifest_sha = _sha(
        value["qualification_cache_manifest_sha256"],
        "qualification-cache manifest",
    )
    if value["qualification_cache_prefix"] != (
        "qualification-cache/seeds/%s/%s/" % (RUN_ID, manifest_sha)
    ):
        raise CampaignRunnerError(
            "qualification-cache prefix is not manifest bound"
        )
    _validate_immutable_marker(
        value["seed_ready"],
        expected_key=(
            "campaigns/%s/qualification/"
            "QUALIFICATION_CACHE_SEED_READY.json" % RUN_ID
        ),
    )
    _validate_immutable_marker(
        value["teacher_ready"],
        expected_key=(
            "qualification-cache/seeds/%s/%s/TEACHER_CACHE_READY.json"
            % (RUN_ID, manifest_sha)
        ),
    )
    return dict(value)


def _qualification_cache_seed(
    *,
    package: Mapping[str, object],
    package_identity_sha256: str,
    services: object,
    journal: object,
) -> dict[str, object]:
    _require_stage(journal, "finalize", package=package)
    plan = package.get("qualification_cache_seed")
    if (
        type(plan) is not dict
        or plan.get("authorized") is not False
        or plan.get("execution_state")
        != "BLOCKED_REQUIRES_REHEARSAL_GATE"
        or plan.get("driver_record_type")
        != "glm52_task13_repository_driver_v2"
        or plan.get("activation_id") != package.get("activation_id")
        or plan.get("driver_operation_kind")
        != "qualification-cache-seed"
        or plan.get("instance_type") != "p5.48xlarge"
        or plan.get("capacity_type") != "ON_DEMAND"
        or plan.get("max_active_instances") != 1
        or plan.get("maximum_gpu_hours") != 6
        or plan.get("required_teacher_rows") != 1
        or plan.get("ready_marker")
        != "QUALIFICATION_CACHE_SEED_READY.json"
        or plan.get("teacher_ready_marker") != "TEACHER_CACHE_READY.json"
        or plan.get("teardown_required") is not True
        or type(plan.get("input")) is not dict
    ):
        raise CampaignRunnerError(
            "qualification-cache seed contract drifted"
        )
    result = _mutation(
        services,
        journal,
        stage="qualification-cache-seed",
        operation_kind="qualification-cache-seed",
        operation_id="qualification-cache-seed:one-row",
        request=dict(plan),
        validate_result=_validate_cache_seed_result,
    )
    assert type(result) is dict
    _exact_read_marker(
        services,
        result["seed_ready"],
        expected_key=(
            "campaigns/%s/qualification/"
            "QUALIFICATION_CACHE_SEED_READY.json" % RUN_ID
        ),
    )
    _exact_read_marker(
        services,
        result["teacher_ready"],
        expected_key=(
            "qualification-cache/seeds/%s/%s/TEACHER_CACHE_READY.json"
            % (
                RUN_ID,
                result["qualification_cache_manifest_sha256"],
            )
        ),
    )
    summary = {
        "qualification_cache_prefix": result[
            "qualification_cache_prefix"
        ],
        "qualification_cache_manifest_sha256": result[
            "qualification_cache_manifest_sha256"
        ],
        "gpu_seconds": result["gpu_seconds"],
        "teacher_row_count": 1,
    }
    _record_stage(
        journal,
        stage="qualification-cache-seed",
        package_identity_sha256=package_identity_sha256,
        reviewed_artifacts_identity_sha256=canonical_sha256(
            package["reviewed_artifacts"]
        ),
        summary=summary,
    )
    return _stage_result(
        stage="qualification-cache-seed",
        package_identity_sha256=package_identity_sha256,
        values=summary,
    )


def _read_cache_seed_result(
    *,
    package: Mapping[str, object],
    services: object,
    journal: object,
) -> dict[str, object]:
    plan = package.get("qualification_cache_seed")
    if type(plan) is not dict:
        raise CampaignRunnerError(
            "qualification-cache seed contract is missing"
        )
    result = _read_committed_mutation(
        services,
        journal,
        stage="qualification-cache-seed",
        operation_kind="qualification-cache-seed",
        operation_id="qualification-cache-seed:one-row",
        request=dict(plan),
        validate_result=_validate_cache_seed_result,
    )
    assert type(result) is dict
    return result


def _h100_qualification(
    *,
    package: Mapping[str, object],
    package_identity_sha256: str,
    services: object,
    journal: object,
) -> dict[str, object]:
    _require_stage(
        journal,
        "qualification-cache-seed",
        package=package,
    )
    seed = _read_cache_seed_result(
        package=package,
        services=services,
        journal=journal,
    )
    plan = package.get("h100_qualification")
    if (
        type(plan) is not dict
        or plan.get("authorized") is not False
        or plan.get("execution_state")
        != "BLOCKED_REQUIRES_REHEARSAL_GATE"
        or plan.get("driver_record_type")
        != "glm52_task13_repository_driver_v2"
        or plan.get("activation_id") != package.get("activation_id")
        or plan.get("driver_operation_kind")
        != "h100-qualification"
        or plan.get("instance_type") != "p5.48xlarge"
        or plan.get("capacity_type") != "ON_DEMAND"
        or plan.get("max_active_instances") != 1
        or plan.get("required_distinct_instance_ids") != 2
        or plan.get("required_training_steps") != 2
        or plan.get("cross_node_resume_required") is not True
        or plan.get("peak_memory_limit_gib_exclusive") != 70
        or plan.get("production_effects_required") != 0
        or plan.get("source_ready_marker") != "SOURCE_NODE_READY.json"
        or plan.get("termination_requested_marker")
        != "QUALIFICATION_TERMINATION_REQUESTED.json"
        or plan.get("resume_ready_marker") != "H100_RESUME_READY.json"
        or plan.get("source_termination_required") is not True
        or plan.get("replacement_instance_required") is not True
    ):
        raise CampaignRunnerError("H100 qualification contract drifted")
    request = {
        **plan,
        "qualification_cache_prefix": seed[
            "qualification_cache_prefix"
        ],
        "qualification_cache_manifest_sha256": seed[
            "qualification_cache_manifest_sha256"
        ],
        "seed_ready": seed["seed_ready"],
        "teacher_ready": seed["teacher_ready"],
    }
    result = _mutation(
        services,
        journal,
        stage="h100-qualification",
        operation_kind="h100-qualification",
        operation_id="h100-qualification:two-step-resume",
        request=request,
        validate_result=_validate_h100_result,
    )
    assert type(result) is dict
    for field, key in (
        (
            "source_ready",
            "campaigns/%s/qualification/SOURCE_NODE_READY.json" % RUN_ID,
        ),
        (
            "termination_requested",
            "campaigns/%s/qualification/"
            "QUALIFICATION_TERMINATION_REQUESTED.json" % RUN_ID,
        ),
        (
            "h100_resume_ready",
            "campaigns/%s/qualification/H100_RESUME_READY.json" % RUN_ID,
        ),
    ):
        _exact_read_marker(
            services,
            result[field],
            expected_key=key,
        )
    if (
        result["qualification_cache_prefix"]
        != seed["qualification_cache_prefix"]
        or result["qualification_cache_manifest_sha256"]
        != seed["qualification_cache_manifest_sha256"]
    ):
        raise CampaignRunnerError(
            "H100 result is not bound to accepted seed"
        )
    summary = {
        "distinct_instance_count": len(result["instance_ids"]),
        "peak_memory_gib": result["peak_memory_gib"],
        "training_steps": result["training_steps"],
        "h100_resume_ready": result["h100_resume_ready"],
    }
    _record_stage(
        journal,
        stage="h100-qualification",
        package_identity_sha256=package_identity_sha256,
        reviewed_artifacts_identity_sha256=canonical_sha256(
            package["reviewed_artifacts"]
        ),
        summary=summary,
    )
    return _stage_result(
        stage="h100-qualification",
        package_identity_sha256=package_identity_sha256,
        values=summary,
    )


def _validate_launch_result(
    value: object,
    *,
    request: Mapping[str, object],
) -> dict[str, object]:
    if (
        type(value) is not dict
        or set(value)
        != {
            "status",
            "accepted",
            "workflow_invocation_count",
            "instance_type",
            "capacity_type",
            "execution_arn",
            "instance_id",
            "availability_zone",
            "capacity_outcomes",
            "workflow_output",
            "same_token_identity_sha256",
            "reserve_identity_sha256",
            "spend_authority_identity_sha256",
            "action_identity_sha256",
            "task9_custody_identity_sha256",
            "spend_authority",
            "action_authority",
            "terminal_contract",
            "workflow_reconciliation",
        }
        or value["workflow_invocation_count"]
        != request["workflow_reconciliation_contract"][
            "workflow_invocation_count"
        ]
        or value["instance_type"] != "p5.48xlarge"
        or value["capacity_type"] != "ON_DEMAND"
        or type(value["execution_arn"]) is not str
        or re.fullmatch(
            r"arn:aws:states:us-west-2:246813579024:execution:"
            r"keep-glm52-h1g-production:[A-Za-z0-9_-]{1,80}",
            value["execution_arn"],
        )
        is None
    ):
        raise CampaignRunnerError("guarded launch result drifted")
    for field in (
        "same_token_identity_sha256",
        "reserve_identity_sha256",
        "spend_authority_identity_sha256",
        "action_identity_sha256",
        "task9_custody_identity_sha256",
    ):
        _sha(value[field], "production " + field)
    if (
        value["same_token_identity_sha256"]
        != request["execution_authority"][
            "same_token_identity_sha256"
        ]
        or any(
            value[field] != request["execution_authority"][field]
            for field in (
                "reserve_identity_sha256",
                "spend_authority_identity_sha256",
                "action_identity_sha256",
                "task9_custody_identity_sha256",
            )
        )
    ):
        raise CampaignRunnerError(
            "production launch authority identity drifted"
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
    reconciliation = value["workflow_reconciliation"]
    if (
        type(reconciliation) is not dict
        or set(reconciliation) != reconciliation_fields
        or reconciliation["classification"] != value["status"]
        or any(
            reconciliation[field] != value[field]
            for field in reconciliation_fields
            if field
            not in {
                "classification",
                "observation_identity_sha256",
            }
        )
        or reconciliation["observation_identity_sha256"]
        != canonical_sha256(
            {
                field: reconciliation[field]
                for field in reconciliation_fields
                if field != "observation_identity_sha256"
            }
        )
    ):
        raise CampaignRunnerError(
            "production workflow reconciliation drifted"
        )
    outcomes = value["capacity_outcomes"]
    attempts = request["capacity_plan"]["attempts"]
    if (
        type(outcomes) is not list
        or not outcomes
        or len(outcomes) > 6
    ):
        raise CampaignRunnerError("production capacity outcomes drifted")
    for index, outcome in enumerate(outcomes):
        expected = attempts[index]
        if (
            type(outcome) is not dict
            or set(outcome)
            != {"attempt", "availability_zone", "outcome"}
            or outcome["attempt"] != expected["attempt"]
            or outcome["availability_zone"]
            != expected["availability_zone"]
            or outcome["outcome"]
            not in {"CAPACITY_REJECTED", "WORKER_ALLOCATED"}
        ):
            raise CampaignRunnerError(
                "production capacity outcome order drifted"
            )
    allocated = [
        outcome
        for outcome in outcomes
        if outcome["outcome"] == "WORKER_ALLOCATED"
    ]
    output = value["workflow_output"]
    if (
        type(output) is not dict
        or set(output)
        != {
            "bucket",
            "key",
            "version_id",
            "file_sha256",
            "body_sha256",
        }
        or output["bucket"]
        != "keep-glm52-models-246813579024-us-west-2"
        or type(output["key"]) is not str
        or output["key"]
        != request["workflow_reconciliation_contract"]["key"]
        or type(output["version_id"]) is not str
        or output["version_id"].lower() in _MUTABLE_VERSION_WORDS
        or not output["version_id"]
    ):
        raise CampaignRunnerError(
            "production workflow output coordinate drifted"
        )
    _sha(output["file_sha256"], "production workflow output file")
    _sha(output["body_sha256"], "production workflow output body")
    accepted = value["accepted"]
    if accepted is True:
        if (
            value["status"] != "WORKER_ALLOCATED"
            or len(allocated) != 1
            or outcomes[-1] != allocated[0]
            or type(value["instance_id"]) is not str
            or _INSTANCE_ID.fullmatch(value["instance_id"]) is None
            or value["availability_zone"]
            != allocated[0]["availability_zone"]
        ):
            raise CampaignRunnerError("accepted launch result drifted")
        spend = value["spend_authority"]
        action = value["action_authority"]
        terminal = value["terminal_contract"]
        if (
            type(spend) is not dict
            or set(spend)
            != {
                "account_id",
                "run_id",
                "remaining_gpu_seconds",
                "ledger_version_id",
                "ledger_body_sha256",
            }
            or spend["account_id"] != ACCOUNT_ID
            or spend["run_id"] != RUN_ID
            or type(spend["remaining_gpu_seconds"]) is not int
            or spend["remaining_gpu_seconds"] <= 3600
            or type(spend["ledger_version_id"]) is not str
            or spend["ledger_version_id"].lower()
            in _MUTABLE_VERSION_WORDS
            or not spend["ledger_version_id"]
        ):
            raise CampaignRunnerError(
                "production spend authority drifted"
            )
        _sha(spend["ledger_body_sha256"], "production spend ledger")
        if (
            type(action) is not dict
            or set(action)
            != {
                "activation_id",
                "action_kind",
                "action_count",
                "action_version_id",
                "action_body_sha256",
            }
            or action["activation_id"] != request["activation_id"]
            or action["action_kind"] != "PRODUCTION_SUBMISSION"
            or action["action_count"] != 1
            or type(action["action_version_id"]) is not str
            or action["action_version_id"].lower()
            in _MUTABLE_VERSION_WORDS
            or not action["action_version_id"]
        ):
            raise CampaignRunnerError(
                "production action authority drifted"
            )
        _sha(action["action_body_sha256"], "production action")
        if (
            type(terminal) is not dict
            or terminal
            != {
                "required_terminal_markers": [
                    "CAMPAIGN_DRAINED.json",
                    "TERMINAL_VERIFIED.json",
                ],
                "monitor_route": (
                    "aws/glm52-gpu/scripts/"
                    "sky_campaign_break_glass.sh status"
                ),
            }
        ):
            raise CampaignRunnerError(
                "production terminal contract drifted"
            )
    elif accepted is False:
        if (
            value["status"] != "CAPACITY_EXHAUSTED"
            or len(outcomes) != 6
            or allocated
            or value["instance_id"] is not None
            or value["availability_zone"] is not None
            or value["spend_authority"] is not None
            or value["action_authority"] is not None
            or value["terminal_contract"] is not None
        ):
            raise CampaignRunnerError("capacity failure result drifted")
    else:
        raise CampaignRunnerError("guarded launch acceptance is not exact")
    return dict(value)


def _validate_presend_launch_authority(
    value: object,
    *,
    attempt: Mapping[str, object],
    activation_id: object,
    immutable_inputs_identity_sha256: str,
    prior_same_token_identity_sha256: Optional[str],
) -> dict[str, object]:
    if type(value) is not dict or set(value) != {
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
        "authority_record",
        "spend_authority",
        "action_authority",
        "liability_action",
        "canonical_identity_sha256",
    }:
        raise CampaignRunnerError("pre-send launch authority field set drifted")
    identity = value["canonical_identity_sha256"]
    body = dict(value)
    del body["canonical_identity_sha256"]
    same_token = _sha(
        value["same_token_identity_sha256"],
        "launch same-token identity",
    )
    for field in (
        "reserve_identity_sha256",
        "spend_authority_identity_sha256",
        "action_identity_sha256",
        "task9_custody_identity_sha256",
    ):
        _sha(value[field], "launch " + field)
    if (
        identity != canonical_sha256(body)
        or value["account_id"] != ACCOUNT_ID
        or value["region"] != REGION
        or value["run_id"] != RUN_ID
        or value["activation_id"] != activation_id
        or value["attempt"] != attempt["attempt"]
        or value["availability_zone"] != attempt["availability_zone"]
        or value["instance_type"] != "p5.48xlarge"
        or value["capacity_type"] != "ON_DEMAND"
        or value["shared_attempt_counter"] != attempt["attempt"]
        or value["immutable_inputs_identity_sha256"]
        != immutable_inputs_identity_sha256
        or (
            prior_same_token_identity_sha256 is not None
            and same_token != prior_same_token_identity_sha256
        )
    ):
        raise CampaignRunnerError("pre-send launch authority identity drifted")
    spend = value["spend_authority"]
    action = value["action_authority"]
    liability = value["liability_action"]
    sole_sender = value["sole_sender_authority"]
    _validate_immutable_marker(
        value["authority_record"],
        expected_key=(
            f"campaigns/{RUN_ID}/authorities/task13/{activation_id}/"
            f"launch-attempt-{attempt['attempt']:02d}.json"
        ),
    )
    if (
        type(sole_sender) is not dict
        or set(sole_sender)
        != {
            "table_name",
            "partition_key",
            "sort_key",
            "authority_identity_sha256",
            "source_closure_identity_sha256",
        }
        or sole_sender["table_name"] != "keep-glm52-h1g-ledger-v1"
        or sole_sender["partition_key"] != RUN_ID
        or sole_sender["sort_key"]
        != (
            f"ACTIVATION#{activation_id}#"
            "TASK10_SOLE_SENDER_AUTHORITY#00000001"
        )
    ):
        raise CampaignRunnerError(
            "pre-send sole-sender authority drifted"
        )
    _sha(
        sole_sender["authority_identity_sha256"],
        "sole-sender authority",
    )
    _sha(
        sole_sender["source_closure_identity_sha256"],
        "sole-sender source closure",
    )
    if (
        type(spend) is not dict
        or set(spend)
        != {
            "remaining_gpu_seconds",
            "remaining_gpu_usd",
            "gpu_reserve_seconds",
            "gpu_reserve_usd",
            "root_volume_tail_usd_max",
            "ledger_version_id",
            "ledger_body_sha256",
        }
        or type(spend["remaining_gpu_seconds"]) is not int
        or spend["remaining_gpu_seconds"] <= 3600
        or not re.fullmatch(r"[0-9]+[.][0-9]{2}", spend["remaining_gpu_usd"])
        or spend["gpu_reserve_seconds"] != 900
        or spend["gpu_reserve_usd"] != "13.76"
        or spend["root_volume_tail_usd_max"] != "0.01"
        or type(spend["ledger_version_id"]) is not str
        or not spend["ledger_version_id"].strip()
    ):
        raise CampaignRunnerError("pre-send launch authority spend drifted")
    _sha(spend["ledger_body_sha256"], "pre-send spend ledger")
    if (
        type(action) is not dict
        or set(action)
        != {
            "action_kind",
            "action_count",
            "action_version_id",
            "action_body_sha256",
        }
        or action["action_kind"] != "PRODUCTION_SUBMISSION"
        or action["action_count"] != 1
        or type(action["action_version_id"]) is not str
        or not action["action_version_id"].strip()
    ):
        raise CampaignRunnerError("pre-send launch authority action drifted")
    _sha(action["action_body_sha256"], "pre-send production action")
    if (
        type(liability) is not dict
        or set(liability)
        != {
            "action_kind",
            "attempt",
            "shared_attempt_counter",
            "action_version_id",
            "action_body_sha256",
        }
        or liability["action_kind"] != "SAME_TOKEN_COMPLETE"
        or liability["attempt"] != attempt["attempt"]
        or liability["shared_attempt_counter"] != attempt["attempt"]
        or type(liability["action_version_id"]) is not str
        or not liability["action_version_id"].strip()
    ):
        raise CampaignRunnerError("pre-send launch authority liability drifted")
    _sha(liability["action_body_sha256"], "pre-send liability action")
    return dict(value)


def _validate_presend_sole_sender_authority(
    value: object,
    *,
    activation_id: object,
) -> dict[str, object]:
    if (
        type(value) is not dict
        or set(value)
        != {
            "table_name",
            "partition_key",
            "sort_key",
            "authority_identity_sha256",
            "source_closure_identity_sha256",
        }
        or value["table_name"] != "keep-glm52-h1g-ledger-v1"
        or value["partition_key"] != RUN_ID
        or value["sort_key"]
        != (
            f"ACTIVATION#{activation_id}#"
            "TASK10_SOLE_SENDER_AUTHORITY#00000001"
        )
    ):
        raise CampaignRunnerError(
            "pre-send sole-sender authority drifted"
        )
    _sha(
        value["authority_identity_sha256"],
        "sole-sender authority",
    )
    _sha(
        value["source_closure_identity_sha256"],
        "sole-sender source closure",
    )
    return dict(value)


def _read_h100_result(
    *,
    package: Mapping[str, object],
    services: object,
    journal: object,
) -> dict[str, object]:
    seed = _read_cache_seed_result(
        package=package,
        services=services,
        journal=journal,
    )
    plan = package.get("h100_qualification")
    if type(plan) is not dict:
        raise CampaignRunnerError("H100 qualification contract is missing")
    request = {
        **plan,
        "qualification_cache_prefix": seed["qualification_cache_prefix"],
        "qualification_cache_manifest_sha256": seed[
            "qualification_cache_manifest_sha256"
        ],
        "seed_ready": seed["seed_ready"],
        "teacher_ready": seed["teacher_ready"],
    }
    result = _read_committed_mutation(
        services,
        journal,
        stage="h100-qualification",
        operation_kind="h100-qualification",
        operation_id="h100-qualification:two-step-resume",
        request=request,
        validate_result=_validate_h100_result,
    )
    assert type(result) is dict
    return result


def _guarded_launch(
    *,
    package: Mapping[str, object],
    package_identity_sha256: str,
    services: object,
    journal: object,
) -> dict[str, object]:
    _require_stage(journal, "h100-qualification", package=package)
    plan = package.get("production_retry_plan")
    reviewed = package.get("reviewed_artifacts")
    if type(reviewed) is not list:
        raise CampaignRunnerError("campaign reviewed artifacts are missing")
    artifacts_by_kind = {
        str(row.get("artifact_kind")): row
        for row in reviewed
        if type(row) is dict
    }
    try:
        execution_custody = validate_reviewed_repository_execution_custody(
            plan.get("execution_custody")
            if type(plan) is dict
            else None,
            artifacts=artifacts_by_kind,
            route=plan.get("route") if type(plan) is dict else None,
        )
    except CampaignPackageError as error:
        raise CampaignRunnerError(
            "reviewed repository execution custody drifted"
        ) from error
    expected_zones = ["us-west-2" + letter for letter in "abcdef"]
    expected_attempts = [
        {
            "attempt": index,
            "availability_zone": zone,
            "on_capacity_failure": (
                "ROTATE_NEXT_AZ"
                if index < 6
                else "STOP_CAPACITY_EXHAUSTED"
            ),
        }
        for index, zone in enumerate(expected_zones, 1)
    ]
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
        "bucket": (
            "keep-glm52-models-246813579024-us-west-2"
        ),
        "key": (
            f"campaigns/{RUN_ID}/submissions/production/"
            "generations/00000001/workflow/LAUNCH_OUTCOME.json"
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
        type(plan) is not dict
        or plan.get("authorized") is not False
        or plan.get("execution_state") != "BLOCKED_UNTIL_ALL_GATES"
        or plan.get("instance_type") != "p5.48xlarge"
        or plan.get("capacity_type") != "ON_DEMAND"
        or plan.get("max_active_instances") != 1
        or plan.get("availability_zones") != expected_zones
        or plan.get("attempts") != expected_attempts
        or plan.get("stop_on_first_success") is not True
        or plan.get("automatic_spot_fallback") is not False
        or plan.get("workflow_reconciliation_contract")
        != expected_reconciliation_contract
        or plan.get("production_authority_contract")
        != {
            "minimum_remaining_gpu_seconds_exclusive": 3600,
            "action_kind": "PRODUCTION_SUBMISSION",
            "action_count": 1,
            "required_terminal_markers": [
                "CAMPAIGN_DRAINED.json",
                "TERMINAL_VERIFIED.json",
            ],
            "monitor_route": (
                "aws/glm52-gpu/scripts/"
                "sky_campaign_break_glass.sh status"
            ),
        }
    ):
        raise CampaignRunnerError("guarded six-AZ retry plan drifted")
    immutable_inputs = plan.get("immutable_inputs")
    if type(immutable_inputs) is not list or len(immutable_inputs) != 12:
        raise CampaignRunnerError("production immutable input set drifted")
    exact_immutable_inputs = [
        _exact_read_artifact(services, coordinate)
        for coordinate in immutable_inputs
    ]
    h100 = _read_h100_result(
        package=package,
        services=services,
        journal=journal,
    )
    seed = _read_cache_seed_result(
        package=package,
        services=services,
        journal=journal,
    )
    launch_inputs = {
        "immutable_inputs": exact_immutable_inputs,
        "qualification_cache_prefix": seed["qualification_cache_prefix"],
        "qualification_cache_manifest_sha256": seed[
            "qualification_cache_manifest_sha256"
        ],
        "seed_ready": seed["seed_ready"],
        "teacher_ready": seed["teacher_ready"],
        "source_ready": h100["source_ready"],
        "termination_requested": h100["termination_requested"],
        "h100_resume_ready": h100["h100_resume_ready"],
    }
    immutable_inputs_identity = canonical_sha256(launch_inputs)
    workflow_attempt = expected_attempts[0]
    authority_request = {
        **workflow_attempt,
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "profile": PROFILE,
        "run_id": RUN_ID,
        "activation_id": package.get("activation_id"),
        "instance_type": "p5.48xlarge",
        "capacity_type": "ON_DEMAND",
        "max_active_instances": 1,
        "immutable_inputs_identity_sha256": immutable_inputs_identity,
        "immutable_inputs": exact_immutable_inputs,
        "capacity_plan": {
            "availability_zones": expected_zones,
            "attempts": expected_attempts,
            "maximum_capacity_attempts": 6,
            "stop_on_first_worker_success": True,
            "automatic_spot_fallback": False,
            "capacity_block_allowed": False,
        },
    }
    operation_id = "launch:task10-workflow"
    existing = _journal_entries(journal, operation_id)
    if not existing or existing[-1]["state"] == "PREPARED":
        zero = _inspect(
            services,
            "p5-zero",
            {
                "account_id": ACCOUNT_ID,
                "region": REGION,
                "profile": PROFILE,
                "instance_type": "p5.48xlarge",
                "workflow_invocation": 1,
            },
        )
        if zero != {"active_p5_instance_ids": []}:
            raise CampaignRunnerError(
                "pre-workflow invariant found an active P5"
            )
    sole_sender = _mutation(
        services,
        journal,
        stage="launch",
        operation_kind="sole-sender-authority-materialize",
        operation_id="sole-sender-authority:generation-00000001",
        request=authority_request,
        validate_result=lambda value: (
            _validate_presend_sole_sender_authority(
                value,
                activation_id=package.get("activation_id"),
            )
        ),
    )
    assert type(sole_sender) is dict
    launch_authority_request = {
        **authority_request,
        "sole_sender_authority": sole_sender,
    }
    authority = _mutation(
        services,
        journal,
        stage="launch",
        operation_kind="launch-authority-materialize",
        operation_id="launch-authority:attempt-01",
        request=launch_authority_request,
        validate_result=lambda value: _validate_presend_launch_authority(
            value,
            attempt=workflow_attempt,
            activation_id=package.get("activation_id"),
            immutable_inputs_identity_sha256=immutable_inputs_identity,
            prior_same_token_identity_sha256=None,
        ),
    )
    assert type(authority) is dict
    _exact_read_marker(
        services,
        authority["authority_record"],
        expected_key=authority["authority_record"]["key"],
    )
    request = {
        **launch_authority_request,
        "route": plan.get("route"),
        "execution_custody": execution_custody,
        "automatic_spot_fallback": False,
        "production_authority_contract": plan[
            "production_authority_contract"
        ],
        "workflow_reconciliation_contract": (
            expected_reconciliation_contract
        ),
        **launch_inputs,
        "execution_authority": authority,
    }
    result = _mutation(
        services,
        journal,
        stage="launch",
        operation_kind="guarded-launch",
        operation_id=operation_id,
        request=request,
        validate_result=lambda value: _validate_launch_result(
            value,
            request=request,
        ),
    )
    assert type(result) is dict
    observed_output = _inspect(
        services,
        "immutable-marker",
        result["workflow_output"],
    )
    if observed_output != result["workflow_output"]:
        raise CampaignRunnerError(
            "workflow output exact VersionId readback drifted"
        )
    if not result["accepted"]:
        raise CampaignRunnerError(
            "guarded launch exhausted all six availability zones"
        )
    summary = {
        "accepted_execution_arn": result["execution_arn"],
        "accepted_instance_id": result["instance_id"],
        "accepted_availability_zone": result["availability_zone"],
        "workflow_invocation_count": 1,
        "capacity_outcome_count": len(result["capacity_outcomes"]),
        "capacity_type": "ON_DEMAND",
        "workflow_reconciliation": result["workflow_reconciliation"],
    }
    _record_stage(
        journal,
        stage="launch",
        package_identity_sha256=package_identity_sha256,
        reviewed_artifacts_identity_sha256=canonical_sha256(
            package["reviewed_artifacts"]
        ),
        summary=summary,
    )
    return _stage_result(
        stage="launch",
        package_identity_sha256=package_identity_sha256,
        values=summary,
    )


def _validate_terminal_proof(
    value: object,
    *,
    activation_id: object,
) -> dict[str, object]:
    fields = {
        "proof",
        "terminal_verified_value",
        "campaign_drained_value",
        "spend_ledger_raw",
    }
    if (
        type(value) is not dict
        or set(value) != fields
        or type(value["proof"]) is not dict
        or type(value["terminal_verified_value"]) is not dict
        or type(value["campaign_drained_value"]) is not dict
        or type(value["spend_ledger_raw"]) is not bytes
    ):
        raise CampaignRunnerError(
            "terminal proof exact-read envelope drifted"
        )
    try:
        proof = validate_task13_terminal_proof(
            value["proof"],
            terminal_verified_value=value["terminal_verified_value"],
            campaign_drained_value=value["campaign_drained_value"],
            spend_ledger_raw=value["spend_ledger_raw"],
        )
    except (TerminalEvidenceError, TypeError, ValueError) as error:
        raise CampaignRunnerError("terminal proof drifted") from error
    if proof["activation_id"] != activation_id:
        raise CampaignRunnerError("terminal proof activation drifted")
    return proof


def _terminal(
    *,
    package: Mapping[str, object],
    package_identity_sha256: str,
    services: object,
    journal: object,
) -> dict[str, object]:
    _require_stage(journal, "launch", package=package)
    request = {
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "profile": PROFILE,
        "run_id": RUN_ID,
        "activation_id": package.get("activation_id"),
        "required_terminal_markers": [
            "CAMPAIGN_DRAINED.json",
            "TERMINAL_VERIFIED.json",
        ],
        "required_sky_state": "SUCCEEDED",
        "required_billable_p5_instance_count": 0,
        "monitor_contract": package.get("monitor_contract"),
    }
    proof = _validate_terminal_proof(
        _inspect(services, "terminal-proof", request),
        activation_id=package.get("activation_id"),
    )
    _exact_read_marker(
        services,
        proof["campaign_drained"],
        expected_key=proof["campaign_drained"]["key"],
    )
    _exact_read_marker(
        services,
        proof["terminal_verified"],
        expected_key=proof["terminal_verified"]["key"],
    )
    settlement = proof["settlement"]
    summary = {
        "sky_state": proof["sky_state"],
        "billable_instance_count": 0,
        "campaign_drained": proof["campaign_drained"],
        "terminal_verified": proof["terminal_verified"],
        "gpu_spend_ledger": settlement["gpu_spend_ledger"],
        "total_gpu_seconds": settlement["total_gpu_seconds"],
        "total_gpu_cost_usd": settlement["total_gpu_cost_usd"],
        "remaining_gpu_seconds": settlement["remaining_gpu_seconds"],
        "remaining_gpu_usd": settlement["remaining_gpu_usd"],
        "retained_costs": settlement["retained_costs"],
    }
    _record_stage(
        journal,
        stage="terminal",
        package_identity_sha256=package_identity_sha256,
        reviewed_artifacts_identity_sha256=canonical_sha256(
            package["reviewed_artifacts"]
        ),
        summary=summary,
    )
    return _stage_result(
        stage="terminal",
        package_identity_sha256=package_identity_sha256,
        values=summary,
    )


def _monitor(
    *,
    package: Mapping[str, object],
    services: object,
    journal: object,
) -> dict[str, object]:
    _require_stage(journal, "launch", package=package)
    contract = package.get("monitor_contract")
    if (
        type(contract) is not dict
        or contract.get("read_only") is not True
        or contract.get("account_id") != ACCOUNT_ID
        or contract.get("region") != REGION
        or contract.get("profile") != PROFILE
        or contract.get("argv")
        != [
            "aws/glm52-gpu/scripts/sky_campaign_break_glass.sh",
            "status",
        ]
        or contract.get("environment")
        != {
            "AWS_PROFILE": PROFILE,
            "CAMPAIGN_DESCRIPTOR": contract.get("descriptor_path"),
            "SKY_BIN": (
                "/Users/jack.mazac/.local/share/keep/"
                "skypilot-0.13.0/bin/sky"
            ),
            "SKYPILOT_CONFIG": (
                "/Users/jack.mazac/.local/share/keep/"
                "skypilot-0.13.0/server-config.yaml"
            ),
        }
    ):
        raise CampaignRunnerError("monitor contract drifted")
    argv = contract.get("argv")
    if (
        type(argv) is not list
        or not argv
        or any(
            type(item) is not str
            or not item
            or "\x00" in item
            or "\n" in item
            or "\r" in item
            for item in argv
        )
        or argv[0].rsplit("/", 1)[-1] in _SHELL_EXECUTABLES
        or "-c" in argv[:3]
    ):
        raise CampaignRunnerError("monitor argv is not safe")
    observed = _inspect(services, "monitor", dict(contract))
    if (
        type(observed) is not dict
        or set(observed) != {"status", "instance_ids"}
        or observed["status"]
        not in {"STARTING", "RUNNING", "SUCCEEDED", "FAILED"}
        or type(observed["instance_ids"]) is not list
        or any(
            type(instance_id) is not str
            or _INSTANCE_ID.fullmatch(instance_id) is None
            for instance_id in observed["instance_ids"]
        )
    ):
        raise CampaignRunnerError("monitor result drifted")
    return dict(observed)


def run_campaign_stage(
    *,
    package: object,
    reviewed_artifacts: object,
    stage: object = "validate",
    services: object,
    journal: object,
    authority: object,
    finalization_capture: Optional[
        Callable[[FinalizationCapture], None]
    ] = None,
) -> dict[str, object]:
    """Run one explicit stage; validation performs no external operation."""

    exact_package, package_identity = _validate_package(
        package,
        reviewed_artifacts,
    )
    if type(stage) is not str or stage not in _STAGES:
        raise CampaignRunnerError("campaign runner stage is not closed")
    if finalization_capture is not None and (
        stage != "finalize" or not callable(finalization_capture)
    ):
        raise CampaignRunnerError(
            "finalization capture is valid only for finalize"
        )
    if (
        exact_package.get("package_phase") == "PREQUALIFICATION"
        and stage in {"launch", "terminal", "monitor"}
    ):
        raise CampaignRunnerError(
            "prequalification package cannot authorize production stages"
        )
    artifacts_identity = canonical_sha256(
        exact_package["reviewed_artifacts"]
    )
    retry = exact_package["production_retry_plan"]
    assert type(retry) is dict
    execution_custody = retry["execution_custody"]
    assert type(execution_custody) is dict
    execution_custody_identity = str(
        execution_custody["canonical_identity_sha256"]
    )
    if stage == "validate":
        body = {
            "status": "VALIDATED_NO_EXTERNAL_EXECUTION",
            "stage": "validate",
            "package_identity_sha256": package_identity,
            "reviewed_artifacts_identity_sha256": artifacts_identity,
        }
        return {
            **body,
            "canonical_identity_sha256": canonical_sha256(body),
        }
    exact_authority = _validate_authority(
        authority,
        stage=stage,
        package_identity_sha256=package_identity,
        reviewed_artifacts_identity_sha256=artifacts_identity,
        execution_custody_identity_sha256=(
            execution_custody_identity
        ),
        activation_id=str(exact_package["activation_id"]),
        services=services,
    )
    bind_authority = getattr(
        services,
        "bind_controller_authority",
        None,
    )
    if callable(bind_authority):
        bind_authority(exact_authority)
    journal_context = _journal_context(journal)
    dispatch = {
        "collect-first-five": _collect_first_five,
        "collect-remaining": _collect_remaining,
        "qualification-cache-seed": _qualification_cache_seed,
        "h100-qualification": _h100_qualification,
        "launch": _guarded_launch,
        "terminal": _terminal,
    }
    if stage == "monitor":
        return _monitor(
            package=exact_package,
            services=services,
            journal=_journal_for_scope(
                journal_context,
                "campaign",
            ),
        )
    if stage == "deploy-disabled":
        return _deploy_disabled(
            package=exact_package,
            package_identity_sha256=package_identity,
            services=services,
            journal=journal_context,
            execution_authority=exact_authority,
        )
    if stage == "finalize":
        return _finalize(
            package=exact_package,
            package_identity_sha256=package_identity,
            services=services,
            journal=_journal_for_scope(journal_context, "campaign"),
            finalization_capture=finalization_capture,
        )
    runner = dispatch[stage]
    return runner(
        package=exact_package,
        package_identity_sha256=package_identity,
        services=services,
        journal=_journal_for_scope(journal_context, "campaign"),
    )


__all__ = [
    "ACCOUNT_ID",
    "REGION",
    "PROFILE",
    "RUN_ID",
    "CampaignRunnerError",
    "CampaignJournalStores",
    "ControllerExecutionAuthority",
    "ControllerOperationCapability",
    "FileJournalStore",
    "FinalizationCapture",
    "SubprocessCoordinator",
    "run_campaign_stage",
]
