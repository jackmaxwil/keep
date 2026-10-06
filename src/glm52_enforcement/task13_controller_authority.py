"""Issue one short-lived, stage-scoped Task 13 controller authority."""

from __future__ import annotations

import base64
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import tempfile
import time
from typing import Callable, Optional

from .canonical import canonical_json_bytes, canonical_sha256
from .task13_campaign_package import (
    CAMPAIGN_OWNER_APPROVAL_SHA256,
    canonical_campaign_package_bytes,
)
from .task13_campaign_runner import (
    ACCOUNT_ID,
    PROFILE,
    REGION,
    RUN_ID,
    ControllerExecutionAuthority,
    ControllerOperationCapability,
)


SOURCE_AUTHORITY_EMAIL = "operator@example.com"
AUTHORITY_THREAT_MODEL = (
    "Protects against accidental or direct protocol bypass and authority "
    "object tampering; malicious code running as the signing-key owner "
    "is outside this boundary."
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
FULL_RUN_WORK_POINTER = Path("/private/tmp/glm52-full-run-current")
OPENSSH_EXECUTABLE = Path("/usr/bin/ssh-keygen")
OPENSSH_EXECUTABLE_SHA256 = (
    "3ebf8d762494bb9e23be1955a8e136a6f3853512ea1f693fab96e0986bb9304f"
)
_AUTHORITY_RECORD_TYPE = "glm52_task13_controller_execution_authority_v2"
_CAPABILITY_RECORD_TYPE = "glm52_task13_operation_capability_v2"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_STAGES = {
    "deploy-disabled",
    "collect-first-five",
    "collect-remaining",
    "finalize",
    "qualification-cache-seed",
    "h100-qualification",
    "launch",
    "terminal",
    "monitor",
}
_PRODUCTION_STAGES = {"launch", "terminal", "monitor"}
_APPROVAL_SNIPPETS = (
    "Principal: Jack Mazac (`operator@example.com`)",
    "Account: `246813579024` (`keep-gpu`)",
    "Region: `us-west-2`",
    "## Production-support-plane approval",
    "## Residual launch-liability approval",
    "## Execution authorization",
)


class ControllerAuthorityIssuerError(ValueError):
    """The issuer input did not prove one exact authority."""


def _validate_openssh_executable() -> None:
    try:
        info = OPENSSH_EXECUTABLE.lstat()
        raw = OPENSSH_EXECUTABLE.read_bytes()
    except OSError as error:
        raise ControllerAuthorityIssuerError(
            "pinned OpenSSH verifier is unavailable"
        ) from error
    if (
        OPENSSH_EXECUTABLE.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or info.st_mode & 0o111 == 0
        or hashlib.sha256(raw).hexdigest() != OPENSSH_EXECUTABLE_SHA256
    ):
        raise ControllerAuthorityIssuerError(
            "pinned OpenSSH verifier identity drifted"
        )


def _public_key_identity(public_key: str) -> tuple[str, str]:
    parts = public_key.strip().split()
    if len(parts) not in {2, 3} or parts[0] != "ssh-ed25519":
        raise ControllerAuthorityIssuerError(
            "controller signing public key is not exact Ed25519"
        )
    try:
        blob = base64.b64decode(parts[1], validate=True)
    except ValueError as error:
        raise ControllerAuthorityIssuerError(
            "controller signing public key is malformed"
        ) from error
    fingerprint = (
        "SHA256:"
        + base64.b64encode(hashlib.sha256(blob).digest())
        .decode("ascii")
        .rstrip("=")
    )
    return " ".join(parts[:2]), fingerprint


_PINNED_PUBLIC_CORE, _PINNED_PUBLIC_FINGERPRINT = _public_key_identity(
    PINNED_SIGNER_PUBLIC_KEY
)
if _PINNED_PUBLIC_FINGERPRINT != PINNED_SIGNER_FINGERPRINT:
    raise RuntimeError("pinned controller signing key fingerprint drifted")


def _sterile_env() -> dict[str, str]:
    return {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }


class ControllerAuthoritySigner:
    """Use the one campaign Ed25519 private key without exposing its bytes."""

    def __init__(
        self,
        *,
        private_key_path: object,
        full_run_work: object,
    ) -> None:
        if (
            not isinstance(private_key_path, (str, os.PathLike))
            or not isinstance(full_run_work, (str, os.PathLike))
        ):
            raise ControllerAuthorityIssuerError(
                "controller signing key custody paths are not explicit"
            )
        path = Path(private_key_path)
        work = Path(full_run_work)
        try:
            work_info = work.lstat()
            info = path.lstat()
            public_info = path.with_suffix(".pub").lstat()
        except OSError as error:
            raise ControllerAuthorityIssuerError(
                "controller signing private key is unavailable"
            ) from error
        if (
            not work.is_absolute()
            or work.is_symlink()
            or work.resolve(strict=True) != work
            or not stat.S_ISDIR(work_info.st_mode)
            or stat.S_IMODE(work_info.st_mode) != 0o700
            or work_info.st_uid != os.getuid()
            or not path.is_absolute()
            or path.is_symlink()
            or path.resolve(strict=True) != path
            or path.parent != work
            or path.name != "controller-authority-ed25519"
            or not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or path.with_suffix(".pub").is_symlink()
            or not stat.S_ISREG(public_info.st_mode)
            or stat.S_IMODE(public_info.st_mode) != 0o600
            or public_info.st_uid != os.getuid()
            or public_info.st_nlink != 1
        ):
            raise ControllerAuthorityIssuerError(
                "controller signing private key custody drifted"
            )
        _validate_openssh_executable()
        derived = subprocess.run(
            [str(OPENSSH_EXECUTABLE), "-y", "-f", str(path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=_sterile_env(),
            timeout=10,
        )
        if derived.returncode != 0:
            raise ControllerAuthorityIssuerError(
                "controller signing private key could not derive public key"
            )
        try:
            public_text = derived.stdout.decode("ascii")
        except UnicodeDecodeError as error:
            raise ControllerAuthorityIssuerError(
                "controller signing public key is malformed"
            ) from error
        public_core, fingerprint = _public_key_identity(public_text)
        try:
            public_sibling = path.with_suffix(".pub").read_text(
                encoding="ascii"
            )
        except (OSError, UnicodeError) as error:
            raise ControllerAuthorityIssuerError(
                "controller signing public key sibling is unreadable"
            ) from error
        if (
            public_core != _PINNED_PUBLIC_CORE
            or fingerprint != PINNED_SIGNER_FINGERPRINT
            or public_sibling != PINNED_SIGNER_PUBLIC_KEY + "\n"
        ):
            raise ControllerAuthorityIssuerError(
                "controller signing private key public key drifted"
            )
        self._private_key_path = path

    def sign(self, payload: object) -> str:
        if type(payload) is not bytes or not payload:
            raise ControllerAuthorityIssuerError(
                "controller authority signed payload is absent"
            )
        with tempfile.TemporaryDirectory(
            prefix="glm52-task13-authority-sign-"
        ) as directory:
            message_path = Path(directory) / "authority.json"
            message_path.write_bytes(payload)
            message_path.chmod(0o600)
            completed = subprocess.run(
                [
                    str(OPENSSH_EXECUTABLE),
                    "-Y",
                    "sign",
                    "-f",
                    str(self._private_key_path),
                    "-n",
                    OPENSSH_SIGNING_NAMESPACE,
                    "-O",
                    "hashalg=sha512",
                    str(message_path),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                env=_sterile_env(),
                timeout=10,
            )
            signature_path = Path(str(message_path) + ".sig")
            if (
                completed.returncode != 0
                or not signature_path.is_file()
                or signature_path.is_symlink()
            ):
                raise ControllerAuthorityIssuerError(
                    "controller authority signing failed"
                )
            signature = signature_path.read_bytes()
        if (
            not signature.startswith(b"-----BEGIN SSH SIGNATURE-----\n")
            or not signature.endswith(b"-----END SSH SIGNATURE-----\n")
            or len(signature) > 4096
        ):
            raise ControllerAuthorityIssuerError(
                "controller authority signature encoding drifted"
            )
        return base64.b64encode(signature).decode("ascii")


def _verify_signature(payload: bytes, signature_base64: object) -> None:
    if type(signature_base64) is not str or len(signature_base64) > 8192:
        raise ControllerAuthorityIssuerError(
            "controller authority signature is absent"
        )
    try:
        signature = base64.b64decode(signature_base64, validate=True)
    except ValueError as error:
        raise ControllerAuthorityIssuerError(
            "controller authority signature is malformed"
        ) from error
    if (
        not signature.startswith(b"-----BEGIN SSH SIGNATURE-----\n")
        or not signature.endswith(b"-----END SSH SIGNATURE-----\n")
        or len(signature) > 4096
    ):
        raise ControllerAuthorityIssuerError(
            "controller authority signature is malformed"
        )
    _validate_openssh_executable()
    with tempfile.TemporaryDirectory(
        prefix="glm52-task13-authority-verify-"
    ) as directory:
        signer_path = Path(directory) / "allowed-signers"
        signature_path = Path(directory) / "authority.sig"
        signer_path.write_text(
            SOURCE_AUTHORITY_EMAIL + " " + PINNED_SIGNER_PUBLIC_KEY + "\n",
            encoding="ascii",
        )
        signature_path.write_bytes(signature)
        signer_path.chmod(0o400)
        signature_path.chmod(0o400)
        completed = subprocess.run(
            [
                str(OPENSSH_EXECUTABLE),
                "-Y",
                "verify",
                "-f",
                str(signer_path),
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
            check=False,
            env=_sterile_env(),
            timeout=10,
        )
    if completed.returncode != 0:
        raise ControllerAuthorityIssuerError(
            "controller authority signature verification failed"
        )


def _sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ControllerAuthorityIssuerError(
            label + " must be lowercase SHA-256"
        )
    return value


def validate_owner_approval(
    raw: object,
    *,
    expected_sha256: object,
) -> str:
    """Validate the pinned committed approval bytes and required decisions."""

    if type(raw) is not bytes:
        raise ControllerAuthorityIssuerError(
            "owner approval must be exact bytes"
        )
    expected = _sha256(expected_sha256, "expected owner approval")
    if expected != CAMPAIGN_OWNER_APPROVAL_SHA256:
        raise ControllerAuthorityIssuerError(
            "owner approval identity drifted"
        )
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected:
        raise ControllerAuthorityIssuerError(
            "owner approval identity drifted"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ControllerAuthorityIssuerError(
            "owner approval is not UTF-8"
        ) from error
    if any(snippet not in text for snippet in _APPROVAL_SNIPPETS):
        raise ControllerAuthorityIssuerError(
            "owner approval decision contract drifted"
        )
    if text.count("Status: **APPROVED**") != 2:
        raise ControllerAuthorityIssuerError(
            "owner approval statuses are not exact"
        )
    return observed


def issue_controller_execution_authority(
    *,
    package: object,
    reviewed_artifacts: object,
    stage: object,
    ttl_seconds: object,
    coordinator_executable_sha256: object,
    owner_approval_sha256: object,
    now: object = None,
    random_bytes: Callable[[int], bytes] = secrets.token_bytes,
    signer: Optional[ControllerAuthoritySigner] = None,
) -> ControllerExecutionAuthority:
    """Build an Ed25519-signed authority for one package, executable, and stage."""

    if type(package) is not dict:
        raise ControllerAuthorityIssuerError(
            "campaign package must be an exact object"
        )
    try:
        canonical_campaign_package_bytes(package)
    except (TypeError, ValueError) as error:
        raise ControllerAuthorityIssuerError(
            "campaign package identity drifted"
        ) from error
    if (
        type(reviewed_artifacts) is not list
        or package.get("reviewed_artifacts") != reviewed_artifacts
    ):
        raise ControllerAuthorityIssuerError(
            "reviewed artifacts do not match the campaign package"
        )
    if type(stage) is not str or stage not in _STAGES:
        raise ControllerAuthorityIssuerError(
            "authority stage is not closed"
        )
    if (
        package.get("package_phase") == "PREQUALIFICATION"
        and stage in _PRODUCTION_STAGES
    ):
        raise ControllerAuthorityIssuerError(
            "prequalification package cannot receive production authority"
        )
    if type(ttl_seconds) is not int or ttl_seconds not in range(60, 901):
        raise ControllerAuthorityIssuerError(
            "authority TTL must be 60 through 900 seconds"
        )
    issued_at = int(time.time()) if now is None else now
    if type(issued_at) is not int:
        raise ControllerAuthorityIssuerError(
            "authority issuance time is invalid"
        )
    coordinator_sha = _sha256(
        coordinator_executable_sha256,
        "coordinator executable identity",
    )
    approval_sha = _sha256(
        owner_approval_sha256,
        "owner approval identity",
    )
    if approval_sha != CAMPAIGN_OWNER_APPROVAL_SHA256:
        raise ControllerAuthorityIssuerError(
            "owner approval identity is not the pinned campaign decision"
        )
    retry = package.get("production_retry_plan")
    custody = (
        retry.get("execution_custody")
        if type(retry) is dict
        else None
    )
    custody_identity = (
        custody.get("canonical_identity_sha256")
        if type(custody) is dict
        else None
    )
    _sha256(custody_identity, "execution custody identity")
    nonce = random_bytes(32)
    if type(nonce) is not bytes or len(nonce) != 32:
        raise ControllerAuthorityIssuerError(
            "authority nonce source is invalid"
        )
    body = {
        "schema_version": 2,
        "record_type": _AUTHORITY_RECORD_TYPE,
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "profile": PROFILE,
        "run_id": RUN_ID,
        "activation_id": package["activation_id"],
        "stage": stage,
        "package_identity_sha256": package[
            "canonical_identity_sha256"
        ],
        "reviewed_artifacts_identity_sha256": canonical_sha256(
            reviewed_artifacts
        ),
        "controller_nonce_sha256": hashlib.sha256(nonce).hexdigest(),
        "source_authority_email": SOURCE_AUTHORITY_EMAIL,
        "owner_approval_sha256": approval_sha,
        "execution_custody_identity_sha256": custody_identity,
        "production_support_plane_approved": True,
        "residual_launch_liability_approved": True,
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
        "production_launch_approved": stage in _PRODUCTION_STAGES,
        "coordinator_executable_sha256": coordinator_sha,
        "issued_at_epoch_seconds": issued_at,
        "expires_at_epoch_seconds": issued_at + ttl_seconds,
        "signer_identity": SOURCE_AUTHORITY_EMAIL,
        "signing_namespace": OPENSSH_SIGNING_NAMESPACE,
        "signer_public_key_fingerprint": PINNED_SIGNER_FINGERPRINT,
    }
    signed_body = {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }
    exact_signer = signer
    if type(exact_signer) is not ControllerAuthoritySigner:
        raise ControllerAuthorityIssuerError(
            "controller authority signer boundary is foreign"
        )
    authority = ControllerExecutionAuthority(
        **signed_body,
        sshsig_base64=exact_signer.sign(
            canonical_json_bytes(signed_body)
        ),
    )
    verify_controller_execution_authority(
        authority,
        now=issued_at,
    )
    return authority


def verify_controller_execution_authority(
    value: object,
    *,
    now: object = None,
) -> ControllerExecutionAuthority:
    """Verify the pinned campaign signer before trusting any authority field."""

    if type(value) is not ControllerExecutionAuthority:
        raise ControllerAuthorityIssuerError(
            "controller authority type is not exact"
        )
    signed = asdict(value)
    signature = signed.pop("sshsig_base64")
    identity = signed.pop("canonical_identity_sha256")
    observed = int(time.time()) if now is None else now
    if (
        value.schema_version != 2
        or value.record_type != _AUTHORITY_RECORD_TYPE
    ):
        raise ControllerAuthorityIssuerError(
            "legacy or foreign controller authority schema is forbidden"
        )
    if (
        value.signer_identity != SOURCE_AUTHORITY_EMAIL
        or value.signing_namespace != OPENSSH_SIGNING_NAMESPACE
        or value.signer_public_key_fingerprint
        != PINNED_SIGNER_FINGERPRINT
    ):
        raise ControllerAuthorityIssuerError(
            "controller authority signer identity drifted"
        )
    if type(observed) is not int:
        raise ControllerAuthorityIssuerError(
            "controller authority verification time is invalid"
        )
    if (
        type(value.issued_at_epoch_seconds) is not int
        or type(value.expires_at_epoch_seconds) is not int
        or value.issued_at_epoch_seconds > observed
        or value.expires_at_epoch_seconds <= observed
        or value.expires_at_epoch_seconds - value.issued_at_epoch_seconds
        not in range(60, 901)
    ):
        raise ControllerAuthorityIssuerError(
            "controller authority TTL is not currently valid"
        )
    if identity != canonical_sha256(signed):
        raise ControllerAuthorityIssuerError(
            "controller authority identity drifted"
        )
    _verify_signature(
        canonical_json_bytes(
            {**signed, "canonical_identity_sha256": identity}
        ),
        signature,
    )
    return value


def issue_controller_operation_capability(
    *,
    authority: object,
    action: object,
    operation_kind: object,
    operation_id: object,
    request_identity_sha256: object,
    journal_store_path: object,
    journal_store_identity_sha256: object,
    journal_record_identities: object,
    journal_state: object,
    prior_execute_capability_identity_sha256: object,
    ttl_seconds: object,
    now: object = None,
    random_bytes: Callable[[int], bytes] = secrets.token_bytes,
    signer: Optional[ControllerAuthoritySigner] = None,
) -> ControllerOperationCapability:
    """Sign one action/kind/id/request/journal tuple for coordinator admission."""

    issued_at = int(time.time()) if now is None else now
    exact_authority = verify_controller_execution_authority(
        authority,
        now=issued_at,
    )
    if type(action) is not str or action not in {
        "execute",
        "inspect",
        "reconcile",
    }:
        raise ControllerAuthorityIssuerError(
            "operation capability action is not closed"
        )
    for value, label in (
        (operation_kind, "operation kind"),
        (operation_id, "operation ID"),
    ):
        if (
            type(value) is not str
            or not value
            or len(value) > 256
            or any(character in value for character in "\x00\r\n")
        ):
            raise ControllerAuthorityIssuerError(
                "operation capability " + label + " is not exact"
            )
    request_sha = _sha256(
        request_identity_sha256,
        "operation request identity",
    )
    if (
        type(journal_store_path) is not str
        or not journal_store_path
        or len(journal_store_path) > 4096
        or "\x00" in journal_store_path
    ):
        raise ControllerAuthorityIssuerError(
            "journal store path is not exact"
        )
    store_sha = _sha256(
        journal_store_identity_sha256,
        "journal store identity",
    )
    if (
        type(journal_record_identities) is not list
        or len(journal_record_identities) not in range(1, 5)
    ):
        raise ControllerAuthorityIssuerError(
            "operation journal record identities are not exact"
        )
    journal_ids = [
        _sha256(value, "journal record identity")
        for value in journal_record_identities
    ]
    if (
        type(journal_state) is not str
        or journal_state
        not in {"PREPARED", "POSSIBLY_SENT", "COMMITTED", "READ_ONLY"}
        or (action == "execute" and journal_state != "POSSIBLY_SENT")
        or (
            action == "reconcile"
            and journal_state not in {"POSSIBLY_SENT", "COMMITTED"}
        )
        or (action == "inspect" and journal_state != "READ_ONLY")
    ):
        raise ControllerAuthorityIssuerError(
            "operation capability journal state drifted"
        )
    if action == "reconcile":
        prior_execute = _sha256(
            prior_execute_capability_identity_sha256,
            "prior execute capability identity",
        )
    elif prior_execute_capability_identity_sha256 is not None:
        raise ControllerAuthorityIssuerError(
            "non-reconcile capability invented prior execute custody"
        )
    else:
        prior_execute = None
    if type(ttl_seconds) is not int or ttl_seconds not in range(60, 901):
        raise ControllerAuthorityIssuerError(
            "operation capability TTL must be 60 through 900 seconds"
        )
    if issued_at + ttl_seconds > exact_authority.expires_at_epoch_seconds:
        raise ControllerAuthorityIssuerError(
            "operation capability exceeds controller authority TTL"
        )
    nonce = random_bytes(32)
    if type(nonce) is not bytes or len(nonce) != 32:
        raise ControllerAuthorityIssuerError(
            "operation capability nonce source is invalid"
        )
    body = {
        "schema_version": 2,
        "record_type": _CAPABILITY_RECORD_TYPE,
        "account_id": exact_authority.account_id,
        "region": exact_authority.region,
        "profile": exact_authority.profile,
        "run_id": exact_authority.run_id,
        "activation_id": exact_authority.activation_id,
        "stage": exact_authority.stage,
        "controller_authority_identity_sha256": (
            exact_authority.canonical_identity_sha256
        ),
        "package_identity_sha256": (
            exact_authority.package_identity_sha256
        ),
        "reviewed_artifacts_identity_sha256": (
            exact_authority.reviewed_artifacts_identity_sha256
        ),
        "execution_custody_identity_sha256": (
            exact_authority.execution_custody_identity_sha256
        ),
        "coordinator_executable_sha256": (
            exact_authority.coordinator_executable_sha256
        ),
        "owner_approval_sha256": exact_authority.owner_approval_sha256,
        "action": action,
        "operation_kind": operation_kind,
        "operation_id": operation_id,
        "request_identity_sha256": request_sha,
        "journal_store_path": journal_store_path,
        "journal_store_identity_sha256": store_sha,
        "journal_record_identities": journal_ids,
        "journal_chain_head_sha256": canonical_sha256(journal_ids),
        "journal_state": journal_state,
        "prior_execute_capability_identity_sha256": prior_execute,
        "capability_nonce_sha256": hashlib.sha256(nonce).hexdigest(),
        "issued_at_epoch_seconds": issued_at,
        "expires_at_epoch_seconds": issued_at + ttl_seconds,
        "signer_identity": SOURCE_AUTHORITY_EMAIL,
        "signing_namespace": OPENSSH_SIGNING_NAMESPACE,
        "signer_public_key_fingerprint": PINNED_SIGNER_FINGERPRINT,
    }
    signed_body = {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }
    exact_signer = signer
    if type(exact_signer) is not ControllerAuthoritySigner:
        raise ControllerAuthorityIssuerError(
            "controller capability signer boundary is foreign"
        )
    capability = ControllerOperationCapability(
        **signed_body,
        sshsig_base64=exact_signer.sign(
            canonical_json_bytes(signed_body)
        ),
    )
    verify_controller_operation_capability(
        capability,
        authority=exact_authority,
        now=issued_at,
    )
    return capability


def verify_controller_operation_capability(
    value: object,
    *,
    authority: object,
    now: object = None,
) -> ControllerOperationCapability:
    """Verify one v2 capability and its ancestry to the signed stage grant."""

    observed = int(time.time()) if now is None else now
    exact_authority = verify_controller_execution_authority(
        authority,
        now=observed,
    )
    if type(value) is not ControllerOperationCapability:
        raise ControllerAuthorityIssuerError(
            "controller operation capability type is not exact"
        )
    signed = asdict(value)
    signature = signed.pop("sshsig_base64")
    identity = signed.pop("canonical_identity_sha256")
    if (
        value.schema_version != 2
        or value.record_type != _CAPABILITY_RECORD_TYPE
    ):
        raise ControllerAuthorityIssuerError(
            "legacy or foreign operation capability schema is forbidden"
        )
    ancestry = {
        "account_id": exact_authority.account_id,
        "region": exact_authority.region,
        "profile": exact_authority.profile,
        "run_id": exact_authority.run_id,
        "activation_id": exact_authority.activation_id,
        "stage": exact_authority.stage,
        "controller_authority_identity_sha256": (
            exact_authority.canonical_identity_sha256
        ),
        "package_identity_sha256": (
            exact_authority.package_identity_sha256
        ),
        "reviewed_artifacts_identity_sha256": (
            exact_authority.reviewed_artifacts_identity_sha256
        ),
        "execution_custody_identity_sha256": (
            exact_authority.execution_custody_identity_sha256
        ),
        "coordinator_executable_sha256": (
            exact_authority.coordinator_executable_sha256
        ),
        "owner_approval_sha256": exact_authority.owner_approval_sha256,
        "signer_identity": SOURCE_AUTHORITY_EMAIL,
        "signing_namespace": OPENSSH_SIGNING_NAMESPACE,
        "signer_public_key_fingerprint": PINNED_SIGNER_FINGERPRINT,
    }
    if any(getattr(value, field) != expected for field, expected in ancestry.items()):
        raise ControllerAuthorityIssuerError(
            "operation capability authority ancestry drifted"
        )
    if (
        type(observed) is not int
        or type(value.issued_at_epoch_seconds) is not int
        or type(value.expires_at_epoch_seconds) is not int
        or value.issued_at_epoch_seconds > observed
        or value.expires_at_epoch_seconds <= observed
        or value.expires_at_epoch_seconds - value.issued_at_epoch_seconds
        not in range(60, 901)
        or value.expires_at_epoch_seconds
        > exact_authority.expires_at_epoch_seconds
    ):
        raise ControllerAuthorityIssuerError(
            "operation capability TTL is not currently valid"
        )
    if (
        value.journal_chain_head_sha256
        != canonical_sha256(value.journal_record_identities)
        or identity != canonical_sha256(signed)
    ):
        raise ControllerAuthorityIssuerError(
            "operation capability identity or journal chain drifted"
        )
    _verify_signature(
        canonical_json_bytes(
            {**signed, "canonical_identity_sha256": identity}
        ),
        signature,
    )
    return value


__all__ = [
    "AUTHORITY_THREAT_MODEL",
    "ControllerAuthorityIssuerError",
    "ControllerAuthoritySigner",
    "OPENSSH_EXECUTABLE",
    "OPENSSH_EXECUTABLE_SHA256",
    "OPENSSH_SIGNING_NAMESPACE",
    "PINNED_SIGNER_FINGERPRINT",
    "PINNED_SIGNER_PUBLIC_KEY",
    "FULL_RUN_WORK_POINTER",
    "SOURCE_AUTHORITY_EMAIL",
    "issue_controller_execution_authority",
    "issue_controller_operation_capability",
    "validate_owner_approval",
    "verify_controller_execution_authority",
    "verify_controller_operation_capability",
]
