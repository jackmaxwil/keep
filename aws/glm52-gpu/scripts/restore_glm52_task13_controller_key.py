#!/usr/bin/env python3
"""Materialize the pinned Task 13 controller signing key from escrow.

The signer requires a real regular file at an exact path with exact
permissions (see `ControllerAuthoritySigner.__init__`), so the durable master
lives in AWS Secrets Manager and is materialized per run. Nothing here ever
prints, logs, or passes key bytes through argv.

Account and region are pinned. A wrong account is fatal, never a fallback.
"""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from glm52_enforcement.task13_controller_authority import (  # noqa: E402
    PINNED_SIGNER_FINGERPRINT,
    PINNED_SIGNER_PUBLIC_KEY,
    ControllerAuthoritySigner,
    _public_key_identity,
)

ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
SECRET_ID = "keep/glm52/task13/controller-authority-ed25519-v3"
KEY_NAME = "controller-authority-ed25519"
OPENSSH_EXECUTABLE = Path("/usr/bin/ssh-keygen")


class RestoreError(RuntimeError):
    """The escrow restore could not be completed exactly."""


def _sterile_env() -> dict[str, str]:
    return {"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"}


def _require_pinned_account(session) -> None:
    identity = session.client("sts", region_name=REGION).get_caller_identity()
    if identity.get("Account") != ACCOUNT_ID:
        raise RestoreError(
            "refusing to restore outside the pinned account: observed "
            f"{identity.get('Account')!r}, required {ACCOUNT_ID!r}"
        )


def _fetch_private_key(session) -> bytes:
    client = session.client("secretsmanager", region_name=REGION)
    response = client.get_secret_value(SecretId=SECRET_ID)
    material = response.get("SecretBinary")
    if material is None:
        raise RestoreError("escrowed controller key is not binary material")
    if type(material) is not bytes or not material:
        raise RestoreError("escrowed controller key material is absent")
    return material


def _create_exclusive_0600(target: Path, payload: bytes) -> None:
    """Create target exclusively at 0600, or fail.

    Creation itself is the exclusivity guarantee: `O_EXCL` means a concurrent
    process cannot have its file replaced by ours, and `O_NOFOLLOW` means a
    planted symlink is never followed. Check-then-replace would leave a race
    between the existence check and the rename, so custody files are never
    written that way.
    """

    handle = os.open(
        str(target),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def _derive_public_key(private_key_path: Path) -> str:
    derived = subprocess.run(
        [str(OPENSSH_EXECUTABLE), "-y", "-f", str(private_key_path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=_sterile_env(),
        timeout=10,
    )
    if derived.returncode != 0:
        raise RestoreError("restored controller key could not derive a public key")
    return derived.stdout.decode("ascii")


def _require_0700_work(work: Path) -> None:
    """Validate the caller's RAW path, before any resolution.

    Resolving first would silently erase the evidence that the caller handed us
    a symlink, so the non-symlink condition is checked on the path as given.
    """

    if not work.is_absolute():
        raise RestoreError(f"work directory is not absolute: {work}")
    info = work.lstat()
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o700
        or info.st_uid != os.getuid()
        or work.resolve(strict=True) != work
    ):
        raise RestoreError(
            f"work directory is not an exact private 0700 directory: {work}"
        )


def restore(full_run_work: Path, *, session) -> Path:
    """Materialize the pinned key into full_run_work and prove custody."""

    work = Path(full_run_work)
    _require_0700_work(work)
    _require_pinned_account(session)

    private_key_path = work / KEY_NAME
    public_key_path = private_key_path.with_suffix(".pub")

    # Never mutate an already-materialized run directory. Failing closed here
    # also guarantees the cleanup below can only ever remove material THIS
    # call created, so a re-restore can never destroy valid custody.
    for existing in (private_key_path, public_key_path):
        if existing.exists() or existing.is_symlink():
            raise RestoreError(
                f"refusing to overwrite existing custody material: {existing}; "
                "run with --wipe first if this directory is genuinely stale"
            )

    # One cleanup boundary: any failure after a half exists must leave no
    # secret material behind. Only halves THIS call actually created are
    # removed -- if `_create_exclusive_0600` lost an O_EXCL race, the winning
    # file belongs to another process and must never be unlinked here.
    private_created = False
    public_created = False
    try:
        _create_exclusive_0600(private_key_path, _fetch_private_key(session))
        private_created = True

        derived_core, derived_fingerprint = _public_key_identity(
            _derive_public_key(private_key_path)
        )
        pinned_core, _ = _public_key_identity(PINNED_SIGNER_PUBLIC_KEY)
        if (
            derived_core != pinned_core
            or derived_fingerprint != PINNED_SIGNER_FINGERPRINT
        ):
            raise RestoreError(
                "escrowed controller key does not match the pinned public identity"
            )

        _create_exclusive_0600(
            public_key_path,
            (PINNED_SIGNER_PUBLIC_KEY + "\n").encode("ascii"),
        )
        public_created = True

        # Constructing the signer is the custody proof: it re-checks every
        # permission, link-count, and identity condition the issuer relies on.
        ControllerAuthoritySigner(
            private_key_path=private_key_path, full_run_work=work
        )
    except BaseException:
        if public_created:
            public_key_path.unlink(missing_ok=True)
        if private_created:
            private_key_path.unlink(missing_ok=True)
        raise
    return private_key_path


def wipe(full_run_work: Path) -> None:
    """Remove materialized halves, only from an exact private 0700 directory."""

    work = Path(full_run_work)
    _require_0700_work(work)
    for name in (KEY_NAME, KEY_NAME + ".pub"):
        (work / name).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-run-work",
        required=True,
        help="existing absolute 0700 directory to materialize the key into",
    )
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="remove a previously materialized key and public half, then exit",
    )
    arguments = parser.parse_args(argv)
    work = Path(arguments.full_run_work)

    if arguments.wipe:
        wipe(work)
        print(f"wiped materialized controller key under {work}")
        return 0

    import boto3

    path = restore(work, session=boto3.Session())
    print(f"restored and custody-verified: {path}")
    print(f"fingerprint: {PINNED_SIGNER_FINGERPRINT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
