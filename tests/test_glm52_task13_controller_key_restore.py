"""Custody guarantees of the Task 13 controller-key escrow restore.

These tests never touch AWS and never read the production private key. They
drive `restore()` with a fake Secrets Manager session holding an ephemeral key,
and assert the custody and cleanup invariants that protect real key material.

Every directory that can hold private material is created 0700 and removed in a
`finally`, so an interrupted run never leaves an unencrypted key in the
temporary directory.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "glm52_task13_controller_key_restore",
    REPO / "aws/glm52-gpu/scripts/restore_glm52_task13_controller_key.py",
)
assert _SPEC is not None and _SPEC.loader is not None
restore_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(restore_module)

ACCOUNT_ID = restore_module.ACCOUNT_ID
KEY_NAME = restore_module.KEY_NAME
RestoreError = restore_module.RestoreError


def _private_dir() -> Path:
    work = Path(tempfile.mkdtemp(prefix="glm52-restore-test-")).resolve(strict=True)
    work.chmod(0o700)
    return work


def _ephemeral_key_bytes(scratch: Path) -> tuple[bytes, str]:
    """Generate a throwaway Ed25519 key and return (private bytes, public text)."""

    key = scratch / "ephemeral"
    subprocess.run(
        [
            "/usr/bin/ssh-keygen",
            "-t", "ed25519",
            "-N", "",
            "-C", "glm52-restore-test-ephemeral",
            "-f", str(key),
            "-q",
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return key.read_bytes(), key.with_suffix(".pub").read_text(encoding="ascii")


class _FakeSecrets:
    def __init__(self, material: bytes) -> None:
        self._material = material
        self.calls = 0

    def get_secret_value(self, **_kwargs: object) -> dict[str, object]:
        self.calls += 1
        return {"SecretBinary": self._material}


class _FakeSts:
    def __init__(self, account: str) -> None:
        self._account = account

    def get_caller_identity(self) -> dict[str, str]:
        return {"Account": self._account}


class _FakeSession:
    def __init__(self, material: bytes, *, account: str = ACCOUNT_ID) -> None:
        self.secrets = _FakeSecrets(material)
        self._account = account

    def client(self, name: str, **_kwargs: object) -> object:
        if name == "secretsmanager":
            return self.secrets
        if name == "sts":
            return _FakeSts(self._account)
        raise AssertionError(f"unexpected client {name!r}")


@pytest.fixture()
def ephemeral(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[bytes, Path]]:
    """An ephemeral key escrow-shaped for restore, with pins patched to match.

    Both private directories are destroyed on the way out, including when a test
    fails or is interrupted.
    """

    scratch = _private_dir()
    work = _private_dir()
    try:
        private_bytes, public_text = _ephemeral_key_bytes(scratch)
        core, fingerprint = restore_module._public_key_identity(public_text)
        signer_module = sys.modules[
            restore_module.ControllerAuthoritySigner.__module__
        ]
        for module in (restore_module, signer_module):
            monkeypatch.setattr(
                module, "PINNED_SIGNER_PUBLIC_KEY", public_text.strip()
            )
            monkeypatch.setattr(module, "PINNED_SIGNER_FINGERPRINT", fingerprint)
        monkeypatch.setattr(signer_module, "_PINNED_PUBLIC_CORE", core)
        yield private_bytes, work
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)


def test_restore_materializes_exact_custody_and_proves_the_signer(
    ephemeral: tuple[bytes, Path],
) -> None:
    """Break caught: a restored key that the issuer would later refuse."""

    private_bytes, work = ephemeral
    session = _FakeSession(private_bytes)

    path = restore_module.restore(work, session=session)

    assert path == work / KEY_NAME
    for target in (path, path.with_suffix(".pub")):
        info = target.lstat()
        assert oct(info.st_mode & 0o777) == "0o600"
        assert info.st_nlink == 1
        assert info.st_uid == os.getuid()
        assert not target.is_symlink()


def test_restore_refuses_a_foreign_account(ephemeral: tuple[bytes, Path]) -> None:
    """Break caught: restoring campaign key material into the wrong account."""

    private_bytes, work = ephemeral
    session = _FakeSession(private_bytes, account="111111111111")

    with pytest.raises(RestoreError, match="pinned account"):
        restore_module.restore(work, session=session)

    assert not (work / KEY_NAME).exists()
    assert session.secrets.calls == 0, "secret must not be fetched off-account"


def test_restore_never_deletes_a_file_it_did_not_create(
    ephemeral: tuple[bytes, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: losing another process's valid custody during cleanup.

    A concurrent writer can win the O_EXCL race between the preflight check and
    creation. Cleanup must then leave the winner's file untouched.
    """

    private_bytes, work = ephemeral
    session = _FakeSession(private_bytes)
    foreign = work / KEY_NAME
    real_create = restore_module._create_exclusive_0600

    def _plant_then_create(target: Path, payload: bytes) -> None:
        # Simulate the racing writer landing after the preflight check.
        if target == foreign and not foreign.exists():
            foreign.write_bytes(b"FOREIGN-WINNER")
            foreign.chmod(0o600)
        real_create(target, payload)

    monkeypatch.setattr(
        restore_module, "_create_exclusive_0600", _plant_then_create
    )
    with pytest.raises(FileExistsError):
        restore_module.restore(work, session=session)

    assert foreign.read_bytes() == b"FOREIGN-WINNER", (
        "cleanup unlinked a file this call did not create"
    )


def test_restore_refuses_existing_custody_material(
    ephemeral: tuple[bytes, Path],
) -> None:
    """Break caught: a second restore silently replacing a live key."""

    private_bytes, work = ephemeral
    session = _FakeSession(private_bytes)
    restore_module.restore(work, session=session)
    original = (work / KEY_NAME).read_bytes()

    with pytest.raises(RestoreError, match="refusing to overwrite"):
        restore_module.restore(work, session=session)

    assert (work / KEY_NAME).read_bytes() == original


def test_restore_refuses_a_symlinked_work_directory(
    ephemeral: tuple[bytes, Path],
) -> None:
    """Break caught: custody redirected through a planted symlink."""

    private_bytes, work = ephemeral
    holder = _private_dir()
    try:
        link = holder / "work"
        link.symlink_to(work)
        with pytest.raises(RestoreError, match="0700 directory"):
            restore_module.restore(link, session=_FakeSession(private_bytes))
    finally:
        shutil.rmtree(holder, ignore_errors=True)


def test_restore_refuses_a_group_readable_work_directory(
    ephemeral: tuple[bytes, Path],
) -> None:
    """Break caught: key material landing in a directory others can enter."""

    private_bytes, work = ephemeral
    work.chmod(0o750)
    try:
        with pytest.raises(RestoreError, match="0700 directory"):
            restore_module.restore(work, session=_FakeSession(private_bytes))
    finally:
        work.chmod(0o700)


def test_wipe_refuses_a_directory_that_is_not_exact_private_custody(
    ephemeral: tuple[bytes, Path],
) -> None:
    """Break caught: wipe deleting paths outside a verified custody directory."""

    private_bytes, work = ephemeral
    restore_module.restore(work, session=_FakeSession(private_bytes))
    work.chmod(0o755)
    try:
        with pytest.raises(RestoreError, match="0700 directory"):
            restore_module.wipe(work)
        assert (work / KEY_NAME).exists()
    finally:
        work.chmod(0o700)

    restore_module.wipe(work)
    assert not (work / KEY_NAME).exists()
    assert not (work / (KEY_NAME + ".pub")).exists()
