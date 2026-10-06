"""Ephemeral Ed25519 signer support for Task 13 authority tests."""

from __future__ import annotations

import atexit
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import glm52_enforcement.task13_controller_authority as controller_authority


PRODUCTION_PINNED_SIGNER_PUBLIC_KEY = (
    controller_authority.PINNED_SIGNER_PUBLIC_KEY
)
PRODUCTION_PINNED_SIGNER_FINGERPRINT = (
    controller_authority.PINNED_SIGNER_FINGERPRINT
)
_SSH_KEYGEN = Path("/usr/bin/ssh-keygen")
_KEY_NAME = "controller-authority-ed25519"
_KEY_COMMENT = "keep-glm52-task13-tests-ephemeral"
_POINTER_NAME = "full-run-work-current"
_EPHEMERAL_SIGNER: Optional[
    tuple[Path, Path, controller_authority.ControllerAuthoritySigner]
] = None


def _read_public_key(public_key_path: Path) -> str:
    public_key_text = public_key_path.read_text(encoding="ascii")
    if not public_key_text.endswith("\n") or public_key_text.count("\n") != 1:
        raise RuntimeError("ephemeral controller public key is not one line")
    return public_key_text[:-1]


def patch_controller_authority_consumer_pins(
    consumer: object,
    public_key_path: Path,
) -> None:
    """Patch a test-loaded verifier that carries its own copied pins."""

    public_key = _read_public_key(public_key_path)
    _, fingerprint = controller_authority._public_key_identity(public_key)
    setattr(consumer, "PINNED_SIGNER_PUBLIC_KEY", public_key)
    setattr(consumer, "PINNED_SIGNER_FINGERPRINT", fingerprint)


def patch_controller_authority_pins(
    public_key_path: Path,
    *,
    full_run_work_pointer: Optional[Path] = None,
) -> None:
    """Patch every runtime pin used to issue or verify test authorities."""

    public_key = _read_public_key(public_key_path)
    public_core, _ = controller_authority._public_key_identity(public_key)
    patch_controller_authority_consumer_pins(
        controller_authority,
        public_key_path,
    )
    controller_authority._PINNED_PUBLIC_CORE = public_core
    if full_run_work_pointer is not None:
        controller_authority.FULL_RUN_WORK_POINTER = full_run_work_pointer
    public_key_path.write_text(public_key + "\n", encoding="ascii")
    public_key_path.chmod(0o600)


def ephemeral_controller_authority_signer(
    *,
    comment: str = _KEY_COMMENT,
) -> tuple[Path, Path, controller_authority.ControllerAuthoritySigner]:
    """Return one process-wide signer backed only by a disposable test key."""

    global _EPHEMERAL_SIGNER
    if _EPHEMERAL_SIGNER is not None:
        return _EPHEMERAL_SIGNER

    work = Path(
        tempfile.mkdtemp(prefix="glm52-task13-ephemeral-signer-")
    )
    work.chmod(0o700)
    work = work.resolve(strict=True)
    atexit.register(shutil.rmtree, work, ignore_errors=True)

    private_key = work / _KEY_NAME
    generated = subprocess.run(
        [
            str(_SSH_KEYGEN),
            "-t",
            "ed25519",
            "-N",
            "",
            "-C",
            comment,
            "-f",
            str(private_key),
            "-q",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        timeout=10,
    )
    if generated.returncode != 0:
        raise RuntimeError(
            "ephemeral controller signing key generation failed: "
            + generated.stderr.decode("ascii", errors="replace")
        )

    public_key = private_key.with_suffix(".pub")
    private_key.chmod(0o600)
    public_key.chmod(0o600)
    patch_controller_authority_pins(public_key)
    pointer = work / _POINTER_NAME
    pointer.write_text(str(work) + "\n", encoding="ascii")
    pointer.chmod(0o600)
    signer = controller_authority.ControllerAuthoritySigner(
        private_key_path=private_key,
        full_run_work=work,
    )
    _EPHEMERAL_SIGNER = (work, private_key, signer)
    return _EPHEMERAL_SIGNER


def patched_python_script_command(
    *,
    public_key_path: Path,
    script_path: Path,
    arguments: Sequence[object],
) -> list[str]:
    """Build a child command that applies the same test pins before a CLI."""

    pointer_path = public_key_path.parent / _POINTER_NAME
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        str(public_key_path),
        str(pointer_path),
        str(script_path),
        *(str(argument) for argument in arguments),
    ]


def _run_patched_script() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(
            "usage: glm52_task13_signer_support.py "
            "PUBLIC_KEY POINTER SCRIPT [ARG ...]"
        )
    public_key_path = Path(sys.argv[1]).resolve(strict=True)
    pointer_path = Path(sys.argv[2]).resolve(strict=True)
    script_path = Path(sys.argv[3]).resolve(strict=True)
    patch_controller_authority_pins(
        public_key_path,
        full_run_work_pointer=pointer_path,
    )
    sys.argv = [str(script_path), *sys.argv[4:]]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    _run_patched_script()
