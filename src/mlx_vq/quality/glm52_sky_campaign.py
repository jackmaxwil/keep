"""Compatibility alias for the enforcement-native campaign contract.

Operator scripts also load this file directly by path, outside an installed
package and without ``PYTHONPATH``.  Populate this module's namespace before
installing the normal-import alias so those path-loaded callers receive the
same validation types and functions as package imports.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path
import sys
from typing import Mapping


_MOUNT_ROOT_NAME = "glm52-worker-start-v2"
_MOUNTED_NATIVE_FILENAME = "glm52_sky_campaign_native.py"
_NATIVE_MODULE_NAME = "glm52_sky_campaign_native"


def _load_native_path(native_path: Path) -> object:
    _existing_native = sys.modules.get(_NATIVE_MODULE_NAME)
    if _existing_native is not None:
        _existing_path = getattr(_existing_native, "__file__", None)
        if (
            not isinstance(_existing_path, str)
            or Path(_existing_path).resolve() != native_path
        ):
            raise RuntimeError("Sky campaign native policy module identity drifted")
        return _existing_native
    _native_spec = importlib.util.spec_from_file_location(
        _NATIVE_MODULE_NAME,
        native_path,
    )
    if _native_spec is None or _native_spec.loader is None:
        raise RuntimeError(f"cannot load {native_path}")
    native = importlib.util.module_from_spec(_native_spec)
    sys.modules[_NATIVE_MODULE_NAME] = native
    _native_spec.loader.exec_module(native)
    return native


_facade_path = Path(__file__).resolve()
_mounted_native_path = _facade_path.with_name(_MOUNTED_NATIVE_FILENAME)
if _facade_path.parent.name == _MOUNT_ROOT_NAME:
    if not _mounted_native_path.is_file() or _mounted_native_path.is_symlink():
        raise RuntimeError("mounted Sky campaign native policy is unavailable")
    _native = _load_native_path(_mounted_native_path)
else:
    try:
        from glm52_enforcement import glm52_sky_campaign as _native
    except ModuleNotFoundError as error:
        if error.name != "glm52_enforcement":
            raise
        _native = _load_native_path(
            _facade_path.parents[2] / "glm52_enforcement/glm52_sky_campaign.py"
        )

for _name, _value in vars(_native).items():
    if _name not in {
        "__builtins__",
        "__cached__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__spec__",
    }:
        globals()[_name] = _value

APPROVED_ACCOUNT_ID = "246813579024"
APPROVED_REGION = "us-west-2"
APPROVED_INSTANCE_TYPE = "p5.48xlarge"
if (
    _native.APPROVED_ACCOUNT_ID != APPROVED_ACCOUNT_ID
    or _native.APPROVED_REGION != APPROVED_REGION
    or _native.APPROVED_INSTANCE_TYPE != APPROVED_INSTANCE_TYPE
):
    raise RuntimeError("enforcement-native Sky campaign approval constants drifted")


def build_sky_campaign_descriptor(
    *,
    run_id: str,
    must_start_by: datetime,
    controller_identity: str,
    worker_identity: str,
    vpc_name: str,
    image_id: str,
    bucket: str,
    jobs_bucket: str,
    repo_tar_key: str,
    repo_tar_sha256: str,
    campaign_descriptor_key: str,
    approval_key: str,
    approval_sha256: str,
    artifacts: Mapping[str, str],
) -> dict[str, object]:
    """Compatibility route for the enforcement-native descriptor builder."""
    return _native.build_sky_campaign_descriptor(
        run_id=run_id,
        must_start_by=must_start_by,
        controller_identity=controller_identity,
        worker_identity=worker_identity,
        vpc_name=vpc_name,
        image_id=image_id,
        bucket=bucket,
        jobs_bucket=jobs_bucket,
        repo_tar_key=repo_tar_key,
        repo_tar_sha256=repo_tar_sha256,
        campaign_descriptor_key=campaign_descriptor_key,
        approval_key=approval_key,
        approval_sha256=approval_sha256,
        artifacts=artifacts,
    )


def validate_sky_campaign_descriptor(
    value: Mapping[str, object], *, verify_body_sha: bool = True
) -> dict[str, object]:
    """Compatibility route for the enforcement-native descriptor validator."""
    return _native.validate_sky_campaign_descriptor(
        value, verify_body_sha=verify_body_sha
    )


def require_approved_aws_identity(
    identity: Mapping[str, object],
) -> dict[str, object]:
    """Compatibility route for the enforcement-native account guard."""
    return _native.require_approved_aws_identity(identity)
