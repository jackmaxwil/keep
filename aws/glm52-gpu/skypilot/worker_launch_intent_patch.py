#!/usr/bin/env python3
"""Repository-owned Sky 0.13.0 intent-only provisioner integration.

This module is installed by the immutable combined-host image before the
Sky service starts.  It does not import Sky at module import time and owns no
AWS client.  The image bootstrap supplies the already authenticated authority
reader, intent provisioner, and Sky result bridge, then installs this function
over the pinned AWS provisioner entrypoint after exact source verification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Optional

from glm52_enforcement.launch_custody import (
    IntentOnlyProvisioner,
    LaunchCustodyError,
    LaunchParameterAuthority,
)
from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement.sky_admission import (
    PinnedSkyIdentity,
    validate_pinned_sky_identity,
)


EXPECTED_SKYPILOT_VERSION = "0.13.0"
EXPECTED_ORIGINAL_PROVISIONER_SHA256 = (
    "fc5d2e4b94f97c10babb583859da24a4fb19256807e763e064e1f294d6c442ef"
)
_INTENT_PROVISIONER: Optional[IntentOnlyProvisioner] = None
_AUTHORITY_READER: Optional[object] = None
_RESULT_BRIDGE: Optional[object] = None


def verify_original_provisioner(path: Path) -> str:
    """Authenticate the exact installed original before replacing its entry."""

    if not isinstance(path, Path) or not path.is_file():
        raise LaunchCustodyError("pinned original provisioner is absent")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_ORIGINAL_PROVISIONER_SHA256:
        raise LaunchCustodyError("pinned original provisioner source drifted")
    return digest


def configure_intent_route(
    *,
    provisioner: IntentOnlyProvisioner,
    authority_reader: object,
    result_bridge: object,
) -> None:
    """Configure the three closed, dependency-injected production boundaries."""

    global _INTENT_PROVISIONER
    global _AUTHORITY_READER
    global _RESULT_BRIDGE
    if not isinstance(provisioner, IntentOnlyProvisioner):
        raise LaunchCustodyError("intent provisioner is not typed")
    if not callable(getattr(authority_reader, "exact_read", None)):
        raise LaunchCustodyError("launch authority reader is incomplete")
    if not callable(getattr(result_bridge, "to_sky_record", None)):
        raise LaunchCustodyError("Sky provision result bridge is incomplete")
    _INTENT_PROVISIONER = provisioner
    _AUTHORITY_READER = authority_reader
    _RESULT_BRIDGE = result_bridge


def run_instances(region, cluster_name, cluster_name_on_cloud, config):
    """Sky entrypoint: create only an authenticated durable launch intent."""

    del config
    if (
        region != "us-west-2"
        or _INTENT_PROVISIONER is None
        or _AUTHORITY_READER is None
        or _RESULT_BRIDGE is None
    ):
        raise LaunchCustodyError("intent-only production route is not configured")
    authority = _AUTHORITY_READER.exact_read(
        region=region,
        cluster_name=cluster_name,
        cluster_name_on_cloud=cluster_name_on_cloud,
    )
    if not isinstance(authority, LaunchParameterAuthority):
        raise LaunchCustodyError("authenticated launch authority is absent")
    result = _INTENT_PROVISIONER.run(authority)
    return _RESULT_BRIDGE.to_sky_record(result)


def authenticate_pinned_runtime(
    *,
    module: object,
    identity: PinnedSkyIdentity,
    runtime_paths: object,
) -> None:
    """Bind the actually loaded Sky module and every pinned runtime source."""

    validate_pinned_sky_identity(identity)
    required = {
        "original_provisioner",
        "wheel_metadata",
        "dist_record",
        "interpreter",
        "dependency_lock",
        "server_config",
        "jobs_server",
        "patched_provisioner",
    }
    if (
        type(runtime_paths) is not dict
        or set(runtime_paths) != required
        or any(
            not isinstance(path, Path) or not path.is_file()
            for path in runtime_paths.values()
        )
    ):
        raise LaunchCustodyError("pinned Sky runtime paths are incomplete")
    if getattr(module, "__name__", None) != "sky.provision.aws.instance":
        raise LaunchCustodyError("foreign Sky provisioner module")
    loaded_path = getattr(module, "__file__", None)
    original = runtime_paths["original_provisioner"].resolve()
    if (
        type(loaded_path) is not str
        or Path(loaded_path).resolve() != original
        or Path(sys.executable).resolve()
        != runtime_paths["interpreter"].resolve()
    ):
        raise LaunchCustodyError("loaded Sky module or interpreter is foreign")
    expected_hashes = {
        "original_provisioner": identity.original_provisioner_sha256,
        "wheel_metadata": identity.wheel_metadata_sha256,
        "dist_record": identity.dist_record_sha256,
        "interpreter": identity.interpreter_sha256,
        "dependency_lock": identity.dependency_lock_sha256,
        "jobs_server": identity.jobs_server_sha256,
        "patched_provisioner": identity.patched_provisioner_sha256,
    }
    for name, expected in expected_hashes.items():
        if hashlib.sha256(runtime_paths[name].read_bytes()).hexdigest() != expected:
            raise LaunchCustodyError("pinned Sky runtime source drifted: " + name)
    try:
        server_config = json.loads(
            runtime_paths["server_config"].read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LaunchCustodyError("pinned Sky server config is unreadable") from exc
    if (
        type(server_config) is not dict
        or canonical_sha256(server_config) != identity.server_config_sha256
    ):
        raise LaunchCustodyError("pinned Sky server config drifted")
    verify_original_provisioner(original)


def install_into_pinned_module(
    module: object,
    *,
    identity: PinnedSkyIdentity,
    runtime_paths: object,
) -> None:
    """Install only after full loaded-runtime authentication."""

    authenticate_pinned_runtime(
        module=module,
        identity=identity,
        runtime_paths=runtime_paths,
    )
    setattr(module, "run_instances", run_instances)


__all__ = [
    "EXPECTED_ORIGINAL_PROVISIONER_SHA256",
    "EXPECTED_SKYPILOT_VERSION",
    "configure_intent_route",
    "authenticate_pinned_runtime",
    "install_into_pinned_module",
    "run_instances",
    "verify_original_provisioner",
]
