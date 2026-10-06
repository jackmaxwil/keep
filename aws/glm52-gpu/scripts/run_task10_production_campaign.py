#!/usr/bin/env python3
"""Authenticate fixed Task 10 runtime authority, then exec the campaign."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Callable, Mapping, NoReturn


IMPORT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = IMPORT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from glm52_enforcement.canonical import canonical_json_bytes  # noqa: E402
from glm52_enforcement.task10_worker import (  # noqa: E402
    CAMPAIGN_ROOT as CAMPAIGN_ROOT_TEXT,
    DEADLINE_STATE_PATH,
    DESCRIPTOR_PATH,
    WORKER_BOOTSTRAP_DESCRIPTOR_PATH,
    WORKER_INSTANCE_OBSERVATION_PATH,
    DeadlineState,
    Task10WorkerError,
    build_worker_runtime_authority,
    validate_deadline_state,
    worker_bootstrap_descriptor_from_mapping,
    worker_instance_observation_from_mapping,
)
from mlx_vq.quality.glm52_sky_campaign import (  # noqa: E402
    validate_sky_campaign_descriptor,
)


REPOSITORY_ROOT = Path("/opt/keep-campaign/repo")
CAMPAIGN_ROOT = Path(CAMPAIGN_ROOT_TEXT)
CAMPAIGN = Path(DESCRIPTOR_PATH)
WRAPPER = Path(WORKER_BOOTSTRAP_DESCRIPTOR_PATH)
OBSERVATION = Path(WORKER_INSTANCE_OBSERVATION_PATH)
STATE = Path(DEADLINE_STATE_PATH)
KEY = Path("/etc/keep-glm52/deadline-state.key")
PYTHON = "/usr/bin/python3"
BENCHMARK = REPOSITORY_ROOT / "benchmarks/run_glm52_campaign.py"
RUNTIME_AUTHORITY_VARIABLES = (
    "GLM52_EXECUTION_DEADLINE",
    "GLM52_GPU_ALLOCATION_SHA256",
    "GLM52_GPU_SPEND_AUTHORITY_SHA256",
)


def _fail(message: str) -> NoReturn:
    print("Task 10 production runtime: " + message, file=sys.stderr)
    raise SystemExit(70)


def _canonical(
    path: Path,
    label: str,
) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        raise Task10WorkerError(label + " is absent or not a regular file")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise Task10WorkerError(label + " is unreadable") from exc
    if type(value) is not dict or raw != canonical_json_bytes(value) + b"\n":
        raise Task10WorkerError(label + " is not canonical JSON")
    return value, raw


def _deadline_state(
    *,
    descriptor_file_sha256: str,
) -> DeadlineState:
    raw, _state_bytes = _canonical(STATE, "persistent deadline state")
    try:
        state = DeadlineState(**raw)
    except TypeError as exc:
        raise Task10WorkerError(
            "persistent deadline state is malformed"
        ) from exc
    try:
        signing_key = KEY.read_bytes()
    except OSError as exc:
        raise Task10WorkerError("persistent deadline key is absent") from exc
    if len(signing_key) != 32:
        raise Task10WorkerError("persistent deadline key is corrupt")
    return validate_deadline_state(
        state,
        descriptor_file_sha256=descriptor_file_sha256,
        signing_key=signing_key,
    )


def _runtime_environment(
    inherited: Mapping[str, str],
) -> dict[str, str]:
    wrapper_raw, wrapper_bytes = _canonical(
        WRAPPER,
        "worker bootstrap descriptor",
    )
    observation_raw, _observation_bytes = _canonical(
        OBSERVATION,
        "worker instance observation",
    )
    wrapper = worker_bootstrap_descriptor_from_mapping(wrapper_raw)
    observation = worker_instance_observation_from_mapping(observation_raw)
    runtime = build_worker_runtime_authority(wrapper, observation)
    wrapper_file_sha256 = hashlib.sha256(wrapper_bytes).hexdigest()
    state = _deadline_state(
        descriptor_file_sha256=wrapper_file_sha256,
    )
    campaign_raw, campaign_bytes = _canonical(
        CAMPAIGN,
        "accepted H.1c campaign descriptor",
    )
    campaign = validate_sky_campaign_descriptor(campaign_raw)
    if (
        hashlib.sha256(campaign_bytes).hexdigest()
        != wrapper.base_descriptor_file_sha256
        or campaign["descriptor_body_sha256"]
        != wrapper.base_descriptor_body_sha256
        or campaign["campaign_identity_sha256"]
        != wrapper.campaign_identity_sha256
        or state.instance_id != runtime.observation.instance_id
        or state.allocation_ordinal
        != runtime.observation.allocation_ordinal
        or state.execution_deadline != wrapper.execution_deadline
    ):
        raise Task10WorkerError(
            "campaign, wrapper, observation, or deadline ancestry drifted"
        )
    environment = dict(inherited)
    for name in RUNTIME_AUTHORITY_VARIABLES:
        environment.pop(name, None)
    environment.update(
        {
            "GLM52_EXECUTION_DEADLINE": wrapper.execution_deadline,
            "GLM52_GPU_ALLOCATION_SHA256": (
                wrapper.gpu_allocation_sha256
            ),
            "GLM52_GPU_SPEND_AUTHORITY_SHA256": (
                wrapper.task8_spend_authority_identity_sha256
            ),
        }
    )
    return environment


def main(
    argv: list[str],
    *,
    execve: Callable[[str, list[str], dict[str, str]], object] = os.execve,
) -> int:
    if argv:
        _fail("this runtime entrypoint accepts no arguments")
    try:
        environment = _runtime_environment(os.environ)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        _fail(str(exc))
    arguments = [
        PYTHON,
        str(BENCHMARK),
        "--descriptor",
        str(CAMPAIGN),
        "--root",
        str(CAMPAIGN_ROOT),
        "--repo-root",
        str(REPOSITORY_ROOT),
    ]
    execve(PYTHON, arguments, environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
