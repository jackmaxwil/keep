"""Authenticate the exact observation-only proof required for active mode."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = (
    REPO_ROOT / "src/mlx_vq/quality/glm52_sky_must_start.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_glm52_must_start_activation_policy",
    POLICY_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load must-start policy from {POLICY_PATH}")
_POLICY = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _POLICY
_SPEC.loader.exec_module(_POLICY)
validate_must_start_controller_observation = (
    _POLICY.validate_must_start_controller_observation
)
validate_must_start_job_binding = _POLICY.validate_must_start_job_binding


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} is not a readable JSON object") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _expected_binding(args: argparse.Namespace) -> dict[str, object]:
    descriptor_prefix = f"campaigns/{args.run_id}/submissions/"
    submission_prefix = (
        f"campaigns/{args.run_id}/monitor/submission-locks/"
    )
    if not args.descriptor_key.startswith(descriptor_prefix):
        raise ValueError("descriptor key has a mixed or foreign run prefix")
    if not args.submission_key.startswith(submission_prefix):
        raise ValueError("submission key has a mixed or foreign run prefix")
    return {
        "run_id": args.run_id,
        "managed_mode": args.managed_mode,
        "account_id": args.account_id,
        "region": args.region,
        "bucket": args.bucket,
        "descriptor_body_sha256": args.descriptor_body_sha256,
        "submission_body_sha256": args.submission_body_sha256,
        "sky_job_name": args.job_name,
        "must_start_by": args.must_start_by,
        "descriptor_key": args.descriptor_key,
        "descriptor_file_sha256": args.descriptor_file_sha256,
        "submission_key": args.submission_key,
        "submission_submitted_at": args.submission_submitted_at,
        "target_job_id": args.target_job_id,
        "workspace": args.workspace,
        "controller_instance_id": args.controller_instance_id,
        "controller_instance_type": args.controller_instance_type,
        "controller_profile_arn": args.controller_profile_arn,
        "controller_cluster_name": args.controller_cluster_name,
    }


def _load_exact_binding(args: argparse.Namespace) -> dict[str, object]:
    binding = validate_must_start_job_binding(
        _load_object(args.job_binding)
    )
    for field, expected in _expected_binding(args).items():
        if binding.get(field) != expected:
            raise ValueError(f"JOB_BINDING.json {field} authority mismatch")
    return binding


def _observation_key(
    args: argparse.Namespace,
    binding: dict[str, object],
) -> str:
    return (
        f"campaigns/{args.run_id}/monitor/must-start/"
        f"{args.managed_mode}/{args.submission_body_sha256}/observations/"
        f"{binding['observation_body_sha256']}.json"
    )


def _validate_observation(
    args: argparse.Namespace,
    binding: dict[str, object],
) -> dict[str, object]:
    observation = validate_must_start_controller_observation(
        _load_object(args.observation)
    )
    for field in (
        "run_id",
        "managed_mode",
        "account_id",
        "region",
        "bucket",
        "descriptor_body_sha256",
        "submission_body_sha256",
        "sky_job_name",
        "must_start_by",
        "target_job_id",
        "workspace",
        "controller_instance_id",
        "controller_instance_type",
        "controller_profile_arn",
        "controller_cluster_name",
    ):
        if observation.get(field) != binding.get(field):
            raise ValueError(f"controller observation {field} mismatch")
    if (
        observation["observation_body_sha256"]
        != binding["observation_body_sha256"]
    ):
        raise ValueError("JOB_BINDING.json does not bind this observation")
    if observation["observed_at"] != binding["bound_at"]:
        raise ValueError("JOB_BINDING.json bound_at does not bind observation")
    return observation


def _add_authority_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--job-binding", type=Path, required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--managed-mode", required=True)
    parser.add_argument("--descriptor-key", required=True)
    parser.add_argument("--descriptor-file-sha256", required=True)
    parser.add_argument("--descriptor-body-sha256", required=True)
    parser.add_argument("--submission-key", required=True)
    parser.add_argument("--submission-body-sha256", required=True)
    parser.add_argument("--submission-submitted-at", required=True)
    parser.add_argument("--target-job-id", type=int, required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--must-start-by", required=True)
    parser.add_argument("--controller-instance-id", required=True)
    parser.add_argument("--controller-instance-type", required=True)
    parser.add_argument("--controller-profile-arn", required=True)
    parser.add_argument("--controller-cluster-name", required=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    key_parser = subparsers.add_parser("observation-key")
    _add_authority_arguments(key_parser)
    validate_parser = subparsers.add_parser("validate")
    _add_authority_arguments(validate_parser)
    validate_parser.add_argument("--observation", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        binding = _load_exact_binding(args)
        observation_key = _observation_key(args, binding)
        if args.action == "observation-key":
            print(observation_key)
            return 0
        observation = _validate_observation(args, binding)
    except (TypeError, ValueError) as error:
        print(f"must-start activation proof invalid: {error}", file=sys.stderr)
        return 65
    print(
        json.dumps(
            {
                "job_binding_body_sha256": binding[
                    "job_binding_body_sha256"
                ],
                "observation_body_sha256": observation[
                    "observation_body_sha256"
                ],
                "observation_key": observation_key,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
