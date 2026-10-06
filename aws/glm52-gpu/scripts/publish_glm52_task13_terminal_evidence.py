#!/usr/bin/env python3
"""Publish exact generation-1 Task 13 terminal verification and proof."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from glm52_enforcement.canonical import canonical_json_bytes  # noqa: E402
from glm52_enforcement.task13_terminal_evidence import (  # noqa: E402
    ACCOUNT_ID,
    CAMPAIGN_DRAINED_KEY,
    REGION,
    TASK13_TERMINAL_PROOF_KEY,
    TERMINAL_VERIFIED_KEY,
    TerminalEvidenceError,
    build_task13_terminal_proof,
    build_terminal_verified,
    publish_task13_terminal_proof,
    publish_terminal_verified,
    read_exact_coordinate,
    read_exact_spend_ledger,
    validate_campaign_drained,
    validate_task13_terminal_proof,
    validate_terminal_settlement,
    validate_terminal_verified,
)

PROFILE = "keep-gpu"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--activation-id", required=True)
    parser.add_argument("--terminal-state", required=True, type=Path)
    parser.add_argument("--settlement", required=True, type=Path)
    parser.add_argument("--verified-at", required=True)
    return parser


def _read(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise TerminalEvidenceError(
            label + " must be a regular non-symlink file"
        )
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TerminalEvidenceError(label + " is not JSON") from error
    if type(value) is not dict or raw != canonical_json_bytes(value) + b"\n":
        raise TerminalEvidenceError(
            label + " is not canonical JSON plus LF"
        )
    return dict(value)


def _default_services(profile: str) -> tuple[object, object]:
    import boto3
    from botocore.config import Config

    session = boto3.Session(profile_name=profile, region_name=REGION)
    one_attempt = Config(
        retries={"mode": "standard", "total_max_attempts": 1}
    )
    return (
        session.client("sts", config=one_attempt),
        session.client("s3", config=one_attempt),
    )


def _guard_identity(sts: object) -> None:
    method = getattr(sts, "get_caller_identity", None)
    if not callable(method):
        raise TerminalEvidenceError("STS identity boundary is absent")
    value = method()
    if (
        type(value) is not dict
        or value.get("Account") != ACCOUNT_ID
        or type(value.get("Arn")) is not str
        or f":{ACCOUNT_ID}:" not in value["Arn"]
    ):
        raise TerminalEvidenceError(
            "caller is outside the approved AWS account"
        )


def main(
    argv: Sequence[str] | None = None,
    *,
    services_factory: Callable[[str], tuple[object, object]] = (
        _default_services
    ),
) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.profile != PROFILE:
            raise TerminalEvidenceError(
                f"--profile must be exactly {PROFILE}"
            )
        terminal_state = _read(
            args.terminal_state,
            label="terminal-state verification",
        )
        settlement = _read(
            args.settlement,
            label="terminal settlement",
        )
        sts, s3 = services_factory(args.profile)
        _guard_identity(sts)
        drained, drained_coordinate = read_exact_coordinate(
            s3,
            key=CAMPAIGN_DRAINED_KEY,
            validator=validate_campaign_drained,
        )
        spend_ledger = settlement.get("gpu_spend_ledger")
        if type(spend_ledger) is not dict:
            raise TerminalEvidenceError(
                "terminal settlement spend source is absent"
            )
        spend_ledger_raw = read_exact_spend_ledger(
            s3,
            spend_ledger,
        )
        validate_terminal_settlement(
            settlement,
            terminal_state=terminal_state,
            spend_ledger_raw=spend_ledger_raw,
        )
        terminal_verified = build_terminal_verified(
            activation_id=args.activation_id,
            campaign_drained=drained_coordinate,
            campaign_drained_value=drained,
            terminal_state=terminal_state,
            verified_at=args.verified_at,
        )
        terminal_verified_coordinate = publish_terminal_verified(
            s3,
            terminal_verified,
        )
        _verified, exact_verified_coordinate = read_exact_coordinate(
            s3,
            key=TERMINAL_VERIFIED_KEY,
            validator=validate_terminal_verified,
        )
        if exact_verified_coordinate != terminal_verified_coordinate:
            raise TerminalEvidenceError(
                "terminal verified coordinate changed after publication"
            )
        proof = build_task13_terminal_proof(
            activation_id=args.activation_id,
            campaign_drained=drained_coordinate,
            campaign_drained_value=drained,
            terminal_verified=terminal_verified_coordinate,
            terminal_verified_value=terminal_verified,
            settlement=settlement,
            spend_ledger_raw=spend_ledger_raw,
        )
        proof_coordinate = publish_task13_terminal_proof(
            s3,
            proof,
            terminal_verified_value=terminal_verified,
            campaign_drained_value=drained,
            spend_ledger_raw=spend_ledger_raw,
        )
        _proof, exact_proof_coordinate = read_exact_coordinate(
            s3,
            key=TASK13_TERMINAL_PROOF_KEY,
        )
        validate_task13_terminal_proof(
            _proof,
            terminal_verified_value=terminal_verified,
            campaign_drained_value=drained,
            spend_ledger_raw=spend_ledger_raw,
        )
        if exact_proof_coordinate != proof_coordinate:
            raise TerminalEvidenceError(
                "Task 13 terminal proof coordinate changed after publication"
            )
    except (OSError, TerminalEvidenceError, ValueError) as error:
        print(
            f"Task 13 terminal publication refused: {error}",
            file=sys.stderr,
        )
        return 70
    print(
        json.dumps(
            {
                "terminal_verified": terminal_verified_coordinate,
                "task13_terminal_proof": proof_coordinate,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
