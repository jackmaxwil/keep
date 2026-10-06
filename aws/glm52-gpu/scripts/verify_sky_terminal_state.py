#!/usr/bin/env python3
"""Authenticate matching local and S3 terminal state for a Sky worker."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SKY = _load(
    "_glm52_sky_terminal_campaign",
    REPO_ROOT / "src/mlx_vq/quality/glm52_sky_campaign.py",
)
TERMINAL = _load(
    "_glm52_sky_terminal_state",
    REPO_ROOT / "src/mlx_vq/quality/glm52_sky_terminal_state.py",
)


def _read_environment(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        name, separator, value = line.partition("=")
        if not separator or not name or name in values:
            raise ValueError("campaign environment file is malformed")
        values[name] = value
    return values


def _aws_download(source: str, destination: Path, *, region: str) -> None:
    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            source,
            str(destination),
            "--region",
            region,
            "--only-show-errors",
        ],
        check=True,
    )


def _aws_upload(source: Path, destination: str, *, region: str) -> None:
    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            str(source),
            destination,
            "--region",
            region,
            "--only-show-errors",
        ],
        check=True,
    )


def _write_atomic(path: Path, value: object) -> None:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--region", default="us-west-2")
    args = parser.parse_args()
    descriptor = SKY.validate_sky_campaign_descriptor(
        json.loads(args.descriptor.read_bytes())
    )
    environment = _read_environment(args.root / "runtime/campaign.env")
    deadline = environment["GLM52_EXECUTION_DEADLINE"]
    spend_authority = environment["GLM52_GPU_SPEND_AUTHORITY_SHA256"]
    local_paths = {
        "drained": args.root / "CAMPAIGN_DRAINED.json",
        "campaign": args.root / "ledger/campaign-ledger.jsonl",
        "spend": args.root / "runtime/GPU_SPEND_LEDGER.jsonl",
    }
    for label, path in local_paths.items():
        if path.is_symlink() or not path.is_file():
            parser.error(f"local {label} authority is missing")
    prefix = (
        f"s3://{descriptor['bucket']}/campaigns/{descriptor['run_id']}"
    )
    with tempfile.TemporaryDirectory(prefix="glm52-terminal-s3-") as directory:
        remote_paths = {
            "drained": Path(directory) / "CAMPAIGN_DRAINED.json",
            "campaign": Path(directory) / "campaign-ledger.jsonl",
            "spend": Path(directory) / "GPU_SPEND_LEDGER.jsonl",
        }
        _aws_download(
            f"{prefix}/CAMPAIGN_DRAINED.json",
            remote_paths["drained"],
            region=args.region,
        )
        _aws_download(
            f"{prefix}/ledger/campaign-ledger.jsonl",
            remote_paths["campaign"],
            region=args.region,
        )
        _aws_download(
            f"{prefix}/runtime/GPU_SPEND_LEDGER.jsonl",
            remote_paths["spend"],
            region=args.region,
        )
        for label in local_paths:
            if local_paths[label].read_bytes() != remote_paths[label].read_bytes():
                parser.error(f"local and S3 {label} authorities differ")
        report = TERMINAL.validate_sky_terminal_state(
            run_id=str(descriptor["run_id"]),
            execution_deadline=deadline,
            gpu_spend_authority_sha256=spend_authority,
            drained_raw=remote_paths["drained"].read_bytes(),
            campaign_ledger_raw=remote_paths["campaign"].read_bytes(),
            spend_ledger_raw=remote_paths["spend"].read_bytes(),
        )
    report["descriptor_body_sha256"] = descriptor["descriptor_body_sha256"]
    report["verification_body_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output = args.root / "runtime/TERMINAL_VERIFIED.json"
    _write_atomic(output, report)
    _aws_upload(
        output,
        f"{prefix}/runtime/TERMINAL_VERIFIED.json",
        region=args.region,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
