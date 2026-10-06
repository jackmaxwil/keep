#!/usr/bin/env python3
"""Build the deterministic import-complete H.1g support Lambda archive."""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[3]
FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _entries() -> tuple[tuple[str, Path], ...]:
    package = ROOT / "src/glm52_enforcement"
    wrappers = ROOT / "aws/glm52-gpu/lambda"
    entries = [
        (
            "support_budget_gate_handler.py",
            wrappers / "support_budget_gate_handler.py",
        ),
        (
            "support_rehearsal_collector_handler.py",
            wrappers / "support_rehearsal_collector_handler.py",
        ),
        (
            "support_rehearsal_probe_handler.py",
            wrappers / "support_rehearsal_probe_handler.py",
        ),
        (
            "support_decision_handler.py",
            wrappers / "support_decision_handler.py",
        ),
        (
            "support_source_publisher_handler.py",
            wrappers / "support_source_publisher_handler.py",
        ),
        (
            "support_effect_writer_handler.py",
            wrappers / "support_effect_writer_handler.py",
        ),
        (
            "support_fence_handler.py",
            wrappers / "support_fence_handler.py",
        ),
        (
            "support_numericbinding_handler.py",
            wrappers / "support_numericbinding_handler.py",
        ),
        (
            "support_attestation_handler.py",
            wrappers / "support_attestation_handler.py",
        ),
        (
            "support_launchadmission_handler.py",
            wrappers / "support_launchadmission_handler.py",
        ),
        (
            "support_custom_resource_handler.py",
            wrappers / "support_custom_resource_handler.py",
        ),
        (
            "support_retainedcancellation_handler.py",
            wrappers / "support_retainedcancellation_handler.py",
        ),
        (
            "support_supportdeadline_handler.py",
            wrappers / "support_supportdeadline_handler.py",
        ),
        (
            "retained_support_lifecycle_handler.py",
            wrappers / "retained_support_lifecycle_handler.py",
        ),
        (
            "retained_execution_observer_handler.py",
            wrappers / "retained_execution_observer_handler.py",
        ),
        (
            "retained_terminal_v2_handler.py",
            wrappers / "retained_terminal_v2_handler.py",
        ),
        (
            "retained_finalizer_handler.py",
            wrappers / "retained_finalizer_handler.py",
        ),
        (
            "retained_h1g_drained_handler.py",
            wrappers / "retained_h1g_drained_handler.py",
        ),
        (
            "retained_worker_drain_handler.py",
            wrappers / "retained_worker_drain_handler.py",
        ),
        (
            "retained_operator_disposition_handler.py",
            wrappers / "retained_operator_disposition_handler.py",
        ),
        (
            "retained_orphan_audit_handler.py",
            wrappers / "retained_orphan_audit_handler.py",
        ),
        (
            "retained_snapshot_cleanup_handler.py",
            wrappers / "retained_snapshot_cleanup_handler.py",
        ),
        (
            "task10_sole_sender_handler.py",
            wrappers / "task10_sole_sender_handler.py",
        ),
        (
            "task10_sole_sender_runtime.py",
            wrappers / "task10_sole_sender_runtime.py",
        ),
        (
            "task9_liability_watcher_handler.py",
            wrappers / "task9_liability_watcher_handler.py",
        ),
        (
            "task9_liability_watcher_runtime.py",
            package / "task9_liability_watcher_runtime.py",
        ),
    ]
    entries.extend(
        ("glm52_enforcement/" + path.name, path)
        for path in sorted(package.glob("*.py"))
    )
    return tuple(entries)


def build(output: Path) -> None:
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for archive_name, source in _entries():
            info = zipfile.ZipInfo(archive_name, FIXED_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
