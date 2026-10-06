from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import audit_glm52_reap_materialization as audit_cli


def _args(**overrides):
    values = {
        "all_groups": True,
        "non_vq_artifact_dir": "/tmp/non-vq",
        "non_vq_evidence_json": "/tmp/non-vq.json",
    }
    values.update(overrides)
    return Namespace(**values)


def test_full_audit_requires_both_non_vq_composite_inputs() -> None:
    with pytest.raises(ValueError, match="full GLM52 audit requires"):
        audit_cli._require_composite_inputs(
            _args(non_vq_artifact_dir=None, non_vq_evidence_json=None)
        )
    with pytest.raises(ValueError, match="must be supplied together"):
        audit_cli._require_composite_inputs(
            _args(non_vq_evidence_json=None)
        )

    audit_cli._require_composite_inputs(_args())
    audit_cli._require_composite_inputs(
        _args(
            all_groups=False,
            non_vq_artifact_dir=None,
            non_vq_evidence_json=None,
        )
    )


def test_composite_accounting_reports_actual_payload_and_physical_values() -> None:
    routed = SimpleNamespace(
        audit_pass=True,
        full_routed_artifact_ready=True,
        actual_routed_payload_bytes=61_312_204_800,
        actual_routed_codebook_bytes=230_400,
        artifact_tree_bytes=61_312_700_000,
    )
    non_vq = SimpleNamespace(
        audit_pass=True,
        tensor_payload_bytes=37_121_488_608,
        artifact_tree_bytes=37_122_403_011,
        parameter_count=18_560_731_704,
    )
    accounting = SimpleNamespace(
        main_non_routed_tensor_payload_bytes=37_121_488_608,
        main_non_routed_parameter_count=18_560_731_704,
        main_model_parameter_count_excluding_mtp=494_194_805_304,
    )

    payload = audit_cli._actual_composite_accounting(
        routed_audit=routed,
        non_vq_audit=non_vq,
        source_accounting=accounting,
    )

    assert payload["actual_whole_model_tensor_payload_bytes"] == 98_433_923_808
    assert payload["actual_whole_model_tensor_payload_bpw"] == pytest.approx(
        1.593443277857996,
        rel=0,
        abs=1e-15,
    )
    assert payload["actual_whole_model_artifact_tree_bytes"] == 98_435_103_011
    assert payload["actual_whole_model_physical_bpw"] == pytest.approx(
        98_435_103_011 * 8 / 494_194_805_304
    )
    assert payload["whole_model_tensor_payload_values_actual"] is True
    assert payload["whole_model_physical_values_actual"] is True


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("full_routed_artifact_ready", False, "full routed artifact"),
        ("audit_pass", False, "routed artifact audit"),
    ],
)
def test_composite_accounting_rejects_non_green_routed_authority(
    field: str,
    value: bool,
    match: str,
) -> None:
    routed = SimpleNamespace(
        audit_pass=True,
        full_routed_artifact_ready=True,
        actual_routed_payload_bytes=10,
        actual_routed_codebook_bytes=1,
        artifact_tree_bytes=20,
    )
    setattr(routed, field, value)
    non_vq = SimpleNamespace(
        audit_pass=True,
        tensor_payload_bytes=30,
        artifact_tree_bytes=40,
        parameter_count=3,
    )
    accounting = SimpleNamespace(
        main_non_routed_tensor_payload_bytes=30,
        main_non_routed_parameter_count=3,
        main_model_parameter_count_excluding_mtp=100,
    )

    with pytest.raises(ValueError, match=match):
        audit_cli._actual_composite_accounting(
            routed_audit=routed,
            non_vq_audit=non_vq,
            source_accounting=accounting,
        )


def test_composite_accounting_rejects_non_vq_or_source_accounting_drift() -> None:
    routed = SimpleNamespace(
        audit_pass=True,
        full_routed_artifact_ready=True,
        actual_routed_payload_bytes=10,
        actual_routed_codebook_bytes=1,
        artifact_tree_bytes=20,
    )
    non_vq = SimpleNamespace(
        audit_pass=True,
        tensor_payload_bytes=30,
        artifact_tree_bytes=40,
        parameter_count=4,
    )
    accounting = SimpleNamespace(
        main_non_routed_tensor_payload_bytes=30,
        main_non_routed_parameter_count=3,
        main_model_parameter_count_excluding_mtp=100,
    )

    with pytest.raises(ValueError, match="parameter count"):
        audit_cli._actual_composite_accounting(
            routed_audit=routed,
            non_vq_audit=non_vq,
            source_accounting=accounting,
        )


def test_audit_outputs_cannot_alias_or_enter_authority_roots(tmp_path: Path) -> None:
    source = tmp_path / "source"
    routed = tmp_path / "routed"
    non_vq = tmp_path / "non-vq"
    for root in (source, routed, non_vq):
        root.mkdir()
    protected = tmp_path / "non-vq-evidence.json"
    protected.write_text("{}\n")
    args = Namespace(
        source_dir=str(source),
        artifact_dir=str(routed),
        non_vq_artifact_dir=str(non_vq),
        profile_path=str(tmp_path / "profile.yaml"),
        config_path=str(source / "config.json"),
        index_path=str(source / "model.safetensors.index.json"),
        non_vq_evidence_json=str(protected),
        materialization_runs_jsonl=None,
        materialization_evidence=None,
        output_json=str(routed / "audit.json"),
        append_jsonl=None,
    )

    with pytest.raises(ValueError, match="outside authority and artifact roots"):
        audit_cli._validated_output_paths(args)

    args.output_json = str(protected)
    with pytest.raises(ValueError, match="aliases protected input"):
        audit_cli._validated_output_paths(args)

    hardlink_alias = tmp_path / "hardlink-audit.json"
    hardlink_alias.hardlink_to(protected)
    args.output_json = str(hardlink_alias)
    with pytest.raises(ValueError, match="aliases protected input"):
        audit_cli._validated_output_paths(args)

    shared_target = tmp_path / "shared-output"
    shared_target.write_text("")
    json_alias = tmp_path / "audit.json"
    jsonl_alias = tmp_path / "audit.jsonl"
    json_alias.symlink_to(shared_target)
    jsonl_alias.symlink_to(shared_target)
    args.output_json = str(json_alias)
    args.append_jsonl = str(jsonl_alias)
    with pytest.raises(ValueError, match="must be distinct"):
        audit_cli._validated_output_paths(args)
