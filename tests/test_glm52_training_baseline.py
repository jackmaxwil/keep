from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


PROVIDER_MODULE = "mlx_vq.quality.glm52_training_baseline"
PROVIDER_SPEC = f"{PROVIDER_MODULE}:accepted_baseline_provider"
IDENTITY_FROM_AUDIT = "a" * 64
CALLER_IDENTITY = "b" * 64
RECOVERY_IDENTITY = "c" * 64


@pytest.fixture(autouse=True)
def _headless_quality_package(monkeypatch: pytest.MonkeyPatch) -> None:
    quality = ModuleType("mlx_vq.quality")
    quality.__path__ = [str(Path(__file__).parents[1] / "src/mlx_vq/quality")]
    monkeypatch.setitem(sys.modules, "mlx_vq.quality", quality)
    sys.modules.pop(PROVIDER_MODULE, None)


@dataclass(frozen=True)
class _FakeBaseline:
    model: object
    candidate_identity_sha256: str


def _write_config(tmp_path: Path, **updates: object) -> Path:
    audit_path = tmp_path / "accepted-composite-audit.json"
    audit_path.write_text(
        json.dumps({"candidate_identity_sha256": CALLER_IDENTITY}),
        encoding="utf-8",
    )
    readiness_path = tmp_path / "tokenizer-readiness.json"
    readiness_path.write_text(json.dumps({"prompt": "authenticated prompt"}), encoding="utf-8")
    payload: dict[str, object] = {
        "profile_path": "/authority/profile.yaml",
        "config_path": "/authority/config.json",
        "source_index_path": "/authority/model.safetensors.index.json",
        "tokenizer_dir": "/authority/tokenizer",
        "tokenizer_readiness_json": str(readiness_path),
        "family_policy_json": "/authority/family-policy.json",
        "non_vq_artifact_dir": "/artifacts/non-vq",
        "non_vq_evidence_json": "/authority/non-vq-evidence.json",
        "routed_artifact_dir": "/artifacts/e8-full-w1",
        "composite_audit_json": str(audit_path),
        "expected_composite_audit_sha256": hashlib.sha256(
            audit_path.read_bytes()
        ).hexdigest(),
        "materialization_runs_jsonl": "/authority/materialization.jsonl",
        "full_bind_preflight_json": "/authority/full-bind-preflight.json",
        "model_id": "0xSero/glm-5.2-reap-504B-v2",
        "revision": "pinned-revision",
    }
    payload.update(updates)
    config_path = tmp_path / "training-baseline.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def _install_fake_runtime(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, object]] = []
    model = object()
    config_path = Path(os.environ["GLM52_TRAINING_BASELINE_JSON"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validated = SimpleNamespace(
        artifact_identity=SimpleNamespace(sha256=IDENTITY_FROM_AUDIT),
        input_evidence_file_sha256={
            "composite_audit_json": config["expected_composite_audit_sha256"]
        },
        profile=SimpleNamespace(name="synthetic-glm52"),
    )

    def validate_glm52_production_inputs(**kwargs: object):
        calls.append(("validate", kwargs))
        return validated

    def load_authenticated_glm52_composite(value: object):
        calls.append(("load", value))
        return model, SimpleNamespace(
            artifact_identity_sha256=IDENTITY_FROM_AUDIT,
            bound_sparse_layer_ids=tuple(range(3, 78)),
            dense_routed_parameter_names=(),
            unbound_vq_experts=False,
        )

    composite = SimpleNamespace(
        GLM52_EXPECTED_SPARSE_LAYERS=tuple(range(3, 78)),
        validate_glm52_production_inputs=validate_glm52_production_inputs,
        load_authenticated_glm52_composite=load_authenticated_glm52_composite,
    )
    training = SimpleNamespace(ValidatedGLM52TrainingBaseline=_FakeBaseline)
    monkeypatch.setitem(
        sys.modules, "mlx_vq.models.glm52_composite_loader", composite
    )
    monkeypatch.setitem(
        sys.modules, "mlx_vq.quality.glm52_adapter_training", training
    )
    return calls, model


def _recovery_candidate_config(tmp_path: Path, *, composite_sha256: str) -> dict[str, str]:
    conversion_dir = tmp_path / "recovery-candidate"
    (conversion_dir / "artifact").mkdir(parents=True)
    policy_path = tmp_path / "recovery-policy.json"
    policy_path.write_text(
        json.dumps({"kind": "synthetic-recovery", "version": "1"}),
        encoding="utf-8",
    )
    return {
        "conversion_dir": str(conversion_dir),
        "expected_composite_audit_sha256": composite_sha256,
        "expected_seed_manifest_sha256": "1" * 64,
        "expected_stats_manifest_sha256": "2" * 64,
        "expected_full_source_blob_inventory_sha256": "3" * 64,
        "expected_routed_source_blob_inventory_sha256": "4" * 64,
        "expected_recovery_lever": "synthetic_recovery",
        "expected_recovery_policy_json": str(policy_path),
    }


def _install_fake_recovery_runtime(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, object]] = []
    layers = tuple(range(3, 78))
    projections = ("gate_proj", "up_proj", "down_proj")
    groups = tuple(
        SimpleNamespace(
            group_key=f"{layer}:{projection}",
            classification="replacement",
            filename=f"layer-{layer}-{projection}.safetensors",
        )
        for layer in layers
        for projection in projections
    )

    def verify_current_identity() -> None:
        calls.append(("verify_recovery_identity", None))

    audit = SimpleNamespace(
        audit_pass=True,
        group_count=225,
        complete_replacement_layer_ids=layers,
        groups=groups,
        recovery_dir="/synthetic/recovery",
        candidate_identity_sha256=RECOVERY_IDENTITY,
        verify_current_identity=verify_current_identity,
    )

    def audit_glm52_recovery_mixed_artifact(*args: object, **kwargs: object):
        calls.append(("audit_recovery", (args, kwargs)))
        return audit

    def authenticated_recovery_candidate_group_paths(value: object):
        value.verify_current_identity()
        calls.append(("recovery_group_paths", value))
        return {group.filename: Path("/synthetic") / group.filename for group in groups}

    def bind_glm52_vq_experts_from_paths(
        model: object,
        paths: object,
        *,
        layers: tuple[int, ...],
        profile: object,
        strict: bool,
    ) -> tuple[int, ...]:
        calls.append(
            (
                "bind_recovery",
                {
                    "model": model,
                    "paths": paths,
                    "layers": layers,
                    "profile": profile,
                    "strict": strict,
                },
            )
        )
        return layers

    monkeypatch.setitem(
        sys.modules,
        "mlx_vq.validate.glm52_recovery_artifact",
        SimpleNamespace(
            audit_glm52_recovery_mixed_artifact=audit_glm52_recovery_mixed_artifact
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vq.quality.glm52_recovery",
        SimpleNamespace(
            PROJECTIONS=projections,
            authenticated_recovery_candidate_group_paths=(
                authenticated_recovery_candidate_group_paths
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vq.models.glm52_vq_adapter",
        SimpleNamespace(
            bind_glm52_vq_experts_from_paths=bind_glm52_vq_experts_from_paths
        ),
    )
    return calls, audit


def test_provider_parses_env_config_and_uses_authenticated_accepted_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path)
    monkeypatch.setenv("GLM52_TRAINING_BASELINE_JSON", str(config_path))
    calls, model = _install_fake_runtime(monkeypatch)

    module = importlib.import_module(PROVIDER_MODULE)
    baseline = module.accepted_baseline_provider(SimpleNamespace())

    assert baseline == _FakeBaseline(
        model=model,
        candidate_identity_sha256=IDENTITY_FROM_AUDIT,
    )
    assert [name for name, _value in calls] == ["validate", "load"]
    validation = calls[0][1]
    assert validation["prompt"] == "authenticated prompt"
    assert validation["routed_artifact_dir"] == Path("/artifacts/e8-full-w1")
    assert validation["composite_audit_json"].name == "accepted-composite-audit.json"


def test_provider_fails_closed_when_config_env_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GLM52_TRAINING_BASELINE_JSON", raising=False)
    module = importlib.import_module(PROVIDER_MODULE)

    with pytest.raises(ValueError, match="GLM52_TRAINING_BASELINE_JSON"):
        module.accepted_baseline_provider(SimpleNamespace())


def test_provider_fails_closed_when_required_hash_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload.pop("expected_composite_audit_sha256")
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("GLM52_TRAINING_BASELINE_JSON", str(config_path))
    module = importlib.import_module(PROVIDER_MODULE)

    with pytest.raises(ValueError, match="missing required fields.*expected_composite"):
        module.accepted_baseline_provider(SimpleNamespace())


def test_provider_rejects_composite_audit_hash_mismatch_before_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(
        tmp_path,
        expected_composite_audit_sha256="0" * 64,
    )
    monkeypatch.setenv("GLM52_TRAINING_BASELINE_JSON", str(config_path))
    calls, _model = _install_fake_runtime(monkeypatch)
    module = importlib.import_module(PROVIDER_MODULE)

    with pytest.raises(ValueError, match="composite audit SHA-256"):
        module.accepted_baseline_provider(SimpleNamespace())

    assert calls == []


def test_identity_comes_from_validated_audit_not_caller_or_audit_strings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path)
    monkeypatch.setenv("GLM52_TRAINING_BASELINE_JSON", str(config_path))
    _calls, _model = _install_fake_runtime(monkeypatch)
    module = importlib.import_module(PROVIDER_MODULE)

    baseline = module.accepted_baseline_provider(SimpleNamespace())

    assert baseline.candidate_identity_sha256 == IDENTITY_FROM_AUDIT
    assert baseline.candidate_identity_sha256 != CALLER_IDENTITY


@pytest.mark.parametrize(
    ("report_updates", "match"),
    [
        ({"artifact_identity_sha256": "c" * 64}, "load report identity"),
        ({"bound_sparse_layer_ids": tuple(range(3, 77))}, "layer inventory"),
        ({"dense_routed_parameter_names": ("dense.weight",)}, "dense routed"),
        ({"unbound_vq_experts": True}, "unbound routed"),
    ],
)
def test_provider_rejects_strict_load_inventory_or_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report_updates: dict[str, object],
    match: str,
) -> None:
    config_path = _write_config(tmp_path)
    monkeypatch.setenv("GLM52_TRAINING_BASELINE_JSON", str(config_path))
    _calls, _model = _install_fake_runtime(monkeypatch)
    composite = sys.modules["mlx_vq.models.glm52_composite_loader"]
    report = {
        "artifact_identity_sha256": IDENTITY_FROM_AUDIT,
        "bound_sparse_layer_ids": tuple(range(3, 78)),
        "dense_routed_parameter_names": (),
        "unbound_vq_experts": False,
    }
    report.update(report_updates)
    composite.load_authenticated_glm52_composite = lambda _validated: (
        object(),
        SimpleNamespace(**report),
    )
    module = importlib.import_module(PROVIDER_MODULE)

    with pytest.raises(ValueError, match=match):
        module.accepted_baseline_provider(SimpleNamespace())


def test_provider_is_importable_with_cli_module_function_syntax() -> None:
    module_name, separator, function_name = PROVIDER_SPEC.partition(":")

    assert separator == ":"
    provider = getattr(importlib.import_module(module_name), function_name)
    assert callable(provider)


def test_environment_provider_is_an_importable_module_function_alias() -> None:
    module = importlib.import_module(PROVIDER_MODULE)

    assert module.environment_baseline_provider is module.accepted_baseline_provider


def test_recovery_provider_without_candidate_returns_accepted_composite_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path)
    monkeypatch.setenv("GLM52_TRAINING_BASELINE_JSON", str(config_path))
    accepted_calls, model = _install_fake_runtime(monkeypatch)
    module = importlib.import_module(PROVIDER_MODULE)

    baseline = module.recovery_candidate_baseline_provider()

    assert baseline == _FakeBaseline(
        model=model,
        candidate_identity_sha256=IDENTITY_FROM_AUDIT,
    )
    assert [name for name, _value in accepted_calls] == ["validate", "load"]


def test_recovery_provider_reports_the_audited_recovery_candidate_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["recovery_candidate"] = _recovery_candidate_config(
        tmp_path,
        composite_sha256=payload["expected_composite_audit_sha256"],
    )
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("GLM52_TRAINING_BASELINE_JSON", str(config_path))
    _accepted_calls, model = _install_fake_runtime(monkeypatch)
    recovery_calls, recovery_audit = _install_fake_recovery_runtime(monkeypatch)
    module = importlib.import_module(PROVIDER_MODULE)

    baseline = module.recovery_candidate_baseline_provider(SimpleNamespace())

    assert baseline == _FakeBaseline(
        model=model,
        candidate_identity_sha256=recovery_audit.candidate_identity_sha256,
    )
    assert [name for name, _value in recovery_calls] == [
        "audit_recovery",
        "verify_recovery_identity",
        "recovery_group_paths",
        "bind_recovery",
        "verify_recovery_identity",
    ]


def test_recovery_provider_rejects_candidate_composite_audit_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["recovery_candidate"] = _recovery_candidate_config(
        tmp_path,
        composite_sha256="0" * 64,
    )
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("GLM52_TRAINING_BASELINE_JSON", str(config_path))
    _install_fake_runtime(monkeypatch)
    recovery_calls, _recovery_audit = _install_fake_recovery_runtime(monkeypatch)
    module = importlib.import_module(PROVIDER_MODULE)

    with pytest.raises(ValueError, match="recovery candidate composite audit SHA-256"):
        module.recovery_candidate_baseline_provider()

    assert recovery_calls == []


def test_recovery_candidate_must_be_an_object_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path, recovery_candidate=None)
    monkeypatch.setenv("GLM52_TRAINING_BASELINE_JSON", str(config_path))
    _install_fake_runtime(monkeypatch)
    module = importlib.import_module(PROVIDER_MODULE)

    with pytest.raises(ValueError, match="recovery_candidate must be an object"):
        module.recovery_candidate_baseline_provider()
