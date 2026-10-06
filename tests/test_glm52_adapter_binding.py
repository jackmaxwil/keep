from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _binding_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    binding = _load_module(
        "_test_glm52_adapter_binding",
        "src/mlx_vq/quality/glm52_adapter_binding.py",
    )
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    manifest = {
        "schema_version": 1,
        "record_type": "glm52_low_rank_adapter_overlay",
        "parent_candidate_identity_sha256": "a" * 64,
        "teacher_manifest_body_sha256": "b" * 64,
        "candidate_identity_sha256": "c" * 64,
        "manifest_body_sha256": "d" * 64,
        "release_eligible": True,
        "routed_projection_inventory": {
            "count": 2,
            "metadata_sha256": "e" * 64,
        },
        "sidecars": [
            {"layer": 3, "projection": "gate_proj"},
            {"layer": 3, "projection": "down_proj"},
        ],
    }
    (adapter_dir / "glm52-low-rank-adapter-manifest.json").write_bytes(
        _canonical_bytes(manifest) + b"\n"
    )
    validated = SimpleNamespace(
        adapter_dir=adapter_dir,
        parent_candidate_identity_sha256="a" * 64,
        teacher_manifest_body_sha256="b" * 64,
        manifest_body_sha256="d" * 64,
        candidate_identity_sha256="c" * 64,
        sidecar_paths={(3, "gate_proj"): object(), (3, "down_proj"): object()},
        release_eligible=True,
    )
    calls: dict[str, object] = {}

    def load(adapter_root: Path, **kwargs: object):
        calls["load"] = (adapter_root, kwargs)
        return validated

    def bind(projection_map: dict[tuple[int, str], object], value: object) -> None:
        calls["bind"] = (projection_map, value)

    training = SimpleNamespace(
        ADAPTER_MANIFEST="glm52-low-rank-adapter-manifest.json",
        _canonical_bytes=_canonical_bytes,
        _manifest_body=lambda value: {
            key: item
            for key, item in value.items()
            if key not in {"manifest_body_sha256", "candidate_identity_sha256"}
        },
        _read_regular_file=lambda path, **_kwargs: Path(path).read_bytes(),
        _require_sha256=lambda value, **_kwargs: (
            None
            if isinstance(value, str) and len(value) == 64
            else (_ for _ in ()).throw(ValueError("invalid SHA-256"))
        ),
        _sha256_bytes=lambda value: hashlib.sha256(value).hexdigest(),
        attest_glm52_projection_inventory=lambda _projection_map: SimpleNamespace(
            projection_count=2,
            projection_metadata_sha256="e" * 64,
        ),
        bind_glm52_low_rank_adapters=bind,
        load_authenticated_glm52_adapter=load,
        target_glm52_projection=lambda model, *, layer, projection: getattr(
            model.model.layers[layer].mlp.switch_mlp, projection
        ),
    )
    monkeypatch.setattr(binding, "_load_adapter_training_api", lambda: training)
    projections = SimpleNamespace(gate_proj=object(), down_proj=object())
    layers = [SimpleNamespace() for _ in range(4)]
    layers[3] = SimpleNamespace(
        mlp=SimpleNamespace(switch_mlp=projections)
    )
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    return binding, adapter_dir, manifest, validated, training, model, calls


def test_bind_authenticated_adapter_uses_external_manifest_and_parent_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, adapter_dir, manifest, validated, _training, model, calls = (
        _binding_fixture(tmp_path, monkeypatch)
    )
    body = {
        key: value
        for key, value in manifest.items()
        if key not in {"manifest_body_sha256", "candidate_identity_sha256"}
    }
    expected_manifest_sha256 = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    manifest["manifest_body_sha256"] = expected_manifest_sha256
    (adapter_dir / "glm52-low-rank-adapter-manifest.json").write_bytes(
        _canonical_bytes(manifest) + b"\n"
    )
    validated.manifest_body_sha256 = expected_manifest_sha256

    result = binding.bind_authenticated_glm52_adapter_sidecars(
        model,
        adapter_dir,
        expected_adapter_manifest_sha256=expected_manifest_sha256,
        expected_parent_candidate_identity_sha256="a" * 64,
        expected_num_experts=256,
    )

    assert result is validated
    assert calls["load"] == (
        adapter_dir,
        {
            "expected_parent_candidate_identity_sha256": "a" * 64,
            "expected_teacher_manifest_body_sha256": "b" * 64,
            "expected_manifest_body_sha256": expected_manifest_sha256,
            "expected_candidate_identity_sha256": "c" * 64,
            "expected_num_experts": 256,
        },
    )
    projection_map, bound = calls["bind"]
    assert set(projection_map) == {(3, "gate_proj"), (3, "down_proj")}
    assert bound is validated


def test_bind_authenticated_adapter_rejects_wrong_parent_before_sidecar_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, adapter_dir, manifest, _validated, _training, model, calls = (
        _binding_fixture(tmp_path, monkeypatch)
    )
    body = {
        key: value
        for key, value in manifest.items()
        if key not in {"manifest_body_sha256", "candidate_identity_sha256"}
    }
    expected_manifest_sha256 = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    manifest["manifest_body_sha256"] = expected_manifest_sha256
    (adapter_dir / "glm52-low-rank-adapter-manifest.json").write_bytes(
        _canonical_bytes(manifest) + b"\n"
    )

    with pytest.raises(ValueError, match="parent candidate identity"):
        binding.bind_authenticated_glm52_adapter_sidecars(
            model,
            adapter_dir,
            expected_adapter_manifest_sha256=expected_manifest_sha256,
            expected_parent_candidate_identity_sha256="f" * 64,
            expected_num_experts=256,
        )

    assert "load" not in calls
    assert "bind" not in calls


def _reevaluate_argv() -> list[str]:
    return [
        "reevaluate",
        "--profile-path", "profile.yaml",
        "--config-path", "config.json",
        "--source-index-path", "index.json",
        "--tokenizer-dir", "tokenizer",
        "--tokenizer-readiness-json", "readiness.json",
        "--family-policy-json", "policy.json",
        "--prompt-pack-json", "prompts.json",
        "--teacher-cache-root", "teacher",
        "--non-vq-artifact-dir", "non-vq",
        "--non-vq-evidence-json", "non-vq.json",
        "--accepted-routed-artifact-dir", "routed",
        "--accepted-composite-audit-json", "composite.json",
        "--accepted-materialization-runs-jsonl", "runs.jsonl",
        "--accepted-full-bind-preflight-json", "preflight.json",
        "--recovery-conversion-dir", "recovery",
        "--output-json", "output.json",
        "--expected-seed-manifest-sha256", "1" * 64,
        "--expected-stats-manifest-sha256", "2" * 64,
        "--expected-full-source-blob-inventory-sha256", "3" * 64,
        "--expected-routed-source-blob-inventory-sha256", "4" * 64,
        "--expected-recovery-lever", "lever",
        "--expected-recovery-policy-json", "recovery-policy.json",
        "--expected-composite-audit-sha256", "5" * 64,
    ]


def test_reevaluate_adapter_cli_flags_are_optional_as_an_exact_pair() -> None:
    cli = _load_module(
        "_test_glm52_recovery_wave1_cli",
        "benchmarks/run_glm52_recovery_wave1.py",
    )
    base = _reevaluate_argv()

    parsed = cli._build_parser().parse_args(base)
    assert parsed.adapter_sidecar_dir is None
    assert parsed.expected_adapter_manifest_sha256 is None
    with pytest.raises(SystemExit):
        cli._build_parser().parse_args(
            [*base, "--adapter-sidecar-dir", "adapter"]
        )
    with pytest.raises(SystemExit):
        cli._build_parser().parse_args(
            [
                *base,
                "--expected-adapter-manifest-sha256",
                "6" * 64,
            ]
        )

    parsed = cli._build_parser().parse_args(
        [
            *base,
            "--adapter-sidecar-dir", "adapter",
            "--expected-adapter-manifest-sha256", "6" * 64,
        ]
    )
    assert parsed.adapter_sidecar_dir == "adapter"
    assert parsed.expected_adapter_manifest_sha256 == "6" * 64


def test_reevaluate_cli_forwards_authenticated_adapter_pair(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_module(
        "_test_glm52_recovery_wave1_forwarding",
        "benchmarks/run_glm52_recovery_wave1.py",
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        cli,
        "reevaluate_recovery_candidate",
        lambda **kwargs: calls.append(kwargs) or {"status": "ok"},
    )

    assert cli.run_cli(
        [
            *_reevaluate_argv(),
            "--adapter-sidecar-dir", "adapter",
            "--expected-adapter-manifest-sha256", "6" * 64,
        ]
    ) == 0

    assert len(calls) == 1
    assert calls[0]["adapter_sidecar_dir"] == "adapter"
    assert calls[0]["expected_adapter_manifest_sha256"] == "6" * 64
    assert json.loads(capsys.readouterr().out) == {"status": "ok"}


def test_optional_adapter_provenance_preserves_no_flag_payload_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recovery = _load_module(
        "_test_glm52_recovery_adapter",
        "src/mlx_vq/quality/glm52_recovery.py",
    )
    payload = {"record_type": "reevaluation", "quality_gate_pass": True}
    before = _canonical_bytes(payload)

    result = recovery.bind_optional_glm52_adapter_sidecars(
        object(),
        adapter_sidecar_dir=None,
        expected_adapter_manifest_sha256=None,
        expected_parent_candidate_identity_sha256="a" * 64,
        expected_num_experts=256,
    )
    recovery.add_glm52_adapter_provenance(payload, result)

    assert _canonical_bytes(payload) == before

    validated = SimpleNamespace(
        adapter_dir=Path("adapter"),
        parent_candidate_identity_sha256="a" * 64,
        teacher_manifest_body_sha256="b" * 64,
        manifest_body_sha256="c" * 64,
        candidate_identity_sha256="d" * 64,
        release_eligible=False,
    )
    binder = ModuleType("mlx_vq.quality.glm52_adapter_binding")
    binder.bind_authenticated_glm52_adapter_sidecars = (
        lambda *_args, **_kwargs: validated
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vq.quality.glm52_adapter_binding",
        binder,
    )
    result = recovery.bind_optional_glm52_adapter_sidecars(
        object(),
        adapter_sidecar_dir="adapter",
        expected_adapter_manifest_sha256="c" * 64,
        expected_parent_candidate_identity_sha256="a" * 64,
        expected_num_experts=256,
    )
    recovery.add_glm52_adapter_provenance(payload, result)

    assert payload["adapter_provenance"] == {
        "adapter_sidecar_dir": "adapter",
        "parent_candidate_identity_sha256": "a" * 64,
        "teacher_manifest_body_sha256": "b" * 64,
        "adapter_manifest_body_sha256": "c" * 64,
        "adapter_candidate_identity_sha256": "d" * 64,
        "release_eligible": False,
    }
