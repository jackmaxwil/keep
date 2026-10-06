from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_recovery_module():
    name = "glm52_recovery_candidate_binding_under_test"
    path = (
        Path(__file__).resolve().parents[1]
        / "src/mlx_vq/quality/glm52_recovery.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _candidate_binding_fixture(
    tmp_path: Path,
    *,
    recovery_seeded: bool,
) -> tuple[object, dict[str, Path]]:
    accepted_root = tmp_path / "accepted"
    parent_root = tmp_path / "parent-recovery"
    candidate_root = tmp_path / "candidate"
    artifact_root = candidate_root / "artifact"
    recovered_root = candidate_root / "recovered-groups"
    for root in (
        accepted_root,
        parent_root / "recovered-groups",
        artifact_root,
        recovered_root,
    ):
        root.mkdir(parents=True)

    filenames = (
        "layer-00003-gate_proj.safetensors",
        "layer-00003-up_proj.safetensors",
        "layer-00003-down_proj.safetensors",
    )
    for name in filenames:
        (accepted_root / name).write_bytes(f"accepted:{name}".encode())
        (parent_root / "recovered-groups" / name).write_bytes(
            f"parent:{name}".encode()
        )
    (recovered_root / filenames[1]).write_bytes(
        f"candidate:{filenames[1]}".encode()
    )

    inherited_root = (
        parent_root / "recovered-groups" if recovery_seeded else accepted_root
    )
    expected_paths: dict[str, Path] = {}
    groups = []
    for name in filenames:
        target = (
            recovered_root / name
            if name == filenames[1]
            else inherited_root / name
        )
        (artifact_root / name).symlink_to(os.path.relpath(target, artifact_root))
        expected_paths[name] = target.resolve()
        payload = target.read_bytes()
        projection = name.removeprefix("layer-00003-").removesuffix(".safetensors")
        groups.append(
            SimpleNamespace(
                group_key=f"3:{projection}",
                filename=name,
                classification=(
                    "replacement" if name == filenames[1] else "inherited"
                ),
                artifact_sha256=hashlib.sha256(payload).hexdigest(),
                artifact_bytes=len(payload),
            )
        )

    allowed_roots = {candidate_root.resolve(), accepted_root.resolve()}
    if recovery_seeded:
        allowed_roots.add(parent_root.resolve())
    verification_calls = 0

    def verify_current_identity() -> None:
        nonlocal verification_calls
        verification_calls += 1
        assert {entry.name for entry in artifact_root.iterdir()} == set(filenames)
        for name in filenames:
            resolved = (artifact_root / name).resolve(strict=True)
            assert any(resolved.is_relative_to(root) for root in allowed_roots)

    audit = SimpleNamespace(
        recovery_dir=str(candidate_root),
        groups=tuple(groups),
        verify_current_identity=verify_current_identity,
        verification_calls=lambda: verification_calls,
    )
    return audit, expected_paths


def _resolved_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {
        name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for name, path in paths.items()
    }


def test_classic_candidate_binding_remains_on_accepted_inherited_bytes(
    tmp_path: Path,
) -> None:
    module = _load_recovery_module()
    audit, expected_paths = _candidate_binding_fixture(
        tmp_path,
        recovery_seeded=False,
    )

    paths = module.authenticated_recovery_candidate_group_paths(audit)

    assert paths == expected_paths
    assert _resolved_hashes(paths) == _resolved_hashes(expected_paths)
    assert "accepted" in paths["layer-00003-gate_proj.safetensors"].parts
    assert audit.verification_calls() == 2


def test_recovery_seeded_candidate_binding_uses_parent_inherited_bytes(
    tmp_path: Path,
) -> None:
    module = _load_recovery_module()
    audit, expected_paths = _candidate_binding_fixture(
        tmp_path,
        recovery_seeded=True,
    )

    paths = module.authenticated_recovery_candidate_group_paths(audit)

    assert paths == expected_paths
    assert _resolved_hashes(paths) == _resolved_hashes(expected_paths)
    inherited = paths["layer-00003-gate_proj.safetensors"]
    assert "parent-recovery" in inherited.parts
    assert inherited.read_bytes().startswith(b"parent:")
    assert audit.verification_calls() == 2
