from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from mlx_vq.recovery_campaign import ProjectionRates, load_campaign_config
from mlx_vq.io.schema import codebook_metadata_for_bits


RECIPE = Path("recipes/glm52_recovery_campaign_v1_20260711.yaml")
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
E8_TENSOR_PAYLOAD_BYTES = 272_499_712
E8P_TENSOR_PAYLOAD_BYTES = 536_740_864
NON_ROUTED_PAYLOAD_BYTES = 37_121_488_608
SOURCE_LINEAGE = {
    "source_model_id": "0xSero/glm-5.2-reap-504B-v2",
    "source_revision": "6c9241aa05fb243a0edb7c804c213ec1cf5c920d",
    "source_config_sha256": "5fa690755d0dab25a8e0e5e0745675bdac03ba2b6f5641da2931278235c71f1b",
    "source_index_sha256": "bb5b4fa9782aea5ffc66f9145d6e630f1045d385c30437c531bebe422c075f3f",
    "source_profile": "glm52-reap-504b-v2",
}
RECOVERY_POLICY = {
    "decoded_expert_working_set": "one_per_worker",
    "recovery_lever": "selection_diagonal_hessian_importance_weighted_reround_v1",
    "rounding_objective": "selection_diagonal_hessian_weighted_squared_error",
    "scale_estimator": "importance_weighted_least_squares",
    "source_config_sha256": SOURCE_LINEAGE["source_config_sha256"],
    "source_decoder": "modelopt_nvfp4_v1",
    "source_index_sha256": SOURCE_LINEAGE["source_index_sha256"],
    "source_model_id": SOURCE_LINEAGE["source_model_id"],
    "source_profile": SOURCE_LINEAGE["source_profile"],
    "source_revision": SOURCE_LINEAGE["source_revision"],
    "source_weight_encoding": "modelopt_nvfp4",
}


def _observe(
    repo_root: Path,
    *,
    ps_output: str = "",
    argv_by_pid: dict[int, tuple[str, ...] | None] | None = None,
    lock_openers: tuple[int, ...] = (),
    use_real_openers: bool = False,
):
    from mlx_vq.recovery_campaign.observer import observe_campaign

    if argv_by_pid is None:
        argv_by_pid = {}
        for line in ps_output.splitlines():
            fields = line.strip().split(None, 2)
            if len(fields) == 3:
                argv_by_pid[int(fields[0])] = tuple(shlex.split(fields[2]))
    patches = [
        patch(
            "mlx_vq.recovery_campaign.observer._read_process_argv",
            side_effect=lambda pid: argv_by_pid.get(pid),
        )
    ]
    if not use_real_openers:
        patches.append(
            patch(
                "mlx_vq.recovery_campaign.observer._read_lock_openers",
                return_value=lock_openers,
            )
        )
    with patches[0]:
        if len(patches) == 2:
            with patches[1]:
                return observe_campaign(
                    load_campaign_config(RECIPE),
                    repo_root=repo_root,
                    ps_output=ps_output,
                )
        return observe_campaign(
            load_campaign_config(RECIPE),
            repo_root=repo_root,
            ps_output=ps_output,
        )


def _output(repo_root: Path, experiment_name: str = "full75-e8") -> Path:
    config = load_campaign_config(RECIPE)
    return repo_root / config.experiment(experiment_name).output_path


def _write_groups(repo_root: Path, names: list[str]) -> None:
    groups = _output(repo_root) / "recovered-groups"
    groups.mkdir(parents=True, exist_ok=True)
    for name in names:
        (groups / name).write_bytes(b"group")


def _write_manifest(
    repo_root: Path,
    *,
    experiment_name: str = "full75-e8",
    schema_version: int = 2,
) -> tuple[str, str]:
    config = load_campaign_config(RECIPE)
    experiment = config.experiment(experiment_name)
    authorities = {authority.name: authority for authority in config.authorities}
    path = _output(repo_root, experiment_name) / "conversion-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    source_verification = {
        "routed_source_blob_inventory_sha256": config.routed_source_blob_inventory_sha256,
        "shard_count": 61,
        "source_blob_inventory_sha256": config.full_source_blob_inventory_sha256,
    }
    provenance = {
        "holdout_used_for_tuning": False,
        "lever": "selection_diagonal_hessian_importance_weighted_reround_v1",
        "report_used_for_tuning": False,
        "rounding_objective": "selection_diagonal_hessian_weighted_squared_error",
        "scale_estimator": "importance_weighted_least_squares",
        "source_lineage": SOURCE_LINEAGE,
        "source_verification": source_verification,
        "stats_manifest_sha256": authorities["recovery-stats"].sha256,
    }
    upgraded_layers = {
        rates.layer
        for rates in experiment.projection_rates
        if rates.values == (16, 16, 16)
    }

    def record(layer: int, projection: str) -> dict[str, object]:
        bits = 16 if layer in upgraded_layers else 8
        value: dict[str, object] = {
            "artifact_bytes": 5,
            "artifact_path": _canonical_group(layer, projection),
            "artifact_sha256": "a" * 64,
            "layer": layer,
            "lever_provenance": provenance,
            "projection": projection,
            "recovery_policy": RECOVERY_POLICY,
            "source_lineage": SOURCE_LINEAGE,
            "status": "materialized",
        }
        if schema_version == 2:
            codebook_name, codebook_sha256 = codebook_metadata_for_bits(bits)
            value.update(
                {
                    "code_bits": bits,
                    "codebook_name": codebook_name,
                    "codebook_sha256": codebook_sha256,
                    "codes_dtype": "uint16" if bits == 16 else "uint8",
                    "tensor_payload_bytes": (
                        E8P_TENSOR_PAYLOAD_BYTES
                        if bits == 16
                        else E8_TENSOR_PAYLOAD_BYTES
                    ),
                    "zero_importance_policy": (
                        "source_rtn_e8p"
                        if bits == 16
                        else "preserve_seed_expert_bytes"
                    ),
                }
            )
        return value

    logical_payload = experiment.expected_payload_bytes
    accounting: dict[str, object] = {
        "incremental_disk_bytes": len(experiment.recovered_layers) * 3 * 5,
        "logical_payload_limit_bytes": config.budget.payload_limit_bytes,
        "logical_whole_model_tensor_payload_bytes": logical_payload,
    }
    if schema_version == 1:
        accounting["non_routed_tensor_payload_bytes"] = NON_ROUTED_PAYLOAD_BYTES
    else:
        accounting.update(
            {
                "main_non_routed_tensor_payload_bytes": NON_ROUTED_PAYLOAD_BYTES,
                "routed_codebook_bytes": 225 * 1024,
                "routed_codes_scales_bytes": (
                    logical_payload - NON_ROUTED_PAYLOAD_BYTES - 225 * 1024
                ),
            }
        )
    seed_dir = repo_root / Path(authorities["accepted-seed-manifest"].path).parent
    body = {
        "accounting": accounting,
        "accepted_composite_audit_sha256": authorities["accepted-composite-audit"].sha256,
        "expected_full_source_blob_inventory_sha256": config.full_source_blob_inventory_sha256,
        "expected_routed_source_blob_inventory_sha256": config.routed_source_blob_inventory_sha256,
        "groups": [
            record(layer, projection)
            for layer in experiment.recovered_layers
            for projection in PROJECTIONS
        ],
        "lever_provenance": provenance,
        "mixed_artifact": {
            "inherited_group_count": 225 - len(experiment.recovered_layers) * 3,
            "output_dir": str((_output(repo_root, experiment_name) / "artifact").resolve()),
            "recovered_groups_dir": str(
                (_output(repo_root, experiment_name) / "recovered-groups").resolve()
            ),
            "replacement_count": len(experiment.recovered_layers) * 3,
            "seed_artifact_dir": str(seed_dir.absolute()),
            "seed_manifest_sha256": authorities["accepted-seed-manifest"].sha256,
        },
        "record_type": "glm52_recovery_conversion_manifest",
        "recovery_policy": RECOVERY_POLICY,
        "resumable": True,
        "schema_version": schema_version,
        "selected_groups": [
            f"{layer}:{projection}"
            for layer in experiment.recovered_layers
            for projection in PROJECTIONS
        ],
        "stats_manifest_sha256": authorities["recovery-stats"].sha256,
        "status": "complete",
        "source_lineage": SOURCE_LINEAGE,
        "source_verification": source_verification,
    }
    if schema_version == 2:
        body["rate_policy"] = {
            "complete_layer_rates_required": True,
            "default_code_bits": 8,
            "layer_code_bits": {
                str(rates.layer): 16
                for rates in experiment.projection_rates
                if rates.values == (16, 16, 16)
            },
        }
    body_sha = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    manifest = {**body, "manifest_body_sha256": body_sha}
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), body_sha


def _rehash_manifest(path: Path, mutate) -> None:
    manifest = json.loads(path.read_text())
    mutate(manifest)
    manifest.pop("manifest_body_sha256", None)
    manifest["manifest_body_sha256"] = hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n")


def _write_complete_physical(
    repo_root: Path,
    *,
    experiment_name: str = "full75-e8",
) -> None:
    config = load_campaign_config(RECIPE)
    experiment = config.experiment(experiment_name)
    selected = {
        _canonical_group(layer, projection)
        for layer in experiment.recovered_layers
        for projection in PROJECTIONS
    }
    output = _output(repo_root, experiment_name)
    recovered = output / "recovered-groups"
    artifact = output / "artifact"
    recovered.mkdir(parents=True, exist_ok=True)
    artifact.mkdir(parents=True, exist_ok=True)
    seed = repo_root / Path(
        next(
            authority.path
            for authority in config.authorities
            if authority.name == "accepted-seed-manifest"
        )
    ).parent
    for name in sorted(selected):
        (recovered / name).write_bytes(b"group")
    for layer in range(3, 78):
        for projection in PROJECTIONS:
            name = _canonical_group(layer, projection)
            target = (
                Path("../recovered-groups") / name
                if name in selected
                else Path(os.path.relpath(seed / name, artifact))
            )
            (artifact / name).symlink_to(target)


def _canonical_group(layer: int, projection: str = "gate_proj") -> str:
    return f"layer-{layer:05d}-{projection}.safetensors"


def _snapshot(root: Path) -> tuple[tuple[str, int, int, int, int], ...]:
    if not root.exists():
        return ()
    return tuple(
        (
            str(path.relative_to(root)),
            path.lstat().st_mode,
            path.lstat().st_ino,
            path.lstat().st_size,
            path.lstat().st_mtime_ns,
        )
        for path in sorted(root.rglob("*"))
    )


def test_observer_does_not_create_missing_lock_output_or_ledger_paths(tmp_path: Path) -> None:
    before = _snapshot(tmp_path)

    observation = _observe(tmp_path)

    assert observation.lock_state == "absent"
    assert observation.recovered_groups_state == "absent"
    assert observation.artifact_links_state == "absent"
    assert observation.recovered_groups == 0
    assert observation.manifest_state == "absent"
    assert _snapshot(tmp_path) == before == ()


def test_observer_reports_real_flock_absent_free_and_held(tmp_path: Path) -> None:
    assert _observe(tmp_path).lock_state == "absent"

    lock = tmp_path / ".keep-heavy-job.lock"
    lock.touch()
    assert _observe(tmp_path).lock_state == "free"

    with lock.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        observation = _observe(tmp_path)
        assert observation.lock_state == "held"
        assert observation.lock_held is True
        assert observation.lock_owner_known is False


def test_observer_rejects_symlink_lock_without_following_it(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.write_text("do not follow")
    (tmp_path / ".keep-heavy-job.lock").symlink_to(outside)

    observation = _observe(tmp_path)

    assert observation.lock_state == "invalid"
    assert any("lock" in item and "invalid" in item for item in observation.contradictions)


@pytest.mark.parametrize(
    ("elapsed", "seconds"),
    (("05:30", 330.0), ("02:05:30", 7_530.0), ("3-02:05:30", 266_730.0)),
)
def test_observer_matches_exact_declared_process_and_parses_elapsed(
    tmp_path: Path, elapsed: str, seconds: float
) -> None:
    config = load_campaign_config(RECIPE)
    transition = config.transition("full75-rematerialize")
    lock = tmp_path / ".keep-heavy-job.lock"
    lock.touch()
    with lock.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        observation = _observe(
            tmp_path,
            ps_output=f"  4321 {elapsed} {' '.join(transition.argv)}\n",
            lock_openers=(4321,),
        )

    assert observation.experiment_name == "full75-e8"
    assert observation.active_processes[0].pid == 4321
    assert observation.active_processes[0].elapsed_seconds == seconds
    assert observation.lock_owner_known is True


def test_observer_does_not_accept_a_process_with_only_a_similar_argv(tmp_path: Path) -> None:
    config = load_campaign_config(RECIPE)
    argv = list(config.transition("full75-rematerialize").argv)
    argv[-1] = "a-different-lock"

    observation = _observe(tmp_path, ps_output=f"123 00:05 {' '.join(argv)}\n")

    assert observation.active_processes == ()


def test_observer_counts_only_canonical_regular_recovered_groups(tmp_path: Path) -> None:
    _write_groups(
        tmp_path,
        [
            _canonical_group(77, "gate_proj"),
            _canonical_group(77, "up_proj"),
            "layer-00077-not-a-projection.safetensors",
        ],
    )
    canonical_symlink = (
        _output(tmp_path) / "recovered-groups" / _canonical_group(77, "down_proj")
    )
    canonical_symlink.symlink_to("missing")

    observation = _observe(tmp_path)

    assert observation.recovered_groups == 2
    assert observation.expected_groups == 225
    assert any("noncanonical recovered-group" in item for item in observation.contradictions)
    assert any("not a regular file" in item for item in observation.contradictions)


def test_observer_counts_canonical_artifact_tree_symlinks_separately(tmp_path: Path) -> None:
    artifact = _output(tmp_path) / "artifact"
    artifact.mkdir(parents=True)
    for layer in range(3, 78):
        for projection in PROJECTIONS:
            (artifact / _canonical_group(layer, projection)).symlink_to("../recovered-groups/missing")
    (artifact / "notes.txt").write_text("not a routed link")

    observation = _observe(tmp_path)

    assert observation.artifact_links == 225
    assert observation.expected_artifact_links == 225


def test_observer_authenticates_manifest_body_and_reports_file_identity(tmp_path: Path) -> None:
    expected_file_sha, expected_body_sha = _write_manifest(tmp_path)
    transition = load_campaign_config(RECIPE).transition("full75-rematerialize")

    observation = _observe(
        tmp_path,
        ps_output=f"42 00:10 {' '.join(transition.argv)}\n",
    )

    assert observation.manifest_state == "valid"
    assert observation.manifest_file_sha256 == expected_file_sha
    assert observation.manifest_body_sha256 == expected_body_sha
    assert observation.manifest_sha256 == expected_file_sha
    assert any("valid manifest" in item and "recovered groups" in item for item in observation.contradictions)
    assert any("valid manifest" in item and "artifact links" in item for item in observation.contradictions)


def test_observer_accepts_schema_v1_only_for_uniform_e8_experiment(
    tmp_path: Path,
) -> None:
    _write_manifest(tmp_path, schema_version=1)
    transition = load_campaign_config(RECIPE).transition("full75-rematerialize")

    full75 = _observe(
        tmp_path,
        ps_output=f"42 00:10 {' '.join(transition.argv)}\n",
    )

    assert full75.manifest_state == "valid"

    _write_manifest(tmp_path, experiment_name="worst8-e8p", schema_version=1)
    worst8_transition = load_campaign_config(RECIPE).transition(
        "worst8-e8p-rematerialize"
    )
    worst8 = _observe(
        tmp_path,
        ps_output=f"43 00:10 {' '.join(worst8_transition.argv)}\n",
    )

    assert worst8.manifest_state == "invalid"
    assert any("schema" in item for item in worst8.contradictions)


def test_observer_rejects_self_hashed_manifest_for_the_wrong_experiment(
    tmp_path: Path,
) -> None:
    _write_manifest(tmp_path)
    path = _output(tmp_path) / "conversion-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["selected_groups"] = ["77:gate_proj"]
    manifest.pop("manifest_body_sha256")
    manifest["manifest_body_sha256"] = hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

    observation = _observe(tmp_path)

    assert observation.manifest_state == "invalid"
    assert any("selected groups" in item for item in observation.contradictions)


def test_observer_rejects_manifest_group_without_canonical_identity(
    tmp_path: Path,
) -> None:
    _write_manifest(tmp_path)
    path = _output(tmp_path) / "conversion-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["groups"][0].pop("artifact_sha256")
    manifest.pop("manifest_body_sha256")
    manifest["manifest_body_sha256"] = hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    transition = load_campaign_config(RECIPE).transition("full75-rematerialize")

    observation = _observe(
        tmp_path,
        ps_output=f"42 00:10 {' '.join(transition.argv)}\n",
    )

    assert observation.manifest_state == "invalid"
    assert any("group" in item and "exact fields" in item for item in observation.contradictions)


@pytest.mark.parametrize(
    "payload",
    (
        b'{"schema_version":2,"schema_version":2,"manifest_body_sha256":"' + b"0" * 64 + b'"}',
        b'{"schema_version":NaN,"manifest_body_sha256":"' + b"0" * 64 + b'"}',
        b'{"schema_version":2,"manifest_body_sha256":"' + b"0" * 64 + b'"}',
    ),
)
def test_observer_rejects_duplicate_nonfinite_or_body_mismatched_manifest(
    tmp_path: Path, payload: bytes
) -> None:
    manifest = _output(tmp_path) / "conversion-manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(payload)

    observation = _observe(tmp_path)

    assert observation.manifest_state == "invalid"
    assert observation.manifest_file_sha256 == hashlib.sha256(payload).hexdigest()
    assert observation.manifest_body_sha256 is None
    assert any("manifest" in item and "invalid" in item for item in observation.contradictions)


def test_observer_rejects_manifest_symlink_without_reading_target(tmp_path: Path) -> None:
    outside = tmp_path / "outside-manifest.json"
    outside.write_text('{"secret":"must not be read"}\n')
    manifest = _output(tmp_path) / "conversion-manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.symlink_to(outside)

    observation = _observe(tmp_path)

    assert observation.manifest_state == "invalid"
    assert observation.manifest_file_sha256 is None
    assert any("manifest" in item and "invalid" in item for item in observation.contradictions)


def test_observer_selects_first_executable_experiment_without_valid_manifest(
    tmp_path: Path,
) -> None:
    _write_manifest(tmp_path, experiment_name="full75-e8")

    observation = _observe(tmp_path)

    assert observation.experiment_name == "worst8-e8p"
    assert observation.expected_groups == 24
    assert any(
        "full75-e8" in item and "valid manifest" in item
        for item in observation.contradictions
    )


def test_observer_derives_throughput_and_eta_only_from_exact_active_process(
    tmp_path: Path,
) -> None:
    config = load_campaign_config(RECIPE)
    transition = config.transition("full75-rematerialize")
    _write_groups(tmp_path, [_canonical_group(77, projection) for projection in PROJECTIONS])
    lock = tmp_path / ".keep-heavy-job.lock"
    lock.touch()
    with lock.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        observation = _observe(
            tmp_path,
            ps_output=f"98 01:00 {' '.join(transition.argv)}\n",
            lock_openers=(98,),
        )

    assert observation.throughput_groups_per_second == pytest.approx(0.05)
    assert observation.eta_seconds == pytest.approx(4_440.0)

    inactive = _observe(tmp_path)
    assert inactive.throughput_groups_per_second is None
    assert inactive.eta_seconds is None


def test_observer_reports_required_contradictions(tmp_path: Path) -> None:
    config = load_campaign_config(RECIPE)
    transition = config.transition("full75-rematerialize")
    _write_groups(
        tmp_path,
        [
            _canonical_group(77, projection)
            for projection in PROJECTIONS
        ]
        + [_canonical_group(2)],
    )
    process = f"{' '.join(transition.argv)}"
    (tmp_path / ".keep-heavy-job.lock").touch()

    observation = _observe(
        tmp_path,
        ps_output=f"11 00:10 {process}\n12 00:11 {process}\n",
    )

    assert any("more than one matching heavy process" in item for item in observation.contradictions)
    assert any("active process" in item and "free lock" in item for item in observation.contradictions)
    assert observation.experiment_name is None


def test_observer_reports_recovered_count_above_declared_experiment(
    tmp_path: Path,
) -> None:
    config = load_campaign_config(RECIPE)
    experiment = config.experiment("worst8-e8p")
    transition = config.transition("worst8-e8p-rematerialize")
    groups = _output(tmp_path, experiment.name) / "recovered-groups"
    groups.mkdir(parents=True)
    for layer in experiment.recovered_layers:
        for projection in PROJECTIONS:
            (groups / _canonical_group(layer, projection)).write_bytes(b"group")
    (groups / _canonical_group(69)).write_bytes(b"extra")

    observation = _observe(
        tmp_path,
        ps_output=f"99 00:10 {' '.join(transition.argv)}\n",
    )

    assert observation.recovered_groups == 25
    assert observation.expected_groups == 24
    assert any("exceed expected" in item for item in observation.contradictions)


def test_lock_owner_requires_exact_pid_in_injected_inode_opener_evidence(
    tmp_path: Path,
) -> None:
    config = load_campaign_config(RECIPE)
    transition = config.transition("full75-rematerialize")
    lock = tmp_path / ".keep-heavy-job.lock"
    lock.touch()
    with lock.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        owned = _observe(
            tmp_path,
            ps_output=f"4321 00:10 displayed text is not authoritative\n",
            argv_by_pid={4321: transition.argv},
            lock_openers=(4321,),
        )
        other = _observe(
            tmp_path,
            ps_output=f"4321 00:10 displayed text is not authoritative\n",
            argv_by_pid={4321: transition.argv},
            lock_openers=(9876,),
        )

    assert owned.lock_owner_known is True
    assert other.lock_owner_known is False
    assert any("different opener" in item for item in other.contradictions)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS lsof/flock proof")
def test_real_flock_holder_does_not_transfer_ownership_to_different_matching_pid(
    tmp_path: Path,
) -> None:
    from mlx_vq.recovery_campaign.observer import observe_campaign

    config = load_campaign_config(RECIPE)
    transition = config.transition("full75-rematerialize")
    lock = tmp_path / ".keep-heavy-job.lock"
    lock.touch()
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl,sys,time; "
                "h=open(sys.argv[1],'rb'); fcntl.flock(h,fcntl.LOCK_EX); "
                "print('ready',flush=True); time.sleep(30)"
            ),
            str(lock),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None and child.stdout.readline().strip() == "ready"
        injected_pid = os.getpid()
        with patch(
            "mlx_vq.recovery_campaign.observer._read_process_argv",
            return_value=transition.argv,
        ):
            observation = observe_campaign(
                config,
                repo_root=tmp_path,
                ps_output=f"{injected_pid} 00:10 spoof\n",
            )
    finally:
        child.terminate()
        child.wait(timeout=5)

    assert observation.lock_state == "held"
    assert observation.lock_owner_known is False
    assert child.pid != injected_pid
    assert any("different opener" in item for item in observation.contradictions)


def test_ps_display_is_not_used_as_actual_argv_and_boundaries_are_exact(
    tmp_path: Path,
) -> None:
    transition = load_campaign_config(RECIPE).transition("full75-rematerialize")
    displayed_exact_actual_wrong = _observe(
        tmp_path,
        ps_output=f"10 00:10 {' '.join(transition.argv)}\n",
        argv_by_pid={10: ("not", "the", "declared", "argv")},
    )
    displayed_spoof_actual_exact = _observe(
        tmp_path,
        ps_output="11 00:10 arbitrary spoofed display\n",
        argv_by_pid={11: transition.argv},
    )
    merged = transition.argv[:-2] + (" ".join(transition.argv[-2:]),)
    boundary_drift = _observe(
        tmp_path,
        ps_output="12 00:10 arbitrary display\n",
        argv_by_pid={12: merged},
    )

    assert displayed_exact_actual_wrong.active_processes == ()
    assert tuple(process.pid for process in displayed_spoof_actual_exact.active_processes) == (11,)
    assert boundary_drift.active_processes == ()


def test_ps_two_field_pid_elapsed_candidate_uses_actual_argv(tmp_path: Path) -> None:
    transition = load_campaign_config(RECIPE).transition("full75-rematerialize")

    observation = _observe(
        tmp_path,
        ps_output="11 00:10\n",
        argv_by_pid={11: transition.argv},
    )

    assert tuple(process.pid for process in observation.active_processes) == (11,)
    assert observation.active_processes[0].elapsed_seconds == 10.0


def test_multiple_actual_argv_matches_are_sorted_and_select_no_experiment(
    tmp_path: Path,
) -> None:
    transition = load_campaign_config(RECIPE).transition("full75-rematerialize")

    observation = _observe(
        tmp_path,
        ps_output="22 00:20 second\n11 00:10 first\n",
        argv_by_pid={22: transition.argv, 11: transition.argv},
    )

    assert tuple(process.pid for process in observation.active_processes) == (11, 22)
    assert observation.experiment_name is None
    assert observation.throughput_groups_per_second is None
    assert observation.eta_seconds is None
    assert any("more than one matching heavy process" in item for item in observation.contradictions)


@pytest.mark.parametrize(
    ("experiment_name", "mutate", "message"),
    (
        ("full75-e8", lambda value: value.update({"extra": True}), "exact fields"),
        (
            "full75-e8",
            lambda value: value["source_lineage"].update({"source_model_id": "wrong"}),
            "source lineage",
        ),
        (
            "full75-e8",
            lambda value: value["source_verification"].update({"shard_count": 0}),
            "shard count",
        ),
        (
            "full75-e8",
            lambda value: value["recovery_policy"].update({"extra": "drift"}),
            "recovery policy",
        ),
        (
            "full75-e8",
            lambda value: value["lever_provenance"].update(
                {"holdout_used_for_tuning": True}
            ),
            "provenance",
        ),
        (
            "full75-e8",
            lambda value: value["accounting"].update(
                {"logical_whole_model_tensor_payload_bytes": 1}
            ),
            "accounting",
        ),
        (
            "full75-e8",
            lambda value: value["mixed_artifact"].update(
                {"seed_artifact_dir": "/tmp/wrong-seed"}
            ),
            "seed_artifact_dir",
        ),
        (
            "full75-e8",
            lambda value: value["groups"][0]["source_lineage"].update(
                {"source_revision": "wrong"}
            ),
            "group source lineage",
        ),
        (
            "full75-e8",
            lambda value: value["groups"][0]["recovery_policy"].update(
                {"extra": "drift"}
            ),
            "group recovery policy",
        ),
        (
            "full75-e8",
            lambda value: value["groups"][0]["lever_provenance"].update(
                {"report_used_for_tuning": True}
            ),
            "group provenance",
        ),
        (
            "worst8-e8p",
            lambda value: value["groups"][0].update(
                {
                    "code_bits": 16,
                    "codes_dtype": "uint16",
                    "tensor_payload_bytes": E8_TENSOR_PAYLOAD_BYTES,
                }
            ),
            "tensor_payload_bytes",
        ),
        (
            "worst8-e8p",
            lambda value: value["groups"][0].update({"extra": True}),
            "group.*exact fields",
        ),
    ),
)
def test_self_rehashed_manifest_contract_drift_is_rejected(
    tmp_path: Path,
    experiment_name: str,
    mutate,
    message: str,
) -> None:
    _write_manifest(tmp_path, experiment_name=experiment_name)
    path = _output(tmp_path, experiment_name) / "conversion-manifest.json"
    _rehash_manifest(path, mutate)
    transition = load_campaign_config(RECIPE).transition(
        "full75-rematerialize"
        if experiment_name == "full75-e8"
        else "worst8-e8p-rematerialize"
    )

    observation = _observe(
        tmp_path,
        ps_output=f"42 00:10 candidate\n",
        argv_by_pid={42: transition.argv},
    )

    assert observation.manifest_state == "invalid"
    assert any(__import__("re").search(message, item) for item in observation.contradictions)


def test_self_rehashed_consistent_source_lineage_hash_substitution_is_rejected(
    tmp_path: Path,
) -> None:
    _write_manifest(tmp_path)
    path = _output(tmp_path) / "conversion-manifest.json"
    replacement_config = "1" * 64
    replacement_index = "2" * 64

    def mutate(value: dict[str, object]) -> None:
        lineages = [
            value["source_lineage"],
            value["lever_provenance"]["source_lineage"],
        ]
        policies = [value["recovery_policy"]]
        for record in value["groups"]:
            lineages.extend(
                [record["source_lineage"], record["lever_provenance"]["source_lineage"]]
            )
            policies.append(record["recovery_policy"])
        for lineage in lineages:
            lineage["source_config_sha256"] = replacement_config
            lineage["source_index_sha256"] = replacement_index
        for policy in policies:
            policy["source_config_sha256"] = replacement_config
            policy["source_index_sha256"] = replacement_index

    _rehash_manifest(path, mutate)
    transition = load_campaign_config(RECIPE).transition("full75-rematerialize")

    observation = _observe(
        tmp_path,
        ps_output="42 00:10\n",
        argv_by_pid={42: transition.argv},
    )

    assert observation.manifest_state == "invalid"
    assert any("source lineage" in item for item in observation.contradictions)


def test_schema_v1_accepts_explicit_uniform_e8_rates(tmp_path: Path) -> None:
    from mlx_vq.recovery_campaign.observer import observe_campaign

    config = load_campaign_config(RECIPE)
    full75 = config.experiment("full75-e8")
    explicit = replace(
        full75,
        projection_rates=tuple(
            ProjectionRates(layer=layer, gate=8, up=8, down=8)
            for layer in full75.recovered_layers
        ),
    )
    config = replace(
        config,
        experiments=tuple(
            explicit if experiment.name == explicit.name else experiment
            for experiment in config.experiments
        ),
    )
    _write_manifest(tmp_path, schema_version=1)
    transition = config.transition("full75-rematerialize")
    with (
        patch(
            "mlx_vq.recovery_campaign.observer._read_process_argv",
            return_value=transition.argv,
        ),
        patch(
            "mlx_vq.recovery_campaign.observer._read_lock_openers",
            return_value=(),
        ),
    ):
        observation = observe_campaign(
            config,
            repo_root=tmp_path,
            ps_output="42 00:10 candidate\n",
        )

    assert observation.manifest_state == "valid"


def test_active_experiment_requires_complete_valid_dependencies(tmp_path: Path) -> None:
    config = load_campaign_config(RECIPE)
    transition = config.transition("worst8-e8p-rematerialize")

    absent = _observe(
        tmp_path,
        ps_output="42 00:10 candidate\n",
        argv_by_pid={42: transition.argv},
    )
    _write_manifest(tmp_path, experiment_name="full75-e8", schema_version=1)
    incomplete = _observe(
        tmp_path,
        ps_output="42 00:10 candidate\n",
        argv_by_pid={42: transition.argv},
    )

    assert any("dependency full75-e8" in item and "absent" in item for item in absent.contradictions)
    assert any(
        "dependency full75-e8" in item and "incomplete" in item
        for item in incomplete.contradictions
    )


@pytest.mark.parametrize("attack", ("extra", "regular", "wrong_target"))
def test_valid_manifest_requires_exact_artifact_symlink_inventory(
    tmp_path: Path,
    attack: str,
) -> None:
    _write_manifest(tmp_path, schema_version=1)
    _write_complete_physical(tmp_path)
    artifact = _output(tmp_path) / "artifact"
    if attack == "extra":
        (artifact / "extra.txt").write_text("extra")
    else:
        link = artifact / _canonical_group(77)
        link.unlink()
        if attack == "regular":
            link.write_bytes(b"not a symlink")
        else:
            link.symlink_to("../recovered-groups/not-the-same-name.safetensors")
    transition = load_campaign_config(RECIPE).transition("full75-rematerialize")

    observation = _observe(
        tmp_path,
        ps_output="42 00:10 candidate\n",
        argv_by_pid={42: transition.argv},
    )

    assert observation.manifest_state == "valid"
    assert observation.contradictions
    assert any("artifact" in item for item in observation.contradictions)


def test_human_render_distinguishes_invalid_directory_states(tmp_path: Path) -> None:
    from mlx_vq.recovery_campaign.render import render_status

    output = _output(tmp_path)
    output.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (output / "recovered-groups").symlink_to(outside, target_is_directory=True)
    (output / "artifact").symlink_to(outside, target_is_directory=True)

    observation = _observe(tmp_path)
    rendered = render_status(observation)

    assert observation.recovered_groups_state == "invalid"
    assert observation.artifact_links_state == "invalid"
    assert "Recovered groups: invalid (expected 225)" in rendered
    assert "Artifact links: invalid (expected 225)" in rendered


def test_scan_time_race_becomes_invalid_directory_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from mlx_vq.recovery_campaign.observer import _scan_directory

    directory = tmp_path / "present"
    directory.mkdir()

    def race(_descriptor):
        raise OSError("directory changed during scan")

    monkeypatch.setattr(os, "scandir", race)

    entries, state, error = _scan_directory(tmp_path, Path("present"))

    assert entries == []
    assert state == "invalid"
    assert error is not None and "changed during scan" in error
