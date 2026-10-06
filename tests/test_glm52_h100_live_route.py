from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from glm52_enforcement.h100_live_route import (
    GuardedQualificationError,
    QualificationAuthorities,
    resolve_qualification_authorities,
    run_guarded_qualification,
)
from mlx_vq.quality.glm52_sky_campaign import build_sky_campaign_descriptor


REPO_ROOT = Path(__file__).parents[1]
ROUTE_SCRIPT = (
    REPO_ROOT
    / "aws/glm52-gpu/scripts/run_h100_qualification_route.py"
)
TASK13_SHELL = (
    REPO_ROOT
    / "aws/glm52-gpu/scripts/run_h100_qualification_campaign.sh"
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
        + b"\n"
    )


def _valid_descriptor() -> dict[str, object]:
    return build_sky_campaign_descriptor(
        run_id="glm52-sky-20260724",
        must_start_by="2026-07-29T23:00:00Z",
        controller_identity=(
            "arn:aws:iam::246813579024:role/keep-glm52-controller"
        ),
        worker_identity=(
            "arn:aws:iam::246813579024:role/keep-glm52-worker"
        ),
        vpc_name="keep-glm52",
        image_id="ami-0123456789abcdef0",
        bucket="keep-glm52-models-246813579024-us-west-2",
        jobs_bucket="keep-glm52-models-246813579024-us-west-2",
        repo_tar_key=(
            "campaigns/glm52-sky-20260724/repository/"
            "keep-repository.tar.gz"
        ),
        repo_tar_sha256="1" * 64,
        campaign_descriptor_key=(
            "campaigns/glm52-sky-20260724/submissions/post-seed/"
            "campaign-descriptor-v2.json"
        ),
        approval_key=(
            "campaigns/glm52-sky-20260724/authorities/"
            "GPU_SPEND_APPROVAL.json"
        ),
        approval_sha256=hashlib.sha256(b'{"approved":true}\n').hexdigest(),
        artifacts={
            "source_snapshot_prefix": "source-snapshot/",
            "source_snapshot_sha256": "2" * 64,
            "non_vq_prefix": "non-vq-package/",
            "non_vq_package_sha256": "3" * 64,
            "teich_pack_key": (
                "teich-pack/glm52-sky-20260724/teich-pack.json"
            ),
            "teich_pack_sha256": "4" * 64,
            "frozen_prompt_pack_key": (
                "quality/glm52-sky-20260724/frozen-66.json"
            ),
            "frozen_prompt_pack_sha256": "5" * 64,
            "training_baseline_prefix": "training-baseline/",
            "training_baseline_sha256": "6" * 64,
            "training_config_key": (
                "campaigns/glm52-sky-20260724/authorities/training.json"
            ),
            "training_config_sha256": "7" * 64,
            "artifact_inventory_key": (
                "campaigns/glm52-sky-20260724/inventories/"
                "artifact-inventory-"
                + "8" * 64
                + ".json"
            ),
            "artifact_inventory_sha256": "8" * 64,
            "qualification_cache_prefix": (
                "qualification-cache/seeds/glm52-sky-20260724/"
                + "9" * 64
                + "/"
            ),
            "qualification_cache_manifest_sha256": "9" * 64,
        },
    )


def _authorities(tmp_path: Path) -> QualificationAuthorities:
    names = {
        "descriptor": "campaign-descriptor-v2.json",
        "seed_descriptor": "seed-campaign-descriptor-v2.json",
        "approval": "GPU_SPEND_APPROVAL.json",
        "staged_ready": "STAGED_CONTROL_PLANE_READY.json",
        "rehearsal_evidence": "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json",
        "cache_seed_accepted": "QUALIFICATION_CACHE_SEED_ACCEPTED.json",
        "gpu_spend_snapshot": "GPU_SPEND_SNAPSHOT.json",
        "qualification_ready": "QUALIFICATION_SUBMISSION_READY.json",
        "task": "glm52-campaign.yaml",
        "config": "skypilot-config.yaml",
    }
    paths: dict[str, Path] = {}
    for field, name in names.items():
        path = tmp_path / name
        path.write_text("{}\n")
        paths[field] = path
    paths["descriptor"].write_bytes(_canonical(_valid_descriptor()))
    return QualificationAuthorities(**paths)


def _handoff(authorities: QualificationAuthorities) -> dict[str, object]:
    descriptor = json.loads(authorities.descriptor.read_bytes())
    descriptor_raw = authorities.descriptor.read_bytes()
    bucket = str(descriptor["bucket"])
    run_id = str(descriptor["run_id"])
    intent_body = "c" * 64
    baseline_body = "e" * 64
    ready_body = "0" * 64
    body = {
        "schema_version": 1,
        "record_type": "glm52_sky_submission_handoff_v1",
        "account_id": "246813579024",
        "region": "us-west-2",
        "bucket": bucket,
        "run_id": run_id,
        "managed_mode": "qualification",
        "descriptor_s3_uri": (
            f"s3://{bucket}/{descriptor['campaign_descriptor_key']}"
        ),
        "descriptor_file_sha256": hashlib.sha256(descriptor_raw).hexdigest(),
        "intent_s3_uri": (
            f"s3://{bucket}/campaigns/{run_id}/submissions/qualification/"
            "intents/"
            + intent_body
            + "/SKYPILOT_SUBMISSION_INTENT.json"
        ),
        "intent_file_sha256": "b" * 64,
        "intent_body_sha256": intent_body,
        "controller_baseline_s3_uri": (
            f"s3://{bucket}/campaigns/{run_id}/qualification/"
            "controller-baselines/"
            + baseline_body
            + "/CONTROLLER_BASELINE.json"
        ),
        "controller_baseline_file_sha256": "d" * 64,
        "controller_baseline_body_sha256": baseline_body,
        "must_start_ready_s3_uri": (
            f"s3://{bucket}/campaigns/{run_id}/monitor/must-start/"
            "qualification/"
            + intent_body
            + "/control-plane-ready/"
            + ready_body
            + "/CONTROL_PLANE_READY.json"
        ),
        "must_start_ready_file_sha256": "f" * 64,
        "must_start_ready_body_sha256": ready_body,
        "requires_separate_active_deployment": True,
        "created_at": "2026-07-29T12:00:00Z",
    }
    body["handoff_body_sha256"] = hashlib.sha256(
        _canonical(body)[:-1]
    ).hexdigest()
    return body


def _rehash_handoff(value: dict[str, object]) -> None:
    body = dict(value)
    body.pop("handoff_body_sha256", None)
    value["handoff_body_sha256"] = hashlib.sha256(
        _canonical(body)[:-1]
    ).hexdigest()


def test_guarded_route_prepares_then_launches_and_retries_only_75(
    tmp_path: Path,
) -> None:
    """Break caught: H100 driver observes markers but never submits a job."""

    authorities = _authorities(tmp_path)
    submitter = REPO_ROOT / "aws/glm52-gpu/scripts/submit_sky_campaign.sh"
    sky_bin = tmp_path / "sky"
    sky_bin.write_text("#!/usr/bin/env bash\n")
    sky_bin.chmod(0o755)
    calls: list[list[str]] = []
    returncodes = iter((0, 75, 0))

    def runner(
        argv: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if argv[2] == "prepare-intent":
            output = Path(argv[argv.index("--output-handoff") + 1])
            output.write_bytes(_canonical(_handoff(authorities)))
        return subprocess.CompletedProcess(
            argv,
            next(returncodes),
            stdout="",
            stderr="",
        )

    result = run_guarded_qualification(
        authorities=authorities,
        submitter=submitter,
        sky_bin=sky_bin,
        profile="keep-gpu",
        work_dir=tmp_path / "route-work",
        runner=runner,
    )

    assert result["status"] == "accepted"
    assert [call[:3] for call in calls] == [
        [str(submitter), "--qualification", "prepare-intent"],
        [str(submitter), "--qualification", "acquire-and-launch"],
        [str(submitter), "--qualification", "acquire-and-launch"],
    ]
    assert calls[0].count("--output-handoff") == 1
    for call in calls[1:]:
        assert "--output-handoff" not in call
        assert call[call.index("--intent-s3-uri") + 1] == (
            _handoff(authorities)["intent_s3_uri"]
        )
        assert call[call.index("--must-start-ready-s3-uri") + 1] == (
            _handoff(authorities)["must_start_ready_s3_uri"]
        )
    rendered = "\n".join(" ".join(call) for call in calls)
    assert "terminate-instances" not in rendered
    assert "run-instances" not in rendered
    assert "jobs launch" not in rendered


def test_guarded_route_refuses_deployment_required_before_launch(
    tmp_path: Path,
) -> None:
    """Break caught: missing supervisor deployment is treated as launchable."""

    authorities = _authorities(tmp_path)
    submitter = REPO_ROOT / "aws/glm52-gpu/scripts/submit_sky_campaign.sh"
    sky_bin = tmp_path / "sky"
    sky_bin.write_text("#!/usr/bin/env bash\n")
    sky_bin.chmod(0o755)
    calls: list[list[str]] = []

    def runner(
        argv: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv,
            78,
            stdout="",
            stderr="deployment-required",
        )

    with pytest.raises(
        GuardedQualificationError,
        match="deployed qualification supervisor",
    ):
        run_guarded_qualification(
            authorities=authorities,
            submitter=submitter,
            sky_bin=sky_bin,
            profile="keep-gpu",
            work_dir=tmp_path / "route-work",
            runner=runner,
        )
    assert len(calls) == 1


def test_guarded_route_refuses_foreign_handoff_before_acquisition(
    tmp_path: Path,
) -> None:
    """Break caught: a foreign prepared intent is passed to the submitter."""

    authorities = _authorities(tmp_path)
    submitter = REPO_ROOT / "aws/glm52-gpu/scripts/submit_sky_campaign.sh"
    sky_bin = tmp_path / "sky"
    sky_bin.write_text("#!/usr/bin/env bash\n")
    sky_bin.chmod(0o755)
    calls: list[list[str]] = []

    def runner(
        argv: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        handoff = _handoff(authorities)
        handoff["bucket"] = "foreign-bucket"
        handoff["descriptor_s3_uri"] = (
            "s3://foreign-bucket/"
            + str(_valid_descriptor()["campaign_descriptor_key"])
        )
        _rehash_handoff(handoff)
        output = Path(argv[argv.index("--output-handoff") + 1])
        output.write_bytes(_canonical(handoff))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="",
            stderr="",
        )

    with pytest.raises(GuardedQualificationError, match="handoff"):
        run_guarded_qualification(
            authorities=authorities,
            submitter=submitter,
            sky_bin=sky_bin,
            profile="keep-gpu",
            work_dir=tmp_path / "route-work",
            runner=runner,
        )
    assert len(calls) == 1


def test_guarded_route_refuses_fake_submitter_before_any_call(
    tmp_path: Path,
) -> None:
    authorities = _authorities(tmp_path)
    submitter = tmp_path / "submit_sky_campaign.sh"
    submitter.write_text("#!/usr/bin/env bash\nexit 0\n")
    submitter.chmod(0o755)
    sky_bin = tmp_path / "sky"
    sky_bin.write_text("#!/usr/bin/env bash\n")
    sky_bin.chmod(0o755)
    calls: list[list[str]] = []

    with pytest.raises(GuardedQualificationError, match="pinned repository"):
        run_guarded_qualification(
            authorities=authorities,
            submitter=submitter,
            sky_bin=sky_bin,
            profile="keep-gpu",
            work_dir=tmp_path / "route-work",
            runner=lambda argv, **_kwargs: calls.append(argv),
        )
    assert calls == []


def test_resolver_materializes_exact_reviewed_authority_chain(
    tmp_path: Path,
) -> None:
    """Break caught: the live driver has no way to supply submitter pins."""

    workspace = tmp_path / "full-run"
    post_seed = workspace / "post-seed"
    authorities_dir = workspace / "reviewed"
    post_seed.mkdir(parents=True)
    authorities_dir.mkdir()
    approval_raw = b'{"approved":true}\n'
    descriptor = build_sky_campaign_descriptor(
        run_id="glm52-sky-20260724",
        must_start_by="2026-07-29T23:00:00Z",
        controller_identity=(
            "arn:aws:iam::246813579024:role/keep-glm52-controller"
        ),
        worker_identity=(
            "arn:aws:iam::246813579024:role/keep-glm52-worker"
        ),
        vpc_name="keep-glm52",
        image_id="ami-0123456789abcdef0",
        bucket="keep-glm52-models-246813579024-us-west-2",
        jobs_bucket="keep-glm52-models-246813579024-us-west-2",
        repo_tar_key=(
            "campaigns/glm52-sky-20260724/repository/"
            "keep-repository.tar.gz"
        ),
        repo_tar_sha256="1" * 64,
        campaign_descriptor_key=(
            "campaigns/glm52-sky-20260724/submissions/post-seed/"
            "campaign-descriptor-v2.json"
        ),
        approval_key=(
            "campaigns/glm52-sky-20260724/authorities/"
            "GPU_SPEND_APPROVAL.json"
        ),
        approval_sha256=hashlib.sha256(approval_raw).hexdigest(),
        artifacts={
            "source_snapshot_prefix": "source-snapshot/",
            "source_snapshot_sha256": "2" * 64,
            "non_vq_prefix": "non-vq-package/",
            "non_vq_package_sha256": "3" * 64,
            "teich_pack_key": (
                "teich-pack/glm52-sky-20260724/teich-pack.json"
            ),
            "teich_pack_sha256": "4" * 64,
            "frozen_prompt_pack_key": (
                "quality/glm52-sky-20260724/frozen-66.json"
            ),
            "frozen_prompt_pack_sha256": "5" * 64,
            "training_baseline_prefix": "training-baseline/",
            "training_baseline_sha256": "6" * 64,
            "training_config_key": (
                "campaigns/glm52-sky-20260724/authorities/training.json"
            ),
            "training_config_sha256": "7" * 64,
            "artifact_inventory_key": (
                "campaigns/glm52-sky-20260724/inventories/"
                "artifact-inventory-"
                + "8" * 64
                + ".json"
            ),
            "artifact_inventory_sha256": "8" * 64,
            "qualification_cache_prefix": (
                "qualification-cache/seeds/glm52-sky-20260724/"
                + "9" * 64
                + "/"
            ),
            "qualification_cache_manifest_sha256": "9" * 64,
        },
    )
    descriptor_raw = _canonical(descriptor)
    descriptor_path = post_seed / "campaign-descriptor-v2.json"
    descriptor_path.write_bytes(descriptor_raw)
    (authorities_dir / "GPU_SPEND_APPROVAL.json").write_bytes(approval_raw)
    seed_descriptor = {"record_type": "seed", "run_id": descriptor["run_id"]}
    seed_raw = _canonical(seed_descriptor)
    records = {
        "STAGED_CONTROL_PLANE_READY.json": {"record_type": "staged"},
        "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json": {
            "record_type": "rehearsal"
        },
        "QUALIFICATION_CACHE_SEED_ACCEPTED.json": {
            "record_type": "accepted"
        },
        "GPU_SPEND_SNAPSHOT.json": {"record_type": "snapshot"},
    }
    raws: dict[str, bytes] = {}
    for name, value in records.items():
        raw = _canonical(value)
        raws[name] = raw
        (authorities_dir / name).write_bytes(raw)
    readiness = {
        "schema_version": 1,
        "record_type": "glm52_qualification_submission_ready_v1",
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": descriptor["run_id"],
        "managed_mode": "qualification",
        "descriptor_key": descriptor["campaign_descriptor_key"],
        "descriptor_file_sha256": hashlib.sha256(descriptor_raw).hexdigest(),
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "approval_sha256": descriptor["approval_sha256"],
        "seed_descriptor": seed_descriptor,
        "seed_descriptor_file_sha256": hashlib.sha256(seed_raw).hexdigest(),
        "staged_readiness_file_sha256": hashlib.sha256(
            raws["STAGED_CONTROL_PLANE_READY.json"]
        ).hexdigest(),
        "cache_seed_acceptance_file_sha256": hashlib.sha256(
            raws["QUALIFICATION_CACHE_SEED_ACCEPTED.json"]
        ).hexdigest(),
        "gpu_spend_snapshot_sha256": hashlib.sha256(
            raws["GPU_SPEND_SNAPSHOT.json"]
        ).hexdigest(),
        "rehearsal_evidence_sha256": hashlib.sha256(
            raws["GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"]
        ).hexdigest(),
    }
    readiness_path = authorities_dir / "QUALIFICATION_SUBMISSION_READY.json"
    readiness_path.write_bytes(_canonical(readiness))
    config = workspace / "skypilot-config.yaml"
    config.write_text("aws:\n  use_internal_ips: true\n")
    repo = Path(__file__).parents[1]

    resolved = resolve_qualification_authorities(
        descriptor=descriptor_path,
        config=config,
        repo_root=repo,
        workspace=workspace,
        materialization_dir=tmp_path / "materialized",
    )

    assert resolved.descriptor == descriptor_path.resolve()
    assert resolved.qualification_ready == readiness_path.resolve()
    assert resolved.seed_descriptor.read_bytes() == seed_raw
    assert resolved.approval.read_bytes() == approval_raw
    assert resolved.task == (
        repo / "aws/glm52-gpu/skypilot/glm52-campaign.yaml"
    ).resolve()


def test_resolver_refuses_readiness_not_bound_to_descriptor(
    tmp_path: Path,
) -> None:
    descriptor = tmp_path / "campaign-descriptor-v2.json"
    descriptor.write_text("{}\n")
    config = tmp_path / "config.yaml"
    config.write_text("{}\n")
    readiness = tmp_path / "QUALIFICATION_SUBMISSION_READY.json"
    readiness.write_bytes(
        _canonical(
            {
                "schema_version": 1,
                "record_type": "glm52_qualification_submission_ready_v1",
                "descriptor_file_sha256": "0" * 64,
            }
        )
    )

    with pytest.raises(
        GuardedQualificationError,
        match="campaign descriptor",
    ):
        resolve_qualification_authorities(
            descriptor=descriptor,
            config=config,
            repo_root=Path(__file__).parents[1],
            workspace=tmp_path,
            materialization_dir=tmp_path / "materialized",
        )


def test_cli_connects_resolver_to_guarded_route_without_effect_bypass(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Break caught: tested route exists but the Task 13 shell cannot call it."""

    spec = importlib.util.spec_from_file_location("_h100_route_cli", ROUTE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    authorities = _authorities(tmp_path)
    events: list[tuple[str, dict[str, object]]] = []

    def resolver(**kwargs: object) -> QualificationAuthorities:
        events.append(("resolve", kwargs))
        return authorities

    def route(**kwargs: object) -> dict[str, object]:
        events.append(("route", kwargs))
        return {"status": "accepted", "attempts": 1}

    paths = {
        name: tmp_path / name
        for name in ("descriptor.json", "config.yaml", "sky", "submitter")
    }
    for path in paths.values():
        path.write_text("{}\n")
    work_root = tmp_path / "work"

    assert (
        module.main(
            [
                "--descriptor",
                str(paths["descriptor.json"]),
                "--config",
                str(paths["config.yaml"]),
                "--workspace",
                str(tmp_path),
                "--sky-bin",
                str(paths["sky"]),
                "--submitter",
                str(paths["submitter"]),
                "--work-root",
                str(work_root),
                "--profile",
                "keep-gpu",
            ],
            resolver=resolver,
            route=route,
        )
        == 0
    )
    assert [name for name, _kwargs in events] == ["resolve", "route"]
    assert events[1][1]["authorities"] is authorities
    assert events[1][1]["submitter"] == paths["submitter"]
    assert events[1][1]["sky_bin"] == paths["sky"]
    assert json.loads(capsys.readouterr().out)["status"] == "accepted"


def test_task13_shell_wires_guarded_originator_before_marker_observation() -> None:
    """Break caught: Task 13 still executes the qualification-only observer."""

    source = TASK13_SHELL.read_text()
    originator = '"$SCRIPT_DIR/run_h100_qualification_route.py"'
    assert originator in source
    assert '"$SCRIPT_DIR/submit_sky_campaign.sh"' in source
    assert source.index(originator) < source.index("DEADLINE=")
    assert "terminate-instances" not in source
    assert "run-instances" not in source
    result = subprocess.run(
        ["bash", "-n", str(TASK13_SHELL)],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
