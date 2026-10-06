from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from glm52_enforcement import records as RECORDS
from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256
from glm52_enforcement.task10_worker import (
    GRACEFUL_SCRIPT_NAMES,
    build_graceful_stop_evidence,
    build_worker_bootstrap_descriptor,
    render_worker_units,
    worker_unit_hashes,
)
from mlx_vq.quality.glm52_sky_campaign import (
    APPROVED_GPU_COST_USD,
    APPROVED_GPU_RUNTIME_SECONDS,
    APPROVED_HOURLY_COST_USD,
    GpuSpendLedger,
    SkyCampaignLedger,
)
from mlx_vq.quality.glm52_teich_campaign import CampaignPhase

ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
BUCKET = "keep-glm52-models-246813579024-us-west-2"
GENERATION = 2
GENERATION_TEXT = "00000002"
GENERATION_KEY = (
    f"campaigns/{RUN_ID}/submissions/production/generations/"
    f"{GENERATION_TEXT}/terminal/CAMPAIGN_DRAINED.json"
)
ROOT_MARKER_KEY = f"campaigns/{RUN_ID}/CAMPAIGN_DRAINED.json"
CAMPAIGN_LEDGER_KEY = f"campaigns/{RUN_ID}/ledger/campaign-ledger.jsonl"
SPEND_LEDGER_KEY = f"campaigns/{RUN_ID}/runtime/GPU_SPEND_LEDGER.jsonl"
TASK10_DESCRIPTOR_KEY = "task13/production/task10-worker-descriptor.json"
TASK9_EVIDENCE_KEY = (
    f"campaigns/{RUN_ID}/submissions/production/generations/"
    f"{GENERATION_TEXT}/terminal-evidence/TASK9_TERMINAL_EVIDENCE.json"
)
GRACEFUL_STOP_KEY = (
    f"campaigns/{RUN_ID}/submissions/production/generations/"
    f"{GENERATION_TEXT}/allocations/00000001/WORKER_GRACEFUL_STOP.json"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = REPO_ROOT / "benchmarks/run_glm52_campaign.py"
STACK = REPO_ROOT / "aws/glm52-gpu/cfn/gpu-teacher-stack.yaml"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _metadata(label: str) -> dict[str, object]:
    return {
        "HTTPStatusCode": 200,
        "RequestId": label,
        "RetryAttempts": 0,
    }


def _checksum(raw: bytes) -> str:
    return base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")


def _load_record_fixtures() -> object:
    path = Path(__file__).with_name("test_glm52_enforcement_records.py")
    spec = importlib.util.spec_from_file_location(
        "_campaign_drained_record_fixtures", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RECORD_FIXTURES = _load_record_fixtures()


def _terminal_v2() -> dict[str, object]:
    value = RECORD_FIXTURES._terminal_for_outcome(
        RECORDS, "DRAINED_COMPLETED", "ONE"
    )
    value.update(
        activation_id="activation-0002",
        activation_ordinal=2,
        generation=GENERATION,
        generation_text=GENERATION_TEXT,
    )
    for liability in value["worker_launch_liabilities"]:
        liability.update(
            activation_id="activation-0002",
            activation_ordinal=2,
            generation=GENERATION,
        )
        liability["canonical_entry_sha256"] = canonical_sha256(
            {
                field: item
                for field, item in liability.items()
                if field != "canonical_entry_sha256"
            }
        )
    value["worker_launch_liabilities_array_sha256"] = canonical_sha256(
        value["worker_launch_liabilities"]
    )
    value["canonical_body_sha256"] = canonical_sha256(
        {
            field: item
            for field, item in value.items()
            if field != "canonical_body_sha256"
        }
    )
    return RECORDS.validate_record("glm52_production_terminal_v2", value)


def _task9_evidence(terminal: dict[str, object]) -> dict[str, object]:
    copied = (
        "handoff",
        "binding",
        "final_sky_state",
        "final_ec2_states",
        "request_cardinality",
        "allocations",
        "worker_launch_evidence",
        "worker_launch_liabilities",
        "final_heartbeat_identity",
        "checkpoint_identity",
        "cache_identity",
        "training_identity",
        "evaluation_identity",
        "drain_identity",
        "request_evidence",
        "post_terminal_quiescence_evidence",
        "prior_terminal_v1_identity",
        "outcome",
        "operator_disposition_required",
    )
    body = {
        "schema_version": 1,
        "record_type": "glm52_task9_terminal_evidence_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "campaign_identity_sha256": terminal[
            "campaign_identity_sha256"
        ],
        "activation_id": "activation-0002",
        "activation_ordinal": 2,
        "generation": GENERATION,
        "generation_text": GENERATION_TEXT,
        "source_family_identity_sha256": _sha("terminal-families"),
        **{name: terminal[name] for name in copied},
        "evidence_created_at": "2026-07-29T11:59:00Z",
    }
    return {**body, "canonical_body_sha256": canonical_sha256(body)}


def _terminal_fixture(tmp_path: Path) -> SimpleNamespace:
    execution_deadline = "2026-07-31T12:00:00Z"
    gpu_allocation_sha256 = _sha("gpu-allocation")
    spend_path = tmp_path / "runtime/GPU_SPEND_LEDGER.jsonl"
    spend = GpuSpendLedger(
        spend_path,
        run_id=RUN_ID,
        approval_sha256=_sha("approval"),
        approved_gpu_runtime_seconds=APPROVED_GPU_RUNTIME_SECONDS,
        approved_gpu_cost_usd=APPROVED_GPU_COST_USD,
        hourly_cost_usd=APPROVED_HOURLY_COST_USD,
    )
    launched = datetime(2026, 7, 29, 8, 0, tzinfo=UTC)
    spend.start_allocation(
        job_id=RUN_ID,
        instance_id="i-0123456789abcdef0",
        launched_at=launched,
    )
    spend.end_allocation(
        instance_id="i-0123456789abcdef0",
        ended_at=launched + timedelta(hours=1),
    )
    descriptor = build_worker_bootstrap_descriptor(
        schema_version=1,
        record_type="glm52_task10_worker_bootstrap_descriptor_v1",
        account_id=ACCOUNT_ID,
        region=REGION,
        run_id=RUN_ID,
        campaign_identity_sha256=_sha("campaign"),
        activation_id="activation-0002",
        activation_ordinal=2,
        generation=GENERATION,
        generation_text=GENERATION_TEXT,
        action_key=f"ACTION#{GENERATION_TEXT}#SKY_POST#00000001",
        sky_job_name=RUN_ID,
        execution_deadline=execution_deadline,
        gpu_allocation_sha256=gpu_allocation_sha256,
        base_descriptor_s3_uri=(
            f"s3://{BUCKET}/campaigns/{RUN_ID}/"
            "submissions/production/campaign-descriptor-v2.json"
        ),
        base_descriptor_version_id="campaign-version-0001",
        base_descriptor_file_sha256=_sha("campaign-file"),
        base_descriptor_body_sha256=_sha("campaign-body"),
        archive_identity_sha256=_sha("archive"),
        repository_archive_version_id="archive-version-0001",
        approval_identity_sha256=_sha("approval-identity"),
        approval_version_id="approval-version-0001",
        intent_identity_sha256=_sha("intent"),
        intent_version_id="intent-version-0001",
        task8_live_h1d_identity_sha256=_sha("h1d"),
        task8_spend_authority_identity_sha256=spend.genesis_sha256,
        task9_launch_identity_sha256=_sha("launch"),
        task9_admission_identity_sha256=_sha("admission"),
        task9_custody_identity_sha256=_sha("custody"),
    )
    descriptor_raw = canonical_json_bytes(asdict(descriptor)) + b"\n"
    campaign_path = tmp_path / "ledger/campaign-ledger.jsonl"
    campaign = SkyCampaignLedger(
        campaign_path,
        run_id=RUN_ID,
        execution_deadline=execution_deadline,
        gpu_spend_authority_sha256=spend.genesis_sha256,
    )
    start = datetime(2026, 7, 29, 8, 0, tzinfo=UTC)
    prior = None
    for index, phase in enumerate(tuple(CampaignPhase)[:-1]):
        prior = campaign.transition(
            phase,
            input_identities={"input": _sha(f"input-{index}")},
            output_identities={"output": _sha(f"output-{index}")},
            gpu_allocation_sha256=gpu_allocation_sha256,
            timestamp=start + timedelta(minutes=index),
        )
    assert prior is not None
    marker = {
        "record_type": "glm52_sky_campaign_drained_v2",
        "run_id": RUN_ID,
        "outcome": "completed",
        "last_completed_phase": "EVALUATION",
        "drain_reason": "campaign_complete",
        "prior_record_sha256": prior.record_sha256,
        "teacher_cache_ready_sha256": _sha("teacher-cache-ready"),
        "automatic_model_promotion": False,
        "model_uploaded": False,
        "execution_deadline": execution_deadline,
        "gpu_allocation_sha256": gpu_allocation_sha256,
    }
    marker_raw = canonical_json_bytes(marker) + b"\n"
    campaign.transition(
        CampaignPhase.DRAINED,
        input_identities={"evaluation": prior.record_sha256},
        output_identities={
            "drained": hashlib.sha256(marker_raw).hexdigest()
        },
        gpu_allocation_sha256=gpu_allocation_sha256,
        timestamp=start + timedelta(hours=1),
    )
    terminal = _terminal_v2()
    terminal["campaign_identity_sha256"] = descriptor.campaign_identity_sha256
    terminal["canonical_body_sha256"] = canonical_sha256(
        {
            field: item
            for field, item in terminal.items()
            if field != "canonical_body_sha256"
        }
    )
    terminal = RECORDS.validate_record(
        "glm52_production_terminal_v2", terminal
    )
    evidence = _task9_evidence(terminal)
    unit_hashes = worker_unit_hashes(dict(render_worker_units()))
    script_hashes = {
        name: _sha(name) for name in GRACEFUL_SCRIPT_NAMES
    }
    graceful = build_graceful_stop_evidence(
        unit_hashes=unit_hashes,
        script_hashes=script_hashes,
        active_state="inactive",
        sub_state="dead",
        result="success",
        exec_main_code=1,
        exec_main_status=0,
        control_group_pids=[],
        stop_file_identity_sha256=_sha("stop"),
        checkpoint_identity_sha256=_sha("checkpoint"),
        latest_marker_identity_sha256=_sha("ledger-latest"),
        campaign_terminal_marker_identity_sha256=hashlib.sha256(
            marker_raw
        ).hexdigest(),
        ssm_command_id="11111111-2222-3333-4444-555555555555",
        authority="SSM",
    )
    return SimpleNamespace(
        descriptor=descriptor,
        descriptor_raw=descriptor_raw,
        marker_raw=marker_raw,
        campaign_raw=campaign_path.read_bytes(),
        spend_raw=spend_path.read_bytes(),
        terminal=terminal,
        evidence_raw=canonical_json_bytes(evidence) + b"\n",
        graceful_raw=canonical_json_bytes(graceful) + b"\n",
    )


class ExactS3:
    def __init__(self, *, lose_put_response: bool = False) -> None:
        self.versions: dict[str, list[dict[str, object]]] = {}
        self.delete_markers: list[dict[str, object]] = []
        self.put_requests: list[dict[str, object]] = []
        self.list_requests: list[dict[str, object]] = []
        self.get_requests: list[dict[str, object]] = []
        self.lose_put_response = lose_put_response
        self._version = 0

    def seed(
        self, key: str, raw: bytes, *, metadata: dict[str, str] | None = None
    ) -> str:
        self._version += 1
        version_id = f"version-{self._version:04d}"
        for row in self.versions.get(key, []):
            row["IsLatest"] = False
        self.versions.setdefault(key, []).append(
            {
                "VersionId": version_id,
                "raw": raw,
                "Metadata": dict(metadata or {}),
                "IsLatest": True,
            }
        )
        return version_id

    def list_object_versions(self, **request: object) -> dict[str, object]:
        self.list_requests.append(dict(request))
        prefix = str(request["Prefix"])
        return {
            "Versions": [
                {
                    "Key": key,
                    "VersionId": row["VersionId"],
                    "Size": len(row["raw"]),
                    "IsLatest": row["IsLatest"],
                }
                for key, rows in self.versions.items()
                if key == prefix
                for row in rows
            ],
            "DeleteMarkers": [
                row
                for row in self.delete_markers
                if row.get("Key") == prefix
            ],
            "IsTruncated": False,
            "ResponseMetadata": _metadata(
                f"list-{len(self.list_requests)}"
            ),
        }

    def get_object(self, **request: object) -> dict[str, object]:
        self.get_requests.append(dict(request))
        key = str(request["Key"])
        version_id = str(request["VersionId"])
        row = next(
            item
            for item in self.versions[key]
            if item["VersionId"] == version_id
        )
        raw = bytes(row["raw"])
        return {
            "Body": io.BytesIO(raw),
            "ContentLength": len(raw),
            "ContentType": "application/json",
            "ChecksumType": "FULL_OBJECT",
            "VersionId": version_id,
            "ChecksumSHA256": _checksum(raw),
            "ETag": '"'
            + hashlib.md5(raw, usedforsecurity=False).hexdigest()
            + '"',
            "Metadata": dict(row["Metadata"]),
            "ResponseMetadata": _metadata(
                f"get-{len(self.get_requests)}"
            ),
        }

    def put_object(self, **request: object) -> dict[str, object]:
        self.put_requests.append(dict(request))
        key = str(request["Key"])
        if self.versions.get(key):
            raise RuntimeError("PreconditionFailed")
        raw = bytes(request["Body"])
        version_id = self.seed(
            key, raw, metadata=dict(request["Metadata"])
        )
        response = {
            "VersionId": version_id,
            "ChecksumSHA256": request["ChecksumSHA256"],
            "ResponseMetadata": _metadata("put"),
        }
        if self.lose_put_response:
            self.lose_put_response = False
            raise TimeoutError("lost PutObject response")
        return response


def _seed_sources(
    fixture: SimpleNamespace, s3: ExactS3
) -> dict[str, object]:
    versions = {
        ROOT_MARKER_KEY: s3.seed(ROOT_MARKER_KEY, fixture.marker_raw),
        CAMPAIGN_LEDGER_KEY: s3.seed(
            CAMPAIGN_LEDGER_KEY, fixture.campaign_raw
        ),
        SPEND_LEDGER_KEY: s3.seed(SPEND_LEDGER_KEY, fixture.spend_raw),
        TASK10_DESCRIPTOR_KEY: s3.seed(
            TASK10_DESCRIPTOR_KEY, fixture.descriptor_raw
        ),
        TASK9_EVIDENCE_KEY: s3.seed(
            TASK9_EVIDENCE_KEY, fixture.evidence_raw
        ),
        GRACEFUL_STOP_KEY: s3.seed(
            GRACEFUL_STOP_KEY, fixture.graceful_raw
        ),
    }
    return {
        "bucket": BUCKET,
        "key": TASK10_DESCRIPTOR_KEY,
        "version_id": versions[TASK10_DESCRIPTOR_KEY],
        "file_sha256": hashlib.sha256(
            fixture.descriptor_raw
        ).hexdigest(),
        "body_sha256": fixture.descriptor.descriptor_body_sha256,
    }


def _publish(
    fixture: SimpleNamespace,
    s3: ExactS3,
    coordinate: dict[str, object],
) -> dict[str, object]:
    from glm52_enforcement.campaign_drained_publication import (
        CampaignDrainedPublicationServices,
        publish_campaign_drained,
    )

    return publish_campaign_drained(
        activation_id="activation-0002",
        activation_ordinal=2,
        generation=GENERATION,
        generation_text=GENERATION_TEXT,
        terminal_v2=fixture.terminal,
        task10_worker_descriptor_coordinate=coordinate,
        services=CampaignDrainedPublicationServices(
            s3=s3, total_max_attempts=1
        ),
    )


def test_retained_generation_two_publication_exact_reads_all_ancestry(
    tmp_path: Path,
) -> None:
    fixture = _terminal_fixture(tmp_path)
    s3 = ExactS3()
    coordinate = _seed_sources(fixture, s3)

    result = _publish(fixture, s3, coordinate)

    assert result["key"] == GENERATION_KEY
    assert result["generation"] == 2
    assert result["generation_text"] == "00000002"
    assert len(s3.put_requests) == 1
    assert {
        request["Key"] for request in s3.get_requests[:-1]
    } == {
        ROOT_MARKER_KEY,
        CAMPAIGN_LEDGER_KEY,
        SPEND_LEDGER_KEY,
        TASK10_DESCRIPTOR_KEY,
        TASK9_EVIDENCE_KEY,
        GRACEFUL_STOP_KEY,
    }
    put = s3.put_requests[0]
    assert put["Key"] == GENERATION_KEY
    assert put["Body"] == fixture.marker_raw
    assert put["IfNoneMatch"] == "*"


def test_lost_response_and_restart_adopt_without_second_put(
    tmp_path: Path,
) -> None:
    fixture = _terminal_fixture(tmp_path)
    s3 = ExactS3(lose_put_response=True)
    coordinate = _seed_sources(fixture, s3)

    first = _publish(fixture, s3, coordinate)
    second = _publish(fixture, s3, coordinate)

    assert second == first
    assert len(s3.put_requests) == 1


def test_open_spend_ledger_fails_before_destination_inventory(
    tmp_path: Path,
) -> None:
    fixture = _terminal_fixture(tmp_path)
    lines = fixture.spend_raw.splitlines(keepends=True)
    fixture.spend_raw = b"".join(lines[:-1])
    s3 = ExactS3()
    coordinate = _seed_sources(fixture, s3)

    with pytest.raises(ValueError, match="open|ended"):
        _publish(fixture, s3, coordinate)

    assert not any(
        request["Prefix"] == GENERATION_KEY
        for request in s3.list_requests
    )
    assert s3.put_requests == []


@pytest.mark.parametrize(
    ("key", "match"),
    [
        (TASK10_DESCRIPTOR_KEY, "descriptor.*history|descriptor.*singular"),
        (ROOT_MARKER_KEY, "marker.*history|marker.*singular"),
        (CAMPAIGN_LEDGER_KEY, "ledger.*history|ledger.*singular"),
        (SPEND_LEDGER_KEY, "spend.*history|spend.*singular"),
    ],
)
def test_rewritten_or_ambiguous_source_history_fails_closed(
    tmp_path: Path, key: str, match: str
) -> None:
    fixture = _terminal_fixture(tmp_path)
    s3 = ExactS3()
    coordinate = _seed_sources(fixture, s3)
    raw = bytes(s3.versions[key][-1]["raw"])
    s3.seed(key, raw)

    with pytest.raises(ValueError, match=match):
        _publish(fixture, s3, coordinate)

    assert s3.put_requests == []


def test_descriptor_generation_substitution_fails_before_root_reads(
    tmp_path: Path,
) -> None:
    fixture = _terminal_fixture(tmp_path)
    value = json.loads(fixture.descriptor_raw)
    value["generation"] = 3
    value["generation_text"] = "00000003"
    fixture.descriptor_raw = canonical_json_bytes(value) + b"\n"
    s3 = ExactS3()
    coordinate = _seed_sources(fixture, s3)

    with pytest.raises(ValueError, match="descriptor|self-hash|generation"):
        _publish(fixture, s3, coordinate)

    assert s3.put_requests == []


def test_task9_terminal_evidence_rewrite_or_drift_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = _terminal_fixture(tmp_path)
    value = json.loads(fixture.evidence_raw)
    value["generation"] = 3
    value["canonical_body_sha256"] = canonical_sha256(
        {
            name: item
            for name, item in value.items()
            if name != "canonical_body_sha256"
        }
    )
    fixture.evidence_raw = canonical_json_bytes(value) + b"\n"
    s3 = ExactS3()
    coordinate = _seed_sources(fixture, s3)

    with pytest.raises(ValueError, match="Task9|generation|ancestry"):
        _publish(fixture, s3, coordinate)

    assert s3.put_requests == []


def test_graceful_stop_must_bind_root_marker(
    tmp_path: Path,
) -> None:
    fixture = _terminal_fixture(tmp_path)
    value = json.loads(fixture.graceful_raw)
    value["campaign_terminal_marker_identity_sha256"] = _sha("foreign")
    body = dict(value)
    body.pop("graceful_stop_body_sha256")
    value["graceful_stop_body_sha256"] = canonical_sha256(body)
    fixture.graceful_raw = canonical_json_bytes(value) + b"\n"
    s3 = ExactS3()
    coordinate = _seed_sources(fixture, s3)

    with pytest.raises(ValueError, match="graceful|marker|ancestry"):
        _publish(fixture, s3, coordinate)

    assert s3.put_requests == []


def test_benchmark_contains_no_generation_publisher_call_or_script() -> None:
    source = BENCHMARK.read_text()
    assert "_publish_generation_campaign_drained" not in source
    assert "publish_glm52_generation_campaign_drained.py" not in source


def test_worker_restart_restores_only_root_compatibility_marker(
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "_campaign_drained_benchmark", BENCHMARK
    )
    assert spec is not None and spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    calls: list[tuple[object, ...]] = []
    controller = SimpleNamespace(
        kind=benchmark.RuntimeCampaignKind.SKYPILOT,
        phase_done=lambda phase: phase is CampaignPhase.DRAINED,
        root=tmp_path,
        s3_root=f"s3://{BUCKET}/campaigns/{RUN_ID}",
        _aws=lambda *args: calls.append(args),
    )

    benchmark.CampaignController.drain(controller)

    assert calls == [
        (
            "s3",
            "cp",
            f"s3://{BUCKET}/campaigns/{RUN_ID}/CAMPAIGN_DRAINED.json",
            str(tmp_path / "CAMPAIGN_DRAINED.json"),
            "--only-show-errors",
        )
    ]
    assert all("generations/" not in str(item) for call in calls for item in call)


class _Loader(yaml.SafeLoader):
    pass


def _unknown(loader: _Loader, suffix: str, node: yaml.Node) -> object:
    if isinstance(node, yaml.ScalarNode):
        return f"!{suffix} {loader.construct_scalar(node)}"
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_mapping(node)


_Loader.add_multi_constructor("!", _unknown)


def test_worker_is_explicitly_denied_generation_terminal_publication() -> None:
    template = yaml.load(STACK.read_text(), Loader=_Loader)
    statements = template["Resources"]["SkyPilotWorkerRole"][
        "Properties"
    ]["Policies"][0]["PolicyDocument"]["Statement"]
    assert not any(
        statement.get("Sid")
        in {
            "ListExactGenerationCampaignDrainedVersions",
            "PublishExactGenerationCampaignDrained",
        }
        for statement in statements
    )
    denial = next(
        statement
        for statement in statements
        if statement.get("Sid") == "DenyGenerationTerminalPublication"
    )
    assert denial == {
        "Sid": "DenyGenerationTerminalPublication",
        "Effect": "Deny",
        "Action": [
            "s3:PutObject",
            "s3:DeleteObject",
            "s3:DeleteObjectVersion",
        ],
        "Resource": (
            "!Sub ${ModelBucket.Arn}/campaigns/${SkyCampaignRunId}/"
            "submissions/production/generations/*/terminal/"
            "CAMPAIGN_DRAINED.json"
        ),
    }


def test_retained_terminal_v2_role_owns_dynamic_generation_publication() -> None:
    from glm52_enforcement.task12_support_plane import _runtime_statements

    inputs = SimpleNamespace(
        ledger_table_arn="arn:ledger",
        model_bucket_arn="arn:model",
        retained_kms_key_arn="arn:kms",
        worker_drain_document_arn="arn:ssm",
        activation_id="activation-0002",
    )
    statements = _runtime_statements(
        inputs, "TerminalV2", "keep-glm52-h1g-terminal-v2-writer"
    )
    publication = next(
        statement
        for statement in statements
        if statement.get("Sid")
        == "PublishExactGenerationCampaignDrained"
    )
    assert publication["Action"] == ["s3:GetObjectVersion", "s3:PutObject"]
    assert publication["Resource"].endswith(
        "/generations/*/terminal/CAMPAIGN_DRAINED.json"
    )
    assert not any(
        statement.get("Sid")
        == "PublishExactGenerationCampaignDrained"
        for statement in _runtime_statements(
            inputs, "WorkerDrain", "keep-glm52-h1g-worker-drain-signal"
        )
    )
