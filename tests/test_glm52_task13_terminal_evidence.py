from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

from glm52_enforcement.task13_terminal_evidence import (
    CAMPAIGN_DRAINED_KEY,
    CAMPAIGN_BUCKET,
    GPU_SPEND_LEDGER_KEY,
    MODEL_BUCKET,
    TASK13_TERMINAL_PROOF_KEY,
    TERMINAL_VERIFIED_KEY,
    TerminalEvidenceError,
    build_task13_terminal_proof,
    build_terminal_settlement,
    build_terminal_verified,
    publish_task13_terminal_proof,
    publish_terminal_verified,
    validate_campaign_drained,
    validate_task13_terminal_proof,
    validate_terminal_settlement,
    validate_terminal_verified,
)


ACCOUNT = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
ACTIVATION = "activation-0001"
REPO_ROOT = Path(__file__).parents[1]
PUBLISHER = (
    REPO_ROOT
    / "aws/glm52-gpu/scripts/publish_glm52_task13_terminal_evidence.py"
)
COORDINATOR = (
    REPO_ROOT
    / "aws/glm52-gpu/scripts/glm52_task13_production_coordinator.py"
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _coordinate(key: str, version_id: str) -> dict[str, object]:
    body = {
        "key": key,
        "version_id": version_id,
        "file_sha256": "a" * 64,
        "body_sha256": "b" * 64,
    }
    return {
        **body,
        "canonical_identity_sha256": hashlib.sha256(
            _canonical(body)
        ).hexdigest(),
    }


def _coordinate_for(
    key: str,
    version_id: str,
    value: dict[str, object],
    *,
    body_field: str | None = None,
) -> dict[str, object]:
    raw = _canonical(value) + b"\n"
    body = {
        "key": key,
        "version_id": version_id,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "body_sha256": (
            value[body_field]
            if body_field is not None
            else hashlib.sha256(_canonical(value)).hexdigest()
        ),
    }
    return {
        **body,
        "canonical_identity_sha256": hashlib.sha256(
            _canonical(body)
        ).hexdigest(),
    }


def _identity(body: dict[str, object]) -> dict[str, object]:
    return {
        **body,
        "canonical_identity_sha256": hashlib.sha256(
            _canonical(body)
        ).hexdigest(),
    }


def _campaign_drained() -> dict[str, object]:
    return {
        "record_type": "glm52_sky_campaign_drained_v2",
        "run_id": RUN_ID,
        "outcome": "completed",
        "last_completed_phase": "EVALUATION",
        "drain_reason": "campaign_complete",
        "prior_record_sha256": "a" * 64,
        "teacher_cache_ready_sha256": "b" * 64,
        "automatic_model_promotion": False,
        "model_uploaded": False,
        "execution_deadline": "2026-07-30T20:00:00Z",
        "gpu_allocation_sha256": "c" * 64,
    }


def _spend_ledger() -> tuple[bytes, dict[str, object]]:
    approval = "d" * 64
    genesis = {
        "record_type": "glm52_gpu_spend_ledger_genesis_v1",
        "run_id": RUN_ID,
        "approval_sha256": approval,
        "approved_gpu_runtime_seconds": 86400,
        "approved_gpu_cost_usd": 1320.96,
        "hourly_cost_usd": 55.04,
    }
    prior = hashlib.sha256(_canonical(genesis)).hexdigest()
    specs = (
        (
            "allocation_started",
            "glm52-cache-seed",
            "i-00000000000000001",
            "2026-07-29T20:00:00Z",
        ),
        (
            "allocation_ended",
            None,
            "i-00000000000000001",
            "2026-07-29T20:15:00Z",
        ),
        (
            "allocation_started",
            "glm52-production",
            "i-00000000000000006",
            "2026-07-29T21:00:00Z",
        ),
        (
            "allocation_ended",
            None,
            "i-00000000000000006",
            "2026-07-29T22:00:00Z",
        ),
    )
    records: list[dict[str, object]] = []
    for event, job_id, instance_id, timestamp in specs:
        body: dict[str, object] = {
            "record_type": "glm52_gpu_spend_event_v1",
            "run_id": RUN_ID,
            "approval_sha256": approval,
            "event": event,
            "instance_id": instance_id,
            "timestamp": timestamp,
            "prior_record_sha256": prior,
        }
        if job_id is not None:
            body["job_id"] = job_id
        record = {
            **body,
            "record_sha256": hashlib.sha256(_canonical(body)).hexdigest(),
        }
        records.append(record)
        prior = str(record["record_sha256"])
    raw = b"".join(_canonical(record) + b"\n" for record in records)
    coordinate_body = {
        "bucket": MODEL_BUCKET,
        "key": GPU_SPEND_LEDGER_KEY,
        "version_id": "spend-v1",
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "body_sha256": hashlib.sha256(_canonical(records)).hexdigest(),
        "head_record_sha256": records[-1]["record_sha256"],
    }
    coordinate = {
        **coordinate_body,
        "canonical_identity_sha256": hashlib.sha256(
            _canonical(coordinate_body)
        ).hexdigest(),
    }
    return raw, coordinate


def _terminal_state() -> dict[str, object]:
    spend_raw, _coordinate_value = _spend_ledger()
    drained_raw = _canonical(_campaign_drained()) + b"\n"
    body = {
        "record_type": "glm52_sky_terminal_verification_v1",
        "run_id": RUN_ID,
        "phase": "DRAINED",
        "outcome": "completed",
        "last_completed_phase": "EVALUATION",
        "terminal_state_authenticated": True,
        "drained_sha256": hashlib.sha256(drained_raw).hexdigest(),
        "campaign_ledger_sha256": "2" * 64,
        "gpu_spend_ledger_sha256": hashlib.sha256(
            spend_raw
        ).hexdigest(),
        "descriptor_body_sha256": "4" * 64,
    }
    return {
        **body,
        "verification_body_sha256": hashlib.sha256(
            _canonical(body)
        ).hexdigest(),
    }


def _settlement() -> dict[str, object]:
    spend_raw, spend_coordinate = _spend_ledger()
    return build_terminal_settlement(
        activation_id=ACTIVATION,
        terminal_state=_terminal_state(),
        gpu_spend_ledger=spend_coordinate,
        spend_ledger_raw=spend_raw,
        allocation_kinds=[
            "qualification-cache-seed",
            "production",
        ],
    )


def _rehash(value: dict[str, object]) -> None:
    body = dict(value)
    body.pop("canonical_identity_sha256", None)
    value["canonical_identity_sha256"] = hashlib.sha256(
        _canonical(body)
    ).hexdigest()


@pytest.mark.parametrize("mutation", ["allocation-split", "retained-cost"])
def test_settlement_rejects_rehashed_cost_manipulation(
    mutation: str,
) -> None:
    spend_raw, _coordinate_value = _spend_ledger()
    settlement = _settlement()
    if mutation == "allocation-split":
        settlement["allocations"][0]["gpu_cost_usd"] = "13.75"
        settlement["allocations"][1]["gpu_cost_usd"] = "55.05"
    else:
        settlement["retained_costs"]["retained_s3"] = "999999.99"
    _rehash(settlement)

    with pytest.raises(
        TerminalEvidenceError,
        match="allocation|provenance",
    ):
        validate_terminal_settlement(
            settlement,
            terminal_state=_terminal_state(),
            spend_ledger_raw=spend_raw,
        )


def test_settlement_rejects_rehashed_semantic_allocation_relabeling() -> None:
    """Break caught: authenticated cache-seed spend is relabeled recovery."""

    spend_raw, _coordinate_value = _spend_ledger()
    settlement = _settlement()
    settlement["allocations"][0]["allocation_kind"] = "recovery"
    _rehash(settlement)

    with pytest.raises(
        TerminalEvidenceError,
        match="allocation.*semantic|spend source",
    ):
        validate_terminal_settlement(
            settlement,
            terminal_state=_terminal_state(),
            spend_ledger_raw=spend_raw,
        )


def test_campaign_drained_accepts_exact_generation_marker_schema() -> None:
    drained = _campaign_drained()

    assert validate_campaign_drained(drained) == drained


def test_campaign_drained_rejects_activation_h1g_schema() -> None:
    drained = {
        "schema_version": 1,
        "record_type": "glm52_production_h1g_drained",
        "account_id": ACCOUNT,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": ACTIVATION,
    }

    with pytest.raises(TerminalEvidenceError, match="campaign drained"):
        validate_campaign_drained(drained)


def test_builders_close_terminal_verified_then_task13_proof() -> None:
    spend_raw, _spend_coordinate = _spend_ledger()
    drained_value = _campaign_drained()
    drained = _coordinate_for(
        CAMPAIGN_DRAINED_KEY,
        "drained-v1",
        drained_value,
    )
    verified = build_terminal_verified(
        activation_id=ACTIVATION,
        campaign_drained=drained,
        campaign_drained_value=drained_value,
        terminal_state=_terminal_state(),
        verified_at="2026-07-29T20:00:00Z",
    )
    assert validate_terminal_verified(verified) == verified
    verified_coordinate = _coordinate_for(
        TERMINAL_VERIFIED_KEY,
        "verified-v1",
        verified,
        body_field="canonical_identity_sha256",
    )

    proof = build_task13_terminal_proof(
        activation_id=ACTIVATION,
        campaign_drained=drained,
        campaign_drained_value=drained_value,
        terminal_verified=verified_coordinate,
        terminal_verified_value=verified,
        settlement=_settlement(),
        spend_ledger_raw=spend_raw,
    )

    assert (
        validate_task13_terminal_proof(
            proof,
            terminal_verified_value=verified,
            campaign_drained_value=drained_value,
            spend_ledger_raw=spend_raw,
        )
        == proof
    )
    assert proof["status"] == "TERMINAL_PROVEN"
    assert proof["sky_state"] == "SUCCEEDED"
    assert proof["ec2_billable_instance_ids"] == []


def test_terminal_proof_rejects_budget_arithmetic_drift() -> None:
    spend_raw, _spend_coordinate = _spend_ledger()
    drained_value = _campaign_drained()
    drained = _coordinate_for(
        CAMPAIGN_DRAINED_KEY,
        "drained-v1",
        drained_value,
    )
    verified = build_terminal_verified(
        activation_id=ACTIVATION,
        campaign_drained=drained,
        campaign_drained_value=drained_value,
        terminal_state=_terminal_state(),
        verified_at="2026-07-29T20:00:00Z",
    )
    proof = build_task13_terminal_proof(
        activation_id=ACTIVATION,
        campaign_drained=drained,
        campaign_drained_value=drained_value,
        terminal_verified=_coordinate_for(
            TERMINAL_VERIFIED_KEY,
            "verified-v1",
            verified,
            body_field="canonical_identity_sha256",
        ),
        terminal_verified_value=verified,
        settlement=_settlement(),
        spend_ledger_raw=spend_raw,
    )
    mutant = json.loads(json.dumps(proof))
    mutant["settlement"]["remaining_gpu_seconds"] = 81899
    _rehash(mutant["settlement"])
    unsigned = dict(mutant)
    unsigned.pop("canonical_identity_sha256")
    mutant["canonical_identity_sha256"] = hashlib.sha256(
        _canonical(unsigned)
    ).hexdigest()

    with pytest.raises(TerminalEvidenceError, match="budget"):
        validate_task13_terminal_proof(
            mutant,
            terminal_verified_value=verified,
            campaign_drained_value=drained_value,
            spend_ledger_raw=spend_raw,
        )


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[str, bytes, dict[str, str]]] = {}
        self.puts: list[dict[str, object]] = []
        self.fail_put = False
        self.delete_markers: list[dict[str, object]] = []
        self.prior_versions: list[dict[str, object]] = []

    @staticmethod
    def _metadata() -> dict[str, object]:
        return {
            "HTTPStatusCode": 200,
            "RequestId": "request-1",
            "RetryAttempts": 0,
        }

    def put_object(self, **kwargs: object) -> dict[str, object]:
        self.puts.append(kwargs)
        if self.fail_put:
            raise RuntimeError("ambiguous put")
        key = str(kwargs["Key"])
        if key in self.objects:
            raise RuntimeError("PreconditionFailed")
        version_id = "published-v1"
        self.objects[key] = (
            version_id,
            bytes(kwargs["Body"]),
            dict(kwargs["Metadata"]),
        )
        return {
            "ResponseMetadata": self._metadata(),
            "VersionId": version_id,
            "ChecksumSHA256": kwargs["ChecksumSHA256"],
        }

    def list_object_versions(self, **kwargs: object) -> dict[str, object]:
        key = str(kwargs["Prefix"])
        versions = list(self.prior_versions)
        if key in self.objects:
            version_id, _raw, _metadata = self.objects[key]
            versions.append(
                {
                    "Key": key,
                    "VersionId": version_id,
                    "IsLatest": True,
                }
            )
        return {
            "ResponseMetadata": self._metadata(),
            "IsTruncated": False,
            "Versions": versions,
            "DeleteMarkers": list(self.delete_markers),
        }

    def get_object(self, **kwargs: object) -> dict[str, object]:
        key = str(kwargs["Key"])
        version_id, raw, metadata = self.objects[key]
        assert kwargs["VersionId"] == version_id
        return {
            "ResponseMetadata": self._metadata(),
            "VersionId": version_id,
            "ChecksumSHA256": base64.b64encode(
                hashlib.sha256(raw).digest()
            ).decode(),
            "Metadata": metadata,
            "Body": io.BytesIO(raw),
        }


def _seed_campaign_drained(
    client: FakeS3,
    *,
    version_id: str = "drained-v1",
) -> tuple[dict[str, object], dict[str, object]]:
    value = _campaign_drained()
    client.objects[CAMPAIGN_DRAINED_KEY] = (
        version_id,
        _canonical(value) + b"\n",
        {"record-type": "glm52_sky_campaign_drained_v2"},
    )
    return value, _coordinate_for(
        CAMPAIGN_DRAINED_KEY,
        version_id,
        value,
    )


def test_publishers_conditionally_create_exact_generation_one_records() -> None:
    client = FakeS3()
    spend_raw, _spend_coordinate_value = _spend_ledger()
    drained_value, drained = _seed_campaign_drained(client)
    verified = build_terminal_verified(
        activation_id=ACTIVATION,
        campaign_drained=drained,
        campaign_drained_value=drained_value,
        terminal_state=_terminal_state(),
        verified_at="2026-07-29T20:00:00Z",
    )

    verified_coordinate = publish_terminal_verified(client, verified)
    proof = build_task13_terminal_proof(
        activation_id=ACTIVATION,
        campaign_drained=drained,
        campaign_drained_value=drained_value,
        terminal_verified=verified_coordinate,
        terminal_verified_value=verified,
        settlement=_settlement(),
        spend_ledger_raw=spend_raw,
    )
    proof_coordinate = publish_task13_terminal_proof(
        client,
        proof,
        terminal_verified_value=verified,
        campaign_drained_value=drained_value,
        spend_ledger_raw=spend_raw,
    )

    assert [put["Key"] for put in client.puts] == [
        TERMINAL_VERIFIED_KEY,
        TASK13_TERMINAL_PROOF_KEY,
    ]
    assert all(put["Bucket"] == CAMPAIGN_BUCKET for put in client.puts)
    assert all(put["IfNoneMatch"] == "*" for put in client.puts)
    assert verified_coordinate["key"] == TERMINAL_VERIFIED_KEY
    assert proof_coordinate["key"] == TASK13_TERMINAL_PROOF_KEY


def test_terminal_verified_publisher_rejects_rehashed_foreign_drained_hash() -> None:
    """Break caught: validator-only publication bypassed the safe builder."""

    client = FakeS3()
    drained_value, drained_coordinate = _seed_campaign_drained(client)
    value = build_terminal_verified(
        activation_id=ACTIVATION,
        campaign_drained=drained_coordinate,
        campaign_drained_value=drained_value,
        terminal_state=_terminal_state(),
        verified_at="2026-07-29T20:00:00Z",
    )
    value["terminal_state"]["drained_sha256"] = "f" * 64
    terminal_body = dict(value["terminal_state"])
    terminal_body.pop("verification_body_sha256")
    value["terminal_state"]["verification_body_sha256"] = hashlib.sha256(
        _canonical(terminal_body)
    ).hexdigest()
    _rehash(value)

    with pytest.raises(TerminalEvidenceError, match="drained hash"):
        validate_terminal_verified(value)
    with pytest.raises(TerminalEvidenceError, match="drained hash"):
        publish_terminal_verified(client, value)
    assert client.puts == []


def test_terminal_verified_publisher_rejects_rehashed_foreign_version() -> None:
    """Break caught: a foreign VersionId could poison the immutable result."""

    client = FakeS3()
    drained_value, drained_coordinate = _seed_campaign_drained(client)
    value = build_terminal_verified(
        activation_id=ACTIVATION,
        campaign_drained=drained_coordinate,
        campaign_drained_value=drained_value,
        terminal_state=_terminal_state(),
        verified_at="2026-07-29T20:00:00Z",
    )
    value["campaign_drained"]["version_id"] = "foreign-v9"
    _rehash(value["campaign_drained"])
    _rehash(value)

    assert validate_terminal_verified(value) == value
    with pytest.raises(TerminalEvidenceError, match="VersionId"):
        publish_terminal_verified(client, value)
    assert client.puts == []


def test_publisher_adopts_only_one_exact_existing_version() -> None:
    client = FakeS3()
    drained_value, drained_coordinate = _seed_campaign_drained(client)
    value = build_terminal_verified(
        activation_id=ACTIVATION,
        campaign_drained=drained_coordinate,
        campaign_drained_value=drained_value,
        terminal_state=_terminal_state(),
        verified_at="2026-07-29T20:00:00Z",
    )
    raw = _canonical(value) + b"\n"
    metadata = {
        "record-type": str(value["record_type"]),
        "canonical-identity-sha256": str(
            value["canonical_identity_sha256"]
        ),
    }
    client.objects[TERMINAL_VERIFIED_KEY] = (
        "existing-v1",
        raw,
        metadata,
    )
    client.fail_put = True

    coordinate = publish_terminal_verified(client, value)

    assert coordinate["version_id"] == "existing-v1"
    assert len(client.puts) == 1


@pytest.mark.parametrize("corruption", ["delete-marker", "prior-version"])
def test_publisher_refuses_nonunique_version_history(corruption: str) -> None:
    client = FakeS3()
    drained_value, drained_coordinate = _seed_campaign_drained(client)
    value = build_terminal_verified(
        activation_id=ACTIVATION,
        campaign_drained=drained_coordinate,
        campaign_drained_value=drained_value,
        terminal_state=_terminal_state(),
        verified_at="2026-07-29T20:00:00Z",
    )
    raw = _canonical(value) + b"\n"
    metadata = {
        "record-type": str(value["record_type"]),
        "canonical-identity-sha256": str(
            value["canonical_identity_sha256"]
        ),
    }
    client.objects[TERMINAL_VERIFIED_KEY] = (
        "existing-v1",
        raw,
        metadata,
    )
    if corruption == "delete-marker":
        client.delete_markers.append(
            {
                "Key": TERMINAL_VERIFIED_KEY,
                "VersionId": "deleted-v0",
                "IsLatest": False,
            }
        )
    else:
        client.prior_versions.append(
            {
                "Key": TERMINAL_VERIFIED_KEY,
                "VersionId": "prior-v0",
                "IsLatest": False,
            }
        )
    client.fail_put = True

    with pytest.raises(TerminalEvidenceError, match="one immutable version"):
        publish_terminal_verified(client, value)


class FakeSts:
    def __init__(self, account: str = ACCOUNT) -> None:
        self.account = account

    def get_caller_identity(self) -> dict[str, object]:
        return {
            "Account": self.account,
            "Arn": f"arn:aws:sts::{self.account}:assumed-role/test/session",
        }


def _publisher_module():
    spec = importlib.util.spec_from_file_location(
        "_task13_terminal_publisher",
        PUBLISHER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _coordinator_module():
    spec = importlib.util.spec_from_file_location(
        "_task13_terminal_coordinator",
        COORDINATOR,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coordinator_returns_only_fully_validated_terminal_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _coordinator_module()
    spend_raw, _spend_coordinate = _spend_ledger()
    drained_value = _campaign_drained()
    drained_coordinate = _coordinate_for(
        CAMPAIGN_DRAINED_KEY,
        "drained-v1",
        drained_value,
    )
    verified_value = build_terminal_verified(
        activation_id=ACTIVATION,
        campaign_drained=drained_coordinate,
        campaign_drained_value=drained_value,
        terminal_state=_terminal_state(),
        verified_at="2026-07-29T20:00:00Z",
    )
    verified_coordinate = _coordinate_for(
        TERMINAL_VERIFIED_KEY,
        "verified-v1",
        verified_value,
        body_field="canonical_identity_sha256",
    )
    proof = build_task13_terminal_proof(
        activation_id=ACTIVATION,
        campaign_drained=drained_coordinate,
        campaign_drained_value=drained_value,
        terminal_verified=verified_coordinate,
        terminal_verified_value=verified_value,
        settlement=_settlement(),
        spend_ledger_raw=spend_raw,
    )
    raw_by_coordinate = {
        (
            CAMPAIGN_BUCKET,
            CAMPAIGN_DRAINED_KEY,
            "drained-v1",
        ): _canonical(drained_value) + b"\n",
        (
            CAMPAIGN_BUCKET,
            TERMINAL_VERIFIED_KEY,
            "verified-v1",
        ): _canonical(verified_value) + b"\n",
        (
            MODEL_BUCKET,
            GPU_SPEND_LEDGER_KEY,
            str(proof["settlement"]["gpu_spend_ledger"]["version_id"]),
        ): spend_raw,
    }
    monkeypatch.setattr(
        module,
        "_monitor",
        lambda _contract: {"status": "SUCCEEDED", "instance_ids": []},
    )
    monkeypatch.setattr(
        module,
        "_read_result_record",
        lambda *, key: (
            proof,
            _coordinate_for(
                key,
                "proof-v1",
                proof,
                body_field="canonical_identity_sha256",
            ),
        ),
    )
    monkeypatch.setattr(
        module,
        "_get_s3_version",
        lambda *, bucket, key, version_id: raw_by_coordinate[
            (bucket, key, version_id)
        ],
    )
    monkeypatch.setenv("GLM52_TASK13_REPO_ROOT", str(REPO_ROOT))

    envelope = module._terminal_proof(
        {
            "account_id": ACCOUNT,
            "region": REGION,
            "profile": "keep-gpu",
            "run_id": RUN_ID,
            "required_terminal_markers": [
                "CAMPAIGN_DRAINED.json",
                "TERMINAL_VERIFIED.json",
            ],
            "required_sky_state": "SUCCEEDED",
            "required_billable_p5_instance_count": 0,
            "activation_id": ACTIVATION,
            "monitor_contract": {"closed": True},
        }
    )

    assert set(envelope) == {
        "proof",
        "terminal_verified_value",
        "campaign_drained_value",
        "spend_ledger_raw",
    }
    assert envelope["proof"] == proof
    assert envelope["terminal_verified_value"] == verified_value
    assert envelope["campaign_drained_value"] == drained_value
    assert envelope["spend_ledger_raw"] == spend_raw


def test_cli_publishes_verified_then_proof_from_exact_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Break caught: validators exist but no repository executable owns them."""

    module = _publisher_module()
    client = FakeS3()
    drained_value = _campaign_drained()
    drained_raw = _canonical(drained_value) + b"\n"
    client.objects[CAMPAIGN_DRAINED_KEY] = (
        "drained-v1",
        drained_raw,
        {"record-type": "glm52_sky_campaign_drained_v2"},
    )
    spend_raw, spend_coordinate = _spend_ledger()
    client.objects[GPU_SPEND_LEDGER_KEY] = (
        str(spend_coordinate["version_id"]),
        spend_raw,
        {"record-type": "glm52_gpu_spend_event_v1"},
    )
    terminal_state = tmp_path / "TERMINAL_VERIFIED.legacy.json"
    terminal_state.write_bytes(_canonical(_terminal_state()) + b"\n")
    settlement = tmp_path / "terminal-settlement.json"
    settlement.write_bytes(_canonical(_settlement()) + b"\n")

    assert (
        module.main(
            [
                "--profile",
                "keep-gpu",
                "--activation-id",
                ACTIVATION,
                "--terminal-state",
                str(terminal_state),
                "--settlement",
                str(settlement),
                "--verified-at",
                "2026-07-29T20:00:00Z",
            ],
            services_factory=lambda _profile: (FakeSts(), client),
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["terminal_verified"]["key"] == TERMINAL_VERIFIED_KEY
    assert output["task13_terminal_proof"]["key"] == (
        TASK13_TERMINAL_PROOF_KEY
    )
    assert [put["Key"] for put in client.puts] == [
        TERMINAL_VERIFIED_KEY,
        TASK13_TERMINAL_PROOF_KEY,
    ]


def test_cli_refuses_fabricated_campaign_drained_before_publication(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _publisher_module()
    client = FakeS3()
    fabricated = {"record_type": "campaign-drained"}
    client.objects[CAMPAIGN_DRAINED_KEY] = (
        "drained-v1",
        _canonical(fabricated) + b"\n",
        {"record-type": "campaign-drained"},
    )
    terminal_state = tmp_path / "terminal.json"
    terminal_state.write_bytes(_canonical(_terminal_state()) + b"\n")
    settlement = tmp_path / "settlement.json"
    settlement.write_bytes(_canonical(_settlement()) + b"\n")

    assert (
        module.main(
            [
                "--profile",
                "keep-gpu",
                "--activation-id",
                ACTIVATION,
                "--terminal-state",
                str(terminal_state),
                "--settlement",
                str(settlement),
                "--verified-at",
                "2026-07-29T20:00:00Z",
            ],
            services_factory=lambda _profile: (FakeSts(), client),
        )
        == 70
    )
    assert client.puts == []
    assert "campaign drained" in capsys.readouterr().err


def test_cli_refuses_foreign_account_before_any_publication(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _publisher_module()
    client = FakeS3()
    terminal_state = tmp_path / "terminal.json"
    terminal_state.write_bytes(_canonical(_terminal_state()) + b"\n")
    settlement = tmp_path / "settlement.json"
    settlement.write_bytes(_canonical(_settlement()) + b"\n")

    assert (
        module.main(
            [
                "--profile",
                "keep-gpu",
                "--activation-id",
                ACTIVATION,
                "--terminal-state",
                str(terminal_state),
                "--settlement",
                str(settlement),
                "--verified-at",
                "2026-07-29T20:00:00Z",
            ],
            services_factory=lambda _profile: (
                FakeSts("000000000000"),
                client,
            ),
        )
        == 70
    )
    assert client.puts == []
    assert "approved AWS account" in capsys.readouterr().err
