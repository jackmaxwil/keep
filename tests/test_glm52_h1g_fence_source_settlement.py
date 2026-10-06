"""Strict H.1g terminal source-settlement boundary contracts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from types import MappingProxyType

import pytest

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256


def SHA(marker: str) -> str:
    return marker * 64


ACCOUNT_ID = "246813579024"
BUCKET = "keep-glm52-models-246813579024-us-west-2"
RUN_ID = "glm52-sky-20260724"
ACTIVATION_ID = "glm52-v2-amber-quartz"
STACK_ID = (
    "arn:aws:cloudformation:us-west-2:246813579024:stack/"
    "keep-glm52-h1g-fence/01234567-89ab-cdef-0123-456789abcdef"
)
KMS_ARN = "arn:aws:kms:us-west-2:246813579024:key/01234567-89ab-cdef-0123-456789abcdef"
BASE_TIME = datetime(2026, 7, 31, 0, 0, 0, tzinfo=UTC)


def _api():
    from glm52_enforcement import fence_source_settlement

    return fence_source_settlement


def _signed(**fields):
    value = dict(fields)
    value["canonical_identity_sha256"] = canonical_sha256(value)
    return value


def _utc(offset: int) -> str:
    return (BASE_TIME + timedelta(seconds=offset)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _source_keys():
    return (
        "campaigns/glm52-sky-20260724/spend-snapshots/"
        + SHA("a")
        + "/GPU_SPEND_SNAPSHOT.json",
        "campaigns/glm52-sky-20260724/submissions/production/intents/"
        + SHA("b")
        + "/SKYPILOT_SUBMISSION_INTENT.json",
        "campaigns/glm52-sky-20260724/production/controller-baselines/"
        + SHA("c")
        + "/CONTROLLER_BASELINE.json",
        "campaigns/glm52-sky-20260724/monitor/must-start/production/"
        + SHA("d")
        + "/control-plane-ready/"
        + SHA("e")
        + "/CONTROL_PLANE_READY.json",
        "campaigns/glm52-sky-20260724/submissions/production/acquisitions/"
        + SHA("f")
        + "/SUBMISSION_ACQUIRED.json",
    )


def _publisher_rows():
    api = _api()
    return tuple(
        {
            "source_family": family,
            "publisher_role_arn": (
                f"arn:aws:iam::{ACCOUNT_ID}:role/keep-glm52-publisher-{index}"
            ),
            "publisher_role_id": f"AROAPUBLISHER{index:05d}",
        }
        for index, family in enumerate(api.SOURCE_FAMILY_ORDER)
    )


def _selected_rows():
    return tuple(
        _signed(
            source_family=publisher["source_family"],
            key=key,
            version_id=f"source-version-{index}",
            file_sha256=hashlib.sha256(f"file-{index}".encode()).hexdigest(),
            body_sha256=hashlib.sha256(f"body-{index}".encode()).hexdigest(),
            publisher_role_arn=publisher["publisher_role_arn"],
            publisher_role_id=publisher["publisher_role_id"],
            s3_request_id=f"s3-request-{index}",
        )
        for index, (publisher, key) in enumerate(zip(_publisher_rows(), _source_keys()))
    )


def _action_rows(status: str = "SUCCEEDED"):
    return tuple(
        _signed(
            source_family=family,
            action_identity_sha256=hashlib.sha256(
                f"action-{index}".encode()
            ).hexdigest(),
            invocation_identity_sha256=hashlib.sha256(
                f"invocation-{index}".encode()
            ).hexdigest(),
            terminal_status=status,
            terminal_observed_at=_utc(1),
        )
        for index, family in enumerate(_api().SOURCE_FAMILY_ORDER)
    )


def _invocation_rows(status: str = "SUCCEEDED"):
    selected = _selected_rows()
    actions = _action_rows(status)
    rows = []
    for index, (source, action) in enumerate(zip(selected, actions)):
        successful = status == "SUCCEEDED"
        rows.append(
            _signed(
                source_family=source["source_family"],
                action_identity_sha256=action["action_identity_sha256"],
                invocation_identity_sha256=action["invocation_identity_sha256"],
                function_version_arn=(
                    f"arn:aws:lambda:us-west-2:{ACCOUNT_ID}:function:"
                    f"keep-glm52-source-{index}:{index + 1}"
                ),
                executed_version=str(index + 1),
                lambda_request_id=f"lambda-request-{index}",
                request_identity_sha256=hashlib.sha256(
                    f"request-{index}".encode()
                ).hexdigest(),
                response_identity_sha256=(
                    hashlib.sha256(f"response-{index}".encode()).hexdigest()
                    if successful
                    else None
                ),
                s3_request_id=(source["s3_request_id"] if successful else None),
                version_id=(source["version_id"] if successful else None),
                terminal_status=status,
                terminal_observed_at=_utc(2),
            )
        )
    return tuple(rows)


def _freeze_execution(
    bootstrap_identity: str,
    freeze_entry_identity: str,
    deployed_policy_sha256: str,
    stack_role_arn: str,
):
    return _signed(
        slot="SOURCE_FAMILIES_FROZEN",
        outcome="APPLIED",
        bootstrap_manifest_identity_sha256=bootstrap_identity,
        freeze_entry_identity_sha256=freeze_entry_identity,
        deployed_policy_sha256=deployed_policy_sha256,
        stack_id=STACK_ID,
        stack_role_arn=stack_role_arn,
        change_set_arn=(
            f"arn:aws:cloudformation:us-west-2:{ACCOUNT_ID}:"
            "changeSet/freeze-request/0f844297-5e20-4d17-853f-fc8a8eaf8934"
        ),
        execute_request_id="freeze-execute-request",
        executed_at=_utc(0),
    )


def _probe_rows():
    api = _api()
    from glm52_enforcement.fence_artifacts import FenceSlot

    freeze_policy_sha256 = (
        _bootstrap_manifest().entry(FenceSlot.SOURCE_FAMILIES_FROZEN).policy_sha256
    )
    return tuple(
        _signed(
            round=round_index + 1,
            observed_at=_utc(round_index * 10),
            policy_sha256=freeze_policy_sha256,
            direct_policy_sha256=freeze_policy_sha256,
            attribution_rows=tuple(
                _signed(
                    source_family=publisher["source_family"],
                    expected_sid=(
                        "DenyAllReservedFamilyMutation_SOURCE_FAMILIES_FROZEN"
                    ),
                    publisher_role_arn=publisher["publisher_role_arn"],
                    publisher_role_id=publisher["publisher_role_id"],
                    existing_credentials_denied=True,
                    fresh_session_denied=True,
                    alternate_principal_denied=True,
                    denied_operations=api.SOURCE_DENIAL_OPERATIONS,
                    service_request_ids=tuple(
                        f"denial-{round_index}-{family_index}-{operation_index}"
                        for operation_index, _ in enumerate(
                            api.SOURCE_DENIAL_OPERATIONS
                        )
                    ),
                )
                for family_index, publisher in enumerate(_publisher_rows())
            ),
        )
        for round_index in range(2)
    )


def _workflow_row():
    return _signed(
        state_machine_version_arn=(
            f"arn:aws:states:us-west-2:{ACCOUNT_ID}:stateMachine:keep-glm52-source:1"
        ),
        post_source_state="SOURCE_PHASE_CLOSED",
        source_task_families=_api().SOURCE_FAMILY_ORDER,
        reachable_source_task_families=(),
        graph_sha256=SHA("5"),
        observed_at=_utc(11),
    )


def _reachability_row():
    return _signed(
        source_families=_api().SOURCE_FAMILY_ORDER,
        publisher_function_version_arns=tuple(
            row["function_version_arn"] for row in _invocation_rows()
        ),
        reachable_source_families=(),
        closed=True,
        no_launch=True,
        observed_at=_utc(11),
    )


def _bootstrap_manifest():
    from test_glm52_h1g_fence_artifacts import _build_bootstrap_manifest

    return _build_bootstrap_manifest()


def _coordinate(
    key: str,
    marker: str,
    *,
    version_id: str | None = None,
    file_sha256: str | None = None,
    canonical_identity_sha256: str | None = None,
):
    from glm52_enforcement.fence_artifacts import ArtifactCoordinate

    return ArtifactCoordinate(
        bucket=BUCKET,
        key=key,
        version_id=version_id or f"version-{marker}",
        file_sha256=file_sha256 or SHA(marker),
        canonical_identity_sha256=canonical_identity_sha256 or SHA(marker),
    )


def _runtime_coordinate():
    from glm52_enforcement.fence_artifacts import SupportRuntimeIdentityCoordinate

    return SupportRuntimeIdentityCoordinate(
        bucket=BUCKET,
        key=(
            "campaigns/glm52-sky-20260724/authorities/fence/runtime/"
            f"{ACTIVATION_ID}/00000001/SUPPORT_RUNTIME_IDENTITY.json"
        ),
        version_id="runtime-version",
        file_sha256=SHA("a"),
        canonical_identity_sha256=SHA("b"),
    )


def _policy_input(head, *, nonselected_keys=()):
    from glm52_enforcement import fence_policy_renderer as renderer

    selected = _selected_rows()
    publishers = tuple(
        renderer.PrincipalIdentity(
            binding_id=row["source_family"],
            arn=row["publisher_role_arn"],
            role_id=row["publisher_role_id"],
        )
        for row in selected
    )
    common = {
        "build_mode": renderer.BuildMode.LEGACY_ENABLED,
        "policy_head": head,
        "bucket_name": BUCKET,
        "account_id": ACCOUNT_ID,
        "kms_key_arn": KMS_ARN,
        "predecessor_policy_sha256": None,
        "freeze_denial_evidence_sha256": None,
        "source_publication_sealed_sha256": None,
        "all_version_inventory_sha256": None,
        "source_settlement_sha256": None,
        "publisher_deny_policy_sha256": SHA("8"),
        "terminal_prerequisite_sha256": (
            SHA("9") if head is renderer.PolicyHead.TERMINAL else None
        ),
        "retired_publishers": publishers,
        "retired_publisher_resources": tuple(
            renderer.FIXED_BUCKET_ARN + "/" + row["key"] for row in selected
        ),
        "selected_source_keys": tuple(row["key"] for row in selected),
        "nonselected_source_keys": nonselected_keys,
    }
    return renderer.FencePolicyInput(**common)


def _request(*, nonselected_keys=()):
    from glm52_enforcement import fence_policy_renderer as renderer

    api = _api()
    bootstrap = _bootstrap_manifest()
    entries = {entry.slot: entry.to_dict() for entry in bootstrap.entries}
    reservation = entries[
        __import__(
            "glm52_enforcement.fence_artifacts", fromlist=["FenceSlot"]
        ).FenceSlot.RESERVATION_ONLY
    ]
    freeze = entries[
        __import__(
            "glm52_enforcement.fence_artifacts", fromlist=["FenceSlot"]
        ).FenceSlot.SOURCE_FAMILIES_FROZEN
    ]
    bootstrap_key = (
        "campaigns/glm52-sky-20260724/authorities/fence/manifests/"
        f"{ACTIVATION_ID}/00000001/FENCE_BOOTSTRAP_MANIFEST.json"
    )
    return api.SourceSettlementRequest(
        bootstrap_manifest=bootstrap,
        bootstrap_manifest_coordinate=_coordinate(
            bootstrap_key,
            "c",
            canonical_identity_sha256=bootstrap.canonical_identity_sha256,
        ),
        support_runtime_identity_coordinate=_runtime_coordinate(),
        reservation_coordinate=_coordinate(
            reservation["template_key"],
            "d",
            version_id=reservation["version_id"],
            file_sha256=reservation["template_sha256"],
            canonical_identity_sha256=reservation["template_body_sha256"],
        ),
        freeze_entry_coordinate=_coordinate(
            freeze["template_key"],
            "e",
            version_id=freeze["version_id"],
            file_sha256=freeze["template_sha256"],
            canonical_identity_sha256=freeze["template_body_sha256"],
        ),
        batch_policy_input=_policy_input(
            renderer.PolicyHead.BATCH,
            nonselected_keys=nonselected_keys,
        ),
        terminal_policy_input=_policy_input(
            renderer.PolicyHead.TERMINAL,
            nonselected_keys=nonselected_keys,
        ),
        batch_successor_contract_sha256=SHA("1"),
        terminal_successor_contract_sha256=SHA("2"),
        materializer_function_version_arn=(
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-h1g-fence-source-settlement:3"
        ),
        evidence_coordinates={
            name: _coordinate(
                (
                    "campaigns/glm52-sky-20260724/authorities/fence/"
                    f"evidence/{ACTIVATION_ID}/00000001/{name}.json"
                ),
                str(index + 3),
            )
            for index, name in enumerate(
                (
                    "read_freeze_execution_evidence",
                    "read_selected_source_rows",
                    "read_freeze_denial_probe_rows",
                    "read_source_action_terminal_rows",
                    "read_source_lambda_execution_terminal_rows",
                    "read_workflow_post_source_state",
                    "read_publisher_reachability_proof",
                )
            )
        },
        kms_key_arn=KMS_ARN,
    )


class _Evidence:
    def __init__(self, *, status="SUCCEEDED"):
        bootstrap = _bootstrap_manifest()
        freeze = bootstrap.entry(
            __import__(
                "glm52_enforcement.fence_artifacts", fromlist=["FenceSlot"]
            ).FenceSlot.SOURCE_FAMILIES_FROZEN
        )
        self.freeze = _freeze_execution(
            bootstrap.canonical_identity_sha256,
            freeze.entry_identity_sha256,
            freeze.policy_sha256,
            bootstrap.to_dict()["fence_service_role"]["arn"],
        )
        self.probes = _probe_rows()
        self.actions = _action_rows(status)
        self.invocations = _invocation_rows(status)
        self.workflow = _workflow_row()
        self.reachability = _reachability_row()
        self.selected = _selected_rows()
        self.read_count = 0
        self.drift_after_first = None

    def _value(self, field):
        self.read_count += 1
        value = getattr(self, field)
        if self.drift_after_first == field and self.read_count > 6:
            changed = dict(value)
            changed["canonical_identity_sha256"] = SHA("0")
            return changed
        return value

    def read_freeze_execution_evidence(self):
        return self._value("freeze")

    def read_freeze_denial_probe_rows(self):
        return self._value("probes")

    def read_source_action_terminal_rows(self):
        return self._value("actions")

    def read_source_lambda_execution_terminal_rows(self):
        return self._value("invocations")

    def read_workflow_post_source_state(self):
        return self._value("workflow")

    def read_publisher_reachability_proof(self):
        return self._value("reachability")

    def read_selected_source_rows(self):
        return self._value("selected")


class _Clock:
    def __init__(self):
        self.current = BASE_TIME
        self.sleeps = []

    def __call__(self):
        return self.current

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.current += timedelta(seconds=seconds)


class _S3:
    def __init__(self):
        self.pass_index = 0
        self.mutant = None
        self.calls = []

    def list_object_versions(self, **request):
        api = _api()
        prefix = request["Prefix"]
        family_index = tuple(api.SOURCE_FAMILY_PREFIXES.values()).index(prefix)
        if family_index == 0 and "KeyMarker" not in request:
            self.pass_index += 1
        key = _source_keys()[family_index]
        version = f"source-version-{family_index}"
        if self.mutant == "second-pass" and self.pass_index >= 2 and family_index == 4:
            version = "drifted-version"
        response = {
            "Name": BUCKET,
            "Prefix": prefix,
            "IsTruncated": False,
            "Versions": [
                {
                    "Key": key,
                    "VersionId": version,
                    "IsLatest": True,
                    "ETag": f'"etag-{family_index}"',
                    "Size": family_index + 1,
                    "LastModified": BASE_TIME,
                }
            ],
            "DeleteMarkers": [],
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": (
                    f"list-{self.pass_index}-{family_index}-"
                    + ("continued" if "KeyMarker" in request else "first")
                ),
                "RetryAttempts": 0,
            },
        }
        if "KeyMarker" in request:
            response["KeyMarker"] = request["KeyMarker"]
            response["VersionIdMarker"] = request["VersionIdMarker"]
        if self.mutant == "paged" and family_index == 0:
            if "KeyMarker" not in request:
                response["IsTruncated"] = True
                response["NextKeyMarker"] = key
                response["NextVersionIdMarker"] = "page-marker"
            else:
                response["Versions"] = []
                response["DeleteMarkers"] = [
                    {
                        "Key": (
                            api.SOURCE_FAMILY_PREFIXES["gpu-spend-snapshot"]
                            + SHA("1")
                            + "/GPU_SPEND_SNAPSHOT.json"
                        ),
                        "VersionId": "delete-marker-version",
                        "IsLatest": True,
                        "LastModified": BASE_TIME,
                    }
                ]
        if self.mutant == "cycle" and family_index == 0 and "KeyMarker" in request:
            response["Versions"] = []
        if self.mutant == "partial-marker" and family_index == 0:
            response["IsTruncated"] = True
            response["NextKeyMarker"] = key
        if self.mutant == "cycle" and family_index == 0:
            response["IsTruncated"] = True
            response["NextKeyMarker"] = request.get("KeyMarker", key)
            response["NextVersionIdMarker"] = request.get("VersionIdMarker", version)
        self.calls.append(dict(request))
        return response


def _services(evidence=None, s3=None, clock=None):
    api = _api()
    clock = clock or _Clock()
    return api.SourceSettlementServices(
        evidence=evidence or _Evidence(),
        s3=s3 or _S3(),
        artifact_services=object(),
        now_utc=clock,
        sleep=clock.sleep,
    )


def _rendered(marker):
    from test_glm52_h1g_fence_policy_renderer import _slot_input

    from glm52_enforcement.fence_policy_renderer import (
        PolicyHead,
        render_fence_policy,
    )

    return render_fence_policy(_slot_input(PolicyHead(marker)))


def _install_late_artifact_fakes(monkeypatch, events):
    api = _api()
    from glm52_enforcement.fence_artifacts import (
        ArtifactCoordinate,
        FenceSlot,
    )
    from glm52_enforcement.fence_artifacts import (
        build_fence_entry as build_fence_entry_real,
    )
    from glm52_enforcement.fence_artifacts import (
        build_fence_manifest as build_fence_manifest_real,
    )

    def render(value):
        events.append("render-" + value.policy_head.value)
        return _rendered(value.policy_head.value)

    def build_entry(**kwargs):
        events.append("build-entry-" + kwargs["slot"].value)
        return build_fence_entry_real(**kwargs)

    def build_manifest(**kwargs):
        events.append("build-manifest")
        return build_fence_manifest_real(**kwargs)

    def publish(**kwargs):
        assert tuple(kwargs["template_bytes"]) == (
            FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION,
            FenceSlot.TERMINAL,
        )
        batch_raw = kwargs["template_bytes"][FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION]
        terminal_raw = kwargs["template_bytes"][FenceSlot.TERMINAL]
        batch = _coordinate(
            "campaigns/glm52-sky-20260724/authorities/fence/templates/"
            f"{ACTIVATION_ID}/00000001/BATCH_FIVE_SOURCE_ACTIVATION.json",
            "a",
            file_sha256=hashlib.sha256(batch_raw).hexdigest(),
            canonical_identity_sha256=hashlib.sha256(batch_raw[:-1]).hexdigest(),
        )
        terminal = _coordinate(
            "campaigns/glm52-sky-20260724/authorities/fence/templates/"
            f"{ACTIVATION_ID}/00000001/TERMINAL.json",
            "b",
            file_sha256=hashlib.sha256(terminal_raw).hexdigest(),
            canonical_identity_sha256=hashlib.sha256(terminal_raw[:-1]).hexdigest(),
        )
        events.extend(("publish-BATCH", "publish-TERMINAL"))
        raw = kwargs["manifest_builder"](
            MappingProxyType(
                {
                    FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION: batch,
                    FenceSlot.TERMINAL: terminal,
                }
            ),
            None,
        )
        assert raw.endswith(b"\n")
        events.append("publish-manifest")
        manifest = ArtifactCoordinate(
            bucket=BUCKET,
            key=(
                "campaigns/glm52-sky-20260724/authorities/fence/manifests/"
                f"{ACTIVATION_ID}/00000001/FENCE_SOURCE_SETTLED_MANIFEST.json"
            ),
            version_id="manifest-version",
            file_sha256=hashlib.sha256(raw).hexdigest(),
            canonical_identity_sha256=json.loads(raw)["canonical_identity_sha256"],
        )
        return batch, terminal, manifest

    monkeypatch.setattr(api, "render_fence_policy", render)
    monkeypatch.setattr(api, "build_fence_entry", build_entry)
    monkeypatch.setattr(api, "build_fence_manifest", build_manifest)
    monkeypatch.setattr(api, "publish_fence_stage", publish)


def test_source_settlement_request_projection_round_trips_exactly() -> None:
    api = _api()
    request = _request()
    projection = api.source_settlement_request_projection(request)
    parsed = api.parse_source_settlement_request_v2(
        json.loads(canonical_json_bytes(projection))
    )

    assert api.source_settlement_request_projection(parsed) == projection
    assert parsed.bootstrap_manifest.canonical_identity_sha256 == (
        request.bootstrap_manifest.canonical_identity_sha256
    )


def test_success_seals_then_renders_and_publishes_exact_late_order(monkeypatch):
    api = _api()
    events = []
    _install_late_artifact_fakes(monkeypatch, events)

    result = api.settle_source_publication(request=_request(), services=_services())

    assert result.disposition is api.SourceSettlementDisposition.SEALED
    assert result.classification == "SOURCE_PUBLICATION_SEALED"
    assert result.source_settlement is not None
    assert result.source_settled_manifest is not None
    assert result.no_launch_evidence is None
    assert result.selected_entry.slot.value == "BATCH_FIVE_SOURCE_ACTIVATION"
    assert (
        tuple(
            row["source_family"]
            for row in result.source_settlement.to_dict()["selected_source_rows"]
        )
        == api.SOURCE_FAMILY_ORDER
    )
    assert events == [
        "render-BATCH",
        "render-TERMINAL",
        "publish-BATCH",
        "publish-TERMINAL",
        "build-entry-BATCH_FIVE_SOURCE_ACTIVATION",
        "build-entry-TERMINAL",
        "build-manifest",
        "publish-manifest",
    ]


@pytest.mark.parametrize(
    "dependency",
    (
        "build_source_settlement_seal",
        "render_fence_policy",
        "build_fence_template_bytes",
    ),
)
def test_unexpected_late_settlement_defect_propagates_unchanged(
    monkeypatch, dependency
):
    api = _api()
    _install_late_artifact_fakes(monkeypatch, [])
    defect = AssertionError("unexpected late settlement defect")

    def fail(*args, **kwargs):
        raise defect

    monkeypatch.setattr(api, dependency, fail)

    with pytest.raises(AssertionError) as raised:
        api.settle_source_publication(request=_request(), services=_services())

    assert raised.value is defect


def test_source_settlement_seal_preserves_independent_inventory_observations(
    monkeypatch,
):
    api = _api()
    events = []
    clock = _Clock()
    _install_late_artifact_fakes(monkeypatch, events)

    result = api.settle_source_publication(
        request=_request(), services=_services(clock=clock)
    )

    assert result.disposition is api.SourceSettlementDisposition.SEALED
    assert result.source_settlement is not None
    seal = result.source_settlement.to_dict()
    first = seal["first_inventory_observation"]
    second = seal["second_inventory_observation"]
    observation_fields = {
        "observed_at",
        "inventory_sha256",
        "page_request_ids",
        "canonical_identity_sha256",
    }

    assert set(first) == observation_fields
    assert set(second) == observation_fields
    assert first != second
    assert first["inventory_sha256"] == seal["inventory_sha256"]
    assert second["inventory_sha256"] == seal["inventory_sha256"]
    assert first["observed_at"] == _utc(0)
    assert second["observed_at"] == _utc(10)
    assert first["observed_at"] < second["observed_at"]
    assert clock.sleeps == [10]
    assert first["page_request_ids"] == [
        f"list-1-{family_index}-first"
        for family_index in range(len(api.SOURCE_FAMILY_ORDER))
    ]
    assert second["page_request_ids"] == [
        f"list-2-{family_index}-first"
        for family_index in range(len(api.SOURCE_FAMILY_ORDER))
    ]
    assert set(first["page_request_ids"]).isdisjoint(second["page_request_ids"])
    for observation in (first, second):
        body = dict(observation)
        identity = body.pop("canonical_identity_sha256")
        assert identity == canonical_sha256(body)


@pytest.mark.parametrize(
    ("status", "classification"),
    (
        ("FAILED", "SOURCE_ACTION_FAILED"),
        ("TIMED_OUT", "SOURCE_ACTION_TIMED_OUT"),
        ("CANCELLED", "SOURCE_ACTION_CANCELLED"),
        ("UNKNOWN_PRE_FREEZE_INVOCATION", "UNKNOWN_PRE_FREEZE_INVOCATION"),
    ),
)
def test_every_non_success_terminal_exit_selects_bootstrap_closed_source(
    monkeypatch, status, classification
):
    api = _api()
    evidence = _Evidence(status=status)
    events = []
    _install_late_artifact_fakes(monkeypatch, events)

    result = api.settle_source_publication(
        request=_request(), services=_services(evidence=evidence)
    )

    assert result.disposition is api.SourceSettlementDisposition.CLOSED_SOURCE
    assert result.classification == classification
    assert result.selected_entry.slot.value == "CLOSED_SOURCE"
    assert result.source_settlement is None
    assert result.publication_coordinates == ()
    assert result.no_launch_evidence["no_batch"] is True
    assert result.no_launch_evidence["no_launch"] is True
    assert events == []


@pytest.mark.parametrize("mutant", ("short-interval", "policy", "sid", "role"))
def test_freeze_probe_rounds_are_temporally_and_principally_bound(monkeypatch, mutant):
    api = _api()
    evidence = _Evidence()
    probes = [dict(row) for row in evidence.probes]
    if mutant == "short-interval":
        probes[1]["observed_at"] = _utc(9)
    elif mutant == "policy":
        probes[1]["policy_sha256"] = SHA("0")
    else:
        attribution = [dict(row) for row in probes[1]["attribution_rows"]]
        if mutant == "sid":
            attribution[0]["expected_sid"] = "DenyTLS"
        else:
            attribution[0]["publisher_role_id"] = "AROADRIFTED"
        attribution[0]["canonical_identity_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in attribution[0].items()
                if key != "canonical_identity_sha256"
            }
        )
        probes[1]["attribution_rows"] = tuple(attribution)
    probes[1]["canonical_identity_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in probes[1].items()
            if key != "canonical_identity_sha256"
        }
    )
    evidence.probes = tuple(probes)
    events = []
    _install_late_artifact_fakes(monkeypatch, events)

    result = api.settle_source_publication(
        request=_request(), services=_services(evidence=evidence)
    )

    assert result.disposition is api.SourceSettlementDisposition.CLOSED_SOURCE
    assert result.classification == "FREEZE_DENIAL_UNPROVED"
    assert events == []


@pytest.mark.parametrize("mutant", ("partial-marker", "cycle", "second-pass"))
def test_inventory_pagination_cycles_and_double_pass_drift_fail_closed(
    monkeypatch, mutant
):
    api = _api()
    s3 = _S3()
    s3.mutant = mutant
    events = []
    _install_late_artifact_fakes(monkeypatch, events)

    result = api.settle_source_publication(
        request=_request(), services=_services(s3=s3)
    )

    assert result.disposition is api.SourceSettlementDisposition.CLOSED_SOURCE
    assert result.classification == "SOURCE_INVENTORY_UNPROVED"
    assert result.no_launch_evidence["no_batch"] is True
    assert events == []


def test_post_seal_evidence_drift_never_renders_or_publishes(monkeypatch):
    api = _api()
    evidence = _Evidence()
    evidence.drift_after_first = "workflow"
    events = []
    _install_late_artifact_fakes(monkeypatch, events)

    result = api.settle_source_publication(
        request=_request(), services=_services(evidence=evidence)
    )

    assert result.disposition is api.SourceSettlementDisposition.CLOSED_SOURCE
    assert result.classification == "SOURCE_SETTLEMENT_DRIFT"
    assert result.selected_entry.slot.value == "CLOSED_SOURCE"
    assert result.publication_coordinates == ()
    assert events == []


def test_old_direct_batch_successor_is_not_source_authority(monkeypatch):
    from glm52_enforcement.source_publishers import BatchSuccessorResult

    api = _api()
    evidence = _Evidence()
    evidence.reachability = BatchSuccessorResult(
        selected_successor_count=1,
        selected_precreated_successor=True,
        family_closing_policy_applied=True,
        activated_sources=(),
        denied_publisher_roles=(),
        probe_sets=(),
        fresh_h1f_unique_zero_child_active_head=True,
        fresh_h1f_audit_body_sha256=SHA("1"),
        active_head_body_sha256=SHA("2"),
        active_head_version_id="legacy",
    )
    events = []
    _install_late_artifact_fakes(monkeypatch, events)

    result = api.settle_source_publication(
        request=_request(), services=_services(evidence=evidence)
    )

    assert result.disposition is api.SourceSettlementDisposition.CLOSED_SOURCE
    assert result.classification == "PUBLISHER_REACHABILITY_UNPROVED"
    assert events == []


def test_unresolved_invocation_is_ineligible_even_when_inventory_is_stable(monkeypatch):
    api = _api()
    evidence = _Evidence()
    invocations = list(evidence.invocations)
    unresolved = dict(invocations[2])
    unresolved.update(
        {
            "terminal_status": "UNKNOWN_PRE_FREEZE_INVOCATION",
            "response_identity_sha256": None,
            "s3_request_id": None,
            "version_id": None,
        }
    )
    unresolved["canonical_identity_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in unresolved.items()
            if key != "canonical_identity_sha256"
        }
    )
    invocations[2] = unresolved
    evidence.invocations = tuple(invocations)
    events = []
    _install_late_artifact_fakes(monkeypatch, events)

    result = api.settle_source_publication(
        request=_request(), services=_services(evidence=evidence)
    )

    failure = result.no_launch_evidence["failure_evidence"]
    assert failure["freeze_execution_evidence"]["slot"] == "SOURCE_FAMILIES_FROZEN"
    assert len(failure["freeze_denial_probe_rows"]) == 2
    assert (
        failure["source_lambda_execution_terminal_rows"][2]["terminal_status"]
        == "UNKNOWN_PRE_FREEZE_INVOCATION"
    )
    assert result.classification == "UNKNOWN_PRE_FREEZE_INVOCATION"
    assert result.disposition is api.SourceSettlementDisposition.CLOSED_SOURCE
    assert result.source_settlement is None
    assert events == []


@pytest.mark.parametrize("collection", ("actions", "invocations"))
def test_terminal_action_and_invocation_sets_are_exact(monkeypatch, collection):
    api = _api()
    evidence = _Evidence()
    setattr(evidence, collection, getattr(evidence, collection)[:-1])
    events = []
    _install_late_artifact_fakes(monkeypatch, events)

    result = api.settle_source_publication(
        request=_request(), services=_services(evidence=evidence)
    )

    assert result.classification == "SOURCE_TERMINALITY_UNPROVED"
    assert result.disposition is api.SourceSettlementDisposition.CLOSED_SOURCE
    assert events == []


@pytest.mark.parametrize(
    "field",
    ("publisher_role_arn", "publisher_role_id", "s3_request_id", "version_id"),
)
def test_selected_publisher_identity_rows_are_fully_bound(monkeypatch, field):
    api = _api()
    evidence = _Evidence()
    selected = list(evidence.selected)
    changed = dict(selected[0])
    changed[field] = (
        f"arn:aws:iam::{ACCOUNT_ID}:role/drifted"
        if field == "publisher_role_arn"
        else "drifted"
    )
    changed["canonical_identity_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in changed.items()
            if key != "canonical_identity_sha256"
        }
    )
    selected[0] = changed
    evidence.selected = tuple(selected)
    events = []
    _install_late_artifact_fakes(monkeypatch, events)

    result = api.settle_source_publication(
        request=_request(), services=_services(evidence=evidence)
    )

    assert result.disposition is api.SourceSettlementDisposition.CLOSED_SOURCE
    assert result.no_launch_evidence["no_batch"] is True
    assert events == []


def test_successful_pagination_preserves_nonselected_delete_markers(monkeypatch):
    api = _api()
    s3 = _S3()
    s3.mutant = "paged"
    events = []
    _install_late_artifact_fakes(monkeypatch, events)

    delete_key = (
        api.SOURCE_FAMILY_PREFIXES["gpu-spend-snapshot"]
        + SHA("1")
        + "/GPU_SPEND_SNAPSHOT.json"
    )
    result = api.settle_source_publication(
        request=_request(nonselected_keys=(delete_key,)),
        services=_services(s3=s3),
    )

    assert result.disposition is api.SourceSettlementDisposition.SEALED
    nonselected = result.source_settlement.to_dict()["nonselected_provisional_rows"]
    assert len(nonselected) == 1
    assert nonselected[0]["is_delete_marker"] is True
    assert any("KeyMarker" in request for request in s3.calls)
