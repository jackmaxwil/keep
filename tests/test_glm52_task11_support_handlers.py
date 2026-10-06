"""Task 11 exact-version support-callee boundary tests."""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256
from glm52_enforcement.s3_adapter import S3PublicationServices
from glm52_enforcement.s3_records import S3ObjectIdentity
from glm52_enforcement.source_publishers import SourcePublicationResult
from glm52_enforcement.support_source_publisher_handler import (
    SourcePublisherHandlerServices,
    main as publish_source,
)
from glm52_enforcement import task11_relay_runtime
from glm52_enforcement.task11_relay_runtime import (
    AdmissionDynamoStore,
    AdmissionRequest,
)


ACCOUNT_ID = "246813579024"
ACTIVATION_ID = "activation-0001"
BUCKET = "keep-glm52-production"
DEPLOYMENT_SHA = hashlib.sha256(b"deployment").hexdigest()


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"ResponseMetadata": []},
        {"ResponseMetadata": {"HTTPStatusCode": 200}},
        {
            "ResponseMetadata": {
                "HTTPStatusCode": True,
                "RequestId": "request-1",
            }
        },
        {
            "ResponseMetadata": {
                "HTTPStatusCode": 500,
                "RequestId": "request-1",
            }
        },
        {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "",
            }
        },
        {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "request-1",
            },
            "ConsumedCapacity": {},
        },
        {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "request-1",
            },
            "ItemCollectionMetrics": [],
        },
        [],
    ],
)
@pytest.mark.parametrize(
    ("before_state", "after_state", "domain"),
    [
        ("CONSUMED", "POST_STARTED", "TASK11_POST_STARTED"),
        ("POST_STARTED", "POST_AUTHORIZED", "TASK11_POST_AUTHORIZED"),
        ("POST_AUTHORIZED", "POST_CLASSIFIED", "TASK11_POST_CLASSIFIED"),
    ],
)
def test_task11_admission_malformed_committed_response_reconciles_once(
    monkeypatch: pytest.MonkeyPatch,
    response: object,
    before_state: str,
    after_state: str,
    domain: str,
) -> None:
    class Client:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def transact_write_items(self, **request: object) -> object:
            self.calls.append(dict(request))
            return response

    request = AdmissionRequest(
        account_id=ACCOUNT_ID,
        region="us-west-2",
        run_id="glm52-sky-20260724",
        activation_id=ACTIVATION_ID,
        generation=1,
        attempt=1,
        action_key="ACTION#00000001#SKY_POST#00000001",
        request_body_sha256="1" * 64,
        relay_envelope_sha256="2" * 64,
        attestation_identity_sha256="3" * 64,
        direct_decision_identity_sha256="4" * 64,
        live_h1d_identity_sha256="5" * 64,
        decision_seal_identity_sha256="6" * 64,
        decision_nonce_sha256="7" * 64,
        barrier_transition_identity_sha256="8" * 64,
        private_nonce=b"private-admission-nonce",
    )
    index = {
        "current_activation_id": ACTIVATION_ID,
        "campaign_identity_sha256": "9" * 64,
        "revision": 3,
    }
    action = {
        "state": before_state,
        "revision": 4,
        "campaign_identity_sha256": "9" * 64,
        "activation_id": ACTIVATION_ID,
        "activation_ordinal": 1,
        "generation": 1,
        "action_kind": "SKY_POST",
        "attempt": 1,
        "candidate_key": "candidate",
        "candidate_file_sha256": "a" * 64,
        "candidate_body_sha256": "b" * 64,
        "request_body_sha256": "1" * 64,
        "relay_envelope_sha256": "2" * 64,
        "owner_epoch": 1,
        "owner_execution_arn": "execution",
        "barrier_nonce_sha256": "c" * 64,
        "owner_invocation_nonce_sha256": "d" * 64,
        "consumed_at": "2026-07-29T12:00:00Z",
    }
    after = {**action, "state": after_state, "revision": 5}
    control = {
        "last_sky_post_state": before_state,
        "revision": 8,
        "campaign_identity_sha256": "9" * 64,
        "activation_id": ACTIVATION_ID,
        "activation_ordinal": 1,
        "active_epoch": 1,
        "active_execution_arn": "execution",
        "active_state_machine_version_arn": "version",
        "phase": "TASK11",
        "fence_head_body_sha256": "e" * 64,
        "fence_head_version_id": "version-1",
        "barrier_nonce_sha256": "c" * 64,
        "barrier_state": "ACQUIRED",
        "decision_seal_state": "SEALED",
    }
    control_after = {
        **control,
        "last_sky_post_state": after_state,
        "revision": 9,
    }
    client = Client()
    store = object.__new__(AdmissionDynamoStore)
    store._client = client
    store._table = "ledger-table"
    store._activation = ACTIVATION_ID
    store.states = []
    store._read = lambda requested: (index, control_after, after)
    monkeypatch.setattr(
        task11_relay_runtime, "validate_record", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        task11_relay_runtime,
        "validate_transition",
        lambda *args, **kwargs: None,
    )

    assert (
        store._transition(
            request=request,
            index_before=index,
            before=action,
            after=after,
            control_before=control,
            control_after=control_after,
            domain=domain,
        )
        == after_state
    )
    assert len(client.calls) == 1
    assert store.states == [after_state]


class ExactInputS3:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw
        self.calls = []

    def get_object(self, **kwargs: object):
        self.calls.append(dict(kwargs))
        return {
            "Body": io.BytesIO(self.raw),
            "VersionId": "source-input-v1",
            "ChecksumSHA256": base64.b64encode(
                hashlib.sha256(self.raw).digest()
            ).decode("ascii"),
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "source-input-get-1",
            },
        }


def _source_input() -> dict[str, object]:
    body = {
        "schema_version": 1,
        "record_type": "glm52_task11_source_input_v1",
        "account_id": ACCOUNT_ID,
        "region": "us-west-2",
        "run_id": "glm52-sky-20260724",
        "activation_id": ACTIVATION_ID,
        "generation": 1,
        "source_kind": "GPU_SPEND",
        "action_key": (
            "ACTIVATION#activation-0001#ACTION#S3_CREATE#GPU_SPEND"
        ),
        "source_template": {
            "schema_version": 1,
            "record_type": "test-gpu-spend",
            "snapshot_body_sha256": "0" * 64,
        },
        "descriptor": None,
    }
    return {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }


def _coordinate(raw: bytes) -> dict[str, object]:
    body = {
        "input_kind": "GPU_SPEND_SOURCE_REQUEST",
        "bucket": BUCKET,
        "key": (
            "campaigns/glm52-sky-20260724/authorities/task11/"
            "activation-0001/00000001/01-gpu-spend-source-request.json"
        ),
        "version_id": "source-input-v1",
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "body_sha256": _source_input()["canonical_identity_sha256"],
    }
    return {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }


def _config() -> dict[str, str]:
    return {
        "account_id": ACCOUNT_ID,
        "region": "us-west-2",
        "run_id": "glm52-sky-20260724",
        "activation_id": ACTIVATION_ID,
        "campaign_bucket": BUCKET,
        "closure_role_arn": (
            "arn:aws:iam::246813579024:role/keep-glm52-h1g-decision"
        ),
        "publisher_role_arn": (
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-h1g-source-gpu-spend"
        ),
        "source_name": "SourceGpuSpend",
        "deployment_identity_sha256": DEPLOYMENT_SHA,
    }


def test_task11_fix1_source_handler_loads_pin_and_calls_task5_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_input()
    raw = canonical_json_bytes(source) + b"\n"
    s3 = ExactInputS3(raw)
    calls = []

    def accepted_publish(*, runtime, request):
        calls.append((runtime, request))
        file_sha = hashlib.sha256(request.raw).hexdigest()
        parsed = __import__("json").loads(request.raw[:-1])
        body_sha = parsed["snapshot_body_sha256"]
        return SourcePublicationResult(
            source_kind="gpu-spend-snapshot",
            publisher_role_arn=runtime.publisher_role_arn,
            object_identity=S3ObjectIdentity(
                bucket=BUCKET,
                key=(
                    "campaigns/glm52-sky-20260724/ledger/"
                    "gpu-spend-snapshot.json"
                ),
                version_id="service-assigned-version-1",
                file_sha256=file_sha,
                body_sha256=body_sha,
                content_length=len(request.raw),
                etag='"etag-1"',
                last_modified="2026-07-29T12:00:00Z",
                checksum_sha256_base64=base64.b64encode(
                    hashlib.sha256(request.raw).digest()
                ).decode("ascii"),
                checksum_type="FULL_OBJECT",
                content_type="application/json",
                metadata=(),
                canonical_identity_sha256=hashlib.sha256(
                    b"object"
                ).hexdigest(),
            ),
            candidate_identity_sha256=hashlib.sha256(
                b"candidate"
            ).hexdigest(),
            provenance="direct-response",
            authority_audit_body_sha256=hashlib.sha256(
                b"audit"
            ).hexdigest(),
            closing_revision=7,
            authorized_revision=8,
            direct_request_id="put-source-1",
            direct_server_date="2026-07-29T12:00:00Z",
            direct_response_authenticated=True,
        )

    monkeypatch.setattr(
        "glm52_enforcement.support_source_publisher_handler."
        "publish_gpu_spend_snapshot",
        accepted_publish,
    )
    result = publish_source(
        {
            "schema_version": 1,
            "record_type": "glm52_task11_source_publisher_request_v1",
            "activation_id": ACTIVATION_ID,
            "generation": 1,
            "source_kind": "GPU_SPEND",
            "predecessor_version_id": "initial-v1",
            "input_coordinate": _coordinate(raw),
            "prior_publications": [],
            "custody_nonce_sha256": hashlib.sha256(
                b"custody"
            ).hexdigest(),
        },
        object(),
        services=SourcePublisherHandlerServices(
            s3=s3,
            publication=S3PublicationServices(
                s3=object(),
                actions=object(),
                h1f=object(),
            ),
        ),
        config=_config(),
    )
    assert len(calls) == 1
    assert result["publication"]["source_kind"] == "GPU_SPEND"
    assert result["publication"]["version_id"] == "service-assigned-version-1"
    assert result["publication"]["direct_response_authenticated"] is True
    assert s3.calls == [
        {
            "Bucket": BUCKET,
            "Key": _coordinate(raw)["key"],
            "VersionId": "source-input-v1",
            "ExpectedBucketOwner": ACCOUNT_ID,
            "ChecksumMode": "ENABLED",
        }
    ]


def test_task11_fix1_source_handler_rejects_caller_outcome() -> None:
    source = _source_input()
    raw = canonical_json_bytes(source) + b"\n"
    event = {
        "schema_version": 1,
        "record_type": "glm52_task11_source_publisher_request_v1",
        "activation_id": ACTIVATION_ID,
        "generation": 1,
        "source_kind": "GPU_SPEND",
        "predecessor_version_id": "initial-v1",
        "input_coordinate": _coordinate(raw),
        "prior_publications": [],
        "custody_nonce_sha256": hashlib.sha256(b"custody").hexdigest(),
        "outcome": "created-direct",
    }
    with pytest.raises(ValueError, match="not closed"):
        publish_source(event, object(), config=_config())


def test_task11_fix1_support_timeouts_are_frozen() -> None:
    from glm52_enforcement.support_plane import RUNTIME_FUNCTIONS

    for name in (
        "SourceGpuSpend",
        "SourceSubmissionIntent",
        "SourceControllerBaseline",
        "SourceControlPlaneReadiness",
        "SourceSubmissionAcquisition",
    ):
        assert RUNTIME_FUNCTIONS[name]["timeout"] == 5
    assert RUNTIME_FUNCTIONS["FenceExecutor"]["timeout"] == 12
    for name in (
        "FenceSuccessor",
        "ClaimWriter",
        "DecisionWriter",
        "TerminalV1Writer",
        "ClosureHandoff",
    ):
        assert RUNTIME_FUNCTIONS[name]["timeout"] == 8
    assert RUNTIME_FUNCTIONS["Attestation"]["timeout"] == 15
    assert RUNTIME_FUNCTIONS["LaunchAdmission"]["timeout"] == 25


def test_task11_fix1_privileged_import_graph_blocks_mlx_vq() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    code = r"""
import builtins
import importlib
import sys

sys.path.insert(0, sys.argv[1])
real_import = builtins.__import__

def blocked(name, *args, **kwargs):
    if name.startswith(("mlx_vq", "numpy")):
        raise AssertionError("privileged import crossed into " + name)
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked
names = (
    "glm52_enforcement.task11_effect_writers",
    "glm52_enforcement.glm52_sky_production_generation",
    "glm52_enforcement.glm52_sky_production_fence",
    "glm52_enforcement.glm52_qualification_cache_seed",
)
imported = tuple(importlib.import_module(name) for name in names)
assert callable(imported[0].write_generation_claim)
assert callable(imported[1].build_production_generation_claim)
assert callable(imported[2].build_fence_successor)
assert callable(imported[3].validate_qualification_cache_seed_accepted)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, str(source_root)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_task11_runtime_archive_contains_no_concrete_deployed_identity(
    tmp_path: Path,
) -> None:
    """Break caught: a predeployment zip embeds impossible live identity."""

    root = Path(__file__).resolve().parents[1]
    archive = tmp_path / "support.zip"
    package = root / "aws/glm52-gpu/scripts/package_h1g_support_lambdas.py"
    build = subprocess.run(
        [sys.executable, str(package), str(archive)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    with zipfile.ZipFile(archive) as packaged:
        names = set(packaged.namelist())
        assert "glm52_enforcement/task9-contract-v1.json" not in names
        for name in names:
            if name.endswith(".json"):
                raw = packaged.read(name)
                assert b"glm52_task9_deployed_identity_v1" not in raw
                assert b"combined_host_instance_id" not in raw


def test_task11_isolated_archive_exact_loads_deployed_task9_identity(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    archive = tmp_path / "support.zip"
    package = root / "aws/glm52-gpu/scripts/package_h1g_support_lambdas.py"
    build = subprocess.run(
        [sys.executable, str(package), str(archive)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    script = """
import base64, hashlib, io, sys
sys.path.insert(0, sys.argv[1])
from glm52_enforcement.canonical import canonical_json_bytes
from glm52_enforcement.task9_contract import (
    build_task9_contract, build_task9_deployed_identity,
)
from glm52_enforcement.task11_boundary import build_task11_input_coordinate
from glm52_enforcement.task11_relay_runtime import (
    load_task9_deployed_identity,
)
activation = "activation-0001"
bucket = "keep-glm52-production"
static = build_task9_contract(__import__("pathlib").Path("/isolated/archive"))
relays = []
for index, (purpose, policy) in enumerate(static["relay_policy"].items(), 1):
    relays.append({
        "purpose": purpose,
        "client_secret_arn": (
            "arn:aws:secretsmanager:us-west-2:246813579024:secret:"
            "/keep/glm52/glm52-sky-20260724/" + activation + "/"
            + purpose.lower() + "-client-ABC123"
        ),
        "client_secret_version_id": str(index) * 64,
        "client_secret_version_stage": (
            "h1g-" + activation + "-"
            + purpose.lower().replace("_", "-") + "-client-tls-v1"
        ),
        "client_certificate_der_sha256": str(index) * 64,
        "server_secret_arn": (
            "arn:aws:secretsmanager:us-west-2:246813579024:secret:"
            "/keep/glm52/glm52-sky-20260724/" + activation
            + "/combined-host-server-tls-ABC123"
        ),
        "server_secret_version_id": "9" * 64,
        "server_secret_version_stage": (
            "h1g-" + activation + "-combined-host-server-tls-v1"
        ),
        "server_certificate_der_sha256": str(index + 4) * 64,
        "port": policy["port"],
        "principal_arn": policy["principal_arn"],
        "allowed_paths": policy["allowed_paths"],
    })
document = build_task9_deployed_identity(
    static_contract=static,
    activation_id=activation,
    support_binding={
        "support_stack_id": (
            "arn:aws:cloudformation:us-west-2:246813579024:stack/"
            "keep-glm52-h1g-support/"
            "11111111-2222-4333-8444-555555555555"
        ),
        "support_template_body_sha256": "a" * 64,
        "postcreate_manifest_sha256": "b" * 64,
        "describe_stacks_response_sha256": "c" * 64,
    },
    combined_host={
        "instance_id": "i-0123456789abcdef0",
        "private_ip": "10.20.101.10",
        "ami_id": "ami-0123456789abcdef0",
        "instance_profile_arn": (
            "arn:aws:iam::246813579024:instance-profile/h1g-host"
        ),
        "boot_identity_sha256": "d" * 64,
        "service_identity_sha256": "e" * 64,
        "consolidation_signal_identity_sha256": "f" * 64,
    },
    tls={
        "issuance_id": "iss-" + "9" * 64,
        "bundle_sha256": "8" * 64,
        "ca_certificate_der_sha256": "7" * 64,
        "not_valid_before": "2026-07-29T12:00:00Z",
        "not_valid_after": "2026-08-01T12:00:00Z",
        "relays": relays,
    },
)
raw = canonical_json_bytes(document) + b"\\n"
coordinate = build_task11_input_coordinate(
    input_kind="TASK9_DEPLOYED_IDENTITY_COORDINATE",
    bucket=bucket,
    key=(
        "campaigns/glm52-sky-20260724/authorities/task9/"
        + activation + "/TASK9_DEPLOYED_IDENTITY.json"
    ),
    version_id="opaque-version-1",
    file_sha256=hashlib.sha256(raw).hexdigest(),
    body_sha256=document["canonical_identity_sha256"],
)
class S3:
    def get_object(self, **request):
        assert request["ExpectedBucketOwner"] == "246813579024"
        assert request["VersionId"] == "opaque-version-1"
        return {
            "Body": io.BytesIO(raw),
            "VersionId": "opaque-version-1",
            "ChecksumSHA256": base64.b64encode(
                hashlib.sha256(raw).digest()
            ).decode("ascii"),
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "isolated-loader-1",
            },
        }
loaded = load_task9_deployed_identity(
    s3=S3(),
    coordinate=__import__("dataclasses").asdict(coordinate),
    activation_id=activation,
    expected_body_sha256=document["canonical_identity_sha256"],
    expected_bucket=bucket,
)
assert loaded.document == document
assert loaded.sky_identity.combined_host_instance_id == "i-0123456789abcdef0"
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(archive)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_task11_fix1_legacy_paths_are_exact_compatibility_reexports() -> None:
    from glm52_enforcement import glm52_sky_production_fence as native_fence
    from glm52_enforcement import (
        glm52_sky_production_generation as native_generation,
    )
    from mlx_vq.quality import glm52_sky_production_fence as legacy_fence
    from mlx_vq.quality import (
        glm52_sky_production_generation as legacy_generation,
    )

    assert (
        legacy_fence.build_fence_successor
        is native_fence.build_fence_successor
    )
    assert (
        legacy_generation.build_production_generation_claim
        is native_generation.build_production_generation_claim
    )
    assert (
        legacy_generation.validate_production_generation_terminal
        is native_generation.validate_production_generation_terminal
    )
