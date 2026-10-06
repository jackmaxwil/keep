from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
from pathlib import Path
import ssl
from types import SimpleNamespace

import pytest

from glm52_enforcement.launch_custody import (
    ACCOUNT_ID,
    REGION,
    RUN_ID,
    AmbiguousRunInstances,
    AttemptPermit,
    LaunchParameterAuthority,
    LaunchCustodyError,
    PositiveRunInstancesRejection,
    SameTokenCompleter,
    build_deterministic_client_token,
    build_launch_parameters,
)
from glm52_enforcement.sky_admission import PinnedSkyIdentity
from glm52_enforcement.task9_contract import build_task9_contract


ROOT = Path(__file__).parents[1]
ADAPTER_PATH = (
    ROOT
    / "aws/glm52-gpu/lambda/worker_launch_custody_handler.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "task9_worker_launch_custody_handler", ADAPTER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_module():
    path = ROOT / "aws/glm52-gpu/skypilot/worker_launch_intent_patch.py"
    spec = importlib.util.spec_from_file_location("task9_sky_patch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Sts:
    def __init__(self, events, account="246813579024"):
        self.events = events
        self.account = account

    def get_caller_identity(self):
        self.events.append("STS")
        return {
            "Account": self.account,
            "Arn": "arn:aws:iam::246813579024:role/task9",
            "UserId": "task9",
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "sts-1",
            },
        }


class _Ec2:
    def __init__(self, events, effect="success", attempts=1):
        self.events = events
        self.effect = effect
        self.meta = SimpleNamespace(
            config=SimpleNamespace(
                retries={"mode": "standard", "total_max_attempts": attempts}
            )
        )

    def run_instances(self, **request):
        self.events.append(("RUN", request))
        if self.effect == "timeout":
            raise TimeoutError("lost")
        if self.effect == "reject":
            error = RuntimeError("reject")
            error.response = {
                "ResponseMetadata": {"HTTPStatusCode": 400},
                "Error": {"Code": "InsufficientInstanceCapacity"},
            }
            raise error
        return {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "run-1",
            },
            "Instances": [{"InstanceId": "i-0123456789abcdef0"}],
        }

    def terminate_instances(self, **request):
        self.events.append(("TERMINATE", request))
        return {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "terminate-1",
            },
            "TerminatingInstances": [
                {
                    "InstanceId": request["InstanceIds"][0],
                    "CurrentState": {"Name": "shutting-down"},
                }
            ],
        }


def test_effect_adapter_sts_is_first_and_foreign_account_blocks_ec2() -> None:
    module = _module()
    events = []
    guard = module.AwsAccountRegionGuard(
        sts=_Sts(events, account="000000000000"),
        region="us-west-2",
    )
    adapter = module.ZeroRetryEc2RunInstances(
        guard=guard,
        ec2=_Ec2(events),
    )
    with pytest.raises(LaunchCustodyError, match="account"):
        adapter.run_instances(ClientToken="1" * 64)
    assert events == ["STS"]


def test_published_function_and_resource_policy_guard_fail_closed() -> None:
    module = _module()
    expected_arn = (
        "arn:aws:lambda:us-west-2:246813579024:"
        "function:glm52-same-token-completer:7"
    )

    class Identity:
        def exact_read(self):
            return {
                "published_version_arn": expected_arn,
                "resource_policy_identity_sha256": "1" * 64,
            }

    guard = module.PublishedFunctionIdentityGuard(
        expected_published_version_arn=expected_arn,
        expected_resource_policy_identity_sha256="1" * 64,
        identity_reader=Identity(),
    )
    guard.prove(SimpleNamespace(invoked_function_arn=expected_arn))
    with pytest.raises(LaunchCustodyError, match="published"):
        guard.prove(
            SimpleNamespace(
                invoked_function_arn=expected_arn.rsplit(":", 1)[0] + ":$LATEST"
            )
        )


@pytest.mark.parametrize(
    ("function_arn", "policy_identity"),
    (
        (
            "arn:aws:lambda:us-west-2:246813579024:"
            "function:glm52-same-token-completer",
            "1" * 64,
        ),
        (
            "arn:aws:lambda:us-west-2:246813579024:"
            "function:glm52-same-token-completer:alias",
            "1" * 64,
        ),
        (
            "arn:aws:lambda:us-west-2:246813579024:"
            "function:glm52-same-token-completer:$LATEST",
            "1" * 64,
        ),
        (
            "arn:aws:lambda:us-west-2:246813579024:"
            "function:glm52-same-token-completer:7:8",
            "1" * 64,
        ),
        (
            "arn:aws:lambda:us-west-2:246813579024:"
            "function:glm52-same-token-completer:7",
            "z" * 64,
        ),
        (
            "arn:aws:lambda:us-west-2:246813579024:"
            "function:glm52-same-token-completer:7",
            "A" * 64,
        ),
    ),
)
def test_published_function_guard_rejects_nonversion_or_nonsha_identity(
    function_arn: str,
    policy_identity: str,
) -> None:
    module = _module()
    with pytest.raises(
        LaunchCustodyError,
        match="published function identity is invalid",
    ):
        module.PublishedFunctionIdentityGuard(
            expected_published_version_arn=function_arn,
            expected_resource_policy_identity_sha256=policy_identity,
            identity_reader=object(),
        )


def test_effect_dispatcher_checks_published_identity_before_closed_core() -> None:
    module = _module()
    events = []

    class Guard:
        def prove(self, context):
            events.append(("IDENTITY", context))

    class Completer:
        def complete(self, value):
            events.append(("COMPLETE", value))
            return SimpleNamespace(
                classification="AMBIGUOUS",
                attempt=1,
                instance_ids=(),
                evidence_identity_sha256="1" * 64,
            )

    dispatcher = module.Task9EffectDispatcher(
        function_identity_guard=Guard(),
        same_token_completer=Completer(),
        liability_watcher=object(),
    )
    context = object()
    result = dispatcher.handle(
        {
            "mode": "SAME_TOKEN_COMPLETE",
            "activation_id": "activation-0001",
            "allocation_ordinal": 1,
        },
        context,
    )
    assert result["classification"] == "AMBIGUOUS"
    assert events == [
        ("IDENTITY", context),
        (
            "COMPLETE",
            {
                "activation_id": "activation-0001",
                "allocation_ordinal": 1,
            },
        ),
    ]
    with pytest.raises(LaunchCustodyError, match="closed"):
        dispatcher.handle(
            {
                "mode": "SAME_TOKEN_COMPLETE",
                "activation_id": "activation-0001",
                "allocation_ordinal": 1,
                "ClientToken": "caller-forged",
            },
            context,
        )


def test_task3_production_store_implements_every_custody_core_operation() -> None:
    module = _module()
    required = {
        "allocate_context",
        "put_prepared",
        "exact_read_launch",
        "bind_committed_wal",
        "commit_possibly_sent",
        "wait_watching",
        "signal_ready",
        "poll_result",
        "read_completion_authority",
        "consume_attempt",
        "record_direct",
        "record_ambiguous",
        "record_rejection",
        "acquire_or_takeover",
        "record_scan",
        "record_incident",
        "open_late_allocation",
        "consume_termination",
        "reconcile_instance",
        "settle_if_complete",
        "read_complete_view",
        "transact_settle_once",
        "coherent_read_settlement",
    }
    assert required <= set(vars(module.Task3LaunchCustodyStore))


def test_production_dispatcher_constructor_wires_actual_task3_store() -> None:
    module = _module()
    clients = {}

    class Config:
        def __init__(self, **kwargs):
            self.retries = kwargs["retries"]

    class Boto:
        @staticmethod
        def client(name, **kwargs):
            if name == "sts":
                value = _Sts([])
            elif name == "ec2":
                value = _Ec2([])
            else:
                value = SimpleNamespace()
            clients[name] = (value, kwargs)
            return value

    dispatcher = module.build_production_task9_dispatcher(
        boto3_module=Boto,
        botocore_config_type=Config,
        table_name="glm52-ledger",
        ledger_plan_authority=object(),
        expected_published_version_arn=(
            "arn:aws:lambda:us-west-2:246813579024:"
            "function:glm52-launch-custody:7"
        ),
        expected_resource_policy_identity_sha256="1" * 64,
        function_identity_reader=object(),
        discovery=object(),
        instance_authority=object(),
        clock=lambda: datetime.now(timezone.utc),
    )

    assert isinstance(dispatcher, module.Task9EffectDispatcher)
    store = dispatcher._completer._store
    assert isinstance(store, module.Task3LaunchCustodyStore)
    assert dispatcher._watcher._store is store
    assert set(clients) == {"sts", "ec2", "dynamodb"}
    assert store._ledger._client is clients["dynamodb"][0]


def test_effect_adapter_one_attempt_direct_reject_and_ambiguity_matrix() -> None:
    module = _module()
    for effect, error in (
        ("reject", PositiveRunInstancesRejection),
        ("timeout", AmbiguousRunInstances),
    ):
        events = []
        adapter = module.ZeroRetryEc2RunInstances(
            guard=module.AwsAccountRegionGuard(
                sts=_Sts(events), region="us-west-2"
            ),
            ec2=_Ec2(events, effect=effect),
        )
        with pytest.raises(error):
            adapter.run_instances(ClientToken="1" * 64)
        assert events[0] == "STS"
        assert sum(item[0] == "RUN" for item in events if isinstance(item, tuple)) == 1


@pytest.mark.parametrize(
    ("status", "code", "expected"),
    (
        pytest.param(200, "Synthetic", AmbiguousRunInstances, id="status-200"),
        pytest.param(301, "Synthetic", AmbiguousRunInstances, id="status-301"),
        pytest.param(399, "Synthetic", AmbiguousRunInstances, id="status-399"),
        pytest.param(
            400,
            "InsufficientInstanceCapacity",
            PositiveRunInstancesRejection,
            id="status-400",
        ),
        pytest.param(
            499,
            "Synthetic",
            PositiveRunInstancesRejection,
            id="status-499",
        ),
        pytest.param(500, "Synthetic", AmbiguousRunInstances, id="status-500"),
        pytest.param(400, None, AmbiguousRunInstances, id="missing-code"),
        pytest.param(400, 7, AmbiguousRunInstances, id="nonstring-code"),
        pytest.param(400, "", AmbiguousRunInstances, id="empty-code"),
    ),
)
def test_ec2_exception_status_code_matrix_is_exact_nonempty_4xx_only(
    status: int,
    code: object,
    expected: type[Exception],
) -> None:
    module = _module()
    error = RuntimeError("synthetic EC2 exception")
    error_body = {} if code is None else {"Code": code}
    error.response = {
        "ResponseMetadata": {"HTTPStatusCode": status},
        "Error": error_body,
    }
    with pytest.raises(expected):
        module._classify_ec2_exception(error)


def _same_token_authority() -> LaunchParameterAuthority:
    return LaunchParameterAuthority(
        account_id=ACCOUNT_ID,
        region=REGION,
        run_id=RUN_ID,
        campaign_identity_sha256="1" * 64,
        activation_id="activation-0001",
        activation_ordinal=1,
        generation=1,
        action_key="ACTION#00000001#SKY_POST#00000001",
        sky_request_id="sky-request-0001",
        sky_job_name="glm52-production",
        sky_task_name="glm52-production",
        task_yaml_sha256="2" * 64,
        request_body_sha256="3" * 64,
        approved_ami_id="ami-0123456789abcdef0",
        subnet_id="subnet-0123456789abcdef0",
        security_group_id="sg-0123456789abcdef0",
        instance_profile_name="keep-glm52-gpu-worker",
        source_identity_sha256="4" * 64,
    )


class _SameTokenStatusStore:
    def __init__(self) -> None:
        authority = _same_token_authority()
        self.parameters = build_launch_parameters(
            authority, allocation_ordinal=1
        )
        self.token = build_deterministic_client_token(
            authority, 1, self.parameters
        )
        self.state = "WATCHING"
        self.recorded = None

    def read_completion_authority(self, activation_id, allocation_ordinal):
        return {
            "activation_id": activation_id,
            "allocation_ordinal": allocation_ordinal,
            "state": self.state,
            "same_token_completion_attempts": 0,
            "possibly_sent_at": "2026-07-29T12:00:00Z",
            "ec2_client_token": self.token,
            "launch_parameters": self.parameters,
            "current_owner": True,
            "reserve_held": True,
        }

    def consume_attempt(self, activation_id, allocation_ordinal):
        del activation_id, allocation_ordinal
        self.state = "SAME_TOKEN_COMPLETION"
        return AttemptPermit(True, 1, "5" * 64)

    def record_direct(self, permit, response):
        del permit, response
        raise AssertionError("exception-shaped EC2 response became direct")

    def record_ambiguous(self, permit, evidence):
        del permit, evidence
        self.recorded = "AMBIGUOUS"
        self.state = "WATCHING"

    def record_rejection(self, permit, evidence):
        del permit, evidence
        self.recorded = "POSITIVE_REJECTION"
        self.state = "REJECTION_PROVED_AWAITING_TERMINAL_V2"


class _ExceptionResponseEc2:
    def __init__(self, status: int) -> None:
        self.status = status
        self.meta = SimpleNamespace(
            config=SimpleNamespace(
                retries={"mode": "standard", "total_max_attempts": 1}
            )
        )

    def run_instances(self, **request):
        del request
        error = RuntimeError("synthetic EC2 exception")
        error.response = {
            "ResponseMetadata": {"HTTPStatusCode": self.status},
            "Error": {"Code": "Synthetic"},
        }
        raise error


@pytest.mark.parametrize(
    ("status", "classification", "durable_state"),
    (
        (301, "AMBIGUOUS", "WATCHING"),
        (
            400,
            "POSITIVE_REJECTION",
            "REJECTION_PROVED_AWAITING_TERMINAL_V2",
        ),
        (500, "AMBIGUOUS", "WATCHING"),
    ),
)
def test_same_token_completer_disables_only_for_real_4xx(
    status: int,
    classification: str,
    durable_state: str,
) -> None:
    module = _module()
    store = _SameTokenStatusStore()
    ec2 = module.ZeroRetryEc2RunInstances(
        guard=module.AwsAccountRegionGuard(
            sts=_Sts([]), region="us-west-2"
        ),
        ec2=_ExceptionResponseEc2(status),
    )
    result = SameTokenCompleter(
        store=store,
        ec2=ec2,
        clock=lambda: datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc),
    ).complete(
        {"activation_id": "activation-0001", "allocation_ordinal": 1}
    )
    assert result.classification == classification
    assert store.recorded == classification
    assert store.state == durable_state


def test_effect_adapter_rejects_any_sdk_retry_configuration() -> None:
    module = _module()
    with pytest.raises(LaunchCustodyError, match="one attempt"):
        module.ZeroRetryEc2RunInstances(
            guard=module.AwsAccountRegionGuard(
                sts=_Sts([]), region="us-west-2"
            ),
            ec2=_Ec2([], attempts=2),
        )


def test_termination_adapter_self_reads_exact_instance_before_effect() -> None:
    module = _module()
    events = []

    class Authority:
        def exact_read(self, instance_id):
            events.append(("READ_AUTHORITY", instance_id))
            return {
                "instance_id": instance_id,
                "termination_authorized": True,
            }

    instance_id = "i-0123456789abcdef0"
    adapter = module.ZeroRetryExactInstanceTerminator(
        guard=module.AwsAccountRegionGuard(
            sts=_Sts(events), region="us-west-2"
        ),
        ec2=_Ec2(events),
        instance_authority=Authority(),
    )
    response = adapter.terminate(
        instance_id,
        deadline=datetime.now(timezone.utc) + timedelta(seconds=20),
    )
    assert response["request_id"] == "terminate-1"
    assert events[:2] == ["STS", ("READ_AUTHORITY", instance_id)]
    assert events[2] == ("TERMINATE", {"InstanceIds": [instance_id]})


class _Socket:
    def __init__(self, certificate):
        self.certificate = certificate

    def getpeercert(self, binary_form=False):
        assert binary_form is True
        return self.certificate


class _Response:
    status = 202

    def read(self, limit):
        assert limit == 1_048_577
        return b'{"request_id":"sky-1"}'

    def getheader(self, name):
        assert name == "x-request-id"
        return "relay-1"


class _Connection:
    def __init__(self, certificate):
        self.sock = _Socket(certificate)
        self.events = []

    def connect(self):
        self.events.append("CONNECT")

    def request(self, method, path, body, headers):
        self.events.append((method, path, body, headers))

    def getresponse(self):
        self.events.append("RESPONSE")
        return _Response()

    def close(self):
        self.events.append("CLOSE")


def test_mtls_relay_is_fixed_host_port_path_and_one_request() -> None:
    module = _module()
    certificate = b"launch-relay-certificate"
    connection = _Connection(certificate)
    calls = []

    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        return connection

    relay = module.FixedMtlsLaunchRelay(
        connection_factory=factory,
        tls_context=ssl.create_default_context(),
        expected_peer_certificate_sha256=hashlib.sha256(
            certificate
        ).hexdigest(),
    )
    result = relay.send(b'{"closed":true}')
    assert result["status_code"] == 202
    assert calls == [
        (
            ("glm52-launch-relay.internal", 18444),
            {"timeout": 12, "context": relay._context},
        )
    ]
    requests = [item for item in connection.events if isinstance(item, tuple)]
    assert len(requests) == 1
    assert requests[0][0:2] == ("POST", "/jobs/launch")


def test_mtls_relay_rejects_foreign_peer_before_post() -> None:
    module = _module()
    connection = _Connection(b"foreign")
    relay = module.FixedMtlsLaunchRelay(
        connection_factory=lambda *args, **kwargs: connection,
        tls_context=ssl.create_default_context(),
        expected_peer_certificate_sha256="0" * 64,
    )
    with pytest.raises(LaunchCustodyError, match="certificate"):
        relay.send(b'{"closed":true}')
    assert not any(isinstance(item, tuple) for item in connection.events)


def test_task9_iam_boundary_does_not_invent_client_token_condition_key() -> None:
    source = ADAPTER_PATH.read_text()
    assert "ec2:ClientToken" not in source
    assert set(_module().ALLOWED_EFFECT_OPERATIONS) == {
        "sts:GetCallerIdentity",
        "ec2:RunInstances",
        "ec2:TerminateInstances",
    }


def test_sky_patch_authenticates_actual_loaded_module_and_full_runtime() -> None:
    patch = _patch_module()
    interpreter = Path(
        "/Users/jack.mazac/.local/share/keep/skypilot-0.13.0/bin/python"
    )
    site_packages = next(
        (interpreter.parent.parent / "lib").glob(
            "python*/site-packages"
        )
    )
    dist = site_packages / "skypilot-0.13.0.dist-info"
    original = site_packages / "sky/provision/aws/instance.py"
    runtime_paths = {
        "original_provisioner": original,
        "wheel_metadata": dist / "METADATA",
        "dist_record": dist / "RECORD",
        "interpreter": interpreter,
        "dependency_lock": (
            ROOT / "aws/glm52-gpu/skypilot/skypilot-0.13.0-lock.txt"
        ),
        "server_config": (
            ROOT / "aws/glm52-gpu/skypilot/server-config-v1.json"
        ),
        "jobs_server": site_packages / "sky/jobs/server/server.py",
        "patched_provisioner": (
            ROOT / "aws/glm52-gpu/skypilot/worker_launch_intent_patch.py"
        ),
    }
    identity = PinnedSkyIdentity(
        **build_task9_contract(ROOT)["pinned_sky"]
    )
    loaded = SimpleNamespace(
        __name__="sky.provision.aws.instance",
        __file__=str(original),
    )
    patch.sys = SimpleNamespace(executable=str(interpreter))
    patch.authenticate_pinned_runtime(
        module=loaded,
        identity=identity,
        runtime_paths=runtime_paths,
    )
    loaded.__file__ = str(ROOT / "foreign-instance.py")
    with pytest.raises(LaunchCustodyError, match="module|foreign"):
        patch.authenticate_pinned_runtime(
            module=loaded,
            identity=identity,
            runtime_paths=runtime_paths,
        )
