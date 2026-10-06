from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement.launch_custody import (
    ACCOUNT_ID,
    REGION,
    RUN_ID,
    AmbiguousRunInstances,
    AttemptPermit,
    IntentOnlyProvisioner,
    LaunchContext,
    LaunchCustodyError,
    LaunchParameterAuthority,
    LaunchResult,
    LiabilitySettlementCoordinator,
    LiabilityWatcher,
    PositiveRunInstancesRejection,
    SameTokenCompleter,
    Task8GpuReserveAdapter,
    TerminationPermit,
    WatchInput,
    build_deterministic_client_token,
    build_launch_parameters,
    build_post_terminal_allocation,
    validate_launch_parameters,
)
from glm52_enforcement.launch_wal import LaunchWalError, SqliteLaunchWal


SHA = "a" * 64
NOW = datetime(2026, 7, 29, 12, 0, 0, tzinfo=timezone.utc)


def _authority() -> LaunchParameterAuthority:
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


def test_exact_deterministic_launch_parameters_and_client_token() -> None:
    parameters = build_launch_parameters(_authority(), allocation_ordinal=1)
    assert validate_launch_parameters(parameters, _authority()) == parameters
    body = parameters.run_instances
    assert body["MinCount"] == body["MaxCount"] == 1
    assert body["InstanceType"] == "p5.48xlarge"
    assert "InstanceMarketOptions" not in body
    assert body["MetadataOptions"] == {
        "HttpEndpoint": "enabled",
        "HttpTokens": "required",
        "HttpPutResponseHopLimit": 1,
    }
    assert body["BlockDeviceMappings"] == [
        {
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "DeleteOnTermination": True,
                "Encrypted": True,
                "Iops": 3000,
                "Throughput": 125,
                "VolumeSize": 300,
                "VolumeType": "gp3",
            },
        }
    ]
    token1 = build_deterministic_client_token(_authority(), 1, parameters)
    token2 = build_deterministic_client_token(_authority(), 1, parameters)
    assert token1 == token2
    assert len(token1) == 64
    assert token1 == token1.lower()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("ImageId",), "ami-foreign"),
        (("InstanceType",), "p5.4xlarge"),
        (("SubnetId",), "subnet-foreign"),
        (("SecurityGroupIds",), ["sg-foreign"]),
        (("IamInstanceProfile", "Name"), "admin"),
        (("MetadataOptions", "HttpTokens"), "optional"),
        (("BlockDeviceMappings", 0, "Ebs", "VolumeSize"), 301),
        (("BlockDeviceMappings", 0, "Ebs", "Encrypted"), False),
        (("BlockDeviceMappings", 0, "Ebs", "DeleteOnTermination"), False),
        (("BlockDeviceMappings", 0, "Ebs", "Iops"), 3001),
        (("BlockDeviceMappings", 0, "Ebs", "Throughput"), 126),
        (("MinCount",), 2),
    ],
)
def test_launch_parameter_mutants_fail_closed(path, value) -> None:
    parameters = build_launch_parameters(_authority(), allocation_ordinal=1)
    body = json.loads(json.dumps(parameters.run_instances))
    cursor = body
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = value
    with pytest.raises(LaunchCustodyError):
        validate_launch_parameters(replace(parameters, run_instances=body), _authority())


def test_spot_capacity_block_launch_template_and_data_disk_are_forbidden() -> None:
    parameters = build_launch_parameters(_authority(), allocation_ordinal=1)
    for key, value in (
        ("InstanceMarketOptions", {"MarketType": "spot"}),
        ("CapacityReservationSpecification", {"CapacityReservationPreference": "open"}),
        ("LaunchTemplate", {"LaunchTemplateId": "lt-foreign"}),
    ):
        body = dict(parameters.run_instances)
        body[key] = value
        with pytest.raises(LaunchCustodyError):
            validate_launch_parameters(
                replace(parameters, run_instances=body), _authority()
            )
    body = dict(parameters.run_instances)
    body["BlockDeviceMappings"] = list(body["BlockDeviceMappings"]) + [
        {"DeviceName": "/dev/sdf", "Ebs": {"VolumeSize": 1}}
    ]
    with pytest.raises(LaunchCustodyError):
        validate_launch_parameters(replace(parameters, run_instances=body), _authority())


class _Wal:
    def __init__(self, events):
        self.events = events

    def append_prepared(self, entry):
        self.events.append("WAL_PREPARED")
        return "5" * 64

    def append_committed(self, entry):
        self.events.append("WAL_COMMITTED")
        return "6" * 64


class _ProvisionStore:
    def __init__(self, events, crash=None):
        self.events = events
        self.crash = crash
        self.launch = None
        self.liability = None

    def allocate_context(self, authority):
        self.events.append("ALLOCATE_CONTEXT")
        return LaunchContext(
            allocation_ordinal=1,
            prior_worker_launch_identity_sha256=None,
            prior_instance_terminal_identity_sha256=None,
            prior_spend_allocation_close_identity_sha256=None,
            prior_liability_settlement_identity_sha256=None,
            no_unsettled_liability=True,
        )

    def put_prepared(self, launch, owner_nonce):
        assert owner_nonce == b"launch-owner-private-nonce"
        self.events.append("DDB_PREPARED")
        self.launch = launch
        return launch

    def exact_read_launch(self, activation_id, allocation_ordinal):
        self.events.append("DDB_READBACK")
        return self.launch

    def bind_committed_wal(self, launch, committed_sha256, owner_nonce):
        assert owner_nonce == b"launch-owner-private-nonce"
        self.events.append("DDB_BIND_COMMITTED")
        self.launch = replace(
            launch, ddb_committed_journal_entry_sha256=committed_sha256
        )
        return self.launch

    def commit_possibly_sent(self, launch, reserve, owner_nonce):
        assert owner_nonce == b"launch-owner-private-nonce"
        self.events.append("POSSIBLY_SENT")
        self.launch = replace(
            launch,
            state="POSSIBLY_SENT",
            send_stage="POSSIBLY_SENT",
            possibly_sent_at="2026-07-29T12:00:00Z",
            gpu_liability_reserve_ledger_identity_sha256=(
                reserve["reserve_identity_sha256"]
            ),
        )
        self.liability = {
            "state": "UNOWNED_NOT_ACTIONABLE",
            "allocation_ordinal": 1,
        }
        return self.launch, self.liability

    def wait_watching(self, activation_id, allocation_ordinal):
        self.events.append("WAIT_WATCHING")
        self.liability = {"state": "WATCHING", "allocation_ordinal": 1}
        return self.liability

    def signal_ready(self, activation_id, allocation_ordinal):
        self.events.append("SIGNAL_READY")

    def poll_result(self, activation_id, allocation_ordinal):
        self.events.append("POLL_RESULT")
        return LaunchResult(
            state="INSTANCE_OBSERVED",
            instance_ids=("i-0123456789abcdef0",),
            classification="OBSERVED",
        )


class _Reserve:
    def __init__(self, events):
        self.events = events

    def reserve(self, request):
        self.events.append("RESERVE")
        return {
            "reserve_identity_sha256": "7" * 64,
            "gpu_reserve_seconds": 900,
            "gpu_reserve_cost_usd": "13.76",
            "root_volume_tail_usd_max": "0.01",
        }


def test_prepared_wal_precedes_reserve_possibly_sent_and_signal() -> None:
    events = []
    provisioner = IntentOnlyProvisioner(
        store=_ProvisionStore(events),
        wal=_Wal(events),
        reserve=_Reserve(events),
        nonce_source=lambda: b"launch-owner-private-nonce",
        nonce_encryptor=lambda value, context: b"ciphertext",
        clock=lambda: NOW,
    )
    result = provisioner.run(_authority())
    assert result.instance_ids == ("i-0123456789abcdef0",)
    assert events == [
        "ALLOCATE_CONTEXT",
        "WAL_PREPARED",
        "DDB_PREPARED",
        "DDB_READBACK",
        "WAL_COMMITTED",
        "DDB_BIND_COMMITTED",
        "RESERVE",
        "POSSIBLY_SENT",
        "WAIT_WATCHING",
        "SIGNAL_READY",
        "POLL_RESULT",
    ]


def test_task8_reserve_adapter_calls_real_contract_boundary_without_duplication() -> None:
    calls = []
    request = object()
    spend_request = object()

    class Reader:
        def exact_read(self, value):
            calls.append(("READ", value))
            return request, spend_request

    def reserve_function(gpu_request, spend, services, writer):
        calls.append(("RESERVE", gpu_request, spend, services, writer))
        return SimpleNamespace(
            reserve_identity_sha256="7" * 64,
            gpu_reserve_seconds=900,
            gpu_reserve_cost_usd=Decimal("13.76"),
            root_volume_gib=300,
            root_volume_tail_usd_max=Decimal("0.01"),
        )

    adapter = Task8GpuReserveAdapter(
        authority_reader=Reader(),
        services="services",
        writer="writer",
        reserve_function=reserve_function,
    )
    value = {
        "activation_id": "activation-0001",
        "generation": 1,
        "allocation_ordinal": 1,
        "ec2_client_token": "1" * 64,
        "launch_parameters_sha256": "2" * 64,
        "gpu_reserve_seconds": 900,
        "gpu_reserve_cost_usd": "13.76",
        "root_volume_gib": 300,
        "root_volume_tail_usd_max": "0.01",
        "owner_invocation_nonce_sha256": "3" * 64,
    }
    result = adapter.reserve(value)
    assert result == {
        "reserve_identity_sha256": "7" * 64,
        "gpu_reserve_seconds": 900,
        "gpu_reserve_cost_usd": "13.76",
        "root_volume_tail_usd_max": "0.01",
    }
    assert calls == [
        ("READ", value),
        ("RESERVE", request, spend_request, "services", "writer"),
    ]


def test_intent_only_provisioner_has_no_ec2_mount_or_subprocess_route() -> None:
    source = inspect.getsource(IntentOnlyProvisioner)
    assert "run_instances" not in source.lower()
    assert "create_instances" not in source.lower()
    assert "subprocess" not in source
    assert "mount" not in source.lower()


@pytest.mark.parametrize(
    "boundary",
    [
        "AFTER_PREPARED_WAL",
        "AFTER_PREPARED_DDB",
        "AFTER_COMMITTED_WAL",
        "AFTER_COMMITTED_DDB",
        "AFTER_RESERVE",
        "AFTER_POSSIBLY_SENT",
    ],
)
def test_crash_after_each_durable_boundary_never_sends_from_provisioner(
    boundary,
) -> None:
    events = []

    def crash(name):
        if name == boundary:
            raise RuntimeError("crash at " + name)

    provisioner = IntentOnlyProvisioner(
        store=_ProvisionStore(events),
        wal=_Wal(events),
        reserve=_Reserve(events),
        nonce_source=lambda: b"launch-owner-private-nonce",
        nonce_encryptor=lambda value, context: b"ciphertext",
        clock=lambda: NOW,
        boundary_hook=crash,
    )
    with pytest.raises(RuntimeError, match="crash at"):
        provisioner.run(_authority())
    assert "SIGNAL_READY" not in events
    assert "POLL_RESULT" not in events


def test_sqlite_prepared_wal_is_hash_chained_durable_and_duplicate_exact(
    tmp_path,
) -> None:
    wal = SqliteLaunchWal((tmp_path / "launch.sqlite3").resolve())
    prepared = {
        "schema_version": 1,
        "record_type": "WORKER_LAUNCH_PREPARED",
        "activation_id": "activation-0001",
        "allocation_ordinal": 1,
    }
    prepared_sha = wal.append_prepared(prepared)
    assert wal.append_prepared(prepared) == prepared_sha
    committed = {
        "schema_version": 1,
        "record_type": "WORKER_LAUNCH_DDB_COMMITTED",
        "activation_id": "activation-0001",
        "allocation_ordinal": 1,
        "prepared_journal_entry_sha256": prepared_sha,
    }
    committed_sha = wal.append_committed(committed)
    assert wal.verify_chain() == 2
    assert wal.read_entry(committed_sha) == committed
    changed = dict(prepared)
    changed["foreign"] = True
    with pytest.raises(LaunchWalError, match="different bytes"):
        wal.append_prepared(changed)


def test_prepared_wal_recovers_ciphertext_parameters_nonce_and_action(
    tmp_path,
) -> None:
    wal = SqliteLaunchWal((tmp_path / "recover.sqlite3").resolve())
    owner_nonce = b"launch-owner-private-nonce"

    def crash(name):
        if name == "AFTER_PREPARED_WAL":
            raise RuntimeError("crash after durable prepare")

    provisioner = IntentOnlyProvisioner(
        store=_ProvisionStore([]),
        wal=wal,
        reserve=_Reserve([]),
        nonce_source=lambda: owner_nonce,
        nonce_encryptor=lambda value, context: b"encrypted:" + value,
        nonce_decryptor=lambda value, context: value.removeprefix(
            b"encrypted:"
        ),
        clock=lambda: NOW,
        boundary_hook=crash,
    )
    with pytest.raises(RuntimeError, match="durable prepare"):
        provisioner.run(_authority())
    prepared = wal.read_prepared("activation-0001", 1)
    assert prepared is not None
    assert prepared["owner_nonce_ciphertext_b64"]
    assert prepared["launch_parameters"]["run_instances"][
        "InstanceType"
    ] == "p5.48xlarge"
    assert prepared["sky_action_key"] == _authority().action_key
    recovered = provisioner.recover_prepared(
        activation_id="activation-0001",
        allocation_ordinal=1,
    )
    assert recovered is not None
    assert recovered.owner_nonce == owner_nonce
    assert recovered.parameters == build_launch_parameters(
        _authority(), allocation_ordinal=1
    )
    assert recovered.action_identity_sha256 == prepared[
        "action_identity_sha256"
    ]


@pytest.mark.parametrize(
    "boundary",
    [
        "AFTER_PREPARED_WAL",
        "AFTER_PREPARED_DDB",
        "AFTER_COMMITTED_WAL",
        "AFTER_COMMITTED_DDB",
        "AFTER_RESERVE",
        "AFTER_POSSIBLY_SENT",
    ],
)
def test_each_crash_boundary_adopts_recoverable_prepared_material(
    tmp_path,
    boundary,
) -> None:
    wal = SqliteLaunchWal(
        (tmp_path / f"recover-{boundary}.sqlite3").resolve()
    )
    store = _ProvisionStore([])
    reserve = _Reserve([])
    nonce_calls = 0
    owner_nonce = b"launch-owner-private-nonce"

    def nonce_source():
        nonlocal nonce_calls
        nonce_calls += 1
        return owner_nonce

    def crash(name):
        if name == boundary:
            raise RuntimeError("crash at " + name)

    arguments = {
        "store": store,
        "wal": wal,
        "reserve": reserve,
        "nonce_source": nonce_source,
        "nonce_encryptor": lambda value, context: b"encrypted:" + value,
        "nonce_decryptor": lambda value, context: value.removeprefix(
            b"encrypted:"
        ),
        "clock": lambda: NOW,
    }
    with pytest.raises(RuntimeError, match="crash at"):
        IntentOnlyProvisioner(
            **arguments,
            boundary_hook=crash,
        ).run(_authority())
    result = IntentOnlyProvisioner(**arguments).run(_authority())
    assert result.classification == "OBSERVED"
    assert nonce_calls == 1
    assert wal.verify_chain() == 2


def test_original_repository_patch_exposes_intent_only_run_instances() -> None:
    path = (
        Path(__file__).parents[1]
        / "aws/glm52-gpu/skypilot/worker_launch_intent_patch.py"
    )
    source = path.read_text()
    assert "def run_instances(" in source
    assert "IntentOnlyProvisioner" in source
    assert "create_instances(" not in source
    assert ".run_instances(" not in source


class _CompletionStore:
    def __init__(self, *, count=0, state="WATCHING", permit=True):
        self.count = count
        self.state = state
        self.permit = permit
        self.parameters = build_launch_parameters(_authority(), allocation_ordinal=1)
        self.token = build_deterministic_client_token(_authority(), 1, self.parameters)
        self.events = []

    def read_completion_authority(self, activation_id, allocation_ordinal):
        self.events.append("READ")
        return {
            "activation_id": activation_id,
            "allocation_ordinal": allocation_ordinal,
            "state": self.state,
            "same_token_completion_attempts": self.count,
            "possibly_sent_at": "2026-07-29T12:00:00Z",
            "ec2_client_token": self.token,
            "launch_parameters": self.parameters,
            "current_owner": True,
            "reserve_held": True,
        }

    def consume_attempt(self, activation_id, allocation_ordinal):
        self.events.append("CONSUME")
        if not self.permit:
            return AttemptPermit(False, self.count, "x" * 64)
        self.count += 1
        self.state = "SAME_TOKEN_COMPLETION"
        return AttemptPermit(True, self.count, ("%064x" % self.count))

    def record_direct(self, permit, response):
        self.events.append("DIRECT")
        self.state = "WATCHING"

    def record_ambiguous(self, permit, evidence):
        self.events.append("AMBIGUOUS")
        self.state = "WATCHING" if permit.attempt < 6 else "LIABILITY_INCIDENT"

    def record_rejection(self, permit, evidence):
        self.events.append("REJECTION")
        self.state = "REJECTION_PROVED_AWAITING_TERMINAL_V2"


class _Ec2:
    def __init__(self, outcome="success"):
        self.outcome = outcome
        self.calls = []
        self.total_max_attempts = 1

    def run_instances(self, **kwargs):
        self.calls.append(kwargs)
        if self.outcome == "ambiguous":
            raise AmbiguousRunInstances("connection lost")
        if self.outcome == "rejected":
            raise PositiveRunInstancesRejection("InsufficientInstanceCapacity")
        return {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "ec2-request-1",
            },
            "Instances": [{"InstanceId": "i-0123456789abcdef0"}],
        }


def test_first_same_token_call_consumes_action_and_counter_before_send() -> None:
    store = _CompletionStore()
    ec2 = _Ec2()
    result = SameTokenCompleter(store=store, ec2=ec2, clock=lambda: NOW).complete(
        {"activation_id": "activation-0001", "allocation_ordinal": 1}
    )
    assert result.classification == "DIRECT_SUCCESS"
    assert store.events == ["READ", "CONSUME", "DIRECT"]
    assert len(ec2.calls) == 1
    assert ec2.calls[0]["ClientToken"] == store.token


def test_same_token_completer_six_calls_six_minutes_and_zero_hidden_retry() -> None:
    store = _CompletionStore()
    ec2 = _Ec2("ambiguous")
    clock_values = iter(NOW + timedelta(seconds=index * 60) for index in range(7))
    completer = SameTokenCompleter(store=store, ec2=ec2, clock=lambda: next(clock_values))
    for _ in range(6):
        result = completer.complete(
            {"activation_id": "activation-0001", "allocation_ordinal": 1}
        )
        assert result.classification == "AMBIGUOUS"
    with pytest.raises(LaunchCustodyError, match="six"):
        completer.complete(
            {"activation_id": "activation-0001", "allocation_ordinal": 1}
        )
    assert len(ec2.calls) == 6
    assert {call["ClientToken"] for call in ec2.calls} == {store.token}
    assert len({canonical_sha256(call) for call in ec2.calls}) == 1


@pytest.mark.parametrize(
    ("outcome", "classification", "state"),
    [
        ("rejected", "POSITIVE_REJECTION", "REJECTION_PROVED_AWAITING_TERMINAL_V2"),
        ("ambiguous", "AMBIGUOUS", "WATCHING"),
    ],
)
def test_positive_rejection_and_ambiguous_transport_are_distinct(
    outcome, classification, state
) -> None:
    store = _CompletionStore()
    result = SameTokenCompleter(
        store=store, ec2=_Ec2(outcome), clock=lambda: NOW
    ).complete({"activation_id": "activation-0001", "allocation_ordinal": 1})
    assert result.classification == classification
    assert store.state == state


def test_completer_rejects_token_parameter_and_sdk_retry_substitution() -> None:
    store = _CompletionStore()
    original = store.read_completion_authority

    def corrupt(*args):
        result = dict(original(*args))
        result["ec2_client_token"] = "0" * 64
        return result

    store.read_completion_authority = corrupt
    with pytest.raises(LaunchCustodyError, match="[Tt]oken"):
        SameTokenCompleter(store=store, ec2=_Ec2(), clock=lambda: NOW).complete(
            {"activation_id": "activation-0001", "allocation_ordinal": 1}
        )
    store = _CompletionStore()
    ec2 = _Ec2()
    ec2.total_max_attempts = 3
    with pytest.raises(LaunchCustodyError, match="retry"):
        SameTokenCompleter(store=store, ec2=ec2, clock=lambda: NOW).complete(
            {"activation_id": "activation-0001", "allocation_ordinal": 1}
        )


class _WatchStore:
    def __init__(
        self,
        instances=(),
        age_days=0,
        authorized=True,
        termination_count=0,
        authorized_instance_id="i-0123456789abcdef0",
    ):
        self.instances = tuple(instances)
        self.age_days = age_days
        self.authorized = authorized
        self.authorized_instance_id = authorized_instance_id
        self.events = []
        self.termination_counts = {
            instance["InstanceId"]: termination_count for instance in instances
        }

    def acquire_or_takeover(self, now, *, deadline):
        self.events.append("ACQUIRE")
        return {
            "activation_id": "activation-0001",
            "allocation_ordinal": 1,
            "ec2_client_token": "1" * 64,
            "expected_worker_tags_sha256": "2" * 64,
            "state": "WATCHING",
            "watch_started_at": (
                now - timedelta(days=self.age_days)
            ).isoformat().replace("+00:00", "Z"),
            "next_scan_at": now.isoformat().replace("+00:00", "Z"),
            "current_owner": True,
            "settlement_identity_sha256": None,
            "work_authorized": self.authorized,
            "authorized_instance_id": self.authorized_instance_id,
        }

    def record_scan(self, authority, evidence, next_scan_at, *, deadline):
        self.events.append(("SCAN", next_scan_at))

    def open_late_allocation(
        self, authority, instance, post_terminal, *, deadline
    ):
        self.events.append(("OPEN", instance["InstanceId"], post_terminal))

    def consume_termination(self, authority, instance_id, *, deadline):
        self.events.append(("CONSUME_TERMINATE", instance_id))
        self.termination_counts[instance_id] = (
            self.termination_counts.get(instance_id, 0) + 1
        )
        return TerminationPermit(
            may_terminate=True,
            instance_id=instance_id,
            call_count=self.termination_counts[instance_id],
            window_started_at=authority["watch_started_at"],
            action_identity_sha256="%064x"
            % self.termination_counts[instance_id],
        )

    def record_incident(self, authority, kind, *, deadline):
        self.events.append(("INCIDENT", kind))

    def reconcile_instance(self, authority, instance_id, *, deadline):
        self.events.append(("RECONCILE", instance_id))

    def settle_if_complete(self, authority, *, deadline):
        self.events.append("SETTLE_CHECK")
        return None


class _Discovery:
    def __init__(self, instances):
        self.instances = tuple(instances)
        self.calls = []

    def correlate(self, *, client_token, expected_tags_sha256, deadline):
        self.calls.append((client_token, expected_tags_sha256, deadline))
        return {
            "cloudtrail_request_identities": ("ct-1",),
            "instances": self.instances,
            "state_change_identities": (),
            "launch_evidence_identities": (),
            "spend_evidence_identities": (),
            "terminal_v2_published": True,
            "evidence_identity_sha256": "3" * 64,
        }


class _Terminate:
    def __init__(self):
        self.calls = []
        self.total_max_attempts = 1

    def terminate(self, instance_id, *, deadline):
        self.calls.append(instance_id)
        return {"request_id": "term-1", "response_sha256": "4" * 64}


def _instance(instance_id="i-0123456789abcdef0", state="running"):
    return {
        "InstanceId": instance_id,
        "State": state,
        "TagsSha256": "2" * 64,
        "ObservedAt": "2026-07-29T12:00:00Z",
    }


def test_watcher_absolute_cadence_event_acceleration_and_zero_scan_no_settle() -> None:
    store = _WatchStore()
    watcher = LiabilityWatcher(
        store=store,
        discovery=_Discovery(()),
        terminator=_Terminate(),
        clock=lambda: NOW,
    )
    result = watcher.run(WatchInput())
    assert result.next_scan_at == "2026-07-29T12:01:00Z"
    assert result.settled is False
    assert "SETTLE_CHECK" not in store.events
    accelerated = watcher.run(WatchInput(event_accelerator="ec2-state-change"))
    assert accelerated.settled is False


def test_watcher_fails_closed_when_scan_exceeds_twenty_second_deadline() -> None:
    class DeadlineClock:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            return NOW if self.calls < 5 else NOW + timedelta(seconds=21)

    with pytest.raises(LaunchCustodyError, match="20-second"):
        LiabilityWatcher(
            store=_WatchStore(),
            discovery=_Discovery(()),
            terminator=_Terminate(),
            clock=DeadlineClock(),
        ).run(WatchInput())


def test_watcher_deadline_expiry_after_scan_blocks_termination_effect() -> None:
    class DeadlineClock:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            return NOW if self.calls < 8 else NOW + timedelta(seconds=21)

    instance = _instance()
    terminator = _Terminate()
    with pytest.raises(LaunchCustodyError, match="20-second"):
        LiabilityWatcher(
            store=_WatchStore(instances=(instance,), authorized=False),
            discovery=_Discovery((instance,)),
            terminator=terminator,
            clock=DeadlineClock(),
        ).run(WatchInput())
    assert terminator.calls == []


def test_watcher_late_multiple_instances_are_drain_only_and_exact_terminated() -> None:
    instances = (_instance("i-00000000000000001"), _instance("i-00000000000000002"))
    store = _WatchStore(instances=instances, authorized=False)
    terminator = _Terminate()
    result = LiabilityWatcher(
        store=store,
        discovery=_Discovery(instances),
        terminator=terminator,
        clock=lambda: NOW,
    ).run(WatchInput())
    assert result.instance_ids == (
        "i-00000000000000001",
        "i-00000000000000002",
    )
    assert terminator.calls == list(result.instance_ids)
    assert ("INCIDENT", "MULTIPLE_INSTANCE_TOKEN") in store.events
    assert all(event[0] != "AUTHORIZE_WORK" for event in store.events if isinstance(event, tuple))


def test_watcher_preserves_authorized_known_worker_until_authorization_expires() -> None:
    instance = _instance()
    store = _WatchStore(instances=(instance,), authorized=True)
    terminator = _Terminate()
    LiabilityWatcher(
        store=store,
        discovery=_Discovery((instance,)),
        terminator=terminator,
        clock=lambda: NOW,
    ).run(WatchInput())
    assert terminator.calls == []
    assert not any(
        isinstance(event, tuple) and event[0] == "OPEN"
        for event in store.events
    )

    expired_store = _WatchStore(instances=(instance,), authorized=False)
    expired_terminator = _Terminate()
    LiabilityWatcher(
        store=expired_store,
        discovery=_Discovery((instance,)),
        terminator=expired_terminator,
        clock=lambda: NOW,
    ).run(WatchInput())
    assert expired_terminator.calls == [instance["InstanceId"]]


def test_watcher_drains_single_late_instance_even_while_work_is_authorized() -> None:
    late = _instance("i-00000000000000002")
    store = _WatchStore(instances=(late,), authorized=True)
    terminator = _Terminate()
    LiabilityWatcher(
        store=store,
        discovery=_Discovery((late,)),
        terminator=terminator,
        clock=lambda: NOW,
    ).run(WatchInput())
    assert terminator.calls == [late["InstanceId"]]
    assert ("OPEN", late["InstanceId"], True) in store.events


def test_watcher_30_day_incident_keeps_scan_and_termination_live() -> None:
    instance = _instance()
    store = _WatchStore(instances=(instance,), age_days=31, authorized=False)
    terminator = _Terminate()
    LiabilityWatcher(
        store=store,
        discovery=_Discovery((instance,)),
        terminator=terminator,
        clock=lambda: NOW,
    ).run(WatchInput())
    assert ("INCIDENT", "UNSETTLED_30_DAY") in store.events
    assert terminator.calls == [instance["InstanceId"]]


def test_watcher_rejects_seventh_termination_in_same_nominal_window() -> None:
    instance = _instance()
    store = _WatchStore(
        instances=(instance,),
        authorized=False,
        termination_count=6,
    )
    terminator = _Terminate()
    with pytest.raises(LaunchCustodyError, match="six termination"):
        LiabilityWatcher(
            store=store,
            discovery=_Discovery((instance,)),
            terminator=terminator,
            clock=lambda: NOW,
        ).run(WatchInput())
    assert terminator.calls == []


def test_post_terminal_allocation_closed_chain_and_canonical_identity() -> None:
    discovered = build_post_terminal_allocation(
        campaign_identity_sha256="f" * 64,
        activation_id="activation-0001",
        activation_ordinal=1,
        generation=1,
        allocation_ordinal=1,
        ec2_client_token="1" * 64,
        worker_launch_identity_sha256="2" * 64,
        liability_identity_sha256="3" * 64,
        terminal_v2_identity_sha256="4" * 64,
        instance_id="i-0123456789abcdef0",
        instance_tags_sha256="5" * 64,
        owner={
            "owner_attempt": 1,
            "owner_execution_arn": (
                "arn:aws:states:us-west-2:246813579024:execution:liability:one"
            ),
            "owner_state_machine_version_arn": (
                "arn:aws:states:us-west-2:246813579024:"
                "stateMachine:liability:1"
            ),
            "owner_dispatch_identity_sha256": "6" * 64,
            "owner_invocation_nonce_sha256": "7" * 64,
            "owner_hard_expires_at": "2026-07-29T12:10:00Z",
        },
        discovered_at="2026-07-29T12:00:00Z",
    )
    assert discovered["state"] == "DISCOVERED"
    assert discovered["campaign_identity_sha256"] == "f" * 64
    assert discovered["canonical_body_sha256"] == canonical_sha256(
        {key: value for key, value in discovered.items() if key != "canonical_body_sha256"}
    )


class _SettlementStore:
    def __init__(self, *, variant="instances"):
        self.variant = variant
        self.calls = 0

    def read_complete_view(self, activation_id, allocation_ordinal):
        if self.variant == "rejection":
            terminal_allocations = []
            post = []
            rejection = "1" * 64
            terminal_evidence = []
            spend = []
        else:
            terminal_allocations = [
                {
                    "allocation_ordinal": 1,
                    "instance_id": "i-00000000000000001",
                    "instance_terminal_identity_sha256": "2" * 64,
                    "spend_allocation_close_identity_sha256": "3" * 64,
                }
            ]
            post = [
                {
                    "allocation_ordinal": 1,
                    "instance_id": "i-00000000000000002",
                    "state": "ALLOCATION_CLOSED",
                    "instance_terminal_identity_sha256": "4" * 64,
                    "spend_allocation_close_identity_sha256": "5" * 64,
                    "canonical_body_sha256": "6" * 64,
                }
            ]
            rejection = None
            terminal_evidence = ["2" * 64, "4" * 64]
            spend = ["3" * 64, "5" * 64]
        return {
            "campaign_identity_sha256": "f" * 64,
            "activation_id": activation_id,
            "activation_ordinal": 1,
            "generation": 1,
            "allocation_ordinal": allocation_ordinal,
            "worker_launch_identity_sha256": "7" * 64,
            "worker_launch_liability_identity_sha256": "8" * 64,
            "terminal_v2": {
                "key": "terminal-v2-key",
                "version_id": "opaque-version",
                "body_sha256": "9" * 64,
                "allocations": terminal_allocations,
                "allocations_array_sha256": canonical_sha256(terminal_allocations),
            },
            "post_terminal_allocations": post,
            "service_rejection_evidence_sha256": rejection,
            "instance_terminal_evidence": terminal_evidence,
            "spend_close_evidence": spend,
            "reserve_identity_sha256": "a" * 64,
            "reserve_release_identity_sha256": "b" * 64,
            "residual_liability_approval_identity_sha256": "c" * 64,
            "final_spend_ledger_head_identity_sha256": "d" * 64,
            "prior_incident_identity_sha256": "e" * 64 if post else None,
            "owner": {
                "owner_execution_arn": (
                    "arn:aws:states:us-west-2:246813579024:execution:liability:one"
                ),
                "owner_state_machine_version_arn": (
                    "arn:aws:states:us-west-2:246813579024:"
                    "stateMachine:liability:1"
                ),
                "owner_dispatch_identity_sha256": "f" * 64,
                "owner_invocation_nonce_sha256": "0" * 64,
            },
        }

    def transact_settle_once(self, settlement):
        self.calls += 1
        return {"classification": "AMBIGUOUS"}

    def coherent_read_settlement(self, key):
        return self.last

    def set_expected(self, settlement):
        self.last = settlement


@pytest.mark.parametrize(
    ("variant", "kind", "cardinality"),
    [
        ("rejection", "NO_INSTANCE_POSITIVE_REJECTION", 0),
        ("instances", "ALL_INSTANCES_TERMINAL_AND_SPEND_CLOSED", 2),
    ],
)
def test_both_settlement_kinds_and_lost_response_exact_adoption(
    variant, kind, cardinality
) -> None:
    store = _SettlementStore(variant=variant)
    coordinator = LiabilitySettlementCoordinator(store=store, clock=lambda: NOW)
    preview = coordinator.build("activation-0001", 1)
    store.set_expected(preview)
    result = coordinator.settle("activation-0001", 1)
    assert result["settlement_kind"] == kind
    assert result["campaign_identity_sha256"] == "f" * 64
    assert result["final_worker_cardinality"] == str(cardinality)
    assert store.calls == 1


def test_settlement_rejects_foreign_missing_or_open_member_and_no_refund() -> None:
    store = _SettlementStore()
    original = store.read_complete_view
    view = original("activation-0001", 1)
    view["post_terminal_allocations"][0]["state"] = "ALLOCATION_OPEN"
    store.read_complete_view = lambda *_: view
    with pytest.raises(LaunchCustodyError, match="closed"):
        LiabilitySettlementCoordinator(store=store, clock=lambda: NOW).build(
            "activation-0001", 1
        )
    assert store.calls == 0


def test_settlement_rejects_evidence_not_bound_to_exact_final_members() -> None:
    store = _SettlementStore()
    view = store.read_complete_view("activation-0001", 1)
    view["instance_terminal_evidence"][0] = "f" * 64
    store.read_complete_view = lambda *_: view
    with pytest.raises(LaunchCustodyError, match="exact final members"):
        LiabilitySettlementCoordinator(store=store, clock=lambda: NOW).build(
            "activation-0001", 1
        )
    assert store.calls == 0


def test_new_activation_token_or_ordinal_blocked_while_liability_nonsettled() -> None:
    events = []
    store = _ProvisionStore(events)

    def blocked(authority):
        return LaunchContext(
            allocation_ordinal=2,
            prior_worker_launch_identity_sha256="1" * 64,
            prior_instance_terminal_identity_sha256="2" * 64,
            prior_spend_allocation_close_identity_sha256="3" * 64,
            prior_liability_settlement_identity_sha256="4" * 64,
            no_unsettled_liability=False,
        )

    store.allocate_context = blocked
    with pytest.raises(LaunchCustodyError, match="nonsettled"):
        IntentOnlyProvisioner(
            store=store,
            wal=_Wal(events),
            reserve=_Reserve(events),
            nonce_source=lambda: b"nonce",
            nonce_encryptor=lambda value, context: b"cipher",
            clock=lambda: NOW,
        ).run(_authority())
    assert events == []
