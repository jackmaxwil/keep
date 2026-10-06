"""Intent-only worker launch, same-token completion, and liability custody."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import re
from typing import Callable, Mapping, Optional, Tuple

from .canonical import canonical_sha256
from .records import (
    canonical_record_identity_unchecked,
    validate_record,
)


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
INSTANCE_TYPE = "p5.48xlarge"
MARKET_TYPE = "on-demand"
GPU_RESERVE_SECONDS = 900
GPU_RESERVE_COST_USD = "13.76"
ROOT_VOLUME_GIB = 300
ROOT_VOLUME_TAIL_USD_MAX = "0.01"
MAX_COMPLETION_CALLS = 6
COMPLETION_WINDOW_SECONDS = 360
WATCH_INTERVAL_SECONDS = 60
WATCH_ACTION_DEADLINE_SECONDS = 20
TERMINATION_SETTLING_SECONDS = 360
CONTROL_PERIOD_SECONDS = 2_592_000
_SHA = re.compile(r"^[0-9a-f]{64}$")
_AMI = re.compile(r"^ami-[0-9a-f]{17}$")
_SUBNET = re.compile(r"^subnet-[0-9a-f]{17}$")
_SG = re.compile(r"^sg-[0-9a-f]{17}$")
_INSTANCE = re.compile(r"^i-[0-9a-f]{17}$")


class LaunchCustodyError(ValueError):
    """The deterministic launch or retained liability contract failed."""


class AmbiguousRunInstances(RuntimeError):
    """The EC2 transport may have sent the exact request."""


class PositiveRunInstancesRejection(RuntimeError):
    """EC2 positively rejected the request before instance acceptance."""


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise LaunchCustodyError(label + " must be a lowercase SHA-256")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or not value.strip():
        raise LaunchCustodyError(label + " must be a nonempty exact string")
    return value


def _utc(value: object, label: str) -> datetime:
    if type(value) is not str:
        raise LaunchCustodyError(label + " must be canonical UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise LaunchCustodyError(label + " must be canonical UTC") from exc
    return parsed.replace(tzinfo=timezone.utc)


def _utc_text(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise LaunchCustodyError("clock must be timezone-aware")
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class LaunchParameterAuthority:
    account_id: str
    region: str
    run_id: str
    campaign_identity_sha256: str
    activation_id: str
    activation_ordinal: int
    generation: int
    action_key: str
    sky_request_id: str
    sky_job_name: str
    sky_task_name: str
    task_yaml_sha256: str
    request_body_sha256: str
    approved_ami_id: str
    subnet_id: str
    security_group_id: str
    instance_profile_name: str
    source_identity_sha256: str


@dataclass(frozen=True)
class LaunchParameters:
    schema_version: int
    record_type: str
    authority: LaunchParameterAuthority
    allocation_ordinal: int
    run_instances: Mapping[str, object]
    canonical_identity_sha256: str


def _validate_authority(value: object) -> LaunchParameterAuthority:
    if not isinstance(value, LaunchParameterAuthority):
        raise LaunchCustodyError("launch authority must be typed")
    if (
        value.account_id != ACCOUNT_ID
        or value.region != REGION
        or value.run_id != RUN_ID
        or type(value.activation_id) is not str
        or not value.activation_id
        or type(value.activation_ordinal) is not int
        or value.activation_ordinal <= 0
        or type(value.generation) is not int
        or value.generation <= 0
        or type(value.action_key) is not str
        or "SKY_POST" not in value.action_key
        or _AMI.fullmatch(value.approved_ami_id) is None
        or _SUBNET.fullmatch(value.subnet_id) is None
        or _SG.fullmatch(value.security_group_id) is None
        or value.instance_profile_name != "keep-glm52-gpu-worker"
    ):
        raise LaunchCustodyError("launch authority coordinates are invalid")
    for field in (
        "campaign_identity_sha256",
        "task_yaml_sha256",
        "request_body_sha256",
        "source_identity_sha256",
    ):
        _sha(getattr(value, field), field)
    for field in (
        "sky_request_id",
        "sky_job_name",
        "sky_task_name",
    ):
        _text(getattr(value, field), field)
    return value


def _tag_map(
    authority: LaunchParameterAuthority, allocation_ordinal: int
) -> Mapping[str, str]:
    return {
        "Project": "KEEP",
        "Campaign": "GLM-5.2",
        "RunId": RUN_ID,
        "Market": MARKET_TYPE,
        "campaign-identity-sha256": authority.campaign_identity_sha256,
        "activation-id": authority.activation_id,
        "activation-ordinal-text": "%08d" % authority.activation_ordinal,
        "generation-text": "%08d" % authority.generation,
        "allocation-ordinal-text": "%08d" % allocation_ordinal,
        "action-key": authority.action_key,
        "sky-request-id": authority.sky_request_id,
        "sky-job-name": authority.sky_job_name,
        "sky-task-name": authority.sky_task_name,
        "task-yaml-sha256": authority.task_yaml_sha256,
        "request-body-sha256": authority.request_body_sha256,
    }


def _expected_run_instances(
    authority: LaunchParameterAuthority, allocation_ordinal: int
) -> Mapping[str, object]:
    tags = [
        {"Key": key, "Value": value}
        for key, value in sorted(_tag_map(authority, allocation_ordinal).items())
    ]
    return {
        "ImageId": authority.approved_ami_id,
        "InstanceType": INSTANCE_TYPE,
        "MinCount": 1,
        "MaxCount": 1,
        "SubnetId": authority.subnet_id,
        "SecurityGroupIds": [authority.security_group_id],
        "IamInstanceProfile": {"Name": authority.instance_profile_name},
        "MetadataOptions": {
            "HttpEndpoint": "enabled",
            "HttpTokens": "required",
            "HttpPutResponseHopLimit": 1,
        },
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "DeleteOnTermination": True,
                    "Encrypted": True,
                    "Iops": 3000,
                    "Throughput": 125,
                    "VolumeSize": ROOT_VOLUME_GIB,
                    "VolumeType": "gp3",
                },
            }
        ],
        "TagSpecifications": [
            {"ResourceType": "instance", "Tags": tags},
            {"ResourceType": "volume", "Tags": tags},
        ],
    }


def _parameter_body(value: LaunchParameters) -> Mapping[str, object]:
    return {
        "schema_version": value.schema_version,
        "record_type": value.record_type,
        "authority": asdict(value.authority),
        "allocation_ordinal": value.allocation_ordinal,
        "run_instances": value.run_instances,
    }


def build_launch_parameters(
    authority: LaunchParameterAuthority, *, allocation_ordinal: int
) -> LaunchParameters:
    authority = _validate_authority(authority)
    if type(allocation_ordinal) is not int or allocation_ordinal <= 0:
        raise LaunchCustodyError("allocation ordinal must be positive")
    provisional = LaunchParameters(
        schema_version=1,
        record_type="glm52_worker_launch_parameters_v1",
        authority=authority,
        allocation_ordinal=allocation_ordinal,
        run_instances=_expected_run_instances(authority, allocation_ordinal),
        canonical_identity_sha256="",
    )
    result = LaunchParameters(
        **{
            **asdict(provisional),
            "authority": authority,
            "run_instances": provisional.run_instances,
            "canonical_identity_sha256": canonical_sha256(
                _parameter_body(provisional)
            ),
        }
    )
    return validate_launch_parameters(result, authority)


def validate_launch_parameters(
    value: object, authority: LaunchParameterAuthority
) -> LaunchParameters:
    authority = _validate_authority(authority)
    if not isinstance(value, LaunchParameters):
        raise LaunchCustodyError("launch parameters must be typed")
    if (
        value.schema_version != 1
        or value.record_type != "glm52_worker_launch_parameters_v1"
        or value.authority != authority
        or type(value.allocation_ordinal) is not int
        or value.allocation_ordinal <= 0
        or type(value.run_instances) is not dict
        or value.run_instances
        != _expected_run_instances(authority, value.allocation_ordinal)
        or value.canonical_identity_sha256
        != canonical_sha256(_parameter_body(value))
    ):
        raise LaunchCustodyError("launch parameters drifted")
    forbidden = {
        "InstanceMarketOptions",
        "CapacityReservationSpecification",
        "LaunchTemplate",
    }
    if forbidden & set(value.run_instances):
        raise LaunchCustodyError(
            "Spot, Capacity Block, and launch templates are forbidden"
        )
    return value


def build_deterministic_client_token(
    authority: LaunchParameterAuthority,
    allocation_ordinal: int,
    parameters: LaunchParameters,
) -> str:
    authority = _validate_authority(authority)
    validate_launch_parameters(parameters, authority)
    if allocation_ordinal != parameters.allocation_ordinal:
        raise LaunchCustodyError("token allocation ordinal mismatched")
    immutable = {
        "domain": "glm52-ec2-client-token-v1",
        "campaign_identity_sha256": authority.campaign_identity_sha256,
        "activation_id": authority.activation_id,
        "activation_ordinal": authority.activation_ordinal,
        "generation": authority.generation,
        "action_key": authority.action_key,
        "sky_request_id": authority.sky_request_id,
        "sky_job_name": authority.sky_job_name,
        "allocation_ordinal": allocation_ordinal,
        "launch_parameters_sha256": parameters.canonical_identity_sha256,
    }
    return canonical_sha256(immutable)


@dataclass(frozen=True)
class LaunchContext:
    allocation_ordinal: int
    prior_worker_launch_identity_sha256: Optional[str]
    prior_instance_terminal_identity_sha256: Optional[str]
    prior_spend_allocation_close_identity_sha256: Optional[str]
    prior_liability_settlement_identity_sha256: Optional[str]
    no_unsettled_liability: bool


@dataclass(frozen=True)
class WorkerLaunchIntent:
    schema_version: int
    record_type: str
    account_id: str
    region: str
    run_id: str
    campaign_identity_sha256: str
    activation_id: str
    activation_ordinal: int
    generation: int
    action_key: str
    allocation_ordinal: int
    ec2_client_token: str
    launch_parameters: LaunchParameters
    launch_parameters_sha256: str
    expected_worker_tags_sha256: str
    prior_worker_launch_identity_sha256: Optional[str]
    prior_instance_terminal_identity_sha256: Optional[str]
    prior_spend_allocation_close_identity_sha256: Optional[str]
    prior_liability_settlement_identity_sha256: Optional[str]
    owner_invocation_nonce_sha256: str
    owner_nonce_ciphertext_sha256: str
    prepared_at: str
    prepared_journal_entry_sha256: str
    ddb_committed_journal_entry_sha256: Optional[str]
    state: str
    send_stage: str
    possibly_sent_at: Optional[str]
    gpu_liability_reserve_ledger_identity_sha256: Optional[str]
    canonical_identity_sha256: str


def _intent_body(value: WorkerLaunchIntent) -> Mapping[str, object]:
    body = asdict(value)
    body["launch_parameters"] = _parameter_body(value.launch_parameters)
    body.pop("canonical_identity_sha256")
    return body


def _rehash_intent(value: WorkerLaunchIntent) -> WorkerLaunchIntent:
    return WorkerLaunchIntent(
        **{
            **asdict(value),
            "launch_parameters": value.launch_parameters,
            "canonical_identity_sha256": canonical_sha256(_intent_body(value)),
        }
    )


@dataclass(frozen=True)
class LaunchResult:
    state: str
    instance_ids: Tuple[str, ...]
    classification: str


@dataclass(frozen=True)
class RecoveredPreparedLaunch:
    parameters: LaunchParameters
    ec2_client_token: str
    owner_nonce: bytes
    owner_nonce_ciphertext: bytes
    encryption_context: Mapping[str, str]
    prepared_at: str
    action_identity_sha256: str


class Task8GpuReserveAdapter:
    """Use Task 8's authenticated reserve function without reimplementing it."""

    def __init__(
        self,
        *,
        authority_reader: object,
        services: object,
        writer: object,
        reserve_function: Optional[Callable[..., object]] = None,
    ) -> None:
        if reserve_function is None:
            from .spend_authority import reserve_gpu_liability

            reserve_function = reserve_gpu_liability
        if not callable(reserve_function):
            raise LaunchCustodyError("Task 8 reserve function is absent")
        self._reader = authority_reader
        self._services = services
        self._writer = writer
        self._reserve_function = reserve_function

    def reserve(self, value: Mapping[str, object]) -> Mapping[str, object]:
        required = {
            "activation_id",
            "generation",
            "allocation_ordinal",
            "ec2_client_token",
            "launch_parameters_sha256",
            "gpu_reserve_seconds",
            "gpu_reserve_cost_usd",
            "root_volume_gib",
            "root_volume_tail_usd_max",
            "owner_invocation_nonce_sha256",
        }
        if type(value) is not dict or set(value) != required:
            raise LaunchCustodyError("Task 8 reserve request is not closed")
        read = getattr(self._reader, "exact_read", None)
        if not callable(read):
            raise LaunchCustodyError("Task 8 reserve authority reader is absent")
        requests = read(value)
        if type(requests) is not tuple or len(requests) != 2:
            raise LaunchCustodyError("Task 8 reserve authority is incomplete")
        result = self._reserve_function(
            requests[0],
            requests[1],
            self._services,
            self._writer,
        )
        reserve_identity = getattr(result, "reserve_identity_sha256", None)
        seconds = getattr(result, "gpu_reserve_seconds", None)
        cost = getattr(result, "gpu_reserve_cost_usd", None)
        root_gib = getattr(result, "root_volume_gib", None)
        root_tail = getattr(result, "root_volume_tail_usd_max", None)
        if (
            _sha(reserve_identity, "Task 8 reserve identity") != reserve_identity
            or seconds != GPU_RESERVE_SECONDS
            or str(cost) != GPU_RESERVE_COST_USD
            or root_gib != ROOT_VOLUME_GIB
            or str(root_tail) != ROOT_VOLUME_TAIL_USD_MAX
        ):
            raise LaunchCustodyError("Task 8 reserve result drifted")
        return {
            "reserve_identity_sha256": reserve_identity,
            "gpu_reserve_seconds": seconds,
            "gpu_reserve_cost_usd": str(cost),
            "root_volume_tail_usd_max": str(root_tail),
        }


class IntentOnlyProvisioner:
    """Durable intent path; it owns no EC2 client or external process."""

    def __init__(
        self,
        *,
        store: object,
        wal: object,
        reserve: object,
        nonce_source: Callable[[], bytes],
        nonce_encryptor: Callable[[bytes, Mapping[str, str]], bytes],
        nonce_decryptor: Optional[
            Callable[[bytes, Mapping[str, str]], bytes]
        ] = None,
        clock: Callable[[], datetime],
        boundary_hook: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._store = store
        self._wal = wal
        self._reserve = reserve
        self._nonce_source = nonce_source
        self._nonce_encryptor = nonce_encryptor
        self._nonce_decryptor = nonce_decryptor
        self._clock = clock
        self._hook = boundary_hook

    def _boundary(self, name: str) -> None:
        if self._hook is not None:
            self._hook(name)

    def recover_prepared(
        self,
        *,
        activation_id: str,
        allocation_ordinal: int,
    ) -> Optional[RecoveredPreparedLaunch]:
        verify = getattr(self._wal, "verify_chain", None)
        if callable(verify):
            verify()
        read = getattr(self._wal, "read_prepared", None)
        if not callable(read):
            return None
        entry = read(activation_id, allocation_ordinal)
        if entry is None:
            return None
        required = {
            "schema_version",
            "record_type",
            "activation_id",
            "allocation_ordinal",
            "ec2_client_token",
            "launch_parameters",
            "launch_parameters_sha256",
            "sky_action_key",
            "action_identity_sha256",
            "owner_invocation_nonce_sha256",
            "owner_nonce_ciphertext_b64",
            "owner_nonce_ciphertext_sha256",
            "encryption_context",
            "prepared_at",
        }
        if (
            type(entry) is not dict
            or set(entry) != required
            or entry["schema_version"] != 1
            or entry["record_type"] != "WORKER_LAUNCH_PREPARED"
            or entry["activation_id"] != activation_id
            or entry["allocation_ordinal"] != allocation_ordinal
            or self._nonce_decryptor is None
        ):
            raise LaunchCustodyError("prepared WAL recovery is unavailable")
        raw_parameters = entry["launch_parameters"]
        if type(raw_parameters) is not dict:
            raise LaunchCustodyError("recovered launch parameters are malformed")
        raw_authority = raw_parameters.get("authority")
        if type(raw_authority) is not dict:
            raise LaunchCustodyError("recovered launch authority is malformed")
        try:
            parameters = LaunchParameters(
                **{
                    **raw_parameters,
                    "authority": LaunchParameterAuthority(**raw_authority),
                }
            )
        except TypeError as exc:
            raise LaunchCustodyError(
                "recovered launch parameters are malformed"
            ) from exc
        validate_launch_parameters(parameters, parameters.authority)
        if (
            parameters.canonical_identity_sha256
            != entry["launch_parameters_sha256"]
            or parameters.allocation_ordinal != allocation_ordinal
            or parameters.authority.activation_id != activation_id
            or parameters.authority.action_key != entry["sky_action_key"]
            or build_deterministic_client_token(
                parameters.authority,
                allocation_ordinal,
                parameters,
            )
            != entry["ec2_client_token"]
        ):
            raise LaunchCustodyError("recovered launch identity drifted")
        expected_action_identity = canonical_sha256(
            {
                "action_key": parameters.authority.action_key,
                "request_body_sha256": (
                    parameters.authority.request_body_sha256
                ),
                "source_identity_sha256": (
                    parameters.authority.source_identity_sha256
                ),
                "task_yaml_sha256": parameters.authority.task_yaml_sha256,
            }
        )
        if entry["action_identity_sha256"] != expected_action_identity:
            raise LaunchCustodyError("recovered action identity drifted")
        try:
            ciphertext = base64.b64decode(
                entry["owner_nonce_ciphertext_b64"],
                validate=True,
            )
        except (TypeError, ValueError) as exc:
            raise LaunchCustodyError(
                "recovered nonce ciphertext is malformed"
            ) from exc
        context = entry["encryption_context"]
        if (
            type(context) is not dict
            or set(context)
            != {"run_id", "activation_id", "allocation_ordinal_text"}
            or hashlib.sha256(ciphertext).hexdigest()
            != entry["owner_nonce_ciphertext_sha256"]
        ):
            raise LaunchCustodyError("recovered nonce ciphertext drifted")
        owner_nonce = self._nonce_decryptor(ciphertext, context)
        if (
            type(owner_nonce) is not bytes
            or hashlib.sha256(owner_nonce).hexdigest()
            != entry["owner_invocation_nonce_sha256"]
        ):
            raise LaunchCustodyError("recovered owner nonce did not authenticate")
        _utc(entry["prepared_at"], "recovered prepared_at")
        return RecoveredPreparedLaunch(
            parameters=parameters,
            ec2_client_token=str(entry["ec2_client_token"]),
            owner_nonce=owner_nonce,
            owner_nonce_ciphertext=ciphertext,
            encryption_context=context,
            prepared_at=str(entry["prepared_at"]),
            action_identity_sha256=expected_action_identity,
        )

    def run(self, authority: LaunchParameterAuthority) -> LaunchResult:
        authority = _validate_authority(authority)
        allocate = getattr(self._store, "allocate_context", None)
        if not callable(allocate):
            raise LaunchCustodyError("launch context allocator is absent")
        context = allocate(authority)
        if not isinstance(context, LaunchContext):
            raise LaunchCustodyError("launch context is not typed")
        if context.no_unsettled_liability is not True:
            raise LaunchCustodyError(
                "a nonsettled liability blocks a token or ordinal"
            )
        if type(context.allocation_ordinal) is not int or (
            context.allocation_ordinal <= 0
        ):
            raise LaunchCustodyError("allocated ordinal is invalid")
        prior = (
            context.prior_worker_launch_identity_sha256,
            context.prior_instance_terminal_identity_sha256,
            context.prior_spend_allocation_close_identity_sha256,
            context.prior_liability_settlement_identity_sha256,
        )
        if context.allocation_ordinal == 1:
            if prior != (None, None, None, None):
                raise LaunchCustodyError("first allocation has prior evidence")
        else:
            if any(item is None for item in prior):
                raise LaunchCustodyError(
                    "replacement lacks terminal, spend-close, or settlement proof"
                )
            for index, item in enumerate(prior):
                _sha(item, "prior evidence %d" % index)
        recovered = self.recover_prepared(
            activation_id=authority.activation_id,
            allocation_ordinal=context.allocation_ordinal,
        )
        if recovered is None:
            parameters = build_launch_parameters(
                authority, allocation_ordinal=context.allocation_ordinal
            )
            token = build_deterministic_client_token(
                authority, context.allocation_ordinal, parameters
            )
            owner_nonce = self._nonce_source()
            if type(owner_nonce) is not bytes or len(owner_nonce) < 16:
                raise LaunchCustodyError("launch owner nonce is invalid")
            encryption_context = {
                "run_id": RUN_ID,
                "activation_id": authority.activation_id,
                "allocation_ordinal_text": "%08d" % context.allocation_ordinal,
            }
            ciphertext = self._nonce_encryptor(
                owner_nonce, encryption_context
            )
            if type(ciphertext) is not bytes or not ciphertext:
                raise LaunchCustodyError("encrypted owner nonce is invalid")
            prepared_at = _utc_text(self._clock())
            action_identity_sha256 = canonical_sha256(
                {
                    "action_key": authority.action_key,
                    "request_body_sha256": authority.request_body_sha256,
                    "source_identity_sha256": authority.source_identity_sha256,
                    "task_yaml_sha256": authority.task_yaml_sha256,
                }
            )
        else:
            parameters = recovered.parameters
            if parameters.authority != authority:
                raise LaunchCustodyError("recovered launch authority is foreign")
            token = recovered.ec2_client_token
            owner_nonce = recovered.owner_nonce
            ciphertext = recovered.owner_nonce_ciphertext
            encryption_context = recovered.encryption_context
            prepared_at = recovered.prepared_at
            action_identity_sha256 = recovered.action_identity_sha256
        owner_nonce_sha = hashlib.sha256(owner_nonce).hexdigest()
        prepared_entry = {
            "schema_version": 1,
            "record_type": "WORKER_LAUNCH_PREPARED",
            "activation_id": authority.activation_id,
            "allocation_ordinal": context.allocation_ordinal,
            "ec2_client_token": token,
            "launch_parameters": asdict(parameters),
            "launch_parameters_sha256": parameters.canonical_identity_sha256,
            "sky_action_key": authority.action_key,
            "action_identity_sha256": action_identity_sha256,
            "owner_invocation_nonce_sha256": owner_nonce_sha,
            "owner_nonce_ciphertext_b64": base64.b64encode(ciphertext).decode(
                "ascii"
            ),
            "owner_nonce_ciphertext_sha256": hashlib.sha256(
                ciphertext
            ).hexdigest(),
            "encryption_context": dict(encryption_context),
            "prepared_at": prepared_at,
        }
        append_prepared = getattr(self._wal, "append_prepared", None)
        if not callable(append_prepared):
            raise LaunchCustodyError("prepared WAL boundary is absent")
        prepared_sha = append_prepared(prepared_entry)
        _sha(prepared_sha, "prepared WAL identity")
        self._boundary("AFTER_PREPARED_WAL")
        intent = _rehash_intent(
            WorkerLaunchIntent(
                schema_version=1,
                record_type="glm52_worker_launch_intent_v1",
                account_id=ACCOUNT_ID,
                region=REGION,
                run_id=RUN_ID,
                campaign_identity_sha256=authority.campaign_identity_sha256,
                activation_id=authority.activation_id,
                activation_ordinal=authority.activation_ordinal,
                generation=authority.generation,
                action_key=authority.action_key,
                allocation_ordinal=context.allocation_ordinal,
                ec2_client_token=token,
                launch_parameters=parameters,
                launch_parameters_sha256=(
                    parameters.canonical_identity_sha256
                ),
                expected_worker_tags_sha256=canonical_sha256(
                    _tag_map(authority, context.allocation_ordinal)
                ),
                prior_worker_launch_identity_sha256=(
                    context.prior_worker_launch_identity_sha256
                ),
                prior_instance_terminal_identity_sha256=(
                    context.prior_instance_terminal_identity_sha256
                ),
                prior_spend_allocation_close_identity_sha256=(
                    context.prior_spend_allocation_close_identity_sha256
                ),
                prior_liability_settlement_identity_sha256=(
                    context.prior_liability_settlement_identity_sha256
                ),
                owner_invocation_nonce_sha256=owner_nonce_sha,
                owner_nonce_ciphertext_sha256=hashlib.sha256(
                    ciphertext
                ).hexdigest(),
                prepared_at=prepared_at,
                prepared_journal_entry_sha256=prepared_sha,
                ddb_committed_journal_entry_sha256=None,
                state="PREPARED_NOT_SENT",
                send_stage="NOT_SENT",
                possibly_sent_at=None,
                gpu_liability_reserve_ledger_identity_sha256=None,
                canonical_identity_sha256="",
            )
        )
        put_prepared = getattr(self._store, "put_prepared", None)
        read_launch = getattr(self._store, "exact_read_launch", None)
        bind_committed = getattr(self._store, "bind_committed_wal", None)
        append_committed = getattr(self._wal, "append_committed", None)
        if not all(
            callable(item)
            for item in (
                put_prepared,
                read_launch,
                bind_committed,
                append_committed,
            )
        ):
            raise LaunchCustodyError("prepared launch store is incomplete")
        put_prepared(intent, owner_nonce)
        self._boundary("AFTER_PREPARED_DDB")
        readback = read_launch(
            authority.activation_id, context.allocation_ordinal
        )
        if readback != intent:
            raise LaunchCustodyError("prepared launch readback mismatched")
        committed_entry = {
            "schema_version": 1,
            "record_type": "WORKER_LAUNCH_DDB_COMMITTED",
            "prepared_journal_entry_sha256": prepared_sha,
            "worker_launch_identity_sha256": intent.canonical_identity_sha256,
            "committed_at": _utc_text(self._clock()),
        }
        committed_sha = append_committed(committed_entry)
        _sha(committed_sha, "committed WAL identity")
        self._boundary("AFTER_COMMITTED_WAL")
        intent = bind_committed(intent, committed_sha, owner_nonce)
        if (
            not isinstance(intent, WorkerLaunchIntent)
            or intent.ddb_committed_journal_entry_sha256 != committed_sha
        ):
            raise LaunchCustodyError("committed WAL binding mismatched")
        self._boundary("AFTER_COMMITTED_DDB")
        reserve_method = getattr(self._reserve, "reserve", None)
        if not callable(reserve_method):
            raise LaunchCustodyError("Task 8 reserve boundary is absent")
        reserve = reserve_method(
            {
                "activation_id": authority.activation_id,
                "generation": authority.generation,
                "allocation_ordinal": context.allocation_ordinal,
                "ec2_client_token": token,
                "launch_parameters_sha256": (
                    parameters.canonical_identity_sha256
                ),
                "gpu_reserve_seconds": GPU_RESERVE_SECONDS,
                "gpu_reserve_cost_usd": GPU_RESERVE_COST_USD,
                "root_volume_gib": ROOT_VOLUME_GIB,
                "root_volume_tail_usd_max": ROOT_VOLUME_TAIL_USD_MAX,
                "owner_invocation_nonce_sha256": owner_nonce_sha,
            }
        )
        if (
            type(reserve) is not dict
            or set(reserve)
            != {
                "reserve_identity_sha256",
                "gpu_reserve_seconds",
                "gpu_reserve_cost_usd",
                "root_volume_tail_usd_max",
            }
            or reserve["gpu_reserve_seconds"] != GPU_RESERVE_SECONDS
            or reserve["gpu_reserve_cost_usd"] != GPU_RESERVE_COST_USD
            or reserve["root_volume_tail_usd_max"]
            != ROOT_VOLUME_TAIL_USD_MAX
        ):
            raise LaunchCustodyError("Task 8 reserve did not authenticate")
        _sha(reserve["reserve_identity_sha256"], "reserve identity")
        self._boundary("AFTER_RESERVE")
        commit_send = getattr(self._store, "commit_possibly_sent", None)
        if not callable(commit_send):
            raise LaunchCustodyError("possibly-sent transaction is absent")
        launch, liability = commit_send(intent, reserve, owner_nonce)
        if (
            not isinstance(launch, WorkerLaunchIntent)
            or launch.state != "POSSIBLY_SENT"
            or launch.send_stage != "POSSIBLY_SENT"
            or type(liability) is not dict
            or liability.get("state") != "UNOWNED_NOT_ACTIONABLE"
            or liability.get("allocation_ordinal") != context.allocation_ordinal
        ):
            raise LaunchCustodyError("possibly-sent transaction mismatched")
        self._boundary("AFTER_POSSIBLY_SENT")
        wait = getattr(self._store, "wait_watching", None)
        signal = getattr(self._store, "signal_ready", None)
        poll = getattr(self._store, "poll_result", None)
        if not all(callable(item) for item in (wait, signal, poll)):
            raise LaunchCustodyError("watcher handoff boundary is incomplete")
        watching = wait(authority.activation_id, context.allocation_ordinal)
        if (
            type(watching) is not dict
            or watching.get("state") != "WATCHING"
            or watching.get("allocation_ordinal")
            != context.allocation_ordinal
        ):
            raise LaunchCustodyError("liability watcher ownership is incoherent")
        signal(authority.activation_id, context.allocation_ordinal)
        result = poll(authority.activation_id, context.allocation_ordinal)
        if not isinstance(result, LaunchResult):
            raise LaunchCustodyError("launch result is not typed")
        return result


@dataclass(frozen=True)
class AttemptPermit:
    may_send: bool
    attempt: int
    action_identity_sha256: str


@dataclass(frozen=True)
class CompletionResult:
    classification: str
    attempt: int
    instance_ids: Tuple[str, ...]
    evidence_identity_sha256: str


class SameTokenCompleter:
    """Sole sender for attempt one and all bounded same-token attempts."""

    def __init__(
        self,
        *,
        store: object,
        ec2: object,
        clock: Callable[[], datetime],
    ) -> None:
        self._store = store
        self._ec2 = ec2
        self._clock = clock

    def complete(self, value: Mapping[str, object]) -> CompletionResult:
        if type(value) is not dict or set(value) != {
            "activation_id",
            "allocation_ordinal",
        }:
            raise LaunchCustodyError(
                "same-token completer accepts only activation and ordinal"
            )
        activation_id = _text(value["activation_id"], "activation id")
        ordinal = value["allocation_ordinal"]
        if type(ordinal) is not int or ordinal <= 0:
            raise LaunchCustodyError("allocation ordinal is invalid")
        read = getattr(self._store, "read_completion_authority", None)
        consume = getattr(self._store, "consume_attempt", None)
        if not callable(read) or not callable(consume):
            raise LaunchCustodyError("same-token authority store is incomplete")
        authority = read(activation_id, ordinal)
        expected_fields = {
            "activation_id",
            "allocation_ordinal",
            "state",
            "same_token_completion_attempts",
            "possibly_sent_at",
            "ec2_client_token",
            "launch_parameters",
            "current_owner",
            "reserve_held",
        }
        if type(authority) is not dict or set(authority) != expected_fields:
            raise LaunchCustodyError("same-token authority is not closed")
        count = authority["same_token_completion_attempts"]
        if type(count) is not int or count < 0 or count >= MAX_COMPLETION_CALLS:
            raise LaunchCustodyError("six same-token calls are exhausted")
        if (
            authority["activation_id"] != activation_id
            or authority["allocation_ordinal"] != ordinal
            or authority["state"] != "WATCHING"
            or authority["current_owner"] is not True
            or authority["reserve_held"] is not True
        ):
            raise LaunchCustodyError("current watching owner is absent")
        parameters = authority["launch_parameters"]
        if not isinstance(parameters, LaunchParameters):
            raise LaunchCustodyError("immutable launch parameters are absent")
        validate_launch_parameters(parameters, parameters.authority)
        expected_token = build_deterministic_client_token(
            parameters.authority, ordinal, parameters
        )
        if authority["ec2_client_token"] != expected_token:
            raise LaunchCustodyError("deterministic ClientToken mismatched")
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise LaunchCustodyError("completion clock must be timezone-aware")
        now = now.astimezone(timezone.utc)
        sent_at = _utc(authority["possibly_sent_at"], "possibly_sent_at")
        if now < sent_at or (
            now - sent_at
        ).total_seconds() >= COMPLETION_WINDOW_SECONDS:
            raise LaunchCustodyError("six-minute completion window elapsed")
        if getattr(self._ec2, "total_max_attempts", None) != 1:
            raise LaunchCustodyError("EC2 adapter retry configuration drifted")
        permit = consume(activation_id, ordinal)
        if (
            not isinstance(permit, AttemptPermit)
            or permit.may_send is not True
            or permit.attempt != count + 1
        ):
            raise LaunchCustodyError(
                "liability action/counter did not authorize this call"
            )
        _sha(permit.action_identity_sha256, "liability action identity")
        request = dict(parameters.run_instances)
        request["ClientToken"] = expected_token
        record_direct = getattr(self._store, "record_direct", None)
        record_ambiguous = getattr(self._store, "record_ambiguous", None)
        record_rejection = getattr(self._store, "record_rejection", None)
        if not all(
            callable(item)
            for item in (
                record_direct,
                record_ambiguous,
                record_rejection,
            )
        ):
            raise LaunchCustodyError("completion result store is incomplete")
        try:
            response = self._ec2.run_instances(**request)
        except PositiveRunInstancesRejection as exc:
            evidence = canonical_sha256(
                {
                    "classification": "POSITIVE_REJECTION",
                    "attempt": permit.attempt,
                    "error": str(exc),
                    "request_sha256": canonical_sha256(request),
                }
            )
            record_rejection(permit, evidence)
            return CompletionResult(
                "POSITIVE_REJECTION", permit.attempt, (), evidence
            )
        except AmbiguousRunInstances as exc:
            evidence = canonical_sha256(
                {
                    "classification": "AMBIGUOUS",
                    "attempt": permit.attempt,
                    "error_type": type(exc).__name__,
                    "request_sha256": canonical_sha256(request),
                }
            )
            record_ambiguous(permit, evidence)
            return CompletionResult("AMBIGUOUS", permit.attempt, (), evidence)
        if (
            type(response) is not dict
            or set(response) != {"ResponseMetadata", "Instances"}
            or type(response["ResponseMetadata"]) is not dict
            or response["ResponseMetadata"].get("HTTPStatusCode") != 200
            or type(response["ResponseMetadata"].get("RequestId")) is not str
            or not response["ResponseMetadata"]["RequestId"]
            or type(response["Instances"]) is not list
            or not response["Instances"]
        ):
            raise LaunchCustodyError("direct EC2 response is malformed")
        instance_ids = tuple(
            sorted(item.get("InstanceId") for item in response["Instances"])
        )
        if (
            any(
                type(item) is not str or _INSTANCE.fullmatch(item) is None
                for item in instance_ids
            )
            or len(instance_ids) != len(set(instance_ids))
        ):
            raise LaunchCustodyError("direct EC2 instance identity is invalid")
        evidence = canonical_sha256(
            {
                "classification": "DIRECT_SUCCESS",
                "attempt": permit.attempt,
                "request_sha256": canonical_sha256(request),
                "request_id": response["ResponseMetadata"]["RequestId"],
                "instance_ids": instance_ids,
                "response_sha256": canonical_sha256(response),
            }
        )
        record_direct(permit, response)
        return CompletionResult(
            "DIRECT_SUCCESS", permit.attempt, instance_ids, evidence
        )


@dataclass(frozen=True)
class WatchInput:
    event_accelerator: Optional[str] = None


@dataclass(frozen=True)
class WatchResult:
    settled: bool
    instance_ids: Tuple[str, ...]
    next_scan_at: str
    incident: bool


@dataclass(frozen=True)
class TerminationPermit:
    may_terminate: bool
    instance_id: str
    call_count: int
    window_started_at: str
    action_identity_sha256: str


class LiabilityWatcher:
    """One-minute retained discovery and exact-instance termination owner."""

    def __init__(
        self,
        *,
        store: object,
        discovery: object,
        terminator: object,
        clock: Callable[[], datetime],
    ) -> None:
        self._store = store
        self._discovery = discovery
        self._terminator = terminator
        self._clock = clock

    def _call_before_deadline(
        self,
        *,
        deadline: datetime,
        label: str,
        method: Callable[..., object],
        args: Tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> object:
        for phase in ("before", "after"):
            observed = self._clock()
            if (
                not isinstance(observed, datetime)
                or observed.tzinfo is None
                or observed.astimezone(timezone.utc) > deadline
            ):
                raise LaunchCustodyError(
                    "20-second watch action deadline elapsed "
                    + phase
                    + " "
                    + label
                )
            if phase == "before":
                result = method(*args, **dict(kwargs))
        return result

    def run(self, value: WatchInput) -> WatchResult:
        if not isinstance(value, WatchInput) or (
            value.event_accelerator
            not in {None, "ec2-state-change", "cloudtrail", "owner-failover"}
        ):
            raise LaunchCustodyError("watch input is not parameterless/closed")
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise LaunchCustodyError("watch clock must be timezone-aware")
        now = now.astimezone(timezone.utc)
        deadline = now + timedelta(seconds=WATCH_ACTION_DEADLINE_SECONDS)
        acquire = getattr(self._store, "acquire_or_takeover", None)
        correlate = getattr(self._discovery, "correlate", None)
        record_scan = getattr(self._store, "record_scan", None)
        if not all(callable(item) for item in (acquire, correlate, record_scan)):
            raise LaunchCustodyError("liability watcher boundary is incomplete")
        authority = self._call_before_deadline(
            deadline=deadline,
            label="owner acquisition",
            method=acquire,
            args=(now,),
            kwargs={"deadline": deadline},
        )
        required = {
            "activation_id",
            "allocation_ordinal",
            "ec2_client_token",
            "expected_worker_tags_sha256",
            "state",
            "watch_started_at",
            "next_scan_at",
            "current_owner",
            "settlement_identity_sha256",
            "work_authorized",
            "authorized_instance_id",
        }
        if (
            type(authority) is not dict
            or set(authority) != required
            or authority["current_owner"] is not True
            or authority["state"]
            not in {
                "WATCHING",
                "SAME_TOKEN_COMPLETION",
                "REJECTION_PROVED_AWAITING_TERMINAL_V2",
                "LATE_INSTANCE_DRAINING",
                "LIABILITY_INCIDENT",
            }
            or authority["settlement_identity_sha256"] is not None
            or type(authority["work_authorized"]) is not bool
            or (
                authority["authorized_instance_id"] is not None
                and (
                    type(authority["authorized_instance_id"]) is not str
                    or _INSTANCE.fullmatch(
                        authority["authorized_instance_id"]
                    )
                    is None
                )
            )
        ):
            raise LaunchCustodyError("retained liability owner is invalid")
        token = _sha(authority["ec2_client_token"], "EC2 ClientToken")
        tags = _sha(
            authority["expected_worker_tags_sha256"], "expected worker tags"
        )
        evidence = self._call_before_deadline(
            deadline=deadline,
            label="correlation",
            method=correlate,
            args=(),
            kwargs={
                "client_token": token,
                "expected_tags_sha256": tags,
                "deadline": deadline,
            },
        )
        if type(evidence) is not dict or set(evidence) != {
            "cloudtrail_request_identities",
            "instances",
            "state_change_identities",
            "launch_evidence_identities",
            "spend_evidence_identities",
            "terminal_v2_published",
            "evidence_identity_sha256",
        }:
            raise LaunchCustodyError("liability correlation is not closed")
        _sha(evidence["evidence_identity_sha256"], "scan evidence")
        instances = evidence["instances"]
        if type(instances) is not tuple:
            raise LaunchCustodyError("instance correlation must be an exact tuple")
        instance_ids = tuple(
            sorted(item.get("InstanceId") for item in instances)
        )
        if (
            any(
                type(item) is not str or _INSTANCE.fullmatch(item) is None
                for item in instance_ids
            )
            or len(instance_ids) != len(set(instance_ids))
        ):
            raise LaunchCustodyError("correlated instance identity is invalid")
        scheduled = _utc(authority["next_scan_at"], "next_scan_at")
        next_scan = scheduled + timedelta(seconds=WATCH_INTERVAL_SECONDS)
        self._call_before_deadline(
            deadline=deadline,
            label="scan record",
            method=record_scan,
            args=(authority, evidence, _utc_text(next_scan)),
            kwargs={"deadline": deadline},
        )
        if not instances:
            return WatchResult(False, (), _utc_text(next_scan), False)
        incident = False
        record_incident = getattr(self._store, "record_incident", None)
        open_late = getattr(self._store, "open_late_allocation", None)
        consume_termination = getattr(
            self._store, "consume_termination", None
        )
        reconcile = getattr(self._store, "reconcile_instance", None)
        settle = getattr(self._store, "settle_if_complete", None)
        terminate = getattr(self._terminator, "terminate", None)
        if not all(
            callable(item)
            for item in (
                record_incident,
                open_late,
                consume_termination,
                reconcile,
                settle,
                terminate,
            )
        ):
            raise LaunchCustodyError("termination custody boundary is incomplete")
        if getattr(self._terminator, "total_max_attempts", None) != 1:
            raise LaunchCustodyError("termination adapter retry drifted")
        if len(instance_ids) > 1:
            self._call_before_deadline(
                deadline=deadline,
                label="multiple-instance incident",
                method=record_incident,
                args=(authority, "MULTIPLE_INSTANCE_TOKEN"),
                kwargs={"deadline": deadline},
            )
            incident = True
        watch_age = now - _utc(
            authority["watch_started_at"], "watch_started_at"
        )
        if watch_age.total_seconds() >= CONTROL_PERIOD_SECONDS:
            self._call_before_deadline(
                deadline=deadline,
                label="control-period incident",
                method=record_incident,
                args=(authority, "UNSETTLED_30_DAY"),
                kwargs={"deadline": deadline},
            )
            incident = True
        terminal_states = {"terminated", "shutting-down"}
        for instance in sorted(instances, key=lambda item: item["InstanceId"]):
            instance_id = instance["InstanceId"]
            if (
                set(instance)
                != {"InstanceId", "State", "TagsSha256", "ObservedAt"}
                or instance["TagsSha256"] != tags
            ):
                raise LaunchCustodyError("instance correlation fields drifted")
            _utc(instance["ObservedAt"], "instance observed at")
            known_worker = (
                instance_id == authority["authorized_instance_id"]
            )
            if not known_worker:
                self._call_before_deadline(
                    deadline=deadline,
                    label="late allocation open",
                    method=open_late,
                    args=(
                        authority,
                        instance,
                        evidence["terminal_v2_published"] is True,
                    ),
                    kwargs={"deadline": deadline},
                )
            drain_required = (
                not known_worker
                or len(instance_ids) > 1
                or authority["work_authorized"] is False
            )
            if (
                drain_required
                and instance["State"] not in terminal_states
            ):
                permit = self._call_before_deadline(
                    deadline=deadline,
                    label="termination action consumption",
                    method=consume_termination,
                    args=(authority, instance_id),
                    kwargs={"deadline": deadline},
                )
                if (
                    not isinstance(permit, TerminationPermit)
                    or permit.may_terminate is not True
                    or permit.instance_id != instance_id
                    or type(permit.call_count) is not int
                    or permit.call_count < 1
                    or permit.call_count > 6
                ):
                    raise LaunchCustodyError(
                        "six termination calls require an exact consumed action"
                    )
                _utc(
                    permit.window_started_at,
                    "termination window started at",
                )
                _sha(
                    permit.action_identity_sha256,
                    "termination action identity",
                )
                response = self._call_before_deadline(
                    deadline=deadline,
                    label="termination effect",
                    method=terminate,
                    args=(instance_id,),
                    kwargs={"deadline": deadline},
                )
                if (
                    type(response) is not dict
                    or set(response) != {"request_id", "response_sha256"}
                ):
                    raise LaunchCustodyError(
                        "termination response is malformed"
                    )
                _text(response["request_id"], "termination request id")
                _sha(response["response_sha256"], "termination response")
            self._call_before_deadline(
                deadline=deadline,
                label="instance reconciliation",
                method=reconcile,
                args=(authority, instance_id),
                kwargs={"deadline": deadline},
            )
        settlement = self._call_before_deadline(
            deadline=deadline,
            label="settlement",
            method=settle,
            args=(authority,),
            kwargs={"deadline": deadline},
        )
        return WatchResult(
            settlement is not None,
            instance_ids,
            _utc_text(next_scan),
            incident,
        )


def build_post_terminal_allocation(
    *,
    campaign_identity_sha256: str,
    activation_id: str,
    activation_ordinal: int,
    generation: int,
    allocation_ordinal: int,
    ec2_client_token: str,
    worker_launch_identity_sha256: str,
    liability_identity_sha256: str,
    terminal_v2_identity_sha256: str,
    instance_id: str,
    instance_tags_sha256: str,
    owner: Mapping[str, object],
    discovered_at: str,
) -> Mapping[str, object]:
    _sha(campaign_identity_sha256, "campaign identity")
    if (
        type(owner) is not dict
        or set(owner)
        != {
            "owner_attempt",
            "owner_execution_arn",
            "owner_state_machine_version_arn",
            "owner_dispatch_identity_sha256",
            "owner_invocation_nonce_sha256",
            "owner_hard_expires_at",
        }
    ):
        raise LaunchCustodyError("post-terminal owner is not closed")
    body = {
        "schema_version": 1,
        "record_type": "glm52_production_post_terminal_allocation",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "campaign_identity_sha256": campaign_identity_sha256,
        "activation_id": activation_id,
        "activation_ordinal": activation_ordinal,
        "generation": generation,
        "allocation_ordinal": allocation_ordinal,
        "ec2_client_token": ec2_client_token,
        "worker_launch_identity_sha256": worker_launch_identity_sha256,
        "worker_launch_liability_identity_sha256": liability_identity_sha256,
        "terminal_v2_identity_sha256": terminal_v2_identity_sha256,
        "instance_id": instance_id,
        "instance_tags_sha256": instance_tags_sha256,
        "discovery_source": "EXACT_TOKEN_TAG_WINDOW",
        "discovered_at": discovered_at,
        "state": "DISCOVERED",
        "allocation_open_identity_sha256": None,
        "charged_interval_started_at": None,
        "instance_terminal_identity_sha256": None,
        "charged_interval_ended_at": None,
        "spend_allocation_close_identity_sha256": None,
        "concurrency_window_identity_sha256": None,
        **dict(owner),
        "revision": 0,
        "updated_at": discovered_at,
        "canonical_body_sha256": "",
    }
    body["canonical_body_sha256"] = canonical_record_identity_unchecked(body)
    try:
        return validate_record(
            "glm52_production_post_terminal_allocation", body
        )
    except (TypeError, ValueError) as exc:
        raise LaunchCustodyError(
            "post-terminal allocation did not validate"
        ) from exc


class LiabilitySettlementCoordinator:
    """Re-read, recompute, and atomically settle the retained final view."""

    def __init__(
        self, *, store: object, clock: Callable[[], datetime]
    ) -> None:
        self._store = store
        self._clock = clock

    def build(
        self, activation_id: str, allocation_ordinal: int
    ) -> Mapping[str, object]:
        read = getattr(self._store, "read_complete_view", None)
        if not callable(read):
            raise LaunchCustodyError("settlement exact-read boundary is absent")
        view = read(activation_id, allocation_ordinal)
        expected = {
            "campaign_identity_sha256",
            "activation_id",
            "activation_ordinal",
            "generation",
            "allocation_ordinal",
            "worker_launch_identity_sha256",
            "worker_launch_liability_identity_sha256",
            "terminal_v2",
            "post_terminal_allocations",
            "service_rejection_evidence_sha256",
            "instance_terminal_evidence",
            "spend_close_evidence",
            "reserve_identity_sha256",
            "reserve_release_identity_sha256",
            "residual_liability_approval_identity_sha256",
            "final_spend_ledger_head_identity_sha256",
            "prior_incident_identity_sha256",
            "owner",
        }
        if type(view) is not dict or set(view) != expected:
            raise LaunchCustodyError("settlement view is not closed")
        if (
            view["activation_id"] != activation_id
            or view["allocation_ordinal"] != allocation_ordinal
        ):
            raise LaunchCustodyError("settlement view is foreign")
        campaign_identity_sha256 = _sha(
            view["campaign_identity_sha256"],
            "campaign identity",
        )
        terminal = view["terminal_v2"]
        if type(terminal) is not dict or set(terminal) != {
            "key",
            "version_id",
            "body_sha256",
            "allocations",
            "allocations_array_sha256",
        }:
            raise LaunchCustodyError("terminal-v2 identity is incomplete")
        if terminal["allocations_array_sha256"] != canonical_sha256(
            terminal["allocations"]
        ):
            raise LaunchCustodyError("terminal-v2 allocation hash drifted")
        _sha(terminal["body_sha256"], "terminal-v2 body")
        post = view["post_terminal_allocations"]
        if type(post) is not list:
            raise LaunchCustodyError("post-terminal array is not exact")
        for item in post:
            if (
                type(item) is not dict
                or item.get("state") != "ALLOCATION_CLOSED"
                or type(item.get("instance_id")) is not str
                or _INSTANCE.fullmatch(item["instance_id"]) is None
            ):
                raise LaunchCustodyError(
                    "every post-terminal member must be closed"
                )
            for field in (
                "instance_terminal_identity_sha256",
                "spend_allocation_close_identity_sha256",
                "canonical_body_sha256",
            ):
                _sha(item.get(field), "post-terminal " + field)
        post = sorted(
            post,
            key=lambda item: (
                item["allocation_ordinal"],
                item["instance_id"],
            ),
        )
        merged = list(terminal["allocations"]) + list(post)
        merged.sort(
            key=lambda item: (
                item["allocation_ordinal"],
                item["instance_id"],
            )
        )
        instances = [item["instance_id"] for item in merged]
        if len(instances) != len(set(instances)):
            raise LaunchCustodyError("merged allocation contains duplicates")
        rejection = view["service_rejection_evidence_sha256"]
        if not merged:
            _sha(rejection, "positive rejection evidence")
            kind = "NO_INSTANCE_POSITIVE_REJECTION"
        else:
            if rejection is not None:
                raise LaunchCustodyError(
                    "instance settlement forbids rejection evidence"
                )
            kind = "ALL_INSTANCES_TERMINAL_AND_SPEND_CLOSED"
        terminal_evidence = view["instance_terminal_evidence"]
        spend_evidence = view["spend_close_evidence"]
        if type(terminal_evidence) is not list or type(spend_evidence) is not list:
            raise LaunchCustodyError("settlement evidence arrays are invalid")
        expected_terminal_evidence = []
        expected_spend_evidence = []
        for item in merged:
            if type(item) is not dict:
                raise LaunchCustodyError("final allocation member is malformed")
            expected_terminal_evidence.append(
                _sha(
                    item.get("instance_terminal_identity_sha256"),
                    "final member terminal identity",
                )
            )
            expected_spend_evidence.append(
                _sha(
                    item.get("spend_allocation_close_identity_sha256"),
                    "final member spend-close identity",
                )
            )
        expected_terminal_evidence.sort()
        expected_spend_evidence.sort()
        terminal_evidence = sorted(terminal_evidence)
        spend_evidence = sorted(spend_evidence)
        if (
            terminal_evidence != expected_terminal_evidence
            or spend_evidence != expected_spend_evidence
        ):
            raise LaunchCustodyError(
                "terminal and spend evidence must bind exact final members"
            )
        for item in terminal_evidence + spend_evidence:
            _sha(item, "settlement member evidence")
        owner = view["owner"]
        if type(owner) is not dict or set(owner) != {
            "owner_execution_arn",
            "owner_state_machine_version_arn",
            "owner_dispatch_identity_sha256",
            "owner_invocation_nonce_sha256",
        }:
            raise LaunchCustodyError("settlement owner is not closed")
        for field in (
            "worker_launch_identity_sha256",
            "worker_launch_liability_identity_sha256",
            "reserve_identity_sha256",
            "reserve_release_identity_sha256",
            "residual_liability_approval_identity_sha256",
            "final_spend_ledger_head_identity_sha256",
        ):
            _sha(view[field], field)
        if view["prior_incident_identity_sha256"] is not None:
            _sha(
                view["prior_incident_identity_sha256"],
                "prior incident identity",
            )
        now = _utc_text(self._clock())
        body = {
            "schema_version": 1,
            "record_type": (
                "glm52_production_worker_launch_liability_settlement"
            ),
            "account_id": ACCOUNT_ID,
            "region": REGION,
            "run_id": RUN_ID,
            "campaign_identity_sha256": campaign_identity_sha256,
            "activation_id": activation_id,
            "activation_ordinal": view["activation_ordinal"],
            "generation": view["generation"],
            "allocation_ordinal": allocation_ordinal,
            "worker_launch_identity_sha256": (
                view["worker_launch_identity_sha256"]
            ),
            "worker_launch_liability_identity_sha256": (
                view["worker_launch_liability_identity_sha256"]
            ),
            "settlement_kind": kind,
            "terminal_v2_key": terminal["key"],
            "terminal_v2_version_id": terminal["version_id"],
            "terminal_v2_body_sha256": terminal["body_sha256"],
            "terminal_v2_allocations_array_sha256": (
                terminal["allocations_array_sha256"]
            ),
            "post_terminal_allocations": post,
            "post_terminal_allocations_array_sha256": canonical_sha256(post),
            "merged_final_allocations": merged,
            "merged_final_allocations_array_sha256": canonical_sha256(merged),
            "final_worker_cardinality": str(len(merged)),
            "service_rejection_evidence_sha256": rejection,
            "instance_terminal_evidence_array_sha256": canonical_sha256(
                terminal_evidence
            ),
            "spend_close_evidence_array_sha256": canonical_sha256(
                spend_evidence
            ),
            "gpu_liability_reserve_ledger_identity_sha256": (
                view["reserve_identity_sha256"]
            ),
            "gpu_liability_reserve_release_identity_sha256": (
                view["reserve_release_identity_sha256"]
            ),
            "ebs_liability_reserve_cost_usd": ROOT_VOLUME_TAIL_USD_MAX,
            "residual_liability_approval_identity_sha256": (
                view["residual_liability_approval_identity_sha256"]
            ),
            "final_spend_ledger_head_identity_sha256": (
                view["final_spend_ledger_head_identity_sha256"]
            ),
            "prior_incident_identity_sha256": (
                view["prior_incident_identity_sha256"]
            ),
            **owner,
            "transaction_client_request_token_sha256": canonical_sha256(
                {
                    "domain": "LIABILITY_SETTLE",
                    "activation_id": activation_id,
                    "allocation_ordinal": allocation_ordinal,
                    "terminal_v2_body_sha256": terminal["body_sha256"],
                    "merged_final_allocations_array_sha256": canonical_sha256(
                        merged
                    ),
                }
            ),
            "settled_at": now,
            "canonical_body_sha256": "",
        }
        body["canonical_body_sha256"] = canonical_record_identity_unchecked(body)
        try:
            return validate_record(
                "glm52_production_worker_launch_liability_settlement",
                body,
            )
        except (TypeError, ValueError) as exc:
            raise LaunchCustodyError(
                "liability settlement record did not validate"
            ) from exc

    def settle(
        self, activation_id: str, allocation_ordinal: int
    ) -> Mapping[str, object]:
        settlement = self.build(activation_id, allocation_ordinal)
        transact = getattr(self._store, "transact_settle_once", None)
        read = getattr(self._store, "coherent_read_settlement", None)
        if not callable(transact) or not callable(read):
            raise LaunchCustodyError("settlement transaction boundary is absent")
        response = transact(settlement)
        if response == settlement:
            return settlement
        if (
            type(response) is not dict
            or response.get("classification") != "AMBIGUOUS"
        ):
            raise LaunchCustodyError("settlement transaction was rejected")
        key = (
            "WORKER_LAUNCH_LIABILITY_SETTLEMENT#"
            + activation_id
            + "#%08d" % allocation_ordinal
        )
        readback = read(key)
        if readback != settlement:
            raise LaunchCustodyError(
                "ambiguous settlement did not reconcile exactly"
            )
        return settlement


__all__ = [
    "ACCOUNT_ID",
    "REGION",
    "RUN_ID",
    "AmbiguousRunInstances",
    "AttemptPermit",
    "CompletionResult",
    "IntentOnlyProvisioner",
    "LaunchContext",
    "LaunchCustodyError",
    "LaunchParameterAuthority",
    "LaunchParameters",
    "LaunchResult",
    "LiabilitySettlementCoordinator",
    "LiabilityWatcher",
    "PositiveRunInstancesRejection",
    "RecoveredPreparedLaunch",
    "SameTokenCompleter",
    "Task8GpuReserveAdapter",
    "TerminationPermit",
    "WatchInput",
    "WatchResult",
    "WorkerLaunchIntent",
    "build_deterministic_client_token",
    "build_launch_parameters",
    "build_post_terminal_allocation",
    "validate_launch_parameters",
]
