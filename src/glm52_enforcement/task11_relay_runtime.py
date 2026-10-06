"""Finite no-retry runtime adapters for Task 11 relay Lambdas."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import base64
import hashlib
import http.client
import json
import os
from pathlib import Path
import ssl
import tempfile
from typing import Mapping, Optional

from .canonical import canonical_json_bytes, canonical_sha256
from .dynamodb import (
    DynamoLedgerAdapter,
    LedgerKey,
    encode_attribute_value,
    encode_item,
)
from .h1f_adapter import (
    FreshH1fAuditService,
    H1fAuditRequest,
    H1fAuditResult,
)
from .records import ledger_pk, validate_record
from .s3_records import build_immutable_json_candidate
from .sky_admission import (
    AdmissionRequest,
    RelayIdentity,
    SkyIdentityContract,
    build_admission_custody_identity,
    validate_sky_identity_contract,
)
from .task9_contract import (
    build_task9_contract,
    validate_task9_deployed_identity,
)
from .task11_support_boundary import (
    ExactInputCoordinate,
    exact_input_coordinate_from_mapping,
    load_exact_input,
)
from .transitions import validate_transition


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
_MAX_BODY = 2 * 1024 * 1024


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_admission_request(
    request: AdmissionRequest,
) -> Mapping[str, object]:
    value = asdict(request)
    value.pop("private_nonce")
    value["private_nonce_sha256"] = hashlib.sha256(
        request.private_nonce
    ).hexdigest()
    return value


@dataclass(frozen=True)
class LoadedTask9DeployedIdentity:
    document: Mapping[str, object]
    sky_identity: SkyIdentityContract
    body_sha256: str
    coordinate: ExactInputCoordinate


def load_task9_deployed_identity(
    *,
    s3: object,
    coordinate: object,
    activation_id: str,
    expected_body_sha256: str,
    expected_bucket: str,
) -> LoadedTask9DeployedIdentity:
    """Exact-load the authenticated postcreate identity; there is no fallback."""

    exact = (
        coordinate
        if type(coordinate) is ExactInputCoordinate
        else exact_input_coordinate_from_mapping(
            coordinate,
            expected_kind="TASK9_DEPLOYED_IDENTITY_COORDINATE",
        )
    )
    expected_key = (
        "campaigns/"
        + RUN_ID
        + "/authorities/task9/"
        + activation_id
        + "/TASK9_DEPLOYED_IDENTITY.json"
    )
    if (
        type(activation_id) is not str
        or not activation_id
        or exact.bucket != expected_bucket
        or exact.key != expected_key
        or exact.body_sha256 != expected_body_sha256
    ):
        raise RuntimeError("deployed Task 9 coordinate is missing or foreign")
    value = load_exact_input(s3=s3, coordinate=exact)
    static_contract = build_task9_contract(
        Path(__file__).resolve().parents[2]
    )
    try:
        validate_task9_deployed_identity(
            value,
            static_contract=static_contract,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("deployed Task 9 identity drifted") from exc
    if value["activation_id"] != activation_id:
        raise RuntimeError("deployed Task 9 activation drifted")
    identity = value["sky_identity"]
    if type(identity) is not dict or type(identity.get("identities")) is not list:
        raise RuntimeError("deployed Task 9 Sky identity drifted")
    try:
        result = SkyIdentityContract(
            **{
                **identity,
                "identities": tuple(
                    RelayIdentity(
                        **{
                            **item,
                            "allowed_paths": tuple(item["allowed_paths"]),
                        }
                    )
                    for item in identity["identities"]
                ),
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("deployed Task 9 identity is malformed") from exc
    return LoadedTask9DeployedIdentity(
        document=value,
        sky_identity=validate_sky_identity_contract(result),
        body_sha256=exact.body_sha256,
        coordinate=exact,
    )


@dataclass(frozen=True)
class RelayRuntimeClients:
    dynamodb: object
    s3: object
    secretsmanager: object


def aws_relay_runtime_clients() -> RelayRuntimeClients:
    """Construct the finite SDK boundary once with no automatic retry."""

    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - Lambda dependency
        raise RuntimeError("boto3 and botocore are required") from exc
    config = Config(
        connect_timeout=1,
        read_timeout=4,
        max_pool_connections=4,
        retries={"mode": "standard", "total_max_attempts": 1},
    )
    session = boto3.session.Session(region_name=REGION)
    return RelayRuntimeClients(
        dynamodb=session.client("dynamodb", config=config),
        s3=session.client("s3", config=config),
        secretsmanager=session.client("secretsmanager", config=config),
    )


class DynamoFreshnessNonceRegistry:
    """Consume a verifier nonce once with one conditional DynamoDB write."""

    def __init__(
        self,
        *,
        dynamodb: object,
        table_name: str,
        activation_id: str,
    ) -> None:
        self._client = dynamodb
        self._table = table_name
        self._activation = activation_id

    def consume(self, nonce: str) -> bool:
        digest = hashlib.sha256(nonce.encode("ascii")).hexdigest()
        item = {
            "PK": ledger_pk(RUN_ID),
            "SK": (
                "ACTIVATION#"
                + self._activation
                + "#TASK11_ATTESTATION_NONCE#"
                + digest
            ),
            "schema_version": 1,
            "record_type": "glm52_task11_attestation_nonce_v1",
            "run_id": RUN_ID,
            "activation_id": self._activation,
            "nonce_sha256": digest,
            "consumed_at": _utc(_now()),
        }
        try:
            response = self._client.put_item(
                TableName=self._table,
                Item=encode_item(item),
                ConditionExpression=(
                    "attribute_not_exists(#pk) AND attribute_not_exists(#sk)"
                ),
                ExpressionAttributeNames={"#pk": "PK", "#sk": "SK"},
                ReturnConsumedCapacity="NONE",
            )
        except Exception:
            return False
        metadata = (
            response.get("ResponseMetadata")
            if type(response) is dict
            else None
        )
        return bool(
            type(metadata) is dict
            and metadata.get("HTTPStatusCode") == 200
            and type(metadata.get("RequestId")) is str
            and metadata["RequestId"]
        )


def _secret_json(
    *,
    client: object,
    id_env_prefix: str,
) -> Mapping[str, object]:
    secret_id = os.environ.get(f"GLM52_{id_env_prefix}_SECRET_ID", "")
    version_id = os.environ.get(f"GLM52_{id_env_prefix}_VERSION_ID", "")
    version_stage = os.environ.get(
        f"GLM52_{id_env_prefix}_VERSION_STAGE",
        "",
    )
    if not all((secret_id, version_id, version_stage)):
        raise RuntimeError("relay secret coordinate is absent")
    response = client.get_secret_value(
        SecretId=secret_id,
        VersionId=version_id,
        VersionStage=version_stage,
    )
    metadata = response.get("ResponseMetadata")
    stages = response.get("VersionStages")
    raw = response.get("SecretString")
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or response.get("VersionId") != version_id
        or stages != [version_stage]
        or type(raw) is not str
    ):
        raise RuntimeError("relay secret response is unauthenticated")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("relay secret is not JSON") from exc
    if type(value) is not dict:
        raise RuntimeError("relay secret is not one object")
    return value


class PinnedMtlsRelay:
    """One-request mTLS relay transport with exact pinned secrets."""

    def __init__(
        self,
        *,
        secretsmanager: object,
        purpose: str,
        identity: SkyIdentityContract,
    ) -> None:
        self._secrets = secretsmanager
        self._purpose = purpose
        matches = tuple(
            item for item in identity.identities if item.purpose == purpose
        )
        if len(matches) != 1:
            raise RuntimeError("relay identity purpose is not unique")
        self._identity = matches[0]

    def request(
        self,
        *,
        method: str,
        path: str,
        body: Optional[bytes],
    ) -> Mapping[str, object]:
        if (
            method not in {"GET", "POST"}
            or path not in self._identity.allowed_paths
            or (method == "GET" and body is not None)
            or (method == "POST" and type(body) is not bytes)
        ):
            raise RuntimeError("relay request crossed its fixed surface")
        token = _secret_json(
            client=self._secrets,
            id_env_prefix="RAW_SKY_TOKEN",
        )
        tls = _secret_json(
            client=self._secrets,
            id_env_prefix={
                "ATTESTATION": "ATTESTATION_CLIENT_TLS",
                "LAUNCH_ADMISSION": "LAUNCH_ADMISSION_CLIENT_TLS",
                "NUMERIC_BINDING": "NUMERIC_BINDING_CLIENT_TLS",
                "RETAINED_CANCELLATION": (
                    "RETAINED_CANCELLATION_CLIENT_TLS"
                ),
            }[self._purpose],
        )
        if set(token) != {"token"} or set(tls) != {
            "client_certificate_pem",
            "client_private_key_pem",
            "ca_certificate_pem",
        }:
            raise RuntimeError("relay secret field set drifted")
        if any(type(value) is not str or not value for value in (*token.values(), *tls.values())):
            raise RuntimeError("relay secret material is empty")
        try:
            client_certificate_der = ssl.PEM_cert_to_DER_cert(
                tls["client_certificate_pem"]
            )
        except ValueError as exc:
            raise RuntimeError("relay client certificate is invalid") from exc
        if (
            hashlib.sha256(client_certificate_der).hexdigest()
            != self._identity.client_certificate_sha256
        ):
            raise RuntimeError("relay client certificate identity drifted")
        context = ssl.create_default_context(
            cadata=tls["ca_certificate_pem"]
        )
        with tempfile.NamedTemporaryFile("w", encoding="ascii") as cert_file:
            with tempfile.NamedTemporaryFile("w", encoding="ascii") as key_file:
                cert_file.write(tls["client_certificate_pem"])
                cert_file.flush()
                key_file.write(tls["client_private_key_pem"])
                key_file.flush()
                context.load_cert_chain(cert_file.name, key_file.name)
                host = os.environ.get("GLM52_COMBINED_HOST_PRIVATE_IP", "")
                port_text = os.environ.get("GLM52_RELAY_PORT", "")
                if (
                    not host
                    or not port_text.isdigit()
                    or int(port_text) != self._identity.port
                ):
                    raise RuntimeError("relay host coordinate is absent")
                connection = http.client.HTTPSConnection(
                    host,
                    self._identity.port,
                    timeout=4,
                    context=context,
                )
                try:
                    connection.request(
                        method,
                        path,
                        body=body,
                        headers={
                            "Authorization": "Bearer " + token["token"],
                            "Content-Type": "application/json",
                        },
                    )
                    socket = connection.sock
                    peer_der = (
                        socket.getpeercert(binary_form=True)
                        if socket is not None
                        else None
                    )
                    if type(peer_der) is not bytes or not peer_der:
                        raise RuntimeError(
                            "relay peer certificate is absent"
                        )
                    peer_identity = hashlib.sha256(peer_der).hexdigest()
                    if (
                        peer_identity
                        != self._identity.server_certificate_sha256
                    ):
                        raise RuntimeError(
                            "relay server certificate identity drifted"
                        )
                    response = connection.getresponse()
                    raw = response.read(_MAX_BODY + 1)
                except (OSError, TimeoutError, ssl.SSLError) as exc:
                    raise TimeoutError(
                        "relay transport outcome is ambiguous"
                    ) from exc
                finally:
                    connection.close()
        if len(raw) > _MAX_BODY:
            raise RuntimeError("relay response exceeded fixed bound")
        request_id = response.getheader("x-request-id")
        if type(request_id) is not str or not request_id:
            raise RuntimeError("relay response request ID is absent")
        return {
            "status_code": response.status,
            "request_id": request_id,
            "raw": raw,
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            "tls_peer_certificate_sha256": peer_identity,
        }


class RelayAttestor:
    def __init__(self, relay: PinnedMtlsRelay) -> None:
        self._relay = relay

    @staticmethod
    def _json(response: Mapping[str, object]) -> Mapping[str, object]:
        try:
            value = json.loads(response["raw"].decode("ascii"))
        except (AttributeError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("attestation relay response is invalid") from exc
        if type(value) is not dict:
            raise RuntimeError("attestation relay response is not closed")
        return value

    def inspect(self, request: object) -> Mapping[str, object]:
        nonce = getattr(request, "freshness_nonce", None)
        health_response = self._relay.request(
            method="GET",
            path="/api/health",
            body=None,
        )
        role_response = self._relay.request(
            method="GET",
            path="/users/role",
            body=None,
        )
        health = self._json(health_response)
        role = self._json(role_response)
        expected_health = {
            "tls_peer_certificate_sha256",
            "effective_controller_identity_sha256",
            "observed_at",
        }
        expected_role = {
            "sky_user_identity",
            "sky_roles",
            "token_expires_at",
        }
        if set(health) != expected_health or set(role) != expected_role:
            raise RuntimeError("attestation relay fields drifted")
        if (
            health["tls_peer_certificate_sha256"]
            != health_response["tls_peer_certificate_sha256"]
            or role_response["tls_peer_certificate_sha256"]
            != health_response["tls_peer_certificate_sha256"]
        ):
            raise RuntimeError(
                "attestation relay peer certificate claim drifted"
            )
        return {
            "freshness_nonce": nonce,
            "direct_response_request_id": canonical_sha256(
                {
                    "health_request_id": health_response["request_id"],
                    "role_request_id": role_response["request_id"],
                }
            ),
            "tls_peer_certificate_sha256": health[
                "tls_peer_certificate_sha256"
            ],
            "sky_user_identity": role["sky_user_identity"],
            "sky_roles": tuple(role["sky_roles"]),
            "token_expires_at": role["token_expires_at"],
            "effective_controller_identity_sha256": health[
                "effective_controller_identity_sha256"
            ],
            "observed_at": health["observed_at"],
        }


class RelayAdmissionSender:
    def __init__(self, relay: PinnedMtlsRelay) -> None:
        self._relay = relay
        self.call_count = 0

    def send(self, body: bytes) -> Mapping[str, object]:
        self.call_count += 1
        response = self._relay.request(
            method="POST",
            path="/jobs/launch",
            body=body,
        )
        return {
            "status_code": response["status_code"],
            "request_id": response["request_id"],
            "response_sha256": response["response_sha256"],
        }


class AdmissionDynamoStore:
    """Three one-shot action transitions with exact readback reconciliation."""

    def __init__(
        self,
        *,
        dynamodb: object,
        table_name: str,
        activation_id: str,
        generation: int,
        function_version_arn: str,
    ) -> None:
        self._client = dynamodb
        self._table = table_name
        self._activation = activation_id
        self._generation = generation
        self._version_arn = function_version_arn
        self._ledger = DynamoLedgerAdapter(
            client=dynamodb,
            table_name=table_name,
        )
        self.states = []
        self.audit_result: Optional[H1fAuditResult] = None

    def _read(self, request: AdmissionRequest):
        index, control, action = self._ledger.read_coherent(
            items=(
                (
                    LedgerKey(RUN_ID, "ACTIVATION_INDEX"),
                    "glm52_production_activation_index",
                ),
                (
                    LedgerKey(
                        RUN_ID,
                        "ACTIVATION#" + self._activation + "#CONTROL",
                    ),
                    "glm52_production_control",
                ),
                (
                    LedgerKey(RUN_ID, request.action_key),
                    "glm52_production_action",
                ),
            )
        )
        if (
            index["current_activation_id"] != self._activation
            or control["activation_id"] != self._activation
            or action["activation_id"] != self._activation
            or action["generation"] != self._generation
            or action["action_kind"] != "SKY_POST"
            or index["campaign_identity_sha256"]
            != control["campaign_identity_sha256"]
            or control["campaign_identity_sha256"]
            != action["campaign_identity_sha256"]
            or control["barrier_state"] != "ACQUIRED"
            or control["decision_seal_state"] != "SEALED"
            or control["barrier_nonce_sha256"]
            != action["barrier_nonce_sha256"]
        ):
            raise RuntimeError("launch-admission live authority drifted")
        return index, control, action

    def read(self, request: AdmissionRequest) -> str:
        return str(self._read(request)[2]["state"])

    def _transition(
        self,
        *,
        request: AdmissionRequest,
        index_before: Mapping[str, object],
        before: Mapping[str, object],
        after: Mapping[str, object],
        control_before: Mapping[str, object],
        control_after: Mapping[str, object],
        domain: str,
    ) -> str:
        validate_record(
            "glm52_production_action",
            after,
            sk=request.action_key,
        )
        validate_record("glm52_production_control", control_after)
        validate_transition(
            "glm52_production_action",
            before,
            after,
        )
        action_changed = {
            key: value
            for key, value in after.items()
            if before.get(key) != value
        }
        control_changed = {
            key: value
            for key, value in control_after.items()
            if control_before.get(key) != value
        }

        def exact_condition(
            *,
            values: Mapping[str, object],
            fields: tuple[str, ...],
            prefix: str,
        ) -> tuple[str, Mapping[str, str], Mapping[str, object]]:
            if any(field not in values for field in fields):
                raise RuntimeError(
                    "launch-admission condition field is absent"
                )
            names = {
                f"#{prefix}{position}": field
                for position, field in enumerate(fields)
            }
            encoded = {
                f":{prefix}{position}": encode_attribute_value(
                    values[field]
                )
                for position, field in enumerate(fields)
            }
            expression = " AND ".join(
                f"#{prefix}{position} = :{prefix}{position}"
                for position in range(len(fields))
            )
            return expression, names, encoded

        def update(
            *,
            key: Mapping[str, object],
            state_field: str,
            before_state: str,
            before_revision: int,
            changed: Mapping[str, object],
            immutable_before: Mapping[str, object],
            immutable_fields: tuple[str, ...],
        ) -> Mapping[str, object]:
            condition, condition_names, condition_values = exact_condition(
                values=immutable_before,
                fields=immutable_fields,
                prefix="c",
            )
            names = {
                "#state": state_field,
                "#revision": "revision",
                **condition_names,
                **{
                    "#f" + str(index): field
                    for index, field in enumerate(changed)
                },
            }
            values = {
                ":before_state": encode_attribute_value(before_state),
                ":before_revision": encode_attribute_value(before_revision),
                **condition_values,
                **{
                    ":v" + str(index): encode_attribute_value(value)
                    for index, value in enumerate(changed.values())
                },
            }
            assignments = ", ".join(
                "#f" + str(index) + " = :v" + str(index)
                for index in range(len(changed))
            )
            return {
                "TableName": self._table,
                "Key": encode_item(key),
                "ConditionExpression": (
                    "#state = :before_state AND "
                    "#revision = :before_revision AND "
                    + condition
                ),
                "UpdateExpression": "SET " + assignments,
                "ExpressionAttributeNames": names,
                "ExpressionAttributeValues": values,
            }

        pk = ledger_pk(RUN_ID)
        index_condition, index_names, index_values = exact_condition(
            values=index_before,
            fields=(
                "current_activation_id",
                "campaign_identity_sha256",
                "revision",
            ),
            prefix="i",
        )
        transact = {
            "TransactItems": [
                {
                    "ConditionCheck": {
                        "TableName": self._table,
                        "Key": encode_item(
                            {"PK": pk, "SK": "ACTIVATION_INDEX"}
                        ),
                        "ConditionExpression": index_condition,
                        "ExpressionAttributeNames": index_names,
                        "ExpressionAttributeValues": index_values,
                    }
                },
                {
                    "Update": update(
                        key={"PK": pk, "SK": request.action_key},
                        state_field="state",
                        before_state=str(before["state"]),
                        before_revision=int(before["revision"]),
                        changed=action_changed,
                        immutable_before=before,
                        immutable_fields=(
                            "campaign_identity_sha256",
                            "activation_id",
                            "activation_ordinal",
                            "generation",
                            "action_kind",
                            "attempt",
                            "candidate_key",
                            "candidate_file_sha256",
                            "candidate_body_sha256",
                            "request_body_sha256",
                            "relay_envelope_sha256",
                            "owner_epoch",
                            "owner_execution_arn",
                            "barrier_nonce_sha256",
                            "owner_invocation_nonce_sha256",
                            "consumed_at",
                        ),
                    )
                },
                {
                    "Update": update(
                        key={
                            "PK": pk,
                            "SK": (
                                "ACTIVATION#"
                                + self._activation
                                + "#CONTROL"
                            ),
                        },
                        state_field="last_sky_post_state",
                        before_state=str(control_before["last_sky_post_state"]),
                        before_revision=int(control_before["revision"]),
                        changed=control_changed,
                        immutable_before=control_before,
                        immutable_fields=(
                            "campaign_identity_sha256",
                            "activation_id",
                            "activation_ordinal",
                            "active_epoch",
                            "active_execution_arn",
                            "active_state_machine_version_arn",
                            "phase",
                            "fence_head_body_sha256",
                            "fence_head_version_id",
                            "barrier_nonce_sha256",
                            "barrier_state",
                            "decision_seal_state",
                        ),
                    )
                },
            ],
            "ClientRequestToken": canonical_sha256(
                {
                    "domain": domain,
                    "request": _safe_admission_request(request),
                    "action_after": after,
                    "control_after": control_after,
                }
            )[:36],
            "ReturnConsumedCapacity": "NONE",
        }
        try:
            response = self._client.transact_write_items(**transact)
        except Exception:
            response = None
        if type(response) is dict:
            metadata = response.get("ResponseMetadata")
            if (
                set(response) != {"ResponseMetadata"}
                or type(metadata) is not dict
                or type(metadata.get("HTTPStatusCode")) is not int
                or metadata["HTTPStatusCode"] != 200
                or type(metadata.get("RequestId")) is not str
                or not metadata["RequestId"]
            ):
                response = None
        else:
            response = None
        observed_index, observed_control, observed_action = self._read(request)
        if (
            observed_index != index_before
            or observed_control != control_after
            or observed_action != after
        ):
            raise RuntimeError(
                "launch-admission transaction did not reconcile exactly"
            )
        self.states.append(str(after["state"]))
        return str(after["state"])

    @staticmethod
    def _control_after(
        control: Mapping[str, object],
        *,
        action_key: str,
        state: str,
        now: str,
    ) -> Mapping[str, object]:
        return {
            **control,
            "last_sky_post_action_key": action_key,
            "last_sky_post_state": state,
            "revision": control["revision"] + 1,
            "updated_at": now,
        }

    def own_post_started(
        self,
        *,
        request: AdmissionRequest,
        private_nonce_sha256: str,
    ) -> str:
        index, control, action = self._read(request)
        now = _utc(_now())
        after = {
            **action,
            "state": "POST_STARTED",
            "post_owner_invocation_nonce_sha256": private_nonce_sha256,
            "post_owner_function_version_arn": self._version_arn,
            "post_owner_dispatch_identity_sha256": (
                build_admission_custody_identity(request)
            ),
            "post_owner_hard_expires_at": _utc(
                _now() + timedelta(seconds=25)
            ),
            "post_started_at": now,
            "revision": action["revision"] + 1,
        }
        return self._transition(
            request=request,
            index_before=index,
            before=action,
            after=after,
            control_before=control,
            control_after=self._control_after(
                control,
                action_key=request.action_key,
                state="POST_STARTED",
                now=now,
            ),
            domain="TASK11_POST_STARTED",
        )

    def authorize_post(
        self,
        *,
        request: AdmissionRequest,
        audit: Mapping[str, object],
        authority: Mapping[str, object],
        relay_envelope_sha256: str,
    ) -> str:
        del authority
        index, control, action = self._read(request)
        if (
            action["state"] != "POST_STARTED"
            or audit["closing_revision"] != control["revision"]
        ):
            raise RuntimeError("launch-admission authorization is stale")
        now = _utc(_now())
        after = {
            **action,
            "state": "POST_AUTHORIZED",
            "relay_envelope_sha256": relay_envelope_sha256,
            "authority_audit_body_sha256": audit[
                "audit_identity_sha256"
            ],
            "authority_audit_closing_revision": audit["closing_revision"],
            "authorized_transition_from_revision": audit[
                "closing_revision"
            ],
            "authorized_transition_to_revision": (
                audit["closing_revision"] + 1
            ),
            "post_authorized_at": now,
            "revision": action["revision"] + 1,
        }
        return self._transition(
            request=request,
            index_before=index,
            before=action,
            after=after,
            control_before=control,
            control_after=self._control_after(
                control,
                action_key=request.action_key,
                state="POST_AUTHORIZED",
                now=now,
            ),
            domain="TASK11_POST_AUTHORIZED",
        )

    def classify(
        self,
        *,
        request: AdmissionRequest,
        classification: str,
        response_identity_sha256: Optional[str],
        request_id: Optional[str],
    ) -> str:
        index, control, action = self._read(request)
        if action["state"] != "POST_AUTHORIZED":
            raise RuntimeError("launch-admission classification is stale")
        now = _utc(_now())
        evidence = {
            "classification": classification,
            "response_identity_sha256": response_identity_sha256,
            "request_id": request_id,
            "relay_body_sha256": hashlib.sha256(
                canonical_json_bytes(
                    {
                        "request": _safe_admission_request(request),
                        "classification": classification,
                    }
                )
            ).hexdigest(),
            "observed_at": now,
        }
        evidence_identity = canonical_sha256(evidence)
        durable_response_identity = (
            response_identity_sha256 or evidence_identity
        )
        after = {
            **action,
            "state": "POST_CLASSIFIED",
            "completed_at": now,
            "outcome_class": classification,
            "classification_evidence_kind": "RELAY_RESPONSE",
            "classification_evidence_body_sha256": evidence_identity,
            "sky_request_id": request_id,
            "response_identity_sha256": durable_response_identity,
            "revision": action["revision"] + 1,
        }
        return self._transition(
            request=request,
            index_before=index,
            before=action,
            after=after,
            control_before=control,
            control_after=self._control_after(
                control,
                action_key=request.action_key,
                state="POST_CLASSIFIED",
                now=now,
            ),
            domain="TASK11_POST_CLASSIFIED",
        )

    def evidence(self) -> Mapping[str, object]:
        return {
            "post_audit": self.audit_result,
            "action_states": tuple(self.states),
        }


class AdmissionFreshAudit:
    def __init__(
        self,
        *,
        store: AdmissionDynamoStore,
        s3: object,
        service: FreshH1fAuditService,
    ) -> None:
        self._store = store
        self._s3 = s3
        self._service = service

    def inspect(self, request: AdmissionRequest) -> Mapping[str, object]:
        _, _, action = self._store._read(request)
        bucket = os.environ["GLM52_CAMPAIGN_BUCKET"]
        key = action["candidate_key"]
        versions = []
        deletes = []
        markers: dict[str, object] = {}
        seen_markers = set()
        for _ in range(64):
            page = self._s3.list_object_versions(
                Bucket=bucket,
                Prefix=key,
                MaxKeys=1000,
                ExpectedBucketOwner=ACCOUNT_ID,
                **markers,
            )
            page_metadata = page.get("ResponseMetadata")
            if (
                type(page_metadata) is not dict
                or page_metadata.get("HTTPStatusCode") != 200
                or type(page_metadata.get("RequestId")) is not str
                or not page_metadata["RequestId"]
                or type(page.get("Versions", [])) is not list
                or type(page.get("DeleteMarkers", [])) is not list
                or type(page.get("IsTruncated")) is not bool
            ):
                raise RuntimeError(
                    "launch-admission candidate inventory drifted"
                )
            versions.extend(
                item
                for item in page.get("Versions", [])
                if type(item) is dict and item.get("Key") == key
            )
            deletes.extend(
                item
                for item in page.get("DeleteMarkers", [])
                if type(item) is dict and item.get("Key") == key
            )
            if page["IsTruncated"] is False:
                break
            next_key = page.get("NextKeyMarker")
            next_version = page.get("NextVersionIdMarker")
            marker_identity = (next_key, next_version)
            if (
                type(next_key) is not str
                or not next_key
                or type(next_version) is not str
                or not next_version
                or marker_identity in seen_markers
            ):
                raise RuntimeError(
                    "launch-admission candidate pagination drifted"
                )
            seen_markers.add(marker_identity)
            markers = {
                "KeyMarker": next_key,
                "VersionIdMarker": next_version,
            }
        else:
            raise RuntimeError(
                "launch-admission candidate pagination exceeded bound"
            )
        if (
            len(versions) != 1
            or deletes
            or versions[0].get("IsLatest") is not True
        ):
            raise RuntimeError(
                "launch-admission candidate inventory drifted"
            )
        version_id = versions[0].get("VersionId")
        if type(version_id) is not str or not version_id:
            raise RuntimeError(
                "launch-admission candidate version is absent"
            )
        response = self._s3.get_object(
            Bucket=bucket,
            Key=key,
            VersionId=version_id,
            ExpectedBucketOwner=ACCOUNT_ID,
            ChecksumMode="ENABLED",
        )
        stream = response.get("Body")
        read = getattr(stream, "read", None)
        metadata = response.get("ResponseMetadata")
        if (
            type(metadata) is not dict
            or metadata.get("HTTPStatusCode") != 200
            or type(metadata.get("RequestId")) is not str
            or not metadata["RequestId"]
            or response.get("VersionId") != version_id
            or not callable(read)
            or response.get("ChecksumSHA256")
            != base64.b64encode(
                bytes.fromhex(action["candidate_file_sha256"])
            ).decode("ascii")
        ):
            raise RuntimeError("launch-admission candidate read failed")
        raw = read(_MAX_BODY + 1)
        if (
            type(raw) is not bytes
            or len(raw) > _MAX_BODY
            or versions[0].get("Size") != len(raw)
            or hashlib.sha256(raw).hexdigest()
            != action["candidate_file_sha256"]
        ):
            raise RuntimeError(
                "launch-admission candidate bytes drifted"
            )
        candidate = build_immutable_json_candidate(
            record_kind="generation-start-decision",
            bucket=bucket,
            key=key,
            raw=raw,
            activation_id=request.activation_id,
            generation=request.generation,
        )
        audit = self._service.fresh_audit(
            s3=self._s3,
            request=H1fAuditRequest(
                operation_kind="SKY_POST",
                action_key=request.action_key,
                candidate=candidate,
            ),
        )
        self._store.audit_result = audit
        return {
            "fresh": True,
            "audit_identity_sha256": audit.canonical_body_sha256,
            "closing_revision": audit.closing_revision,
        }


class AdmissionCurrentAuthority:
    def __init__(
        self,
        *,
        store: AdmissionDynamoStore,
        decision_nonce_sha256: str,
        live_h1d_identity_sha256: str,
    ) -> None:
        self._store = store
        self._decision_nonce = decision_nonce_sha256
        self._live_h1d = live_h1d_identity_sha256

    def inspect(self, request: AdmissionRequest) -> Mapping[str, object]:
        index, control, action = self._store._read(request)
        consumed_fields = (
            action.get("owner_invocation_nonce_sha256"),
            action.get("post_owner_invocation_nonce_sha256"),
            action.get("candidate_file_sha256"),
            action.get("candidate_body_sha256"),
        )
        if (
            request.decision_nonce_sha256 != self._decision_nonce
            or request.live_h1d_identity_sha256 != self._live_h1d
            or action["state"] != "POST_STARTED"
            or action["post_owner_dispatch_identity_sha256"]
            != build_admission_custody_identity(request)
            or action.get("candidate_body_sha256")
            != request.direct_decision_identity_sha256
            or any(
                type(value) is not str
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
                for value in consumed_fields
            )
            or type(action.get("candidate_key")) is not str
            or not action["candidate_key"]
            or type(action.get("consumed_at")) is not str
            or not action["consumed_at"]
            or type(action.get("post_started_at")) is not str
            or not action["post_started_at"]
        ):
            raise RuntimeError("launch-admission current owner drifted")
        consumed_reservation = {
            "record_type": "glm52_task11_consumed_admission_reservation_v1",
            "action_key": request.action_key,
            "state_history": ("CONSUMED", "POST_STARTED"),
            "current_action_state": action["state"],
            "decision_owner_nonce_sha256": action[
                "owner_invocation_nonce_sha256"
            ],
            "post_owner_nonce_sha256": action[
                "post_owner_invocation_nonce_sha256"
            ],
            "consumed_at": action["consumed_at"],
            "post_started_at": action["post_started_at"],
            "candidate_key": action["candidate_key"],
            "candidate_file_sha256": action["candidate_file_sha256"],
            "candidate_body_sha256": action["candidate_body_sha256"],
            "direct_decision_identity_sha256": (
                request.direct_decision_identity_sha256
            ),
            "decision_nonce_sha256": request.decision_nonce_sha256,
            "control_revision": control["revision"],
            "activation_index_revision": index["revision"],
            "campaign_identity_sha256": action[
                "campaign_identity_sha256"
            ],
            "activation_id": request.activation_id,
            "generation": request.generation,
            "barrier_nonce_sha256": action["barrier_nonce_sha256"],
        }
        return {
            "current_owner": True,
            "reserve_held": True,
            "action_consumed": True,
            "live_h1d_identity_sha256": self._live_h1d,
            "reserve_identity_sha256": canonical_sha256(
                consumed_reservation
            ),
            "owner_identity_sha256": (
                build_admission_custody_identity(request)
            ),
        }
