"""Task 11 exact assumed closure-session production boundary."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib

import pytest

from glm52_enforcement.task11_production import (
    Task11ProductionConfig,
    _aws_clients,
)


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
ACTIVATION_ID = "approved-20260728"
REQUEST_ID = "11111111-2222-4333-8444-555555555555"
DECISION_VERSION_ARN = (
    "arn:aws:lambda:us-west-2:246813579024:function:"
    "keep-glm52-h1g-decision:11"
)
ACTIVATION_SHA = hashlib.sha256(ACTIVATION_ID.encode("ascii")).hexdigest()[:16]
CLOSURE_ROLE_NAME = (
    "keep-glm52-h1g-closure-session-" + ACTIVATION_SHA
)
CLOSURE_ROLE_ARN = (
    "arn:aws:iam::246813579024:role/" + CLOSURE_ROLE_NAME
)
DEPLOYMENT_IDENTITY = hashlib.sha256(b"support-deployment").hexdigest()
NOW = datetime(2026, 7, 29, 16, 0, 0, tzinfo=timezone.utc)


def _config() -> Task11ProductionConfig:
    prefix = (
        "arn:aws:lambda:us-west-2:246813579024:function:"
        "keep-glm52-h1g-"
    )
    return Task11ProductionConfig(
        account_id=ACCOUNT_ID,
        region=REGION,
        run_id="glm52-sky-20260724",
        activation_id=ACTIVATION_ID,
        ledger_table_name="keep-glm52-h1g-ledger-v1",
        campaign_bucket="keep-glm52-model-evidence",
        model_bucket="keep-glm52-model-evidence",
        model_prefix="campaigns/glm52-sky-20260724/",
        fence_stack_id="keep-glm52-gpu-fence",
        support_stack_id="keep-glm52-h1g-support",
        closure_role_arn=CLOSURE_ROLE_ARN,
        attestation_version_arn=prefix + "attestation:3",
        launch_admission_version_arn=prefix + "launch-admission:7",
        numeric_binding_version_arn=prefix + "numeric-binding:4",
        source_gpu_spend_version_arn=prefix + "source-gpu-spend:2",
        source_submission_intent_version_arn=(
            prefix + "source-submission-intent:2"
        ),
        source_controller_baseline_version_arn=(
            prefix + "source-controller-baseline:2"
        ),
        source_control_plane_readiness_version_arn=(
            prefix + "source-control-plane-readiness:2"
        ),
        source_submission_acquisition_version_arn=(
            prefix + "source-submission-acquisition:2"
        ),
        fence_executor_version_arn=prefix + "fence-executor:5",
        fence_successor_version_arn=prefix + "fence-successor:5",
        claim_writer_version_arn=prefix + "claim-writer:5",
        decision_writer_version_arn=prefix + "decision-writer:5",
        terminal_v1_writer_version_arn=prefix + "terminal-v1-writer:5",
        closure_handoff_version_arn=prefix + "closure-handoff:5",
        deployment_identity_sha256=DEPLOYMENT_IDENTITY,
        decision_function_version_arn=DECISION_VERSION_ARN,
    )


class LambdaContext:
    aws_request_id = REQUEST_ID
    invoked_function_arn = DECISION_VERSION_ARN

    def __init__(self, remaining_millis: int = 839_000) -> None:
        self.remaining_millis = remaining_millis

    def get_remaining_time_in_millis(self) -> int:
        return self.remaining_millis


def _session_name() -> str:
    return (
        "h1g-decision-"
        + ACTIVATION_SHA
        + "-"
        + hashlib.sha256(REQUEST_ID.encode("ascii")).hexdigest()[:16]
    )


def _metadata(request_id: str) -> dict[str, object]:
    return {
        "HTTPStatusCode": 200,
        "RequestId": request_id,
        "RetryAttempts": 0,
    }


class BaseSts:
    def __init__(self, *, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []
        self.identity_calls = 0

    def assume_role(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return self.response

    def get_caller_identity(self) -> object:
        self.identity_calls += 1
        return {
            "Account": ACCOUNT_ID,
            "Arn": (
                "arn:aws:sts::246813579024:assumed-role/"
                "keep-glm52-h1g-support-decision/lambda-session"
            ),
            "UserId": "AROADECISIONROLE123:lambda-session",
            "ResponseMetadata": _metadata("source-identity-request"),
        }


class TargetSts:
    def __init__(self, *, response: object) -> None:
        self.response = response
        self.calls = 0

    def get_caller_identity(self) -> object:
        self.calls += 1
        return self.response


class DataClient:
    def __init__(self, service: str) -> None:
        self.service = service


class BaseSession:
    def __init__(self, sts: BaseSts) -> None:
        self.sts = sts
        self.client_calls: list[str] = []

    def get_credentials(self) -> object:
        raise AssertionError("private EnvProvider credentials are forbidden")

    def client(self, service: str, **kwargs: object) -> object:
        del kwargs
        self.client_calls.append(service)
        if service != "sts":
            raise AssertionError("base session created a data client")
        return self.sts


class AssumedSession:
    def __init__(self, target_sts: TargetSts) -> None:
        self.target_sts = target_sts
        self.client_calls: list[str] = []
        self.clients: dict[str, object] = {}

    def client(self, service: str, **kwargs: object) -> object:
        del kwargs
        self.client_calls.append(service)
        if service == "sts":
            return self.target_sts
        client = DataClient(service)
        self.clients[service] = client
        return client


class SessionFactory:
    def __init__(
        self,
        *,
        assume_response: object | None = None,
        caller_response: object | None = None,
    ) -> None:
        name = _session_name()
        assumed_role_id = "AROATESTCLOSURE:" + name
        self.assume_response = (
            {
                "Credentials": {
                    "AccessKeyId": "ASIATESTCLOSURE",
                    "SecretAccessKey": "secret",
                    "SessionToken": "token",
                    "Expiration": NOW + timedelta(seconds=900),
                },
                "AssumedRoleUser": {
                    "AssumedRoleId": assumed_role_id,
                    "Arn": (
                        "arn:aws:sts::246813579024:assumed-role/"
                        + CLOSURE_ROLE_NAME
                        + "/"
                        + name
                    ),
                },
                "PackedPolicySize": 0,
                "ResponseMetadata": _metadata("assume-request"),
            }
            if assume_response is None
            else assume_response
        )
        self.caller_response = (
            {
                "Account": ACCOUNT_ID,
                "Arn": (
                    "arn:aws:sts::246813579024:assumed-role/"
                    + CLOSURE_ROLE_NAME
                    + "/"
                    + name
                ),
                "UserId": assumed_role_id,
                "ResponseMetadata": _metadata("identity-request"),
            }
            if caller_response is None
            else caller_response
        )
        self.base_sts = BaseSts(response=self.assume_response)
        self.target_sts = TargetSts(response=self.caller_response)
        self.base_session = BaseSession(self.base_sts)
        self.assumed_session = AssumedSession(self.target_sts)
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        if len(self.calls) == 1:
            return self.base_session
        if len(self.calls) == 2:
            return self.assumed_session
        raise AssertionError("a third SDK session was constructed")


def _clients(
    factory: SessionFactory,
    *,
    config: Task11ProductionConfig | None = None,
    context: object | None = None,
):
    return _aws_clients(
        config=_config() if config is None else config,
        context=LambdaContext() if context is None else context,
        session_factory=factory,
        utc_clock=lambda: NOW,
    )


def test_task11_closure_session_uses_one_authenticated_assume_role() -> None:
    """Break caught: data clients use EnvProvider credentials or STS is repeated."""

    factory = SessionFactory()
    clients = _clients(factory)

    assert factory.base_sts.calls == [
        {
            "RoleArn": CLOSURE_ROLE_ARN,
            "RoleSessionName": _session_name(),
            "DurationSeconds": 900,
            "ExternalId": DEPLOYMENT_IDENTITY,
        }
    ]
    assert factory.base_session.client_calls == ["sts"]
    assert factory.target_sts.calls == 1
    assert factory.calls == [
        {"region_name": REGION},
        {
            "aws_access_key_id": "ASIATESTCLOSURE",
            "aws_secret_access_key": "secret",
            "aws_session_token": "token",
            "region_name": REGION,
        },
    ]
    assert clients.sts is factory.target_sts
    assert clients.s3 is factory.assumed_session.clients["s3"]
    assert clients.dynamodb is factory.assumed_session.clients["dynamodb"]
    assert clients.credential_expiration == "2026-07-29T16:15:00Z"
    assert clients.caller_identity == factory.caller_response
    assert clients.credential_issue_time == "2026-07-29T16:00:00Z"
    assert clients.assume_role_request_id == "assume-request"
    assert clients.source_caller_identity["Arn"].endswith(
        "keep-glm52-h1g-support-decision/lambda-session"
    )
    assert factory.base_sts.identity_calls == 1


@pytest.mark.parametrize(
    ("context", "match"),
    (
        (LambdaContext(0), "remaining time"),
        (LambdaContext(840_001), "remaining time"),
        (object(), "request ID"),
    ),
)
def test_task11_closure_session_rejects_invalid_lambda_context(
    context: object,
    match: str,
) -> None:
    """Break caught: a fake or expired invocation can mint closure credentials."""

    with pytest.raises(RuntimeError, match=match):
        _clients(SessionFactory(), context=context)


def test_task11_closure_session_rejects_wrong_invoked_version() -> None:
    """Break caught: an alias or another Decision version can assume the role."""

    context = LambdaContext()
    context.invoked_function_arn = DECISION_VERSION_ARN.rsplit(":", 1)[0]
    with pytest.raises(RuntimeError, match="invoked function"):
        _clients(SessionFactory(), context=context)


def test_task11_closure_session_rejects_non_uuid_request_id() -> None:
    """Break caught: a caller-selected session suffix is accepted."""

    context = LambdaContext()
    context.aws_request_id = "not-a-request-id"
    with pytest.raises(RuntimeError, match="request ID"):
        _clients(SessionFactory(), context=context)


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (
            lambda value: {
                **value,
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "",
                    "RetryAttempts": 0,
                },
            },
            "AssumeRole response",
        ),
        (
            lambda value: {
                **value,
                "Credentials": {
                    **value["Credentials"],
                    "Expiration": NOW + timedelta(seconds=700),
                },
            },
            "expiration",
        ),
        (
            lambda value: {
                **value,
                "Credentials": {
                    **value["Credentials"],
                    "Unexpected": "field",
                },
            },
            "credential",
        ),
        (
            lambda value: {
                **value,
                "AssumedRoleUser": {
                    **value["AssumedRoleUser"],
                    "Arn": value["AssumedRoleUser"]["Arn"].replace(
                        "closure-session",
                        "foreign-role",
                    ),
                },
            },
            "assumed role",
        ),
        (
            lambda value: {
                **value,
                "PackedPolicySize": 1,
            },
            "policy",
        ),
    ),
)
def test_task11_closure_session_rejects_malformed_assume_role(
    mutate,
    match: str,
) -> None:
    """Break caught: unauthenticated or policy-bearing STS output is trusted."""

    baseline = SessionFactory().assume_response
    factory = SessionFactory(assume_response=mutate(baseline))
    with pytest.raises(RuntimeError, match=match):
        _clients(factory)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ({"Account": "111122223333"}, "caller identity"),
        (
            {
                "Arn": (
                    "arn:aws:sts::246813579024:assumed-role/"
                    + CLOSURE_ROLE_NAME
                    + "/other-session"
                )
            },
            "caller identity",
        ),
        ({"UserId": "AROATESTCLOSURE:other-session"}, "caller identity"),
        (
            {
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "",
                    "RetryAttempts": 0,
                }
            },
            "caller identity",
        ),
    ),
)
def test_task11_closure_session_rejects_wrong_target_identity(
    mutation: dict[str, object],
    match: str,
) -> None:
    """Break caught: an alternate assumed role/session/account reaches closure."""

    baseline = dict(SessionFactory().caller_response)
    baseline.update(mutation)
    with pytest.raises(RuntimeError, match=match):
        _clients(SessionFactory(caller_response=baseline))


def test_task11_closure_session_requires_exact_configured_role() -> None:
    """Break caught: an alternate target role is selected at runtime."""

    config = replace(
        _config(),
        closure_role_arn=(
            "arn:aws:iam::246813579024:role/keep-glm52-other-role"
        ),
    )
    with pytest.raises(RuntimeError, match="closure role"):
        _clients(SessionFactory(), config=config)
