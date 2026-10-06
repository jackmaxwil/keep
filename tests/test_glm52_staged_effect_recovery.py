from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace

import pytest

from glm52_enforcement import staged_effect_recovery as recovery
from glm52_enforcement.task13_production_operations import ProductionServices
from glm52_enforcement.task13_staged_deployment import DeploymentStep

_MUTATING_STEPS = frozenset(
    {
        DeploymentStep.RETAINED_BOOTSTRAP_RUNTIME_DEPLOYED,
        DeploymentStep.BRIDGE_SEED_PUBLISHED,
        DeploymentStep.BRIDGE_SEED_ESTABLISHED,
        DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6,
        DeploymentStep.BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED,
        DeploymentStep.RETAINED_FENCE_RUNTIME_DEPLOYED,
        DeploymentStep.PREPARE_EXECUTED_STABILIZED,
        DeploymentStep.DISABLED_SUPPORT_DEPLOYED,
        DeploymentStep.STACK_MIGRATION_OPERATION_7,
        DeploymentStep.SUPPORT_RUNTIME_IDENTITY_COMMITTED,
    }
)


@dataclass(frozen=True)
class _Evidence:
    projection: Mapping[str, object]


class _MutationSpy:
    _FORBIDDEN = frozenset(
        {
            "put_item",
            "update_item",
            "transact_write_items",
            "put_object",
            "create_stack",
            "update_stack",
            "create_change_set",
            "execute_change_set",
            "delete_stack",
            "update_termination_protection",
            "publish_version",
            "create_function",
            "update_function_code",
            "update_function_configuration",
            "run_instances",
            "start_instances",
            "request_spot_instances",
            "purchase_capacity_block",
        }
    )

    def __init__(self) -> None:
        self.forbidden_calls: list[str] = []

    def __getattr__(self, name: str):
        if name not in self._FORBIDDEN:
            raise AttributeError(name)

        def forbidden(**_kwargs: object) -> object:
            self.forbidden_calls.append(name)
            raise AssertionError("mutation method was invoked: " + name)

        return forbidden


def _services(spy: _MutationSpy) -> ProductionServices:
    return ProductionServices(
        sts=spy,
        cloudformation=spy,
        iam=spy,
        s3=spy,
        organizations=spy,
        ec2=spy,
        ssm=spy,
        kms=spy,
        dynamodb=spy,
        lambda_client=spy,
        states=spy,
        cloudtrail=spy,
        total_max_attempts=1,
    )


def _request() -> dict[str, object]:
    return {
        "schema_version": 2,
        "record_type": "glm52_task13_production_operations_v2",
        "activation_id": "glm52-v2-amber-quartz",
    }


def _prior(step: DeploymentStep) -> dict[DeploymentStep, object]:
    return {
        prior: _Evidence({"step": prior.value, "effect": "committed"})
        for prior in tuple(DeploymentStep)[: tuple(DeploymentStep).index(step)]
    }


def _parser(value: object) -> _Evidence:
    if type(value) is not dict or set(value) != {"step", "effect"}:
        raise recovery.StagedEffectRecoveryError("stored projection is not exact")
    return _Evidence(dict(value))


def _binding_table(
    *,
    readers: Mapping[DeploymentStep, Callable[..., object]],
) -> Mapping[DeploymentStep, recovery.RecoveryStepBinding]:
    return MappingProxyType(
        {
            step: recovery.RecoveryStepBinding(
                parser=_parser,
                live_reader=readers[step],
                mutating=step in _MUTATING_STEPS,
            )
            for step in DeploymentStep
        }
    )


def _install_fake_table(
    monkeypatch: pytest.MonkeyPatch,
    *,
    live_projections: Mapping[DeploymentStep, Mapping[str, object]],
    calls: list[DeploymentStep],
) -> None:
    def reader_for(step: DeploymentStep):
        def read_effect(**_kwargs: object) -> _Evidence:
            calls.append(step)
            return _Evidence(dict(live_projections[step]))

        return read_effect

    monkeypatch.setattr(
        recovery,
        "_RECOVERY_STEP_BINDINGS",
        _binding_table(readers={step: reader_for(step) for step in DeploymentStep}),
    )
    monkeypatch.setattr(recovery, "_guard_request", lambda value: dict(value))

    def validated_projection(
        step: DeploymentStep,
        evidence: object,
        committed: Mapping[DeploymentStep, object],
    ) -> Mapping[str, object]:
        assert (
            tuple(committed)
            == tuple(DeploymentStep)[: tuple(DeploymentStep).index(step)]
        )
        if type(evidence) is not _Evidence:
            raise recovery.StagedEffectRecoveryError("evidence type is foreign")
        if evidence.projection.get("step") != step.value:
            raise recovery.StagedEffectRecoveryError("live effect is foreign")
        return dict(evidence.projection)

    monkeypatch.setattr(recovery, "_validated_projection", validated_projection)


def test_recovery_table_is_closed_over_all_thirteen_steps() -> None:
    table = recovery._RECOVERY_STEP_BINDINGS
    assert type(table) is MappingProxyType
    assert tuple(table) == tuple(DeploymentStep)
    assert {
        step for step, binding in table.items() if binding.mutating
    } == _MUTATING_STEPS
    assert all(callable(binding.parser) for binding in table.values())
    assert all(callable(binding.live_reader) for binding in table.values())


def test_disabled_effect_reader_uses_committed_checkpoint_and_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from glm52_enforcement import support_runtime_transition

    checkpoint = object()
    snapshot = object()
    captured: dict[str, object] = {}
    expected = object()

    def read_disabled(**kwargs: object) -> object:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        support_runtime_transition,
        "read_disabled_support_deployment_v2",
        read_disabled,
    )
    result = recovery._read_disabled_support(
        request={"disabled_support_deployment": {"request": "exact"}},
        committed={
            DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6: checkpoint,
            DeploymentStep.SUPPORT_INPUT_SNAPSHOT_TWO: snapshot,
        },
        services=_services(_MutationSpy()),
    )

    assert result is expected
    assert captured["request"] == {"request": "exact"}
    assert captured["checkpoint"] is checkpoint
    assert captured["support_inputs"] is snapshot


def test_two_phase_bootstrap_readers_use_only_bounded_live_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from glm52_enforcement import (
        fence_bootstrap_publication,
        support_plane,
        task13_production_operations,
    )

    checkpoint = SimpleNamespace(fence_stack_id="fence-stack")
    publication = SimpleNamespace(
        manifest_coordinate=SimpleNamespace(
            to_dict=lambda: {"manifest": "coordinate"}
        )
    )
    bootstrap_runtime = {
        "materializer_function_version_arn": "materializer-version"
    }
    expected_bootstrap = object()
    expected_seed = object()
    expected_publication = object()
    expected_runtime = object()
    captured: dict[str, dict[str, object]] = {}

    def read_bootstrap(**kwargs: object) -> object:
        captured["bootstrap"] = dict(kwargs)
        return expected_bootstrap

    def read_seed(**kwargs: object) -> object:
        captured["seed"] = dict(kwargs)
        return expected_seed

    def read_runtime(**kwargs: object) -> object:
        captured["runtime"] = dict(kwargs)
        return expected_runtime

    def read_publication(**kwargs: object) -> object:
        captured["publication"] = dict(kwargs)
        return expected_publication

    monkeypatch.setattr(
        task13_production_operations,
        "read_retained_bootstrap_runtime_deployment_v2",
        read_bootstrap,
        raising=False,
    )
    monkeypatch.setattr(
        fence_bootstrap_publication,
        "read_bridge_seed_publication_v2",
        read_seed,
    )
    monkeypatch.setattr(
        fence_bootstrap_publication,
        "read_bootstrap_fence_publication_v2",
        read_publication,
    )
    monkeypatch.setattr(
        task13_production_operations,
        "read_retained_fence_runtime_deployment_v2",
        read_runtime,
        raising=False,
    )
    monkeypatch.setattr(
        fence_bootstrap_publication,
        "materialize_bridge_seed_publication_request_v2",
        lambda **kwargs: {
            "seed": kwargs["authority"],
            "version": kwargs["materializer_function_version_arn"],
        },
    )
    monkeypatch.setattr(
        fence_bootstrap_publication,
        "materialize_bootstrap_publication_request_v2",
        lambda **kwargs: {
            "publication": kwargs["authority"],
            "version": kwargs["materializer_function_version_arn"],
            "checkpoint": kwargs["checkpoint"],
        },
    )
    monkeypatch.setattr(
        support_plane,
        "materialize_retained_fence_runtime_inputs",
        lambda **kwargs: {
            "runtime": kwargs["authority"],
            "fence_stack_id": kwargs["fence_stack_id"],
            "manifest": kwargs["bootstrap_manifest_coordinate"],
        },
    )
    services = _services(_MutationSpy())
    request = {
        "retained_bootstrap_runtime_deployment": {"runtime": "bootstrap"},
        "bridge_seed_publication": {"seed": "request"},
        "retained_fence_runtime_deployment": {"runtime": "fence"},
    }

    assert (
        recovery._read_retained_bootstrap_runtime(
            request=request,
            committed={},
            services=services,
        )
        is expected_bootstrap
    )
    assert (
        recovery._read_bridge_seed_publication(
            request=request,
            committed={
                DeploymentStep.RETAINED_BOOTSTRAP_RUNTIME_DEPLOYED: (
                    bootstrap_runtime
                )
            },
            services=services,
        )
        is expected_seed
    )
    request["bootstrap_fence_publication"] = {"publication": "authority"}
    assert (
        recovery._read_bootstrap_publication(
            request=request,
            committed={
                DeploymentStep.RETAINED_BOOTSTRAP_RUNTIME_DEPLOYED: (
                    bootstrap_runtime
                ),
                DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6: checkpoint,
            },
            services=services,
        )
        is expected_publication
    )
    assert (
        recovery._read_retained_fence_runtime(
            request=request,
            committed={
                DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6: checkpoint,
                DeploymentStep.BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED: publication,
            },
            services=services,
        )
        is expected_runtime
    )
    assert captured["bootstrap"]["request"] == {"runtime": "bootstrap"}
    assert captured["bootstrap"]["services"] is services
    assert captured["seed"]["request"] == {
        "seed": {"seed": "request"},
        "version": "materializer-version",
    }
    assert captured["publication"]["request"] == {
        "publication": {"publication": "authority"},
        "version": "materializer-version",
        "checkpoint": checkpoint,
    }
    assert captured["runtime"]["request"] == {
        "runtime": {"runtime": "fence"},
        "fence_stack_id": "fence-stack",
        "manifest": {"manifest": "coordinate"},
    }
    assert captured["runtime"]["checkpoint"] is checkpoint
    assert captured["runtime"]["publication"] is publication


def test_bridge_seed_recovery_binds_the_committed_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from glm52_enforcement import (
        fence_bootstrap_publication,
        staged_migration_operations,
    )

    artifact = object()
    publication = SimpleNamespace(artifact=artifact)
    expected = object()
    captured: dict[str, object] = {}

    def materialize(**kwargs: object) -> dict[str, object]:
        captured["prototype"] = kwargs["prototype"]
        captured["artifact"] = kwargs["bridge_seed"]
        return {"request": "publication-bound"}

    def read_seed(**kwargs: object) -> object:
        captured["request"] = kwargs["request"]
        captured["services"] = kwargs["services"]
        return expected

    monkeypatch.setattr(
        staged_migration_operations,
        "materialize_bridge_seed_establishment_request_v2",
        materialize,
    )
    monkeypatch.setattr(
        staged_migration_operations,
        "read_bridge_seed_v2",
        read_seed,
    )
    monkeypatch.setattr(
        fence_bootstrap_publication,
        "BridgeSeedPublicationV2",
        SimpleNamespace,
    )
    services = _services(_MutationSpy())

    assert (
        recovery._read_bridge_seed(
            request={"bridge_seed": {"request": "prototype"}},
            committed={DeploymentStep.BRIDGE_SEED_PUBLISHED: publication},
            services=services,
        )
        is expected
    )
    assert captured == {
        "prototype": {"request": "prototype"},
        "artifact": artifact,
        "request": {"request": "publication-bound"},
        "services": services,
    }


@pytest.mark.parametrize("step", tuple(_MUTATING_STEPS), ids=lambda step: step.value)
def test_possibly_sent_present_effect_is_read_without_any_send(
    monkeypatch: pytest.MonkeyPatch,
    step: DeploymentStep,
) -> None:
    spy = _MutationSpy()
    calls: list[DeploymentStep] = []
    projections = {
        item: {"step": item.value, "effect": "committed"} for item in DeploymentStep
    }
    _install_fake_table(
        monkeypatch,
        live_projections=projections,
        calls=calls,
    )
    request = _request()
    committed = _prior(step)
    seal = {
        "operation_identity_sha256": recovery._expected_operation_identity_sha256(
            step=step,
            request=request,
            committed=committed,
        )
    }

    result = recovery.reconcile_staged_mutation_v2(
        step=step,
        request=request,
        committed=committed,
        possible_send_evidence=seal,
        services=_services(spy),
    )

    assert type(result) is _Evidence
    assert result.projection == projections[step]
    assert calls == [step]
    assert spy.forbidden_calls == []


@pytest.mark.parametrize("step", tuple(_MUTATING_STEPS), ids=lambda step: step.value)
@pytest.mark.parametrize("reason", ("absent", "duplicated"))
def test_possibly_sent_missing_or_duplicated_effect_fails_without_any_send(
    monkeypatch: pytest.MonkeyPatch,
    step: DeploymentStep,
    reason: str,
) -> None:
    spy = _MutationSpy()

    def unavailable(**_kwargs: object) -> object:
        raise recovery.StagedEffectRecoveryError("durable effect is " + reason)

    monkeypatch.setattr(
        recovery,
        "_RECOVERY_STEP_BINDINGS",
        _binding_table(readers={item: unavailable for item in DeploymentStep}),
    )
    monkeypatch.setattr(recovery, "_guard_request", lambda value: dict(value))
    monkeypatch.setattr(
        recovery,
        "_validated_projection",
        lambda _step, evidence, _committed: evidence.projection,
    )
    request = _request()
    committed = _prior(step)
    seal = {
        "operation_identity_sha256": recovery._expected_operation_identity_sha256(
            step=step,
            request=request,
            committed=committed,
        )
    }

    with pytest.raises(
        recovery.StagedEffectRecoveryError, match="durable effect is " + reason
    ):
        recovery.reconcile_staged_mutation_v2(
            step=step,
            request=request,
            committed=committed,
            possible_send_evidence=seal,
            services=_services(spy),
        )

    assert spy.forbidden_calls == []


@pytest.mark.parametrize("step", tuple(_MUTATING_STEPS), ids=lambda step: step.value)
def test_possibly_sent_foreign_effect_fails_without_any_send(
    monkeypatch: pytest.MonkeyPatch,
    step: DeploymentStep,
) -> None:
    spy = _MutationSpy()
    calls: list[DeploymentStep] = []
    projections = {
        item: {"step": item.value, "effect": "committed"} for item in DeploymentStep
    }
    projections[step] = {"step": "FOREIGN", "effect": "committed"}
    _install_fake_table(
        monkeypatch,
        live_projections=projections,
        calls=calls,
    )
    request = _request()
    committed = _prior(step)
    seal = {
        "operation_identity_sha256": recovery._expected_operation_identity_sha256(
            step=step,
            request=request,
            committed=committed,
        )
    }

    with pytest.raises(recovery.StagedEffectRecoveryError, match="foreign"):
        recovery.reconcile_staged_mutation_v2(
            step=step,
            request=request,
            committed=committed,
            possible_send_evidence=seal,
            services=_services(spy),
        )

    assert calls == [step]
    assert spy.forbidden_calls == []


def test_possibly_sent_seal_rejects_missing_unknown_and_foreign_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = DeploymentStep.BRIDGE_SEED_ESTABLISHED
    spy = _MutationSpy()
    calls: list[DeploymentStep] = []
    projections = {
        item: {"step": item.value, "effect": "committed"} for item in DeploymentStep
    }
    _install_fake_table(monkeypatch, live_projections=projections, calls=calls)
    arguments = {
        "step": step,
        "request": _request(),
        "committed": _prior(step),
        "services": _services(spy),
    }

    for seal in (
        {},
        {"operation_identity_sha256": "0" * 64, "unknown": True},
        {"operation_identity_sha256": "f" * 64},
    ):
        with pytest.raises(recovery.StagedEffectRecoveryError, match="seal"):
            recovery.reconcile_staged_mutation_v2(
                **arguments,
                possible_send_evidence=seal,
            )

    assert calls == []
    assert spy.forbidden_calls == []


@pytest.mark.parametrize("step", tuple(DeploymentStep), ids=lambda step: step.value)
def test_committed_adoption_reparses_and_fresh_reads_byte_identical_effect(
    monkeypatch: pytest.MonkeyPatch,
    step: DeploymentStep,
) -> None:
    spy = _MutationSpy()
    calls: list[DeploymentStep] = []
    projection = {"step": step.value, "effect": "committed"}
    projections = {
        item: (
            projection if item is step else {"step": item.value, "effect": "committed"}
        )
        for item in DeploymentStep
    }
    _install_fake_table(monkeypatch, live_projections=projections, calls=calls)

    result = recovery.adopt_staged_evidence_v2(
        step=step,
        request=_request(),
        evidence=dict(projection),
        committed=_prior(step),
        services=_services(spy),
    )

    assert type(result) is _Evidence
    assert result.projection == projection
    assert calls == [step]
    assert spy.forbidden_calls == []


def test_support_snapshots_are_recollected_independently_and_no_launch_is_reproved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[DeploymentStep] = []
    projections = {
        item: {"step": item.value, "effect": "committed"} for item in DeploymentStep
    }
    _install_fake_table(monkeypatch, live_projections=projections, calls=calls)
    spy = _MutationSpy()

    for step in (
        DeploymentStep.SUPPORT_INPUT_SNAPSHOT_ONE,
        DeploymentStep.SUPPORT_INPUT_SNAPSHOT_TWO,
        DeploymentStep.NO_WORKER_ACTIVATION_PROVED,
    ):
        recovery.adopt_staged_evidence_v2(
            step=step,
            request=_request(),
            evidence=dict(projections[step]),
            committed=_prior(step),
            services=_services(spy),
        )

    assert calls == [
        DeploymentStep.SUPPORT_INPUT_SNAPSHOT_ONE,
        DeploymentStep.SUPPORT_INPUT_SNAPSHOT_TWO,
        DeploymentStep.NO_WORKER_ACTIVATION_PROVED,
    ]
    assert spy.forbidden_calls == []


def test_committed_adoption_rejects_live_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = DeploymentStep.SUPPORT_RUNTIME_IDENTITY_COMMITTED
    calls: list[DeploymentStep] = []
    projections = {
        item: {"step": item.value, "effect": "committed"} for item in DeploymentStep
    }
    projections[step] = {"step": step.value, "effect": "drifted"}
    _install_fake_table(monkeypatch, live_projections=projections, calls=calls)

    with pytest.raises(recovery.StagedEffectRecoveryError, match="byte-identical"):
        recovery.adopt_staged_evidence_v2(
            step=step,
            request=_request(),
            evidence={"step": step.value, "effect": "committed"},
            committed=_prior(step),
            services=_services(_MutationSpy()),
        )

    assert calls == [step]
