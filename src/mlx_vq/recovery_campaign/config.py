"""Strict loader for the tracked GLM-5.2 recovery campaign matrix."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from .models import (
    Authority,
    CampaignConfig,
    Experiment,
    PayloadBudget,
    ProjectionRates,
    SelectionPolicy,
    Transition,
)


class CampaignConfigError(ValueError):
    """The campaign matrix is malformed or violates a frozen guard."""


_ROOT_FIELDS = {
    "schema_version",
    "campaign",
    "model",
    "tuning",
    "budget",
    "authorities",
    "experiments",
    "transitions",
}
_MODEL_FIELDS = {
    "id",
    "revision",
    "source_dir",
    "index_path",
    "config_sha256",
    "index_sha256",
    "full_source_blob_inventory_sha256",
    "routed_source_blob_inventory_sha256",
}
_TUNING_FIELDS = {
    "split",
    "prompt_count",
    "position_count",
    "holdout_used_for_tuning",
    "report_used_for_tuning",
}
_BUDGET_FIELDS = {
    "accepted_payload_bytes",
    "e8p_increment_per_projection_bytes",
    "payload_limit_bytes",
}
_AUTHORITY_FIELDS = {"name", "path", "sha256", "role", "split"}
_EXPERIMENT_FIELDS = {
    "name",
    "kind",
    "depends_on",
    "tuning_authorities",
    "output_path",
    "recovered_layers",
    "projection_rates",
    "recovery_levers",
    "expected_payload_bytes",
    "transition",
}
_RATE_FIELDS = {"gate", "up", "down"}
_TRANSITION_FIELDS = {
    "name",
    "experiment",
    "action",
    "heavy",
    "argv",
    "expected_artifacts",
}
_SHA256_CHARS = frozenset("0123456789abcdef")
_SAFE_ARTIFACT_PREFIX = ("artifacts", "quality")
_FORBIDDEN_ARG_FRAGMENTS = (
    "GLM_MLX_WIRED_LIMIT_GB",
    "GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB",
    "iogpu.wired_limit_mb",
)
_SHELL_EXECUTABLES = frozenset({"bash", "dash", "fish", "sh", "zsh"})
_VALUELESS_OPTIONS = frozenset({"--resume"})
_REMATERIALIZE_PREFIX = (
    ".venv/bin/python",
    "benchmarks/run_glm52_recovery_wave1.py",
    "rematerialize",
)
_REMATERIALIZE_REQUIRED_OPTIONS = frozenset(
    {
        "--source-dir",
        "--index-path",
        "--seed-artifact-dir",
        "--stats-dir",
        "--attribution-json",
        "--output-dir",
        "--worst-layer-count",
        "--resume",
        "--expected-stats-manifest-sha256",
        "--expected-attribution-sha256",
        "--expected-seed-manifest-sha256",
        "--expected-full-source-blob-inventory-sha256",
        "--expected-routed-source-blob-inventory-sha256",
        "--accepted-composite-audit-json",
        "--expected-composite-audit-sha256",
        "--heavy-lock-path",
    }
)
_E8P_WORST_LAYER_COUNT_OPTION = "--e8p-worst-layer-count"


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys at every mapping level."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Hashable, object]:
    loader.flatten_mapping(node)
    mapping: dict[Hashable, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, Hashable):
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            )
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _mapping(value: object, *, label: str, fields: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignConfigError(f"{label} must be a mapping")
    keys = set(value)
    unknown = sorted(str(key) for key in keys - fields)
    missing = sorted(fields - keys)
    if unknown:
        raise CampaignConfigError(f"{label} has unknown fields: {', '.join(unknown)}")
    if missing:
        raise CampaignConfigError(f"{label} is missing fields: {', '.join(missing)}")
    return value


def _sequence(value: object, *, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise CampaignConfigError(f"{label} must be a list")
    return value


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CampaignConfigError(f"{label} must be a non-empty string")
    return value


def _optional_string(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label=label)


def _integer(value: object, *, label: str, positive: bool = False) -> int:
    if type(value) is not int or (positive and value <= 0):
        suffix = " a positive integer" if positive else " an integer"
        raise CampaignConfigError(f"{label} must be{suffix}")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise CampaignConfigError(f"{label} must be a boolean")
    return value


def _strings(value: object, *, label: str) -> tuple[str, ...]:
    return tuple(
        _string(item, label=f"{label}[{index}]")
        for index, item in enumerate(_sequence(value, label=label))
    )


def _integers(value: object, *, label: str) -> tuple[int, ...]:
    result = tuple(
        _integer(item, label=f"{label}[{index}]")
        for index, item in enumerate(_sequence(value, label=label))
    )
    if any(item < 0 for item in result):
        raise CampaignConfigError(f"{label} must contain non-negative layers")
    if len(set(result)) != len(result):
        raise CampaignConfigError(f"{label} contains duplicate layers")
    return result


def _sha256(value: object, *, label: str) -> str:
    digest = _string(value, label=label)
    if len(digest) != 64 or not set(digest) <= _SHA256_CHARS:
        raise CampaignConfigError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _safe_artifact_path(value: object, *, label: str) -> str:
    raw = _string(value, label=label)
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or "\\" in raw
        or ".." in path.parts
        or tuple(path.parts[:2]) != _SAFE_ARTIFACT_PREFIX
        or len(path.parts) < 3
    ):
        raise CampaignConfigError(
            f"{label} must be a safe artifact path under artifacts/quality"
        )
    return raw


def _selection_policy(value: object) -> SelectionPolicy:
    raw = _mapping(value, label="tuning", fields=_TUNING_FIELDS)
    policy = SelectionPolicy(
        split=_string(raw["split"], label="tuning.split"),
        prompt_count=_integer(
            raw["prompt_count"], label="tuning.prompt_count", positive=True
        ),
        position_count=_integer(
            raw["position_count"], label="tuning.position_count", positive=True
        ),
        holdout_used_for_tuning=_boolean(
            raw["holdout_used_for_tuning"],
            label="tuning.holdout_used_for_tuning",
        ),
        report_used_for_tuning=_boolean(
            raw["report_used_for_tuning"],
            label="tuning.report_used_for_tuning",
        ),
    )
    if (
        policy.split != "selection"
        or policy.holdout_used_for_tuning
        or policy.report_used_for_tuning
    ):
        raise CampaignConfigError(
            "selection is the only permitted recovery tuning split"
        )
    return policy


def _payload_budget(value: object) -> PayloadBudget:
    raw = _mapping(value, label="budget", fields=_BUDGET_FIELDS)
    budget = PayloadBudget(
        accepted_payload_bytes=_integer(
            raw["accepted_payload_bytes"],
            label="budget.accepted_payload_bytes",
            positive=True,
        ),
        e8p_increment_per_projection_bytes=_integer(
            raw["e8p_increment_per_projection_bytes"],
            label="budget.e8p_increment_per_projection_bytes",
            positive=True,
        ),
        payload_limit_bytes=_integer(
            raw["payload_limit_bytes"],
            label="budget.payload_limit_bytes",
            positive=True,
        ),
    )
    if budget.accepted_payload_bytes > budget.payload_limit_bytes:
        raise CampaignConfigError("accepted payload exceeds the payload limit")
    return budget


def _authorities(value: object) -> tuple[Authority, ...]:
    authorities: list[Authority] = []
    for index, item in enumerate(_sequence(value, label="authorities")):
        label = f"authorities[{index}]"
        raw = _mapping(item, label=label, fields=_AUTHORITY_FIELDS)
        authority = Authority(
            name=_string(raw["name"], label=f"{label}.name"),
            path=_string(raw["path"], label=f"{label}.path"),
            sha256=_sha256(raw["sha256"], label=f"{label}.sha256"),
            role=_string(raw["role"], label=f"{label}.role"),
            split=_optional_string(raw["split"], label=f"{label}.split"),
        )
        if authority.role not in {"artifact", "tuning", "reporting"}:
            raise CampaignConfigError(f"{label}.role is not supported")
        if authority.role == "tuning" and authority.split != "selection":
            raise CampaignConfigError(
                f"{label}: tuning authorities must use the selection split"
            )
        if authority.role == "artifact" and authority.split is not None:
            raise CampaignConfigError(f"{label}: artifact authorities have no split")
        authorities.append(authority)
    names = [authority.name for authority in authorities]
    if len(set(names)) != len(names):
        raise CampaignConfigError("duplicate authority name")
    return tuple(authorities)


def _projection_rates(value: object, *, label: str) -> tuple[ProjectionRates, ...]:
    if not isinstance(value, Mapping):
        raise CampaignConfigError(f"{label} must be a mapping")
    rates: list[ProjectionRates] = []
    for layer_key, item in value.items():
        layer_label = f"{label}.{layer_key}"
        try:
            layer = int(layer_key)
        except (TypeError, ValueError) as error:
            raise CampaignConfigError(f"{layer_label} is not a layer number") from error
        if layer < 0 or str(layer) != str(layer_key):
            raise CampaignConfigError(f"{layer_label} is not a canonical layer number")
        if not isinstance(item, Mapping) or set(item) != _RATE_FIELDS:
            raise CampaignConfigError(
                f"{layer_label} must contain exactly gate, up, and down rates"
            )
        raw = _mapping(item, label=layer_label, fields=_RATE_FIELDS)
        values = {
            projection: _integer(
                raw[projection], label=f"{layer_label}.{projection}", positive=True
            )
            for projection in ("gate", "up", "down")
        }
        if any(bits not in {8, 16} for bits in values.values()):
            raise CampaignConfigError(f"{layer_label} rates must be 8 or 16 bits")
        if len(set(values.values())) != 1:
            raise CampaignConfigError(
                f"{layer_label} must use a uniform whole-layer triplet of "
                "all 8-bit or all 16-bit rates"
            )
        rates.append(ProjectionRates(layer=layer, **values))
    return tuple(rates)


def _experiments(value: object) -> tuple[Experiment, ...]:
    experiments: list[Experiment] = []
    for index, item in enumerate(_sequence(value, label="experiments")):
        label = f"experiments[{index}]"
        raw = _mapping(item, label=label, fields=_EXPERIMENT_FIELDS)
        recovered_layers = _integers(
            raw["recovered_layers"], label=f"{label}.recovered_layers"
        )
        rates = _projection_rates(
            raw["projection_rates"], label=f"{label}.projection_rates"
        )
        if not set(rate.layer for rate in rates) <= set(recovered_layers):
            raise CampaignConfigError(
                f"{label}.projection_rates contains a layer outside recovered_layers"
            )
        experiments.append(
            Experiment(
                name=_string(raw["name"], label=f"{label}.name"),
                kind=_string(raw["kind"], label=f"{label}.kind"),
                depends_on=_strings(raw["depends_on"], label=f"{label}.depends_on"),
                tuning_authorities=_strings(
                    raw["tuning_authorities"],
                    label=f"{label}.tuning_authorities",
                ),
                output_path=_safe_artifact_path(
                    raw["output_path"], label=f"{label}.output_path"
                ),
                recovered_layers=recovered_layers,
                projection_rates=rates,
                recovery_levers=_strings(
                    raw["recovery_levers"], label=f"{label}.recovery_levers"
                ),
                expected_payload_bytes=_integer(
                    raw["expected_payload_bytes"],
                    label=f"{label}.expected_payload_bytes",
                    positive=True,
                ),
                transition_name=_optional_string(
                    raw["transition"], label=f"{label}.transition"
                ),
            )
        )
    names = [experiment.name for experiment in experiments]
    if len(set(names)) != len(names):
        raise CampaignConfigError("duplicate experiment name")
    known: set[str] = set()
    for experiment in experiments:
        unknown = set(experiment.depends_on) - known
        if unknown:
            raise CampaignConfigError(
                f"experiment {experiment.name!r} has unknown or forward dependencies: "
                + ", ".join(sorted(unknown))
            )
        known.add(experiment.name)
    return tuple(experiments)


def _transitions(value: object) -> tuple[Transition, ...]:
    transitions: list[Transition] = []
    for index, item in enumerate(_sequence(value, label="transitions")):
        label = f"transitions[{index}]"
        raw = _mapping(item, label=label, fields=_TRANSITION_FIELDS)
        argv = _strings(raw["argv"], label=f"{label}.argv")
        if not argv:
            raise CampaignConfigError(f"{label}.argv must not be empty")
        if PurePosixPath(argv[0]).name in _SHELL_EXECUTABLES:
            raise CampaignConfigError(f"{label}.argv must not execute through a shell")
        if any(
            fragment in arg
            for fragment in _FORBIDDEN_ARG_FRAGMENTS
            for arg in argv
        ):
            raise CampaignConfigError(
                f"{label}.argv contains a forbidden wired-memory limit"
            )
        transitions.append(
            Transition(
                name=_string(raw["name"], label=f"{label}.name"),
                experiment_name=_string(
                    raw["experiment"], label=f"{label}.experiment"
                ),
                action=_string(raw["action"], label=f"{label}.action"),
                heavy=_boolean(raw["heavy"], label=f"{label}.heavy"),
                argv=argv,
                expected_artifacts=tuple(
                    _safe_artifact_path(path, label=f"{label}.expected_artifacts")
                    for path in _sequence(
                        raw["expected_artifacts"],
                        label=f"{label}.expected_artifacts",
                    )
                ),
            )
        )
    names = [transition.name for transition in transitions]
    if len(set(names)) != len(names):
        raise CampaignConfigError("duplicate transition name")
    return tuple(transitions)


def _parse_option_vector(transition: Transition) -> dict[str, str | None]:
    if transition.action != "rematerialize":
        raise CampaignConfigError(
            f"transition {transition.name!r} has unsupported action "
            f"{transition.action!r}"
        )
    if transition.argv[:3] != _REMATERIALIZE_PREFIX:
        raise CampaignConfigError(
            f"transition {transition.name!r} rematerialize argv has the wrong command"
        )
    options: dict[str, str | None] = {}
    index = len(_REMATERIALIZE_PREFIX)
    while index < len(transition.argv):
        flag = transition.argv[index]
        if not flag.startswith("--") or "=" in flag:
            raise CampaignConfigError(
                f"transition {transition.name!r} argv expected an option, got {flag!r}"
            )
        if flag in options:
            raise CampaignConfigError(
                f"transition {transition.name!r} argv has duplicate option {flag}"
            )
        if flag in _VALUELESS_OPTIONS:
            options[flag] = None
            index += 1
            continue
        if index + 1 >= len(transition.argv) or transition.argv[index + 1].startswith(
            "--"
        ):
            raise CampaignConfigError(
                f"transition {transition.name!r} argv is missing required value "
                f"for {flag}"
            )
        options[flag] = transition.argv[index + 1]
        index += 2
    return options


def _authority(
    authorities: Mapping[str, Authority], *, name: str, transition_name: str
) -> Authority:
    authority = authorities.get(name)
    if authority is None:
        raise CampaignConfigError(
            f"transition {transition_name!r} requires authority {name!r}"
        )
    return authority


def _require_option(
    options: Mapping[str, str | None],
    *,
    flag: str,
    expected: str,
    transition_name: str,
    authority_name: str | None = None,
) -> None:
    if options.get(flag) != expected:
        provenance = f" from authority {authority_name!r}" if authority_name else ""
        raise CampaignConfigError(
            f"transition {transition_name!r} {flag} must equal {expected!r}{provenance}"
        )


def _transition_tuning_authorities(
    experiment: Experiment,
    *,
    authorities: Mapping[str, Authority],
    transition_name: str,
) -> tuple[Authority, Authority]:
    stats_names = tuple(
        name for name in experiment.tuning_authorities if name == "recovery-stats"
    )
    if len(stats_names) != 1:
        raise CampaignConfigError(
            f"transition {transition_name!r} requires recovery-stats exactly once "
            "in the owning experiment tuning_authorities"
        )
    attribution_names = tuple(
        name for name in experiment.tuning_authorities if name != "recovery-stats"
    )
    if len(attribution_names) != 1:
        raise CampaignConfigError(
            f"transition {transition_name!r} requires exactly one attribution "
            "authority alongside recovery-stats"
        )
    recovery_stats = _authority(
        authorities,
        name=stats_names[0],
        transition_name=transition_name,
    )
    attribution = _authority(
        authorities,
        name=attribution_names[0],
        transition_name=transition_name,
    )
    for authority in (recovery_stats, attribution):
        if authority.role != "tuning" or authority.split != "selection":
            raise CampaignConfigError(
                f"transition {transition_name!r} authority {authority.name!r} "
                "must be a selection tuning authority"
            )
    return recovery_stats, attribution


def _validate_rematerialize_transition(
    transition: Transition,
    *,
    config: CampaignConfig,
    experiment: Experiment,
    authorities: Mapping[str, Authority],
) -> None:
    if not transition.heavy:
        raise CampaignConfigError(
            f"transition {transition.name!r} rematerialize action must be heavy"
        )
    if not experiment.recovered_layers:
        raise CampaignConfigError(
            f"transition {transition.name!r} recovered_layers must be nonempty"
        )
    options = _parse_option_vector(transition)
    e8p_layers = tuple(
        rates.layer
        for rates in experiment.projection_rates
        if rates.values == (16, 16, 16)
    )
    e8p_layer_count = len(e8p_layers)
    if not 0 <= e8p_layer_count <= 16:
        raise CampaignConfigError(
            f"transition {transition.name!r} may upgrade at most 16 E8P layers"
        )
    if e8p_layers != experiment.recovered_layers[:e8p_layer_count]:
        raise CampaignConfigError(
            f"transition {transition.name!r} ordered E8P layer identities must "
            "equal the recovered_layers prefix"
        )
    required_options = set(_REMATERIALIZE_REQUIRED_OPTIONS)
    if e8p_layer_count:
        required_options.add(_E8P_WORST_LAYER_COUNT_OPTION)
    unknown_options = sorted(set(options) - required_options)
    if unknown_options:
        raise CampaignConfigError(
            f"transition {transition.name!r} argv has unknown option(s): "
            + ", ".join(unknown_options)
        )
    missing_options = sorted(required_options - set(options))
    if missing_options:
        raise CampaignConfigError(
            f"transition {transition.name!r} argv is missing required option(s): "
            + ", ".join(missing_options)
        )
    recovery_stats, attribution = _transition_tuning_authorities(
        experiment,
        authorities=authorities,
        transition_name=transition.name,
    )
    seed_manifest = _authority(
        authorities, name="accepted-seed-manifest", transition_name=transition.name
    )
    composite_audit = _authority(
        authorities, name="accepted-composite-audit", transition_name=transition.name
    )
    bindings = (
        (
            "--stats-dir",
            str(PurePosixPath(recovery_stats.path).parent),
            recovery_stats.name,
        ),
        (
            "--expected-stats-manifest-sha256",
            recovery_stats.sha256,
            recovery_stats.name,
        ),
        ("--attribution-json", attribution.path, attribution.name),
        (
            "--expected-attribution-sha256",
            attribution.sha256,
            attribution.name,
        ),
        (
            "--seed-artifact-dir",
            str(PurePosixPath(seed_manifest.path).parent),
            seed_manifest.name,
        ),
        (
            "--expected-seed-manifest-sha256",
            seed_manifest.sha256,
            seed_manifest.name,
        ),
        (
            "--accepted-composite-audit-json",
            composite_audit.path,
            composite_audit.name,
        ),
        (
            "--expected-composite-audit-sha256",
            composite_audit.sha256,
            composite_audit.name,
        ),
    )
    for flag, expected, authority_name in bindings:
        _require_option(
            options,
            flag=flag,
            expected=expected,
            transition_name=transition.name,
            authority_name=authority_name,
        )
    source_bindings = (
        ("--source-dir", config.source_dir),
        ("--index-path", config.index_path),
        (
            "--expected-full-source-blob-inventory-sha256",
            config.full_source_blob_inventory_sha256,
        ),
        (
            "--expected-routed-source-blob-inventory-sha256",
            config.routed_source_blob_inventory_sha256,
        ),
        ("--worst-layer-count", str(len(experiment.recovered_layers))),
    )
    for flag, expected in source_bindings:
        _require_option(
            options,
            flag=flag,
            expected=expected,
            transition_name=transition.name,
        )
    if e8p_layer_count:
        _require_option(
            options,
            flag=_E8P_WORST_LAYER_COUNT_OPTION,
            expected=str(e8p_layer_count),
            transition_name=transition.name,
        )
    _require_option(
        options,
        flag="--output-dir",
        expected=experiment.output_path,
        transition_name=transition.name,
    )
    _require_option(
        options,
        flag="--heavy-lock-path",
        expected=".keep-heavy-job.lock",
        transition_name=transition.name,
    )


def _validate_expected_artifacts(
    transition: Transition, *, experiment: Experiment
) -> None:
    expected = (f"{experiment.output_path}/conversion-manifest.json",)
    if transition.expected_artifacts != expected:
        raise CampaignConfigError(
            f"transition {transition.name!r} expected_artifacts must equal the "
            f"exact conversion-manifest.json strict descendant {expected!r}"
        )


def _cross_validate(config: CampaignConfig) -> None:
    authorities = {authority.name: authority for authority in config.authorities}
    transitions = {transition.name: transition for transition in config.transitions}
    source_dir = PurePosixPath(config.source_dir)
    if not source_dir.is_absolute() or source_dir.name != config.model_revision:
        raise CampaignConfigError(
            "model revision must match the absolute source_dir snapshot name"
        )
    expected_model_component = f"models--{config.model_id.replace('/', '--')}"
    if tuple(source_dir.parts[-3:]) != (
        expected_model_component,
        "snapshots",
        config.model_revision,
    ):
        raise CampaignConfigError(
            "model_id must match the source_dir Hugging Face cache component"
        )
    expected_index_path = source_dir / "model.safetensors.index.json"
    if PurePosixPath(config.index_path) != expected_index_path:
        raise CampaignConfigError(
            "model index_path must equal source_dir/model.safetensors.index.json"
        )
    transition_owners: dict[str, list[Experiment]] = {
        transition.name: [] for transition in config.transitions
    }
    for experiment in config.experiments:
        for authority_name in experiment.tuning_authorities:
            authority = authorities.get(authority_name)
            if authority is None:
                raise CampaignConfigError(
                    f"experiment {experiment.name!r} references unknown authority "
                    f"{authority_name!r}"
                )
            if experiment.transition_name is None and (
                authority.role != "tuning" or authority.split != "selection"
            ):
                raise CampaignConfigError(
                    f"experiment {experiment.name!r} may tune only from selection authorities"
                )
        canonical = config.canonical_payload_bytes(experiment)
        if experiment.expected_payload_bytes != canonical:
            raise CampaignConfigError(
                f"experiment {experiment.name!r} expected payload does not match "
                f"canonical payload {canonical}"
            )
        if canonical > config.budget.payload_limit_bytes:
            raise CampaignConfigError(
                f"experiment {experiment.name!r} exceeds the payload limit"
            )
        if experiment.transition_name is not None:
            transition = transitions.get(experiment.transition_name)
            if transition is None:
                raise CampaignConfigError(
                    f"experiment {experiment.name!r} references unknown transition"
                )
            transition_owners[transition.name].append(experiment)
    for transition in config.transitions:
        owners = transition_owners[transition.name]
        if len(owners) != 1:
            raise CampaignConfigError(
                f"transition {transition.name!r} must be named by exactly one experiment"
            )
        experiment = owners[0]
        if transition.experiment_name != experiment.name:
            raise CampaignConfigError(
                f"transition {transition.name!r} targets the wrong experiment"
            )
        _validate_expected_artifacts(transition, experiment=experiment)
        if transition.action != "rematerialize":
            raise CampaignConfigError(
                f"transition {transition.name!r} has unsupported action "
                f"{transition.action!r}"
            )
        _validate_rematerialize_transition(
            transition,
            config=config,
            experiment=experiment,
            authorities=authorities,
        )


def load_campaign_config(path: str | Path) -> CampaignConfig:
    """Load and fail-closed validate one recovery campaign YAML matrix."""

    config_path = Path(path)
    try:
        value = yaml.load(
            config_path.read_text(encoding="utf-8"),
            Loader=_UniqueKeySafeLoader,
        )
    except (OSError, yaml.YAMLError) as error:
        raise CampaignConfigError(f"could not load campaign config {config_path}: {error}") from error
    raw = _mapping(value, label="campaign config", fields=_ROOT_FIELDS)
    model = _mapping(raw["model"], label="model", fields=_MODEL_FIELDS)
    config = CampaignConfig(
        schema_version=_integer(
            raw["schema_version"], label="schema_version", positive=True
        ),
        campaign=_string(raw["campaign"], label="campaign"),
        model_id=_string(model["id"], label="model.id"),
        model_revision=_string(model["revision"], label="model.revision"),
        source_dir=_string(model["source_dir"], label="model.source_dir"),
        index_path=_string(model["index_path"], label="model.index_path"),
        config_sha256=_sha256(
            model["config_sha256"],
            label="model.config_sha256",
        ),
        index_sha256=_sha256(
            model["index_sha256"],
            label="model.index_sha256",
        ),
        full_source_blob_inventory_sha256=_sha256(
            model["full_source_blob_inventory_sha256"],
            label="model.full_source_blob_inventory_sha256",
        ),
        routed_source_blob_inventory_sha256=_sha256(
            model["routed_source_blob_inventory_sha256"],
            label="model.routed_source_blob_inventory_sha256",
        ),
        tuning=_selection_policy(raw["tuning"]),
        budget=_payload_budget(raw["budget"]),
        authorities=_authorities(raw["authorities"]),
        experiments=_experiments(raw["experiments"]),
        transitions=_transitions(raw["transitions"]),
        campaign_config_sha256=hashlib.sha256(
            json.dumps(
                raw,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        campaign_config_path=str(config_path.absolute()),
    )
    if config.schema_version != 1:
        raise CampaignConfigError("schema_version must be 1")
    _cross_validate(config)
    return config


__all__ = ["CampaignConfigError", "load_campaign_config"]
