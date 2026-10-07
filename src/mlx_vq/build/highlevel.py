"""High-level KEEP recipe compiler.

The public YAML surface is intentionally small and human-composable. This
module expands it into the existing low-level step graph schema used by the
recipe runner; runners and ops remain the source of execution behavior.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable

import yaml

from mlx_vq.build.recipe import RecipeError
from mlx_vq.models.profiles import ModelProfile, ProfileError, get_profile

QUALITY_PRESETS = frozenset({"fast", "balanced", "wow"})

GLM52_RELEASE_INPUT_KEYS = (
    "teacher_artifact_identities_json",
    "teacher_ledger_path",
    "teacher_checkpoint_dir",
    "candidate_route_input_root",
    "source_route_authority_json",
    "candidate_route_authority_json",
    "benchmark_prompt_token_ids_json",
    "benchmark_source_contract_json",
)


def is_high_level_recipe(path: str | Path) -> bool:
    """Return true when ``path`` uses the beginner high-level format."""

    raw = _load_yaml_mapping(path)
    return "model" in raw and "steps" not in raw


def compile_high_level(path: str | Path) -> dict[str, Any]:
    """Compile a high-level YAML file into a low-level recipe mapping."""

    source_path = Path(path)
    raw = _load_yaml_mapping(source_path)
    if not ("model" in raw and "steps" not in raw):
        raise RecipeError([f"{source_path}: not a high-level recipe"])

    model = raw.get("model")
    if not isinstance(model, str) or not model:
        raise RecipeError([f"{source_path}: high-level recipe requires model"])

    quality = str(raw.get("quality") or "fast")
    if quality not in QUALITY_PRESETS:
        choices = ", ".join(sorted(QUALITY_PRESETS))
        raise RecipeError([f"unknown quality {quality!r}; choices: {choices}"])

    recovery = str(raw.get("recovery") or "auto")
    if recovery not in {"auto", "off"}:
        raise RecipeError(["recovery must be 'auto' or 'off'"])

    calibration = str(raw.get("calibration") or "default")
    output = str(raw.get("output") or f"runs/{model}-{quality}")
    overrides = _parse_overrides(raw.get("overrides") or {})

    try:
        profile = get_profile(model)
    except ProfileError as error:
        raise RecipeError([str(error)]) from error

    try:
        builder = TEMPLATES[profile.converter]
    except KeyError as error:
        raise RecipeError([f"no high-level template for converter {profile.converter!r}"]) from error

    release_inputs = _parse_glm52_release_inputs(raw.get("release_inputs"), profile)

    compiled = builder(
        profile,
        HighLevelOptions(
            quality=quality,
            output=output,
            calibration=calibration,
            recovery=recovery,
            release_inputs=release_inputs,
        ),
    )
    _apply_overrides(compiled, overrides)
    return compiled


class HighLevelOptions:
    def __init__(
        self,
        *,
        quality: str,
        output: str,
        calibration: str,
        recovery: str,
        release_inputs: dict[str, str] | None = None,
    ) -> None:
        self.quality = quality
        self.output = output
        self.calibration = calibration
        self.recovery = recovery
        self.release_inputs = release_inputs or {}


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    recipe_path = Path(path)
    try:
        raw = yaml.safe_load(recipe_path.read_text())
    except OSError as error:
        raise RecipeError([f"could not read {recipe_path}: {error}"]) from error
    if not isinstance(raw, dict):
        raise RecipeError([f"{recipe_path}: recipe must be a YAML mapping"])
    return raw


def _parse_overrides(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        raise RecipeError(["overrides must be a mapping of step id to params"])
    parsed: dict[str, dict[str, Any]] = {}
    for step_id, params in raw.items():
        if not isinstance(step_id, str) or not step_id:
            raise RecipeError(["overrides step ids must be non-empty strings"])
        if not isinstance(params, dict):
            raise RecipeError([f"overrides.{step_id} must be a mapping"])
        parsed[step_id] = dict(params)
    return parsed


def _parse_glm52_release_inputs(raw: Any, profile: ModelProfile) -> dict[str, str]:
    """Validate the explicit raw inputs not produced by the build graph.

    Route traces and the frozen benchmark prompt/control authorities are emitted
    by the release capture environment, not by a model-conversion primitive.
    Making them explicit prevents a high-level recipe from quietly compiling a
    release graph that has no real source for those evidence lanes.
    """

    if profile.converter != "glm52_vq_groups":
        return {}
    if not isinstance(raw, dict):
        raise RecipeError([
            "GLM52 high-level recipe requires release_inputs mapping for raw "
            "route traces, benchmark authorities, and teacher-cache state"
        ])
    missing = [key for key in GLM52_RELEASE_INPUT_KEYS if key not in raw]
    if missing:
        raise RecipeError([
            "GLM52 high-level recipe requires release_inputs keys: "
            + ", ".join(missing)
        ])
    parsed: dict[str, str] = {}
    for key in GLM52_RELEASE_INPUT_KEYS:
        value = raw[key]
        if not isinstance(value, str) or not value:
            raise RecipeError([f"release_inputs.{key} must be a non-empty path string"])
        parsed[key] = value
    _validate_glm52_benchmark_prompt(parsed["benchmark_prompt_token_ids_json"])
    return parsed


def _validate_glm52_benchmark_prompt(path_value: str) -> None:
    """Reject wrong-shaped benchmark input before any model step is planned."""

    path = Path(path_value).expanduser()
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RecipeError([
            f"release_inputs.benchmark_prompt_token_ids_json must be readable JSON: {error}"
        ]) from error
    if (
        not isinstance(value, list)
        or len(value) != 1024
        or any(type(token) is not int for token in value)
    ):
        raise RecipeError([
            "release_inputs.benchmark_prompt_token_ids_json must contain exactly 1024 JSON integers"
        ])


def _apply_overrides(recipe: dict[str, Any], overrides: dict[str, dict[str, Any]]) -> None:
    if not overrides:
        return
    steps = recipe.get("steps") or []
    by_id = {step.get("id"): step for step in steps if isinstance(step, dict)}
    for step_id, params in overrides.items():
        step = by_id.get(step_id)
        if step is None:
            raise RecipeError([f"overrides references unknown step {step_id!r}"])
        step_params = step.setdefault("params", {})
        if not isinstance(step_params, dict):
            raise RecipeError([f"step {step_id!r} params are not a mapping"])
        step_params.update(params)


def _profile_revision(profile: ModelProfile) -> str:
    return profile.revision or "main"


def _hf_snapshot_path(profile: ModelProfile) -> str:
    repo_key = "models--" + profile.hf_model_id.replace("/", "--")
    return str(Path.home() / ".cache" / "huggingface" / "hub" / repo_key / "snapshots" / _profile_revision(profile))


def _profile_projection_names(profile: ModelProfile) -> list[str]:
    names: list[str] = []
    for projection in profile.recovery_projections:
        names.append(projection if projection.endswith("_proj") else f"{projection}_proj")
    return names


def _air_gate_thresholds(quality: str) -> dict[str, Any]:
    if quality == "fast":
        return {"allow_dirty_rows": True, "clean_rows": 0}
    return {}


def _air_lane_s_thresholds(quality: str) -> dict[str, Any]:
    if quality == "fast":
        return {"allow_dirty_rows": True}
    return {}


def _air_lane_s_params(quality: str) -> dict[str, Any]:
    params: dict[str, Any] = {
        "engine": "vq_e1_routed_nax_e8p",
        "scenario": "prefill_1k",
        "warmup_repetitions": 1,
        "repetitions": 2,
        "clean_repetition_max_attempts": 1,
        "memory_quiet_window_seconds": 0,
        "memory_quiet_max_attempts": 1,
        "prefill_memory_quiet_window_seconds": 0,
        "prefill_memory_quiet_max_attempts": 1,
        "parent_vm_stat_diagnostics": True,
        "memory_phase_trace": True,
    }
    if quality == "wow":
        params.update(
            {
                "repetitions": 3,
                "clean_repetition_max_attempts": 3,
                "memory_quiet_window_seconds": 5,
                "prefill_memory_quiet_window_seconds": 5,
            }
        )
    return params


def _glm45_air_template(profile: ModelProfile, options: HighLevelOptions) -> dict[str, Any]:
    prompt_sets = profile.prompt_sets or {}
    prompt_set = (
        prompt_sets.get("imatrix", "air_imatrix_calib_v1")
        if options.calibration == "default"
        else options.calibration
    )
    recovery_enabled = options.recovery == "auto"
    recovery_source = (
        "step:train_l45_gud_math8/artifact"
        if recovery_enabled
        else "step:materialize_sweep/artifact_bpw2p0"
    )
    promotion_artifact_step = "train_l45_gud_math8" if recovery_enabled else "materialize_sweep"
    projections = _profile_projection_names(profile)

    steps: list[dict[str, Any]] = [
        {
            "id": "collect_imatrix",
            "op": "collect-imatrix",
            "class": "promotable",
            "inputs": {
                "artifact": "external:baseline_artifact",
                "prompt_selection": "external:prompt_selection",
            },
            "params": {
                "prompt_set": prompt_set,
                "layers": list(profile.imatrix_layers),
                "projections": projections,
            },
        },
        {
            "id": "materialize_sweep",
            "op": "materialize-sweep",
            "class": "promotable",
            "inputs": {
                "imatrix_manifest": "step:collect_imatrix/imatrix_manifest",
                "baseline_artifact": "external:high_bit_artifact",
                "high_bit_artifact": "external:high_bit_artifact",
            },
            "params": {
                "candidate_prefix": "air-base-to-rc-agq-ebss32-affinity-reround8",
                "budget": [2.0],
                "mid_error_factor": 0.5,
                "reround_top_low_experts": 8,
                "importance_key": "affinity_weighted_importance",
            },
        },
    ]
    if recovery_enabled:
        steps.append(
            {
                "id": "train_l45_gud_math8",
                "op": "train-low-rank",
                "class": "promotable",
                "inputs": {
                    "seed_artifact": "step:materialize_sweep/artifact_bpw2p0",
                    "selection_teacher": "external:teacher_selection",
                    "validation_teacher": "external:teacher_report",
                },
                "params": {
                    "prefill_engine": "nax_e8p",
                    "layer": profile.recovery_layer,
                    "projections": projections,
                    "trainable": "low_rank_residual",
                    "low_rank": 4,
                    "low_rank_init_scale": 0.05,
                    "steps": 12,
                    "learning_rate": 0.5,
                    "target_nll_weight": 0.5,
                    "teacher_top1_margin_weight": 2,
                    "teacher_top1_margin": 1,
                    "loss_scope": "final_layer_selected",
                    "surrogate_projections": projections,
                    "surrogate_output_chunk_size": 256,
                    "train_cache": "both",
                    "max_train_rows": 8,
                    "train_row_indices": [44, 45, 46, 47, 48, 49, 50, 51],
                    "max_positions": 16,
                },
                "gate": {"profile": "train_sane"},
            }
        )

    steps.extend(
        [
            {
                "id": "audit",
                "op": "audit",
                "class": "verify",
                "inputs": {"artifact": recovery_source},
                "gate": {"profile": "audit_ok"},
            },
            {
                "id": "eval_report_raw",
                "op": "eval",
                "class": "verify",
                "inputs": {"artifact": recovery_source, "teacher": "external:teacher_report"},
                "params": {
                    "engine": profile.default_engine,
                    "max_rows": 128,
                    "mlx_cache_limit_gb": 48,
                    "clear_mlx_cache_between_rows": False,
                },
            },
            {
                "id": "eval_selection_raw",
                "op": "eval",
                "class": "verify",
                "inputs": {"artifact": recovery_source, "teacher": "external:teacher_selection"},
                "params": {
                    "engine": profile.default_engine,
                    "max_rows": 128,
                    "mlx_cache_limit_gb": 48,
                    "clear_mlx_cache_between_rows": False,
                },
            },
            {
                "id": "eval_holdout_raw",
                "op": "eval",
                "class": "verify",
                "inputs": {"artifact": recovery_source, "teacher": "external:teacher_holdout"},
                "params": {
                    "engine": profile.default_engine,
                    "max_rows": 128,
                    "mlx_cache_limit_gb": 48,
                    "clear_mlx_cache_between_rows": False,
                },
            },
            {
                "id": "lane_s_candidate",
                "op": "benchmark-lane-s",
                "class": "verify",
                "inputs": {
                    "artifact": recovery_source,
                    "control_evidence": "external:lane_s_q2_control",
                },
                "params": _air_lane_s_params(options.quality),
                "gate": {
                    "profile": "lane_s",
                    "thresholds": _air_lane_s_thresholds(options.quality),
                },
            },
        ]
    )

    gate_thresholds = _air_gate_thresholds(options.quality)
    description = (
        "High-level GLM-4.5-Air recipe compiled from profile glm45-air. "
        "Fast quality keeps current speed-over-cleanliness allow_dirty_rows gates."
        if options.quality == "fast"
        else "High-level GLM-4.5-Air recipe compiled from profile glm45-air with strict gates."
    )
    if not recovery_enabled:
        description += " Recovery is disabled; evals target the materialized artifact."

    return {
        "schema_version": 1,
        "name": f"{profile.name}-{options.quality}",
        "description": description,
        "build_root": options.output,
        "external_inputs": {
            "baseline_artifact": {"kind": "artifact_dir", "path": "artifacts/glm-4.5-air-vq"},
            "high_bit_artifact": {
                "kind": "artifact_dir",
                "path": "artifacts/glm-4.5-air-vq2-e8p-rtn-uniform-parallel8",
            },
            "prompt_selection": {
                "kind": "file",
                "path": "artifacts/build/glm45air__dmx2p0__agq_ebss32_requant_20260707/glm45air__dmx2p0__agq_ebss32_requant_20260707/steps/agq_select_prompts_01-af0dd195/evidence/selection_json.json",
            },
            "teacher_report": {
                "kind": "teacher_cache",
                "path": "artifacts/quality/glm45-air-teacher-cache-air-vq-ladder-report-v1-full-logits-route-trace-clean/metadata.jsonl",
            },
            "teacher_selection": {
                "kind": "teacher_cache",
                "path": "artifacts/quality/glm45-air-teacher-cache-air-vq-ladder-select-v1-full-logits-route-trace-clean/metadata.jsonl",
            },
            "teacher_holdout": {
                "kind": "teacher_cache",
                "path": "artifacts/quality/glm45-air-teacher-cache-air-vq-ladder-holdout-v1-full-logits-clean-merged-20260701/metadata.jsonl",
            },
            "lane_s_q2_control": {
                "kind": "file",
                "path": "artifacts/benchmarks/glm45-air-target23-nextcycle-r26-q2-control-prefill1k-quiet-20260701.jsonl",
            },
        },
        "steps": steps,
        "promotion": {
            "artifact_step": promotion_artifact_step,
            "gates": [
                {
                    "evidence_step": "eval_report_raw",
                    "profile": "community_wow",
                    "thresholds": deepcopy(gate_thresholds),
                },
                {
                    "evidence_step": "eval_selection_raw",
                    "profile": "community_wow",
                    "thresholds": deepcopy(gate_thresholds),
                },
                {
                    "evidence_step": "eval_holdout_raw",
                    "profile": "community_wow",
                    "thresholds": deepcopy(gate_thresholds),
                },
                {
                    "evidence_step": "lane_s_candidate",
                    "profile": "lane_s",
                    "thresholds": _air_lane_s_thresholds(options.quality),
                },
            ],
            "target_metric_profile": "community_wow",
        },
    }


def _qwen_quality_policy(quality: str) -> dict[str, Any]:
    if quality == "fast":
        return {
            "minimum_clean_rows_per_split": 22,
            "minimum_total_clean_rows": 64,
            "minimum_repetitions_per_scenario": 2,
            "comparison_baseline": "same_machine_qwen_source_switch_projection_control",
            "maximum_candidate_to_reference_ratio": 3.0,
        }
    if quality == "balanced":
        return {
            "minimum_clean_rows_per_split": 22,
            "minimum_total_clean_rows": 64,
            "minimum_repetitions_per_scenario": 2,
            "comparison_baseline": "same_machine_qwen_source_switch_projection_control",
            "maximum_candidate_to_reference_ratio": 2.0,
        }
    return {
        "minimum_clean_rows_per_split": 22,
        "minimum_total_clean_rows": 64,
        "minimum_repetitions_per_scenario": 3,
        "comparison_baseline": "same_machine_qwen_source_switch_projection_control",
        "maximum_candidate_to_reference_ratio": 1.15,
    }


def _qwen36_template(profile: ModelProfile, options: HighLevelOptions) -> dict[str, Any]:
    source_dir = _hf_snapshot_path(profile)
    expected_layers = list(range(profile.num_layers))
    source_projection_groups = profile.num_layers * (2 if profile.fused_gate_up else 3)
    target_projection_groups = profile.num_layers * 3
    group_size = profile.group_size_policy.get("gate_up") or next(iter(profile.group_size_policy.values()))
    model_id = profile.hf_model_id
    revision = _profile_revision(profile)

    common_model_params = {"model_id": model_id, "revision": revision}
    steps = [
        {
            "id": "source_audit",
            "op": "qwen-moe-source-audit",
            "class": "diagnostic",
            "params": {
                **common_model_params,
                "expected_model_type": profile.architecture,
                "expected_language_layers": profile.num_layers,
                "expected_expert_tensors": source_projection_groups,
                "verify_hf_shapes": True,
            },
        },
        {
            "id": "source_payload_audit",
            "op": "qwen-moe-source-payload-audit",
            "class": "diagnostic",
            "inputs": {"source_dir": "external:source_dir", "index_path": "external:index_path"},
            "params": {
                **common_model_params,
                "expected_language_layers": profile.num_layers,
                "max_groups": source_projection_groups,
                "code_bits": profile.default_code_bits,
                "group_size": group_size,
            },
        },
        {
            "id": "materialize_groups",
            "op": "qwen-moe-materialize-groups",
            "class": "diagnostic",
            "inputs": {"source_dir": "external:source_dir", "index_path": "external:index_path"},
            "params": {
                **common_model_params,
                "expected_language_layers": profile.num_layers,
                "max_groups": source_projection_groups,
                "skip_existing": True,
                "code_bits": profile.default_code_bits,
                "group_size": group_size,
            },
        },
        {
            "id": "artifact_audit",
            "op": "qwen-moe-artifact-audit",
            "class": "diagnostic",
            "inputs": {"artifact": "step:materialize_groups/artifact"},
            "params": {
                "expected_source_projection_groups": source_projection_groups,
                "expected_target_projection_groups": target_projection_groups,
            },
        },
        {
            "id": "bind_probe",
            "op": "qwen-moe-bind-probe",
            "class": "diagnostic",
            "inputs": {"artifact": "step:materialize_groups/artifact"},
            "params": {"expected_layers": expected_layers, "model_layer_count": profile.num_layers},
        },
        {
            "id": "non_expert_bind_probe",
            "op": "qwen-non-expert-bind-probe",
            "class": "diagnostic",
            "inputs": {"source_dir": "external:source_dir", "index_path": "external:index_path"},
            "params": {"expected_layers": expected_layers, "layer_only": True, "max_tensors": 700},
        },
        {
            "id": "runtime_bind_forward_probe",
            "op": "qwen-runtime-bind-forward-probe",
            "class": "diagnostic",
            "inputs": {
                "artifact": "step:materialize_groups/artifact",
                "source_dir": "external:source_dir",
                "index_path": "external:index_path",
            },
            "params": {
                "expected_layers": expected_layers,
                "model_layer_count": profile.num_layers,
                "forward_layers": [0, profile.num_layers - 1],
                "forward_top_k": profile.experts_per_tok,
                "input_scale": 0.125,
                "layer_only": True,
                "max_tensors": 700,
            },
        },
        {
            "id": "runtime_global_logit_probe",
            "op": "qwen-runtime-bind-forward-probe",
            "class": "diagnostic",
            "inputs": {
                "artifact": "step:materialize_groups/artifact",
                "source_dir": "external:source_dir",
                "index_path": "external:index_path",
            },
            "params": {
                "expected_layers": [],
                "model_layer_count": profile.num_layers,
                "max_tensors": 2,
                "logit_probe_token_id": 0,
                "logit_probe_top_k": 5,
            },
        },
        {
            "id": "upstream_transformer_probe",
            "op": "qwen-upstream-transformer-bind-probe",
            "class": "diagnostic",
            "inputs": {
                "artifact": "step:materialize_groups/artifact",
                "config_path": "external:config_path",
            },
            "params": {"expected_layers": [0, profile.num_layers - 1]},
        },
        {
            "id": "tokenizer_readiness_probe",
            "op": "qwen-tokenizer-readiness-probe",
            "class": "diagnostic",
            "inputs": {"tokenizer_dir": "external:source_dir"},
            "params": {
                "prompt": "The capital of France is",
                "min_token_count": 1,
                "expected_vocab_size": profile.vocab_size,
            },
        },
        {
            "id": "tokenized_logits_probe",
            "op": "qwen-runtime-bind-forward-probe",
            "class": "diagnostic",
            "inputs": {
                "artifact": "step:materialize_groups/artifact",
                "source_dir": "external:source_dir",
                "index_path": "external:index_path",
                "tokenizer_dir": "external:source_dir",
            },
            "params": {
                "expected_layers": [],
                "model_layer_count": profile.num_layers,
                "max_tensors": 2,
                "tokenizer_prompt": "The capital of France is",
                "tokenized_logit_position": "last",
                "tokenized_logit_top_k": 5,
            },
        },
        {
            "id": "qwen_family_eval_prompts",
            "op": "qwen-family-eval-prompt-probe",
            "class": "diagnostic",
            "inputs": {"tokenizer_dir": "external:source_dir"},
            "params": {
                **common_model_params,
                "min_prompts_per_split": 22,
                "min_tokens_per_prompt": 2,
                "expected_vocab_size": profile.vocab_size,
            },
        },
        {
            "id": "qwen_family_policy",
            "op": "qwen-family-gate-policy",
            "class": "diagnostic",
            "params": {**common_model_params, **_qwen_quality_policy(options.quality)},
        },
        {
            "id": "qwen_family_eval_teacher_metadata",
            "op": "qwen-family-eval-teacher-metadata-probe",
            "class": "diagnostic",
            "inputs": {
                "family_policy_json": "step:qwen_family_policy/evidence_json",
                "eval_prompt_pack_json": "step:qwen_family_eval_prompts/evidence_json",
                "tokenizer_dir": "external:source_dir",
                "source_dir": "external:source_dir",
            },
        },
        {
            "id": "qwen_family_eval_teacher_logits",
            "op": "qwen-family-eval-teacher-logits-probe",
            "class": "diagnostic",
            "inputs": {
                "family_policy_json": "step:qwen_family_policy/evidence_json",
                "eval_prompt_pack_json": "step:qwen_family_eval_prompts/evidence_json",
                "source_dir": "external:source_dir",
                "index_path": "external:index_path",
            },
            "params": {"selected_token_position": "last", "top_k": 32},
        },
        {
            "id": "qwen_family_eval_candidate_logits",
            "op": "qwen-family-eval-candidate-logits-probe",
            "class": "diagnostic",
            "inputs": {
                "family_policy_json": "step:qwen_family_policy/evidence_json",
                "eval_prompt_pack_json": "step:qwen_family_eval_prompts/evidence_json",
                "source_dir": "external:source_dir",
                "index_path": "external:index_path",
                "teacher_logits_jsonl": "step:qwen_family_eval_teacher_logits/teacher_logits_jsonl",
            },
            "params": {"selected_token_position": "last", "top_k": 32},
        },
        {
            "id": "qwen_family_eval_metrics",
            "op": "qwen-family-eval-metric-row-probe",
            "class": "diagnostic",
            "inputs": {
                "family_policy_json": "step:qwen_family_policy/evidence_json",
                "eval_prompt_pack_json": "step:qwen_family_eval_prompts/evidence_json",
                "candidate_logits_jsonl": "step:qwen_family_eval_candidate_logits/candidate_logits_jsonl",
                "teacher_logits_jsonl": "step:qwen_family_eval_teacher_logits/teacher_logits_jsonl",
            },
        },
        {
            "id": "qwen_family_eval_rows",
            "op": "qwen-family-eval-row-probe",
            "class": "diagnostic",
            "inputs": {
                "family_policy_json": "step:qwen_family_policy/evidence_json",
                "eval_prompt_pack_json": "step:qwen_family_eval_prompts/evidence_json",
                "eval_metric_jsonl": "step:qwen_family_eval_metrics/metric_jsonl",
                "teacher_cache_metadata_jsonl": "step:qwen_family_eval_teacher_metadata/metadata_jsonl",
            },
        },
        {
            "id": "qwen_family_eval_gate",
            "op": "qwen-family-eval-gate-check",
            "class": "diagnostic",
            "inputs": {
                "family_policy_json": "step:qwen_family_policy/evidence_json",
                "eval_prompt_pack_json": "step:qwen_family_eval_prompts/evidence_json",
                "eval_jsonl": "step:qwen_family_eval_rows/eval_jsonl",
                "eval_row_probe_json": "step:qwen_family_eval_rows/evidence_json",
            },
        },
        {
            "id": "qwen_family_benchmark_invariants",
            "op": "qwen-family-benchmark-invariant-probe",
            "class": "diagnostic",
            "inputs": {
                "family_policy_json": "step:qwen_family_policy/evidence_json",
                "artifact_audit_json": "step:artifact_audit/evidence_json",
                "non_expert_bind_json": "step:non_expert_bind_probe/evidence_json",
            },
        },
        {
            "id": "qwen_family_candidate_benchmark_latency",
            "op": "qwen-family-candidate-benchmark-latency-probe",
            "class": "diagnostic",
            "inputs": {
                "family_policy_json": "step:qwen_family_policy/evidence_json",
                "artifact": "step:materialize_groups/artifact",
            },
            "params": {
                "layer": 0,
                "repetitions": _qwen_quality_policy(options.quality)["minimum_repetitions_per_scenario"],
                "warmup_repetitions": 0,
                "top_k": profile.experts_per_tok,
            },
        },
        {
            "id": "qwen_family_control_benchmark_latency",
            "op": "qwen-family-control-benchmark-latency-probe",
            "class": "diagnostic",
            "inputs": {
                "family_policy_json": "step:qwen_family_policy/evidence_json",
                "source_dir": "external:source_dir",
                "index_path": "external:index_path",
            },
            "params": {
                "layer": 0,
                "repetitions": _qwen_quality_policy(options.quality)["minimum_repetitions_per_scenario"],
                "warmup_repetitions": 0,
                "top_k": profile.experts_per_tok,
            },
        },
        {
            "id": "qwen_family_benchmark_rows",
            "op": "qwen-family-benchmark-row-probe",
            "class": "diagnostic",
            "inputs": {
                "family_policy_json": "step:qwen_family_policy/evidence_json",
                "candidate_latency_jsonl": "step:qwen_family_candidate_benchmark_latency/latency_jsonl",
                "control_latency_jsonl": "step:qwen_family_control_benchmark_latency/latency_jsonl",
                "candidate_invariant_jsonl": "step:qwen_family_benchmark_invariants/candidate_invariant_jsonl",
            },
        },
        {
            "id": "qwen_family_benchmark_gate",
            "op": "qwen-family-benchmark-gate-check",
            "class": "diagnostic",
            "inputs": {
                "family_policy_json": "step:qwen_family_policy/evidence_json",
                "benchmark_jsonl": "step:qwen_family_benchmark_rows/benchmark_jsonl",
                "benchmark_row_probe_json": "step:qwen_family_benchmark_rows/evidence_json",
            },
        },
        {
            "id": "qwen_family_gate",
            "op": "qwen-family-gate-check",
            "class": "diagnostic",
            "inputs": {
                "source_audit_json": "step:source_audit/evidence_json",
                "source_payload_audit_json": "step:source_payload_audit/evidence_json",
                "artifact_audit_json": "step:artifact_audit/evidence_json",
                "bind_probe_json": "step:bind_probe/evidence_json",
                "non_expert_bind_probe_json": "step:non_expert_bind_probe/evidence_json",
                "runtime_forward_json": "step:runtime_bind_forward_probe/evidence_json",
                "runtime_global_logit_json": "step:runtime_global_logit_probe/evidence_json",
                "upstream_transformer_json": "step:upstream_transformer_probe/evidence_json",
                "tokenizer_json": "step:tokenizer_readiness_probe/evidence_json",
                "tokenized_logits_json": "step:tokenized_logits_probe/evidence_json",
                "family_policy_json": "step:qwen_family_policy/evidence_json",
                "eval_prompt_pack_json": "step:qwen_family_eval_prompts/evidence_json",
                "family_eval_gate_json": "step:qwen_family_eval_gate/evidence_json",
                "family_benchmark_gate_json": "step:qwen_family_benchmark_gate/evidence_json",
            },
            "params": {"expected_layer_count": profile.num_layers},
        },
    ]

    return {
        "schema_version": 1,
        "name": f"{profile.name}-{options.quality}",
        "description": (
            "High-level Qwen3.6-35B-A3B diagnostic recipe compiled from profile "
            f"{profile.name}. Recovery is off for qwen v1 because no Qwen KD "
            "trainer exists yet."
        ),
        "build_root": options.output,
        "external_inputs": {
            "source_dir": {"kind": "artifact_dir", "path": source_dir},
            "index_path": {"kind": "file", "path": str(Path(source_dir) / "model.safetensors.index.json")},
            "config_path": {"kind": "file", "path": str(Path(source_dir) / "config.json")},
        },
        "steps": steps,
    }


def _glm52_reap_template(profile: ModelProfile, options: HighLevelOptions) -> dict[str, Any]:
    """Compile the pinned raw-HF GLM52 source-to-release evidence graph.

    All model work remains in registered subprocess operations.  This compiler
    only declares immutable lineage and the dependency order, so dry-run
    planning never imports MLX or loads the source model.
    """

    source_dir = _hf_snapshot_path(profile)
    common_model_params = {
        "model_id": profile.hf_model_id,
        "revision": _profile_revision(profile),
    }
    conversion_params = {
        **common_model_params,
        "all_groups": True,
        "code_bits": profile.default_code_bits,
        "group_size": 512,
        "scale_estimator": "max_abs",
        "expert_workers": 1,
    }
    audit_params = {
        **common_model_params,
        "all_groups": True,
        "code_bits": profile.default_code_bits,
        "group_size": 512,
        "scale_estimator": "max_abs",
    }
    release = options.release_inputs

    steps: list[dict[str, Any]] = [
        {
            "id": "source_audit",
            "op": "glm52-source-audit",
            "class": "diagnostic",
            "inputs": {
                "profile_path": "external:profile_path",
                "config_path": "external:config_path",
                "index_path": "external:index_path",
            },
            "params": common_model_params,
        },
        {
            "id": "source_payload_audit",
            "op": "glm52-source-payload-audit",
            "class": "diagnostic",
            "inputs": {
                "source_dir": "external:source_dir",
                "profile_path": "external:profile_path",
                "config_path": "external:config_path",
                "index_path": "external:index_path",
                "source_audit_evidence": "step:source_audit/evidence_json",
            },
            "params": common_model_params,
        },
        {
            "id": "materialize_groups",
            "op": "glm52-moe-materialize-groups",
            "class": "diagnostic",
            "inputs": {
                "source_dir": "external:source_dir",
                "profile_path": "external:profile_path",
                "config_path": "external:config_path",
                "index_path": "external:index_path",
                "payload_audit_evidence": "step:source_payload_audit/evidence_json",
            },
            "params": conversion_params,
        },
        {
            "id": "non_vq_pack",
            "op": "glm52-non-vq-pack",
            "class": "diagnostic",
            "inputs": {
                "source_dir": "external:source_dir",
                "profile_path": "external:profile_path",
                "config_path": "external:config_path",
                "index_path": "external:index_path",
                "payload_audit_evidence": "step:source_payload_audit/evidence_json",
            },
            "params": {
                **common_model_params,
                "config_sha256": "5fa690755d0dab25a8e0e5e0745675bdac03ba2b6f5641da2931278235c71f1b",
                "index_sha256": "bb5b4fa9782aea5ffc66f9145d6e630f1045d385c30437c531bebe422c075f3f",
                "max_shard_payload_bytes": 4294967296,
            },
        },
        {
            "id": "composite_audit",
            "op": "glm52-moe-artifact-audit",
            "class": "diagnostic",
            "inputs": {
                "artifact": "step:materialize_groups/artifact",
                "source_dir": "external:source_dir",
                "profile_path": "external:profile_path",
                "config_path": "external:config_path",
                "index_path": "external:index_path",
                "materialization_runs": "step:materialize_groups/evidence",
                "non_vq_artifact": "step:non_vq_pack/artifact",
                "non_vq_evidence_json": "step:non_vq_pack/evidence_json",
            },
            "params": {
                **audit_params,
                "require_resume_proof": True,
            },
        },
        {
            "id": "indexshare_runtime",
            "op": "glm52-indexshare-runtime-probe",
            "class": "diagnostic",
            "inputs": {
                "config_path": "external:config_path",
                "index_path": "external:index_path",
            },
            "params": common_model_params,
        },
        {
            "id": "tokenizer_readiness",
            "op": "glm52-tokenizer-readiness-probe",
            "class": "diagnostic",
            "inputs": {
                "tokenizer_dir": "external:source_dir",
                "config_path": "external:config_path",
            },
            "params": {**common_model_params, "prompt": "The capital of France is"},
        },
        {
            "id": "synthetic_generation",
            "op": "glm52-synthetic-generation-probe",
            "class": "diagnostic",
        },
        {
            "id": "full_bind_evidence",
            "op": "glm52-full-bind-evidence",
            "class": "diagnostic",
            "inputs": {
                "profile_path": "external:profile_path",
                "config_path": "external:config_path",
                "source_index_path": "external:index_path",
                "non_vq_artifact": "step:non_vq_pack/artifact",
                "routed_artifact": "step:materialize_groups/artifact",
            },
            "params": common_model_params,
        },
        {
            "id": "family_policy",
            "op": "glm52-family-gate-policy",
            "class": "diagnostic",
            "params": common_model_params,
        },
        {
            "id": "eval_prompts",
            "op": "glm52-family-eval-prompt-pack",
            "class": "diagnostic",
            "inputs": {
                "tokenizer_dir": "external:source_dir",
                "tokenizer_readiness_json": "step:tokenizer_readiness/evidence_json",
                "family_policy_json": "step:family_policy/evidence_json",
            },
            "params": common_model_params,
        },
        {
            "id": "teacher_metadata",
            "op": "glm52-family-eval-teacher-metadata",
            "class": "diagnostic",
            "inputs": {
                "source_dir": "external:source_dir",
                "profile_path": "external:profile_path",
                "config_path": "external:config_path",
                "index_path": "external:index_path",
                "source_audit_json": "step:source_audit/evidence_json",
                "source_payload_audit_json": "step:source_payload_audit/evidence_json",
                "tokenizer_readiness_json": "step:tokenizer_readiness/evidence_json",
                "family_policy_json": "step:family_policy/evidence_json",
                "eval_prompt_pack_json": "step:eval_prompts/evidence_json",
            },
            "params": common_model_params,
        },
        {
            "id": "teacher_cache",
            "op": "glm52-teacher-cache-produce",
            "class": "diagnostic",
            "inputs": {
                "snapshot_dir": "external:source_dir",
                "prompt_pack_json": "step:eval_prompts/evidence_json",
                "non_vq_package_dir": "step:non_vq_pack/artifact",
                "artifact_identities_json": "external:teacher_artifact_identities_json",
                "profile_path": "external:profile_path",
                "ledger_path": "external:teacher_ledger_path",
                "checkpoint_dir": "external:teacher_checkpoint_dir",
            },
        },
        {
            "id": "teacher_cache_audit",
            "op": "glm52-teacher-cache-audit",
            "class": "diagnostic",
            "inputs": {
                "teacher_cache_root": "step:teacher_cache/artifact",
                "teacher_cache_manifest_json": "step:teacher_cache/manifest",
                "eval_prompt_pack_json": "step:eval_prompts/evidence_json",
            },
        },
        {
            "id": "production_generation",
            "op": "glm52-production-generation-probe",
            "class": "diagnostic",
            "inputs": {
                "profile_path": "external:profile_path",
                "config_path": "external:config_path",
                "source_index_path": "external:index_path",
                "tokenizer_dir": "external:source_dir",
                "tokenizer_readiness_json": "step:tokenizer_readiness/evidence_json",
                "family_policy_json": "step:family_policy/evidence_json",
                "non_vq_artifact": "step:non_vq_pack/artifact",
                "non_vq_evidence_json": "step:non_vq_pack/evidence_json",
                "routed_artifact": "step:materialize_groups/artifact",
                "composite_audit_json": "step:composite_audit/evidence_json",
                "materialization_runs": "step:materialize_groups/evidence",
                "full_bind_preflight_json": "step:full_bind_evidence/evidence_json",
            },
            "params": {**common_model_params, "prompt": "The capital of France is"},
        },
        {
            "id": "candidate_eval_produce",
            "op": "glm52-candidate-full-vocab-eval-produce",
            "class": "diagnostic",
            "inputs": {
                "profile_path": "external:profile_path",
                "config_path": "external:config_path",
                "source_index_path": "external:index_path",
                "tokenizer_dir": "external:source_dir",
                "tokenizer_readiness_json": "step:tokenizer_readiness/evidence_json",
                "family_policy_json": "step:family_policy/evidence_json",
                "prompt_pack_json": "step:eval_prompts/evidence_json",
                "teacher_cache_root": "step:teacher_cache/artifact",
                "non_vq_artifact": "step:non_vq_pack/artifact",
                "non_vq_evidence_json": "step:non_vq_pack/evidence_json",
                "routed_artifact": "step:materialize_groups/artifact",
                "composite_audit_json": "step:composite_audit/evidence_json",
                "materialization_runs": "step:materialize_groups/evidence",
                "full_bind_preflight_json": "step:full_bind_evidence/evidence_json",
                "ledger_path": "external:teacher_ledger_path",
            },
        },
        {
            "id": "candidate_eval_compare",
            "op": "glm52-candidate-full-vocab-eval-compare",
            "class": "diagnostic",
            "inputs": {
                "teacher_cache_root": "step:teacher_cache/artifact",
                "candidate_cache_root": "step:candidate_eval_produce/artifact",
                "prompt_pack_json": "step:eval_prompts/evidence_json",
                "family_policy_json": "step:family_policy/evidence_json",
            },
        },
        {
            "id": "candidate_route_capture",
            "op": "glm52-route-math-capture",
            "class": "diagnostic",
            "inputs": {
                "input_root": "external:candidate_route_input_root",
                "authority_json": "external:candidate_route_authority_json",
            },
            "params": {"side": "candidate", "expert_count": profile.num_experts},
        },
        {
            "id": "route_math_compare",
            "op": "glm52-route-math-compare",
            "class": "diagnostic",
            "inputs": {
                "source_trace_root": "step:teacher_cache/route_trace_root",
                "candidate_trace_root": "step:candidate_route_capture/artifact",
                "source_authority_json": "external:source_route_authority_json",
                "candidate_authority_json": "external:candidate_route_authority_json",
                "family_policy_json": "step:family_policy/evidence_json",
            },
        },
        {
            "id": "same_machine_benchmark_pair",
            "op": "glm52-same-machine-benchmark-pair",
            "class": "diagnostic",
            "inputs": {
                "family_policy_json": "step:family_policy/evidence_json",
                "profile_path": "external:profile_path",
                "non_vq_artifact": "step:non_vq_pack/artifact",
                "prompt_token_ids_json": "external:benchmark_prompt_token_ids_json",
                "config_path": "external:config_path",
                "source_index_path": "external:index_path",
                "tokenizer_dir": "external:source_dir",
                "tokenizer_readiness_json": "step:tokenizer_readiness/evidence_json",
                "non_vq_evidence_json": "step:non_vq_pack/evidence_json",
                "routed_artifact": "step:materialize_groups/artifact",
                "composite_audit_json": "step:composite_audit/evidence_json",
                "materialization_runs": "step:materialize_groups/evidence",
                "full_bind_preflight_json": "step:full_bind_evidence/evidence_json",
                "snapshot_dir": "external:source_dir",
                "prompt_pack_json": "step:eval_prompts/evidence_json",
                "source_contract_json": "external:benchmark_source_contract_json",
            },
        },
        {
            "id": "same_machine_benchmark_compare",
            "op": "glm52-same-machine-benchmark-compare",
            "class": "diagnostic",
            "inputs": {
                "family_policy_json": "step:family_policy/evidence_json",
                "candidate_json": "step:same_machine_benchmark_pair/candidate_json",
                "control_json": "step:same_machine_benchmark_pair/control_json",
                "session_manifest_json": "step:same_machine_benchmark_pair/session_manifest_json",
            },
        },
        {
            "id": "family_gate",
            "op": "glm52-family-gate-check",
            "class": "diagnostic",
            "gate": {"profile": "glm52_family"},
            "inputs": {
                "family_policy_json": "step:family_policy/evidence_json",
                "eval_prompt_pack_json": "step:eval_prompts/evidence_json",
                "teacher_metadata_json": "step:teacher_metadata/evidence_json",
                "full_bind_preflight_json": "step:full_bind_evidence/evidence_json",
                "indexshare_runtime_json": "step:indexshare_runtime/evidence_json",
                "synthetic_generation_json": "step:synthetic_generation/evidence_json",
                "non_vq_evidence_json": "step:non_vq_pack/evidence_json",
                "composite_artifact_audit_json": "step:composite_audit/evidence_json",
                "production_generation_json": "step:production_generation/evidence_json",
                "teacher_cache_root": "step:teacher_cache/artifact",
                "teacher_cache_manifest_json": "step:teacher_cache/manifest",
                "candidate_cache_root": "step:candidate_eval_produce/artifact",
                "source_route_trace_root": "step:teacher_cache/route_trace_root",
                "candidate_route_trace_root": "step:candidate_route_capture/artifact",
                "source_route_authority_json": "external:source_route_authority_json",
                "candidate_route_authority_json": "external:candidate_route_authority_json",
                "benchmark_evidence_json": "step:same_machine_benchmark_compare/evidence_json",
            },
        },
    ]

    return {
        "schema_version": 1,
        "name": f"{profile.name}-{options.quality}",
        "description": (
            "Raw-HF GLM-5.2-REAP source-to-release evidence chain. Heavy model "
            "operations are only run by keep build without --dry-run."
        ),
        "build_root": options.output,
        "external_inputs": {
            "source_dir": {"kind": "artifact_dir", "path": source_dir},
            "profile_path": {"kind": "file", "path": f"models/{profile.name}.yaml"},
            "config_path": {"kind": "file", "path": str(Path(source_dir) / "config.json")},
            "index_path": {"kind": "file", "path": str(Path(source_dir) / "model.safetensors.index.json")},
            "teacher_artifact_identities_json": {"kind": "file", "path": release["teacher_artifact_identities_json"]},
            "teacher_ledger_path": {"kind": "file", "path": release["teacher_ledger_path"]},
            "teacher_checkpoint_dir": {"kind": "artifact_dir", "path": release["teacher_checkpoint_dir"]},
            "candidate_route_input_root": {"kind": "artifact_dir", "path": release["candidate_route_input_root"]},
            "source_route_authority_json": {"kind": "file", "path": release["source_route_authority_json"]},
            "candidate_route_authority_json": {"kind": "file", "path": release["candidate_route_authority_json"]},
            "benchmark_prompt_token_ids_json": {"kind": "file", "path": release["benchmark_prompt_token_ids_json"]},
            "benchmark_source_contract_json": {"kind": "file", "path": release["benchmark_source_contract_json"]},
        },
        "steps": steps,
    }


TEMPLATES: dict[str, Callable[[ModelProfile, HighLevelOptions], dict[str, Any]]] = {
    "glm_stream": _glm45_air_template,
    "qwen_moe_groups": _qwen36_template,
    "glm52_vq_groups": _glm52_reap_template,
}
