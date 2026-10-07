"""Op registry: thin argv builders over the existing deterministic scripts.

Every op runs its primitive as a subprocess with a fully resolved argv —
never an imported ``main()`` — so Lane S memory-quiet measurement discipline
holds (fresh process per benchmark) and every ledger entry is copy-paste
reproducible. The runner never passes ``--allow-existing``/``--overwrite``:
output freshness comes from content-key-named step directories, and the
primitives' refusal of existing dirs double-checks that invariant.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlx_vq.build.recipe import StepSpec

PYTHON_PREFIX = ("uv", "run", "python")

# How a param value renders into argv.
VALUE = "value"  # --flag <str(value)>
ARGS = "args"  # --flag <v1> <v2> ... (argparse nargs)
CSV = "csv"  # --flag v1,v2,...
SWITCH = "switch"  # --flag (present when truthy)


@dataclass
class StepContext:
    spec: StepSpec
    inputs: dict[str, Path]
    out_dir: Path
    evidence_dir: Path

    def evidence_path(self, name: str) -> Path:
        return self.evidence_dir / name


@dataclass(frozen=True)
class OpDef:
    name: str
    version: int
    allowed_classes: frozenset[str]
    script: str
    required_inputs: tuple[str, ...] = ()
    optional_inputs: tuple[str, ...] = ()
    # input name -> flag; inputs without a flag are gate-time context only
    # (e.g. the Lane S control evidence) and never appear in argv.
    input_flags: Mapping[str, str] = field(default_factory=dict)
    param_specs: Mapping[str, tuple[str, str]] = field(default_factory=dict)
    required_params: tuple[str, ...] = ()
    # evidence name -> flag used to route the file into the step's dir
    evidence_flags: Mapping[str, str] = field(default_factory=dict)
    # output name -> relative file under the step output dir
    file_outputs: Mapping[str, str] = field(default_factory=dict)
    # Named output directories which also receive an explicit CLI flag.  These
    # remain file_outputs so the executor verifies their existence without
    # applying this op's artifact-manifest convention to a sibling artifact.
    output_path_flags: Mapping[str, str] = field(default_factory=dict)
    output_dir_flag: str | None = None
    produces_artifact: bool = False
    # Heavy artifact producers that acquire the repository lock themselves
    # must not be wrapped in the executor's copy of the same exclusive lock.
    self_managed_heavy_lock: bool = False
    # Manifest file that proves an artifact output is complete. GLM-family
    # converters write conversion-manifest.json; other families differ (the
    # Qwen MoE materializer writes qwen-moe-materialization-manifest.json).
    artifact_manifest_name: str = "conversion-manifest.json"
    extra_argv: tuple[str, ...] = ()
    output_namer: Callable[[Mapping[str, Any]], tuple[str, ...]] | None = None
    # Cheap live-state diagnostics can opt out of ledger resume so external
    # payload disappearance/corruption is rechecked on every build invocation.
    cacheable: bool = True
    # Group-wise materializers can preserve complete atomically published
    # checkpoints in an interrupted step directory and resume with validation.
    resume_partial: bool = False
    # Strict artifact writers can own recognition/removal of their exact atomic
    # temp filenames; the executor must not glob-delete unrelated entries first.
    manages_partial_files: bool = False
    # Some evidence producers use a nonzero code for a valid blocked verdict.
    # Such codes are only accepted when the recipe declares a structured gate.
    accepted_returncodes: frozenset[int] = frozenset({0})
    # Evidence-specific ops may require their matching evaluator rather than
    # merely any syntactically valid gate profile.
    required_gate_profile: str | None = None
    # Threshold-only gates may be promoted by --regate. Composite release
    # gates whose evidence contract itself must change opt out.
    regate_can_complete: bool = True

    @property
    def allowed_params(self) -> frozenset[str]:
        return frozenset(self.param_specs)

    def output_names(self, params: Mapping[str, Any]) -> tuple[str, ...]:
        if self.output_namer is not None:
            return self.output_namer(params)
        names: list[str] = []
        if self.produces_artifact:
            names.append("artifact")
        names.extend(self.file_outputs)
        names.extend(self.evidence_flags)
        return tuple(names)

    def output_paths(self, ctx: StepContext) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for name in self.output_names(ctx.spec.params):
            if name in self.evidence_flags:
                paths[name] = ctx.evidence_path(_evidence_filename(name))
            elif name in self.file_outputs:
                paths[name] = ctx.out_dir / self.file_outputs[name]
            elif name.startswith("artifact_bpw"):
                paths[name] = ctx.out_dir / _sweep_candidate_dirname(
                    ctx.spec.params, name
                )
            else:
                # "artifact", "imatrix": the step's own output directory.
                paths[name] = ctx.out_dir
        return paths

    def build_argv(self, ctx: StepContext) -> list[str]:
        argv: list[str] = [*PYTHON_PREFIX, self.script, *self.extra_argv]
        for input_name in (*self.required_inputs, *self.optional_inputs):
            flag = self.input_flags.get(input_name)
            if flag is None or input_name not in ctx.inputs:
                continue
            argv.extend([flag, str(ctx.inputs[input_name])])
        if self.output_dir_flag is not None:
            argv.extend([self.output_dir_flag, str(ctx.out_dir)])
        for name, flag in self.evidence_flags.items():
            argv.extend([flag, str(ctx.evidence_path(_evidence_filename(name)))])
        for name, flag in self.output_path_flags.items():
            argv.extend([flag, str(ctx.out_dir / self.file_outputs[name])])
        for param_name in sorted(ctx.spec.params):
            value = ctx.spec.params[param_name]
            flag, kind = self.param_specs[param_name]
            if kind == SWITCH:
                if value:
                    argv.append(flag)
            elif kind == ARGS:
                argv.append(flag)
                argv.extend(str(item) for item in _as_sequence(value, param_name))
            elif kind == CSV:
                argv.extend(
                    [flag, ",".join(str(item) for item in _as_sequence(value, param_name))]
                )
            else:
                if kind == VALUE and isinstance(value, (list, tuple)):
                    # repeatable value flags (e.g. --budget)
                    for item in value:
                        argv.extend([flag, str(item)])
                else:
                    argv.extend([flag, str(value)])
        return argv


def _as_sequence(value: Any, param_name: str) -> Sequence[Any]:
    if isinstance(value, (list, tuple)):
        return value
    raise ValueError(f"param {param_name!r} expects a list")


def derive_implicit_inputs(op_name: str, inputs: dict[str, Path]) -> dict[str, Path]:
    """Fill inputs that follow mechanically from declared ones.

    Teacher-cache logit shards live beside their ``metadata.jsonl``, so the
    trainer's cache roots default to the metadata file's parent directory.
    """

    derived = dict(inputs)
    if op_name == "train-low-rank":
        for teacher, cache_root in (
            ("selection_teacher", "selection_cache_root"),
            ("validation_teacher", "validation_cache_root"),
        ):
            if cache_root not in derived and teacher in derived:
                derived[cache_root] = derived[teacher].parent
    if op_name == "fit-block-local-sidecar":
        for index in range(1, 5):
            teacher = f"teacher_{index}"
            cache_root = f"teacher_cache_root_{index}"
            if cache_root not in derived and teacher in derived:
                derived[cache_root] = derived[teacher].parent
    return derived


def _evidence_filename(name: str) -> str:
    suffix = ".json" if name.endswith(("_json", "attribution", "gap", "rows")) else ".jsonl"
    return f"{name}{suffix}"


def _sweep_candidate_dirname(params: Mapping[str, Any], output_name: str) -> str:
    # output name artifact_bpw2p0 -> <candidate_prefix>-bpw-2p0 (matches
    # materialize_glm45_air_dynamic_imatrix_sweep.py's candidate naming).
    prefix = str(params.get("candidate_prefix", "air-dynamic-imatrix"))
    tag = output_name.removeprefix("artifact_bpw")
    return f"{prefix}-bpw-{tag}"


def _sweep_output_names(params: Mapping[str, Any]) -> tuple[str, ...]:
    budgets = params.get("budget") or []
    if not isinstance(budgets, (list, tuple)):
        budgets = [budgets]
    return tuple(
        "artifact_bpw" + f"{float(budget):.1f}".replace(".", "p") for budget in budgets
    )


def _pipeline_source_view_output_names(params: Mapping[str, Any]) -> tuple[str, ...]:
    names = ["artifact"]
    if params.get("local_sequential_stage_views"):
        names.append("stage_view_roots_json")
    return tuple(names)


REGISTRY: dict[str, OpDef] = {}


def _register(op: OpDef) -> OpDef:
    REGISTRY[op.name] = op
    return op


_register(
    OpDef(
        name="stream-convert-vq",
        version=1,
        allowed_classes=frozenset({"promotable", "diagnostic"}),
        script="scripts/plan_stream_convert.py",
        optional_inputs=("source_dir", "config_path", "index_path"),
        input_flags={
            "source_dir": "--source-dir",
            "config_path": "--config-path",
            "index_path": "--index-path",
        },
        output_dir_flag="--output-dir",
        produces_artifact=True,
        extra_argv=("--convert-all",),
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
            "group_size": ("--group-size", VALUE),
            "code_bits": ("--code-bits", VALUE),
            "code_bits_policy": ("--code-bits-policy", VALUE),
            "scale_estimator": ("--scale-estimator", VALUE),
            "max_groups": ("--max-groups", VALUE),
            "download_missing_source_shards": (
                "--download-missing-source-shards",
                SWITCH,
            ),
            "download_dry_run": ("--download-dry-run", SWITCH),
            "max_download_shards": ("--max-download-shards", VALUE),
            "download_workers": ("--download-workers", VALUE),
            "expert_workers": ("--expert-workers", VALUE),
        },
        required_params=("model_id",),
    )
)

_register(
    OpDef(
        name="export-teacher-cache-cleanroom",
        version=1,
        allowed_classes=frozenset({"promotable", "diagnostic"}),
        script="benchmarks/export_glm45_air_cleanroom_cache.py",
        optional_inputs=(
            "rank_view_dir",
            "jaccl_hostfile",
            "local_sequential_stage_view_roots_json",
        ),
        input_flags={
            "rank_view_dir": "--rank-view-dir",
            "jaccl_hostfile": "--jaccl-hostfile",
            "local_sequential_stage_view_roots_json": (
                "--local-sequential-stage-view-roots-json"
            ),
        },
        output_dir_flag="--output-dir",
        file_outputs={"teacher_cache": "metadata.jsonl"},
        evidence_flags={
            "validation": "--validation-jsonl",
            "local_eval": "--local-jsonl",
        },
        param_specs={
            "jaccl_hostfile": ("--jaccl-hostfile", VALUE),
            "peer_ssh": ("--peer-ssh", VALUE),
            "peer_repo": ("--peer-repo", VALUE),
            "peer_rank0_view_root": ("--peer-rank0-view-root", VALUE),
            "peer_rank1_view_root": ("--peer-rank1-view-root", VALUE),
            "peer_source_dir": ("--peer-source-dir", VALUE),
            "peer_source_stage_ssh": ("--peer-source-stage-ssh", VALUE),
            "stage_peer_source": ("--stage-peer-source", SWITCH),
            "peer_source_min_free_gb": ("--peer-source-min-free-gb", VALUE),
            "peer_uv": ("--peer-uv", VALUE),
            "known_hosts": ("--known-hosts", VALUE),
            "tmp_src_root": ("--tmp-src-root", VALUE),
            "tmp_src_archive": ("--tmp-src-archive", VALUE),
            "rank_view_roots_json": ("--rank-view-roots-json", VALUE),
            "prompt_set": ("--prompt-set", VALUE),
            "prompt_id": ("--prompt-id", VALUE),
            "top_k": ("--top-k", VALUE),
            "max_positions": ("--max-positions", VALUE),
            "layer_split": ("--layer-split", VALUE),
            "lm_head_chunk_rows": ("--lm-head-chunk-rows", VALUE),
            "mlx_wired_limit_gb": ("--mlx-wired-limit-gb", VALUE),
            "mlx_cache_limit_gb": ("--mlx-cache-limit-gb", VALUE),
            "required_wired_mb": ("--required-wired-mb", VALUE),
            "local_direct_if": ("--local-direct-if", VALUE),
            "peer_direct_if": ("--peer-direct-if", VALUE),
            "local_direct_ip": ("--local-direct-ip", VALUE),
            "peer_direct_ip": ("--peer-direct-ip", VALUE),
            "skip_direct_link_check": ("--skip-direct-link-check", SWITCH),
            "preflight_only": ("--preflight-only", SWITCH),
            "no_full_logits": ("--no-full-logits", SWITCH),
            "include_route_trace": ("--include-route-trace", SWITCH),
            "route_trace_only": ("--route-trace-only", SWITCH),
            "route_trace_layer": ("--route-trace-layer", VALUE),
            "skip_local_consume": ("--skip-local-consume", SWITCH),
            "allow_dirty_cache": ("--allow-dirty-cache", SWITCH),
            "memory_quiet_preflight": ("--memory-quiet-preflight", SWITCH),
            "require_memory_quiet_preflight": (
                "--require-memory-quiet-preflight",
                SWITCH,
            ),
            "memory_quiet_seconds": ("--memory-quiet-seconds", VALUE),
            "memory_quiet_max_attempts": ("--memory-quiet-max-attempts", VALUE),
            "memory_quiet_min_free_gb": ("--memory-quiet-min-free-gb", VALUE),
            "memory_quiet_stage_view_margin_gb": (
                "--memory-quiet-stage-view-margin-gb",
                VALUE,
            ),
            "single_host_fallback": ("--single-host-fallback", SWITCH),
            "single_host_model_path": ("--single-host-model-path", VALUE),
            "single_host_revision": ("--single-host-revision", VALUE),
            "single_host_teacher_kind": ("--single-host-teacher-kind", VALUE),
            "single_host_lazy": ("--single-host-lazy", SWITCH),
            "single_host_pipeline_local": ("--single-host-pipeline-local", SWITCH),
            "single_host_pipeline_local_stage_processes": (
                "--single-host-pipeline-local-stage-processes",
                SWITCH,
            ),
            "single_host_pipeline_local_head_process": (
                "--single-host-pipeline-local-head-process",
                SWITCH,
            ),
            "single_host_pipeline_local_lower_split_layer": (
                "--single-host-pipeline-local-lower-split-layer",
                VALUE,
            ),
            "single_host_pipeline_local_upper_split_layer": (
                "--single-host-pipeline-local-upper-split-layer",
                VALUE,
            ),
            "single_host_pipeline_local_lower_split_layers": (
                "--single-host-pipeline-local-lower-split-layers",
                CSV,
            ),
            "single_host_pipeline_local_upper_split_layers": (
                "--single-host-pipeline-local-upper-split-layers",
                CSV,
            ),
            "single_host_pipeline_local_abort_on_dirty_stage": (
                "--single-host-pipeline-local-abort-on-dirty-stage",
                SWITCH,
            ),
            "local_sequential_remote_workers": (
                "--local-sequential-remote-workers",
                VALUE,
            ),
            "local_sequential_remote_tmp_dir": (
                "--local-sequential-remote-tmp-dir",
                VALUE,
            ),
            "local_sequential_remote_dirty_retries": (
                "--local-sequential-remote-dirty-retries",
                VALUE,
            ),
            "local_sequential_remote_retry_sleep_seconds": (
                "--local-sequential-remote-retry-sleep-seconds",
                VALUE,
            ),
            "single_host_mlx_wired_limit_gb": (
                "--single-host-mlx-wired-limit-gb",
                VALUE,
            ),
            "single_host_source_memory_guard_ratio": (
                "--single-host-source-memory-guard-ratio",
                VALUE,
            ),
            "vq_artifact_dir": ("--vq-artifact-dir", VALUE),
            "engine": ("--engine", VALUE),
        },
    )
)

_register(
    OpDef(
        name="collect-route-records",
        version=1,
        allowed_classes=frozenset({"promotable", "diagnostic"}),
        script="benchmarks/collect_glm45_air_route_records.py",
        required_inputs=("artifact",),
        input_flags={"artifact": "--artifact-dir"},
        evidence_flags={"route_records": "--output-jsonl"},
        param_specs={
            "prompt_set": ("--prompt-set", VALUE),
            "prompt_id": ("--prompt-id", VALUE),
            "max_prompts": ("--max-prompts", VALUE),
            "layers": ("--layers", CSV),
        },
    )
)

_register(
    OpDef(
        name="select-ebss-prompts",
        version=1,
        allowed_classes=frozenset({"promotable", "diagnostic"}),
        script="benchmarks/select_glm45_air_ebss_prompts.py",
        required_inputs=("route_records",),
        input_flags={"route_records": "--records-jsonl"},
        evidence_flags={"selection_json": "--out-json"},
        param_specs={"max_prompts": ("--max-prompts", VALUE)},
        required_params=("max_prompts",),
    )
)

_register(
    OpDef(
        name="collect-imatrix",
        version=1,
        allowed_classes=frozenset({"promotable", "diagnostic"}),
        script="benchmarks/collect_glm45_air_imatrix.py",
        required_inputs=("artifact",),
        optional_inputs=("prompt_selection",),
        input_flags={
            "artifact": "--artifact-dir",
            "prompt_selection": "--prompt-selection-json",
        },
        output_dir_flag="--output-dir",
        produces_artifact=False,
        file_outputs={"imatrix_manifest": "imatrix-manifest.json"},
        param_specs={
            "prompt_set": ("--prompt-set", VALUE),
            "prompt_id": ("--prompt-id", VALUE),
            "max_prompts": ("--max-prompts", VALUE),
            "layers": ("--layers", CSV),
            "projections": ("--projections", CSV),
        },
    )
)

_register(
    OpDef(
        name="materialize-sweep",
        version=1,
        allowed_classes=frozenset({"promotable", "diagnostic"}),
        script="benchmarks/materialize_glm45_air_dynamic_imatrix_sweep.py",
        required_inputs=("imatrix_manifest", "baseline_artifact"),
        optional_inputs=("high_bit_artifact",),
        input_flags={
            "imatrix_manifest": "--imatrix-manifest",
            "baseline_artifact": "--baseline-artifact-dir",
            "high_bit_artifact": "--high-bit-artifact-dir",
        },
        output_dir_flag="--output-root",
        param_specs={
            "candidate_prefix": ("--candidate-prefix", VALUE),
            "budget": ("--budget", VALUE),
            "mid_error_factor": ("--mid-error-factor", VALUE),
            "reround_top_low_experts": ("--reround-top-low-experts", VALUE),
            "importance_key": ("--importance-key", VALUE),
        },
        required_params=("budget",),
        output_namer=_sweep_output_names,
    )
)

_register(
    OpDef(
        name="materialize-learned-rotation",
        version=1,
        allowed_classes=frozenset({"promotable", "diagnostic"}),
        script="benchmarks/materialize_glm45_air_learned_rotation.py",
        required_inputs=(
            "source_dir",
            "config_path",
            "index_path",
            "seed_artifact",
            "rotation_manifest",
        ),
        input_flags={
            "source_dir": "--source-dir",
            "config_path": "--config-path",
            "index_path": "--index-path",
            "seed_artifact": "--seed-artifact-dir",
            "rotation_manifest": "--rotation-manifest",
        },
        output_dir_flag="--output-dir",
        produces_artifact=True,
        param_specs={
            "model_id": ("--model-id", VALUE),
            "group_size": ("--group-size", VALUE),
            "code_bits": ("--code-bits", VALUE),
            "scale_estimator": ("--scale-estimator", VALUE),
            "expert_workers": ("--expert-workers", VALUE),
            "no_strict_config": ("--no-strict-config", SWITCH),
        },
    )
)

_register(
    OpDef(
        name="train-learned-rotation",
        version=1,
        allowed_classes=frozenset({"promotable", "diagnostic"}),
        script="benchmarks/train_glm45_air_learned_rotation.py",
        required_inputs=("source_dir", "config_path", "index_path"),
        input_flags={
            "source_dir": "--source-dir",
            "config_path": "--config-path",
            "index_path": "--index-path",
        },
        output_dir_flag="--output-dir",
        file_outputs={"rotation_manifest": "rotation-manifest.json"},
        evidence_flags={"training_evidence": "--evidence-jsonl"},
        param_specs={
            "model_id": ("--model-id", VALUE),
            "group_size": ("--group-size", VALUE),
            "code_bits": ("--code-bits", VALUE),
            "target": ("--target", VALUE),
            "steps": ("--steps", VALUE),
            "learning_rate": ("--learning-rate", VALUE),
            "max_experts": ("--max-experts", VALUE),
            "max_rows_per_expert": ("--max-rows-per-expert", VALUE),
            "sample_seed": ("--sample-seed", VALUE),
            "init": ("--init", VALUE),
            "structured_rht_signs": ("--structured-rht-signs", SWITCH),
            "sign_candidates_per_step": ("--sign-candidates-per-step", VALUE),
            "rht_seed_prefix": ("--rht-seed-prefix", VALUE),
        },
        required_params=("target",),
    )
)

_register(
    OpDef(
        name="train-low-rank",
        version=1,
        allowed_classes=frozenset({"promotable", "diagnostic"}),
        script="benchmarks/finetune_glm45_air_vq_continuous.py",
        required_inputs=(
            "seed_artifact",
            "selection_teacher",
            "validation_teacher",
        ),
        optional_inputs=("selection_cache_root", "validation_cache_root"),
        input_flags={
            "seed_artifact": "--seed-artifact-dir",
            "selection_teacher": "--selection-teacher-jsonl",
            "selection_cache_root": "--selection-cache-root",
            "validation_teacher": "--validation-teacher-jsonl",
            "validation_cache_root": "--validation-cache-root",
        },
        output_dir_flag="--output-dir",
        produces_artifact=True,
        evidence_flags={"train_log": "--append-jsonl"},
        param_specs={
            "prefill_engine": ("--prefill-engine", VALUE),
            "layer": ("--layer", VALUE),
            "projection": ("--projection", VALUE),
            "projections": ("--projections", ARGS),
            "trainable": ("--trainable", VALUE),
            "low_rank": ("--low-rank", VALUE),
            "low_rank_init_scale": ("--low-rank-init-scale", VALUE),
            "steps": ("--steps", VALUE),
            "learning_rate": ("--learning-rate", VALUE),
            "target_nll_weight": ("--target-nll-weight", VALUE),
            "teacher_top1_margin_weight": ("--teacher-top1-margin-weight", VALUE),
            "teacher_top1_margin": ("--teacher-top1-margin", VALUE),
            "tail_kld_weight": ("--tail-kld-weight", VALUE),
            "grad_clip_norm": ("--grad-clip-norm", VALUE),
            "loss_scope": ("--loss-scope", VALUE),
            "surrogate_projections": ("--surrogate-projections", ARGS),
            "surrogate_output_chunk_size": ("--surrogate-output-chunk-size", VALUE),
            "train_cache": ("--train-cache", VALUE),
            "max_train_rows": ("--max-train-rows", VALUE),
            "train_row_indices": ("--train-row-indices", CSV),
            "max_positions": ("--max-positions", VALUE),
            "aux_loss_position_indices": ("--aux-loss-position-indices", CSV),
            "min_top_k": ("--min-top-k", VALUE),
        },
    )
)

_FIT_TEACHER_INPUTS = tuple(f"teacher_{index}" for index in range(1, 5))
_FIT_TEACHER_CACHE_ROOT_INPUTS = tuple(
    f"teacher_cache_root_{index}" for index in range(1, 5)
)

_register(
    OpDef(
        name="fit-block-local-sidecar",
        version=1,
        allowed_classes=frozenset({"promotable", "diagnostic"}),
        script="benchmarks/fit_glm45_air_block_local_sidecar.py",
        required_inputs=("seed_artifact", "teacher_1"),
        optional_inputs=(
            "source_dir",
            "config_path",
            "index_path",
            *_FIT_TEACHER_INPUTS[1:],
            *_FIT_TEACHER_CACHE_ROOT_INPUTS,
        ),
        input_flags={
            "seed_artifact": "--seed-artifact-dir",
            "source_dir": "--source-dir",
            "config_path": "--config-path",
            "index_path": "--index-path",
            **{name: "--teacher-jsonl" for name in _FIT_TEACHER_INPUTS},
            **{name: "--teacher-cache-root" for name in _FIT_TEACHER_CACHE_ROOT_INPUTS},
        },
        output_dir_flag="--output-dir",
        produces_artifact=True,
        evidence_flags={"fit_log": "--append-jsonl"},
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
            "layer": ("--layer", VALUE),
            "projection": ("--projection", VALUE),
            "trainable": ("--trainable", VALUE),
            "low_rank": ("--low-rank", VALUE),
            "max_train_rows": ("--max-train-rows", VALUE),
            "row_indices": ("--row-indices", CSV),
            "max_positions": ("--max-positions", VALUE),
            "min_top_k": ("--min-top-k", VALUE),
            "allow_dirty_cache": ("--allow-dirty-cache", SWITCH),
        },
        required_params=("layer", "projection"),
    )
)

_register(
    OpDef(
        name="scale-block-local-sidecars",
        version=1,
        allowed_classes=frozenset({"promotable", "diagnostic"}),
        script="benchmarks/materialize_glm45_air_scaled_sidecars.py",
        required_inputs=("seed_artifact", "source_artifact"),
        input_flags={
            "seed_artifact": "--seed-artifact-dir",
            "source_artifact": "--source-artifact-dir",
        },
        output_dir_flag="--output-dir",
        produces_artifact=True,
        evidence_flags={"scale_log": "--append-jsonl"},
        param_specs={
            "scale_policy_json": ("--scale-policy-json", VALUE),
        },
        required_params=("scale_policy_json",),
    )
)

_MERGE_SOURCE_INPUTS = tuple(f"source_artifact_{index}" for index in range(1, 9))

_register(
    OpDef(
        name="merge-block-local-sidecars",
        version=1,
        allowed_classes=frozenset({"promotable", "diagnostic"}),
        script="benchmarks/materialize_glm45_air_merged_sidecars.py",
        required_inputs=("seed_artifact", "source_artifact_1"),
        optional_inputs=_MERGE_SOURCE_INPUTS[1:],
        input_flags={
            "seed_artifact": "--seed-artifact-dir",
            **{name: "--source-artifact-dir" for name in _MERGE_SOURCE_INPUTS},
        },
        output_dir_flag="--output-dir",
        produces_artifact=True,
        evidence_flags={"merge_log": "--append-jsonl"},
        param_specs={
            "train_split": ("--train-split", VALUE),
            "train_row_count": ("--train-row-count", VALUE),
            "max_positions": ("--max-positions", VALUE),
            "trainable": ("--trainable", VALUE),
            "teacher_jsonl": ("--teacher-jsonl", VALUE),
            "teacher_cache_root": ("--teacher-cache-root", VALUE),
        },
    )
)

_register(
    OpDef(
        name="eval",
        version=1,
        allowed_classes=frozenset({"verify", "diagnostic"}),
        script="benchmarks/eval_glm45_air_teacher_cache.py",
        required_inputs=("artifact", "teacher"),
        optional_inputs=("cache_root",),
        input_flags={
            "artifact": "--artifact-dir",
            "teacher": "--teacher-jsonl",
            "cache_root": "--cache-root",
        },
        evidence_flags={"evidence": "--append-jsonl"},
        param_specs={
            "engine": ("--engine", VALUE),
            "max_rows": ("--max-rows", VALUE),
            "row_indices": ("--row-indices", CSV),
            "min_top_k": ("--min-top-k", VALUE),
            "clear_mlx_cache_between_rows": ("--clear-mlx-cache-between-rows", SWITCH),
            "mlx_cache_limit_gb": ("--mlx-cache-limit-gb", VALUE),
            "mlx_memory_limit_gb": ("--mlx-memory-limit-gb", VALUE),
            "mlx_wired_limit_gb": ("--mlx-wired-limit-gb", VALUE),
            "mlx_clear_cache_before_load": ("--mlx-clear-cache-before-load", SWITCH),
            "truncate_input_to_selected_positions": (
                "--truncate-input-to-selected-positions",
                SWITCH,
            ),
        },
        required_params=("engine",),
    )
)

_register(
    OpDef(
        name="eval-repair",
        version=1,
        allowed_classes=frozenset({"verify", "diagnostic"}),
        script="benchmarks/repair_glm45_air_eval_evidence.py",
        required_inputs=("artifact", "teacher", "existing_evidence"),
        optional_inputs=("cache_root",),
        input_flags={
            "artifact": "--artifact-dir",
            "teacher": "--teacher-jsonl",
            "existing_evidence": "--existing-jsonl",
            "cache_root": "--cache-root",
        },
        evidence_flags={"evidence": "--append-jsonl"},
        param_specs={
            "engine": ("--engine", VALUE),
            "max_rows": ("--max-rows", VALUE),
            "row_indices": ("--row-indices", CSV),
            "repair_attempts": ("--repair-attempts", VALUE),
            "eval_script": ("--eval-script", VALUE),
            "min_top_k": ("--min-top-k", VALUE),
            "clear_mlx_cache_between_rows": ("--clear-mlx-cache-between-rows", SWITCH),
            "mlx_cache_limit_gb": ("--mlx-cache-limit-gb", VALUE),
            "mlx_memory_limit_gb": ("--mlx-memory-limit-gb", VALUE),
            "mlx_wired_limit_gb": ("--mlx-wired-limit-gb", VALUE),
            "mlx_clear_cache_before_load": ("--mlx-clear-cache-before-load", SWITCH),
            "truncate_input_to_selected_positions": (
                "--truncate-input-to-selected-positions",
                SWITCH,
            ),
        },
        required_params=("engine",),
    )
)

_register(
    OpDef(
        name="benchmark-lane-s",
        version=1,
        allowed_classes=frozenset({"verify", "diagnostic"}),
        script="benchmarks/bench_glm45_air_quant_compare.py",
        optional_inputs=("artifact", "control_evidence"),
        input_flags={"artifact": "--artifact-dir"},
        evidence_flags={"evidence": "--append-jsonl"},
        param_specs={
            "engine": ("--engine", VALUE),
            "scenario": ("--scenario", VALUE),
            "repetitions": ("--repetitions", VALUE),
            "warmup_repetitions": ("--warmup-repetitions", VALUE),
            "timing_stability_max_relative_spread": (
                "--timing-stability-max-relative-spread",
                VALUE,
            ),
            "clean_repetition_max_attempts": ("--clean-repetition-max-attempts", VALUE),
            "memory_quiet_preflight": ("--memory-quiet-preflight", SWITCH),
            "require_memory_quiet_preflight": (
                "--require-memory-quiet-preflight",
                SWITCH,
            ),
            "memory_quiet_window_seconds": ("--memory-quiet-window-seconds", VALUE),
            "memory_quiet_max_attempts": ("--memory-quiet-max-attempts", VALUE),
            "prefill_memory_quiet_window_seconds": (
                "--prefill-memory-quiet-window-seconds",
                VALUE,
            ),
            "prefill_memory_quiet_max_attempts": (
                "--prefill-memory-quiet-max-attempts",
                VALUE,
            ),
            "prefill_chunk_size": ("--prefill-chunk-size", VALUE),
            "parent_vm_stat_diagnostics": ("--parent-vm-stat-diagnostics", SWITCH),
            "memory_phase_trace": ("--memory-phase-trace", SWITCH),
            "mlx_cache_limit_gb": ("--mlx-cache-limit-gb", VALUE),
            "mlx_clear_cache_before_run": ("--mlx-clear-cache-before-run", SWITCH),
        },
        required_params=("engine", "scenario"),
    )
)

_register(
    OpDef(
        name="audit",
        version=1,
        allowed_classes=frozenset({"verify"}),
        script="src/mlx_vq/build/audit_cli.py",
        required_inputs=("artifact",),
        input_flags={"artifact": "--artifact-dir"},
        evidence_flags={"evidence": "--output-json"},
    )
)

_register(
    OpDef(
        name="resident-byte-audit",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/audit_glm45_air_resident_bytes.py",
        required_inputs=("artifact",),
        optional_inputs=(
            "source_dir",
            "config_path",
            "index_path",
            "baseline_evidence",
        ),
        input_flags={
            "artifact": "--artifact-dir",
            "source_dir": "--source-dir",
            "config_path": "--config-path",
            "index_path": "--index-path",
            "baseline_evidence": "--baseline-json",
        },
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
            "max_counted_resident_total_bytes": (
                "--max-counted-resident-total-bytes",
                VALUE,
            ),
            "min_total_byte_reduction": ("--min-total-byte-reduction", VALUE),
        },
    )
)

_register(
    OpDef(
        name="qwen-moe-source-audit",
        version=3,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/inspect_qwen_moe_source.py",
        optional_inputs=("config_path", "index_path"),
        input_flags={
            "config_path": "--config-path",
            "index_path": "--index-path",
        },
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
            "expected_model_type": ("--expected-model-type", VALUE),
            "expected_language_layers": ("--expected-language-layers", VALUE),
            "expected_expert_tensors": ("--expected-expert-tensors", VALUE),
            "verify_hf_shapes": ("--verify-hf-shapes", SWITCH),
        },
        required_params=("model_id", "revision"),
    )
)

_register(
    OpDef(
        name="glm52-source-audit",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/inspect_glm52_reap_source.py",
        required_inputs=("profile_path", "config_path", "index_path"),
        input_flags={
            "profile_path": "--profile-path",
            "config_path": "--config-path",
            "index_path": "--index-path",
        },
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
        },
        required_params=("model_id", "revision"),
    )
)

_register(
    OpDef(
        name="glm52-source-payload-audit",
        version=2,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/audit_glm52_reap_source_payloads.py",
        required_inputs=(
            "source_dir",
            "profile_path",
            "config_path",
            "index_path",
            "source_audit_evidence",
        ),
        input_flags={
            "source_dir": "--source-dir",
            "profile_path": "--profile-path",
            "config_path": "--config-path",
            "index_path": "--index-path",
        },
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
            "max_groups": ("--max-groups", VALUE),
            "groups": ("--groups", CSV),
        },
        required_params=("model_id", "revision"),
        cacheable=False,
    )
)

_register(
    OpDef(
        name="glm52-moe-materialize-groups",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/materialize_glm52_reap_groups.py",
        extra_argv=("--skip-existing",),
        required_inputs=(
            "source_dir",
            "profile_path",
            "config_path",
            "index_path",
            "payload_audit_evidence",
        ),
        input_flags={
            "source_dir": "--source-dir",
            "profile_path": "--profile-path",
            "config_path": "--config-path",
            "index_path": "--index-path",
        },
        output_dir_flag="--output-dir",
        produces_artifact=True,
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
            "groups": ("--groups", CSV),
            "max_groups": ("--max-groups", VALUE),
            "all_groups": ("--all-groups", SWITCH),
            "code_bits": ("--code-bits", VALUE),
            "group_size": ("--group-size", VALUE),
            "scale_estimator": ("--scale-estimator", VALUE),
            "expert_workers": ("--expert-workers", VALUE),
        },
        required_params=("model_id", "revision"),
        resume_partial=True,
    )
)

_register(
    OpDef(
        name="glm52-non-vq-pack",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="src/mlx_vq/convert/glm52_non_vq.py",
        extra_argv=("--resume",),
        required_inputs=(
            "source_dir",
            "profile_path",
            "config_path",
            "index_path",
            "payload_audit_evidence",
        ),
        input_flags={
            "source_dir": "--source-dir",
            "profile_path": "--profile-path",
            "config_path": "--config-path",
            "index_path": "--index-path",
        },
        output_dir_flag="--output-dir",
        produces_artifact=True,
        artifact_manifest_name="non-vq-manifest.json",
        file_outputs={
            "manifest": "non-vq-manifest.json",
            "index": "model.safetensors.index.json",
        },
        evidence_flags={"evidence_json": "--output-json"},
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
            "config_sha256": ("--config-sha256", VALUE),
            "index_sha256": ("--index-sha256", VALUE),
            "max_shard_payload_bytes": ("--max-shard-payload-bytes", VALUE),
        },
        required_params=(
            "model_id",
            "revision",
            "config_sha256",
            "index_sha256",
            "max_shard_payload_bytes",
        ),
        resume_partial=True,
        manages_partial_files=True,
    )
)

_register(
    OpDef(
        name="glm52-moe-artifact-audit",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/audit_glm52_reap_materialization.py",
        required_inputs=(
            "artifact",
            "source_dir",
            "profile_path",
            "config_path",
            "index_path",
            "materialization_runs",
        ),
        optional_inputs=("non_vq_artifact", "non_vq_evidence_json"),
        input_flags={
            "artifact": "--artifact-dir",
            "non_vq_artifact": "--non-vq-artifact-dir",
            "non_vq_evidence_json": "--non-vq-evidence-json",
            "source_dir": "--source-dir",
            "profile_path": "--profile-path",
            "config_path": "--config-path",
            "index_path": "--index-path",
            "materialization_runs": "--materialization-runs-jsonl",
        },
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
            "groups": ("--groups", CSV),
            "max_groups": ("--max-groups", VALUE),
            "all_groups": ("--all-groups", SWITCH),
            "code_bits": ("--code-bits", VALUE),
            "group_size": ("--group-size", VALUE),
            "scale_estimator": ("--scale-estimator", VALUE),
            "require_resume_proof": ("--require-resume-proof", SWITCH),
        },
        required_params=(
            "model_id",
            "revision",
            "code_bits",
            "group_size",
            "scale_estimator",
        ),
        cacheable=False,
    )
)

_register(
    OpDef(
        name="glm52-layer-forward-probe",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/probe_glm52_layer_forward.py",
        required_inputs=(
            "non_vq_artifact",
            "non_vq_evidence_json",
            "routed_artifact",
            "routed_audit_json",
            "profile_path",
            "config_path",
        ),
        input_flags={
            "non_vq_artifact": "--non-vq-artifact",
            "non_vq_evidence_json": "--non-vq-evidence-json",
            "routed_artifact": "--routed-artifact",
            "routed_audit_json": "--routed-audit-json",
            "profile_path": "--profile-path",
            "config_path": "--config-path",
        },
        evidence_flags={"evidence_json": "--output-json"},
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
            "layer": ("--layer", VALUE),
            "tokens": ("--tokens", VALUE),
            "seed": ("--seed", VALUE),
            "input_scale": ("--input-scale", VALUE),
        },
        required_params=(
            "model_id",
            "revision",
            "layer",
            "tokens",
            "seed",
            "input_scale",
        ),
        cacheable=False,
    )
)

_register(
    OpDef(
        name="glm52-indexshare-runtime-probe",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/probe_glm52_indexshare_runtime.py",
        required_inputs=("config_path", "index_path"),
        input_flags={
            "config_path": "--config-path",
            "index_path": "--index-path",
        },
        evidence_flags={"evidence_json": "--output-json"},
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
        },
        required_params=("model_id", "revision"),
        cacheable=False,
    )
)

_register(
    OpDef(
        name="glm52-tokenizer-readiness-probe",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/probe_glm52_tokenizer_readiness.py",
        required_inputs=("tokenizer_dir", "config_path"),
        input_flags={
            "tokenizer_dir": "--tokenizer-dir",
            "config_path": "--config-path",
        },
        evidence_flags={"evidence_json": "--output-json"},
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
            "prompt": ("--prompt", VALUE),
        },
        required_params=("model_id", "revision", "prompt"),
        cacheable=False,
    )
)

_register(
    OpDef(
        name="glm52-synthetic-generation-probe",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/probe_glm52_synthetic_generation.py",
        evidence_flags={"evidence_json": "--output-json"},
        cacheable=False,
    )
)

_register(
    OpDef(
        name="glm52-full-bind-preflight",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/probe_glm52_full_bind_preflight.py",
        required_inputs=(
            "profile_path",
            "config_path",
            "source_index_path",
            "non_vq_artifact",
            "routed_artifact",
        ),
        input_flags={
            "profile_path": "--profile-path",
            "config_path": "--config-path",
            "source_index_path": "--source-index-path",
            "non_vq_artifact": "--non-vq-artifact-dir",
            "routed_artifact": "--routed-artifact-dir",
        },
        evidence_flags={"evidence_json": "--output-json"},
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
        },
        required_params=("model_id", "revision"),
        cacheable=False,
    )
)

_register(
    OpDef(
        name="glm52-full-bind-evidence",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/probe_glm52_full_bind_preflight.py",
        required_inputs=(
            "profile_path",
            "config_path",
            "source_index_path",
            "non_vq_artifact",
            "routed_artifact",
        ),
        input_flags={
            "profile_path": "--profile-path",
            "config_path": "--config-path",
            "source_index_path": "--source-index-path",
            "non_vq_artifact": "--non-vq-artifact-dir",
            "routed_artifact": "--routed-artifact-dir",
        },
        evidence_flags={"evidence_json": "--output-json"},
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
        },
        required_params=("model_id", "revision"),
        extra_argv=("--evidence-only",),
        cacheable=False,
    )
)

_register(
    OpDef(
        name="glm52-production-generation-probe",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/probe_glm52_production_generation.py",
        required_inputs=(
            "profile_path",
            "config_path",
            "source_index_path",
            "tokenizer_dir",
            "tokenizer_readiness_json",
            "family_policy_json",
            "non_vq_artifact",
            "non_vq_evidence_json",
            "routed_artifact",
            "composite_audit_json",
            "materialization_runs",
            "full_bind_preflight_json",
        ),
        input_flags={
            "profile_path": "--profile-path",
            "config_path": "--config-path",
            "source_index_path": "--source-index-path",
            "tokenizer_dir": "--tokenizer-dir",
            "tokenizer_readiness_json": "--tokenizer-readiness-json",
            "family_policy_json": "--family-policy-json",
            "non_vq_artifact": "--non-vq-artifact-dir",
            "non_vq_evidence_json": "--non-vq-evidence-json",
            "routed_artifact": "--routed-artifact-dir",
            "composite_audit_json": "--composite-audit-json",
            "materialization_runs": "--materialization-runs-jsonl",
            "full_bind_preflight_json": "--full-bind-preflight-json",
        },
        evidence_flags={"evidence_json": "--output-json"},
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
            "prompt": ("--prompt", VALUE),
        },
        required_params=("model_id", "revision", "prompt"),
        cacheable=False,
    )
)

_register(
    OpDef(
        name="glm52-family-gate-policy",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/define_glm52_family_gate_policy.py",
        evidence_flags={"evidence_json": "--output-json"},
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
        },
        required_params=("model_id", "revision"),
        cacheable=False,
    )
)

_register(
    OpDef(
        name="glm52-family-eval-prompt-pack",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/prepare_glm52_family_eval_prompts.py",
        required_inputs=(
            "tokenizer_dir",
            "tokenizer_readiness_json",
            "family_policy_json",
        ),
        input_flags={
            "tokenizer_dir": "--tokenizer-dir",
            "tokenizer_readiness_json": "--tokenizer-readiness-json",
            "family_policy_json": "--family-policy-json",
        },
        evidence_flags={"evidence_json": "--output-json"},
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
        },
        required_params=("model_id", "revision"),
        cacheable=False,
    )
)

_register(
    OpDef(
        name="glm52-family-eval-teacher-metadata",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/prepare_glm52_family_eval_teacher_metadata.py",
        required_inputs=(
            "source_dir",
            "profile_path",
            "config_path",
            "index_path",
            "source_audit_json",
            "source_payload_audit_json",
            "tokenizer_readiness_json",
            "family_policy_json",
            "eval_prompt_pack_json",
        ),
        input_flags={
            "source_dir": "--source-dir",
            "profile_path": "--profile-path",
            "config_path": "--config-path",
            "index_path": "--index-path",
            "source_audit_json": "--source-audit-json",
            "source_payload_audit_json": "--source-payload-audit-json",
            "tokenizer_readiness_json": "--tokenizer-readiness-json",
            "family_policy_json": "--family-policy-json",
            "eval_prompt_pack_json": "--eval-prompt-pack-json",
        },
        output_dir_flag="--output-dir",
        file_outputs={
            "metadata_jsonl": "glm52_teacher_source_metadata.jsonl",
        },
        evidence_flags={"evidence_json": "--output-json"},
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
        },
        required_params=("model_id", "revision"),
        cacheable=False,
    )
)

_register(
    OpDef(
        name="glm52-teacher-cache-produce",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/produce_glm52_teacher_cache.py",
        required_inputs=(
            "snapshot_dir",
            "prompt_pack_json",
            "non_vq_package_dir",
            "artifact_identities_json",
            "profile_path",
            "ledger_path",
            "checkpoint_dir",
        ),
        input_flags={
            "snapshot_dir": "--snapshot-dir",
            "prompt_pack_json": "--prompt-pack-json",
            "non_vq_package_dir": "--non-vq-package-dir",
            "artifact_identities_json": "--artifact-identities-json",
            "profile_path": "--profile-path",
            "ledger_path": "--ledger-path",
            "checkpoint_dir": "--checkpoint-dir",
        },
        output_dir_flag="--cache-root",
        produces_artifact=True,
        self_managed_heavy_lock=True,
        artifact_manifest_name="glm52-teacher-cache-fp32-manifest.json",
        file_outputs={
            "manifest": "glm52-teacher-cache-fp32-manifest.json",
            # Must remain a sibling: the producer rejects trace roots inside
            # the immutable cache root.
            "route_trace_root": "../source-route-traces",
        },
        output_path_flags={"route_trace_root": "--route-trace-root"},
        param_specs={"limit_prompts": ("--limit-prompts", VALUE)},
        resume_partial=True,
        manages_partial_files=True,
    )
)

_register(
    OpDef(
        name="glm52-teacher-cache-audit",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/check_glm52_family_gate.py",
        required_inputs=(
            "teacher_cache_root",
            "teacher_cache_manifest_json",
            "eval_prompt_pack_json",
        ),
        input_flags={
            "teacher_cache_root": "--teacher-cache-root",
            "teacher_cache_manifest_json": "--teacher-cache-manifest-json",
            "eval_prompt_pack_json": "--eval-prompt-pack-json",
        },
        evidence_flags={"evidence_json": "--output-json"},
        extra_argv=("--audit-teacher-cache-only",),
        cacheable=False,
    )
)

_register(
    OpDef(
        name="glm52-candidate-full-vocab-eval-produce",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/eval_glm52_candidate_full_vocab.py",
        extra_argv=("produce",),
        required_inputs=(
            "profile_path",
            "config_path",
            "source_index_path",
            "tokenizer_dir",
            "tokenizer_readiness_json",
            "family_policy_json",
            "prompt_pack_json",
            "teacher_cache_root",
            "non_vq_artifact",
            "non_vq_evidence_json",
            "routed_artifact",
            "composite_audit_json",
            "materialization_runs",
            "full_bind_preflight_json",
            "ledger_path",
        ),
        input_flags={
            "profile_path": "--profile-path",
            "config_path": "--config-path",
            "source_index_path": "--source-index-path",
            "tokenizer_dir": "--tokenizer-dir",
            "tokenizer_readiness_json": "--tokenizer-readiness-json",
            "family_policy_json": "--family-policy-json",
            "prompt_pack_json": "--prompt-pack-json",
            "teacher_cache_root": "--teacher-cache-root",
            "non_vq_artifact": "--non-vq-artifact-dir",
            "non_vq_evidence_json": "--non-vq-evidence-json",
            "routed_artifact": "--routed-artifact-dir",
            "composite_audit_json": "--composite-audit-json",
            "materialization_runs": "--materialization-runs-jsonl",
            "full_bind_preflight_json": "--full-bind-preflight-json",
            "ledger_path": "--ledger-path",
        },
        output_dir_flag="--candidate-cache-root",
        produces_artifact=True,
        self_managed_heavy_lock=True,
        artifact_manifest_name="glm52-teacher-cache-fp32-manifest.json",
        resume_partial=True,
        manages_partial_files=True,
    )
)

_register(
    OpDef(
        name="glm52-candidate-full-vocab-eval-compare",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/eval_glm52_candidate_full_vocab.py",
        extra_argv=("compare",),
        required_inputs=(
            "teacher_cache_root",
            "candidate_cache_root",
            "prompt_pack_json",
            "family_policy_json",
        ),
        input_flags={
            "teacher_cache_root": "--teacher-cache-root",
            "candidate_cache_root": "--candidate-cache-root",
            "prompt_pack_json": "--prompt-pack-json",
            "family_policy_json": "--family-policy-json",
        },
        evidence_flags={"evidence_json": "--output-json"},
        cacheable=False,
    )
)

_register(
    OpDef(
        name="glm52-route-math-capture",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/check_glm52_route_math.py",
        extra_argv=("capture",),
        required_inputs=("input_root",),
        optional_inputs=("authority_json",),
        input_flags={
            "input_root": "--input-root",
            "authority_json": "--authority-json",
        },
        output_dir_flag="--trace-root",
        produces_artifact=True,
        artifact_manifest_name="glm52-route-trace-manifest.json",
        param_specs={
            "side": ("--side", VALUE),
            "routed_scaling_factor": ("--routed-scaling-factor", VALUE),
            "expert_count": ("--expert-count", VALUE),
        },
        required_params=("side",),
    )
)

_register(
    OpDef(
        name="glm52-route-math-compare",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/check_glm52_route_math.py",
        extra_argv=("compare",),
        required_inputs=(
            "source_trace_root",
            "candidate_trace_root",
            "source_authority_json",
            "candidate_authority_json",
            "family_policy_json",
        ),
        input_flags={
            "source_trace_root": "--source-trace-root",
            "candidate_trace_root": "--candidate-trace-root",
            "source_authority_json": "--source-authority-json",
            "candidate_authority_json": "--candidate-authority-json",
            "family_policy_json": "--family-policy-json",
        },
        evidence_flags={"evidence_json": "--evidence-json"},
        cacheable=False,
    )
)

_register(
    OpDef(
        name="glm52-same-machine-benchmark-pair",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/bench_glm52_same_machine_fp4.py",
        extra_argv=("pair",),
        required_inputs=(
            "family_policy_json",
            "profile_path",
            "non_vq_artifact",
            "prompt_token_ids_json",
            "config_path",
            "source_index_path",
            "tokenizer_dir",
            "tokenizer_readiness_json",
            "non_vq_evidence_json",
            "routed_artifact",
            "composite_audit_json",
            "materialization_runs",
            "full_bind_preflight_json",
            "snapshot_dir",
            "prompt_pack_json",
            "source_contract_json",
        ),
        input_flags={
            "family_policy_json": "--family-policy-json",
            "profile_path": "--profile-path",
            "non_vq_artifact": "--non-vq-artifact-dir",
            "prompt_token_ids_json": "--prompt-token-ids-json",
            "config_path": "--config-path",
            "source_index_path": "--source-index-path",
            "tokenizer_dir": "--tokenizer-dir",
            "tokenizer_readiness_json": "--tokenizer-readiness-json",
            "non_vq_evidence_json": "--non-vq-evidence-json",
            "routed_artifact": "--routed-artifact-dir",
            "composite_audit_json": "--composite-audit-json",
            "materialization_runs": "--materialization-runs-jsonl",
            "full_bind_preflight_json": "--full-bind-preflight-json",
            "snapshot_dir": "--snapshot-dir",
            "prompt_pack_json": "--prompt-pack-json",
            "source_contract_json": "--source-contract-json",
        },
        evidence_flags={
            "candidate_json": "--candidate-output-json",
            "control_json": "--control-output-json",
            "session_manifest_json": "--session-manifest-json",
        },
        self_managed_heavy_lock=True,
        cacheable=False,
    )
)

_register(
    OpDef(
        name="glm52-same-machine-benchmark-compare",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/bench_glm52_same_machine_fp4.py",
        extra_argv=("compare",),
        required_inputs=(
            "family_policy_json",
            "candidate_json",
            "control_json",
            "session_manifest_json",
        ),
        input_flags={
            "family_policy_json": "--family-policy-json",
            "candidate_json": "--candidate-json",
            "control_json": "--control-json",
            "session_manifest_json": "--session-manifest-json",
        },
        evidence_flags={"evidence_json": "--output-json"},
        cacheable=False,
    )
)

_register(
    OpDef(
        name="glm52-family-gate-check",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/check_glm52_family_gate.py",
        required_inputs=(
            "family_policy_json",
            "eval_prompt_pack_json",
            "teacher_metadata_json",
            "full_bind_preflight_json",
            "indexshare_runtime_json",
            "synthetic_generation_json",
        ),
        optional_inputs=(
            "family_eval_gate_json",
            "family_benchmark_gate_json",
            "non_vq_evidence_json",
            "composite_artifact_audit_json",
            "production_generation_json",
            "teacher_cache_root",
            "teacher_cache_manifest_json",
            "candidate_cache_root",
            "source_route_trace_root",
            "candidate_route_trace_root",
            "source_route_authority_json",
            "candidate_route_authority_json",
            "benchmark_evidence_json",
        ),
        input_flags={
            "family_policy_json": "--family-policy-json",
            "eval_prompt_pack_json": "--eval-prompt-pack-json",
            "teacher_metadata_json": "--teacher-metadata-json",
            "full_bind_preflight_json": "--full-bind-preflight-json",
            "indexshare_runtime_json": "--indexshare-runtime-json",
            "synthetic_generation_json": "--synthetic-generation-json",
            "family_eval_gate_json": "--family-eval-gate-json",
            "family_benchmark_gate_json": "--family-benchmark-gate-json",
            "non_vq_evidence_json": "--non-vq-evidence-json",
            "composite_artifact_audit_json": (
                "--composite-artifact-audit-json"
            ),
            "production_generation_json": "--production-generation-json",
            "teacher_cache_root": "--teacher-cache-root",
            "teacher_cache_manifest_json": "--teacher-cache-manifest-json",
            "candidate_cache_root": "--candidate-cache-root",
            "source_route_trace_root": "--source-route-trace-root",
            "candidate_route_trace_root": "--candidate-route-trace-root",
            "source_route_authority_json": "--source-route-authority-json",
            "candidate_route_authority_json": "--candidate-route-authority-json",
            "benchmark_evidence_json": "--benchmark-evidence-json",
        },
        evidence_flags={"evidence_json": "--output-json"},
        cacheable=False,
        accepted_returncodes=frozenset({0, 2}),
        required_gate_profile="glm52_family",
        regate_can_complete=False,
    )
)

_register(
    OpDef(
        name="qwen-moe-materialize-group",
        version=2,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/materialize_qwen_moe_group.py",
        required_inputs=("source_dir", "index_path"),
        optional_inputs=("config_path",),
        input_flags={
            "source_dir": "--source-dir",
            "index_path": "--index-path",
            "config_path": "--config-path",
        },
        output_dir_flag="--output-dir",
        produces_artifact=True,
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
            "expected_language_layers": ("--expected-language-layers", VALUE),
            "layer": ("--layer", VALUE),
            "source_projection": ("--source-projection", VALUE),
            "code_bits": ("--code-bits", VALUE),
            "group_size": ("--group-size", VALUE),
            "scale_estimator": ("--scale-estimator", VALUE),
        },
        required_params=("model_id", "revision", "layer", "source_projection"),
    )
)

_register(
    OpDef(
        name="qwen-moe-source-payload-audit",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/audit_qwen_moe_source_payloads.py",
        required_inputs=("source_dir", "index_path"),
        optional_inputs=("config_path",),
        input_flags={
            "source_dir": "--source-dir",
            "index_path": "--index-path",
            "config_path": "--config-path",
        },
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
            "expected_language_layers": ("--expected-language-layers", VALUE),
            "max_groups": ("--max-groups", VALUE),
            "code_bits": ("--code-bits", VALUE),
            "group_size": ("--group-size", VALUE),
        },
        required_params=("model_id", "revision"),
    )
)

_register(
    OpDef(
        name="qwen-moe-materialize-groups",
        version=2,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/materialize_qwen_moe_group.py",
        extra_argv=("--all-groups",),
        required_inputs=("source_dir", "index_path"),
        optional_inputs=("config_path",),
        input_flags={
            "source_dir": "--source-dir",
            "index_path": "--index-path",
            "config_path": "--config-path",
        },
        output_dir_flag="--output-dir",
        produces_artifact=True,
        artifact_manifest_name="qwen-moe-materialization-manifest.json",
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
            "expected_language_layers": ("--expected-language-layers", VALUE),
            "max_groups": ("--max-groups", VALUE),
            "skip_existing": ("--skip-existing", SWITCH),
            "code_bits": ("--code-bits", VALUE),
            "group_size": ("--group-size", VALUE),
            "scale_estimator": ("--scale-estimator", VALUE),
        },
        required_params=("model_id", "revision"),
    )
)

_register(
    OpDef(
        name="qwen-moe-artifact-audit",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/audit_qwen_moe_materialization.py",
        required_inputs=("artifact",),
        input_flags={
            "artifact": "--artifact-dir",
        },
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
        param_specs={
            "expected_source_projection_groups": (
                "--expected-source-projection-groups",
                VALUE,
            ),
            "expected_target_projection_groups": (
                "--expected-target-projection-groups",
                VALUE,
            ),
        },
    )
)

_register(
    OpDef(
        name="qwen-moe-bind-probe",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/probe_qwen_moe_binding.py",
        required_inputs=("artifact",),
        input_flags={
            "artifact": "--artifact-dir",
        },
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
        param_specs={
            "expected_layers": ("--expected-layers", CSV),
            "model_layer_count": ("--model-layer-count", VALUE),
        },
    )
)

_register(
    OpDef(
        name="qwen-non-expert-bind-probe",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/probe_qwen_non_expert_binding.py",
        required_inputs=("source_dir", "index_path"),
        input_flags={
            "source_dir": "--source-dir",
            "index_path": "--index-path",
        },
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
        param_specs={
            "expected_layers": ("--expected-layers", CSV),
            "layer_only": ("--layer-only", SWITCH),
            "max_tensors": ("--max-tensors", VALUE),
        },
    )
)

_register(
    OpDef(
        name="qwen-runtime-bind-forward-probe",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/probe_qwen_runtime_binding.py",
        required_inputs=("artifact", "source_dir", "index_path"),
        input_flags={
            "artifact": "--artifact-dir",
            "source_dir": "--source-dir",
            "index_path": "--index-path",
            "tokenizer_dir": "--tokenizer-dir",
        },
        optional_inputs=("tokenizer_dir",),
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
        param_specs={
            "expected_layers": ("--expected-layers", CSV),
            "model_layer_count": ("--model-layer-count", VALUE),
            "forward_layers": ("--forward-layers", CSV),
            "forward_top_k": ("--forward-top-k", VALUE),
            "input_scale": ("--input-scale", VALUE),
            "layer_only": ("--layer-only", SWITCH),
            "max_tensors": ("--max-tensors", VALUE),
            "logit_probe_token_id": ("--logit-probe-token-id", VALUE),
            "logit_probe_top_k": ("--logit-probe-top-k", VALUE),
            "tokenizer_prompt": ("--tokenizer-prompt", VALUE),
            "tokenized_logit_position": ("--tokenized-logit-position", VALUE),
            "tokenized_logit_top_k": ("--tokenized-logit-top-k", VALUE),
        },
    )
)

_register(
    OpDef(
        name="qwen-upstream-transformer-bind-probe",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/probe_qwen_upstream_transformer_binding.py",
        required_inputs=("artifact", "config_path"),
        input_flags={
            "artifact": "--artifact-dir",
            "config_path": "--config-path",
        },
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
        param_specs={
            "expected_layers": ("--expected-layers", CSV),
        },
    )
)

_register(
    OpDef(
        name="qwen-tokenizer-readiness-probe",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/probe_qwen_tokenizer_readiness.py",
        required_inputs=("tokenizer_dir",),
        input_flags={
            "tokenizer_dir": "--tokenizer-dir",
        },
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
        param_specs={
            "prompt": ("--prompt", VALUE),
            "min_token_count": ("--min-token-count", VALUE),
            "expected_vocab_size": ("--expected-vocab-size", VALUE),
        },
    )
)

_register(
    OpDef(
        name="qwen-family-gate-policy",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/define_qwen_family_gate_policy.py",
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
            "minimum_clean_rows_per_split": (
                "--minimum-clean-rows-per-split",
                VALUE,
            ),
            "minimum_total_clean_rows": (
                "--minimum-total-clean-rows",
                VALUE,
            ),
            "minimum_repetitions_per_scenario": (
                "--minimum-repetitions-per-scenario",
                VALUE,
            ),
            "comparison_baseline": ("--comparison-baseline", VALUE),
            "maximum_candidate_to_reference_ratio": (
                "--maximum-candidate-to-reference-ratio",
                VALUE,
            ),
        },
    )
)

_register(
    OpDef(
        name="qwen-family-eval-prompt-probe",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/prepare_qwen_family_eval_prompts.py",
        required_inputs=("tokenizer_dir",),
        input_flags={
            "tokenizer_dir": "--tokenizer-dir",
        },
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
            "min_prompts_per_split": ("--min-prompts-per-split", VALUE),
            "min_tokens_per_prompt": ("--min-tokens-per-prompt", VALUE),
            "expected_vocab_size": ("--expected-vocab-size", VALUE),
        },
    )
)

_register(
    OpDef(
        name="qwen-family-eval-gate-check",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/check_qwen_family_eval_gate.py",
        required_inputs=("family_policy_json",),
        optional_inputs=("eval_prompt_pack_json", "eval_jsonl", "eval_row_probe_json"),
        input_flags={
            "family_policy_json": "--family-policy-json",
            "eval_prompt_pack_json": "--eval-prompt-pack-json",
            "eval_jsonl": "--eval-jsonl",
            "eval_row_probe_json": "--eval-row-probe-json",
        },
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
    )
)

_register(
    OpDef(
        name="qwen-family-eval-row-probe",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/prepare_qwen_family_eval_rows.py",
        required_inputs=("family_policy_json", "eval_prompt_pack_json"),
        optional_inputs=("eval_metric_jsonl", "teacher_cache_metadata_jsonl"),
        input_flags={
            "family_policy_json": "--family-policy-json",
            "eval_prompt_pack_json": "--eval-prompt-pack-json",
            "eval_metric_jsonl": "--eval-metric-jsonl",
            "teacher_cache_metadata_jsonl": "--teacher-cache-metadata-jsonl",
        },
        output_dir_flag="--output-dir",
        file_outputs={"eval_jsonl": "qwen_family_eval_rows.jsonl"},
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
    )
)

_register(
    OpDef(
        name="qwen-family-eval-teacher-metadata-probe",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/prepare_qwen_family_eval_teacher_metadata.py",
        required_inputs=("family_policy_json", "eval_prompt_pack_json", "tokenizer_dir", "source_dir"),
        input_flags={
            "family_policy_json": "--family-policy-json",
            "eval_prompt_pack_json": "--eval-prompt-pack-json",
            "tokenizer_dir": "--tokenizer-dir",
            "source_dir": "--source-dir",
        },
        output_dir_flag="--output-dir",
        file_outputs={"metadata_jsonl": "qwen_teacher_cache_metadata.jsonl"},
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
    )
)

_register(
    OpDef(
        name="qwen-family-eval-metric-row-probe",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/prepare_qwen_family_eval_metric_rows.py",
        required_inputs=("family_policy_json", "eval_prompt_pack_json"),
        optional_inputs=("candidate_logits_jsonl", "teacher_logits_jsonl"),
        input_flags={
            "family_policy_json": "--family-policy-json",
            "eval_prompt_pack_json": "--eval-prompt-pack-json",
            "candidate_logits_jsonl": "--candidate-logits-jsonl",
            "teacher_logits_jsonl": "--teacher-logits-jsonl",
        },
        output_dir_flag="--output-dir",
        file_outputs={"metric_jsonl": "qwen_family_eval_metric_rows.jsonl"},
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
    )
)

_register(
    OpDef(
        name="qwen-family-eval-candidate-logits-probe",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/prepare_qwen_family_eval_candidate_logits.py",
        required_inputs=("family_policy_json", "eval_prompt_pack_json"),
        optional_inputs=("source_dir", "index_path", "teacher_logits_jsonl"),
        input_flags={
            "family_policy_json": "--family-policy-json",
            "eval_prompt_pack_json": "--eval-prompt-pack-json",
            "source_dir": "--source-dir",
            "index_path": "--index-path",
            "teacher_logits_jsonl": "--teacher-logits-jsonl",
        },
        param_specs={
            "selected_token_position": ("--selected-token-position", VALUE),
            "top_k": ("--top-k", VALUE),
        },
        output_dir_flag="--output-dir",
        file_outputs={"candidate_logits_jsonl": "qwen_candidate_eval_logits.jsonl"},
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
    )
)

_register(
    OpDef(
        name="qwen-family-eval-teacher-logits-probe",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/prepare_qwen_family_eval_teacher_logits.py",
        required_inputs=("family_policy_json", "eval_prompt_pack_json", "source_dir", "index_path"),
        input_flags={
            "family_policy_json": "--family-policy-json",
            "eval_prompt_pack_json": "--eval-prompt-pack-json",
            "source_dir": "--source-dir",
            "index_path": "--index-path",
        },
        output_dir_flag="--output-dir",
        file_outputs={"teacher_logits_jsonl": "qwen_teacher_eval_logits.jsonl"},
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
        param_specs={
            "selected_token_position": ("--selected-token-position", VALUE),
            "top_k": ("--top-k", VALUE),
        },
    )
)

_register(
    OpDef(
        name="qwen-family-benchmark-gate-check",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/check_qwen_family_benchmark_gate.py",
        required_inputs=("family_policy_json",),
        optional_inputs=("benchmark_jsonl", "benchmark_row_probe_json"),
        input_flags={
            "family_policy_json": "--family-policy-json",
            "benchmark_jsonl": "--benchmark-jsonl",
            "benchmark_row_probe_json": "--benchmark-row-probe-json",
        },
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
    )
)

_register(
    OpDef(
        name="qwen-family-benchmark-invariant-probe",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/prepare_qwen_family_benchmark_invariants.py",
        required_inputs=(
            "family_policy_json",
            "artifact_audit_json",
            "non_expert_bind_json",
        ),
        input_flags={
            "family_policy_json": "--family-policy-json",
            "artifact_audit_json": "--artifact-audit-json",
            "non_expert_bind_json": "--non-expert-bind-json",
        },
        output_dir_flag="--output-dir",
        file_outputs={
            "candidate_invariant_jsonl": "qwen_candidate_benchmark_invariants.jsonl"
        },
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
        param_specs={
            "run_index": ("--run-index", VALUE),
            "repetitions": ("--repetitions", VALUE),
        },
    )
)

_register(
    OpDef(
        name="qwen-family-candidate-benchmark-latency-probe",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/bench_qwen_family_candidate_latency.py",
        required_inputs=("family_policy_json", "artifact"),
        input_flags={
            "family_policy_json": "--family-policy-json",
            "artifact": "--artifact-dir",
        },
        output_dir_flag="--output-dir",
        file_outputs={"latency_jsonl": "qwen_candidate_benchmark_latency_rows.jsonl"},
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
        param_specs={
            "layer": ("--layer", VALUE),
            "repetitions": ("--repetitions", VALUE),
            "warmup_repetitions": ("--warmup-repetitions", VALUE),
            "top_k": ("--top-k", VALUE),
            "input_scale": ("--input-scale", VALUE),
        },
    )
)

_register(
    OpDef(
        name="qwen-family-control-benchmark-latency-probe",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/bench_qwen_family_control_latency.py",
        required_inputs=("family_policy_json", "source_dir", "index_path"),
        input_flags={
            "family_policy_json": "--family-policy-json",
            "source_dir": "--source-dir",
            "index_path": "--index-path",
        },
        output_dir_flag="--output-dir",
        file_outputs={"latency_jsonl": "qwen_control_benchmark_latency_rows.jsonl"},
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
        param_specs={
            "layer": ("--layer", VALUE),
            "repetitions": ("--repetitions", VALUE),
            "warmup_repetitions": ("--warmup-repetitions", VALUE),
            "top_k": ("--top-k", VALUE),
            "input_scale": ("--input-scale", VALUE),
        },
    )
)

_register(
    OpDef(
        name="qwen-family-benchmark-row-probe",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/prepare_qwen_family_benchmark_rows.py",
        required_inputs=("family_policy_json",),
        optional_inputs=(
            "latency_jsonl",
            "candidate_latency_jsonl",
            "control_latency_jsonl",
            "candidate_invariant_jsonl",
        ),
        input_flags={
            "family_policy_json": "--family-policy-json",
            "latency_jsonl": "--latency-jsonl",
            "candidate_latency_jsonl": "--latency-jsonl",
            "control_latency_jsonl": "--latency-jsonl",
            "candidate_invariant_jsonl": "--candidate-invariant-jsonl",
        },
        output_dir_flag="--output-dir",
        file_outputs={"benchmark_jsonl": "qwen_family_benchmark_rows.jsonl"},
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
    )
)

_register(
    OpDef(
        name="qwen-family-gate-check",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/check_qwen_family_gate.py",
        required_inputs=(
            "source_audit_json",
            "source_payload_audit_json",
            "artifact_audit_json",
            "bind_probe_json",
            "non_expert_bind_probe_json",
            "runtime_forward_json",
            "runtime_global_logit_json",
        ),
        optional_inputs=(
            "upstream_transformer_json",
            "tokenizer_json",
            "tokenized_logits_json",
            "family_policy_json",
            "eval_prompt_pack_json",
            "family_eval_gate_json",
            "family_benchmark_gate_json",
        ),
        input_flags={
            "source_audit_json": "--source-audit-json",
            "source_payload_audit_json": "--source-payload-audit-json",
            "artifact_audit_json": "--artifact-audit-json",
            "bind_probe_json": "--bind-probe-json",
            "non_expert_bind_probe_json": "--non-expert-bind-probe-json",
            "runtime_forward_json": "--runtime-forward-json",
            "runtime_global_logit_json": "--runtime-global-logit-json",
            "upstream_transformer_json": "--upstream-transformer-json",
            "tokenizer_json": "--tokenizer-json",
            "tokenized_logits_json": "--tokenized-logits-json",
            "family_policy_json": "--family-policy-json",
            "eval_prompt_pack_json": "--eval-prompt-pack-json",
            "family_eval_gate_json": "--family-eval-gate-json",
            "family_benchmark_gate_json": "--family-benchmark-gate-json",
        },
        evidence_flags={
            "evidence_json": "--output-json",
            "evidence": "--append-jsonl",
        },
        param_specs={
            "expected_layer_count": ("--expected-layer-count", VALUE),
            "upstream_transformer_integration": (
                "--upstream-transformer-integration",
                SWITCH,
            ),
            "tokenizer_to_logits_forward": ("--tokenizer-to-logits-forward", SWITCH),
            "family_eval_gate": ("--family-eval-gate", SWITCH),
            "family_benchmark_gate": ("--family-benchmark-gate", SWITCH),
        },
    )
)

_register(
    OpDef(
        name="glm52-vq-validate",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="scripts/validate_glm52_vq.py",
        required_inputs=("source_dir", "index_path", "artifact"),
        optional_inputs=("config_path",),
        input_flags={
            "source_dir": "--source-dir",
            "config_path": "--config-path",
            "index_path": "--index-path",
            "artifact": "--artifact-dir",
        },
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
            "layer": ("--layer", VALUE),
            "tokens": ("--tokens", VALUE),
            "seed": ("--seed", VALUE),
            "input_scale": ("--input-scale", VALUE),
            "no_strict_config": ("--no-strict-config", SWITCH),
            "min_artifact_cosine": ("--min-artifact-cosine", VALUE),
            "min_source_weighted_cosine": ("--min-source-weighted-cosine", VALUE),
        },
        required_params=("model_id", "revision"),
    )
)

_register(
    OpDef(
        name="jaccl-hostfile",
        version=1,
        allowed_classes=frozenset({"promotable", "diagnostic"}),
        script="benchmarks/write_jaccl_hostfile.py",
        output_dir_flag="--output-dir",
        file_outputs={"hostfile": "hostfile.json"},
        param_specs={
            "local_ssh": ("--local-ssh", VALUE),
            "local_ip": ("--local-ip", VALUE),
            "local_rdma_device": ("--local-rdma-device", VALUE),
            "peer_ssh": ("--peer-ssh", VALUE),
            "peer_rdma_device": ("--peer-rdma-device", VALUE),
        },
        required_params=(
            "local_ip",
            "local_rdma_device",
            "peer_ssh",
            "peer_rdma_device",
        ),
    )
)

_register(
    OpDef(
        name="rdma-topology-audit",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/audit_glm45_air_rdma_topology.py",
        evidence_flags={"evidence_json": "--output-json"},
        param_specs={
            "local_if": ("--local-if", VALUE),
            "peer_if": ("--peer-if", VALUE),
            "local_ip": ("--local-ip", VALUE),
            "peer_ip": ("--peer-ip", VALUE),
            "local_rdma_device": ("--local-rdma-device", VALUE),
            "peer_rdma_device": ("--peer-rdma-device", VALUE),
            "peer_ssh": ("--peer-ssh", VALUE),
            "timeout": ("--timeout", VALUE),
            "require_ready": ("--require-ready", SWITCH),
        },
        required_params=(
            "local_if",
            "peer_if",
            "local_ip",
            "peer_ip",
            "local_rdma_device",
            "peer_rdma_device",
            "peer_ssh",
        ),
    )
)

_register(
    OpDef(
        name="single-host-teacher-source-audit",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/audit_glm45_air_single_host_teacher_source.py",
        evidence_flags={"evidence_json": "--output-json"},
        param_specs={
            "model_path": ("--model-path", VALUE),
            "revision": ("--revision", VALUE),
            "teacher_kind": ("--teacher-kind", VALUE),
            "scan_root": ("--scan-root", VALUE),
        },
        required_params=("model_path",),
    )
)

_register(
    OpDef(
        name="pipeline-source-views",
        version=1,
        allowed_classes=frozenset({"promotable", "diagnostic"}),
        script="benchmarks/materialize_glm45_air_pipeline_views.py",
        optional_inputs=("source_dir", "index_path"),
        input_flags={
            "source_dir": "--source-dir",
            "index_path": "--index-path",
        },
        output_dir_flag="--output-dir",
        produces_artifact=True,
        file_outputs={
            "stage_view_roots_json": "local_sequential_stage_view_roots.json",
        },
        output_namer=_pipeline_source_view_output_names,
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
            "pipeline_size": ("--pipeline-size", VALUE),
            "layer_split": ("--layer-split", VALUE),
            "rank0_stop_after_layer": ("--rank0-stop-after-layer", VALUE),
            "rank0_route_trace_only": ("--rank0-route-trace-only", SWITCH),
            "route_trace_only_views": ("--route-trace-only-views", SWITCH),
            "rank0_stop_after_route_gate": (
                "--rank0-stop-after-route-gate",
                SWITCH,
            ),
            "rank0_only_logits_views": ("--rank0-only-logits-views", SWITCH),
            "local_sequential_stage_views": (
                "--local-sequential-stage-views",
                SWITCH,
            ),
            "local_sequential_lower_split_layer": (
                "--local-sequential-lower-split-layer",
                VALUE,
            ),
            "local_sequential_upper_split_layer": (
                "--local-sequential-upper-split-layer",
                VALUE,
            ),
            "local_sequential_lower_split_layers": (
                "--local-sequential-lower-split-layers",
                CSV,
            ),
            "local_sequential_upper_split_layers": (
                "--local-sequential-upper-split-layers",
                CSV,
            ),
            "local_sequential_head_process": (
                "--local-sequential-head-process",
                SWITCH,
            ),
            "rank_budget_gb": ("--rank-budget-gb", VALUE),
            "materialize": ("--materialize", SWITCH),
            "copy": ("--copy", SWITCH),
        },
    )
)

_register(
    OpDef(
        name="pipeline-view-load-probe",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/probe_glm45_air_pipeline_view_load.py",
        required_inputs=("view_dir",),
        optional_inputs=("source_dir", "index_path"),
        input_flags={
            "source_dir": "--source-dir",
            "view_dir": "--output-dir",
            "index_path": "--index-path",
        },
        evidence_flags={"evidence_json": "--output-json"},
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
            "pipeline_size": ("--pipeline-size", VALUE),
            "layer_split": ("--layer-split", VALUE),
            "rank_budget_gb": ("--rank-budget-gb", VALUE),
            "reuse_existing": ("--reuse-existing", SWITCH),
            "copy": ("--copy", SWITCH),
        },
    )
)

_register(
    OpDef(
        name="attribute",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/analyze_glm45_air_teacher_cache_attribution.py",
        required_inputs=("baseline_evidence",),
        optional_inputs=("candidate_evidence",),
        input_flags={
            "baseline_evidence": "--baseline-jsonl",
            "candidate_evidence": "--candidate-jsonl",
        },
        evidence_flags={"attribution": "--output-json"},
        param_specs={
            "baseline_label": ("--baseline-label", VALUE),
            "top_tokens": ("--top-tokens", VALUE),
            "top_prompts": ("--top-prompts", VALUE),
        },
    )
)

_register(
    OpDef(
        name="top1-gap",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/analyze_glm45_air_top1_gap.py",
        optional_inputs=("baseline_evidence", "candidate_evidence", "evidence"),
        input_flags={
            "baseline_evidence": "--baseline-jsonl",
            "candidate_evidence": "--candidate-jsonl",
            "evidence": "--jsonl",
        },
        evidence_flags={"gap": "--output-json"},
        param_specs={
            "label": ("--label", VALUE),
            "focus_position": ("--focus-position", VALUE),
            "watch_token_id": ("--watch-token-id", VALUE),
            "teacher_token_id": ("--teacher-token-id", VALUE),
            "select_tail_cleanup_targets": ("--select-tail-cleanup-targets", SWITCH),
            "max_train_rows": ("--max-train-rows", VALUE),
            "max_validation_rows": ("--max-validation-rows", VALUE),
            "top_tail_records": ("--top-tail-records", VALUE),
        },
    )
)

_register(
    OpDef(
        name="select-rows",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/select_glm45_air_teacher_cache_rows.py",
        required_inputs=("report_evidence", "selection_evidence"),
        input_flags={
            "report_evidence": "--report-eval-jsonl",
            "selection_evidence": "--selection-eval-jsonl",
        },
        evidence_flags={"rows": "--output-json"},
        param_specs={
            "count": ("--count", VALUE),
            "pool_size": ("--pool-size", VALUE),
            "avoid_row_indices": ("--avoid-row-indices", CSV),
            "top1_weight": ("--top1-weight", VALUE),
            "kld_weight": ("--kld-weight", VALUE),
            "shared_signal_only": ("--shared-signal-only", SWITCH),
        },
    )
)

_register(
    OpDef(
        name="materialize-logit-bias",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/materialize_glm45_air_logit_bias.py",
        required_inputs=("seed_artifact",),
        input_flags={"seed_artifact": "--seed-artifact-dir"},
        output_dir_flag="--output-dir",
        produces_artifact=True,
        evidence_flags={"bias_log": "--append-jsonl"},
        param_specs={
            "token_bias": ("--token-bias", VALUE),
            "position_indices": ("--position-indices", CSV),
            "token_position": ("--token-position", VALUE),
        },
        required_params=("token_bias",),
    )
)

_register(
    OpDef(
        name="bump-non-expert-precision",
        version=1,
        allowed_classes=frozenset({"promotable", "diagnostic"}),
        script="benchmarks/materialize_glm45_air_non_expert_precision.py",
        required_inputs=("seed_artifact",),
        input_flags={"seed_artifact": "--seed-artifact-dir"},
        output_dir_flag="--output-dir",
        produces_artifact=True,
        evidence_flags={"precision_log": "--append-jsonl"},
        param_specs={
            "surfaces": ("--surfaces", CSV),
            "dtype": ("--dtype", VALUE),
            "reason": ("--reason", VALUE),
        },
        required_params=("surfaces", "dtype"),
    )
)

_register(
    OpDef(
        name="isolate-sparse-fp16",
        version=1,
        allowed_classes=frozenset({"promotable", "diagnostic"}),
        script="benchmarks/materialize_glm45_air_high_precision_projection.py",
        required_inputs=("seed_artifact",),
        optional_inputs=("source_dir", "index_path"),
        input_flags={
            "seed_artifact": "--seed-artifact-dir",
            "source_dir": "--source-dir",
            "index_path": "--index-path",
        },
        output_dir_flag="--output-dir",
        produces_artifact=True,
        file_outputs={"manifest": "high-precision-projection-manifest.json"},
        param_specs={
            "projection": ("--projection", VALUE),
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
            "expert_count": ("--expert-count", VALUE),
        },
        required_params=("projection", "model_id", "revision"),
    )
)

_register(
    OpDef(
        name="layer-probe-attribution",
        version=1,
        allowed_classes=frozenset({"diagnostic"}),
        script="benchmarks/probe_glm45_air_layer_attribution.py",
        required_inputs=("teacher", "eval", "source_dir", "artifact"),
        optional_inputs=("index_path", "config_path", "capture_artifact", "state_npz"),
        input_flags={
            "teacher": "--teacher-jsonl",
            "eval": "--eval-jsonl",
            "source_dir": "--source-dir",
            "artifact": "--artifact-dir",
            "index_path": "--index-path",
            "config_path": "--config-path",
            "capture_artifact": "--capture-artifact-dir",
            "state_npz": "--state-npz",
        },
        evidence_flags={"plan_json": "--output-json"},
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
            "layers": ("--layers", VALUE),
            "target": ("--target", VALUE),
            "no_strict_config": ("--no-strict-config", SWITCH),
        },
        required_params=("target",),
    )
)

_register(
    OpDef(
        name="sparse-residual",
        version=1,
        allowed_classes=frozenset({"promotable", "diagnostic"}),
        script="benchmarks/materialize_glm45_air_vq_sparse_residual.py",
        required_inputs=("seed_artifact",),
        optional_inputs=("source_dir", "index_path", "config_path", "plan_json"),
        input_flags={
            "seed_artifact": "--seed-artifact-dir",
            "source_dir": "--source-dir",
            "index_path": "--index-path",
            "config_path": "--config-path",
            "plan_json": "--plan-json",
        },
        output_dir_flag="--output-dir",
        produces_artifact=True,
        evidence_flags={"residual_log": "--append-jsonl"},
        param_specs={
            "model_id": ("--model-id", VALUE),
            "revision": ("--revision", VALUE),
            "layer": ("--layer", VALUE),
            "projection": ("--projection", VALUE),
            "experts": ("--experts", CSV),
            "max_rows_per_expert": ("--max-rows-per-expert", VALUE),
            "residual_scale": ("--residual-scale", VALUE),
            "plan_target": ("--plan-target", VALUE),
            "plan_expert": ("--plan-expert", VALUE),
            "plan_route_rank": ("--plan-route-rank", VALUE),
            "plan_max_rows": ("--plan-max-rows", VALUE),
        },
        required_params=("model_id", "revision"),
    )
)

_register(
    OpDef(
        name="train-router-kd",
        version=1,
        allowed_classes=frozenset({"promotable", "diagnostic"}),
        script="benchmarks/materialize_glm45_air_router_correction.py",
        required_inputs=("seed_artifact", "disagreement"),
        input_flags={
            "seed_artifact": "--seed-artifact-dir",
            "disagreement": "--disagreement-json",
        },
        output_dir_flag="--output-dir",
        produces_artifact=True,
        evidence_flags={"router_kd_log": "--append-jsonl"},
        param_specs={
            "scale": ("--scale", VALUE),
            "max_abs_delta": ("--max-abs-delta", VALUE),
            "num_experts": ("--num-experts", VALUE),
        },
    )
)
