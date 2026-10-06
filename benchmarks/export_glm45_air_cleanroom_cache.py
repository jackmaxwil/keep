"""Flag-based adapter for the GLM-4.5-Air clean-room teacher-cache script."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def _set(env: dict[str, str], name: str, value: object | None) -> None:
    if value is not None:
        env[name] = str(value)


def _set_switch(env: dict[str, str], name: str, enabled: bool) -> None:
    if enabled:
        env[name] = "1"


_RANK_VIEW_DIR_RE = re.compile(r"^rank-(\d+)$")


def _rank_view_roots_json_from_dir(view_dir: Path) -> str:
    if not view_dir.is_dir():
        raise ValueError(f"rank view directory does not exist: {view_dir}")
    rank_roots: dict[int, str] = {}
    for child in view_dir.iterdir():
        if not child.is_dir():
            continue
        match = _RANK_VIEW_DIR_RE.match(child.name)
        if match is None:
            continue
        rank = int(match.group(1))
        if rank in rank_roots:
            raise ValueError(f"duplicate rank view directory for rank {rank}: {child}")
        rank_roots[rank] = str(child)
    if not rank_roots:
        raise ValueError(f"no rank view directories found under {view_dir}")
    return json.dumps(
        {str(rank): rank_roots[rank] for rank in sorted(rank_roots)},
        separators=(",", ":"),
    )


def _json_value_or_file(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.lstrip()
    if stripped.startswith(("{", "[")):
        return value
    path = Path(value)
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run scripts/run_glm45_air_distributed_cleanroom_cache.sh from a "
            "fully rendered recipe argv."
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-jsonl")
    parser.add_argument("--local-jsonl")
    parser.add_argument("--jaccl-hostfile")
    parser.add_argument("--peer-ssh")
    parser.add_argument("--peer-repo")
    parser.add_argument("--peer-rank0-view-root")
    parser.add_argument("--peer-rank1-view-root")
    parser.add_argument("--peer-source-dir")
    parser.add_argument("--peer-source-stage-ssh")
    parser.add_argument("--stage-peer-source", action="store_true")
    parser.add_argument("--peer-source-min-free-gb", type=float)
    parser.add_argument("--peer-uv")
    parser.add_argument("--known-hosts")
    parser.add_argument("--tmp-src-root")
    parser.add_argument("--tmp-src-archive")
    rank_view_group = parser.add_mutually_exclusive_group()
    rank_view_group.add_argument("--rank-view-roots-json")
    rank_view_group.add_argument("--rank-view-dir", type=Path)
    parser.add_argument("--local-sequential-stage-view-roots-json")
    parser.add_argument("--prompt-set")
    parser.add_argument("--prompt-id", action="append")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-positions", type=int)
    parser.add_argument("--layer-split", type=int)
    parser.add_argument("--lm-head-chunk-rows", type=int)
    parser.add_argument("--mlx-wired-limit-gb", type=float)
    parser.add_argument("--mlx-cache-limit-gb", type=float)
    parser.add_argument("--required-wired-mb", type=int)
    parser.add_argument("--local-direct-if")
    parser.add_argument("--peer-direct-if")
    parser.add_argument("--local-direct-ip")
    parser.add_argument("--peer-direct-ip")
    parser.add_argument("--skip-direct-link-check", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--no-full-logits", action="store_true")
    parser.add_argument("--include-route-trace", action="store_true")
    parser.add_argument("--route-trace-only", action="store_true")
    parser.add_argument("--route-trace-layer", action="append")
    parser.add_argument("--skip-local-consume", action="store_true")
    parser.add_argument("--allow-dirty-cache", action="store_true")
    parser.add_argument("--memory-quiet-preflight", action="store_true")
    parser.add_argument("--require-memory-quiet-preflight", action="store_true")
    parser.add_argument("--memory-quiet-seconds", type=float)
    parser.add_argument("--memory-quiet-max-attempts", type=int)
    parser.add_argument("--memory-quiet-min-free-gb", type=float)
    parser.add_argument("--memory-quiet-stage-view-margin-gb", type=float)
    parser.add_argument("--single-host-fallback", action="store_true")
    parser.add_argument("--single-host-model-path")
    parser.add_argument("--single-host-revision")
    parser.add_argument("--single-host-teacher-kind")
    parser.add_argument("--single-host-lazy", action="store_true")
    parser.add_argument("--single-host-pipeline-local", action="store_true")
    parser.add_argument("--single-host-pipeline-local-stage-processes", action="store_true")
    parser.add_argument("--single-host-pipeline-local-head-process", action="store_true")
    parser.add_argument("--single-host-pipeline-local-lower-split-layer", type=int)
    parser.add_argument("--single-host-pipeline-local-upper-split-layer", type=int)
    parser.add_argument("--single-host-pipeline-local-lower-split-layers")
    parser.add_argument("--single-host-pipeline-local-upper-split-layers")
    parser.add_argument("--single-host-pipeline-local-abort-on-dirty-stage", action="store_true")
    parser.add_argument("--local-sequential-remote-workers")
    parser.add_argument("--local-sequential-remote-tmp-dir")
    parser.add_argument("--local-sequential-remote-dirty-retries", type=int)
    parser.add_argument("--local-sequential-remote-retry-sleep-seconds", type=float)
    parser.add_argument("--single-host-mlx-wired-limit-gb", type=float)
    parser.add_argument("--single-host-source-memory-guard-ratio", type=float)
    parser.add_argument("--vq-artifact-dir")
    parser.add_argument("--engine")
    args = parser.parse_args()

    rank_view_roots_json = args.rank_view_roots_json
    if args.rank_view_dir is not None:
        try:
            rank_view_roots_json = _rank_view_roots_json_from_dir(args.rank_view_dir)
        except ValueError as exc:
            parser.error(str(exc))
    local_sequential_stage_view_roots_json = _json_value_or_file(
        args.local_sequential_stage_view_roots_json
    )

    env = os.environ.copy()
    _set(env, "GLM_OUTPUT_DIR", args.output_dir)
    _set(env, "GLM_VALIDATION_JSONL", args.validation_jsonl)
    _set(env, "GLM_LOCAL_JSONL", args.local_jsonl)
    _set(env, "GLM_JACCL_HOSTFILE", args.jaccl_hostfile)
    _set(env, "GLM_PEER_SSH", args.peer_ssh)
    _set(env, "GLM_PEER_REPO", args.peer_repo)
    _set(env, "GLM_PEER_RANK0_VIEW_ROOT", args.peer_rank0_view_root)
    _set(env, "GLM_PEER_RANK1_VIEW_ROOT", args.peer_rank1_view_root)
    _set(env, "GLM_PEER_SOURCE_DIR", args.peer_source_dir)
    _set(env, "GLM_PEER_SOURCE_STAGE_SSH", args.peer_source_stage_ssh)
    _set_switch(env, "GLM_STAGE_PEER_SOURCE", args.stage_peer_source)
    _set(env, "GLM_PEER_SOURCE_MIN_FREE_GB", args.peer_source_min_free_gb)
    _set(env, "GLM_PEER_UV", args.peer_uv)
    _set(env, "GLM_KNOWN_HOSTS", args.known_hosts)
    _set(env, "GLM_TMP_SRC_ROOT", args.tmp_src_root)
    _set(env, "GLM_TMP_SRC_ARCHIVE", args.tmp_src_archive)
    _set(env, "GLM_RANK_VIEW_ROOTS_JSON", rank_view_roots_json)
    _set(
        env,
        "GLM_LOCAL_SEQUENTIAL_STAGE_VIEW_ROOTS_JSON",
        local_sequential_stage_view_roots_json,
    )
    _set(env, "GLM_PROMPT_SET", args.prompt_set)
    if args.prompt_id:
        env["GLM_PROMPT_IDS"] = ",".join(args.prompt_id)
    _set(env, "GLM_TOP_K", args.top_k)
    _set(env, "GLM_MAX_POSITIONS", args.max_positions)
    _set(env, "GLM_LAYER_SPLIT", args.layer_split)
    _set(env, "GLM_LM_HEAD_CHUNK_ROWS", args.lm_head_chunk_rows)
    _set(env, "GLM_MLX_WIRED_LIMIT_GB", args.mlx_wired_limit_gb)
    _set(env, "GLM_MLX_CACHE_LIMIT_GB", args.mlx_cache_limit_gb)
    _set(env, "GLM_REQUIRED_WIRED_MB", args.required_wired_mb)
    _set(env, "GLM_LOCAL_DIRECT_IF", args.local_direct_if)
    _set(env, "GLM_PEER_DIRECT_IF", args.peer_direct_if)
    _set(env, "GLM_LOCAL_DIRECT_IP", args.local_direct_ip)
    _set(env, "GLM_PEER_DIRECT_IP", args.peer_direct_ip)
    _set_switch(env, "GLM_SKIP_DIRECT_LINK_CHECK", args.skip_direct_link_check)
    _set_switch(env, "GLM_PREFLIGHT_ONLY", args.preflight_only)
    _set_switch(env, "GLM_NO_FULL_LOGITS", args.no_full_logits)
    _set_switch(env, "GLM_INCLUDE_ROUTE_TRACE", args.include_route_trace)
    _set_switch(env, "GLM_ROUTE_TRACE_ONLY", args.route_trace_only)
    _set_switch(env, "GLM_SKIP_LOCAL_CONSUME", args.skip_local_consume)
    _set_switch(env, "GLM_ALLOW_DIRTY_CACHE", args.allow_dirty_cache)
    _set_switch(env, "GLM_MEMORY_QUIET_PREFLIGHT", args.memory_quiet_preflight)
    _set_switch(
        env,
        "GLM_REQUIRE_MEMORY_QUIET_PREFLIGHT",
        args.require_memory_quiet_preflight,
    )
    _set(env, "GLM_MEMORY_QUIET_SECONDS", args.memory_quiet_seconds)
    _set(env, "GLM_MEMORY_QUIET_MAX_ATTEMPTS", args.memory_quiet_max_attempts)
    _set(env, "GLM_MEMORY_QUIET_MIN_FREE_GB", args.memory_quiet_min_free_gb)
    _set(
        env,
        "GLM_MEMORY_QUIET_STAGE_VIEW_ROOTS_JSON",
        local_sequential_stage_view_roots_json,
    )
    _set(
        env,
        "GLM_MEMORY_QUIET_STAGE_VIEW_MARGIN_GB",
        args.memory_quiet_stage_view_margin_gb,
    )
    _set_switch(env, "GLM_SINGLE_HOST_FALLBACK", args.single_host_fallback)
    _set(env, "GLM_SINGLE_HOST_MODEL_PATH", args.single_host_model_path)
    _set(env, "GLM_SINGLE_HOST_REVISION", args.single_host_revision)
    _set(env, "GLM_SINGLE_HOST_TEACHER_KIND", args.single_host_teacher_kind)
    _set_switch(env, "GLM_SINGLE_HOST_LAZY", args.single_host_lazy)
    _set_switch(env, "GLM_SINGLE_HOST_PIPELINE_LOCAL", args.single_host_pipeline_local)
    _set_switch(
        env,
        "GLM_SINGLE_HOST_PIPELINE_LOCAL_STAGE_PROCESSES",
        args.single_host_pipeline_local_stage_processes,
    )
    _set_switch(
        env,
        "GLM_SINGLE_HOST_PIPELINE_LOCAL_HEAD_PROCESS",
        args.single_host_pipeline_local_head_process,
    )
    _set(
        env,
        "GLM_SINGLE_HOST_PIPELINE_LOCAL_LOWER_SPLIT_LAYER",
        args.single_host_pipeline_local_lower_split_layer,
    )
    _set(
        env,
        "GLM_SINGLE_HOST_PIPELINE_LOCAL_UPPER_SPLIT_LAYER",
        args.single_host_pipeline_local_upper_split_layer,
    )
    _set(
        env,
        "GLM_SINGLE_HOST_PIPELINE_LOCAL_LOWER_SPLIT_LAYERS",
        args.single_host_pipeline_local_lower_split_layers,
    )
    _set(
        env,
        "GLM_SINGLE_HOST_PIPELINE_LOCAL_UPPER_SPLIT_LAYERS",
        args.single_host_pipeline_local_upper_split_layers,
    )
    _set_switch(
        env,
        "GLM_SINGLE_HOST_PIPELINE_LOCAL_ABORT_ON_DIRTY_STAGE",
        args.single_host_pipeline_local_abort_on_dirty_stage,
    )
    _set(env, "GLM_LOCAL_SEQUENTIAL_REMOTE_WORKERS", args.local_sequential_remote_workers)
    _set(env, "GLM_LOCAL_SEQUENTIAL_REMOTE_TMP_DIR", args.local_sequential_remote_tmp_dir)
    _set(env, "GLM_LOCAL_SEQUENTIAL_REMOTE_DIRTY_RETRIES", args.local_sequential_remote_dirty_retries)
    _set(
        env,
        "GLM_LOCAL_SEQUENTIAL_REMOTE_RETRY_SLEEP_SECONDS",
        args.local_sequential_remote_retry_sleep_seconds,
    )
    _set(env, "GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB", args.single_host_mlx_wired_limit_gb)
    _set(
        env,
        "GLM_SINGLE_HOST_SOURCE_MEMORY_GUARD_RATIO",
        args.single_host_source_memory_guard_ratio,
    )
    if args.route_trace_layer:
        env["GLM_ROUTE_TRACE_LAYERS"] = ",".join(str(item) for item in args.route_trace_layer)
    _set(env, "GLM_VQ_ARTIFACT_DIR", args.vq_artifact_dir)
    _set(env, "GLM_ENGINE", args.engine)

    root = Path(__file__).resolve().parents[1]
    command = ["bash", str(root / "scripts/run_glm45_air_distributed_cleanroom_cache.sh")]
    raise SystemExit(subprocess.run(command, env=env).returncode)


if __name__ == "__main__":
    main()
