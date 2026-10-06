#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

default_seed_artifact_dir="artifacts/glm-4.5-air-dynamic3p0-target23-joint-gate-up-down-nextcycle-r25-r26-s48-lr1-w2-m1-nll0p5-20260701"
default_report_teacher_jsonl="artifacts/quality/glm45-air-teacher-cache-air-vq-ladder-report-v1-full-logits-route-trace-clean/metadata.jsonl"
default_selection_teacher_jsonl="artifacts/quality/glm45-air-teacher-cache-air-vq-ladder-select-v1-full-logits-route-trace-clean/metadata.jsonl"
default_report_eval_jsonl="artifacts/quality/glm45-air-dynamic3p0-r26-lora-l45-gud-math8-r4-init0p05-s12-lr0p5-w2-m1-nll0p5-report128-20260701.jsonl"
default_selection_eval_jsonl="artifacts/quality/glm45-air-dynamic3p0-r26-lora-l45-gud-math8-r4-init0p05-s12-lr0p5-w2-m1-nll0p5-select128-20260701.jsonl"
default_holdout_eval_jsonl="artifacts/quality/glm45-air-dynamic3p0-r26-lora-l45-gud-math8-r4-init0p05-s12-lr0p5-w2-m1-nll0p5-holdout128-20260701.jsonl"
default_imatrix_output_dir="artifacts/imatrix/glm45-air-public-calibration"
default_imatrix_manifest="$default_imatrix_output_dir/imatrix-manifest.json"
default_high_bit_artifact_dir="artifacts/glm-4.5-air-highbit-routed-imatrix-reference-20260701"
default_reproduced_artifact_dir="artifacts/glm45-air-public-reproduction-low-rank-residual-r4"
default_reproduced_train_jsonl="artifacts/quality/glm45-air-public-reproduction-low-rank-residual-r4.jsonl"
new_model_template="docs/KEEP_NEW_MODEL_FAMILY_TEMPLATE.md"

output_dir_from_args() {
  local output_dir="artifacts/rc/glm45-air-balanced-r4-20260701"
  while (($# > 0)); do
    case "$1" in
      --output-dir)
        if (($# >= 2)); then
          output_dir="$2"
          shift 2
          continue
        fi
        ;;
      --output-dir=*)
        output_dir="${1#--output-dir=}"
        ;;
    esac
    shift
  done
  printf '%s\n' "$output_dir"
}

imatrix_manifest_from_args() {
  local imatrix_manifest="$default_imatrix_manifest"
  while (($# > 0)); do
    case "$1" in
      --imatrix-manifest)
        if (($# >= 2)); then
          imatrix_manifest="$2"
          shift 2
          continue
        fi
        ;;
      --imatrix-manifest=*)
        imatrix_manifest="${1#--imatrix-manifest=}"
        ;;
    esac
    shift
  done
  printf '%s\n' "$imatrix_manifest"
}

filter_imatrix_manifest_args() {
  while (($# > 0)); do
    case "$1" in
      --imatrix-manifest)
        if (($# >= 2)); then
          shift 2
        else
          shift
        fi
        continue
        ;;
      --imatrix-manifest=*)
        shift
        continue
        ;;
    esac
    printf '%s\0' "$1"
    shift
  done
}

require_file_for_real_run() {
  local path="$1"
  local hint="$2"
  if [[ "${GLM45_AIR_RC_DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi
  if [[ ! -f "$path" ]]; then
    printf 'required file is missing: %s\n%s\n' "$path" "$hint" >&2
    exit 1
  fi
}

usage() {
  cat <<'EOF'
Usage: scripts/glm45_air_rc.sh <command> [pipeline args]

Commands:
  env-preflight        Check MLX, mlx-lm, Metal kernel, Hadamard, and
                       safetensors prerequisites.
  preflight            Check local RC inputs without loading the model.
  preflight-md         Render the preflight as Markdown.
  memory-preflight     Check host pageout/swapout quietness without loading
                       the model.
  packet               Generate rc-summary, RC_SUMMARY, quality-focus,
                       quality-frontier, and quality-plan outputs.
  summary              Generate only rc-summary.json and RC_SUMMARY.md.
  focus                Export only quality-focus.json with the summary.
  attribute            Export prompt/token attribution JSON for report,
                       selection, and holdout accepted RC rows.
  attribute-report     Export prompt/token attribution JSON for report rows.
  attribute-selection  Export prompt/token attribution JSON for selection rows.
  attribute-holdout    Export prompt/token attribution JSON for holdout rows.
  frontier             Export only quality-frontier.json with the summary.
  plan                 Export only quality-plan.json with the summary.
  reproduce            Train the public rank-4 sidecar recipe from the protected
                       seed, then run full verification on the reproduced artifact.
  verify               Run audit, report/selection/holdout evals, paired Lane S,
                       and write the full RC packet from fresh outputs.
  audit                Run the artifact audit through the RC pipeline.
  eval-report          Rerun report split eval through the RC pipeline.
  eval-selection       Rerun selection split eval through the RC pipeline.
  eval-holdout         Rerun holdout split eval through the RC pipeline.
  benchmark-lane-s     Rerun q2 and balanced-RC Lane S benchmarks together.
  benchmark-q2         Rerun the q2 Lane S control benchmark.
  benchmark-candidate  Rerun the balanced-RC Lane S candidate benchmark.
  benchmark-candidate-cache1
                       Run the candidate Lane S cache-limit diagnostic.
  benchmark-candidate-cache-sweep
                       Run candidate cache diagnostics at 1, 2, and 4 GiB.
  collect-imatrix      Collect routed projection calibration/imatrix sidecars.
  materialize-sweep    Materialize a bounded dynamic-imatrix VQ sweep.
  train-low-rank       Train the public rank-4 low-rank residual sidecar recipe.
  build                Build a declarative recipe via `keep build`. With
                       GLM45_AIR_RC_DRY_RUN=1 this runs `keep build --dry-run`.
  new-model-template   Print the reusable checklist for adding a model family.
  help                 Show this message.

Examples:
  scripts/glm45_air_rc.sh env-preflight
  scripts/glm45_air_rc.sh preflight
  scripts/glm45_air_rc.sh packet --overwrite
  scripts/glm45_air_rc.sh attribute --overwrite
  scripts/glm45_air_rc.sh reproduce --overwrite --output-dir artifacts/rc/glm45-air-public-reproduction-r4
  scripts/glm45_air_rc.sh verify --overwrite --output-dir artifacts/rc/glm45-air-balanced-r4-fresh
  scripts/glm45_air_rc.sh audit --overwrite
  scripts/glm45_air_rc.sh build recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml

Extra args are passed through to benchmarks/run_glm45_air_rc_pipeline.py.
Use --output-dir to choose where generated RC outputs are written.
For collect-imatrix, materialize-sweep, and train-low-rank, extra args are
passed through to the underlying workflow script after the default public recipe.
EOF
}

run_command() {
  local command=("$@")
  if [[ "${GLM45_AIR_RC_DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    return 0
  fi
  "${command[@]}"
}

run_pipeline() {
  run_command uv run python benchmarks/run_glm45_air_rc_pipeline.py "$@"
}

run_low_rank_recipe() {
  local artifact_dir="$1"
  local append_jsonl="$2"
  shift 2
  run_command uv run python benchmarks/finetune_glm45_air_vq_continuous.py \
    --selection-teacher-jsonl "$default_selection_teacher_jsonl" \
    --selection-cache-root "$(dirname "$default_selection_teacher_jsonl")" \
    --validation-teacher-jsonl "$default_report_teacher_jsonl" \
    --validation-cache-root "$(dirname "$default_report_teacher_jsonl")" \
    --seed-artifact-dir "$default_seed_artifact_dir" \
    --output-dir "$artifact_dir" \
    --prefill-engine nax_e8p \
    --layer 45 \
    --projections gate_proj up_proj down_proj \
    --trainable low_rank_residual \
    --low-rank 4 \
    --low-rank-init-scale 0.05 \
    --steps 12 \
    --learning-rate 0.5 \
    --target-nll-weight 0.5 \
    --teacher-top1-margin-weight 2 \
    --teacher-top1-margin 1 \
    --loss-scope final_layer_selected \
    --surrogate-projections gate_proj up_proj down_proj \
    --surrogate-output-chunk-size 256 \
    --train-cache both \
    --max-train-rows 8 \
    --train-row-indices 44,45,46,47,48,49,50,51 \
    --max-positions 16 \
    --append-jsonl "$append_jsonl" \
    "$@"
}

run_attribution() {
  local split_name="$1"
  local baseline_jsonl="$2"
  local output_json="$3"
  shift 3
  local attribution_args=()
  local overwrite=0
  while (($# > 0)); do
    case "$1" in
      --output-dir)
        if (($# >= 2)); then
          shift 2
        else
          shift
        fi
        continue
        ;;
      --output-dir=*)
        shift
        continue
        ;;
      --overwrite)
        overwrite=1
        shift
        continue
        ;;
    esac
    attribution_args+=("$1")
    shift
  done
  local command=(
    uv run python benchmarks/analyze_glm45_air_teacher_cache_attribution.py
    --baseline-jsonl "$baseline_jsonl" \
    --baseline-label "balanced_rc_${split_name}" \
    --output-json "$output_json"
  )
  if ((${#attribution_args[@]} > 0)); then
    command+=("${attribution_args[@]}")
  fi
  if [[ "${GLM45_AIR_RC_DRY_RUN:-0}" != "1" ]]; then
    mkdir -p "$(dirname "$output_json")"
    if [[ -e "$output_json" && "$overwrite" != "1" ]]; then
      printf 'attribution output already exists: %s; pass --overwrite to replace it\n' "$output_json" >&2
      exit 1
    fi
  fi
  run_command "${command[@]}"
}

command="${1:-help}"
if (($# > 0)); then
  shift
fi

output_dir="$(output_dir_from_args "$@")"

case "$command" in
  env-preflight)
    run_command uv run python scripts/verify_env.py "$@"
    ;;
  preflight)
    run_pipeline --preflight "$@"
    ;;
  preflight-md)
    run_pipeline --preflight --preflight-format markdown "$@"
    ;;
  memory-preflight)
    run_command uv run python benchmarks/bench_glm45_air_quant_compare.py \
      --memory-quiet-preflight \
      --memory-quiet-window-seconds 3 \
      --memory-quiet-max-attempts 5 \
      "$@"
    ;;
  packet)
    run_pipeline "$@" \
      --write-focus-json "$output_dir/quality-focus.json" \
      --write-quality-frontier-json "$output_dir/quality-frontier.json" \
      --write-quality-plan-json "$output_dir/quality-plan.json"
    ;;
  summary)
    run_pipeline "$@"
    ;;
  focus)
    run_pipeline "$@" --write-focus-json "$output_dir/quality-focus.json"
    ;;
  attribute)
    run_attribution report "$default_report_eval_jsonl" "$output_dir/report-attribution.json" "$@"
    run_attribution selection "$default_selection_eval_jsonl" "$output_dir/selection-attribution.json" "$@"
    run_attribution holdout "$default_holdout_eval_jsonl" "$output_dir/holdout-attribution.json" "$@"
    ;;
  attribute-report)
    run_attribution report "$default_report_eval_jsonl" "$output_dir/report-attribution.json" "$@"
    ;;
  attribute-selection)
    run_attribution selection "$default_selection_eval_jsonl" "$output_dir/selection-attribution.json" "$@"
    ;;
  attribute-holdout)
    run_attribution holdout "$default_holdout_eval_jsonl" "$output_dir/holdout-attribution.json" "$@"
    ;;
  frontier)
    run_pipeline "$@" --write-quality-frontier-json "$output_dir/quality-frontier.json"
    ;;
  plan)
    run_pipeline "$@" --write-quality-plan-json "$output_dir/quality-plan.json"
    ;;
  reproduce)
    reproduce_args=()
    while (($# > 0)); do
      case "$1" in
        --output-dir)
          shift 2
          continue
          ;;
        --output-dir=*)
          shift
          continue
          ;;
      esac
      reproduce_args+=("$1")
      shift
    done
    run_low_rank_recipe "$default_reproduced_artifact_dir" "$default_reproduced_train_jsonl"
    run_pipeline \
      --execute audit \
      --execute eval_report \
      --execute eval_selection \
      --execute eval_holdout \
      --execute lane_s_q2_control \
      --execute lane_s_candidate \
      "${reproduce_args[@]}" \
      --artifact-dir "$default_reproduced_artifact_dir" \
      --seed-artifact-dir "$default_seed_artifact_dir" \
      --output-dir "$output_dir" \
      --write-focus-json "$output_dir/quality-focus.json" \
      --write-quality-frontier-json "$output_dir/quality-frontier.json" \
      --write-quality-plan-json "$output_dir/quality-plan.json"
    ;;
  verify)
    run_pipeline \
      --execute audit \
      --execute eval_report \
      --execute eval_selection \
      --execute eval_holdout \
      --execute lane_s_q2_control \
      --execute lane_s_candidate \
      "$@" \
      --write-focus-json "$output_dir/quality-focus.json" \
      --write-quality-frontier-json "$output_dir/quality-frontier.json" \
      --write-quality-plan-json "$output_dir/quality-plan.json"
    ;;
  audit)
    run_pipeline --execute audit "$@"
    ;;
  eval-report)
    run_pipeline --execute eval_report "$@"
    ;;
  eval-selection)
    run_pipeline --execute eval_selection "$@"
    ;;
  eval-holdout)
    run_pipeline --execute eval_holdout "$@"
    ;;
  benchmark-lane-s)
    run_pipeline --execute lane_s_q2_control --execute lane_s_candidate "$@"
    ;;
  benchmark-q2)
    run_pipeline --execute lane_s_q2_control "$@"
    ;;
  benchmark-candidate)
    run_pipeline --execute lane_s_candidate "$@"
    ;;
  benchmark-candidate-cache1)
    run_pipeline --execute lane_s_candidate_cache1_diagnostic "$@"
    ;;
  benchmark-candidate-cache-sweep)
    run_pipeline \
      --execute lane_s_candidate_cache1_diagnostic \
      --execute lane_s_candidate_cache2_diagnostic \
      --execute lane_s_candidate_cache4_diagnostic \
      "$@"
    ;;
  collect-imatrix)
    run_command uv run python benchmarks/collect_glm45_air_imatrix.py \
      --artifact-dir "$default_seed_artifact_dir" \
      --output-dir "$default_imatrix_output_dir" \
      --prompt-set air_vq_ladder_select_v1 \
      --max-prompts 128 \
      --layers 31,36,41,45 \
      --projections gate_proj,up_proj,down_proj \
      "$@"
    ;;
  materialize-sweep)
    imatrix_manifest="$(imatrix_manifest_from_args "$@")"
    materialize_args=()
    while IFS= read -r -d '' arg; do
      materialize_args+=("$arg")
    done < <(filter_imatrix_manifest_args "$@")
    require_file_for_real_run \
      "$imatrix_manifest" \
      "Run scripts/glm45_air_rc.sh collect-imatrix first, or pass --imatrix-manifest <path>."
    materialize_command=(
      uv run python benchmarks/materialize_glm45_air_dynamic_imatrix_sweep.py
      --imatrix-manifest "$imatrix_manifest"
      --baseline-artifact-dir "$default_seed_artifact_dir"
      --high-bit-artifact-dir "$default_high_bit_artifact_dir"
      --output-root artifacts/glm45-air-public-dynamic-imatrix-sweep
      --candidate-prefix public-dynamic-imatrix
      --budget 2.0
      --budget 2.4
    )
    if ((${#materialize_args[@]} > 0)); then
      materialize_command+=("${materialize_args[@]}")
    fi
    run_command "${materialize_command[@]}"
    ;;
  train-low-rank)
    run_low_rank_recipe \
      artifacts/glm45-air-public-workflow-low-rank-residual-r4 \
      artifacts/quality/glm45-air-public-workflow-low-rank-residual-r4.jsonl \
      "$@"
    ;;
  build)
    build_args=("$@")
    if [[ "${GLM45_AIR_RC_DRY_RUN:-0}" == "1" ]]; then
      build_args+=(--dry-run)
    fi
    uv run keep build "${build_args[@]}"
    ;;
  new-model-template)
    run_command sed -n '1,260p' "$new_model_template" "$@"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    printf 'Unknown command: %s\n\n' "$command" >&2
    usage >&2
    exit 2
    ;;
esac
