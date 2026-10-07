#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

repo_abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$ROOT" "$1" ;;
  esac
}

peer_repo_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PEER_REPO" "$1" ;;
  esac
}

HOSTFILE="${GLM_JACCL_HOSTFILE:-/tmp/glm-jaccl-hostfile.json}"
PEER_SSH="${GLM_PEER_SSH:-jackmazac@peer-host.local}"
PEER_REPO="${GLM_PEER_REPO:-/Users/jackmazac/Development/mlx}"
PEER_UV="${GLM_PEER_UV:-/Users/jackmazac/.local/bin/uv}"
KNOWN_HOSTS="${GLM_KNOWN_HOSTS:-/tmp/glm-codex-peer-known-hosts}"

TMP_PYTHON="${GLM_TMP_PYTHON:-/tmp/glm-mlx-python}"
TMP_EXPORTER="${GLM_TMP_EXPORTER:-/tmp/glm_export_distributed_teacher_cache.py}"
TMP_PROBE="${GLM_TMP_PROBE:-/tmp/glm_jaccl_cleanroom_probe.py}"
TMP_WIRED_PROBE="${GLM_TMP_WIRED_PROBE:-/tmp/glm_mlx_wired_limit_probe.py}"
TMP_SRC_ROOT="${GLM_TMP_SRC_ROOT:-/tmp/keep-src}"
TMP_SRC_ARCHIVE="${GLM_TMP_SRC_ARCHIVE:-/tmp/keep-src.tar}"

REQUIRED_WIRED_MB="${GLM_REQUIRED_WIRED_MB:-0}"
MLX_WIRED_LIMIT_GB="${GLM_MLX_WIRED_LIMIT_GB:-}"
MLX_CACHE_LIMIT_GB="${GLM_MLX_CACHE_LIMIT_GB:-${GLM_CACHE_LIMIT_GB:-0}}"
LM_HEAD_CHUNK_ROWS="${GLM_LM_HEAD_CHUNK_ROWS:-8192}"
TOP_K="${GLM_TOP_K:-128}"
MAX_POSITIONS="${GLM_MAX_POSITIONS:-128}"
PROMPT_SET="${GLM_PROMPT_SET:-base}"
PROMPT_IDS="${GLM_PROMPT_IDS:-}"
LAYER_SPLIT="${GLM_LAYER_SPLIT:-}"
DEFAULT_RANK_VIEW_ROOTS_JSON='{"0":"/tmp/glm45-air-pipeline-view-load-probe/rank-0","1":"/Users/jackmazac/Development/mlx/glm45-air-pipeline-views/rank-1"}'
RANK_VIEW_ROOTS_JSON="${GLM_RANK_VIEW_ROOTS_JSON:-$DEFAULT_RANK_VIEW_ROOTS_JSON}"
PEER_RANK0_VIEW_ROOT="${GLM_PEER_RANK0_VIEW_ROOT:-}"
PEER_RANK1_VIEW_ROOT="${GLM_PEER_RANK1_VIEW_ROOT:-}"
PEER_SOURCE_DIR="${GLM_PEER_SOURCE_DIR:-}"
PEER_SOURCE_STAGE_SSH="${GLM_PEER_SOURCE_STAGE_SSH:-$PEER_SSH}"
STAGE_PEER_SOURCE="${GLM_STAGE_PEER_SOURCE:-0}"
PEER_SOURCE_MIN_FREE_GB="${GLM_PEER_SOURCE_MIN_FREE_GB:-16}"
LOCAL_SEQUENTIAL_STAGE_VIEW_ROOTS_JSON="${GLM_LOCAL_SEQUENTIAL_STAGE_VIEW_ROOTS_JSON:-}"
LOCAL_SEQUENTIAL_REMOTE_WORKERS="${GLM_LOCAL_SEQUENTIAL_REMOTE_WORKERS:-}"
LOCAL_SEQUENTIAL_REMOTE_SSH="${GLM_LOCAL_SEQUENTIAL_REMOTE_SSH:-$PEER_SSH}"
LOCAL_SEQUENTIAL_REMOTE_REPO="${GLM_LOCAL_SEQUENTIAL_REMOTE_REPO:-$PEER_REPO}"
LOCAL_SEQUENTIAL_REMOTE_PYTHON="${GLM_LOCAL_SEQUENTIAL_REMOTE_PYTHON:-$PEER_UV run python}"
LOCAL_SEQUENTIAL_REMOTE_TMP_DIR="${GLM_LOCAL_SEQUENTIAL_REMOTE_TMP_DIR:-/tmp/glm-local-sequential-workers}"
LOCAL_SEQUENTIAL_REMOTE_STAGE_VIEW_ROOTS_JSON="${GLM_LOCAL_SEQUENTIAL_REMOTE_STAGE_VIEW_ROOTS_JSON:-}"
LOCAL_SEQUENTIAL_REMOTE_DIRTY_RETRIES="${GLM_LOCAL_SEQUENTIAL_REMOTE_DIRTY_RETRIES:-0}"
LOCAL_SEQUENTIAL_REMOTE_RETRY_SLEEP_SECONDS="${GLM_LOCAL_SEQUENTIAL_REMOTE_RETRY_SLEEP_SECONDS:-0}"
PREFLIGHT_ONLY="${GLM_PREFLIGHT_ONLY:-0}"
NO_FULL_LOGITS="${GLM_NO_FULL_LOGITS:-0}"
INCLUDE_ROUTE_TRACE="${GLM_INCLUDE_ROUTE_TRACE:-0}"
ROUTE_TRACE_ONLY="${GLM_ROUTE_TRACE_ONLY:-0}"
ROUTE_TRACE_LAYERS="${GLM_ROUTE_TRACE_LAYERS:-}"
SKIP_LOCAL_CONSUME="${GLM_SKIP_LOCAL_CONSUME:-0}"
ALLOW_DIRTY_CACHE="${GLM_ALLOW_DIRTY_CACHE:-0}"
MEMORY_QUIET_PREFLIGHT="${GLM_MEMORY_QUIET_PREFLIGHT:-0}"
REQUIRE_MEMORY_QUIET_PREFLIGHT="${GLM_REQUIRE_MEMORY_QUIET_PREFLIGHT:-0}"
MEMORY_QUIET_SECONDS="${GLM_MEMORY_QUIET_SECONDS:-3}"
MEMORY_QUIET_MAX_ATTEMPTS="${GLM_MEMORY_QUIET_MAX_ATTEMPTS:-1}"
MEMORY_QUIET_MIN_FREE_GB="${GLM_MEMORY_QUIET_MIN_FREE_GB:-0}"
MEMORY_QUIET_STAGE_VIEW_ROOTS_JSON="${GLM_MEMORY_QUIET_STAGE_VIEW_ROOTS_JSON:-$LOCAL_SEQUENTIAL_STAGE_VIEW_ROOTS_JSON}"
MEMORY_QUIET_STAGE_VIEW_MARGIN_GB="${GLM_MEMORY_QUIET_STAGE_VIEW_MARGIN_GB:-0}"
SINGLE_HOST_FALLBACK="${GLM_SINGLE_HOST_FALLBACK:-0}"
SINGLE_HOST_MODEL_PATH="${GLM_SINGLE_HOST_MODEL_PATH:-zai-org/GLM-4.5-Air}"
SINGLE_HOST_REVISION="${GLM_SINGLE_HOST_REVISION:-}"
SINGLE_HOST_TEACHER_KIND="${GLM_SINGLE_HOST_TEACHER_KIND:-bf16_source}"
SINGLE_HOST_LAZY="${GLM_SINGLE_HOST_LAZY:-0}"
SINGLE_HOST_PIPELINE_LOCAL="${GLM_SINGLE_HOST_PIPELINE_LOCAL:-0}"
SINGLE_HOST_PIPELINE_LOCAL_STAGE_PROCESSES="${GLM_SINGLE_HOST_PIPELINE_LOCAL_STAGE_PROCESSES:-0}"
SINGLE_HOST_PIPELINE_LOCAL_HEAD_PROCESS="${GLM_SINGLE_HOST_PIPELINE_LOCAL_HEAD_PROCESS:-0}"
SINGLE_HOST_PIPELINE_LOCAL_LOWER_SPLIT_LAYER="${GLM_SINGLE_HOST_PIPELINE_LOCAL_LOWER_SPLIT_LAYER:-}"
SINGLE_HOST_PIPELINE_LOCAL_UPPER_SPLIT_LAYER="${GLM_SINGLE_HOST_PIPELINE_LOCAL_UPPER_SPLIT_LAYER:-}"
SINGLE_HOST_PIPELINE_LOCAL_LOWER_SPLIT_LAYERS="${GLM_SINGLE_HOST_PIPELINE_LOCAL_LOWER_SPLIT_LAYERS:-}"
SINGLE_HOST_PIPELINE_LOCAL_UPPER_SPLIT_LAYERS="${GLM_SINGLE_HOST_PIPELINE_LOCAL_UPPER_SPLIT_LAYERS:-}"
SINGLE_HOST_PIPELINE_LOCAL_ABORT_ON_DIRTY_STAGE="${GLM_SINGLE_HOST_PIPELINE_LOCAL_ABORT_ON_DIRTY_STAGE:-0}"
MEMORY_QUIET_STAGE_VIEW_WORKER="${GLM_MEMORY_QUIET_STAGE_VIEW_WORKER:-}"
if [[ -z "$MEMORY_QUIET_STAGE_VIEW_WORKER" && "$SINGLE_HOST_PIPELINE_LOCAL_STAGE_PROCESSES" == "1" && -n "$LOCAL_SEQUENTIAL_STAGE_VIEW_ROOTS_JSON" && -n "$SINGLE_HOST_PIPELINE_LOCAL_LOWER_SPLIT_LAYERS" ]]; then
  MEMORY_QUIET_STAGE_VIEW_WORKER="lower-embed"
fi
SINGLE_HOST_MLX_WIRED_LIMIT_GB="${GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB:-}"
SINGLE_HOST_SOURCE_MEMORY_GUARD_RATIO="${GLM_SINGLE_HOST_SOURCE_MEMORY_GUARD_RATIO:-}"
LOCAL_DIRECT_IF="${GLM_LOCAL_DIRECT_IF:-en6}"
PEER_DIRECT_IF="${GLM_PEER_DIRECT_IF:-en1}"
LOCAL_DIRECT_IP="${GLM_LOCAL_DIRECT_IP:-198.51.100.1}"
PEER_DIRECT_IP="${GLM_PEER_DIRECT_IP:-198.51.100.2}"
LOCAL_RDMA_DEVICE="${GLM_LOCAL_RDMA_DEVICE:-rdma_en1}"
PEER_RDMA_DEVICE="${GLM_PEER_RDMA_DEVICE:-rdma_en1}"
SKIP_DIRECT_LINK_CHECK="${GLM_SKIP_DIRECT_LINK_CHECK:-0}"
LAUNCH_LOG_DIR="${GLM_LAUNCH_LOG_DIR:-}"
LAUNCH_LOG_SEQ=0

OUTPUT_DIR="${GLM_OUTPUT_DIR:-$ROOT/artifacts/quality/glm45-air-teacher-cache-distributed-cleanroom}"
VALIDATION_JSONL="${GLM_VALIDATION_JSONL:-$OUTPUT_DIR-validation.jsonl}"
LOCAL_JSONL="${GLM_LOCAL_JSONL:-$OUTPUT_DIR-local.jsonl}"
VQ_ARTIFACT_DIR="${GLM_VQ_ARTIFACT_DIR:-$ROOT/artifacts/glm-4.5-air-vq}"
ENGINE="${GLM_ENGINE:-vq_e1_routed}"

OUTPUT_DIR="$(repo_abs_path "$OUTPUT_DIR")"
VALIDATION_JSONL="$(repo_abs_path "$VALIDATION_JSONL")"
LOCAL_JSONL="$(repo_abs_path "$LOCAL_JSONL")"
VQ_ARTIFACT_DIR="$(repo_abs_path "$VQ_ARTIFACT_DIR")"

SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=5
  -o UserKnownHostsFile="$KNOWN_HOSTS"
  -o StrictHostKeyChecking=accept-new
)

info() {
  printf '[cleanroom] %s\n' "$*" >&2
}

die() {
  printf '[cleanroom] ERROR: %s\n' "$*" >&2
  exit 1
}

mlx_launch() {
  uv run python -c 'from mlx._distributed_utils.launch import main; main()' "$@"
}

mlx_launch_checked() {
  local output_file status
  output_file="$(mktemp)"
  set +e
  mlx_launch "$@" 2>&1 | tee "$output_file"
  status="${PIPESTATUS[0]}"
  set -e
  if [[ -n "$LAUNCH_LOG_DIR" ]]; then
    mkdir -p "$LAUNCH_LOG_DIR"
    LAUNCH_LOG_SEQ=$((LAUNCH_LOG_SEQ + 1))
    cp "$output_file" "$LAUNCH_LOG_DIR/launch-${LAUNCH_LOG_SEQ}.log"
  fi
  if (( status != 0 )); then
    rm -f "$output_file"
    return "$status"
  fi
  if grep -Eq "Node with rank .*exited with code -?[1-9][0-9]*" "$output_file"; then
    rm -f "$output_file"
    return 1
  fi
  if grep -Fq "Traceback (most recent call last)" "$output_file"; then
    rm -f "$output_file"
    return 1
  fi
  rm -f "$output_file"
}

require_file() {
  [[ -f "$1" ]] || die "missing required file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "missing required directory: $1"
}

rank_view_roots_json() {
  "$ROOT/.venv/bin/python" - "$RANK_VIEW_ROOTS_JSON" "$PEER_RANK1_VIEW_ROOT" "$ROOT" "$PEER_REPO" <<'PY'
import json
import sys
from pathlib import Path

roots = json.loads(sys.argv[1])
peer_rank1_root = sys.argv[2]
local_root = Path(sys.argv[3])
peer_repo = sys.argv[4]

def local_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else local_root / path)

def peer_path(value: str) -> str:
    path = Path(value)
    return str(path) if path.is_absolute() else f"{peer_repo}/{value}"

if "0" in roots:
    roots["0"] = local_path(str(roots["0"]))
if peer_rank1_root:
    roots["1"] = peer_path(peer_rank1_root)
elif "1" in roots:
    roots["1"] = peer_path(str(roots["1"]))
print(json.dumps(roots, separators=(",", ":"), sort_keys=True))
PY
}

rank_view_root() {
  local rank="$1"
  "$ROOT/.venv/bin/python" - "$rank" "$(rank_view_roots_json)" <<'PY'
import json
import sys

rank = sys.argv[1]
roots = json.loads(sys.argv[2])
value = roots.get(rank)
if not value:
    raise SystemExit(1)
print(value)
PY
}

local_rank_view_root() {
  local rank="$1"
  "$ROOT/.venv/bin/python" - "$rank" "$RANK_VIEW_ROOTS_JSON" "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

rank = sys.argv[1]
roots = json.loads(sys.argv[2])
repo_root = Path(sys.argv[3])
value = roots.get(rank)
if not value:
    raise SystemExit(1)
path = Path(str(value))
print(path if path.is_absolute() else repo_root / path)
PY
}

peer_home() {
  ssh "${SSH_OPTS[@]}" "$PEER_SSH" 'printf "%s\n" "$HOME"'
}

rsync_remote_shell() {
  local rendered="ssh"
  local opt
  for opt in "${SSH_OPTS[@]}"; do
    printf -v rendered '%s %q' "$rendered" "$opt"
  done
  printf '%s\n' "$rendered"
}

peer_source_dir_for_rank1() {
  local local_rank1_root
  if [[ -n "$PEER_SOURCE_DIR" ]]; then
    peer_repo_path "$PEER_SOURCE_DIR"
    return
  fi
  local_rank1_root="$(local_rank_view_root 1)" || die "rank view roots JSON has no local rank 1 root"
  peer_source_dir_for_local_rank_root "$local_rank1_root"
}

peer_source_dir_for_local_rank_root() {
  local local_rank_root="$1"
  local peer_home_value
  peer_home_value="$(peer_home)"
  "$ROOT/.venv/bin/python" - "$local_rank_root" "$peer_home_value" <<'PY'
import json
import sys
from pathlib import Path

rank_root = Path(sys.argv[1])
peer_home = sys.argv[2]
metadata = json.loads((rank_root / "model.safetensors.index.json").read_text())["metadata"]
source_dir = str(metadata.get("source_dir") or "")
marker = "/.cache/huggingface/hub/"
if marker not in source_dir:
    raise SystemExit(f"cannot derive peer source dir from {source_dir!r}")
print(f"{peer_home}{marker}{source_dir.split(marker, 1)[1]}")
PY
}

peer_rank0_view_root() {
  if [[ -n "$PEER_RANK0_VIEW_ROOT" ]]; then
    peer_repo_path "$PEER_RANK0_VIEW_ROOT"
    return
  fi
  peer_repo_path "glm45-air-pipeline-views/rank-0"
}

stage_peer_rank1_source_files() {
  local local_rank1_root="$1"
  local peer_source_dir="$2"
  local stage_dir required_bytes required_count peer_source_quoted rsync_shell
  stage_dir="$(mktemp -d -t glm-rank1-source.XXXXXX)"
  read -r required_bytes required_count < <("$ROOT/.venv/bin/python" - "$local_rank1_root" "$stage_dir" <<'PY'
import os
import sys
from pathlib import Path

rank_root = Path(sys.argv[1])
stage_dir = Path(sys.argv[2])
total = 0
count = 0
stage_dir.mkdir(parents=True, exist_ok=True)
for child in sorted(rank_root.iterdir()):
    if not child.is_symlink():
        continue
    target = child.resolve(strict=True)
    if not target.is_file():
        raise SystemExit(f"rank1 source target is not a file: {child} -> {target}")
    link = stage_dir / child.name
    if link.exists() or link.is_symlink():
        link.unlink()
    os.symlink(target, link)
    total += target.stat().st_size
    count += 1
print(total, count)
PY
  )
  if [[ "$required_count" == "0" ]]; then
    rm -rf "$stage_dir"
    die "rank 1 view has no source symlinks to stage: $local_rank1_root"
  fi

  info "staging $required_count rank1 source files to peer ($required_bytes bytes, min free after stage ${PEER_SOURCE_MIN_FREE_GB} GiB)"
  printf -v peer_source_quoted '%q' "$peer_source_dir"
  ssh "${SSH_OPTS[@]}" "$PEER_SOURCE_STAGE_SSH" "python3 - ${peer_source_quoted} ${required_bytes} ${PEER_SOURCE_MIN_FREE_GB} <<'PY'
import shutil
import sys
from pathlib import Path

path = Path(sys.argv[1])
required = int(sys.argv[2])
min_free_gb = float(sys.argv[3])
probe = path
while not probe.exists() and probe != probe.parent:
    probe = probe.parent
usage = shutil.disk_usage(probe)
min_free = int(min_free_gb * 1024**3)
if usage.free < required + min_free:
    raise SystemExit(
        f'insufficient peer disk for rank1 source stage: free={usage.free} '
        f'required={required} min_free={min_free} probe={probe}'
    )
PY"
  ssh "${SSH_OPTS[@]}" "$PEER_SOURCE_STAGE_SSH" "mkdir -p ${peer_source_quoted}"
  rsync_shell="$(rsync_remote_shell)"
  rsync -aL --partial -e "$rsync_shell" "$stage_dir"/ "$PEER_SOURCE_STAGE_SSH:$peer_source_dir"/
  rm -rf "$stage_dir"
}

stage_peer_rank1_view_root() {
  local local_rank1_root peer_rank1_root peer_source_dir
  local peer_rank1_quoted peer_source_quoted archive archive_remote
  local_rank1_root="$(local_rank_view_root 1)" || die "rank view roots JSON has no local rank 1 root"
  peer_rank1_root="$(rank_view_root 1)" || die "rank view roots JSON has no peer rank 1 root"
  peer_source_dir="$(peer_source_dir_for_rank1)"

  info "peer rank1 source snapshot: $peer_source_dir"
  printf -v peer_source_quoted '%q' "$peer_source_dir"
  if [[ "$STAGE_PEER_SOURCE" == "1" ]]; then
    stage_peer_rank1_source_files "$local_rank1_root" "$peer_source_dir"
  else
    ssh "${SSH_OPTS[@]}" "$PEER_SSH" "test -d ${peer_source_quoted}" \
      || die "missing peer source snapshot: $peer_source_dir"
  fi

  archive="$(mktemp -t glm-rank1-view.XXXXXX.tar)"
  archive_remote="/tmp/glm-rank1-view.tar"
  tar -cf "$archive" -C "$local_rank1_root" .
  scp "${SSH_OPTS[@]}" "$archive" "$PEER_SSH:$archive_remote" >/dev/null
  rm -f "$archive"

  printf -v peer_rank1_quoted '%q' "$peer_rank1_root"
  ssh "${SSH_OPTS[@]}" "$PEER_SSH" "rm -rf ${peer_rank1_quoted} && mkdir -p ${peer_rank1_quoted} && tar -xf '$archive_remote' -C ${peer_rank1_quoted}"
  info "rewiring peer rank1 view symlinks to $peer_source_dir"
  ssh "${SSH_OPTS[@]}" "$PEER_SSH" "python3 - ${peer_rank1_quoted} ${peer_source_quoted} <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
source = Path(sys.argv[2])
missing = []
for child in root.iterdir():
    if child.is_symlink():
        target = source / child.name
        if not target.exists():
            missing.append(str(target))
            continue
        child.unlink()
        child.symlink_to(target)
if missing:
    raise SystemExit('missing peer source files: ' + ', '.join(missing[:8]))
PY" || die "failed to rewire peer rank1 view symlinks"
}

stage_peer_rank0_view_root() {
  local local_rank0_root peer_rank0_root peer_source_dir
  local peer_rank0_quoted peer_source_quoted archive archive_remote
  local_rank0_root="$(local_rank_view_root 0)" || die "rank view roots JSON has no local rank 0 root"
  peer_rank0_root="$(peer_rank0_view_root)"
  if [[ -n "$PEER_SOURCE_DIR" ]]; then
    peer_source_dir="$(peer_repo_path "$PEER_SOURCE_DIR")"
  else
    peer_source_dir="$(peer_source_dir_for_local_rank_root "$local_rank0_root")"
  fi

  info "peer rank0 source snapshot: $peer_source_dir"
  printf -v peer_source_quoted '%q' "$peer_source_dir"
  if [[ "$STAGE_PEER_SOURCE" == "1" ]]; then
    stage_peer_rank1_source_files "$local_rank0_root" "$peer_source_dir"
  else
    ssh "${SSH_OPTS[@]}" "$PEER_SSH" "test -d ${peer_source_quoted}" \
      || die "missing peer source snapshot: $peer_source_dir"
  fi

  archive="$(mktemp -t glm-rank0-view.XXXXXX.tar)"
  archive_remote="/tmp/glm-rank0-view.tar"
  tar -cf "$archive" -C "$local_rank0_root" .
  scp "${SSH_OPTS[@]}" "$archive" "$PEER_SSH:$archive_remote" >/dev/null
  rm -f "$archive"

  printf -v peer_rank0_quoted '%q' "$peer_rank0_root"
  ssh "${SSH_OPTS[@]}" "$PEER_SSH" "rm -rf ${peer_rank0_quoted} && mkdir -p ${peer_rank0_quoted} && tar -xf '$archive_remote' -C ${peer_rank0_quoted}"
  info "rewiring peer rank0 view symlinks to $peer_source_dir"
  ssh "${SSH_OPTS[@]}" "$PEER_SSH" "python3 - ${peer_rank0_quoted} ${peer_source_quoted} <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
source = Path(sys.argv[2])
missing = []
for child in root.iterdir():
    if child.is_symlink():
        target = source / child.name
        if not target.exists():
            missing.append(str(target))
            continue
        child.unlink()
        child.symlink_to(target)
if missing:
    raise SystemExit('missing peer source files: ' + ', '.join(missing[:8]))
PY" || die "failed to rewire peer rank0 view symlinks"
  printf '%s\n' "$peer_rank0_root"
}

local_sequential_remote_stage_roots_json() {
  local peer_rank0_root="$1"
  "$ROOT/.venv/bin/python" - "$LOCAL_SEQUENTIAL_REMOTE_WORKERS" "$SINGLE_HOST_PIPELINE_LOCAL_UPPER_SPLIT_LAYERS" "$peer_rank0_root" <<'PY'
import json
import sys

workers_raw = sys.argv[1]
upper_splits_raw = sys.argv[2]
peer_rank0_root = sys.argv[3]
split_count = len([item for item in upper_splits_raw.split(",") if item])
window_count = max(1, split_count + 1)
roots = {
    "upper": {"0": peer_rank0_root},
    "upper-pre": {"0": peer_rank0_root},
    "upper-final": {"0": peer_rank0_root},
    "head": {"0": peer_rank0_root},
}
for index in range(window_count):
    roots[f"upper-window-{index}"] = {"0": peer_rank0_root}
print(json.dumps(roots, sort_keys=True))
PY
}

require_rank_view_roots() {
  local rank0_root rank1_root rank1_quoted
  rank0_root="$(rank_view_root 0)" || die "rank view roots JSON has no rank 0 root"
  rank1_root="$(rank_view_root 1)" || die "rank view roots JSON has no rank 1 root"

  info "local rank0 view root: $rank0_root"
  require_file "$rank0_root/model.safetensors.index.json"
  require_file "$rank0_root/pipeline_view_plan.json"

  info "peer rank1 view root: $rank1_root"
  printf -v rank1_quoted '%q' "$rank1_root"
  ssh "${SSH_OPTS[@]}" "$PEER_SSH" \
    "test -f ${rank1_quoted}/model.safetensors.index.json && test -f ${rank1_quoted}/pipeline_view_plan.json" \
    || die "missing peer rank1 view root files under $rank1_root"
}

numeric_or_die() {
  local label="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || die "$label is not numeric: $value"
}

local_sysctl_mb() {
  sysctl -n iogpu.wired_limit_mb | tr -d '[:space:]'
}

peer_sysctl_mb() {
  ssh "${SSH_OPTS[@]}" "$PEER_SSH" 'sysctl -n iogpu.wired_limit_mb' | tr -d '[:space:]'
}

interface_status() {
  ifconfig "$1" | awk -F': ' '/status:/{print $2; exit}'
}

peer_interface_status() {
  ssh "${SSH_OPTS[@]}" "$PEER_SSH" "ifconfig '$PEER_DIRECT_IF'" \
    | awk -F': ' '/status:/{print $2; exit}'
}

require_direct_link() {
  if [[ "$SKIP_DIRECT_LINK_CHECK" == "1" ]]; then
    info "GLM_SKIP_DIRECT_LINK_CHECK=1; skipping direct Thunderbolt link preflight"
    return
  fi

  local local_status peer_status
  local_status="$(interface_status "$LOCAL_DIRECT_IF")"
  peer_status="$(peer_interface_status)"
  info "local ${LOCAL_DIRECT_IF} status=${local_status:-unknown} ip=${LOCAL_DIRECT_IP}"
  info "peer  ${PEER_DIRECT_IF} status=${peer_status:-unknown} ip=${PEER_DIRECT_IP}"
  if [[ "$local_status" != "active" || "$peer_status" != "active" ]]; then
    cat >&2 <<EOF
[cleanroom] Direct Thunderbolt link is not active on both hosts.
[cleanroom]
[cleanroom] Check locally:
[cleanroom]   ifconfig ${LOCAL_DIRECT_IF}
[cleanroom]   route -n get ${PEER_DIRECT_IP}
[cleanroom]
[cleanroom] Check peer:
[cleanroom]   ssh ${PEER_SSH} 'ifconfig ${PEER_DIRECT_IF}; route -n get ${LOCAL_DIRECT_IP}'
[cleanroom]
[cleanroom] If the cable/port is correct but the static addresses disappeared:
[cleanroom]   sudo ifconfig bridge0 down
[cleanroom]   sudo ifconfig ${LOCAL_DIRECT_IF} inet ${LOCAL_DIRECT_IP} netmask 255.255.255.252 up
[cleanroom]   ssh ${PEER_SSH} 'sudo ifconfig bridge0 down; sudo ifconfig ${PEER_DIRECT_IF} inet ${PEER_DIRECT_IP} netmask 255.255.255.252 up'
[cleanroom]
[cleanroom] Re-run this script after both interfaces show status: active.
EOF
    exit 21
  fi

  if ! ping -c 1 -W 1000 "$PEER_DIRECT_IP" >/dev/null 2>&1; then
    cat >&2 <<EOF
[cleanroom] Direct Thunderbolt interfaces are active, but ${LOCAL_DIRECT_IP} cannot ping ${PEER_DIRECT_IP}.
[cleanroom]
[cleanroom] Inspect ARP and routes:
[cleanroom]   arp -an | grep '${PEER_DIRECT_IP}'
[cleanroom]   route -n get ${PEER_DIRECT_IP}
[cleanroom]   ssh ${PEER_SSH} 'arp -an | grep ${LOCAL_DIRECT_IP}; route -n get ${LOCAL_DIRECT_IP}'
[cleanroom]
[cleanroom] Do not launch the distributed teacher-cache run until ping works.
EOF
    exit 22
  fi
}

require_wired_caps() {
  local local_mb peer_mb
  local_mb="$(local_sysctl_mb)"
  peer_mb="$(peer_sysctl_mb)"
  numeric_or_die "local iogpu.wired_limit_mb" "$local_mb"
  numeric_or_die "peer iogpu.wired_limit_mb" "$peer_mb"
  info "local iogpu.wired_limit_mb=$local_mb"
  info "peer  iogpu.wired_limit_mb=$peer_mb"
  if (( local_mb < REQUIRED_WIRED_MB || peer_mb < REQUIRED_WIRED_MB )); then
    local requested_limit
    if [[ -n "$MLX_WIRED_LIMIT_GB" ]]; then
      requested_limit="${MLX_WIRED_LIMIT_GB} GiB"
    else
      requested_limit="the system default"
    fi
    cat >&2 <<EOF
[cleanroom] The requested MLX wired limit is ${requested_limit}, but both
[cleanroom] Macs must first have iogpu.wired_limit_mb >= ${REQUIRED_WIRED_MB}.
[cleanroom]
[cleanroom] Run locally:
[cleanroom]   sudo sysctl iogpu.wired_limit_mb=${REQUIRED_WIRED_MB}
[cleanroom]
[cleanroom] Run on peer:
[cleanroom]   ssh ${PEER_SSH} 'sudo sysctl iogpu.wired_limit_mb=${REQUIRED_WIRED_MB}'
[cleanroom]
[cleanroom] This is intentionally not done by this script because high wired
[cleanroom] memory reduces macOS reclaimable headroom. Reboot or set the value
[cleanroom] back to 0 after the clean-room attempt if needed.
EOF
    exit 20
  fi
}

write_wrappers_and_probes() {
  rm -rf "$TMP_SRC_ROOT"
  mkdir -p "$TMP_SRC_ROOT"
  rm -f "$TMP_SRC_ARCHIVE"
  (cd "$ROOT/src" && tar -cf "$TMP_SRC_ARCHIVE" keep ramp)
  tar -xf "$TMP_SRC_ARCHIVE" -C "$TMP_SRC_ROOT"
  ssh "${SSH_OPTS[@]}" "$PEER_SSH" "rm -rf '$TMP_SRC_ROOT' && mkdir -p '$TMP_SRC_ROOT'"
  scp "${SSH_OPTS[@]}" "$TMP_SRC_ARCHIVE" "$PEER_SSH:$TMP_SRC_ARCHIVE" >/dev/null
  ssh "${SSH_OPTS[@]}" "$PEER_SSH" "tar -xf '$TMP_SRC_ARCHIVE' -C '$TMP_SRC_ROOT'"

  cat > "$TMP_PYTHON" <<EOF
#!/usr/bin/env bash
export PYTHONPATH="$TMP_SRC_ROOT:\${PYTHONPATH:-}"
exec "$ROOT/.venv/bin/python" "\$@"
EOF
  chmod +x "$TMP_PYTHON"

  ssh "${SSH_OPTS[@]}" "$PEER_SSH" "cat > '$TMP_PYTHON' <<EOF
#!/usr/bin/env bash
export PYTHONPATH=\"$TMP_SRC_ROOT:\${PYTHONPATH:-}\"
exec \"$PEER_REPO/.venv/bin/python\" \"\\\$@\"
EOF
chmod +x '$TMP_PYTHON'"

  cp "$ROOT/benchmarks/export_glm45_air_distributed_teacher_cache.py" "$TMP_EXPORTER"
  scp "${SSH_OPTS[@]}" "$ROOT/benchmarks/export_glm45_air_distributed_teacher_cache.py" "$PEER_SSH:$TMP_EXPORTER" >/dev/null

  cat > "$TMP_WIRED_PROBE" <<'PY'
import sys
import mlx.core as mx

gib = float(sys.argv[1])
limit = int(gib * 1024**3)
previous = mx.set_wired_limit(limit)
print({"requested_gib": gib, "requested_bytes": limit, "previous": int(previous)}, flush=True)
PY
  scp "${SSH_OPTS[@]}" "$TMP_WIRED_PROBE" "$PEER_SSH:$TMP_WIRED_PROBE" >/dev/null

  cat > "$TMP_PROBE" <<'PY'
import json
import socket
import mlx.core as mx

group = mx.distributed.init(backend="jaccl", strict=True)
x = mx.distributed.all_sum(mx.ones((1,), dtype=mx.float32), group=group, stream=mx.cpu)
mx.eval(x)
payload = {
    "host": socket.gethostname(),
    "rank": int(group.rank()),
    "size": int(group.size()),
    "sum0": float(x[0]),
}
print(json.dumps(payload), flush=True)
if int(group.size()) != 2:
    raise SystemExit(f"expected size 2, got {group.size()}")
if float(x[0]) != 2.0:
    raise SystemExit(f"expected all_sum 2.0, got {float(x[0])}")
PY
  scp "${SSH_OPTS[@]}" "$TMP_PROBE" "$PEER_SSH:$TMP_PROBE" >/dev/null
}

check_mlx_wired_limit() {
  if [[ -z "$MLX_WIRED_LIMIT_GB" ]]; then
    info "using system MLX wired memory limit; skipping mx.set_wired_limit probe"
    return
  fi
  info "checking mx.set_wired_limit(${MLX_WIRED_LIMIT_GB} GiB) on both hosts"
  uv run python "$TMP_WIRED_PROBE" "$MLX_WIRED_LIMIT_GB"
  ssh "${SSH_OPTS[@]}" "$PEER_SSH" "cd '$PEER_REPO' && '$PEER_UV' run python '$TMP_WIRED_PROBE' '$MLX_WIRED_LIMIT_GB'"
}

run_memory_quiet_preflight() {
  if [[ "$MEMORY_QUIET_PREFLIGHT" != "1" && "$REQUIRE_MEMORY_QUIET_PREFLIGHT" != "1" ]]; then
    return
  fi
  info "checking host memory quiet window before model/exporter work"
  local output status
  set +e
  output="$("$ROOT/.venv/bin/python" - "$MEMORY_QUIET_SECONDS" "$MEMORY_QUIET_MAX_ATTEMPTS" "$MEMORY_QUIET_MIN_FREE_GB" "$MEMORY_QUIET_STAGE_VIEW_ROOTS_JSON" "$MEMORY_QUIET_STAGE_VIEW_MARGIN_GB" "$ROOT" "$MEMORY_QUIET_STAGE_VIEW_WORKER" <<'PY'
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

window_seconds = float(sys.argv[1])
max_attempts = max(1, int(sys.argv[2]))
min_free_gb = float(sys.argv[3])
stage_view_roots_raw = sys.argv[4]
stage_view_margin_gb = float(sys.argv[5])
repo_root = Path(sys.argv[6])
stage_view_worker = sys.argv[7]
pattern = re.compile(r"^\s*(?P<name>[^:]+):\s+(?P<value>[0-9]+)\.?\s*$")
page_size_pattern = re.compile(r"page size of (?P<page_size>[0-9]+) bytes")

def resolve_repo_path(path: str) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return repo_root / resolved

def load_stage_view_roots(raw: str) -> dict:
    if not raw:
        return {}
    try:
        if raw.lstrip().startswith("{"):
            return json.loads(raw)
        return json.loads(resolve_repo_path(raw).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}

def stage_view_memory_requirement() -> dict:
    roots = load_stage_view_roots(stage_view_roots_raw)
    selected_roots = list(roots.items())
    worker_found = None
    if stage_view_worker:
        selected = roots.get(stage_view_worker)
        worker_found = isinstance(selected, dict)
        if worker_found:
            selected_roots = [(stage_view_worker, selected)]
    max_visible_bytes = 0
    max_required_bytes = 0
    plan_count = 0
    for _, stage_roots in selected_roots:
        if not isinstance(stage_roots, dict):
            continue
        for root in stage_roots.values():
            plan_path = resolve_repo_path(str(root)) / "pipeline_stage_view_plan.json"
            try:
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            plan_count += 1
            max_visible_bytes = max(
                max_visible_bytes,
                int(plan.get("visible_shard_file_bytes") or 0),
            )
            max_required_bytes = max(
                max_required_bytes,
                int(plan.get("required_present_tensor_bytes") or 0),
            )
    max_visible_gb = max_visible_bytes / (1024**3)
    max_required_gb = max_required_bytes / (1024**3)
    derived_min_free_gb = max_visible_gb + stage_view_margin_gb if plan_count else 0.0
    return {
        "stage_view_plan_count": plan_count,
        "stage_view_max_visible_shard_gb": max_visible_gb,
        "stage_view_max_required_tensor_gb": max_required_gb,
        "stage_view_margin_gb": stage_view_margin_gb,
        "stage_view_min_free_gb": derived_min_free_gb,
        "stage_view_worker": stage_view_worker or None,
        "stage_view_worker_found": worker_found,
    }

def collect():
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["vm_stat"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    counts = {"page_size": 16384}
    for line in result.stdout.splitlines():
        page_size_match = page_size_pattern.search(line)
        if page_size_match is not None:
            counts["page_size"] = int(page_size_match.group("page_size"))
            continue
        match = pattern.match(line)
        if match is None:
            continue
        counts[match.group("name").strip().lower().replace(" ", "_")] = int(
            match.group("value")
        )
    if "pageouts" not in counts or "swapouts" not in counts:
        return None
    return counts

last = {
    "available": False,
    "quiet": False,
    "attempts": 0,
    "window_seconds": window_seconds,
    **stage_view_memory_requirement(),
    "configured_min_free_gb": min_free_gb,
    "effective_min_free_gb": None,
    "free_gb": None,
    "inactive_gb": None,
    "purgeable_gb": None,
    "available_gb": None,
    "availability_basis": "free+speculative+inactive",
    "pageouts_delta": None,
    "swapouts_delta": None,
}
effective_min_free_gb = max(
    min_free_gb,
    float(last.get("stage_view_min_free_gb") or 0.0),
)
last["effective_min_free_gb"] = effective_min_free_gb
for attempt in range(1, max_attempts + 1):
    before = collect()
    time.sleep(window_seconds)
    after = collect()
    if before is None or after is None:
        last = {
            **last,
            "available": False,
            "attempts": attempt,
        }
        break
    pageouts_delta = int(after.get("pageouts", 0)) - int(before.get("pageouts", 0))
    swapouts_delta = int(after.get("swapouts", 0)) - int(before.get("swapouts", 0))
    free_pages = int(after.get("pages_free", 0)) + int(after.get("pages_speculative", 0))
    inactive_pages = int(after.get("pages_inactive", 0))
    purgeable_pages = int(after.get("pages_purgeable", 0))
    page_size = int(after.get("page_size", 16384))
    free_gb = free_pages * page_size / (1024**3)
    inactive_gb = inactive_pages * page_size / (1024**3)
    purgeable_gb = purgeable_pages * page_size / (1024**3)
    available_gb = (free_pages + inactive_pages) * page_size / (1024**3)
    last = {
        "available": True,
        "quiet": (
            pageouts_delta == 0
            and swapouts_delta == 0
            and available_gb >= effective_min_free_gb
        ),
        "configured_min_free_gb": min_free_gb,
        "effective_min_free_gb": effective_min_free_gb,
        "attempts": attempt,
        "window_seconds": window_seconds,
        "free_gb": free_gb,
        "inactive_gb": inactive_gb,
        "purgeable_gb": purgeable_gb,
        "available_gb": available_gb,
        "availability_basis": "free+speculative+inactive",
        "pageouts_delta": pageouts_delta,
        "swapouts_delta": swapouts_delta,
        "stage_view_plan_count": last.get("stage_view_plan_count", 0),
        "stage_view_max_visible_shard_gb": last.get(
            "stage_view_max_visible_shard_gb",
            0.0,
        ),
        "stage_view_max_required_tensor_gb": last.get(
            "stage_view_max_required_tensor_gb",
            0.0,
        ),
        "stage_view_margin_gb": stage_view_margin_gb,
        "stage_view_min_free_gb": last.get("stage_view_min_free_gb", 0.0),
        "stage_view_worker": last.get("stage_view_worker"),
        "stage_view_worker_found": last.get("stage_view_worker_found"),
    }
    if last["quiet"]:
        break
print(json.dumps(last, sort_keys=True))
raise SystemExit(0 if last.get("quiet") else 1)
PY
)"
  status=$?
  set -e
  info "memory quiet preflight: $output"
  if (( status != 0 )) && [[ "$REQUIRE_MEMORY_QUIET_PREFLIGHT" == "1" ]]; then
    die "required memory quiet preflight failed before model/exporter work"
  fi
}

run_jaccl_probe() {
  info "checking explicit two-rank JACCL"
  mlx_launch_checked \
    --verbose \
    --backend jaccl \
    --hostfile "$HOSTFILE" \
    --env GLM_MLX_DISTRIBUTED_BACKEND=jaccl \
    --env MLX_METAL_FAST_SYNCH=1 \
    --python "$TMP_PYTHON" \
    --cwd /tmp \
    --no-verify-script \
    "$TMP_PROBE"
}

run_uc_pingpong() {
  info "checking UC pingpong"
  local peer_log local_log peer_status local_status peer_pid peer_device_quoted
  peer_log="$(mktemp)"
  local_log="$(mktemp)"
  printf -v peer_device_quoted '%q' "$PEER_RDMA_DEVICE"

  set +e
  ssh "${SSH_OPTS[@]}" "$PEER_SSH" \
    "perl -e 'alarm 30; exec @ARGV' /usr/bin/ibv_uc_pingpong -d ${peer_device_quoted} -n 100 -g 1" \
    >"$peer_log" 2>&1 &
  peer_pid=$!
  sleep 1
  perl -e 'alarm 30; exec @ARGV' /usr/bin/ibv_uc_pingpong \
    -d "$LOCAL_RDMA_DEVICE" -n 100 -g 1 "$PEER_DIRECT_IP" \
    >"$local_log" 2>&1
  local_status=$?
  wait "$peer_pid"
  peer_status=$?
  set -e

  if (( local_status != 0 || peer_status != 0 )); then
    {
      printf '[cleanroom] UC pingpong failed: local_status=%s peer_status=%s\n' \
        "$local_status" "$peer_status"
      printf '[cleanroom] --- local ibv_uc_pingpong ---\n'
      cat "$local_log"
      printf '[cleanroom] --- peer ibv_uc_pingpong ---\n'
      cat "$peer_log"
    } >&2
    rm -f "$local_log" "$peer_log"
    exit 23
  fi

  info "UC pingpong passed"
  rm -f "$local_log" "$peer_log"
}

prompt_args() {
  local normalized prompt
  normalized="${PROMPT_IDS//,/ }"
  for prompt in $normalized; do
    printf '%s\0%s\0' "--prompt-id" "$prompt"
  done
}

route_trace_layer_args() {
  local normalized layer
  normalized="${ROUTE_TRACE_LAYERS//,/ }"
  for layer in $normalized; do
    printf '%s\0%s\0' "--route-trace-layer" "$layer"
  done
}

run_exporter() {
  info "launching distributed teacher-cache exporter"
  mkdir -p "$(dirname "$OUTPUT_DIR")"
  local rank_view_json
  rank_view_json="$(rank_view_roots_json)"
  local args=()
  while IFS= read -r -d '' item; do
    args+=("$item")
  done < <(prompt_args)
  if [[ "$NO_FULL_LOGITS" == "1" ]]; then
    info "GLM_NO_FULL_LOGITS=1; exporter will emit top-k-only lower-bound rows"
    args+=("--no-full-logits")
  fi
  if [[ "$INCLUDE_ROUTE_TRACE" == "1" ]]; then
    info "GLM_INCLUDE_ROUTE_TRACE=1; exporter will include token-level route traces"
    args+=("--include-route-trace")
  fi
  if [[ "$ROUTE_TRACE_ONLY" == "1" ]]; then
    info "GLM_ROUTE_TRACE_ONLY=1; exporter will skip lm_head/logits and write route-trace-only metadata"
    args+=("--route-trace-only")
  fi
  if [[ -n "$LAYER_SPLIT" ]]; then
    numeric_or_die "GLM_LAYER_SPLIT" "$LAYER_SPLIT"
    info "GLM_LAYER_SPLIT=$LAYER_SPLIT; exporter will use a custom two-rank layer split"
    args+=("--layer-split" "$LAYER_SPLIT")
  fi
  while IFS= read -r -d '' item; do
    args+=("$item")
  done < <(route_trace_layer_args)
  if [[ -n "$MLX_WIRED_LIMIT_GB" ]]; then
    args+=("--mlx-wired-limit-gb" "$MLX_WIRED_LIMIT_GB")
  fi

  local launch_cmd=(
    mlx_launch_checked
    --verbose \
    --backend jaccl \
    --hostfile "$HOSTFILE" \
    --env GLM_MLX_DISTRIBUTED_BACKEND=jaccl \
    --env MLX_METAL_FAST_SYNCH=1 \
    --python "$TMP_PYTHON" \
    --cwd /tmp \
    --no-verify-script \
    "$TMP_EXPORTER" \
    --backend jaccl \
    --rank-view-roots-json "$rank_view_json" \
    --output-dir "$OUTPUT_DIR" \
    --require-two-ranks \
    --rank0-only-logits \
    --stream-lm-head \
    --lm-head-chunk-rows "$LM_HEAD_CHUNK_ROWS" \
    --mlx-cache-limit-gb "$MLX_CACHE_LIMIT_GB" \
    --top-k "$TOP_K" \
    --prompt-set "$PROMPT_SET" \
    --max-positions "$MAX_POSITIONS"
  )
  if (( ${#args[@]} > 0 )); then
    launch_cmd+=("${args[@]}")
  fi
  "${launch_cmd[@]}"
}

run_single_host_exporter() {
  if [[ "$INCLUDE_ROUTE_TRACE" == "1" || "$ROUTE_TRACE_ONLY" == "1" ]]; then
    die "single-host fallback does not support route-trace export; use the distributed path"
  fi
  info "launching single-host teacher-cache exporter"
  mkdir -p "$(dirname "$OUTPUT_DIR")"
  local args=()
  while IFS= read -r -d '' item; do
    args+=("$item")
  done < <(prompt_args)
  if [[ "$NO_FULL_LOGITS" == "1" ]]; then
    info "GLM_NO_FULL_LOGITS=1; single-host exporter will emit top-k-only lower-bound rows"
    args+=("--no-full-logits")
  fi
  if [[ "$SINGLE_HOST_PIPELINE_LOCAL" == "1" ]]; then
    if [[ -z "$LAYER_SPLIT" ]]; then
      die "GLM_SINGLE_HOST_PIPELINE_LOCAL=1 requires GLM_LAYER_SPLIT"
    fi
    info "GLM_SINGLE_HOST_PIPELINE_LOCAL=1; running rank views sequentially on one host"
    args+=("--mlx-cache-limit-gb" "$MLX_CACHE_LIMIT_GB")
    if [[ -n "$SINGLE_HOST_MLX_WIRED_LIMIT_GB" ]]; then
      args+=("--mlx-wired-limit-gb" "$SINGLE_HOST_MLX_WIRED_LIMIT_GB")
    fi
    if [[ -n "$SINGLE_HOST_REVISION" ]]; then
      args+=("--revision" "$SINGLE_HOST_REVISION")
    fi
    if [[ "$SINGLE_HOST_PIPELINE_LOCAL_STAGE_PROCESSES" == "1" ]]; then
      info "GLM_SINGLE_HOST_PIPELINE_LOCAL_STAGE_PROCESSES=1; isolating lower and upper stages"
      args+=("--local-sequential-stage-processes")
      if [[ -n "$LOCAL_SEQUENTIAL_STAGE_VIEW_ROOTS_JSON" ]]; then
        info "GLM_LOCAL_SEQUENTIAL_STAGE_VIEW_ROOTS_JSON set; using worker-specific stage view roots"
        args+=("--local-sequential-stage-view-roots-json" "$LOCAL_SEQUENTIAL_STAGE_VIEW_ROOTS_JSON")
      fi
      if [[ -n "$LOCAL_SEQUENTIAL_REMOTE_WORKERS" ]]; then
        info "GLM_LOCAL_SEQUENTIAL_REMOTE_WORKERS=$LOCAL_SEQUENTIAL_REMOTE_WORKERS; running selected stage workers over SSH"
        [[ "$LOCAL_SEQUENTIAL_REMOTE_DIRTY_RETRIES" =~ ^[0-9]+$ ]] \
          || die "GLM_LOCAL_SEQUENTIAL_REMOTE_DIRTY_RETRIES must be a non-negative integer"
        [[ "$LOCAL_SEQUENTIAL_REMOTE_RETRY_SLEEP_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] \
          || die "GLM_LOCAL_SEQUENTIAL_REMOTE_RETRY_SLEEP_SECONDS must be a non-negative number"
        local remote_stage_view_roots_json="$LOCAL_SEQUENTIAL_REMOTE_STAGE_VIEW_ROOTS_JSON"
        if [[ -z "$remote_stage_view_roots_json" ]]; then
          local peer_rank0_root
          peer_rank0_root="$(stage_peer_rank0_view_root)"
          remote_stage_view_roots_json="$(local_sequential_remote_stage_roots_json "$peer_rank0_root")"
        fi
        args+=(
          "--local-sequential-remote-worker" "$LOCAL_SEQUENTIAL_REMOTE_WORKERS"
          "--local-sequential-remote-ssh" "$LOCAL_SEQUENTIAL_REMOTE_SSH"
          "--local-sequential-remote-repo" "$LOCAL_SEQUENTIAL_REMOTE_REPO"
          "--local-sequential-remote-python" "$LOCAL_SEQUENTIAL_REMOTE_PYTHON"
          "--local-sequential-remote-tmp-dir" "$LOCAL_SEQUENTIAL_REMOTE_TMP_DIR"
          "--local-sequential-remote-dirty-retries" "$LOCAL_SEQUENTIAL_REMOTE_DIRTY_RETRIES"
          "--local-sequential-remote-retry-sleep-seconds" "$LOCAL_SEQUENTIAL_REMOTE_RETRY_SLEEP_SECONDS"
        )
        for ssh_option in "${SSH_OPTS[@]}"; do
          args+=("--local-sequential-remote-ssh-option=$ssh_option")
        done
        if [[ -n "$remote_stage_view_roots_json" ]]; then
          info "using peer-local worker rank views for remote local-sequential workers"
          args+=(
            "--local-sequential-remote-stage-view-roots-json"
            "$remote_stage_view_roots_json"
          )
        fi
      fi
      if [[ "$REQUIRE_MEMORY_QUIET_PREFLIGHT" == "1" ]]; then
        info "GLM_REQUIRE_MEMORY_QUIET_PREFLIGHT=1; checking memory before each stage worker"
        args+=(
          "--local-sequential-stage-memory-quiet-preflight"
          "--local-sequential-stage-memory-quiet-seconds" "$MEMORY_QUIET_SECONDS"
          "--local-sequential-stage-memory-quiet-max-attempts" "$MEMORY_QUIET_MAX_ATTEMPTS"
          "--local-sequential-stage-memory-quiet-min-free-gb" "$MEMORY_QUIET_MIN_FREE_GB"
          "--local-sequential-stage-memory-quiet-stage-view-margin-gb" "$MEMORY_QUIET_STAGE_VIEW_MARGIN_GB"
        )
      fi
    fi
    if [[ "$SINGLE_HOST_PIPELINE_LOCAL_HEAD_PROCESS" == "1" ]]; then
      if [[ "$SINGLE_HOST_PIPELINE_LOCAL_STAGE_PROCESSES" != "1" ]]; then
        die "GLM_SINGLE_HOST_PIPELINE_LOCAL_HEAD_PROCESS=1 requires GLM_SINGLE_HOST_PIPELINE_LOCAL_STAGE_PROCESSES=1"
      fi
      info "GLM_SINGLE_HOST_PIPELINE_LOCAL_HEAD_PROCESS=1; streaming lm_head in a third child process"
      args+=("--local-sequential-head-process")
    fi
    if [[ -n "$SINGLE_HOST_PIPELINE_LOCAL_LOWER_SPLIT_LAYER" ]]; then
      if [[ -n "$SINGLE_HOST_PIPELINE_LOCAL_LOWER_SPLIT_LAYERS" ]]; then
        die "Use either GLM_SINGLE_HOST_PIPELINE_LOCAL_LOWER_SPLIT_LAYER or GLM_SINGLE_HOST_PIPELINE_LOCAL_LOWER_SPLIT_LAYERS, not both"
      fi
      if [[ "$SINGLE_HOST_PIPELINE_LOCAL_STAGE_PROCESSES" != "1" ]]; then
        die "GLM_SINGLE_HOST_PIPELINE_LOCAL_LOWER_SPLIT_LAYER requires GLM_SINGLE_HOST_PIPELINE_LOCAL_STAGE_PROCESSES=1"
      fi
      if ! [[ "$SINGLE_HOST_PIPELINE_LOCAL_LOWER_SPLIT_LAYER" =~ ^[1-9][0-9]*$ ]]; then
        die "GLM_SINGLE_HOST_PIPELINE_LOCAL_LOWER_SPLIT_LAYER must be a positive integer"
      fi
      info "GLM_SINGLE_HOST_PIPELINE_LOCAL_LOWER_SPLIT_LAYER=$SINGLE_HOST_PIPELINE_LOCAL_LOWER_SPLIT_LAYER; splitting rank1 lower stage"
      args+=("--local-sequential-lower-split-layer" "$SINGLE_HOST_PIPELINE_LOCAL_LOWER_SPLIT_LAYER")
    fi
    if [[ -n "$SINGLE_HOST_PIPELINE_LOCAL_LOWER_SPLIT_LAYERS" ]]; then
      if [[ "$SINGLE_HOST_PIPELINE_LOCAL_STAGE_PROCESSES" != "1" ]]; then
        die "GLM_SINGLE_HOST_PIPELINE_LOCAL_LOWER_SPLIT_LAYERS requires GLM_SINGLE_HOST_PIPELINE_LOCAL_STAGE_PROCESSES=1"
      fi
      if [[ "$SINGLE_HOST_PIPELINE_LOCAL_HEAD_PROCESS" != "1" ]]; then
        die "GLM_SINGLE_HOST_PIPELINE_LOCAL_LOWER_SPLIT_LAYERS requires GLM_SINGLE_HOST_PIPELINE_LOCAL_HEAD_PROCESS=1"
      fi
      if ! [[ "$SINGLE_HOST_PIPELINE_LOCAL_LOWER_SPLIT_LAYERS" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
        die "GLM_SINGLE_HOST_PIPELINE_LOCAL_LOWER_SPLIT_LAYERS must be comma-separated positive integers"
      fi
      info "GLM_SINGLE_HOST_PIPELINE_LOCAL_LOWER_SPLIT_LAYERS=$SINGLE_HOST_PIPELINE_LOCAL_LOWER_SPLIT_LAYERS; splitting rank1 lower stage into multiple windows"
      args+=("--local-sequential-lower-split-layers" "$SINGLE_HOST_PIPELINE_LOCAL_LOWER_SPLIT_LAYERS")
    fi
    if [[ -n "$SINGLE_HOST_PIPELINE_LOCAL_UPPER_SPLIT_LAYER" ]]; then
      if [[ -n "$SINGLE_HOST_PIPELINE_LOCAL_UPPER_SPLIT_LAYERS" ]]; then
        die "Use either GLM_SINGLE_HOST_PIPELINE_LOCAL_UPPER_SPLIT_LAYER or GLM_SINGLE_HOST_PIPELINE_LOCAL_UPPER_SPLIT_LAYERS, not both"
      fi
      if [[ "$SINGLE_HOST_PIPELINE_LOCAL_STAGE_PROCESSES" != "1" ]]; then
        die "GLM_SINGLE_HOST_PIPELINE_LOCAL_UPPER_SPLIT_LAYER requires GLM_SINGLE_HOST_PIPELINE_LOCAL_STAGE_PROCESSES=1"
      fi
      if ! [[ "$SINGLE_HOST_PIPELINE_LOCAL_UPPER_SPLIT_LAYER" =~ ^[1-9][0-9]*$ ]]; then
        die "GLM_SINGLE_HOST_PIPELINE_LOCAL_UPPER_SPLIT_LAYER must be a positive integer"
      fi
      info "GLM_SINGLE_HOST_PIPELINE_LOCAL_UPPER_SPLIT_LAYER=$SINGLE_HOST_PIPELINE_LOCAL_UPPER_SPLIT_LAYER; splitting rank0 upper stage"
      args+=("--local-sequential-upper-split-layer" "$SINGLE_HOST_PIPELINE_LOCAL_UPPER_SPLIT_LAYER")
    fi
    if [[ -n "$SINGLE_HOST_PIPELINE_LOCAL_UPPER_SPLIT_LAYERS" ]]; then
      if [[ "$SINGLE_HOST_PIPELINE_LOCAL_STAGE_PROCESSES" != "1" ]]; then
        die "GLM_SINGLE_HOST_PIPELINE_LOCAL_UPPER_SPLIT_LAYERS requires GLM_SINGLE_HOST_PIPELINE_LOCAL_STAGE_PROCESSES=1"
      fi
      if [[ "$SINGLE_HOST_PIPELINE_LOCAL_HEAD_PROCESS" != "1" ]]; then
        die "GLM_SINGLE_HOST_PIPELINE_LOCAL_UPPER_SPLIT_LAYERS requires GLM_SINGLE_HOST_PIPELINE_LOCAL_HEAD_PROCESS=1"
      fi
      if ! [[ "$SINGLE_HOST_PIPELINE_LOCAL_UPPER_SPLIT_LAYERS" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
        die "GLM_SINGLE_HOST_PIPELINE_LOCAL_UPPER_SPLIT_LAYERS must be comma-separated positive integers"
      fi
      info "GLM_SINGLE_HOST_PIPELINE_LOCAL_UPPER_SPLIT_LAYERS=$SINGLE_HOST_PIPELINE_LOCAL_UPPER_SPLIT_LAYERS; splitting rank0 upper stage into multiple windows"
      args+=("--local-sequential-upper-split-layers" "$SINGLE_HOST_PIPELINE_LOCAL_UPPER_SPLIT_LAYERS")
    fi
    if [[ "$SINGLE_HOST_PIPELINE_LOCAL_ABORT_ON_DIRTY_STAGE" == "1" ]]; then
      if [[ "$SINGLE_HOST_PIPELINE_LOCAL_STAGE_PROCESSES" != "1" ]]; then
        die "GLM_SINGLE_HOST_PIPELINE_LOCAL_ABORT_ON_DIRTY_STAGE=1 requires GLM_SINGLE_HOST_PIPELINE_LOCAL_STAGE_PROCESSES=1"
      fi
      info "GLM_SINGLE_HOST_PIPELINE_LOCAL_ABORT_ON_DIRTY_STAGE=1; aborting after the first dirty stage worker"
      args+=("--local-sequential-abort-on-dirty-stage")
    fi
    uv run python benchmarks/export_glm45_air_distributed_teacher_cache.py \
      --local-sequential \
      --rank-view-roots-json "$RANK_VIEW_ROOTS_JSON" \
      --output-dir "$OUTPUT_DIR" \
      --layer-split "$LAYER_SPLIT" \
      --rank0-only-logits \
      --stream-lm-head \
      --lm-head-chunk-rows "$LM_HEAD_CHUNK_ROWS" \
      --model-id "$SINGLE_HOST_MODEL_PATH" \
      --teacher-kind "$SINGLE_HOST_TEACHER_KIND" \
      --top-k "$TOP_K" \
      --prompt-set "$PROMPT_SET" \
      --max-positions "$MAX_POSITIONS" \
      "${args[@]}"
    return
  fi
  if [[ "$SINGLE_HOST_LAZY" == "1" ]]; then
    args+=("--lazy")
  fi
  if [[ -n "$SINGLE_HOST_REVISION" ]]; then
    args+=("--revision" "$SINGLE_HOST_REVISION")
  fi
  args+=("--mlx-cache-limit-gb" "$MLX_CACHE_LIMIT_GB")
  if [[ -n "$SINGLE_HOST_MLX_WIRED_LIMIT_GB" ]]; then
    args+=("--mlx-wired-limit-gb" "$SINGLE_HOST_MLX_WIRED_LIMIT_GB")
  fi
  if [[ -n "$SINGLE_HOST_SOURCE_MEMORY_GUARD_RATIO" ]]; then
    args+=("--source-memory-guard-ratio" "$SINGLE_HOST_SOURCE_MEMORY_GUARD_RATIO")
  fi
  uv run python benchmarks/export_glm45_air_teacher_cache.py \
    --model-path "$SINGLE_HOST_MODEL_PATH" \
    --teacher-kind "$SINGLE_HOST_TEACHER_KIND" \
    --output-dir "$OUTPUT_DIR" \
    --top-k "$TOP_K" \
    --prompt-set "$PROMPT_SET" \
    --max-positions "$MAX_POSITIONS" \
    "${args[@]}"
}

validate_teacher_cache() {
  info "validating teacher cache"
  uv run python benchmarks/validate_glm45_air_teacher_cache.py \
    --teacher-jsonl "$OUTPUT_DIR/metadata.jsonl" \
    --cache-root "$OUTPUT_DIR" \
    --min-top-k "$TOP_K" \
    --check-values \
    --append-jsonl "$VALIDATION_JSONL"

  uv run python - "$VALIDATION_JSONL" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not rows:
    raise SystemExit(f"no validation rows in {path}")
latest = rows[-1]
print(json.dumps({
    "validation_jsonl": str(path),
    "ok": latest.get("ok"),
    "record_count": latest.get("record_count"),
    "all_memory_clean": latest.get("all_memory_clean"),
    "error_count": latest.get("error_count"),
}, indent=2, sort_keys=True))
allow_dirty = os.environ.get("GLM_ALLOW_DIRTY_CACHE") == "1"
if not latest.get("all_memory_clean") and not allow_dirty:
    raise SystemExit("teacher-cache validation passed, but producer memory counters are dirty")
if not latest.get("all_memory_clean") and allow_dirty:
    print("GLM_ALLOW_DIRTY_CACHE=1; teacher-cache validation passed with dirty producer memory counters")
PY
}

consume_teacher_cache() {
  info "consuming teacher cache locally"
  local consume_args=()
  if [[ "$INCLUDE_ROUTE_TRACE" == "1" ]]; then
    info "GLM_INCLUDE_ROUTE_TRACE=1; local eval will include student token-level route traces"
    consume_args+=("--include-route-trace")
  fi
  uv run python benchmarks/eval_glm45_air_teacher_cache.py \
    --teacher-jsonl "$OUTPUT_DIR/metadata.jsonl" \
    --cache-root "$OUTPUT_DIR" \
    --artifact-dir "$VQ_ARTIFACT_DIR" \
    --engine "$ENGINE" \
    --min-top-k "$TOP_K" \
    --check-values \
    --append-jsonl "$LOCAL_JSONL" \
    "${consume_args[@]}"
}

validate_and_consume() {
  validate_teacher_cache
  if [[ "$SKIP_LOCAL_CONSUME" == "1" ]]; then
    info "GLM_SKIP_LOCAL_CONSUME=1; skipping local VQ consumption after validation"
    return
  fi
  consume_teacher_cache
}

main() {
  require_file "$ROOT/benchmarks/export_glm45_air_distributed_teacher_cache.py"
  require_file "$ROOT/benchmarks/export_glm45_air_teacher_cache.py"
  require_file "$ROOT/benchmarks/validate_glm45_air_teacher_cache.py"
  require_file "$ROOT/benchmarks/eval_glm45_air_teacher_cache.py"
  require_dir "$ROOT/.venv"
  if [[ "$ROUTE_TRACE_ONLY" != "1" && "$SKIP_LOCAL_CONSUME" != "1" ]]; then
    require_dir "$VQ_ARTIFACT_DIR"
  fi
  if [[ "$SINGLE_HOST_FALLBACK" == "1" ]]; then
    info "GLM_SINGLE_HOST_FALLBACK=1; skipping peer, direct-link, and JACCL preflight"
    if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
      info "GLM_PREFLIGHT_ONLY=1; single-host fallback prerequisites are present"
      return
    fi
    run_memory_quiet_preflight
    run_single_host_exporter
    validate_and_consume
    info "single-host fallback teacher cache is ready at $OUTPUT_DIR"
    return
  fi
  require_file "$HOSTFILE"
  require_direct_link
  require_wired_caps
  write_wrappers_and_probes
  check_mlx_wired_limit
  run_uc_pingpong
  run_jaccl_probe
  if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    info "GLM_PREFLIGHT_ONLY=1; skipping exporter, validation, and local consumption"
    return
  fi
  stage_peer_rank1_view_root
  require_rank_view_roots
  run_memory_quiet_preflight
  run_exporter
  if [[ "$ROUTE_TRACE_ONLY" == "1" ]]; then
    info "route-trace-only export requested; skipping teacher-cache validation and local consumption"
    return
  fi
  validate_and_consume
  info "clean distributed teacher cache is ready at $OUTPUT_DIR"
}

main "$@"
