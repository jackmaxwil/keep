#!/usr/bin/env bash
# RUN ON THE GPU NODE. Environment bootstrap — battle-tested on g6e.2xlarge
# (2026-07-13). Uses SYSTEM python (the DLAMI's venv/ensurepip is broken) and
# aligns the CUDA symlink to MLX's bundled NVRTC so runtime JIT compiles.
set -uxo pipefail

BUCKET="${BUCKET:?set BUCKET=keep-glm52-models-<acct>-<region>}"
ROOT="${ROOT:-/mnt/nvme}"
CAMPAIGN_DESCRIPTOR="${CAMPAIGN_DESCRIPTOR:-/etc/keep-glm52/campaign.json}"
REPO_DIR="${KEEP_REPO_DIR:-$ROOT/keep}"
MANAGED_MODE="${GLM52_MANAGED_MODE:-production}"
fail() { echo "GPU-ENV-FAILED: $1"; exit 1; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }
descriptor() { jq -er "$1" "$CAMPAIGN_DESCRIPTOR"; }

test -f "$CAMPAIGN_DESCRIPTOR" || fail campaign-descriptor
unset GLM_MLX_WIRED_LIMIT_GB GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB

# s5cmd (the launch-template UserData installs it; be defensive). The clean
# control-plane rehearsal deliberately stops before host/GPU package mutation.
if [ "${GLM52_BOOTSTRAP_REHEARSAL:-0}" != 1 ]; then
  command -v s5cmd >/dev/null || \
    curl -sSL https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz \
    | tar -xz -C /usr/local/bin s5cmd || fail s5cmd
fi

echo ">> repo"
if [ "${KEEP_REPO_PRESTAGED:-0}" != 1 ]; then
  REPO_KEY=$(descriptor '.repo_tar_key')
  REPO_SHA=$(descriptor '.repo_tar_sha256')
  s5cmd cp "s3://$BUCKET/$REPO_KEY" "$ROOT/keep.tar.gz" || fail repo-pull
  [ "$(sha256_file "$ROOT/keep.tar.gz")" = "$REPO_SHA" ] || fail repo-sha256
  mkdir -p "$REPO_DIR" && tar -xzf "$ROOT/keep.tar.gz" -C "$REPO_DIR" || fail repo-extract
fi
test -f "$REPO_DIR/pyproject.toml" || fail repo-check

if [ "${GLM52_BOOTSTRAP_REHEARSAL:-0}" = 1 ]; then
  for required in \
    "$REPO_DIR/benchmarks/run_glm52_campaign.py" \
    "$REPO_DIR/aws/glm52-gpu/scripts/run_campaign.sh" \
    "$REPO_DIR/aws/glm52-gpu/scripts/watch_campaign.py" \
    "$REPO_DIR/aws/glm52-gpu/scripts/campaign_break_glass.py" \
    "$REPO_DIR/aws/glm52-gpu/scripts/sky_campaign_break_glass.sh" \
    "$REPO_DIR/aws/glm52-gpu/scripts/manage_gpu_spend.py" \
    "$REPO_DIR/aws/glm52-gpu/scripts/verify_sky_terminal_state.py" \
    "$REPO_DIR/aws/glm52-gpu/skypilot/glm52-campaign.yaml" \
    "$REPO_DIR/aws/glm52-gpu/skypilot/bootstrap_campaign.sh" \
    "$REPO_DIR/aws/glm52-gpu/skypilot/prepare_nvme_storage.sh" \
    "$REPO_DIR/aws/glm52-gpu/skypilot/run_managed_campaign.sh" \
    "$REPO_DIR/aws/glm52-gpu/skypilot/keep-glm52-campaign.service"; do
    test -f "$required" || fail "rehearsal-missing-$required"
  done
  bash -n "$REPO_DIR/aws/glm52-gpu/scripts/run_campaign.sh" || fail rehearsal-shell
  echo "GPU-ENV-REHEARSAL-OK next=MLX-CUDA/H100 root=$ROOT repo=$REPO_DIR"
  exit 0
fi

echo ">> MLX CUDA (system python; DLAMI venv/ensurepip is broken)"
# NOTE: pick the wheel matching the pinned lock. mlx-cuda-12 releases are
# sparse (0.30.x, 0.31.1, 0.32.0); 0.32.0 verified working on L40S.
python3 -m pip install -q \
  "mlx[cuda12]==0.32.0" mlx-lm numpy safetensors huggingface_hub pyyaml \
  || fail pip-mlx
python3 -m pip install -q --no-deps -e "$REPO_DIR" || fail pip-keep

echo ">> align /usr/local/cuda to MLX's bundled NVRTC (JIT header parity)"
# mlx-cuda-12 bundles nvidia-cuda-nvrtc-cu12 (e.g. 12.9.x). If the system
# symlink points at a NEWER toolkit (DLAMI ships 13.x), MLX's runtime JIT
# fails parsing cuda_fp8/fp6/fp4.hpp. Point the symlink at the matching 12.x.
NVRTC_MM=$(python3 -c "import importlib.metadata as m; v=m.version(\"nvidia-cuda-nvrtc-cu12\"); print(\".\".join(v.split(\".\")[:2]))" 2>/dev/null || true)
if [ -n "$NVRTC_MM" ] && [ -d "/usr/local/cuda-$NVRTC_MM" ]; then
  ln -sfn "/usr/local/cuda-$NVRTC_MM" /usr/local/cuda
fi
echo "cuda symlink -> $(readlink /usr/local/cuda)"

python3 -c "import mlx.core as mx; print(\"device:\", mx.default_device()); a=mx.random.normal((512,512)); mx.eval(a@a.T); print(\"gpu-matmul-ok\")" || fail mlx-gpu
python3 -c "import mlx_lm; from mlx_lm.models.base import create_causal_mask; print(\"imports-ok\")" || fail imports

echo ">> data pulls"
SOURCE_PREFIX=$(descriptor '.artifacts.source_snapshot_prefix')
NONVQ_PREFIX=$(descriptor '.artifacts.non_vq_prefix')
TEICH_KEY=$(descriptor '.artifacts.teich_pack_key')
FROZEN_KEY=$(descriptor '.artifacts.frozen_prompt_pack_key')
BASELINE_PREFIX=$(descriptor '.artifacts.training_baseline_prefix')
mkdir -p "$ROOT/non-vq-package" "$ROOT/source-snapshot" "$ROOT/training-baseline"
aws s3 sync "s3://$BUCKET/${NONVQ_PREFIX%/}/" "$ROOT/non-vq-package/" --only-show-errors || fail nonvq
test -f "$ROOT/non-vq-package/non-vq-manifest.json" || fail nonvq-check
s5cmd cp "s3://$BUCKET/$TEICH_KEY" "$ROOT/teich-pack.json" || fail teich-pack
s5cmd cp "s3://$BUCKET/$FROZEN_KEY" "$ROOT/frozen-66.json" || fail frozen-pack
aws s3 sync "s3://$BUCKET/${BASELINE_PREFIX%/}/" "$ROOT/training-baseline/" --only-show-errors || fail baseline
NONVQ_PACKAGE_SHA=$(
  jq -er .package_set_sha256 "$ROOT/non-vq-package/non-vq-manifest.json"
) || fail nonvq-package-authority
[ "$NONVQ_PACKAGE_SHA" = "$(descriptor '.artifacts.non_vq_package_sha256')" ] || fail nonvq-sha256
[ "$(sha256_file "$ROOT/teich-pack.json")" = "$(descriptor '.artifacts.teich_pack_sha256')" ] || fail teich-sha256
[ "$(sha256_file "$ROOT/frozen-66.json")" = "$(descriptor '.artifacts.frozen_prompt_pack_sha256')" ] || fail frozen-sha256
test -f "$ROOT/training-baseline/training-baseline.json" || fail baseline-config
[ "$(sha256_file "$ROOT/training-baseline/training-baseline.json")" = "$(descriptor '.artifacts.training_baseline_sha256')" ] || fail baseline-sha256
if [ "$(descriptor '.record_type // ""')" = glm52_sky_campaign_descriptor_v2 ]; then
  TRAINING_CONFIG_KEY=$(descriptor '.artifacts.training_config_key')
  s5cmd cp "s3://$BUCKET/$TRAINING_CONFIG_KEY" "$ROOT/training-config.json" || fail training-config
  [ "$(sha256_file "$ROOT/training-config.json")" = "$(descriptor '.artifacts.training_config_sha256')" ] || fail training-config-sha256
  QUALIFICATION_CACHE_SHA=$(descriptor '.artifacts.qualification_cache_manifest_sha256')
  EMPTY_QUALIFICATION_CACHE_SHA=0000000000000000000000000000000000000000000000000000000000000000
  if [ "$MANAGED_MODE" = cache-seed ]; then
    [ "$QUALIFICATION_CACHE_SHA" = "$EMPTY_QUALIFICATION_CACHE_SHA" ] || fail cache-seed-authority
  else
    [ "$QUALIFICATION_CACHE_SHA" != "$EMPTY_QUALIFICATION_CACHE_SHA" ] || fail qualification-cache-authority
    QUALIFICATION_CACHE_PREFIX=$(descriptor '.artifacts.qualification_cache_prefix')
    mkdir -p "$ROOT/qualification-cache"
    aws s3 sync "s3://$BUCKET/${QUALIFICATION_CACHE_PREFIX%/}/" \
      "$ROOT/qualification-cache/" --only-show-errors || fail qualification-cache
    test -f "$ROOT/qualification-cache/glm52-teacher-signal-cache-v3-manifest.json" || fail qualification-cache-manifest
    [ "$(sha256_file "$ROOT/qualification-cache/glm52-teacher-signal-cache-v3-manifest.json")" = "$QUALIFICATION_CACHE_SHA" ] || fail qualification-cache-sha256
  fi
fi

echo ">> authenticated source snapshot"
aws s3 sync "s3://$BUCKET/${SOURCE_PREFIX%/}/" "$ROOT/source-snapshot/" --only-show-errors || fail source-snapshot
test -f "$ROOT/source-snapshot/model.safetensors.index.json" || fail source-index
[ "$(sha256_file "$ROOT/source-snapshot/model.safetensors.index.json")" = "$(descriptor '.artifacts.source_snapshot_sha256')" ] || fail source-sha256
echo "GPU-ENV-OK snapshot=$ROOT/source-snapshot repo=$REPO_DIR"
