#!/usr/bin/env bash
# RUN ON A GPU/SEEDER INSTANCE (in-AWS). Pulls the REAP-504B source straight from
# HuggingFace onto local NVMe, then pushes it to S3 so every later node pulls fast.
# One-time. A cheap g6e.xlarge is fine for this.
set -euo pipefail

REGION="${REGION:-us-west-2}"
BUCKET="${BUCKET:?set BUCKET=keep-glm52-models-<acct>-<region>}"
HF_REPO="${HF_REPO:-0xSero/glm-5.2-reap-504B-v2}"
STAGE="${STAGE:-/mnt/nvme/source-snapshot}"

python3 -m pip install -q -U huggingface_hub
# hf_transfer is deprecated; Xet high-performance mode is the current fast path
# (measured on g6e.2xlarge: 1 conn / stalls without it, 39 conns / 172MB/s with it).
export HF_XET_HIGH_PERFORMANCE=1
export HF_HOME="$STAGE-hf-home"        # keep chunk cache off the root EBS
mkdir -p "$HF_HOME"

mkdir -p "$STAGE"
echo ">> Downloading $HF_REPO from HuggingFace -> $STAGE"
huggingface-cli download "$HF_REPO" --local-dir "$STAGE" --exclude "*.gguf" "original/*"

echo ">> Uploading snapshot -> s3://$BUCKET/source-snapshot/ (S3 gateway endpoint = free/fast)"
s5cmd --numworkers 64 cp "$STAGE/" "s3://$BUCKET/source-snapshot/"
echo ">> Seed complete. Verify:  s5cmd ls s3://$BUCKET/source-snapshot/"
