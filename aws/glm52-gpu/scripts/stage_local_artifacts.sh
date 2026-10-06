#!/usr/bin/env bash
# RUN ON JACK'S MAC. Uploads the KEEP-local artifacts (small) that are NOT on
# HuggingFace: the non-VQ package, the profile, and the teich prompt pack.
# The big model weights come from HF via seed_from_hf.sh (do not upload those here).
set -euo pipefail

REGION="${REGION:-us-west-2}"
STACK="${STACK:-keep-glm52-gpu}"
REPO="${REPO:-/Users/jack.mazac/Developer/keep}"

BUCKET="${BUCKET:-$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='ModelBucketName'].OutputValue" --output text)}"
[ -n "$BUCKET" ] && [ "$BUCKET" != "None" ] || { echo "resolve BUCKET failed"; exit 1; }

NONVQ="$REPO/artifacts/build/glm52_reap_504b_materialization_probe_20260709/steps/non_vq_pack-a678bc3c/out"
PROFILE="$REPO/models/glm52-reap-504b-v2.yaml"
TEICH="/private/tmp/claude-502/-Users-jack-mazac-Developer-keep/b4d50550-2788-47d9-b001-3dbad2f173e6/scratchpad/teich-corpus-v2/initial-cut/glm52-coding-agent-initial-v2-20260713.json"

echo ">> Bucket: s3://$BUCKET"
echo ">> non-VQ package ($(du -sh "$NONVQ" | cut -f1)) -> s3://$BUCKET/non-vq-package/"
aws s3 sync "$NONVQ" "s3://$BUCKET/non-vq-package/" --region "$REGION"
echo ">> profile -> s3://$BUCKET/profile/"
aws s3 cp "$PROFILE" "s3://$BUCKET/profile/" --region "$REGION"
echo ">> teich pack -> s3://$BUCKET/teich-pack/"
aws s3 cp "$TEICH" "s3://$BUCKET/teich-pack/" --region "$REGION"

VBASE="$REPO/runs/teich-teacher-localref-20260713"
echo ">> validation packs + v5-derived reference -> s3://$BUCKET/validation/"
aws s3 cp "$VBASE/validation-pack.json" "s3://$BUCKET/validation/" --region "$REGION"
aws s3 cp "$VBASE/v5-parity-pack.json" "s3://$BUCKET/validation/" --region "$REGION"
aws s3 sync "$VBASE/v5-ref" "s3://$BUCKET/validation/v5-ref/" --region "$REGION"

echo ">> repo tarball (code the GPU node runs) -> s3://$BUCKET/repo/keep.tar.gz"
TARBALL=$(mktemp -d)/keep.tar.gz
tar -czf "$TARBALL" -C "$REPO" \
  --exclude ".git" --exclude "runs" --exclude "node_modules" \
  --exclude "__pycache__" --exclude ".venv" --exclude "*.safetensors" \
  src benchmarks native models tests pyproject.toml uv.lock aws artifacts/quality/instruction_hf_dolly48_prompts.jsonl
aws s3 cp "$TARBALL" "s3://$BUCKET/repo/keep.tar.gz" --region "$REGION"
echo ">> Done. Contents:"
aws s3 ls "s3://$BUCKET/" --recursive --region "$REGION" --human-readable --summarize | tail -20
