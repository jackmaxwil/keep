#!/usr/bin/env bash
# Stage code and non-model campaign authorities to both repository buckets.
# This script intentionally does NOT upload the 57 GiB accepted routed model.
set -euo pipefail

REPO="${REPO:-/Users/jack.mazac/Developer/keep}"
RUN_ID="${RUN_ID:?set RUN_ID}"
WEST_BUCKET="${WEST_BUCKET:-keep-glm52-models-246813579024-us-west-2}"
REPOSITORY_BUCKET="${REPOSITORY_BUCKET:-keep-glm52-models-246813579024-us-east-1}"
SOURCE_SNAPSHOT="${SOURCE_SNAPSHOT:-/Users/jack.mazac/.cache/huggingface/hub/models--0xSero--glm-5.2-reap-504B-v2/snapshots/6c9241aa05fb243a0edb7c804c213ec1cf5c920d}"
NONVQ="$REPO/artifacts/build/glm52_reap_504b_materialization_probe_20260709/steps/non_vq_pack-a678bc3c/out"
TEICH="${TEICH:-/private/tmp/claude-502/-Users-jack-mazac-Developer-keep/b4d50550-2788-47d9-b001-3dbad2f173e6/scratchpad/teich-corpus-v2/initial-cut/glm52-coding-agent-initial-v2-20260713.json}"
FROZEN="$REPO/artifacts/quality/glm52-family-eval-prompts-20260709-v2.json"
ACCEPTED="$REPO/artifacts/quality/glm52-training-baseline-accepted-20260711.json"
WORK="${WORK:-$(mktemp -d)}"
trap 'rm -rf "$WORK"' EXIT

for path in \
  "$SOURCE_SNAPSHOT/model.safetensors.index.json" \
  "$NONVQ/non-vq-manifest.json" "$TEICH" "$FROZEN" "$ACCEPTED"; do
  test -f "$path" || { echo "missing authority: $path" >&2; exit 1; }
done

TAR="$WORK/keep-campaign.tar.gz"
COPYFILE_DISABLE=1 tar -czf "$TAR" -C "$REPO" \
  --exclude '.git' --exclude 'runs' --exclude '.venv' --exclude 'node_modules' \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '*.safetensors' \
  src benchmarks native models tests pyproject.toml uv.lock aws \
  artifacts/quality/glm52-tokenizer-readiness-20260709.json \
  artifacts/quality/glm52-family-policy-20260709-v2.json \
  artifacts/quality/glm52-wave6-full-artifact-audit-20260709.json \
  artifacts/quality/glm52-wave6-full-bind-preflight-20260709.json \
  artifacts/quality/glm52-wave6-full-materialization-20260709/materialization.jsonl
REPO_SHA=$(shasum -a 256 "$TAR" | awk '{print $1}')
REPO_KEY="campaigns/$RUN_ID/repo/keep-$REPO_SHA.tar.gz"

BASELINE_DIR="$WORK/training-baseline"
mkdir -p "$BASELINE_DIR"
python3 "$REPO/aws/glm52-gpu/scripts/build_gpu_training_baseline_config.py" \
  --accepted-config "$ACCEPTED" --output "$BASELINE_DIR/training-baseline.json"
cp "$REPO/artifacts/build/glm52_reap_504b_materialization_probe_20260709/steps/non_vq_pack-a678bc3c/evidence/evidence_json.json" \
  "$BASELINE_DIR/non-vq-evidence.json"
BASELINE_SHA=$(shasum -a 256 "$BASELINE_DIR/training-baseline.json" | awk '{print $1}')

LAMBDA_ZIP="$WORK/capacity-block-controller.zip"
"$REPO/aws/glm52-gpu/scripts/package_controller_lambda.sh" "$LAMBDA_ZIP" >/dev/null
LAMBDA_SHA=$(shasum -a 256 "$LAMBDA_ZIP" | awk '{print $1}')
LAMBDA_KEY="campaigns/$RUN_ID/controller/capacity-block-controller-$LAMBDA_SHA.zip"

for destination in \
  "us-west-2:$WEST_BUCKET" \
  "us-east-1:$REPOSITORY_BUCKET"; do
  REGION=${destination%%:*}
  BUCKET=${destination#*:}
  aws s3 cp "$TAR" "s3://$BUCKET/$REPO_KEY" --region "$REGION" --only-show-errors
  aws s3 cp "$LAMBDA_ZIP" "s3://$BUCKET/$LAMBDA_KEY" --region "$REGION" --only-show-errors
  aws s3 cp "$TEICH" "s3://$BUCKET/teich-pack/glm52-coding-agent-initial-v2-20260713.json" \
    --region "$REGION" --only-show-errors
  aws s3 cp "$FROZEN" "s3://$BUCKET/quality/glm52-family-eval-prompts-20260709-v2.json" \
    --region "$REGION" --only-show-errors
done
aws s3 sync "$BASELINE_DIR/" "s3://$WEST_BUCKET/training-baseline/" \
  --region us-west-2 --only-show-errors

SOURCE_SHA=$(shasum -a 256 "$SOURCE_SNAPSHOT/model.safetensors.index.json" | awk '{print $1}')
NONVQ_SHA=$(shasum -a 256 "$NONVQ/non-vq-manifest.json" | awk '{print $1}')
TEICH_SHA=$(shasum -a 256 "$TEICH" | awk '{print $1}')
FROZEN_SHA=$(shasum -a 256 "$FROZEN" | awk '{print $1}')
RESULT="${OUTPUT:-$REPO_SHA.staging.json}"
python3 - "$RESULT" <<PY
import json, sys
value = {
  "run_id": "$RUN_ID",
  "repo_tar_key": "$REPO_KEY",
  "repo_tar_sha256": "$REPO_SHA",
  "controller_code_s3_key": "$LAMBDA_KEY",
  "controller_zip_sha256": "$LAMBDA_SHA",
  "source_snapshot_prefix": "source-snapshot/",
  "source_snapshot_sha256": "$SOURCE_SHA",
  "non_vq_prefix": "non-vq-package/",
  "non_vq_package_sha256": "$NONVQ_SHA",
  "teich_pack_key": "teich-pack/glm52-coding-agent-initial-v2-20260713.json",
  "teich_pack_sha256": "$TEICH_SHA",
  "frozen_prompt_pack_key": "quality/glm52-family-eval-prompts-20260709-v2.json",
  "frozen_prompt_pack_sha256": "$FROZEN_SHA",
  "training_baseline_prefix": "training-baseline/",
  "training_baseline_sha256": "$BASELINE_SHA",
  "accepted_routed_model_uploaded": False,
}
open(sys.argv[1], "w").write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
print(sys.argv[1])
PY
