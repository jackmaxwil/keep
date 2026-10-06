#!/usr/bin/env bash
# Deploy (or update) the GLM-5.2 GPU teacher-gen stack. COST: $0 — this creates
# only VPC/S3-endpoint/bucket/IAM/SG/launch-template. No GPU is launched here.
set -euo pipefail

STACK="${STACK:-keep-glm52-gpu}"
REGION="${REGION:-us-west-2}"
AZ="${AZ:-us-west-2c}"
AZ_ALT="${AZ_ALT:-us-west-2a}"
AZ3="${AZ3:-us-west-2b}"
AZ4="${AZ4:-us-west-2d}"
AZ5="${AZ5:-}"
AZ6="${AZ6:-}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

echo ">> Deploying stack '$STACK' in $REGION (AZs $AZ/$AZ_ALT/$AZ3/$AZ4/$AZ5/$AZ6) — no compute cost"
aws cloudformation deploy \
  --region "$REGION" \
  --stack-name "$STACK" \
  --template-file "$HERE/cfn/gpu-teacher-stack.yaml" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides "AvailabilityZone=$AZ" "AvailabilityZoneAlt=$AZ_ALT" "AvailabilityZone3=$AZ3" "AvailabilityZone4=$AZ4" "AvailabilityZone5=$AZ5" "AvailabilityZone6=$AZ6" \
  --tags project=keep-glm52 owner=jack.mazac

echo ">> Outputs:"
aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
  --query "Stacks[0].Outputs[].{Key:OutputKey,Value:OutputValue}" --output table
