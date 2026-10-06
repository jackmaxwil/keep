#!/usr/bin/env bash
# SUPERSEDED — MUST NOT BE USED FOR THE CURRENT GLM-5.2 CAMPAIGN.
# Binding decision:
# docs/superpowers/decisions/2026-07-30-glm52-member-account-lifecycle-controls.md
# Approved replacement: SkyPilot-owned provisioning through the guarded
# aws/glm52-gpu/scripts/submit_sky_campaign.sh --production entrypoint.
# This legacy script violates the decision by calling raw ec2:RunInstances and
# allowing Spot, alternate worker shapes, alternate regions, and multi-AZ retry
# instead of at most one on-demand p5.48xlarge in us-west-2.
#
# Launch ONE GPU instance from the stack's launch template. THIS SPENDS MONEY.
# Usage:
#   INSTANCE_TYPE=g6e.xlarge USE_SPOT=1 ./launch_instance.sh    # cheap spike/seed
#   INSTANCE_TYPE=p5.48xlarge ./launch_instance.sh              # full run (on-demand)
set -euo pipefail

STACK="${STACK:-keep-glm52-gpu}"
REGION="${REGION:-us-west-2}"
INSTANCE_TYPE="${INSTANCE_TYPE:-p5.48xlarge}"
USE_SPOT="${USE_SPOT:-0}"

OUT=$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
  --query "Stacks[0].Outputs")
LT_ID=$(echo "$OUT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(next(o['OutputValue'] for o in d if o['OutputKey']=='LaunchTemplateId'))")
SG_ID=$(echo "$OUT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(next(o['OutputValue'] for o in d if o['OutputKey']=='SecurityGroupId'))")
[ -n "$LT_ID" ] && [ "$LT_ID" != "None" ] || { echo "stack not deployed?"; exit 1; }
# Try every configured AZ in turn — p5/p5en capacity is genuinely scarce.
SUBNETS=$(echo "$OUT" | python3 -c "
import json, sys
d = json.load(sys.stdin)
keys = ('SubnetId', 'SubnetIdAltAz', 'SubnetIdAz3', 'SubnetIdAz4', 'SubnetIdAz5', 'SubnetIdAz6')
for k in keys:
    v = next((o['OutputValue'] for o in d if o['OutputKey'] == k), None)
    if v:
        print(v)
")

MARKET=()
if [ "$USE_SPOT" = "1" ]; then
  # one-time spot request; instance persists until you stop/terminate it
  MARKET=(--instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}')
fi

try_launch() {
  local subnet="$1"
  aws ec2 run-instances --region "$REGION" \
    --launch-template "LaunchTemplateId=$LT_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --network-interfaces "DeviceIndex=0,SubnetId=$subnet,Groups=$SG_ID,AssociatePublicIpAddress=true" \
    ${MARKET[@]+"${MARKET[@]}"} \
    --query "Instances[0].InstanceId" --output text 2>/tmp/keep-launch-err.txt
}

ID=""
while IFS= read -r SUBNET; do
  [ -n "$SUBNET" ] || continue
  echo ">> Launching $INSTANCE_TYPE (spot=$USE_SPOT) from LT $LT_ID in $SUBNET"
  ID=$(try_launch "$SUBNET") || true
  if [ -n "$ID" ]; then break; fi
  if grep -q InsufficientInstanceCapacity /tmp/keep-launch-err.txt 2>/dev/null; then
    echo ">> No capacity in $SUBNET, trying next AZ"
  else
    echo ">> Launch error in $SUBNET (not a capacity issue):"; cat /tmp/keep-launch-err.txt; break
  fi
done <<< "$SUBNETS"

if [ -z "${ID:-}" ]; then
  echo ">> Launch failed in all AZs:"; cat /tmp/keep-launch-err.txt 2>/dev/null; exit 1
fi

echo ">> InstanceId: $ID  (waiting for running + SSM registration...)"
aws ec2 wait instance-running --region "$REGION" --instance-ids "$ID"
echo ">> Running. Connect with:   REGION=$REGION ID=$ID ./scripts/connect.sh"
echo ">> REMEMBER to terminate when done:   aws ec2 terminate-instances --region $REGION --instance-ids $ID"
