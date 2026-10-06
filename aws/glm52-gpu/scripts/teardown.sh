#!/usr/bin/env bash
# Delete the stack. The model S3 bucket is RETAINED (DeletionPolicy: Retain) so
# your ~250GB survives. Terminate any launched GPU instances FIRST (this does not).
set -euo pipefail
STACK="${STACK:-keep-glm52-gpu}"
REGION="${REGION:-us-west-2}"

RUNNING=$(aws ec2 describe-instances --region "$REGION" \
  --filters "Name=tag:project,Values=keep-glm52" "Name=instance-state-name,Values=running,pending" \
  --query "Reservations[].Instances[].InstanceId" --output text)
if [ -n "$RUNNING" ]; then
  echo "!! GPU instances still running: $RUNNING"
  echo "!! Terminate them first:  aws ec2 terminate-instances --region $REGION --instance-ids $RUNNING"
  exit 1
fi
echo ">> Deleting stack $STACK (S3 bucket retained)"
aws cloudformation delete-stack --region "$REGION" --stack-name "$STACK"
aws cloudformation wait stack-delete-complete --region "$REGION" --stack-name "$STACK"
echo ">> Done."
