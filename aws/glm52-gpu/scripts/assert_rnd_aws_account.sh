#!/usr/bin/env bash
# Fail closed unless an explicit named AWS profile resolves to the R&D account.
set -euo pipefail

: "${AWS_PROFILE:?set AWS_PROFILE to a named R&D AWS SSO profile}"
if [ "$AWS_PROFILE" = "default" ]; then
  echo "refusing AWS operation: AWS_PROFILE=default is not permitted" >&2
  exit 2
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
AWS_PAGER="" aws sts get-caller-identity \
  --profile "$AWS_PROFILE" \
  --output json |
  "$SCRIPT_DIR/assert_rnd_aws_account.py"
