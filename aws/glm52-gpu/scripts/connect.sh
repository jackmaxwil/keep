#!/usr/bin/env bash
# Open an interactive shell on the GPU node via SSM Session Manager (no SSH keys).
# Requires the Session Manager plugin: https://docs.aws.amazon.com/cli/latest/userguide/session-manager-plugin.html
set -euo pipefail
REGION="${REGION:-us-west-2}"
ID="${ID:?set ID=<instance-id>}"
exec aws ssm start-session --region "$REGION" --target "$ID"
