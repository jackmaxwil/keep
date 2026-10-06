#!/usr/bin/env bash
set -euo pipefail

DESCRIPTOR=/etc/keep-glm52/campaign.json
DEADLINE_STATE=/var/lib/keep-glm52/deadline-state.json
if [ "$#" -ne 0 ]; then
  if [ "$#" -ne 4 ] ||
     [ "$1" != "--descriptor" ] ||
     [ "$2" != "$DESCRIPTOR" ] ||
     [ "$3" != "--deadline-state" ] ||
     [ "$4" != "$DEADLINE_STATE" ]; then
    echo "refusing caller-selected campaign arguments" >&2
    exit 64
  fi
fi

export CAMPAIGN_DESCRIPTOR="$DESCRIPTOR"
export ROOT=/mnt/nvme/glm52-campaign
export KEEP_REPO_DIR=/opt/keep-campaign/repo
export GLM52_DEADLINE_STATE="$DEADLINE_STATE"
unset GLM_MLX_WIRED_LIMIT_GB GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB

install -d -m 0755 "$ROOT/runtime"
exec 9>"$ROOT/runtime/heavy-job.lock"
flock -n 9 || {
  echo "another heavy campaign process owns the fixed lock" >&2
  exit 75
}

exec /usr/bin/python3 "$KEEP_REPO_DIR/aws/glm52-gpu/scripts/run_task10_production_campaign.py"
