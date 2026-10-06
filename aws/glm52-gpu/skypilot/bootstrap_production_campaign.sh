#!/usr/bin/env bash
# Install and prove the immutable Task 10 worker contract.
set -euo pipefail

if [ "$#" -ne 0 ]; then
  echo "production bootstrap accepts no arguments" >&2
  exit 64
fi

: "${GLM52_DESCRIPTOR_FILE_SHA256:?authenticated descriptor hash is required}"
: "${GLM52_DESCRIPTOR_VERSION_ID:?opaque descriptor VersionId is required}"
: "${GLM52_APPROVAL_S3_URI:?authenticated approval URI is required}"
: "${GLM52_APPROVAL_VERSION_ID:?opaque approval VersionId is required}"
: "${GLM52_APPROVAL_FILE_SHA256:?authenticated approval hash is required}"
: "${GLM52_SUBMISSION_INTENT_S3_URI:?authenticated intent URI is required}"
: "${GLM52_SUBMISSION_INTENT_VERSION_ID:?opaque intent VersionId is required}"
: "${GLM52_SUBMISSION_INTENT_FILE_SHA256:?authenticated intent hash is required}"
: "${GLM52_SUBMISSION_INTENT_BODY_SHA256:?authenticated intent body hash is required}"
: "${GLM52_REPOSITORY_ARCHIVE_FILE_SHA256:?authenticated archive hash is required}"
: "${GLM52_REPOSITORY_ARCHIVE_VERSION_ID:?opaque archive VersionId is required}"

REPO=/opt/keep-campaign/repo
DOWNLOAD=/opt/keep-campaign/download
ROOT=/mnt/nvme/glm52-campaign
DESCRIPTOR=/etc/keep-glm52/campaign.json
WORKER_DESCRIPTOR=/etc/keep-glm52/worker-bootstrap.json
STATE=/var/lib/keep-glm52/deadline-state.json
UNITS="$REPO/aws/glm52-gpu/skypilot/production"

install -d -m 0755 /etc/keep-glm52 /var/lib/keep-glm52
install -d -m 0755 \
  "$ROOT" \
  "$ROOT/runtime" \
  "$ROOT/ledger" \
  "$ROOT/teacher-cache" \
  "$ROOT/spike-resume" \
  "$ROOT/training"
printf '%s  %s\n' \
  "$GLM52_DESCRIPTOR_FILE_SHA256" "$DOWNLOAD/campaign.json" |
  sha256sum -c -
"$REPO/aws/glm52-gpu/scripts/install_task10_worker_descriptors.py"
test -s "$WORKER_DESCRIPTOR"
test -s "$DESCRIPTOR"

"$REPO/aws/glm52-gpu/scripts/download_task10_runtime_inputs.py"
printf '%s  %s\n' \
  "$GLM52_APPROVAL_FILE_SHA256" "$ROOT/runtime/approval.json" |
  sha256sum -c -
printf '%s  %s\n' \
  "$GLM52_SUBMISSION_INTENT_FILE_SHA256" "$ROOT/runtime/intent.json" |
  sha256sum -c -
"$REPO/aws/glm52-gpu/scripts/authenticate_task10_worker_inputs.py"

for unit in \
  keep-glm52-campaign.service \
  keep-glm52-deadline.service \
  keep-glm52-deadline.timer
do
  install -m 0644 "$UNITS/$unit" "/etc/systemd/system/$unit"
  cmp -s "$UNITS/$unit" "/etc/systemd/system/$unit"
done

"$REPO/aws/glm52-gpu/scripts/materialize_task10_worker_observation.py"
"$REPO/aws/glm52-gpu/scripts/glm52_deadline_guard.py" initialize
test -s "$STATE"

systemctl daemon-reload
systemctl enable keep-glm52-deadline.timer
systemctl start keep-glm52-deadline.timer
systemctl start keep-glm52-campaign.service

"$REPO/aws/glm52-gpu/scripts/publish_task10_bootstrap_ready.py"
test -s "$ROOT/BOOTSTRAP_READY.json"
"$REPO/aws/glm52-gpu/scripts/wait_task10_bootstrap_ledger.py"
test -s "$ROOT/runtime/BOOTSTRAP_LEDGER_DURABLE.json"
