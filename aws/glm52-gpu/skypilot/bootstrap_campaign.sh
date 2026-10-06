#!/usr/bin/env bash
# Authenticate staged inputs and bootstrap the exact extracted campaign tree.
set -euo pipefail

DESCRIPTOR="${CAMPAIGN_DESCRIPTOR:-/etc/keep-glm52/campaign.json}"
APPROVAL="${GPU_SPEND_APPROVAL:-/etc/keep-glm52/GPU_SPEND_APPROVAL.json}"
INTENT="${SKY_SUBMISSION_INTENT:-/etc/keep-glm52/SKYPILOT_SUBMISSION_INTENT.json}"
LATCH="${WORKER_START_LATCH:-/etc/keep-glm52/WORKER_START_LATCH.json}"
LATCH_TRANSPORT="${WORKER_START_LATCH_TRANSPORT:-/etc/keep-glm52/WORKER_START_LATCH_TRANSPORT.json}"
ACCEPTED="${WORKER_START_ACCEPTED:-/etc/keep-glm52/WORKER_START_ACCEPTED.json}"
RECEIPT="${WORKER_START_ADMISSION_RECEIPT:-/etc/keep-glm52/WORKER_START_ADMISSION_RECEIPT.json}"
ROOT="${ROOT:-/mnt/nvme/glm52-campaign}"
REPO="${KEEP_REPO_DIR:-/opt/keep-campaign/repo}"
MANAGED_MODE="${GLM52_MANAGED_MODE:?GLM52_MANAGED_MODE is required}"
LIVE_WORKER="${GLM52_LIVE_SKY_WORKER:-0}"
LOCAL_REHEARSAL=0
unset GLM_MLX_WIRED_LIMIT_GB GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB

test "$MANAGED_MODE" = qualification
case "$LIVE_WORKER" in
  0|1) ;;
  *) echo "invalid GLM52_LIVE_SKY_WORKER" >&2; exit 70 ;;
esac
if [ "$LIVE_WORKER" -eq 1 ]; then
  test "${GLM52_BOOTSTRAP_REHEARSAL:-0}" != 1
  /usr/bin/python3 - "${SKYPILOT_MANAGED_JOB_ID:?required}" <<'PY'
import re
import sys

if re.fullmatch(r"[1-9][0-9]*", sys.argv[1]) is None:
    raise SystemExit("SkyPilot managed job ID is not canonical")
PY
elif [ "${GLM52_BOOTSTRAP_REHEARSAL:-}" = 1 ]; then
  LOCAL_REHEARSAL=1
fi
if [ "$LOCAL_REHEARSAL" -eq 0 ]; then
  test "$DESCRIPTOR" = /etc/keep-glm52/campaign.json
  test "$APPROVAL" = /etc/keep-glm52/GPU_SPEND_APPROVAL.json
  test "$INTENT" = /etc/keep-glm52/SKYPILOT_SUBMISSION_INTENT.json
  test "$LATCH" = /etc/keep-glm52/WORKER_START_LATCH.json
  test "$LATCH_TRANSPORT" = \
    /etc/keep-glm52/WORKER_START_LATCH_TRANSPORT.json
  test "$ACCEPTED" = /etc/keep-glm52/WORKER_START_ACCEPTED.json
  test "$RECEIPT" = /etc/keep-glm52/WORKER_START_ADMISSION_RECEIPT.json
  test "$ROOT" = /mnt/nvme/glm52-campaign
  test "$REPO" = /opt/keep-campaign/repo
fi
for path in \
  "$DESCRIPTOR" "$APPROVAL" "$INTENT" "$LATCH" "$LATCH_TRANSPORT" \
  "$ACCEPTED" "$RECEIPT" "$ROOT" "$REPO"; do
  case "$path" in
    /*) ;;
    *) echo "bootstrap path is not absolute: $path" >&2; exit 70 ;;
  esac
done
test -f "$DESCRIPTOR"
test -f "$APPROVAL"
test -f "$REPO/src/mlx_vq/quality/glm52_sky_campaign.py"
/usr/bin/python3 - "$DESCRIPTOR" "$APPROVAL" "$REPO" <<'PY'
import hashlib
import importlib.util
import json
import pathlib
import sys

descriptor_path = pathlib.Path(sys.argv[1])
approval_path = pathlib.Path(sys.argv[2])
repo = pathlib.Path(sys.argv[3])
frozen_sources = {
    repo / "aws/glm52-gpu/skypilot/publish_worker_start_v2.py":
        "4d1d47aef6dd211c20b52f05ede9bd6d32c35433e305f2b1bd56b48c83b9051e",
    repo / "src/mlx_vq/quality/glm52_sky_campaign.py":
        "77412115f6b8ce325177864418d838e0335d536329e8e7565d7811bcc806cd94",
    repo / "src/glm52_enforcement/glm52_sky_campaign.py":
        "f023eaeddf8fac73fa6546e3c4a0b60dd32e5266f06e1519fe21e0444d445d03",
    repo / "src/mlx_vq/quality/glm52_sky_must_start.py":
        "478ac3a16ce8a77a4e844c1e0a3dca99454d50e7233fd7cd67dfa184b486860a",
    repo / "src/glm52_enforcement/glm52_sky_must_start.py":
        "e910d3d03f7b30b7b9b668e214b61ecc7cbd641a53dedabd620b8e14573d33f3",
    repo / "src/mlx_vq/quality/glm52_sky_must_start_dynamic.py":
        "527019a41b54f810e9f98734c2ca99acc5a7b370344adf48c80f46f925396e18",
    repo / "src/mlx_vq/quality/glm52_sky_worker_must_start_v2.py":
        "8dfad6682d8afc9774dc31385cfc9ecc8c8e2f80d9a535e3c8d4cf6cf0fa97df",
    repo / "aws/glm52-gpu/lambda/sky_worker_start_v2_coordinator.py":
        "efa7c48ad259e10c6ce0e4cb1db9ff0a6e6a9fa203274329a7c2eb5203366406",
}
for path, expected in frozen_sources.items():
    if (
        not path.is_file()
        or path.is_symlink()
        or hashlib.sha256(path.read_bytes()).hexdigest() != expected
    ):
        raise SystemExit(f"frozen worker-v2 source mismatch: {path}")
module_path = repo / "src/mlx_vq/quality/glm52_sky_campaign.py"
spec = importlib.util.spec_from_file_location("_glm52_sky_bootstrap", module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
descriptor_raw = descriptor_path.read_bytes()
approval_raw = approval_path.read_bytes()
descriptor = module.validate_sky_campaign_descriptor(json.loads(descriptor_raw))
module.validate_gpu_spend_approval(json.loads(approval_raw))
if hashlib.sha256(approval_raw).hexdigest() != descriptor["approval_sha256"]:
    raise SystemExit("GPU spend approval file SHA-256 mismatch")
if descriptor["descriptor_body_sha256"] != json.loads(descriptor_raw)[
    "descriptor_body_sha256"
]:
    raise SystemExit("descriptor_body_sha256 authority drift")
PY
aws sts get-caller-identity |
  /usr/bin/python3 \
    "$REPO/aws/glm52-gpu/scripts/assert_rnd_aws_account.py" >/dev/null

if [ "$LOCAL_REHEARSAL" -eq 1 ]; then
  export GLM52_BOOTSTRAP_REHEARSAL=1
else
  unset GLM52_BOOTSTRAP_REHEARSAL
  test -f "$INTENT"
  test -f "$LATCH"
  test -f "$LATCH_TRANSPORT"
  test -f "$ACCEPTED"
  test -f "$RECEIPT"
  /usr/bin/python3 -I "$REPO/aws/glm52-gpu/skypilot/publish_worker_start_v2.py" \
  verify-receipt-v2 \
    --publisher-file-sha256 4d1d47aef6dd211c20b52f05ede9bd6d32c35433e305f2b1bd56b48c83b9051e \
    --campaign-policy "$REPO/src/mlx_vq/quality/glm52_sky_campaign.py" \
    --campaign-policy-file-sha256 77412115f6b8ce325177864418d838e0335d536329e8e7565d7811bcc806cd94 \
    --campaign-policy-native "$REPO/src/glm52_enforcement/glm52_sky_campaign.py" \
    --campaign-policy-native-file-sha256 f023eaeddf8fac73fa6546e3c4a0b60dd32e5266f06e1519fe21e0444d445d03 \
    --must-start-policy "$REPO/src/mlx_vq/quality/glm52_sky_must_start.py" \
    --must-start-policy-file-sha256 478ac3a16ce8a77a4e844c1e0a3dca99454d50e7233fd7cd67dfa184b486860a \
    --must-start-policy-native "$REPO/src/glm52_enforcement/glm52_sky_must_start.py" \
    --must-start-policy-native-file-sha256 e910d3d03f7b30b7b9b668e214b61ecc7cbd641a53dedabd620b8e14573d33f3 \
    --dynamic-policy "$REPO/src/mlx_vq/quality/glm52_sky_must_start_dynamic.py" \
    --dynamic-policy-file-sha256 527019a41b54f810e9f98734c2ca99acc5a7b370344adf48c80f46f925396e18 \
    --worker-policy "$REPO/src/mlx_vq/quality/glm52_sky_worker_must_start_v2.py" \
    --worker-policy-file-sha256 8dfad6682d8afc9774dc31385cfc9ecc8c8e2f80d9a535e3c8d4cf6cf0fa97df \
    --coordinator "$REPO/aws/glm52-gpu/lambda/sky_worker_start_v2_coordinator.py" \
    --coordinator-file-sha256 efa7c48ad259e10c6ce0e4cb1db9ff0a6e6a9fa203274329a7c2eb5203366406 \
    --descriptor "$DESCRIPTOR" \
    --descriptor-s3-uri "${GLM52_CAMPAIGN_DESCRIPTOR_S3_URI:?required}" \
    --descriptor-file-sha256 "${GLM52_CAMPAIGN_DESCRIPTOR_SHA256:?required}" \
    --intent "$INTENT" \
    --intent-s3-uri "${GLM52_SKY_SUBMISSION_INTENT_S3_URI:?required}" \
    --intent-file-sha256 "${GLM52_SKY_SUBMISSION_INTENT_FILE_SHA256:?required}" \
    --intent-body-sha256 "${GLM52_SKY_SUBMISSION_INTENT_BODY_SHA256:?required}" \
    --latch "$LATCH" \
    --latch-transport "$LATCH_TRANSPORT" \
    --accepted "$ACCEPTED" \
    --admission-receipt "$RECEIPT"
  "$REPO/aws/glm52-gpu/skypilot/prepare_nvme_storage.sh"
fi

BUCKET=$(/usr/bin/python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["bucket"])' \
  "$DESCRIPTOR")
BUCKET="$BUCKET" ROOT="$ROOT" CAMPAIGN_DESCRIPTOR="$DESCRIPTOR" \
  KEEP_REPO_DIR="$REPO" KEEP_REPO_PRESTAGED=1 \
  "$REPO/aws/glm52-gpu/scripts/gpu_env_setup.sh"

if [ "$LOCAL_REHEARSAL" -eq 1 ]; then
  echo "SKYPILOT-BOOTSTRAP-REHEARSAL-OK"
  exit 0
fi

install -d -m 0755 "$ROOT/runtime" /run/keep-glm52
install -m 0644 \
  "$REPO/aws/glm52-gpu/skypilot/keep-glm52-campaign.service" \
  /etc/systemd/system/keep-glm52-campaign.service
systemctl daemon-reload
echo "SKYPILOT-BOOTSTRAP-OK"
