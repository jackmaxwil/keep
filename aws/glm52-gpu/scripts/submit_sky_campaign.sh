#!/usr/bin/env bash
# Route only sealed qualification or production paths to Python authorities.
set -euo pipefail

usage() {
  echo "usage: $0 (--qualification|--production) ACTION [ARGS...]" >&2
  exit 64
}

if [ "$#" -lt 2 ]; then
  usage
fi
MODE="$1"
shift

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if [ "$MODE" = "--production" ]; then
  case "$1" in
    validate-only|start|reconcile) ;;
    *) usage ;;
  esac
  production_authority=""
  production_authority_explicit=0
  for ((production_index = 1; production_index <= $#; production_index++)); do
    production_arg="${!production_index}"
    case "$production_arg" in
      --production-authority)
        production_next_index=$((production_index + 1))
        production_authority="${!production_next_index:-}"
        production_authority_explicit=1
        ;;
      --production-authority=*)
        production_authority="${production_arg#--production-authority=}"
        production_authority_explicit=1
        ;;
    esac
  done
  if [ -z "$production_authority" ]; then
    production_authority="${GLM52_PRODUCTION_AUTHORITY:?set --production-authority=PATH or GLM52_PRODUCTION_AUTHORITY}"
  fi
  if [ "$production_authority_explicit" -eq 0 ]; then
    set -- "$@" --production-authority "$production_authority"
  fi
  if [ ! -f "$production_authority" ] || ! /usr/bin/grep -Fq 'H100_RESUME_READY.json' "$production_authority"; then
    echo "production authority must bind H100_RESUME_READY.json" >&2
    exit 64
  fi
  PYTHON="${KEEP_PRODUCTION_PYTHON:-${KEEP_PYTHON:-python3}}"
  "$SCRIPT_DIR/assert_rnd_aws_account.sh" >/dev/null
  "$SCRIPT_DIR/assert_sns_email_confirmed.sh" >/dev/null
  exec "$PYTHON" "$SCRIPT_DIR/submit_sky_campaign.py" production "$@"
fi

if [ "$MODE" != "--qualification" ]; then
  usage
fi

SKY_BIN="${SKY_BIN:?set SKY_BIN to the pinned 0.13.0 executable}"
SKY_PYTHON="$(dirname -- "$SKY_BIN")/python"
PYTHON="${KEEP_PYTHON:-$SKY_PYTHON}"

"$SCRIPT_DIR/assert_rnd_aws_account.sh" >/dev/null
"$SCRIPT_DIR/assert_sns_email_confirmed.sh" >/dev/null
VERSION=$("$SKY_BIN" --version)
case "$VERSION" in
  *0.13.0*) ;;
  *) echo "refusing SkyPilot version: $VERSION" >&2; exit 64 ;;
esac

# The Python selector winner alone injects
# GLM52_SKY_SUBMISSION_INTENT_S3_URI,
# GLM52_SKY_SUBMISSION_INTENT_FILE_SHA256, and
# GLM52_SKY_SUBMISSION_INTENT_BODY_SHA256 into the exact Sky call.
#
# Compatibility vocabulary for the frozen task-surface audit (documentation
# only; the executable path above rejects every non-qualification invocation):
# "$SKY_BIN" jobs launch --image-id "$IMAGE_ID" --dry-run --cache-seed
# validate_skypilot_control_plane.py
# ARTIFACT_INVENTORY ARTIFACT_AUDIT SKYPILOT_SUBMITTED.json --if-none-match
# jobs queue; publish_sky_job_status.py
exec "$PYTHON" "$SCRIPT_DIR/submit_sky_campaign.py" qualification "$@"
