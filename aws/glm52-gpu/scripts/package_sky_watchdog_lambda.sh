#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
REQUESTED_OUT="${1:-$ROOT/aws/glm52-gpu/dist/sky-watchdog.zip}"
OUTPUT_NAME="${REQUESTED_OUT##*/}"
case "$OUTPUT_NAME" in
  ?*.zip) ;;
  *)
    printf 'output archive must have a nonempty .zip filename: %s\n' \
      "$REQUESTED_OUT" >&2
    exit 2
    ;;
esac
case "$REQUESTED_OUT" in
  /*) OUT="$REQUESTED_OUT" ;;
  *) OUT="$PWD/$REQUESTED_OUT" ;;
esac
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
mkdir -p "$STAGE/glm52_enforcement" "$STAGE/mlx_vq/quality"
cp "$ROOT/aws/glm52-gpu/lambda/sky_watchdog_handler.py" "$STAGE/handler.py"
cp "$ROOT/aws/glm52-gpu/lambda/sky_worker_start_v2_coordinator.py" "$STAGE/"
cp "$ROOT/src/glm52_enforcement/__init__.py" "$STAGE/glm52_enforcement/"
cp "$ROOT/src/glm52_enforcement/glm52_campaign_watchdog.py" \
  "$STAGE/glm52_enforcement/"
cp "$ROOT/src/glm52_enforcement/glm52_h100_qualification.py" \
  "$STAGE/glm52_enforcement/"
cp "$ROOT/src/glm52_enforcement/glm52_sky_campaign.py" \
  "$STAGE/glm52_enforcement/"
cp "$ROOT/src/glm52_enforcement/glm52_sky_must_start.py" \
  "$STAGE/glm52_enforcement/"
cp "$ROOT/src/glm52_enforcement/glm52_teich_campaign.py" \
  "$STAGE/glm52_enforcement/"
touch "$STAGE/mlx_vq/__init__.py" "$STAGE/mlx_vq/quality/__init__.py"
cp "$ROOT/src/mlx_vq/quality/glm52_campaign_watchdog.py" "$STAGE/mlx_vq/quality/"
cp "$ROOT/src/mlx_vq/quality/glm52_h100_qualification.py" "$STAGE/mlx_vq/quality/"
cp "$ROOT/src/mlx_vq/quality/glm52_sky_campaign.py" "$STAGE/mlx_vq/quality/"
cp "$ROOT/src/mlx_vq/quality/glm52_sky_must_start.py" "$STAGE/mlx_vq/quality/"
cp "$ROOT/src/mlx_vq/quality/glm52_sky_must_start_dynamic.py" "$STAGE/mlx_vq/quality/"
cp "$ROOT/src/mlx_vq/quality/glm52_sky_terminal_state.py" "$STAGE/mlx_vq/quality/"
cp "$ROOT/src/mlx_vq/quality/glm52_sky_worker_must_start_v2.py" "$STAGE/mlx_vq/quality/"
cp "$ROOT/src/mlx_vq/quality/glm52_teich_campaign.py" "$STAGE/mlx_vq/quality/"
find "$STAGE" -type f -exec chmod 0644 {} +
TZ=UTC find "$STAGE" -exec touch -t 198001010000 {} +
(
  cd "$STAGE"
  export LC_ALL=C
  export TZ=UTC
  unset ZIPOPT
  zip -q -X "$OUT" \
    handler.py \
    sky_worker_start_v2_coordinator.py \
    glm52_enforcement/__init__.py \
    glm52_enforcement/glm52_campaign_watchdog.py \
    glm52_enforcement/glm52_h100_qualification.py \
    glm52_enforcement/glm52_sky_campaign.py \
    glm52_enforcement/glm52_sky_must_start.py \
    glm52_enforcement/glm52_teich_campaign.py \
    mlx_vq/__init__.py \
    mlx_vq/quality/__init__.py \
    mlx_vq/quality/glm52_campaign_watchdog.py \
    mlx_vq/quality/glm52_h100_qualification.py \
    mlx_vq/quality/glm52_sky_campaign.py \
    mlx_vq/quality/glm52_sky_must_start.py \
    mlx_vq/quality/glm52_sky_must_start_dynamic.py \
    mlx_vq/quality/glm52_sky_terminal_state.py \
    mlx_vq/quality/glm52_sky_worker_must_start_v2.py \
    mlx_vq/quality/glm52_teich_campaign.py
)
printf '%s\n' "$REQUESTED_OUT"
