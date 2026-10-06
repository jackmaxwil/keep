#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
OUT_INPUT="${1:-$ROOT/aws/glm52-gpu/dist/sky-must-start-cancel.zip}"
mkdir -p "$(dirname -- "$OUT_INPUT")"
OUT_DIR=$(CDPATH= cd -- "$(dirname -- "$OUT_INPUT")" && pwd)
OUT="$OUT_DIR/$(basename -- "$OUT_INPUT")"
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/mlx_vq/quality"
cp "$ROOT/aws/glm52-gpu/lambda/sky_must_start_cancel_handler.py" \
  "$STAGE/handler.py"
touch "$STAGE/mlx_vq/__init__.py" "$STAGE/mlx_vq/quality/__init__.py"
cp "$ROOT/src/mlx_vq/quality/glm52_campaign_watchdog.py" \
  "$STAGE/mlx_vq/quality/"
cp "$ROOT/src/mlx_vq/quality/glm52_sky_campaign.py" \
  "$STAGE/mlx_vq/quality/"
cp "$ROOT/src/mlx_vq/quality/glm52_sky_must_start.py" \
  "$STAGE/mlx_vq/quality/"
ARCHIVE="$STAGE/sky-must-start-cancel.zip"
(
  cd "$STAGE"
  zip -q -X -r "$ARCHIVE" handler.py mlx_vq
)
mv -f "$ARCHIVE" "$OUT"
SHA256=$(shasum -a 256 "$OUT" | awk '{print $1}')
python3 - "$OUT" "$SHA256" <<'PY'
import json
import sys

path, digest = sys.argv[1:]
print(
    json.dumps(
        {
            "path": path,
            "requires_s3_object_version": True,
            "s3_key": f"lambda/sky-must-start-cancel/{digest}.zip",
            "sha256": digest,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)
PY
