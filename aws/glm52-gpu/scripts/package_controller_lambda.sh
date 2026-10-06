#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="${1:-$ROOT/aws/glm52-gpu/dist/capacity-block-controller.zip}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$(dirname "$OUT")"
cp "$ROOT/aws/glm52-gpu/lambda/handler.py" "$STAGE/handler.py"
cp "$ROOT/src/mlx_vq/quality/glm52_capacity_block_controller.py" \
  "$STAGE/glm52_capacity_block_controller.py"
(
  cd "$STAGE"
  zip -q -X "$OUT" handler.py glm52_capacity_block_controller.py
)
shasum -a 256 "$OUT"
