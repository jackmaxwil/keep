#!/bin/bash
# Serve DeepSeek-V4.1-Flash-1017 from a pool bundle with the no-wait expert cache, under oMLX at a
# pinned commit: every one of the 384 experts per layer reachable, as many resident as RAM allows.
#
#   run_v41.sh BUNDLE [PORT]
#
# BUNDLE holds models/v41-lattice (slim_checkpoint.py output), pool.json and what it names. The
# first run clones oMLX into OMLX_DIR (default ~/.cache/v41/omlx) and builds its venv with uv.
# OMLX_LATTICE_SLOTS and OMLX_LATTICE_BUDGET_GIB override the resident count (omlx_lattice.py).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
B=$(cd "${1:?usage: run_v41.sh BUNDLE [PORT]}" && pwd); PORT=${2:-8017}
OMLX=${OMLX_DIR:-$HOME/.cache/v41/omlx}; PIN=4213c3284abc7b3746a91464a7af7ccc3732b759
[ "$(uname -sm)" = "Darwin arm64" ] || { echo "needs an Apple-silicon Mac" >&2; exit 1; }
RAM=$(( $(sysctl -n hw.memsize) >> 30 ))
[ "$RAM" -ge 96 ] || echo "warning: $RAM GB of RAM; generated code was unreliable below 96 GB in our tests" >&2
for f in models/v41-lattice/compact-map.json pool.json; do
  [ -f "$B/$f" ] || { echo "$B/$f is missing" >&2; exit 1; }
done
if [ ! -f "$B/base/model_settings.json" ]; then  # the drafter costs 8 GiB: on from 128 GB of RAM
  mkdir -p "$B/base"
  [ "$RAM" -ge 120 ] && MTP=', "mtp_enabled": true, "mtp_fixed_depth": 3' || MTP=
  echo "{\"version\": 2, \"models\": {\"v41-lattice\": {\"deepseek_v41_engram_ssd_offload\": true$MTP}}}" > "$B/base/model_settings.json"
fi
if [ ! -x "$OMLX/.venv/bin/python" ]; then
  command -v git > /dev/null && command -v uv > /dev/null || { echo "needs git and uv (https://docs.astral.sh/uv/)" >&2; exit 1; }
  [ -d "$OMLX/.git" ] || git clone --quiet https://github.com/jundot/omlx "$OMLX"
  git -C "$OMLX" checkout --quiet "$PIN"
  (cd "$OMLX" && uv sync --quiet)
fi
echo "endpoint http://127.0.0.1:$PORT/v1, model v41-lattice" >&2
cd "$OMLX"
OMLX_COMPACT=$B/models/v41-lattice/compact-map.json OMLX_LATTICE=$B/pool.json \
  exec .venv/bin/python "$REPO/scripts/v41_build/omlx_lattice.py" serve --model-dir "$B/models" --base-path "$B/base" \
  --port "$PORT" --memory-guard-gb $((RAM * 3 / 4)) --max-concurrent-requests 1 --no-cache --headless
