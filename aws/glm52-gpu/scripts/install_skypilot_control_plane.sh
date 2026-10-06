#!/usr/bin/env bash
# Install the pinned SkyPilot control plane outside the model-training runtime.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
: "${AWS_PROFILE:?set AWS_PROFILE to the named R&D profile}"
if [ "$AWS_PROFILE" = default ]; then
  echo "the default AWS profile is forbidden" >&2
  exit 64
fi
ACCOUNT_ID=$("$SCRIPT_DIR/assert_rnd_aws_account.sh")
CONTROL_VENV="${SKYPILOT_CONTROL_VENV:?set SKYPILOT_CONTROL_VENV to a dedicated path}"
PYTHON_VERSION="${SKYPILOT_CONTROL_PYTHON:-3.11}"
QUEUE_PORT="${SKYPILOT_QUEUE_MANAGER_PORT:-50012}"
API_PORT="${SKYPILOT_API_SERVER_PORT:-46580}"
CACHE="${UV_CACHE_DIR:-/tmp/keep-uv-cache}"

command -v uv >/dev/null
case "$QUEUE_PORT" in
  *[!0-9]*|"") echo "SKYPILOT_QUEUE_MANAGER_PORT must be an integer" >&2; exit 64 ;;
esac
if [ "$QUEUE_PORT" -lt 1024 ] || [ "$QUEUE_PORT" -gt 65535 ]; then
  echo "SKYPILOT_QUEUE_MANAGER_PORT must be between 1024 and 65535" >&2
  exit 64
fi
case "$CONTROL_VENV" in
  */.venv|*/.venv/*)
    echo "refusing to install SkyPilot in a model-training .venv" >&2
    exit 64
    ;;
esac

if [ ! -x "$CONTROL_VENV/bin/python" ]; then
  UV_CACHE_DIR="$CACHE" uv venv --python "$PYTHON_VERSION" "$CONTROL_VENV"
fi
UV_CACHE_DIR="$CACHE" uv pip install \
  --python "$CONTROL_VENV/bin/python" \
  "skypilot[aws]==0.13.0"
BOTOCORE_VERSION=$(
  "$CONTROL_VENV/bin/python" -c \
    'import botocore; print(botocore.__version__)'
)
UV_CACHE_DIR="$CACHE" uv pip install \
  --python "$CONTROL_VENV/bin/python" \
  "botocore[crt]==$BOTOCORE_VERSION"
"$CONTROL_VENV/bin/python" - "$QUEUE_PORT" \
  "$CONTROL_VENV/control-environment.json" "$ACCOUNT_ID" "$AWS_PROFILE" \
  "$API_PORT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

import sky

if sky.__version__ != "0.13.0":
    raise SystemExit(f"unexpected SkyPilot version: {sky.__version__}")

queue_port = int(sys.argv[1])
manifest_path = Path(sys.argv[2])
account_id = sys.argv[3]
aws_profile = sys.argv[4]
api_server_port = int(sys.argv[5])
queue_module = (
    Path(sky.__file__).resolve().parent
    / "server"
    / "requests"
    / "queues"
    / "mp_queue.py"
)
source = queue_module.read_text()
original = "DEFAULT_QUEUE_MANAGER_PORT = 50011"
replacement = f"DEFAULT_QUEUE_MANAGER_PORT = {queue_port}"
if replacement not in source:
    if source.count(original) != 1:
        raise SystemExit(
            f"refusing unexpected SkyPilot queue module: {queue_module}"
        )
    queue_module.write_text(source.replace(original, replacement, 1))
patched = queue_module.read_bytes()
body = {
    "schema_version": 1,
    "record_type": "glm52_skypilot_control_environment_v1",
    "account_id": account_id,
    "aws_profile": aws_profile,
    "skypilot_version": sky.__version__,
    "api_server_port": api_server_port,
    "queue_manager_port": queue_port,
    "python_executable": str(Path(sys.executable).absolute()),
    "sky_executable": str((Path(sys.executable).parent / "sky").resolve()),
    "queue_module": str(queue_module),
    "queue_module_sha256": hashlib.sha256(patched).hexdigest(),
}
body["body_sha256"] = hashlib.sha256(
    json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
manifest_path.write_text(
    json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n"
)
print(json.dumps(body, sort_keys=True))
PY
printf '%s\n' "$CONTROL_VENV/bin/sky"
