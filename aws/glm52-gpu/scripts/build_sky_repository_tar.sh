#!/usr/bin/env bash
# Build the exact secret-free repository payload consumed by the GPU worker.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
OUTPUT="${1:?usage: $0 OUTPUT_TAR_GZ}"
mkdir -p "$(dirname -- "$OUTPUT")"
test ! -e "$OUTPUT"

COPYFILE_DISABLE=1 tar -czf "$OUTPUT" -C "$REPO" \
  --exclude '.git' \
  --exclude 'runs' \
  --exclude '.venv' \
  --exclude '.env' \
  --exclude '.env.*' \
  --exclude 'node_modules' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '*.safetensors' \
  --exclude '*credentials*' \
  --exclude '*secret*' \
  src benchmarks native models tests pyproject.toml uv.lock aws \
  artifacts/quality/glm52-tokenizer-readiness-20260709.json \
  artifacts/quality/glm52-family-policy-20260709-v2.json \
  artifacts/quality/glm52-wave6-full-artifact-audit-20260709.json \
  artifacts/quality/glm52-wave6-full-bind-preflight-20260709.json \
  artifacts/quality/glm52-wave6-full-materialization-20260709/materialization.jsonl \
  artifacts/quality/glm52-training-baseline-accepted-20260711.json \
  artifacts/quality/glm52-family-eval-prompts-20260709-v2.json

python3 - "$OUTPUT" <<'PY'
import hashlib, pathlib, sys, tarfile
path = pathlib.Path(sys.argv[1])
with tarfile.open(path, "r:gz") as archive:
    members = archive.getmembers()
    if not members:
        raise SystemExit("repository tar is empty")
    for member in members:
        parts = pathlib.PurePosixPath(member.name).parts
        if member.name.startswith("/") or ".." in parts or member.isdev():
            raise SystemExit(f"unsafe repository tar member: {member.name}")
        if member.issym() or member.islnk():
            link = pathlib.PurePosixPath(member.linkname)
            if member.linkname.startswith("/") or ".." in link.parts:
                raise SystemExit(f"unsafe repository tar link: {member.name}")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
print(digest)
PY
