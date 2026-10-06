#!/usr/bin/env bash
# Restore only the immutable record chain named by the published latest marker.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REQUIRE_LATEST=0
PROFILE=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --require-latest)
      REQUIRE_LATEST=1
      shift
      ;;
    --profile)
      PROFILE="${2:?--profile requires a value}"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done
ROOT="${1:?campaign root required}"
S3_RUNTIME="${2:?S3 runtime prefix required}"
shift 2
if [ "${1:-}" = "--require-latest" ]; then
  REQUIRE_LATEST=1
  shift
fi
if [ "$#" -ne 0 ]; then
  echo "usage: $0 [--require-latest] [--profile PROFILE] ROOT S3_RUNTIME [--require-latest]" >&2
  exit 64
fi
if [ "$REQUIRE_LATEST" -eq 1 ] && [ "$PROFILE" != "keep-gpu" ]; then
  echo "--profile must be exactly keep-gpu in --require-latest mode" >&2
  exit 64
fi
AWS_PROFILE_ARGS=()
if [ -n "$PROFILE" ]; then
  AWS_PROFILE_ARGS=(--profile "$PROFILE")
fi
REGION=us-west-2
LATEST_URI="${S3_RUNTIME%/}/GPU_SPEND_LEDGER_LATEST.json"
LEDGER_URI="${S3_RUNTIME%/}/GPU_SPEND_LEDGER.jsonl"
bucket_and_key=${LATEST_URI#s3://}
bucket=${bucket_and_key%%/*}
latest_key=${bucket_and_key#*/}
aws sts get-caller-identity "${AWS_PROFILE_ARGS[@]}" |
  "$SCRIPT_DIR/assert_rnd_aws_account.py" >/dev/null
mkdir -p "$ROOT/runtime"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
set +e
aws s3api head-object --bucket "$bucket" --key "$latest_key" \
  --region "$REGION" "${AWS_PROFILE_ARGS[@]}" >"$WORK/latest-head.stdout" \
  2>"$WORK/latest-head.stderr"
LATEST_HEAD_STATUS=$?
set -e
if [ "$LATEST_HEAD_STATUS" -eq 0 ]; then
  mkdir -p "$WORK/records"
  aws s3 cp "$LATEST_URI" "$WORK/latest.json" \
    --region "$REGION" --only-show-errors "${AWS_PROFILE_ARGS[@]}"
  if [ "$REQUIRE_LATEST" -eq 1 ]; then
    record_prefix=${latest_key%GPU_SPEND_LEDGER_LATEST.json}
    record_prefix="${record_prefix}spend-ledger/records/"
    aws s3api list-objects-v2 \
      --bucket "$bucket" --prefix "$record_prefix" \
      --region "$REGION" --output json \
      "${AWS_PROFILE_ARGS[@]}" >"$WORK/record-inventory.json"
    "$SCRIPT_DIR/restore_gpu_spend_ledger.py" \
      --latest "$WORK/latest.json" \
      --s3-key-inventory "$WORK/record-inventory.json" \
      --s3-record-prefix "$record_prefix" \
      --require-nonempty --list-records >"$WORK/record-keys.txt"
  else
    "$SCRIPT_DIR/restore_gpu_spend_ledger.py" \
      --latest "$WORK/latest.json" --list-records >"$WORK/record-keys.txt"
  fi
  while IFS= read -r name; do
    test -n "$name"
    aws s3 cp \
      "${S3_RUNTIME%/}/spend-ledger/records/$name" \
      "$WORK/records/$name" --region "$REGION" --only-show-errors \
      "${AWS_PROFILE_ARGS[@]}"
  done < "$WORK/record-keys.txt"
  "$SCRIPT_DIR/restore_gpu_spend_ledger.py" \
    --latest "$WORK/latest.json" --records-dir "$WORK/records" \
    --output "$ROOT/runtime/GPU_SPEND_LEDGER.jsonl"
  exit 0
fi

if [ "$REQUIRE_LATEST" -eq 1 ]; then
  LATEST_HEAD_ERROR=$(
    cat "$WORK/latest-head.stderr" "$WORK/latest-head.stdout"
  )
  case "$LATEST_HEAD_ERROR" in
    *"(404)"*|*" 404 "*|*"Not Found"*|*"NoSuchKey"*)
      echo "required GPU spend latest marker is missing: $LATEST_URI" >&2
      exit 66
      ;;
    *)
      printf '%s\n' "$LATEST_HEAD_ERROR" >&2
      exit "$LATEST_HEAD_STATUS"
      ;;
  esac
fi

# Compatibility for a ledger created before immutable spend records existed.
ledger_key=${LEDGER_URI#s3://$bucket/}
if aws s3api head-object --bucket "$bucket" --key "$ledger_key" \
  --region "$REGION" "${AWS_PROFILE_ARGS[@]}" >/dev/null 2>&1; then
  aws s3 cp "$LEDGER_URI" "$ROOT/runtime/GPU_SPEND_LEDGER.jsonl" \
    --region "$REGION" --only-show-errors "${AWS_PROFILE_ARGS[@]}"
fi
