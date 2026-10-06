#!/usr/bin/env bash
# Publish immutable spend records first, consolidated ledger second, latest last.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT="${1:?campaign root required}"
S3_RUNTIME="${2:?S3 runtime prefix required}"
REGION=us-west-2
LOCAL="$ROOT/runtime/spend-ledger"
test -f "$LOCAL/GPU_SPEND_LEDGER.jsonl"
test -f "$LOCAL/latest.json"
aws sts get-caller-identity |
  "$SCRIPT_DIR/assert_rnd_aws_account.py" >/dev/null

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
while IFS= read -r record; do
  name=${record##*/}
  remote="${S3_RUNTIME%/}/spend-ledger/records/$name"
  bucket_and_key=${remote#s3://}
  bucket=${bucket_and_key%%/*}
  key=${bucket_and_key#*/}
  if aws s3api head-object \
    --bucket "$bucket" --key "$key" --region "$REGION" >/dev/null 2>&1; then
    aws s3 cp "$remote" "$WORK/$name" \
      --region "$REGION" --only-show-errors
    cmp "$record" "$WORK/$name"
    continue
  fi
  set +e
  RESULT=$(aws s3api put-object \
    --bucket "$bucket" --key "$key" --body "$record" \
    --if-none-match '*' --region "$REGION" 2>&1)
  STATUS=$?
  set -e
  if [ "$STATUS" -ne 0 ]; then
    case "$RESULT" in
      *PreconditionFailed*|*" 412 "*)
        aws s3 cp "$remote" "$WORK/$name" \
          --region "$REGION" --only-show-errors
        cmp "$record" "$WORK/$name"
        ;;
      *)
        echo "$RESULT" >&2
        exit "$STATUS"
        ;;
    esac
  fi
done < <(find "$LOCAL/records" -type f -name '*.json' -print | sort)

aws s3 cp "$LOCAL/GPU_SPEND_LEDGER.jsonl" \
  "${S3_RUNTIME%/}/GPU_SPEND_LEDGER.jsonl" \
  --region "$REGION" --only-show-errors
aws s3 cp "$LOCAL/latest.json" \
  "${S3_RUNTIME%/}/GPU_SPEND_LEDGER_LATEST.json" \
  --region "$REGION" --only-show-errors
