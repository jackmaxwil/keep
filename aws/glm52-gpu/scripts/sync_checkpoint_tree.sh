#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="${1:?local checkpoint root required}"
S3_ROOT="${2:?S3 checkpoint prefix required}"

# Immutable payloads and ledger records go first. Mutable latest markers and
# immutable LM-head completion markers are published only after their objects.
aws s3 sync "$LOCAL_ROOT" "$S3_ROOT" --only-show-errors \
  --exclude 'latest.json' --exclude 'markers/*' --exclude '*/markers/*'

while IFS= read -r marker; do
  relative="${marker#"$LOCAL_ROOT"/}"
  aws s3 cp "$marker" "${S3_ROOT%/}/$relative" --only-show-errors
done < <(find "$LOCAL_ROOT" -type f -path '*/markers/*.json' -print | sort)

while IFS= read -r marker; do
  relative="${marker#"$LOCAL_ROOT"/}"
  aws s3 cp "$marker" "${S3_ROOT%/}/$relative" --only-show-errors
done < <(find "$LOCAL_ROOT" -type f -name latest.json -print | sort)
