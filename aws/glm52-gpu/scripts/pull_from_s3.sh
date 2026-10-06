#!/usr/bin/env bash
# RUN ON THE GPU NODE. Fast-pulls everything from S3 onto local NVMe via the S3
# gateway endpoint (free, multi-GB/s). ~250GB typically lands in 1-3 minutes.
set -euo pipefail

BUCKET="${BUCKET:?set BUCKET=keep-glm52-models-<acct>-<region>}"
DEST="${DEST:-/mnt/nvme}"

for sub in source-snapshot non-vq-package profile teich-pack; do
  mkdir -p "$DEST/$sub"
  echo ">> pulling $sub ..."
  s5cmd --numworkers 64 cp "s3://$BUCKET/$sub/*" "$DEST/$sub/"
done
echo ">> Local layout under $DEST:"
du -sh "$DEST"/* 2>/dev/null || true
