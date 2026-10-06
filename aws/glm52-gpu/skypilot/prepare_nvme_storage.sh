#!/usr/bin/env bash
# Resolve the campaign root onto authenticated instance-store capacity.
set -euo pipefail

TARGET="${GLM52_NVME_TARGET:-/mnt/nvme}"
MIN_AVAILABLE_BYTES="${GLM52_MIN_AVAILABLE_BYTES:-751619276800}"
DLAMI_SOURCE=/opt/dlami/nvme

fail() {
  echo "GLM52-NVME-FAILED: $1" >&2
  exit 1
}

verify_target() {
  local root_source target_source available
  root_source=$(findmnt -n -o SOURCE /)
  target_source=$(findmnt -n -o SOURCE -T "$TARGET")
  if [ "$target_source" = "$root_source" ]; then
    fail "campaign storage resolved to the root filesystem"
  fi
  available=$(df -B1 --output=avail "$TARGET" | tail -1 | tr -d ' ')
  case "$available" in
    ''|*[!0-9]*) fail "campaign storage free-space result is invalid" ;;
  esac
  if [ "$available" -lt "$MIN_AVAILABLE_BYTES" ]; then
    fail "campaign storage has only $available bytes free"
  fi
  chmod 0755 "$TARGET"
  echo "GLM52-NVME-READY target=$TARGET source=$target_source available=$available"
}

mkdir -p "$TARGET"
if mountpoint -q "$TARGET"; then
  verify_target
  exit 0
fi

# AWS Deep Learning AMIs normally assemble the instance-store devices into an
# LVM volume at this location. Bind a campaign-specific directory rather than
# touching those already claimed block devices.
if mountpoint -q /opt/dlami/nvme; then
  mkdir -p "$DLAMI_SOURCE/keep-glm52"
  mount --bind "$DLAMI_SOURCE/keep-glm52" "$TARGET"
  verify_target
  exit 0
fi

# Some SkyPilot images mount instance storage at /mnt themselves. In that
# case TARGET already inherits the non-root filesystem and only needs proof.
if [ "$(findmnt -n -o SOURCE -T "$TARGET")" != "$(findmnt -n -o SOURCE /)" ]; then
  verify_target
  exit 0
fi

# Final fallback for a clean image: select only completely unclaimed EC2
# instance-store disks. Never format EBS, the root disk, partitions, LVM, or
# an existing RAID member.
mapfile -t DEVICES < <(
  lsblk -dn -o NAME,MODEL,TYPE |
    awk '$0 ~ /Amazon EC2 NVMe Instance Storage/ && $NF == "disk" {print "/dev/"$1}'
)
if [ "${#DEVICES[@]}" -eq 0 ]; then
  fail "no unclaimed Amazon EC2 NVMe Instance Storage devices were found"
fi
for device in "${DEVICES[@]}"; do
  test -b "$device" || fail "instance-store device is not a block device: $device"
  if lsblk -nr -o TYPE,MOUNTPOINT "$device" | tail -n +2 | grep -q .; then
    fail "instance-store device already has children: $device"
  fi
done
command -v mkfs.xfs >/dev/null || fail "mkfs.xfs is unavailable"
if [ "${#DEVICES[@]}" -eq 1 ]; then
  mkfs.xfs -f "${DEVICES[0]}"
  mount -o noatime "${DEVICES[0]}" "$TARGET"
else
  command -v mdadm >/dev/null || fail "mdadm is unavailable"
  mdadm --create /dev/md0 --run --force --level=0 \
    --raid-devices="${#DEVICES[@]}" "${DEVICES[@]}"
  mkfs.xfs -f /dev/md0
  mount -o noatime /dev/md0 "$TARGET"
fi
verify_target
