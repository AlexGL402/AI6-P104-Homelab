#!/usr/bin/env bash
set -Eeo pipefail

DEV="$1"
ISO="$2"

if [[ -z "$DEV" ]]; then
  echo "Usage: sudo bash installer/write-usb.sh /dev/sdX [path/to/AI6.iso]"
  lsblk -o NAME,SIZE,MODEL,TRAN,MOUNTPOINTS
  exit 2
fi

if [[ -z "$ISO" ]]; then
  ISO=dist/AI6-Ubuntu-24.04.5-amd64.iso
fi

[[ "$DEV" == /dev/* ]] || { echo "Invalid device: $DEV"; exit 2; }
[[ -b "$DEV" ]] || { echo "Not a block device: $DEV"; exit 2; }
[[ -f "$ISO" ]] || { echo "ISO not found: $ISO"; exit 2; }

ROOT_SRC="$(findmnt -n -o SOURCE / || true)"
if [[ "$ROOT_SRC" == "$DEV"* ]]; then
  echo "REFUSING: $DEV appears to contain the running root filesystem ($ROOT_SRC)"
  exit 1
fi

echo "THIS WILL ERASE THE ENTIRE DEVICE: $DEV"
lsblk "$DEV" -o NAME,SIZE,MODEL,TRAN,MOUNTPOINTS
echo
read -r -p "Type ERASE to continue: " answer
[[ "$answer" == "ERASE" ]] || { echo "Cancelled."; exit 1; }

while read -r part; do
  umount "$part" 2>/dev/null || true
done < <(lsblk -lnpo NAME "$DEV" | tail -n +2)

echo "Writing $ISO -> $DEV"
dd if="$ISO" of="$DEV" bs=16M status=progress conv=fsync
sync

echo
echo "Done. USB is ready."
