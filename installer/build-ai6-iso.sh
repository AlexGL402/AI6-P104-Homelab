#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INSTALLER_DIR="$ROOT_DIR/installer"
CACHE_DIR="$ROOT_DIR/.cache/ai6-installer"
DIST_DIR="$ROOT_DIR/dist"

UBUNTU_VERSION=24.04.5
if [[ -n "$UBUNTU_VERSION_OVERRIDE" ]]; then UBUNTU_VERSION="$UBUNTU_VERSION_OVERRIDE"; fi
ISO_NAME="ubuntu-$UBUNTU_VERSION-live-server-amd64.iso"
ISO_URL="https://releases.ubuntu.com/$UBUNTU_VERSION/$ISO_NAME"
SUMS_URL="https://releases.ubuntu.com/$UBUNTU_VERSION/SHA256SUMS"
BASE_ISO="$CACHE_DIR/$ISO_NAME"
OUT_ISO="$DIST_DIR/AI6-Ubuntu-$UBUNTU_VERSION-amd64.iso"
if [[ $# -ge 1 ]]; then OUT_ISO="$1"; fi

for c in xorriso curl sha256sum sed awk grep python3; do
  command -v "$c" >/dev/null 2>&1 || {
    echo "Missing command: $c"
    echo "Install: sudo apt install -y xorriso curl coreutils python3"
    exit 1
  }
done

mkdir -p "$CACHE_DIR" "$DIST_DIR"

echo "[1/5] Download Ubuntu Server $UBUNTU_VERSION"
curl -fL --retry 5 --retry-delay 3 -C - -o "$BASE_ISO" "$ISO_URL"

echo "[2/5] Verify SHA256"
curl -fsSL "$SUMS_URL" -o "$CACHE_DIR/SHA256SUMS"
(
  cd "$CACHE_DIR"
  grep "  $ISO_NAME$" SHA256SUMS | sha256sum -c -
)

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/boot/grub" "$TMP/nocloud" "$TMP/ai6"

echo "[3/5] Patch GRUB for NoCloud autoinstall"
xorriso -osirrox on -indev "$BASE_ISO" -extract /boot/grub/grub.cfg "$TMP/boot/grub/grub.cfg" >/dev/null 2>&1
HAVE_LOOPBACK=0
if xorriso -osirrox on -indev "$BASE_ISO" -extract /boot/grub/loopback.cfg "$TMP/boot/grub/loopback.cfg" >/dev/null 2>&1; then
  HAVE_LOOPBACK=1
fi

patch_grub() {
  python3 - "$1" <<'PY'
import sys
p=sys.argv[1]
lines=open(p,encoding="utf-8").read().splitlines()
token=r"autoinstall ds=nocloud\;s=/cdrom/nocloud/"
out=[]
for line in lines:
    if line.lstrip().startswith("linux") and "autoinstall ds=nocloud" not in line:
        if " ---" in line:
            line=line.replace(" ---"," "+token+" ---",1)
        else:
            line=line+" "+token
    out.append(line)
open(p,"w",encoding="utf-8").write("\n".join(out)+"\n")
PY
}

patch_grub "$TMP/boot/grub/grub.cfg"
if [[ "$HAVE_LOOPBACK" == 1 ]]; then patch_grub "$TMP/boot/grub/loopback.cfg"; fi

cp "$INSTALLER_DIR/user-data" "$TMP/nocloud/user-data"
cp "$INSTALLER_DIR/meta-data" "$TMP/nocloud/meta-data"
cp "$INSTALLER_DIR/firstboot.sh" "$TMP/ai6/firstboot.sh"
cp "$INSTALLER_DIR/ai6-firstboot.service" "$TMP/ai6/ai6-firstboot.service"

rm -f "$OUT_ISO"

echo "[4/5] Create custom bootable ISO"
if [[ "$HAVE_LOOPBACK" == 1 ]]; then
  xorriso -indev "$BASE_ISO" -outdev "$OUT_ISO"     -boot_image any replay     -map "$TMP/boot/grub/grub.cfg" /boot/grub/grub.cfg     -map "$TMP/boot/grub/loopback.cfg" /boot/grub/loopback.cfg     -map "$TMP/nocloud" /nocloud     -map "$TMP/ai6" /ai6     -commit -end
else
  xorriso -indev "$BASE_ISO" -outdev "$OUT_ISO"     -boot_image any replay     -map "$TMP/boot/grub/grub.cfg" /boot/grub/grub.cfg     -map "$TMP/nocloud" /nocloud     -map "$TMP/ai6" /ai6     -commit -end
fi

echo "[5/5] Done"
ls -lh "$OUT_ISO"
sha256sum "$OUT_ISO" | tee "$OUT_ISO.sha256"
echo
echo "ISO: $OUT_ISO"
echo "Write with: sudo bash installer/write-usb.sh /dev/sdX '$OUT_ISO'"
