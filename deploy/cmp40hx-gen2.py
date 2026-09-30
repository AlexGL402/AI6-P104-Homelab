#!/usr/bin/env python3
import mmap
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

GPU = os.environ.get("CMP40_GPU", "0000:03:00.0")

REGS = [
    (0x8872C, 0x00000006, "XVE_OVR"),
    (0x8C040, 0x80085800, "LINK_CONFIG_0"),
    (0x8841C, 0xE0B42D00, "PRIV_MISC_1"),
    (0x8C2C0, 0x068731B3, "CYA_0"),
]

def sh(*args):
    return subprocess.check_output(args, text=True).strip()

def bdf_short(dev):
    return dev.split("0000:")[-1]

def detect_root(gpu):
    p = Path(f"/sys/bus/pci/devices/{gpu}").resolve()
    parent = p.parent.name
    if not parent.startswith("0000:"):
        raise RuntimeError(f"cannot detect upstream bridge from {p}")
    return parent

def setpci(dev, expr):
    return sh("setpci", "-s", bdf_short(dev), expr)

def rd16(dev, off):
    return int(setpci(dev, f"CAP_EXP+{off:02x}.W"), 16)

def wr16(dev, off, val):
    subprocess.check_call([
        "setpci", "-s", bdf_short(dev),
        f"CAP_EXP+{off:02x}.W={val:04x}",
    ])

def set_tls(dev, gen=2):
    old = rd16(dev, 0x30)
    new = (old & 0xFFF0) | (gen & 0xF)
    wr16(dev, 0x30, new)
    rb = rd16(dev, 0x30)
    print(f"{dev}: LNKCTL2 {old:04x} -> {rb:04x}, TLS={rb & 0xF}")

def retrain(dev):
    ctl = rd16(dev, 0x10)
    wr16(dev, 0x10, ctl & ~0x20)
    time.sleep(0.300)
    ctl2 = rd16(dev, 0x10)
    wr16(dev, 0x10, ctl2 | 0x20)

def current_speed():
    p = Path(f"/sys/bus/pci/devices/{GPU}/current_link_speed")
    try:
        return p.read_text().strip()
    except Exception:
        return "unknown"

def main():
    if os.geteuid() != 0:
        print("Run as root", file=sys.stderr)
        return 1

    root = detect_root(GPU)
    print(f"GPU  = {GPU}")
    print(f"ROOT = {root}")

    resource = f"/sys/bus/pci/devices/{GPU}/resource0"
    fd = os.open(resource, os.O_RDWR | os.O_SYNC)
    mm = mmap.mmap(fd, 0x90000)

    def rd32(off):
        return struct.unpack_from("<I", mm, off)[0]

    def wr32(off, val):
        struct.pack_into("<I", mm, off, val)
        mm.flush()

    try:
        boot0 = rd32(0)
        print(f"BOOT_0 = 0x{boot0:08X}")
        if (boot0 & 0xFF000000) != 0x16000000:
            print("ABORT: BOOT_0 is not TU10x/TU106")
            return 2

        print("\n=== BAR0 BEFORE ===")
        for off, _, name in REGS:
            print(f"{name:14s} 0x{off:05X}: 0x{rd32(off):08X}")

        print("\n=== WRITE PL0 ===")
        for off, val, name in REGS:
            wr32(off, val)
            rb = rd32(off)
            print(f"{name:14s}: wrote 0x{val:08X}, read 0x{rb:08X}")
            if rb != val:
                print("ABORT: readback mismatch")
                return 3

        print("\n=== TLS GEN2 ===")
        set_tls(GPU, 2)
        set_tls(root, 2)

        print(f"\nInitial link: {current_speed()}")
        for i in range(6):
            dev = root if i % 2 == 0 else GPU
            tag = "ROOT" if i % 2 == 0 else "GPU"
            print(f"\nRetrain #{i+1} via {tag}")
            retrain(dev)
            time.sleep(2.2)
            speed = current_speed()
            print("link =", speed)
            if speed.startswith("5.0"):
                print("\n*** GEN2 ACHIEVED ***")
                break

        print("\n=== FINAL ===")
        print("GPU TLS :", rd16(GPU, 0x30) & 0xF)
        print("ROOT TLS:", rd16(root, 0x30) & 0xF)
        print("Link    :", current_speed())

        return 0 if current_speed().startswith("5.0") else 4
    finally:
        mm.close()
        os.close(fd)

if __name__ == "__main__":
    raise SystemExit(main())
