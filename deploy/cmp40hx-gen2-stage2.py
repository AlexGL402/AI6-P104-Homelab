#!/usr/bin/env python3
"""CMP40HX Linux Stage2 Gen2 bring-up.

Tested on Huanan X79 v2.49 + CMP 40HX (TU106) at 03:00.0, root 00:02.0.
Requires the UEFI compute unlock to have succeeded first (SS0=0x88888888, SS1=0x8).

Sequence proven on 2026-09-30:
  unbind nvidia
  -> PL0 publish
  -> root Link Disable/Enable
  -> PCI remove/rescan (nvidia re-probe)
  -> PL0 publish again
  -> TLS=Gen2 on GPU/root
  -> root retrain
  -> verify Gen2 x16

Do NOT run automatically at boot until the UEFI ordering is made deterministic.
"""

import mmap
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

GPU = os.environ.get("CMP40_GPU", "0000:03:00.0")
ROOT = os.environ.get("CMP40_ROOT", "0000:00:02.0")

REGS = [
    (0x8872C, 0x00000006, "XVE_OVR"),
    (0x8C040, 0x80085800, "LINK_CONFIG_0"),
    (0x8841C, 0xE0B42D00, "PRIV_MISC_1"),
    (0x8C2C0, 0x068731B3, "CYA_0"),
]

def log(s=""):
    print(s, flush=True)

def short(dev):
    return dev.split("0000:")[-1]

def run(args, check=True):
    log("+ " + " ".join(args))
    return subprocess.run(args, text=True, capture_output=True, check=check)

def setpci(dev, expr):
    r = run(["setpci", "-s", short(dev), expr])
    return r.stdout.strip()

def rd16(dev, off):
    return int(setpci(dev, f"CAP_EXP+{off:02x}.W"), 16)

def wr16(dev, off, val):
    setpci(dev, f"CAP_EXP+{off:02x}.W={val:04x}")

def set_tls(dev, gen=2):
    old = rd16(dev, 0x30)
    new = (old & 0xFFF0) | (gen & 0xF)
    wr16(dev, 0x30, new)
    rb = rd16(dev, 0x30)
    log(f"{dev}: LNKCTL2 {old:04x} -> {rb:04x}, TLS={rb & 0xF}")

def retrain(dev):
    ctl = rd16(dev, 0x10)
    wr16(dev, 0x10, ctl & ~0x20)
    time.sleep(0.300)
    ctl = rd16(dev, 0x10)
    wr16(dev, 0x10, ctl | 0x20)
    time.sleep(2.2)

def link(dev):
    p = Path(f"/sys/bus/pci/devices/{dev}")
    try:
        return ((p / "current_link_speed").read_text().strip(),
                (p / "current_link_width").read_text().strip())
    except Exception:
        return ("missing", "missing")

def driver_name(dev):
    p = Path(f"/sys/bus/pci/devices/{dev}/driver")
    try:
        return p.resolve().name
    except Exception:
        return None

def pl0():
    resource = f"/sys/bus/pci/devices/{GPU}/resource0"
    fd = os.open(resource, os.O_RDWR | os.O_SYNC)
    mm = mmap.mmap(fd, 0x90000)
    try:
        def rd32(off):
            return struct.unpack_from("<I", mm, off)[0]
        def wr32(off, val):
            struct.pack_into("<I", mm, off, val)
            mm.flush()

        boot0 = rd32(0)
        log(f"BOOT0 = 0x{boot0:08X}")
        if (boot0 & 0xFF000000) != 0x16000000:
            raise RuntimeError("BOOT0 is not TU10x/TU106")

        for off, val, name in REGS:
            before = rd32(off)
            wr32(off, val)
            after = rd32(off)
            log(f"{name:14s} {before:08X} -> {after:08X}")
    finally:
        mm.close()
        os.close(fd)

def unbind_nvidia():
    if driver_name(GPU) == "nvidia":
        Path("/sys/bus/pci/drivers/nvidia/unbind").write_text(GPU)
        time.sleep(1)

def root_link_disable_enable():
    ctl = rd16(ROOT, 0x10)
    log(f"ROOT LNKCTL before=0x{ctl:04X}")
    wr16(ROOT, 0x10, ctl | 0x10)
    time.sleep(0.5)
    set_tls(ROOT, 2)
    ctl = rd16(ROOT, 0x10)
    wr16(ROOT, 0x10, ctl & ~0x10)
    time.sleep(3)

def remove_rescan():
    remove = Path(f"/sys/bus/pci/devices/{GPU}/remove")
    if remove.exists():
        remove.write_text("1")
        time.sleep(2)
    Path("/sys/bus/pci/rescan").write_text("1")
    subprocess.run(["udevadm", "settle"], check=False)
    time.sleep(5)

def nvidia_smi():
    return subprocess.run(
        ["nvidia-smi",
         "--query-gpu=pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max",
         "--format=csv,noheader"],
        text=True, capture_output=True
    )

def main():
    if os.geteuid() != 0:
        log("Run as root")
        return 1

    if not Path(f"/sys/bus/pci/devices/{GPU}").exists():
        log(f"{GPU} is not present")
        return 2

    log(f"GPU={GPU} ROOT={ROOT}")
    log(f"initial GPU link={link(GPU)} driver={driver_name(GPU)}")

    log("\n=== 1. UNBIND NVIDIA ===")
    unbind_nvidia()
    log(f"driver={driver_name(GPU)}")

    log("\n=== 2. FIRST PL0 ===")
    set_tls(ROOT, 2)
    pl0()

    log("\n=== 3. ROOT LINK DISABLE/ENABLE ===")
    root_link_disable_enable()
    log(f"after LD: GPU={link(GPU)} ROOT={link(ROOT)}")

    log("\n=== 4. REMOVE/RESCAN ===")
    remove_rescan()
    if not Path(f"/sys/bus/pci/devices/{GPU}").exists():
        log("GPU did not return after rescan")
        return 3
    log(f"after rescan GPU={link(GPU)} driver={driver_name(GPU)}")

    log("\n=== 5. POST-PNP PL0 ===")
    pl0()
    set_tls(GPU, 2)
    set_tls(ROOT, 2)

    log("\n=== 6. ROOT RETRAIN ===")
    retrain(ROOT)
    log(f"final GPU={link(GPU)} ROOT={link(ROOT)}")

    smi = nvidia_smi()
    if smi.stdout.strip():
        log("nvidia-smi: " + smi.stdout.strip())
    if smi.stderr.strip():
        log("nvidia-smi stderr: " + smi.stderr.strip())

    speed, width = link(GPU)
    ok = speed.startswith("5.0") and width == "16" and smi.returncode == 0
    if ok:
        log("\n*** GEN2 X16 ACHIEVED ***")
        return 0

    log("\nGen2 x16 not confirmed; do not keep retrying blindly.")
    return 4

if __name__ == "__main__":
    raise SystemExit(main())
