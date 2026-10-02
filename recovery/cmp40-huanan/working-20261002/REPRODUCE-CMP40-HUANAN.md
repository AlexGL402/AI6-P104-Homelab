# Huanan CMP40 working recovery procedure — 2026-10-02

## Verified working configuration

Host: Huanan
GPU: NVIDIA CMP 40HX
PCI address: 0000:03:00.0

Kernel:
6.8.0-139-generic

DO NOT boot 6.8.0-142-generic until Forge modules have been
built and verified for that kernel.

Forge modules:
 /lib/modules/6.8.0-139-generic/updates/forge-cmp/nvidia.ko
 /lib/modules/6.8.0-139-generic/updates/forge-cmp/nvidia-uvm.ko

## Critical discovery

forge-cmp/nvidia.ko depends on Linux kernel module:

 ecc

If nvidia.ko is inserted before ecc at boot, loading fails with:

 crypto_ecdh_shared_secret
 ecc_is_pubkey_valid_full
 ecc_make_pub_key
 ecc_get_curve
 ecc_gen_privkey

and nvidia-smi cannot see the CMP40.

## REQUIRED boot order

1. Boot Linux 6.8.0-139-generic
2. systemd starts forge-cmp-driver.service
3. modprobe ecc
4. insmod forge-cmp/nvidia.ko
5. insmod forge-cmp/nvidia-uvm.ko
6. nvidia-smi sees CMP40
7. AI6 monitor and compute services may start

Exact sequence:

 modprobe ecc
 insmod /lib/modules/$(uname -r)/updates/forge-cmp/nvidia.ko
 insmod /lib/modules/$(uname -r)/updates/forge-cmp/nvidia-uvm.ko

## Verification

 uname -r
 systemctl status forge-cmp-driver.service --no-pager
 lsmod | grep -E 'ecc|nvidia'
 nvidia-smi -L

Expected:

 kernel = 6.8.0-139-generic
 forge-cmp-driver.service = active (exited)
 ecc loaded before nvidia
 NVIDIA CMP 40HX visible

## Important

Do not repeatedly rmmod/insmod the CMP driver during debugging.
A bad hot-reload can leave GPU/WPR state stuck. Prefer a clean reboot
back into the verified kernel and boot sequence.

Do not move to kernel 6.8.0-142 until matching Forge modules exist.

## Known working BIOS/PCI setup

Huanan X79 BIOS v2.49.
Above 4G Decoding: OFF.
CMP40 in the known-working PCIe slot.
Known working CMP40 link: PCIe Gen1 x16.

Do not change BIOS/PCIe settings while reproducing this recovery.
