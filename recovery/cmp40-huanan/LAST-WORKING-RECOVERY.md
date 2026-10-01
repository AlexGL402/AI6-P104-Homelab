# CMP40HX / Huanan — last working recovery procedure (2026-10-02)

This note records the exact recovery path used immediately before the CMP40 Soup benchmark returned to baseline.

## Known working state

- Host: Huanan X79
- GPU: NVIDIA CMP 40HX 8 GB (TU106, PCI ID 10de:1f0b)
- GPU PCI address in this host: 03:00.0
- Ubuntu 24.04, kernel 6.8.0-139-generic
- NVIDIA 610.43.03 Open/MIT-GPL kernel modules
- CUDA UMD 13.3
- ForgeMiner 1.8.0
- BIOS: Above 4G Decoding = Disabled
- Normal Ubuntu boot; do NOT boot the CMP40 Unlock EFI / Boot0000
- PCIe Gen2 is not required for recovery.

## What fixed the GPU/AI state

Several forge-cmp module sets existed. The current modules and the older "known-good" archive were different. Hot-unloading/reloading NVIDIA modules left the CMP40 in a bad WPR2/GSP state:

    unexpected WPR2 already up, cannot proceed with booting GSP
    GPU is likely in a bad state and may need to be reset
    RmInitAdapter failed

PCI remove/rescan did not clear that state. A reboot was required.

The module set saved as:

    forge-cmp-610.43.03-6.8.0-139-generic-huanan-working.tar.gz

was installed into:

    /lib/modules/6.8.0-139-generic/updates/forge-cmp/

After a clean reboot, NVIDIA autoload remained intentionally blocked. The working modules were loaded once manually:

    sudo insmod /lib/modules/$(uname -r)/updates/forge-cmp/nvidia.ko
    sudo insmod /lib/modules/$(uname -r)/updates/forge-cmp/nvidia-uvm.ko
    sleep 3
    nvidia-smi

CMP40 then initialized normally.

IMPORTANT: do not repeatedly rmmod/insmod this driver on the initialized CMP40. If WPR2/GSP becomes stuck, reboot and load the modules once.

## Recovery archive

Repository file:

    recovery/cmp40-huanan/CMP40-HUANAN-WORKING-2026-10-01.tar.gz

SHA256:

    b1207faac9dd1ea9da71f452f51584448b627b92c8101297e55e743dc2fa9a19

Verify:

    sha256sum CMP40-HUANAN-WORKING-2026-10-01.tar.gz

The archive contains the forge-cmp modules, captured configuration/state, SHA256 data, and README-RU.md.

## After reboot

Load the driver only once:

    sudo insmod /lib/modules/$(uname -r)/updates/forge-cmp/nvidia.ko
    sudo insmod /lib/modules/$(uname -r)/updates/forge-cmp/nvidia-uvm.ko
    nvidia-smi
    sudo systemctl start ai6-monitor.service

For initial validation leave CMP Tune at stock. Observed power limit is 184 W.

## Soup validation after recovery

The Soup NF4 streamed LoRA benchmark was rerun after recovery:

- Model: Llama 3.1 8B
- quant: NF4
- seq x batch: 512 x 1
- buffers: 2
- LoRA: r=16, alpha=32, q_proj,v_proj
- 10 warm-up + 50 measured steps
- Current throughput: 173.90 tok/s
- Current median step: 2945 ms
- Saved baseline: 174.67 tok/s / 2931.6 ms
- Difference in throughput: about -0.44%
- GPU utilization: 100%
- Median SM clock: 1875 MHz
- Peak VRAM: 2.40 GB
- Max temperature: 76 C
- Effective throughput: 7.65 TFLOPS

Result: PASS. Soup/CMP40 AI performance returned to the saved baseline.

## Forge/PearlHash caveat

A normally initialized GPU is not by itself proof that PearlHash full-speed unlock is active.

Historical CMP40 PearlHash result:
- about 51.0–51.7 TH/s at about 183 W

The most recent Miner UI observation before the Soup validation was about 1.04 TH/s. Therefore the AI/Soup recovery is confirmed, while PearlHash full-speed recovery still needs separate work.

For TU106, `forge --cmp-verify` reports that live unlock state is not observable and says to use miner TUI hashrate.

Do not change BIOS, enable Above 4G, force Gen2, or boot the EFI unlock merely to address this mining discrepancy. Preserve this working AI state first.
