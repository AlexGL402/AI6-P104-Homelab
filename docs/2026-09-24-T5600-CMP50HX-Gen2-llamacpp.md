# T5600 / CMP 50HX / Gen2 x4 / llama.cpp notes

Date: 2026-09-24

## Host

- Dell Precision T5600
- Ubuntu 24.04.4 LTS
- Kernel: `6.8.0-142-generic`
- NVIDIA driver/KMD: `610.43.03`
- CUDA UMD: `13.3`
- GPU: NVIDIA CMP 50HX 10 GB
- GPU bus: `0000:04:00.0`

## CMP50HX unlock / PCIe

Built and installed `xrip/cmp50hx-unlock` for the current kernel.

Build result:

```text
PASS_CMP50HX_ALL_BUILD
artifacts/610.43.03-6.8.0-142-generic
```

Install result:

```text
PASS_CMP_INITRAMFS
PASS_CMP50HX_INSTALL
```

After reboot, `cmp50hx-gen2.service` retrained the link successfully:

```text
cmp50hx-gen2: PASS, link at 5.0 GT/s PCIe
```

Verified PCIe state:

```text
index, name, pcie.link.gen.current, pcie.link.width.current, pcie.link.gen.max, pcie.link.width.max
0, NVIDIA CMP 50HX, 2, 4, 2, 16
```

So this T5600/CMP50HX path is working at **PCIe Gen2 x4**.

## llama.cpp

Build:

- llama.cpp build: `0.5.0-dev`
- commit: `84e76d8a2`
- CUDA architecture: SM75
- model: `qwen2.5-coder-14b-instruct-q4_k_m.gguf`
- model size: ~8.4 GB
- context: 4096
- `ngl=999`
- split: layer
- single GPU

## Gen1 x4 -> Gen2 x4 result

Before the Gen2 unlock, the same model/test on this host measured approximately:

```text
PCIe Gen1 x4
Prompt:     ~119 tok/s
Generation: ~17.19 tok/s
```

After the Gen2 unlock at 150 W:

```text
PCIe Gen2 x4
Prompt:     ~277.4 tok/s
Generation: ~41.91 tok/s
```

Observed uplift was about 2.3x for prompt processing and 2.4x for generation in this test.

## Power-limit sweep

Same Qwen2.5-Coder 14B Q4_K_M workload, PCIe Gen2 x4.

| Power limit | Generation throughput | Notes |
|---:|---:|---|
| 130 W | ~32.77 tok/s | 5 runs completed |
| 140 W | ~37.71 tok/s | 5 runs completed |
| 145 W | ~40.50 tok/s | initial 5-run average |
| 150 W | ~41.91 tok/s | fastest measured short run |

At 145 W, a longer 20-run stability test completed without a host reboot.

20-run generation throughput:

```text
Average: 39.94 tok/s
Min:     39.46 tok/s
Max:     40.69 tok/s
```

During the long 145 W run the monitor showed approximately:

```text
GPU load: 97-98%
Power:    ~142.5-143.5 W / 145 W
Temp:     ~52-53 C
VRAM:     8.98 GiB / 10.00 GiB
Fan:      ~50%
```

A higher-power test around 180 W did not improve generation throughput in the observed short run, and the host later rebooted abruptly under load. The previous-boot journal ended without a clean shutdown sequence or a useful GPU/kernel error immediately before the interruption, so the working hypothesis is a PSU/power-path limitation rather than a confirmed software crash.

For now, **145 W is the T5600 working profile**: close to the 150 W performance while preserving more power margin.

## Boot-persistent power limit

Installed:

```text
/etc/systemd/system/ai6-gpu-power.service
```

Current target:

```text
nvidia-smi -i 0 -pl 145
```

The service is enabled so the limit is restored after reboot.

## Boot-persistent services verified

After reboot:

```text
ai6-web-terminal.service   enabled / active
ai6-monitor.service        enabled / active
cmp50hx-gen2.service       enabled; oneshot retrain
```

Web terminal listens on TCP 8091 and the monitor on TCP 8090.

## Other GPU in T5600

A legacy Quadro 2000 is present at `03:00.0`:

```text
NVIDIA GF106GL [Quadro 2000] [10de:0dd8]
```

The NVIDIA open 610 module reports that this older GPU is unsupported because it lacks the required GSP. This does not prevent the CMP 50HX at `04:00.0` from loading and operating at Gen2 x4.
