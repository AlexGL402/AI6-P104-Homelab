# CMP 50HX vLLM power profile — 2026-09-22

## Host / software

- GPU: NVIDIA CMP 50HX, 10 GB
- Board: ASRock H110M-DVS R3.0
- CPU: Intel Core i3-6100T, 2C/4T
- RAM: 15.32 GiB
- OS: Ubuntu 24.04.5 LTS
- Kernel: 6.8.0-139-generic
- NVIDIA driver/KMD: 610.43.03
- CUDA UMD/runtime: 13.3 / 13.0
- Torch: 2.13.0+cu130
- PCIe: Gen1 x4 current, Gen2 x16 max
- Forge CMP unlock: restored known-good 610.43.03 patched modules

## Benchmark shape

- Model: Qwen3-4B-AWQ
- Quantization: AWQ
- Context: 4096
- Mode: compiled/graphs
- Concurrency: 12
- Prompt profile: short-v1
- Prompt tokens total: 603
- Output tokens/request: 512
- Output tokens total: 6144
- Temperature: 0.0
- Thinking: OFF

## 150 W efficiency reference

Two confirmed runs after restoring the Forge CMP stack:

| PL | Aggregate | Avg power | Efficiency | Temp |
|---:|---:|---:|---:|---:|
| 150 W | 632.58 tok/s | 136.4 W | 4.636 tok/s/W | 47 C |
| 150 W | 615.10 tok/s | 134.8 W | 4.561 tok/s/W | 47 C |

This is the current efficiency-oriented profile.

## 225 W max-throughput reference

Run ID: `1790069826-12-512`

- Wall time: 8.801 s
- Aggregate output throughput: **698.09 tok/s**
- Per-request output throughput: 59.31-59.57 tok/s
- Average decode speed/request: 59.80 tok/s
- TTFT avg/max: 0.065 / 0.067 s
- Average / peak GPU load: 84.36 / 100.0 %
- Average / peak GPU power: 179.31 / 206.75 W
- Power limit: 225 W
- Peak VRAM: 8.95 GiB
- Peak temperature: 48 C
- Peak fan: 35 %
- Efficiency: **3.893 tok/s/W**

## Practical conclusion

For this CMP 50HX and this exact workload:

- **150 W** is the better efficiency point: ~615-633 tok/s at ~135-136 W average and ~4.56-4.64 tok/s/W.
- **225 W** is the max-throughput point: 698.09 tok/s, but average power rises to 179.31 W and efficiency falls to 3.893 tok/s/W.
- Raising PL from 150 W to 225 W adds roughly 10-13% throughput while increasing average power by roughly 31-33%.

The restored `610.43.03 + forge-cmp` stack is required for expected CMP compute throughput. A previous stock 610.57 test on the same class of benchmark produced only about 145 tok/s.
