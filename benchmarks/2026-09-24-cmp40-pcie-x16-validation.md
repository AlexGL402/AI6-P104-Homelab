# CMP 40HX — PCIe x16 mod validation (2026-09-24)

This note records the post-mod PCIe x16 validation for the NVIDIA CMP 40HX on AI6.

## Platform

- GPU under test: NVIDIA CMP 40HX 8 GB
- PCI bus: 00000000:01:00.0
- Host board: ASRock H110M-DVS R3.0
- CPU: Intel Core i3-6100T
- OS: Ubuntu 24.04.5 LTS
- Kernel: 6.8.0-139-generic
- NVIDIA driver: 610.43.03
- PyTorch: 2.13.0+cu130
- CUDA runtime: 13.0
- Link reported by nvidia-smi under load: Gen1 x16 current, Gen2 x16 max
- lspci: LnkSta Speed 2.5GT/s, Width x16

## Important CUDA device-ordering note

With the mixed CMP40 + CMP50 configuration, CUDA device numbering did not initially match the nvidia-smi order. A run with only CUDA_VISIBLE_DEVICES=0 actually executed on the CMP50.

For deterministic mapping to PCI/nvidia-smi order, tests were run with:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
```

The incorrect CMP50 run is excluded from the CMP40 x16 results below.

## Direct PCIe bandwidth test

Pinned host memory and non-blocking PyTorch copies were used after warm-up. Results were identical across 256 MiB, 512 MiB and 1024 MiB transfer sizes.

| Transfer size | H2D | D2H |
|---:|---:|---:|
| 256 MiB | 3.12 GB/s | 3.13 GB/s |
| 512 MiB | 3.12 GB/s | 3.13 GB/s |
| 1024 MiB | 3.12 GB/s | 3.13 GB/s |

This confirms that the x16 modification is carrying real traffic across a Gen1 x16 link rather than only reporting x16 width.

## vLLM control workload

Common settings:

- Model: Qwen3-4B-AWQ
- Quantization: AWQ
- Context: 4096
- Mode: compiled / CUDA graphs
- Concurrency: 12
- Prompt profile: short
- Prompt tokens total: 603
- Output: 512 tokens/request
- Temperature: 0
- Thinking: OFF
- Power limit: 150 W
- PCIe: Gen1 x16

### x16 runs

| Run ID | Aggregate output | Wall | TTFT avg/max | Avg / peak power | Peak VRAM | Peak temp |
|---|---:|---:|---:|---:|---:|---:|
| 1790247762-12-512 | 480.53 tok/s | 12.786 s | 0.264 / 0.279 s | 134.53 / 150.37 W | 7.14 GiB | 50 C |
| 1790247859-12-512 | 504.24 tok/s | 12.185 s | 0.086 / 0.088 s | 139.71 / 152.45 W | 7.14 GiB | 50 C |

Mean aggregate throughput: **492.39 tok/s**.

## Width comparison

Existing controlled results for the same Qwen3-4B-AWQ short / concurrency 12 / output 512 / PL150 workload:

| PCIe | Aggregate runs | Mean |
|---|---|---:|
| Gen1 x4 | 484.67, 488.68 tok/s | 486.68 tok/s |
| Gen1 x8 | 499.65, 490.34 tok/s | 494.99 tok/s |
| Gen1 x16 | 480.53, 504.24 tok/s | 492.39 tok/s |

Observed mean differences:

- x4 -> x8: about +1.7%
- x8 -> x16: about -0.5%
- x4 -> x16: about +1.2%

For this fully GPU-resident 4B AWQ decode workload, the differences are within ordinary run-to-run variation. Increasing PCIe width beyond x4 does not provide a meaningful vLLM throughput gain.

## Interpretation

The hardware modification is successful: the CMP40 trains at Gen1 x16 and sustains about 3.12 GB/s H2D and 3.13 GB/s D2H in a direct pinned-memory transfer test.

The lack of corresponding vLLM scaling is workload-specific. Once Qwen3-4B-AWQ is resident in VRAM, steady-state decode does not move enough data over PCIe for x4 -> x8 -> x16 bandwidth to materially change aggregate output throughput.

Direct bandwidth measurements for x4 and x8 are still pending. Those should be added later to quantify physical bus scaling independently of the LLM workload.
