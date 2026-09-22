# Qwen3-14B-AWQ on 2× CMP 50HX — vLLM benchmark

Date: 2026-09-22

## Test platform

- GPUs: 2× NVIDIA CMP 50HX 10GB
- GPU0 PCIe: Gen2 x4
- GPU1 PCIe: Gen2 x1
- Power limit: 150 W per GPU (300 W total)
- Model: Qwen3-14B-AWQ
- Quantization: AWQ
- vLLM mode: compiled/graphs
- Context: 8192
- Thinking: OFF
- Driver: 610.43.03
- Torch: 2.13.0+cu130
- CUDA runtime: 13.0
- CPU: Intel Core i3-6100T
- RAM: 15.32 GiB
- OS: Ubuntu 24.04.5 LTS
- Kernel: 6.8.0-139-generic

## Results

| Concurrency | Output/request | Prompt tokens total | TTFT avg/max | Wall time | Aggregate output | Per-request output | Avg decode/request | Avg power | Peak power | Peak VRAM | Peak temp |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 50 | 1.441 / 1.441 s | 19.783 s | 25.88 tok/s | 26.49 tok/s | 28.62 tok/s | 239.97 W | 256.66 W | 17.46 GiB | 49 C |
| 4 | 512 | 200 | 0.551 / 0.656 s | 17.484 s | 117.14 tok/s | 29.79–29.90 tok/s | 30.81 tok/s | 266.61 W | 274.33 W | 17.46 GiB | 51 C |
| 1 | 1024 | 50 | 0.098 / 0.098 s | 19.214 s | 53.29 tok/s | 53.41 tok/s | 53.68 tok/s | 291.66 W | 303.43 W | 17.46 GiB | 51 C |
| 4 | 1024 | 200 | 0.129 / 0.157 s | 55.914 s | 73.26 tok/s | 18.34–18.37 tok/s | 18.39 tok/s | 229.52 W | 235.18 W | 17.46 GiB | 50 C |

## Practical notes

- Best interactive result in this set: concurrency 1, 1024 output tokens, ~53.7 tok/s decode and 0.098 s TTFT.
- Concurrency 4 with 512 output tokens reaches 117.14 tok/s aggregate while keeping per-request decode near 30 tok/s.
- Both cards stayed within 150 W PL and around 50–51 C peak temperature.
- This setup is suitable for an interactive coding-agent workload; C1 is the primary target, with C4 useful for parallel work.
