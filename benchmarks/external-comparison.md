# External benchmark comparison

This document compares the AI6-P104-Homelab measurements with public Qwen3/Qwen3-Coder 30B-A3B results. These are **reference comparisons, not strict apples-to-apples tests**: quantization, context length, runtime, batching, kernels, and hardware differ.

## AI6 measured results

| Configuration | Model/runtime | Prompt tok/s | Generation tok/s | Notes |
|---|---|---:|---:|---|
| 3× P104-100 | Qwen3-Coder 30B, Ollama | 105.97 | 33.25 | 32K context |
| 4× P104-100 | Qwen3-Coder 30B, Ollama | 137.94 | 36.53 | Best single-request Ollama result |
| 6× P104-100 | Qwen3-Coder 30B, Ollama | 113.96 | 35.23 | More GPUs did not improve decode |
| 3× P104-100 | llama.cpp tensor, CUDA 12.8 + NCCL 2.26.2 | 62.2 | 31.7 | Stable |
| 4× P104-100 | llama.cpp tensor | 57.8 | 29.1 | Stable, slower because of PCIe x1 communication |
| 6× P104-100 | llama.cpp tensor | 51.6 | 27.2 | Negative scaling |
| 2× P104-100 | llama.cpp tensor | — | — | OOM |
| 3+3 P104 workers | two simultaneous llama.cpp servers | — | **64.08 aggregate** | 32.22 + 31.86 tok/s, two requests in parallel |

## Public reference results

### NVIDIA DGX Spark — official llama.cpp bench

For `Qwen3-Coder-30B-A3B-Instruct-Q8_0`, the llama.cpp DGX Spark benchmark reports approximately:

- `tg32`: 61.06 tok/s at shallow context
- `tg32 @ d32768`: 30.21 tok/s at 32K depth

Source: https://github.com/ggml-org/llama.cpp/blob/master/benches/dgx-spark/dgx-spark.md

The interesting point for AI6 is that one 3×P104 worker reaches about **31.7–32.2 tok/s** in our real 32K coding setup, which is in the same order of magnitude as the DGX Spark deep-context result. This does **not** mean the P104 system equals DGX Spark overall: prompt processing, memory capacity, efficiency, batching, precision, and software paths are very different.

### DGX Spark community benchmark

A recent DGX Spark field-notes benchmark reports Qwen3-30B-A3B around 85 tok/s in Ollama and 77 tok/s in vLLM for single-stream testing, with much higher aggregate throughput under batching.

Source: https://github.com/sergioamsilva/dgx-spark-field-notes

### High-end datacenter/workstation GPUs

GPU Battle reports Qwen3-Coder 30B-A3B generation rates such as:

| GPU | Reported generation tok/s |
|---|---:|
| RTX PRO 6000 Blackwell | 317.68 |
| H200 | 297.9 |
| H100 80GB | 296.22 |
| L40S | 219.47 |
| A100 80GB SXM4 | 182.26 |
| A100 40GB SXM4 | 176.74 |
| A10G | 145 |
| L4 | 98.84 |

Source: https://gpubattle.com/guides/qwen3-coder-30b-a3b-gpu-benchmarks

These numbers use a different benchmark harness and should be treated as scale references only.

## Interpretation

The six-P104 system is not competitive with modern H100/H200/Blackwell hardware in absolute single-stream speed, power efficiency, or prompt processing. Its strength is **cost efficiency and parallelism**.

The key architectural result is that the H81A mining platform's PCIe x1 topology makes large tensor-parallel groups inefficient. Splitting the six identical GPUs into two independent 3-GPU workers avoids most cross-GPU communication and gives about **64.08 tok/s aggregate across two concurrent requests**, more than 2.3× the 6-GPU tensor result of 27.2 tok/s.

For this platform, throughput-oriented multi-agent use is therefore more attractive than trying to maximize the speed of one response with all six GPUs.

## Caveats

- AI6 uses P104-100 Pascal cards with 8 GB each and no P2P path between cards.
- All GPU links are PCIe x1 at 2.5 GT/s on this board.
- Quantization and context differ across external references.
- Aggregate throughput (two simultaneous requests) must not be compared directly to single-request tok/s.
- Agentic coding adds multiple model turns for tool calls, so end-to-end task latency is higher than raw decode speed alone suggests.
