# CMP 40HX — vLLM interim benchmark set (2026-09-21)

This file captures the current intermediate vLLM results before the full CMP40/CMP50/P104 comparison matrix is complete.

## Test platform

- GPU: 1× NVIDIA CMP 40HX 8 GB
- Host: AI6 / Ubuntu 24.04
- PCIe: Gen1 x4 current; device max reported as Gen2 x16
- NVIDIA driver: 610.43.03
- PyTorch: 2.13.0+cu130
- Torch CUDA runtime: 13.0
- vLLM model: Qwen3-4B-AWQ
- Quantization: AWQ
- Context: 4096
- Mode: compiled / CUDA graphs
- Thinking: OFF for the main throughput matrix below
- Temperature: 0
- Output: 512 tokens/request
- Power limit: 150 W unless otherwise noted
- Hardware state: before any capacitor / board-level modification

## Prompt-size scaling

The benchmark UI uses deterministic prompt profiles. Prompt token totals below are totals across all concurrent requests, so they grow with concurrency.

### 1 concurrent request

| Profile | Prompt tokens | TTFT avg/max | Wall | Aggregate output | Per-request | Avg / peak power | Peak temp |
|---|---:|---:|---:|---:|---:|---:|---:|
| Short | 50 | 0.076 / 0.076 s | 5.800 s | 88.27 tok/s | 92.04 tok/s | 132.5 / 150.9 W | 47 C |
| Medium | 339 | 0.164 / 0.164 s | 6.522 s | 78.50 tok/s | 83.48 tok/s | 127.5 / 151.2 W | 50 C |
| Long | 1311 | 0.873 / 0.873 s | 8.950 s | 57.21 tok/s | 60.31 tok/s | 131.6 / 151.2 W | 52 C |

### 4 concurrent requests

| Profile | Prompt tokens | TTFT avg/max | Wall | Aggregate output | Per-request | Avg / peak power | Peak temp |
|---|---:|---:|---:|---:|---:|---:|---:|
| Short | 200 | 0.110 / 0.120 s | 7.701 s | 265.95 tok/s | 71.15–71.31 tok/s | 126.5 / 150.5 W | 52 C |
| Medium | 1356 | 0.333 / 0.421 s | 9.974 s | 205.32 tok/s | 55.41–55.51 tok/s | 134.3 / 151.6 W | 55 C |
| Long | 5244 | 1.524 / 2.533 s | 17.765 s | 115.28 tok/s | 29.28–29.35 tok/s | 134.5 / 141.8 W | 58 C |

### 12 concurrent requests

| Profile | Prompt tokens | TTFT avg/max | Wall | Aggregate output | Per-request | Avg / peak power | Peak temp | Efficiency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 603 | 0.248 / 0.266 s | 12.677 s | 484.67 tok/s | 42.38–42.52 tok/s | 133.6 / 151.2 W | 50 C | 3.629 tok/s/W |
| Medium | 4071 | 1.081 / 1.451 s | 18.822 s | 326.43 tok/s | 27.57–27.64 tok/s | 140.6 / 151.7 W | 55 C | 2.322 tok/s/W |
| Long | 15735 | 5.817 / 9.966 s | 45.525 s | 134.96 tok/s | 11.32–11.38 tok/s | 130.7 / 141.1 W | 63 C | 1.033 tok/s/W |

## Interim interpretation

- Short prompts scale strongly with batching: 88.27 tok/s at concurrency 1 to 484.67 tok/s at concurrency 12.
- Medium prompts also scale well, but TTFT rises to roughly 1.08 s average at concurrency 12.
- Long-context style prompts scale much less efficiently: 57.21 tok/s at concurrency 1 to 134.96 tok/s at concurrency 12.
- Long prompts are dominated by prefill / attention / context handling rather than raw decode alone. At concurrency 12, average TTFT is 5.817 s and max TTFT is 9.966 s.
- Raising concurrency therefore helps aggregate throughput much more for short/medium workloads than for long-context workloads.

## Power-limit spot checks already captured

Earlier Qwen3-4B-AWQ vLLM tests on the same CMP40 showed that raising the power limit from 150 W to 184 W produced almost no aggregate-throughput benefit for an 8-request batch:

| PL | Aggregate | Observed power |
|---:|---:|---:|
| 150 W | 455.65 tok/s | ~151.24 W |
| 184 W | 458.33 tok/s | ~158.23 W |

The difference was only about +0.6% throughput.

A later 4-request spot comparison also showed very similar throughput across 130 / 150 / 184 W, with 130 W giving the best efficiency in that small sample. A full controlled PL matrix is still pending.

## Thinking-mode note

A pair of single-request 512-token tests was captured with Thinking OFF vs ON. Throughput was nearly unchanged in that short workload, but prompt token counts differed because the chat template changes between thinking modes. The benchmark framework now records prompt profile IDs and SHA256 hashes so future A/B tests can be checked for input identity.

For the main hardware-throughput series, Thinking OFF remains the baseline. Thinking ON should be treated as a separate workload class.

## P104 reference — what is directly comparable today

The repository already contains a one-GPU P104-100 reference using Qwen3.5 9B in Ollama:

| GPU | Runtime / model | PCIe | Prompt tok/s | Generation tok/s |
|---|---|---|---:|---:|
| CMP 40HX | Ollama / Qwen3.5 9B | Gen1 x4 | 205.11 | 45.42 |
| P104-100 | Ollama / Qwen3.5 9B | Gen1 x4 | 217.63 | 28.17 |

On that system-level A/B, CMP40 generation was about 1.61× the saved P104 x4 result, while P104 prompt processing was about 5.8% faster.

This is the current direct CMP40-vs-P104 comparison because it uses the same model family/runtime class and the same x4 link width.

## Important comparison caveat

The Qwen3-4B-AWQ vLLM numbers in this file must **not** be directly compared numerically with the P104 Ollama numbers above. They use different models, quantization formats, runtimes, kernels and batching behavior.

Modern vLLM does not natively support the P104-100 Pascal CC 6.1 path used here, so a strict vLLM-vs-vLLM P104 comparison is not available. For future GPU-to-GPU comparison, use a common model/runtime supported by both cards (Ollama or llama.cpp), while treating vLLM as an additional capability/performance result for Turing-class CMP cards.

## Next controlled matrix

Recommended next checkpoints:

1. Qwen3-4B-AWQ at concurrency 12, Short / Medium / Long, PL 130 / 150 / 184 W.
2. Same profile matrix at concurrency 1 and 4 if needed for PL sensitivity.
3. Repeat the common-model Ollama or llama.cpp comparison when P104 is installed in the same host/PCIe configuration.
4. When CMP50 cards arrive, repeat the exact vLLM benchmark profiles using the saved prompt hashes and report format.
