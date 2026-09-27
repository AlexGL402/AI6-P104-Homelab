# AI6 benchmark insights

This file records the current operational conclusions derived from the saved benchmark series. Raw measurements remain in the dated benchmark files and per-run reports.

## Current conclusions — 2026-09-27

### J1900 + P104 mini-node

The J1900 mini platform is viable as an economical always-on local-AI node with one P104-100 8 GB.

Current working software baseline:

- NVIDIA proprietary driver 555.42.06
- Ollama 0.24.0
- CUDA backend detected as CUDA 12.5
- Context 4096
- Current PCIe link: Gen1 x1
- BIOS Max TOLUD: 2 GB

Recent Ollama 0.30.6 and 0.33.3 reject this Pascal / compute 6.1 GPU on driver 555 and request NVIDIA driver 570 or newer. Ollama 0.24.0 detects the P104 normally through the CUDA 12 backend.

Saved single-GPU results:

| Model | Prompt tok/s | Generation tok/s | Total time | Approx. VRAM |
|---|---:|---:|---:|---:|
| Qwen3 4B | 475.15 | 37.31 | 15.573 s | ~3.08 GiB |
| Qwen3 8B | 294.21 | 26.92 | 20.957 s | ~5.3-5.45 GiB |

Qwen3 8B is the current preferred primary mini-node model: it remains fully GPU-resident and still delivers ~26.9 tok/s decode. Qwen3 4B is the faster lightweight fallback.

Because the current PCIe path is Gen1 x1, keep the primary model resident for long periods rather than repeatedly reloading it. A 24-hour Ollama keep-alive is recommended.

See `benchmarks/2026-09-27-j1900-p104-mini.md` for the full BIOS, PCIe, driver, Ollama compatibility and benchmark notes.

### CMP 40HX — vLLM sweet spot

Current practical AI power target:

**130 W**

Workload used for the main decision:

- Qwen3-4B-AWQ
- AWQ
- vLLM compiled / CUDA graphs
- Context 4096
- Medium prompt profile
- Concurrency 12
- Thinking OFF
- Temperature 0
- PCIe Gen1 x4

The 130 W setting repeatedly delivered nearly the same throughput as 150 W and 184 W while improving tok/s/W and reducing temperature.

For 2048 output tokens/request:

| PL | Aggregate | Avg power | Efficiency | Peak temp |
|---:|---:|---:|---:|---:|
| 130 W | 182.59 tok/s | 127.43 W | 1.433 tok/s/W | 67 C |
| 150 W | 183.88 tok/s | 142.66 W | 1.289 tok/s/W | 70 C |
| 184 W | 185.09 tok/s | 145.59 W | 1.271 tok/s/W | 73 C |

The 184 W result is only about 1.4% faster than 130 W on this workload.

### Practical usage guidance

- Interactive coding: prefer low TTFT and high per-request decode speed rather than maximum aggregate throughput.
- Batch/API serving: higher concurrency can strongly improve aggregate throughput, especially on short and medium prompts.
- Long-context / long-output workloads: throughput and tok/s/W decline as sequence length grows; additional PL above 130 W gives little benefit in the current CMP40 tests.
- AI idle: use the Miner tab to run PearlHash when inference is not using the GPU.
- Do not run vLLM and the miner on the same GPU at the same time.

## PRE-MOD baseline

Status: **captured**

The following points are the primary controls to repeat after the planned capacitor modification:

1. Qwen3-4B-AWQ / Medium / conc 12 / 512 output / 130 W
2. Qwen3-4B-AWQ / Medium / conc 12 / 1024 output / 130 W
3. Qwen3-4B-AWQ / Medium / conc 12 / 2048 output / 130 W
4. Qwen3-4B-AWQ / Medium / conc 12 / 2048 output / 150 W
5. PearlHash mining / 130 W

A post-mod change should be treated as meaningful only when it is repeatable and larger than normal run-to-run variation.

## Current direct CMP40 vs P104 reference

The cleanest existing direct comparison uses Qwen3.5 9B under Ollama on Gen1 x4:

| GPU | Prompt tok/s | Generation tok/s |
|---|---:|---:|
| CMP 40HX 8 GB | 205.11 | 45.42 |
| P104-100 8 GB | 217.63 | 28.17 |

On that system-level comparison, CMP40 generation is about 1.61x the saved P104 result, while the P104 prompt-processing result is slightly higher.

Do not directly compare the Qwen3-4B-AWQ vLLM throughput numbers with the P104 Ollama numbers as if they were the same benchmark. The models, kernels, runtimes and batching behavior differ.

## Confidence convention

- **High** — conclusion supported by 3 or more comparable runs.
- **Medium** — 2 comparable runs.
- **Low** — single run or incomplete comparison.

The monitor Insights tab derives its current sweet spot from saved comparable runs and displays a confidence level.

## Future comparison

When CMP50 cards arrive:

1. Repeat the same prompt profiles and output lengths.
2. Keep context, temperature, thinking mode and runtime configuration identical.
3. Preserve prompt profile IDs and SHA256 hashes.
4. Compare both absolute throughput and tok/s/W.
5. Record PCIe current/max link, driver, CUDA, CPU/RAM platform and power telemetry.
6. Use the same PRE-MOD-style control points if any hardware modification is performed later.

This file should contain conclusions and decisions, not every raw benchmark row.
