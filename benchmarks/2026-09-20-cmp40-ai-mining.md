# CMP 40HX — mining + AI benchmark notes (2026-09-20)

## Hardware / software

- Host: AI6 disk / Ubuntu 24.04
- GPU: NVIDIA CMP 40HX, 8 GB
- PCI bus: 01:00.0
- NVIDIA driver / KMD: 610.43.03
- CUDA UMD reported by driver: 13.3
- Compute capability reported by llama.cpp: 7.5
- VRAM reported by NVIDIA-SMI: 8192 MiB
- VRAM available to llama.cpp: 7797 MiB
- Power limit during tests: 150 W unless otherwise noted
- ForgeMiner: 1.8.0
- llama.cpp build: 10884, commit 434ddbbc0

Important: these tests were performed **before any capacitor / board-level modification**. The card is still in its current stock hardware state.

## Exact GPU identification

Captured before any board-level modification:

```text
NVIDIA CMP 40HX
PCI bus: 00000000:01:00.0
PCI device ID: 0x1F0B10DE
PCI sub-device ID: 0x88041043
PCI ID: 10de:1f0b
Subsystem: ASUSTeK Computer Inc. [1043:8804]
GPU UUID: GPU-bf89a502-a3ca-35ca-ef44-776e1e596693
VBIOS: 90.06.67.00.04
VRAM: 8192 MiB
Default power limit: 184.00 W
Maximum power limit: 220.00 W
Chip: TU106, revision a1
Device class: 3D controller [0302]
Kernel driver: nvidia
```

PCIe capability / negotiated link:

```text
LnkCap: Speed 2.5 GT/s, Width x16
LnkSta: Speed 2.5 GT/s, Width x4 (downgraded)
Supported link speeds: 2.5 GT/s
```

So in the current machine the card is negotiating **PCIe Gen1 x4**, although the device capability reports x16 width.

## CMP unlock

ForgeMiner CMP hardware unlock is working with NVIDIA driver 610.43.03 and the open NVIDIA kernel modules.

Patched modules are installed under:

```text
/lib/modules/6.8.0-139-generic/updates/forge-cmp/
```

The unlock survives reboot.

## PearlHash mining

Pool:

```text
prl.kryptex.network:7048
```

Worker:

```text
ai6-cmp40
```

### Observed performance

| Power-limit setting | Observed power | Hashrate | Efficiency | Notes |
|---:|---:|---:|---:|---|
| Full / ~183 W | ~183 W | ~51.0–51.7 TH/s | ~278–283 GH/W | Highest raw hashrate observed |
| 165 W | ~160 W | ~47.36 TH/s | ~296 GH/W | Accepted share observed |
| 150 W | ~141–148 W | ~43.5–46.2 TH/s | ~308–312 GH/W | Good speed/efficiency compromise; accepted share observed |
| 140 W | ~133–135 W | ~41.3–42.4 TH/s | ~310–315 GH/W | Short test |
| 130 W | ~127 W | ~41.02 TH/s | ~324 GH/W | Best efficiency in short tests |

Accepted-share examples:

```text
[2026-09-20 09:56:59] GPU0 share accepted [122ms] [pearlhash] luck 36%
[2026-09-20 10:18:52] GPU0 share accepted [102ms] [pearlhash] luck 371%
```

At 150 W the miner showed about 44 TH/s average during a ~10 minute sample, with no pool drops, stale shares, or rejected shares.

## AI benchmark — Qwen3 14B Q4_K_M

Model:

```text
/home/ai6/models/qwen3-14b/Qwen3-14B-Q4_K_M.gguf
```

Model size / params:

```text
8.38 GiB / 14.77B
```

Benchmark command pattern:

```bash
./build/bin/llama-bench \
  -m /home/ai6/models/qwen3-14b/Qwen3-14B-Q4_K_M.gguf \
  -ngl <N> \
  -p 512 \
  -n 128 \
  -r 3
```

### Results

| NGL | pp512 | tg128 | Approx. VRAM observed | Result |
|---:|---:|---:|---:|---|
| 20 | 85.76 ± 0.18 tok/s | 4.17 ± 0.00 tok/s | ~4.86 GiB | Stable |
| 28 | 129.50 ± 0.48 tok/s | 6.23 ± 0.00 tok/s | ~5.82 GiB | Stable |
| 32 | 173.04 ± 0.75 tok/s | 8.09 ± 0.02 tok/s | ~6.37 GiB observed during run | Stable |
| 36 | — | — | ~7.08 GiB before failure | CUDA OOM / VMM allocation failure |

The `-ngl 36` run aborted with a CUDA allocation error in `ggml_cuda_pool_vmm::alloc`.

### Interpretation

The 14B Q4_K_M model is larger than the usable VRAM on this 8 GB card, so it cannot be fully offloaded. Generation speed therefore remains strongly limited by CPU/RAM/PCIe transfers. Increasing `ngl` from 20 to 32 produced substantial gains in both prompt processing and token generation.

For evaluating the GPU itself, a model that fits fully inside VRAM is more representative.

## Next AI test — Qwen3.5 9B

Ollama already has:

```text
qwen3.5:9b
```

Reported model size:

```text
6.6 GB
```

This is the preferred next one-GPU comparison because it should fit substantially better within the CMP 40HX 8 GB VRAM and there is an existing P104 reference in this repository.

Existing P104 reference from `2026-09-10-summary.md`:

| System | GPU | PCIe | Prompt tok/s | Generation tok/s |
|---|---:|---|---:|---:|
| Huanan / Windows | 1× P104-100 8 GB | Gen1 x4 | 217.63 | 28.17 |
| AI6 / Ubuntu | 1× P104-100 8 GB | Gen1 x1 | 144.09 | 23.74 |



## Qwen3.5 9B — Ollama single-GPU benchmark

Model:

```text
qwen3.5:9b
```

Ollama residency during the test:

```text
100% GPU
Context: 4096
```

Observed VRAM residency while loaded was about 6.37 GiB idle and about 7.05 GiB during generation. The card reached about 124 W during generation with the 150 W power limit.

Benchmark request:

```text
Write a Python function that recursively scans a directory, calculates SHA256 for every file, skips symlinks, handles permission errors, and returns a dict sorted by path. Include type hints and a short explanation.
```

Options:

```text
temperature = 0
num_ctx = 4096
num_predict = 512
```

Measured result:

```text
prompt tokens : 54
output tokens : 512
prompt speed  : 205.11 tok/s
generation    : 45.42 tok/s
total time    : 31.753 s
```

### Direct reference against saved P104-100 x4 result

| GPU | PCIe | Prompt tok/s | Generation tok/s |
|---|---|---:|---:|
| CMP 40HX 8 GB | Gen1 x4 | 205.11 | 45.42 |
| P104-100 8 GB | Gen1 x4 | 217.63 | 28.17 |

On this test, CMP 40HX prompt processing is about 5.8% lower than the saved P104 x4 result, while generation is about 61.2% higher (1.61x). The model is fully GPU-resident according to `ollama ps`, making this a much more representative GPU comparison than the partially-offloaded Qwen3 14B test.

This result was captured before any capacitor / board-level modification.


## Qwen3-4B-AWQ — vLLM 0.29.0

Environment:

```text
vLLM: 0.29.0
PyTorch: 2.13.0+cu130
Torch CUDA runtime: 13.0
GPU: NVIDIA CMP 40HX
Compute capability: 7.5
Attention backend: TRITON_ATTN
Quantization kernel: AutoAWQ Marlin
Server port: 8012
Max model length: 4096
Power limit: 150 W
```

FlashAttention 2 is unavailable on this Turing / CC 7.5 GPU, so vLLM falls back to Triton attention. AWQ is handled by the Marlin linear kernel.

The larger Qwen3-8B-AWQ checkpoint (5.68 GiB weights) could load its weights but did not leave enough memory for a usable KV cache on the 8 GB card. Qwen3-4B-AWQ fits and runs normally.

### Eager-mode baseline

Server was launched with `--enforce-eager`, disabling torch.compile and CUDA Graphs.

Warm 512-token run:

```text
Output tokens: 512
TPOT: ~43.13 ms/token
Generation: ~23.19 tok/s
End-to-end: ~23.11 tok/s
```

### Optimized mode — torch.compile + CUDA Graphs

Restarting the same model without `--enforce-eager` produced a major decode-speed improvement.

Warm 512-token run:

```text
Output tokens: 512
Wall time: 5.555 s
Mean TPOT: ~10.80 ms/token
Generation: ~92.6 tok/s
End-to-end: ~92.2 tok/s
```

This is roughly a 4x decode-speed increase over the eager-mode result.

### Long 1024-token runs

Two consecutive 1024-token runs were used to check steady-state stability.

| Run | Prompt tokens | Output tokens | Wall time | TPOT | Generation |
|---:|---:|---:|---:|---:|---:|
| 1 | 61 | 1024 | 11.984 s | 11.638 ms/token | 85.92 tok/s |
| 2 | 61 | 1024 | 11.913 s | 11.599 ms/token | 86.21 tok/s |

The two long runs differ by only about 0.3%, so a reasonable steady-state decode baseline is **~86.1 tok/s** for this workload.

### GPU telemetry during optimized inference

Observed during the optimized vLLM run:

```text
GPU load: 100%
Power: 148.42 W
Power limit: 150 W
VRAM: 6.54 / 8.00 GiB
Temperature: 48 C
Fan: 36%
```

Using the ~86.1 tok/s long-run result and ~148.4 W observed power gives approximately **0.58 tok/s/W**.

All vLLM results above were captured before any capacitor / board-level modification.


### Concurrent-request scaling — vLLM continuous batching

All tests below used Qwen3-4B-AWQ with 512 output tokens per request unless noted otherwise. The main scaling series used a 150 W power limit.

| Concurrent requests | Total output tokens | Wall time | Aggregate throughput | Approx. per-request speed | Notes |
|---:|---:|---:|---:|---:|---|
| 1 | 1024 | 11.913 s | ~86.0 tok/s | ~86.0 tok/s | Stable long-run single-request baseline |
| 4 | 2048 | 6.986 s | 293.14 tok/s | ~73.3–73.8 tok/s | GPU load 100% |
| 8 | 4096 | 8.989 s | 455.65 tok/s | ~57.1 tok/s | 150 W PL; ~151.24 W observed |
| 16 | 8192 | 13.701 s | 597.93 tok/s | ~37.45 tok/s | 150 W PL; ~148.39 W observed |
| 32 | 16384 | 22.821 s | 717.92 tok/s | ~22.46–22.55 tok/s | 150 W PL; ~150.26 W observed |
| 64 | 32768 | 47.306 s | 692.68 tok/s | ~10.83–14.84 tok/s | 150 W PL; ~144.44 W observed; throughput regressed |

The highest aggregate throughput measured so far is **717.92 tok/s at 32 concurrent requests**. Going from 32 to 64 concurrent requests reduced aggregate throughput by about 3.5% while substantially increasing per-request latency, indicating that the useful batching sweet spot is below 64 for this workload.

#### Power-limit A/B at 8 concurrent requests

| Power limit | Observed power | Aggregate throughput | Change vs 150 W |
|---:|---:|---:|---:|
| 150 W | ~151.24 W | 455.65 tok/s | baseline |
| 184 W | ~158.23 W | 458.33 tok/s | +0.6% |

Raising the power limit from 150 W to 184 W produced only a negligible throughput gain. For this workload, 150 W is therefore the more efficient operating point.

All scaling results were captured before any capacitor / board-level modification.
