# BTCpro CMP vLLM parallelism tests — 2026-09-27

## Host

- Board: ASRock H510 Pro BTC+
- CPU: Intel Celeron G5905, 2C/2T @ 3.50 GHz
- RAM: 7.32 GiB
- Swap: 16 GiB swapfile
- OS: Ubuntu 24.04.5 LTS
- Kernel: 6.8.0-139-generic
- NVIDIA driver/KMD: 610.43.03
- CUDA UMD: 13.3
- vLLM: 0.29.0
- Torch: 2.13.0+cu130

## GPUs / PCIe

- GPU0: NVIDIA CMP 50HX, 10 GiB, PCIe Gen2 x4
- GPU1: NVIDIA CMP 50HX, 10 GiB, PCIe Gen2 x1
- GPU2: NVIDIA CMP 40HX, 8 GiB, PCIe Gen1 x1

P2P/custom all-reduce is unavailable on this platform, so vLLM falls back to PYNCCL.

## Qwen3-4B-AWQ

Workload: short profile, concurrency 12, 512 output tokens/request, context 4096, thinking off.

### Single CMP50HX, GPU0 Gen2 x4

- Aggregate output throughput: **608.35 tok/s**
- Per-request throughput: **58.19–58.36 tok/s**
- TTFT avg/max: **0.165 / 0.175 s**
- Wall time: **10.100 s**
- Peak VRAM: **8.95 GiB**
- Average GPU power: **136.67 W**

### Single CMP50HX, GPU1 Gen2 x1

- Aggregate output throughput: **607.21 tok/s**
- Per-request throughput: **57.56–57.73 tok/s**
- TTFT avg/max: **0.095 / 0.101 s**
- Wall time: **10.118 s**
- Peak VRAM: **8.95 GiB**
- Average GPU power: **138.58 W**

Result: once the model is resident in VRAM, Gen2 x1 vs Gen2 x4 had essentially no decode-throughput penalty for this single-GPU workload.

### 2× CMP50HX, Tensor Parallel = 2

- GPU0 Gen2 x4 + GPU1 Gen2 x1
- Aggregate output throughput: **237.20 tok/s**
- Per-request throughput: **20.13–20.21 tok/s**
- TTFT avg/max: **1.568 / 1.683 s**
- Wall time: **25.902 s**
- Peak VRAM: **9.26 GiB per GPU**
- Average total GPU power: **150.64 W**

Result: TP=2 was much slower than a single GPU on this small model because of inter-GPU synchronization over the x1 link.

## Qwen3-14B-AWQ

The model does not fit on a single CMP50HX in vLLM. Single-GPU load failed while trying to allocate another 1.45 GiB with only about 1.40 GiB free.

Workload below: concurrency 12, 512 output tokens/request, context 4096, thinking off.

### 2× CMP50HX, Tensor Parallel = 2

- GPU0 Gen2 x4 + GPU1 Gen2 x1
- Aggregate output throughput: **133.57 tok/s**
- Per-request throughput: **11.41–11.46 tok/s**
- Average decode speed/request: **12.25 tok/s**
- TTFT avg/max: **3.029 / 3.247 s**
- Wall time: **45.998 s**
- Peak VRAM: **9.25 GiB per GPU**
- Average total GPU power: **168.96 W**
- Average CPU load: **98.05%**

During TP initialization, vLLM reported that custom all-reduce was unavailable because GPU P2P was unavailable/failed, and used PYNCCL.

### 2× CMP50HX, Pipeline Parallel = 2

Manual launch:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,1 \
/home/ai6/vllm-env/bin/vllm serve \
  /home/ai6/models/vllm/Qwen3-14B-AWQ \
  --host 0.0.0.0 \
  --port 8013 \
  --dtype half \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 2
```

Benchmark result:

- Aggregate output throughput: **311.33 tok/s**
- Per-request throughput: **25.94 tok/s**
- Wall time: **19.735 s**
- 6144 output tokens total

Compared with TP=2 on the same two GPUs:

- PP=2: **311.33 tok/s**
- TP=2: **133.57 tok/s**
- PP=2 is about **2.33× faster**
- Wall time dropped from **45.998 s** to **19.735 s**

This is the key result: on a mixed-width PCIe topology with no useful GPU P2P, layer/pipeline splitting is far more suitable than tensor parallelism.

### 2× CMP50HX + 1× CMP40HX, Pipeline Parallel = 3

Manual PP=3 server used GPU0 + GPU1 + GPU2.

Benchmark result:

- Wall time: **129.609 s**
- Aggregate output throughput: **47.40 tok/s**
- Per-request throughput: **3.95 tok/s**
- 6144 output tokens total

Live vLLM telemetry during the run stabilized around 49–50 tok/s aggregate. All GPUs showed near-100% utilization, but relatively low power draw, indicating pipeline stalls/synchronization rather than full compute saturation.

Result: **PP=3 is much worse than PP=2** on this mixed 50HX + 50HX + 40HX setup. The slower CMP40HX / Gen1 x1 stage appears to bottleneck the entire pipeline.


### 1× CMP50HX + 1× CMP40HX, Pipeline Parallel = 2

Manual PP=2 server used GPU0 (CMP50HX, Gen2 x4) + GPU2 (CMP40HX, Gen1 x1).

Benchmark result:

- Wall time: **186.017 s**
- Aggregate output throughput: **33.03 tok/s**
- Per-request throughput: **2.75 tok/s**
- 6144 output tokens total

Live telemetry during the run showed GPU0 at about 59.65 W and GPU2 at about 80.99 W despite 99–100% reported GPU load, again consistent with pipeline waiting/stalls rather than full compute saturation.

This was even slower than the 3-GPU PP=3 run (47.40 tok/s). A likely explanation is that with PP=2 the slower CMP40HX must process roughly half of the model stages, while with PP=3 it receives a smaller share; either way the heterogeneous CMP40HX/Gen1 x1 stage is a severe bottleneck.


## Reproducible benchmark protocol

Use the committed harness:

```bash
python3 benchmarks/vllm_chat_bench.py \
  --url http://127.0.0.1:8013/v1/chat/completions \
  --model /home/ai6/models/vllm/Qwen3-14B-AWQ \
  -n 12 \
  -o 512
```

Fixed benchmark prompt:

```text
Write a production-quality Python implementation of an asynchronous HTTP crawler with retries, timeout handling, URL deduplication, SHA256 hashing, logging, graceful shutdown, and type hints.
```

Benchmark rules for future GPU comparisons:

1. Keep the exact model, prompt, context limit, output length and concurrency unless explicitly testing one of those variables.
2. Keep `temperature=0` and `enable_thinking=False`.
3. Record GPU model, PCIe generation/width, TP/PP layout, vLLM version, driver, CUDA runtime, CPU/RAM and power limits.
4. Run one warm-up pass first. The first pass may include Triton/JIT/graph compilation and can under-report steady-state throughput.
5. Record at least the next two warm runs, including wall time, aggregate tok/s and per-request tok/s.
6. Do not compare a prefix-cache-heavy run to a cold-cache run as if they were identical. Record prefix-cache hit rate when visible.
7. For multi-GPU tests, record whether P2P/custom all-reduce is available and which NCCL/communication backend vLLM selects.

The purpose is to make the later P104 tests directly comparable to the CMP results.

### Warm-repeat PP=2 results, 2× CMP50HX

After restarting the known-good 2× CMP50HX PP=2 configuration, the same 12×512 test was repeated:

- First run after startup: **285.25 tok/s**, wall **21.539 s**, per request **23.77 tok/s**
- Immediate second/warm run: **338.02 tok/s**, wall **18.176 s**, per request **28.17 tok/s**
- Earlier reference run: **311.33 tok/s**, wall **19.735 s**, per request **25.94 tok/s**

The warm repeat confirms that the 2× CMP50HX PP=2 setup consistently operates in roughly the 300+ tok/s class for this exact workload, while startup/JIT/cache state can move a short benchmark by around 10% or more.

During the 338 tok/s run, telemetry showed approximately:

- GPU0 CMP50HX Gen2 x4: ~132 W, ~45% memory utilization
- GPU1 CMP50HX Gen2 x1: ~151 W, ~64% memory utilization

This is materially different from the heterogeneous CMP40 pipeline tests, where reported GPU load stayed near 100% but board power remained only around 60–80 W, consistent with waiting/stalls rather than dense compute.



## Concurrency sweep — 2× CMP50HX, PP=2, Qwen3-14B-AWQ

Fixed workload: 512 output tokens/request, temperature 0, thinking disabled, same coding prompt.

| Concurrency | Output tokens | Wall time | Aggregate | Per request |
|---:|---:|---:|---:|---:|
| 12 (warm) | 6144 | 18.176 s | **338.02 tok/s** | 28.17 tok/s |
| 24 | 12288 | 28.144 s | **436.61 tok/s** | 18.19 tok/s |
| 32 | 16384 | 31.262 s | **524.08 tok/s** | 16.38 tok/s |
| 48 | 24576 | 37.457 s | **656.11 tok/s** | 13.67 tok/s |
| 64 | 32768 | 47.033 s | **696.70 tok/s** | 10.89 tok/s |
| 96 | 49152 | 71.966 s | **682.99 tok/s** | 7.11 tok/s |

The aggregate-throughput peak in this sweep is at concurrency 64: **696.70 tok/s**. Raising concurrency from 64 to 96 reduced aggregate throughput by about **2.0%** while per-request throughput fell from 10.89 to 7.11 tok/s. For this exact benchmark and host, concurrency 64 is therefore the measured throughput sweet spot; lower concurrency remains preferable when per-request latency matters.


## Practical conclusion

For this BTCpro host:

- Small models that fit in one GPU: use **one GPU per worker**.
- Qwen3-14B-AWQ on 2× CMP50HX: use **Pipeline Parallel = 2**, not Tensor Parallel = 2.
- Do not use the mixed 50HX + 50HX + 40HX as PP=3 for this workload.
- The CMP40HX is better kept as a separate worker when the two CMP50HX are serving a PP=2 model.
- 8 GiB system RAM was insufficient for multi-process vLLM startup; adding a 16 GiB swapfile prevented Linux OOM-killer crashes, though more physical RAM would be preferable.
