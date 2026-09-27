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

## Practical conclusion

For this BTCpro host:

- Small models that fit in one GPU: use **one GPU per worker**.
- Qwen3-14B-AWQ on 2× CMP50HX: use **Pipeline Parallel = 2**, not Tensor Parallel = 2.
- Do not use the mixed 50HX + 50HX + 40HX as PP=3 for this workload.
- The CMP40HX is better kept as a separate worker when the two CMP50HX are serving a PP=2 model.
- 8 GiB system RAM was insufficient for multi-process vLLM startup; adding a 16 GiB swapfile prevented Linux OOM-killer crashes, though more physical RAM would be preferable.
