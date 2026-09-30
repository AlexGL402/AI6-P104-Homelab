# huanan cmp40 - Soup

Soup streamed-NF4 LoRA training benchmark on the `huanan` host.

## Hardware

- Host: `huanan`
- GPU: NVIDIA CMP 40HX 8 GB
- GPU bus: `0000:03:00.0`
- PCIe: Gen1 x16
- NVIDIA driver: 610.43.03
- GPU power limit: 184 W
- System RAM: ~62 GiB usable
- Swap: disabled
- Model/shards stored on NVMe

## Software

- Soup: 0.75.0
- Soup commit: `f0bb65061caa10d9e231425269827f9288927551`
- Python: 3.12.3
- PyTorch: 2.14.1+cu130
- bitsandbytes: 0.50.2
- transformers: 5.18.0
- peft: 0.21.1
- trl: 0.29.1
- accelerate: 1.15.0

## Model

`unsloth/Llama-3.1-8B-Instruct`

The Hugging Face snapshot was materialized as a normal local checkpoint at:

`/mnt/nvme/data/models/llama31-8b-unsloth`

Model parameter split reported by the Soup harness:

- decoder: 6.980 B
- embedding: 0.525 B
- lm_head: 0.525 B
- effective workload: 43.98 GFLOP/token

## Benchmark protocol

Soup `benchmarks/harness/issue361_nf4_throughput.py`, issue #361 protocol.

- quantization: NF4
- stream dtype: FP16
- double quant: enabled
- sequence length: 512
- batch: 1
- buffers: 2
- LoRA rank: 16
- LoRA targets: `q_proj,v_proj`
- warm-up: 10 steps
- measured: 50 steps

## Result

Clean repeat run with prebuilt shards:

- throughput: **174.67 tok/s**
- median step time: **2931.6 ms**
- step min/max: 2925.6 / 2939.2 ms
- pinned host store: **5.702 GB**
- peak VRAM allocated: **2.397 GB**
- peak VRAM reserved: **2.703 GB**
- GPU utilization: **100%**
- median SM clock: **1875 MHz**
- max GPU temperature: **75 C**
- effective compute: **7.682 TFLOPS**
- same-session BF16 GEMM ceiling: **48.411 TFLOPS**
- model build: 4.8 s
- shard build on repeat: 0.0 s

Initial run, including shard creation:

- throughput: **174.7 tok/s**
- median step time: ~2931 ms
- peak VRAM: 2.40 GB allocated / 2.61 GB reserved
- GPU utilization: 100%
- median SM clock: 1905 MHz
- max temperature: 69 C
- BF16 GEMM ceiling: 48.88 TFLOPS
- shard build: 48.0 s
- model build: 4.8 s

The two runs produced essentially identical throughput, establishing a stable
**CMP 40HX PCIe Gen1 x16 baseline of ~174.7 tok/s**.

## Storage / RAM observations

The generated NF4 shard directory is approximately 5.4 GB.

During shard/model loading Linux used a large amount of otherwise-free RAM as
filesystem cache. `free -h` showed approximately:

- used: 9.8 GiB
- buff/cache: 58 GiB
- available: 52 GiB

The Soup Python process itself had approximately 7.8 GB RSS, so the apparent
~62 GiB RAM usage shown by the AI6 monitor was mostly filesystem cache rather
than application memory pressure.

This also identifies a monitor issue: the RAM widget should distinguish
reclaimable cache from actually used memory.

## Notes

The official Meta Llama repository was gated. An open Unsloth Llama 3.1 8B
checkpoint was used instead.

Soup 0.75.0 initially rejected the Hugging Face cached snapshot because
`tokenizer.json` was considered outside the expected HF blob store. The
snapshot was materialized into a regular local directory (`cp -L`) and Soup
then resolved the model successfully.

The first benchmark created the NF4 shards. The repeat benchmark reused them
and is the JSON result preserved here.

Next planned A/B test: repeat the identical protocol after attempting to move
the CMP 40HX PCIe link from Gen1 x16 to Gen2 x16.
