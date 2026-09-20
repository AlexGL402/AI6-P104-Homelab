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

## Identification details still to capture

For the final hardware record, capture exact PCI/device/subsystem/VBIOS information with:

```bash
nvidia-smi --query-gpu=name,pci.bus_id,pci.device_id,pci.sub_device_id,uuid,vbios_version,memory.total,power.default_limit,power.max_limit --format=csv

lspci -nnk -s 01:00.0

sudo lspci -vv -s 01:00.0
```

Append these values after collection so the exact CMP 40HX board identity is preserved before any hardware modification.
