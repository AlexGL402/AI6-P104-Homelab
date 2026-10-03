# Dual CMP40HX benchmarks — Huanan X79 — 2026-10-04

## Platform

- Host: Huanan X79
- GPUs: 2x NVIDIA CMP 40HX 8 GB
- GPU0 PCIe: Gen1 x16
- GPU1 PCIe: Gen1 x4
- llama.cpp build: 434ddbbc0 (10884)
- CUDA backend
- Full GPU offload: `-ngl 999`
- Split mode: `-sm layer`

## Llama 3.1 8B Instruct Q6_K_L

Model:
`Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf`

### 2x CMP40HX, split 1/1

Command parameters:
`-ngl 999 -sm layer -ts 1/1 -p 512 -n 256`

Results:

- pp512: 1427.99 ± 10.16 tok/s
- tg256: 52.76 ± 0.09 tok/s
- VRAM: ~3.03 GiB + ~3.29 GiB
- GPU load: ~49% / 51%
- Power observed: ~160 W / ~165 W

For comparison:

- single CMP40 Gen1 x16: pp512 1492.64, tg256 47.94
- single CMP40 Gen1 x4: pp512 1457.26, tg256 47.70

Conclusion:
For an 8B model that fits entirely on one CMP40, using two CMP40 cards gives only about 10% decode improvement and slightly worse prompt processing.

## Mistral Small 24B Instruct 2501 IQ3_XS

Model:
`Mistral-Small-24B-Instruct-2501-IQ3_XS.gguf`

Model size reported by llama-bench:
9.22 GiB, 23.57B params, IQ3_XS 3.3 bpw.

### Split 1/1, PL 184 W

- pp512: 525.74 ± 1.67 tok/s
- tg256: 15.89 ± 0.33 tok/s
- VRAM: ~4.68 GiB + ~4.80 GiB
- Load: ~50% / 50%
- Peak observed temperature on hotter card: ~83-88 C

### Split 1/1, PL 150 W

- pp512: 492.91 ± 1.21 tok/s
- tg256: 15.39 ± 0.22 tok/s

Relative to 184 W:
- pp512: about -6.2%
- tg256: about -3.1%

### Split 1.5/1, PL 150 W

- pp512: 505.92 ± 1.17 tok/s
- tg256: 15.54 ± 0.17 tok/s
- VRAM: ~5.52 GiB + ~3.97 GiB
- Load: ~60% / 40%

This split reduced load on the hotter x4-position card while keeping nearly the same decode speed.

## Mistral Small 24B Instruct 2501 Q4_K_M

Model:
`Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf`

llama-bench reports:
13.34 GiB, 23.57B params, Q4_K Medium.

The model fits fully in the combined VRAM of 2x CMP40HX with `-ngl 999`.

### Short validation, split 1/1

Parameters:
`-p 128 -n 32`

- pp128: 490.36 ± 4.76 tok/s
- tg32: 25.76 ± 0.04 tok/s

### Full benchmark, split 1/1

Parameters:
`-p 512 -n 256`

- pp512: 530.69 ± 1.59 tok/s
- tg256: 25.47 ± 0.12 tok/s
- VRAM: ~6.79 GiB + ~6.74 GiB
- GPU load: ~49% / 51%

### Split 1.15/1

- pp512: 529.52 ± 1.30 tok/s
- tg256: 25.37 ± 0.17 tok/s
- VRAM: ~7.08 GiB + ~6.44 GiB

This was a good practical split for shifting some work away from the hotter x4-position GPU.

### Split 1.20/1

- pp512: 527.89 ± 0.90 tok/s
- tg256: 25.36 ± 0.20 tok/s
- VRAM: ~7.41 GiB + ~6.11 GiB

The performance difference from 1/1 and 1.15/1 is negligible, while GPU0 gets close to its VRAM limit.

### Split 1.5/1

Failed to load the model, consistent with GPU0 running out of usable VRAM.

## Thermal / placement observation

After physically swapping the two CMP40 cards, the hotter behavior followed the slot/position rather than the physical card.

This strongly suggests the temperature difference is caused primarily by case airflow / card spacing rather than a defective CMP40.

Observed examples under similar load:

- better-cooled position: roughly mid-60s C
- hotter position: roughly high-70s to mid-80s C

A 150 W power limit is therefore preferred for sustained dual-CMP40 operation in the current chassis.

## Practical conclusions

- Gen1 x4 vs Gen1 x16 has little effect on fully GPU-resident 8B inference on CMP40.
- Two CMP40 cards provide little benefit for 8B models that already fit on one card.
- The pair becomes useful for models larger than 8 GB.
- Mistral Small 24B IQ3_XS runs fully on 2x CMP40.
- Mistral Small 24B Q4_K_M also fits fully on 2x CMP40 and reaches about 25.4 tok/s decode.
- For Q4_K_M, `-ts 1.15/1` is a good balance between VRAM usage, temperature distribution and performance.
