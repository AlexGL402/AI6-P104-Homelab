# PCIe bandwidth comparison — CMP40 / CMP50 / P104 — 2026-10-04

## Test method

Custom CUDA memcpy benchmark compiled with CUDA 12.8.

Buffer size: 64 MiB.

Measurements:
- H2D = host to device
- D2H = device to host
- D2D = device to device

## Results

| GPU / Host | PCIe link | H2D GiB/s | D2H GiB/s | D2D GiB/s |
|---|---|---:|---:|---:|
| CMP40HX / Huanan | Gen1 x16 | 2.93 | 3.12 | 184.95 |
| CMP40HX / Huanan | Gen1 x4 | 0.74 | 0.78 | 185.44 |
| CMP50HX #0 / T5600 | Gen2 x4 | 1.50 | 1.55 | 8.96 |
| CMP50HX #1 / T5600 | Gen2 x4 | 1.47 | 1.55 | 8.97 |
| P104-100 / J1900 | Gen1 x1 | 0.16 | 0.20 | 134.11 |

## PCIe topology notes

### Huanan CMP40

CMP40 #1:
- LnkCap: Gen1 x16
- LnkSta: Gen1 x16

CMP40 #2:
- LnkCap: Gen1 x16
- LnkSta: Gen1 x4

Measured PCIe bandwidth difference between x16 and x4:

- H2D: 2.93 / 0.74 = ~3.96x
- D2H: 3.12 / 0.78 = 4.00x

### T5600 CMP50

Both CMP50 cards operate at Gen2 x4.

Measured:
- ~1.5 GiB/s H2D
- ~1.55 GiB/s D2H

### J1900 P104

P104 card capability:
- Gen1 x4

Actual link:
- Gen1 x1 downgraded

Upstream port:
- capable of Gen2 x1
- currently Gen1 x1

Measured:
- H2D: 0.16 GiB/s
- D2H: 0.20 GiB/s

This configuration is heavily PCIe constrained.

## Practical conclusions

- CMP40 Gen1 x16 provides about 4x the host-device bandwidth of CMP40 Gen1 x4.
- CMP50 Gen2 x4 provides about 2x the host-device bandwidth of CMP40 Gen1 x4.
- J1900 P104 at Gen1 x1 is severely bottlenecked for host-device transfers.
- Fully GPU-resident LLM decode is much less sensitive to PCIe width than prompt processing, model loading, CPU offload, or workloads with frequent host-device transfers.
- Upgrading the second CMP40 from x4 to x16 would materially improve PCIe bandwidth, even though token-generation speed may improve only slightly for models fully resident in VRAM.
