# Hardware

## Core system

- Motherboard: BIOSTAR H81A Ver. 6.1 mining board
- CPU: Intel Core i3-4130, 2C/4T, 3.4 GHz, Intel HD 4400
- RAM: 8 GB DDR3
- System SSD: 240 GB SATA
- GPUs: 6× NVIDIA P104-100, 8 GB each after validated VBIOS configuration
- Aggregate physical VRAM: 48 GB

## GPU topology

The motherboard exposes one long PCIe slot and five PCIe x1 slots. All six GPUs are connected through mining-style risers.

Observed topology:

- all GPU pairs appear as PHB in `nvidia-smi topo -m`
- each GPU negotiates PCIe x1 at 2.5 GT/s
- kernel reports about 2.0 Gb/s available PCIe bandwidth per card
- no CUDA P2P read/write path is available between any GPU pair
- no NVLink is available on P104-100

This topology strongly affects tensor-parallel scaling.

## Power

The system has been tested with mining PSUs. Because PSU quality and cabling vary, nominal wattage should not be treated as guaranteed continuous safe output.

Practical wiring rules used for this build:

- never hot-plug GPU or riser power
- avoid SATA-to-riser adapters
- prefer dedicated PCIe power cables where possible
- keep each GPU and its riser on the same PSU when multiple PSUs are used
- stop immediately on smell, crackling, excessive connector heat, or repeated resets

## GPU validation

All six P104-100 cards were validated under sustained mining workloads before AI deployment. A six-card Etchash run delivered roughly 207 MH/s aggregate with zero reported GPU errors.

## Upgrade notes

A CPU upgrade is not currently a priority for GPU inference. The dominant bottleneck for one-model multi-GPU tensor parallelism is the PCIe x1/no-P2P platform, not the i3-4130.

A future platform with more real CPU PCIe lanes and P2P-capable topology would be the meaningful hardware upgrade if the goal becomes maximizing single-request multi-GPU performance.