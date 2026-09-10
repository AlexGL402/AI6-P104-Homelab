# AI6 P104 Homelab

Ultra-budget local AI homelab built around six NVIDIA P104-100 8 GB GPUs, llama.cpp, Qwen3-Coder, Open WebUI, and Open Terminal.

## Quick links

- [Русская инструкция по развёртыванию на второй машине](INSTALL_RU.md)
- [Deployment notes](deploy/README.md)
- [Benchmark summary](benchmarks/2026-09-10-summary.md)
- [External benchmark comparison](benchmarks/external-comparison.md)
- [Agent tests](benchmarks/agent-tests.md)
- [Coding-agent system prompt](configs/coding-agent-system-prompt.md)

## Highlights

- 6× NVIDIA P104-100 8 GB = 48 GB aggregate VRAM
- BIOSTAR H81A Ver. 6.1 mining motherboard
- Intel Core i3-4130
- 8 GB DDR3
- Ubuntu Server 24.04 LTS
- NVIDIA driver 580.173.02
- llama.cpp CUDA 12.8 build with NCCL 2.26.2
- Two independent 3-GPU Qwen3-Coder 30B workers
- Open WebUI frontend
- Open Terminal for real file/command execution
- Agentic coding tested with FastAPI projects

## Best measured configuration

The most useful configuration on this PCIe x1 platform is not one 6-GPU tensor-parallel model, but two independent 3-GPU workers:

- Worker A: GPUs 0,1,2
- Worker B: GPUs 3,4,5
- Model: Qwen3-Coder 30B Q4_K
- Context: 32K
- Approx. generation speed per worker: ~32 tok/s
- Two simultaneous requests: 32.22 + 31.86 = **64.08 tok/s aggregate**

This is much better for total throughput than using all six GPUs for one tensor-parallel request on PCIe x1 without P2P.

## Benchmark summary

| Runtime / mode | GPUs | Prompt tok/s | Generation tok/s | Notes |
|---|---:|---:|---:|---|
| Ollama | 3 | 105.97 | 33.25 | Stable |
| Ollama | 4 | 137.94 | 36.53 | Best single-request Ollama result |
| Ollama | 6 | 113.96 | 35.23 | No single-request gain vs 4 GPUs |
| llama.cpp tensor, clean stack | 3 | 62.2 | 31.7 | Stable |
| llama.cpp tensor, clean stack | 4 | 57.8 | 29.1 | Stable |
| llama.cpp tensor, clean stack | 6 | 51.6 | 27.2 | Negative scaling from PCIe x1 / no P2P |
| llama.cpp tensor, clean stack | 2 | — | — | OOM |
| Two parallel 3-GPU workers | 3+3 | — | **64.08 aggregate** | 32.22 + 31.86 tok/s |
| Qwen3-Coder-Next 80B.A3B via Ollama | 6 | 2.58 | 8.59 | Heavy quality mode |

## Agentic coding

The system was tested as a real coding agent rather than only as a chat model.

Verified workflow:

1. Qwen3-Coder reads existing project files.
2. It creates and edits files through Open Terminal.
3. It launches commands and development servers.
4. It detects failures such as occupied ports.
5. It retries or changes configuration.
6. It validates the result with real HTTP requests.
7. It reports working LAN links back to the user.

### FastAPI validation

The agent successfully:

- created a FastAPI project;
- created `main.py`, `requirements.txt`, and tests;
- launched uvicorn inside the Open Terminal container;
- detected a port conflict;
- restarted on another port;
- verified endpoints;
- added a Pydantic-backed `POST /users` endpoint;
- updated tests;
- verified Swagger UI from another machine on the LAN.

Open Terminal development ports are exposed as:

- `8000` — Open Terminal API
- `8001-8010` — development applications

Example LAN URLs:

```text
http://10.36.1.164:8001
http://10.36.1.164:8001/docs
```

## Architecture

```text
6× P104-100 / 48 GB
        |
   +----+----+
   |         |
GPU 0-2    GPU 3-5
   |         |
Qwen 30B   Qwen 30B
~32 t/s    ~32 t/s
   |         |
   +----+----+
        |
    Open WebUI
        |
   Open Terminal
        |
 Linux / Python / files
        |
  ports 8001-8010
```

## Platform limitation

All GPUs negotiate PCIe x1 on this mining motherboard and CUDA P2P is unavailable between cards. Tensor-parallel scaling therefore becomes worse as more GPUs participate in a single request. Splitting the machine into multiple independent workers is the practical workaround.

## Repository layout

```text
benchmarks/
configs/
deploy/
docs/
INSTALL_RU.md
```

The goal of this repository is to preserve the exact hardware, software stack, benchmark results, and practical agent setup for future comparison and upgrades.
