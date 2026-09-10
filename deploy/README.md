# Deploying AI6 on a second machine

Yes: this repository is intended to make a second BIOSTAR/P104 machine reproducible. The Git repository stores configuration, documentation and deployment files, but it deliberately does **not** store model blobs, secrets, NVIDIA drivers, CUDA/NCCL packages, or the compiled llama.cpp binaries.

## What can be reproduced from Git

- Open WebUI container configuration
- Open Terminal container configuration
- port layout (`3000`, `8000`, `8001-8010`)
- two llama.cpp systemd worker definitions (GPU 0-2 and GPU 3-5)
- coding-agent system prompt
- benchmark/reference documentation

## What must exist outside Git

1. Ubuntu Server 24.04 LTS
2. working NVIDIA driver with all six P104-100 cards visible
3. CUDA 12.8 toolchain for Pascal-compatible llama.cpp builds
4. NCCL compatible with the CUDA 12.8 build (the validated AI6 system used NCCL 2.26.2+cuda12.8)
5. llama.cpp built with CUDA + NCCL for compute capability 6.1
6. the Qwen3-Coder 30B GGUF/model blob on local storage
7. fresh secrets for Open WebUI and Open Terminal

Do not copy API keys or WebUI secrets from another host into a public repository.

## 1. Clone

```bash
git clone https://github.com/AlexGL402/AI6-P104-Homelab.git
cd AI6-P104-Homelab
```

## 2. Verify GPUs

```bash
nvidia-smi
nvidia-smi topo -m
```

The reference AI6 machine has six P104-100 8 GB cards. The current worker layout assumes GPU indexes `0,1,2` and `3,4,5`.

## 3. Build llama.cpp

Reference build configuration:

```bash
git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
cd ~/llama.cpp
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_NCCL=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build build -j$(nproc)
```

CUDA 13.x should not be used to compile for Pascal `compute_61`; the reference build uses CUDA 12.8.

## 4. Place the model

The reference services point to the exact local model blob used on the original machine:

```text
/var/lib/ollama-models/blobs/sha256-1194192cf2a187eb02722edcc3f77b11d21f537048ce04b67ccf8ba78863006a
```

If the second host has a different model path, edit both files in `configs/` before installing them.

To inspect an Ollama model's source blob/path:

```bash
ollama show qwen3-coder:30b --modelfile
```

Model data should normally be copied over the LAN or downloaded separately; it is too large for this Git repository.

## 5. Install the two llama.cpp workers

Review the service files first, especially `User=`, `WorkingDirectory=`, model path, and CUDA device indexes.

```bash
sudo cp configs/llama-8081.service /etc/systemd/system/
sudo cp configs/llama-8082.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llama-8081 llama-8082
```

Check:

```bash
systemctl --no-pager --full status llama-8081 llama-8082
curl http://127.0.0.1:8081/health
curl http://127.0.0.1:8082/health
```

## 6. Start Open WebUI + Open Terminal

```bash
cd deploy
cp .env.example .env
```

Generate two fresh secrets:

```bash
openssl rand -hex 32
openssl rand -hex 32
```

Put them into `.env`, then:

```bash
docker compose up -d
```

Check:

```bash
docker ps
curl http://127.0.0.1:8000/openapi.json
```

## 7. Configure Open WebUI

Open:

```text
http://SECOND_MACHINE_IP:3000
```

Add the llama.cpp OpenAI-compatible endpoints:

```text
http://SECOND_MACHINE_IP:8081/v1
http://SECOND_MACHINE_IP:8082/v1
```

or use `host.docker.internal` when it works correctly in that installation.

Add Open Terminal under Admin -> Settings -> Integrations -> Open Terminal:

```text
http://SECOND_MACHINE_IP:8000
```

Use Bearer auth with the `OPEN_TERMINAL_API_KEY` generated for that host.

Assign the Terminal capability to the coding model and copy the system prompt from:

```text
configs/coding-agent-system-prompt.md
```

## 8. LAN application preview ports

Open Terminal exposes ports `8001-8010`, so an agent-created development service should bind to `0.0.0.0` and use the first free port in that range.

Example:

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

From another LAN computer:

```text
http://SECOND_MACHINE_IP:8001
http://SECOND_MACHINE_IP:8001/docs
```

## What is not one-click yet

The current repo makes the application/configuration layer reproducible, but a bare second machine still needs the NVIDIA/CUDA/NCCL setup and the model files. A future bootstrap script can automate most of those checks and service installation, but GPU driver/toolkit installation should remain explicit because package versions and host state matter.
