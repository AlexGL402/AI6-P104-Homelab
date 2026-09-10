# Развёртывание AI6-P104-Homelab на второй машине

Эта инструкция описывает, как развернуть аналогичную AI6-систему на второй BIOSTAR H81A с 6× NVIDIA P104-100 8 GB.

> Репозиторий хранит конфиги, systemd-сервисы, Docker-стек, prompt и результаты тестов. Модели, секреты, драйверы и собранные бинарники llama.cpp в Git не кладутся.

## 0. Эталонная конфигурация

Проверенная исходная система:

- BIOSTAR H81A Ver. 6.1
- Intel Core i3-4130
- 8 GB DDR3
- 6× NVIDIA P104-100 8 GB
- Ubuntu Server 24.04 LTS
- NVIDIA driver 580.173.02
- CUDA 12.8 для сборки Pascal `compute_61`
- NCCL 2.26.2+cuda12.8
- llama.cpp с CUDA + NCCL
- Qwen3-Coder 30B Q4_K
- два независимых worker-а по 3 GPU
- Open WebUI
- Open Terminal

Рабочая схема:

```text
GPU 0,1,2 -> llama-server :8081 -> qwen3-coder-30b-gpu012
GPU 3,4,5 -> llama-server :8082 -> qwen3-coder-30b-gpu345
Open WebUI                -> :3000
Open Terminal             -> :8000
Agent web-preview ports   -> :8001-8010
```

## 1. Установить Ubuntu Server

Рекомендуется Ubuntu Server 24.04 LTS.

После установки:

```bash
sudo apt update
sudo apt full-upgrade -y
sudo reboot
```

Полезные пакеты:

```bash
sudo apt install -y git curl wget build-essential cmake ninja-build pkg-config \
  python3 python3-pip openssl docker.io docker-compose-v2
```

Добавить текущего пользователя в группу Docker:

```bash
sudo usermod -aG docker $USER
```

После этого выйти из SSH и зайти снова.

Проверка:

```bash
docker --version
docker compose version
```

## 2. Проверить все 6 GPU

После установки NVIDIA driver:

```bash
nvidia-smi
```

Должны быть видны все 6 P104-100 и около 8192 MiB VRAM на каждой.

Проверить топологию:

```bash
nvidia-smi topo -m
```

На BIOSTAR H81A ожидается PCIe x1 и отсутствие CUDA P2P между картами. Это нормально для этой сборки.

Полезно сохранить диагностику:

```bash
nvidia-smi -q > ~/nvidia-smi-full.txt
lspci -nn > ~/lspci.txt
```

## 3. Клонировать этот репозиторий

```bash
cd ~
git clone https://github.com/AlexGL402/AI6-P104-Homelab.git
cd AI6-P104-Homelab
```

## 4. Установить CUDA 12.8 и совместимый NCCL

Для P104-100 нужен Pascal `compute_61`.

Эталонная сборка llama.cpp делалась именно CUDA 12.8. CUDA 13.x для компиляции `compute_61` не использовать.

После установки CUDA проверить:

```bash
/usr/local/cuda-12.8/bin/nvcc --version
```

NCCL должен быть из CUDA 12.x-ветки. На эталонной машине использовался:

```text
NCCL 2.26.2+cuda12.8
```

Проверить установленные пакеты:

```bash
dpkg -l | grep -E 'nccl|cuda'
```

## 5. Собрать llama.cpp

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp.git
cd ~/llama.cpp
```

Чистая сборка:

```bash
rm -rf build
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_NCCL=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_CUDA_ARCHITECTURES=61

cmake --build build -j$(nproc)
```

Проверить:

```bash
~/llama.cpp/build/bin/llama-cli --list-devices
```

Должны быть видны все 6 P104-100.

## 6. Подготовить модель

Эталонная модель:

```text
Qwen3-Coder 30B Q4_K
```

На первой машине использовался blob:

```text
/var/lib/ollama-models/blobs/sha256-1194192cf2a187eb02722edcc3f77b11d21f537048ce04b67ccf8ba78863006a
```

На второй машине можно либо скопировать этот файл по LAN, либо получить модель отдельно.

Если модель уже есть в Ollama:

```bash
ollama show qwen3-coder:30b --modelfile
```

Команда покажет исходный blob/path.

Создать каталог при необходимости:

```bash
sudo mkdir -p /var/lib/ollama-models/blobs
sudo chown -R $USER:$USER /var/lib/ollama-models
```

После копирования убедиться, что файл существует:

```bash
ls -lh /var/lib/ollama-models/blobs/
```

Если путь модели отличается от эталонного, изменить `ExecStart` в обоих файлах `configs/llama-8081.service` и `configs/llama-8082.service` до их установки.

## 7. Проверить systemd-конфиги

Перед копированием открыть:

```bash
nano ~/AI6-P104-Homelab/configs/llama-8081.service
nano ~/AI6-P104-Homelab/configs/llama-8082.service
```

Проверить:

- `User=` — должен совпадать с пользователем второй машины;
- `WorkingDirectory=/home/USER/llama.cpp`;
- путь к модели;
- `CUDA_VISIBLE_DEVICES=0,1,2` для :8081;
- `CUDA_VISIBLE_DEVICES=3,4,5` для :8082;
- aliases моделей;
- context `32768`.

## 8. Установить два llama.cpp worker-а

```bash
cd ~/AI6-P104-Homelab
sudo cp configs/llama-8081.service /etc/systemd/system/
sudo cp configs/llama-8082.service /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable --now llama-8081 llama-8082
```

Проверить статус:

```bash
systemctl --no-pager --full status llama-8081
systemctl --no-pager --full status llama-8082
```

Проверить health:

```bash
curl http://127.0.0.1:8081/health
curl http://127.0.0.1:8082/health
```

Во время загрузки модели временно может быть `503 Loading model`. После загрузки ожидается `{"status":"ok"}`.

Проверить VRAM:

```bash
nvidia-smi
```

В idle после загрузки модели ожидается примерно 7.1-7.3 GB VRAM на каждой карте.

## 9. Поднять Open WebUI и Open Terminal

```bash
cd ~/AI6-P104-Homelab/deploy
cp .env.example .env
```

Сгенерировать новые секреты именно для второй машины:

```bash
openssl rand -hex 32
openssl rand -hex 32
```

Не публиковать эти значения в Git.

Открыть `.env`:

```bash
nano .env
```

Вписать новые секреты и сохранить.

Запуск:

```bash
docker compose up -d
```

Проверить:

```bash
docker ps
```

Ожидаемые внешние порты:

```text
3000       Open WebUI
8000       Open Terminal API
8001-8010  приложения, создаваемые агентом
```

Проверить Open Terminal:

```bash
curl -s http://127.0.0.1:8000/openapi.json | \
python3 -c "import json,sys; print(json.load(sys.stdin)['info']['version'])"
```

## 10. Первый вход в Open WebUI

С другого компьютера в локальной сети открыть:

```text
http://SECOND_MACHINE_IP:3000
```

Создать первый admin-аккаунт.

## 11. Подключить оба llama.cpp worker-а

В Open WebUI:

```text
Админ -> Настройки -> Подключения -> OpenAI API
```

Добавить:

```text
http://SECOND_MACHINE_IP:8081/v1
http://SECOND_MACHINE_IP:8082/v1
```

Для локального llama.cpp можно использовать произвольный API key, например:

```text
sk-local
```

Если `host.docker.internal` работает корректно в конкретной установке, его тоже можно использовать. На эталонной машине для некоторых интеграций надёжнее оказался LAN IP.

После подключения должны появиться две модели:

```text
qwen3-coder-30b-gpu012
qwen3-coder-30b-gpu345
```

## 12. Подключить Open Terminal

В Open WebUI:

```text
Админ -> Настройки -> Интеграции -> Open Terminal -> +
```

Заполнить:

```text
Имя: AI6 Terminal
URL: http://SECOND_MACHINE_IP:8000
Вход: Bearer
API Key: значение OPEN_TERMINAL_API_KEY из .env
Chat Uploads: По умолчанию
```

Нажать кнопку проверки подключения, затем `Сохранить`.

Включить переключатель `AI6 Terminal`.

Если файловая панель показывает `401 Unauthorized`, сначала проверить ключ напрямую:

```bash
KEY=$(docker inspect open-terminal \
  --format '{{range .Config.Env}}{{println .}}{{end}}' | \
  sed -n 's/^OPEN_TERMINAL_API_KEY=//p')

curl -i \
  -H "Authorization: Bearer $KEY" \
  http://127.0.0.1:8000/system
```

Ожидается `HTTP/1.1 200 OK`.

## 13. Назначить Terminal модели

В Open WebUI:

```text
Админ -> Настройки -> Модели -> qwen3-coder-30b-gpu012
```

Возможности:

```text
Терминал: включить
```

В поле выбора Terminal выбрать:

```text
AI6 Terminal
```

В `Системный prompt` вставить содержимое:

```text
configs/coding-agent-system-prompt.md
```

Нажать `Сохранить и обновить`.

То же можно сделать для `qwen3-coder-30b-gpu345`.

## 14. Проверить agentic coding

Первый простой тест:

```text
Создай через терминал файл hello.py с содержимым
print("Hello from AI6 Terminal"), запусти его и покажи фактический вывод.
```

Ожидаемая цепочка:

```text
write_file -> run_command -> get_process_status -> stdout
```

Ожидаемый stdout:

```text
Hello from AI6 Terminal
```

## 15. Проверить web-preview

Тестовый запрос агенту:

```text
Создай небольшой FastAPI проект, запусти его на первом свободном порту
из диапазона 8001-8010 с bind на 0.0.0.0, проверь endpoint и верни
LAN-ссылку и Swagger /docs.
```

Если агент выбрал 8001, открыть с другого ПК:

```text
http://SECOND_MACHINE_IP:8001
http://SECOND_MACHINE_IP:8001/docs
```

## 16. Полезные проверки после перезагрузки

```bash
nvidia-smi
systemctl is-active llama-8081
systemctl is-active llama-8082
docker ps
curl http://127.0.0.1:8081/health
curl http://127.0.0.1:8082/health
```

Open WebUI и Open Terminal должны подняться через Docker restart policy, а оба llama.cpp worker-а — через systemd.

## 17. Архитектурное замечание

На этой mining-платформе все P104 работают через PCIe x1 и CUDA P2P недоступен. Поэтому использовать все 6 карт для одного tensor-parallel inference невыгодно.

Проверенный лучший практический режим:

```text
3 GPU + 3 GPU
```

Два независимых Qwen3-Coder 30B worker-а дают около 32 tok/s каждый и примерно 64 tok/s aggregate при двух одновременных запросах.

## 18. Что пока НЕ автоматизировано

На чистой Ubuntu пока вручную выполняются:

- установка NVIDIA driver;
- установка CUDA 12.8;
- установка подходящего NCCL;
- копирование/загрузка модели;
- первичная проверка GPU topology.

Остальная конфигурация уже сохранена в репозитории.

В дальнейшем можно добавить `bootstrap.sh`, который будет автоматически проверять зависимости, устанавливать services, поднимать Docker-стек и выдавать итоговый PASS/FAIL отчёт.
