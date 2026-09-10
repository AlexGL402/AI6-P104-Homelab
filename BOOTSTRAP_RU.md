# Быстрое развёртывание через bootstrap.sh

`bootstrap.sh` автоматизирует проверку второй AI6-машины и установку той части конфигурации, которая уже хранится в Git.

> Скрипт специально **не устанавливает автоматически NVIDIA driver, CUDA, NCCL и модель**. Для GPU-стека это сделано намеренно: версии драйвера/пакетов и состояние конкретного хоста лучше проверять явно.

## 1. Клонировать репозиторий

```bash
cd ~
git clone https://github.com/AlexGL402/AI6-P104-Homelab.git
cd AI6-P104-Homelab
```

Запускать можно без изменения executable-bit:

```bash
bash bootstrap.sh check
```

Или один раз:

```bash
chmod +x bootstrap.sh
./bootstrap.sh check
```

## 2. Сначала только диагностика

```bash
bash bootstrap.sh check
```

Проверяется:

- наличие `nvidia-smi`;
- количество GPU (по умолчанию ожидается 6);
- CUDA 12.8 `nvcc`;
- NCCL runtime;
- собранный `llama-server`;
- модель Qwen3-Coder 30B;
- Docker;
- Docker Compose;
- `curl` и `openssl`;
- RAM;
- GPU topology.

В конце выводится итог:

```text
PASS: ...
WARN: ...
FAIL: ...
Result: PASS/FAIL
```

Если есть `FAIL`, сначала исправить соответствующий пункт.

## 3. Значения по умолчанию

Скрипт ожидает:

```text
llama.cpp:
~/llama.cpp

llama-server:
~/llama.cpp/build/bin/llama-server

model:
/var/lib/ollama-models/blobs/sha256-1194192cf2a187eb02722edcc3f77b11d21f537048ce04b67ccf8ba78863006a

GPU workers:
0,1,2 -> :8081
3,4,5 -> :8082
```

Если на второй машине пути отличаются, их можно передать без редактирования скрипта.

Пример:

```bash
MODEL_PATH=/data/models/qwen3-coder-30b.gguf \
LLAMA_DIR=/home/$USER/llama.cpp \
bash bootstrap.sh check
```

Поддерживаемые переменные:

```text
MODEL_PATH
LLAMA_DIR
LLAMA_SERVER
EXPECTED_GPUS
```

## 4. Установить только llama.cpp services

После того как `check` проходит:

```bash
bash bootstrap.sh install-services
```

Скрипт автоматически генерирует systemd-unit'ы под текущего пользователя и текущий `$HOME`, поэтому вручную менять `User=ai6` на второй машине не требуется.

Устанавливаются и включаются:

```text
llama-8081.service  GPU 0,1,2
llama-8082.service  GPU 3,4,5
```

После установки проверяются:

```text
http://127.0.0.1:8081/health
http://127.0.0.1:8082/health
```

Во время первой загрузки модели `llama-server` может некоторое время отвечать `503`; bootstrap ждёт запуск до нескольких минут.

## 5. Поднять только Docker-часть

```bash
bash bootstrap.sh docker
```

Если `deploy/.env` ещё отсутствует, скрипт сам создаст два новых случайных секрета:

```text
WEBUI_SECRET_KEY
OPEN_TERMINAL_API_KEY
```

Секреты не печатаются в терминал и `.env` получает права `600`.

Затем запускаются:

```text
Open WebUI    :3000
Open Terminal :8000
Agent apps    :8001-8010
```

Если `.env` уже существует, bootstrap сохраняет его и не перезаписывает секреты.

## 6. Полное развёртывание после подготовки GPU-стека

Когда драйвер, CUDA 12.8, NCCL, llama.cpp и модель уже готовы:

```bash
bash bootstrap.sh all
```

Команда:

1. проверит prerequisites;
2. создаст и установит два systemd worker-а;
3. включит автозапуск worker-ов;
4. создаст `deploy/.env`, если его нет;
5. поднимет Open WebUI + Open Terminal;
6. проверит оба llama-worker-а;
7. проверит Open Terminal;
8. проверит Open WebUI;
9. покажет контейнеры и итоговый PASS/WARN/FAIL.

## 7. После bootstrap

Узнать IP второй машины:

```bash
hostname -I
```

Открыть:

```text
http://SECOND_MACHINE_IP:3000
```

Дальше в Open WebUI остаётся выполнить UI-настройку:

- добавить `http://SECOND_MACHINE_IP:8081/v1`;
- добавить `http://SECOND_MACHINE_IP:8082/v1`;
- подключить Open Terminal `http://SECOND_MACHINE_IP:8000` с Bearer key из `deploy/.env`;
- назначить `AI6 Terminal` coding-моделям;
- вставить `configs/coding-agent-system-prompt.md`.

Посмотреть локальный ключ Open Terminal на второй машине можно так:

```bash
grep '^OPEN_TERMINAL_API_KEY=' ~/AI6-P104-Homelab/deploy/.env
```

Не публиковать этот ключ в Git или публичных логах.

## 8. Быстрая проверка после перезагрузки

```bash
systemctl is-active llama-8081
systemctl is-active llama-8082
docker ps
curl http://127.0.0.1:8081/health
curl http://127.0.0.1:8082/health
curl -s http://127.0.0.1:8000/openapi.json | head
```

Если оба worker-а `active`, Docker-контейнеры `Up`, а health endpoints отвечают `200`, основная система восстановлена.

## Важное ограничение

`bootstrap.sh` рассчитан на эталонную архитектуру AI6: 6× P104-100 и два независимых 3-GPU Qwen3-Coder worker-а. Если количество GPU, модель или архитектура поменяются, сначала запускайте `check` и задавайте нужные переменные окружения вместо слепого `all`.
