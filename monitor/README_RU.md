# AI6 Host Monitor — установка

Этот сервис работает **на Ubuntu-хосте**, а не внутри Open Terminal. Поэтому он видит реальные данные `nvidia-smi` по всем NVIDIA GPU и одновременно читает CPU/RAM хоста.

## Что показывает

- CPU usage, load average и температуру, если она доступна
- RAM usage
- disk usage
- network counters
- все NVIDIA GPU: utilization, temperature, power draw, power limit, VRAM, fan, clocks
- суммарную мощность всех GPU
- максимальную температуру GPU
- автоматически обнаруженные llama-server процессы (модель, GPU, ctx, PID, uptime, tok/s)
- ручное сохранение показания линии `12V` с мультиметра в CSV

Dashboard обновляется каждые 2 секунды.

## Быстрая установка

На AI6-хосте:

```bash
cd ~/AI6-P104-Homelab
git pull
sudo apt update
sudo apt install -y python3-venv
chmod +x monitor/install.sh
./monitor/install.sh
```

Установщик автоматически:

1. проверит `nvidia-smi` и наличие NVIDIA GPU;
2. создаст Python venv;
3. установит FastAPI, uvicorn и psutil;
4. создаст `/var/lib/ai6-monitor`;
5. сгенерирует systemd service под **текущего пользователя и фактический путь репозитория**;
6. включит автозапуск;
7. проверит `/health` и `/api/stats`;
8. выведет актуальный LAN URL.

Текущий LAN IP можно определить так:

```bash
ip route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if($i=="src") {print $(i+1); exit}}'
```

Dashboard:

```text
http://<AI6_LAN_IP>:8090/
```

API и Swagger:

```text
http://<AI6_LAN_IP>:8090/api/stats
http://<AI6_LAN_IP>:8090/docs
```

Адрес хоста не должен быть жёстко прописан: после перехода AI6 в другую LAN используйте новый адрес, выданный DHCP.

## Проверка

```bash
systemctl --no-pager --full status ai6-monitor
curl http://127.0.0.1:8090/health
curl http://127.0.0.1:8090/api/stats | python3 -m json.tool
```

Логи:

```bash
journalctl -u ai6-monitor -f
```

Перезапуск:

```bash
sudo systemctl restart ai6-monitor
```

## Тест БП по ступеням нагрузки

На dashboard есть поле `PSU 12V sample`.

Порядок теста:

1. Создать стабильную GPU-нагрузку.
2. Дождаться стабилизации `GPU total power`.
3. Мультиметром измерить +12 V на свободном PCIe/Molex-разъёме **этого же БП**.
4. Ввести, например, `12.06` в dashboard.
5. В `note` написать `100W`, `200W`, `beep starts` и т.п.
6. Нажать `Save sample`.

Сервис сохранит timestamp, 12 V, суммарную GPU power, max GPU temperature, CPU usage, RAM usage и note.

CSV:

```text
/var/lib/ai6-monitor/psu-test.csv
```

Посмотреть:

```bash
column -s, -t < /var/lib/ai6-monitor/psu-test.csv
```

или:

```bash
cat /var/lib/ai6-monitor/psu-test.csv
```

Пример серии измерений:

| Этап | GPU total | 12 V | Писк |
|---|---:|---:|---|
| idle | ~60 W | ... | нет |
| 1 | ~100 W | ... | ... |
| 2 | ~200 W | ... | ... |
| 3 | ~300 W | ... | ... |
| 4 | ~400 W | ... | ... |
| 5 | ~500 W | ... | ... |
| 6 | ~600 W | ... | ... |

Не вскрывайте БП и не измеряйте первичную/сетевую часть. Для этого теста достаточно внешнего низковольтного +12 V разъёма.

## Почему монитор не внутри Open Terminal

Open Terminal специально остаётся изолированным coding-container. Сейчас внутри него нет `nvidia-smi`. Пробрасывать GPU runtime в coding-container только ради мониторинга не нужно.

```text
Ubuntu host
  ├─ nvidia-smi -> NVIDIA GPUs
  ├─ psutil -> CPU/RAM/disk/network
  └─ AI6 Host Monitor :8090
          ├─ dashboard /
          ├─ /api/stats
          └─ /api/psu-sample

Open Terminal остаётся отдельным Docker-контейнером.
```

## Опциональный модуль Miner

Основное назначение AI6 Host Monitor — мониторинг и управление **AI/LLM-нагрузкой**. Модуль ForgeMiner/Pearl оставлен только как опциональный тест GPU и **по умолчанию отключён**.

В обычном режиме ничего настраивать не нужно: если `AI6_ENABLE_MINER` не задан, вкладка Miner и её API вообще не загружаются, а майнер не запускается.

Чтобы временно включить модуль вручную:

```bash
sudo systemctl edit ai6-monitor
```

Добавить:

```ini
[Service]
Environment=AI6_ENABLE_MINER=1
```

Затем:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ai6-monitor
```

Чтобы вернуться к обычному AI-only режиму, удалите override с `AI6_ENABLE_MINER=1` (или задайте `AI6_ENABLE_MINER=0`) и перезапустите `ai6-monitor`. Наличие исходника `ai6_miner.py` само по себе ничего не запускает.

## Безопасность

Порт `8090` слушает `0.0.0.0`, чтобы dashboard был доступен в LAN. В сервисе нет авторизации. Не пробрасывайте `8090` напрямую в Интернет; для внешнего доступа используйте VPN или reverse proxy с аутентификацией.


## Web Terminal в браузере

Для прямого shell-доступа с dashboard используется отдельный `ttyd` на порту `8091`.
Он работает от обычного пользователя Linux и защищён отдельной HTTP Basic авторизацией.

Установка:

```bash
cd ~/AI6-P104-Homelab
git pull
chmod +x monitor/install-web-terminal.sh
./monitor/install-web-terminal.sh
```

Скрипт сам установит `ttyd`, сгенерирует случайный пароль, создаст systemd service
`ai6-web-terminal` и выведет готовую LAN-ссылку.

После установки в AI6 Host Monitor появится блок **Web Terminal**:
- **Open Web Terminal** — открыть отдельной вкладкой;
- **Show / hide below** — встроить терминал прямо внизу страницы мониторинга.

По умолчанию URL имеет вид:

```text
http://<AI6_LAN_IP>:8091/
```

Проверить сервис:

```bash
systemctl status ai6-web-terminal --no-pager
```

Посмотреть сохранённые учётные данные:

```bash
sudo cat /etc/ai6-web-terminal.env
```

Остановить/запустить:

```bash
sudo systemctl stop ai6-web-terminal
sudo systemctl start ai6-web-terminal
```

**Важно:** порт 8091 даёт интерактивный shell на AI6. Не пробрасывайте его напрямую в Интернет.


## Auto-discovery llama workers

Список worker-ов больше не захардкожен: монитор находит все запущенные локальные
`llama-server` процессы через `psutil`.

### Как работает обнаружение

1. Монитор перебирает все процессы хоста (`psutil.process_iter`).
2. Процесс считается llama worker-ом, если `llama-server` встречается в имени
   процесса или в его командной строке.
3. Из командной строки парсятся аргументы в обоих форматах — `--flag value` и
   `--flag=value`. Понимаются флаги: `--port`/`-p`, `--model`/`-m`,
   `--alias`, `--ctx-size`/`-c`.
4. Если порт (`--port`/`-p`) не распознан, процесс игнорируется: без него нельзя
   опрашивать health и `/v1/models`.

### Что определяется у каждого worker-а

- **порт** — `--port`/`-p` из командной строки;
- **модель** — сначала живое имя модели с `GET /v1/models` (если worker в
  состоянии `ready`), иначе `--alias` или имя файла из `--model`/`-m`;
- **GPU** — из `CUDA_VISIBLE_DEVICES`/`NVIDIA_VISIBLE_DEVICES` процесса;
- **контекст** — `--ctx-size`/`-c`;
- **PID и uptime** — из `psutil` (`create_time`);
- **статус** — `active`/`ready`/`loading` по ответу health-check API;
- **скорость** — последние `tok/s` (gen и prompt) и размер prompt-батча, извлечённые
  из журнала: из `journalctl` у managed-сервисов, иначе из файла stdout процесса
  (`/proc/<pid>/fd/1`). Путь к этому файлу показывается в поле `log_path` API;
  если stdout не ведётся в файл (например, в TTY), `tok/s` недоступны;
- **managed** — есть ли у порта зарегистрированный systemd-сервис.

## Managed vs manual workers

- **Managed worker** — запущен как systemd-сервис AI6 (например, `ai6-llama-8081`),
  опознан по порту через `_service_for_port`. В карточке доступны кнопки
  **Start / Stop / Restart**, а логи берутся из `journalctl`.
- **Manual worker** — запущен вручную (например, в терминале). В карточке кнопки
  управления заменяются пометкой *Manual process • start/stop from terminal*;
  логи читаются из файла stdout процесса.

Оба типа одинаково отображаются в dashboard: модель, GPU, ctx, PID, uptime,
gen/prompt `tok/s`.

### Troubleshooting: worker не отображается

Если запущенный `llama-server` не появился в dashboard:

1. **Нет порта в командной строке** — обнаружение требует `--port`/`-p`.
   Запустите сервер явно: `llama-server ... --port 8081`.
2. **Процесс виден, но порт занят** — два сервера на одном порту невозможны;
   проверьте `ss -ltnp | grep 8081`.
3. **Процесс другого пользователя** — `psutil` может не прочитать `cmdline`
   чужого процесса (`AccessDenied`); запускать worker-ов и монитор нужно под
   одним пользователем или с соответствующими правами.
4. **Worker в состоянии loading** — карточка отображается, но статус будет
   `loading`, пока модель не загрузилась; имя модели появится после `ready`.
5. **Не видно tok/s** — убедитесь, что в логе есть строки llama.cpp вида
   `... tokens per second` (нужен хотя бы один запрос), и что stdout ведётся в
   файл (для manual-процессов) или сервис логгируется в journal (managed).

Проверить вручную:

```bash
ps -ef | grep llama-server
curl -s http://127.0.0.1:8081/health
curl -s http://127.0.0.1:8081/v1/models
```

## Управление llama workers из dashboard

В карточке `llama workers` для **managed** worker-ов доступны кнопки **Start / Stop / Restart** и переключатели профиля:

- `3+3` — текущий проверенный профиль: GPU `0,1,2` -> `8081`, GPU `3,4,5` -> `8082`;
- `2+2+2` — три worker-а: GPU `0,1` -> `8081`, GPU `2,3` -> `8082`, GPU `4,5` -> `8083`.

Установка безопасного helper-а управления:

```bash
cd ~/AI6-P104-Homelab
git pull
chmod +x monitor/install-worker-control.sh
./monitor/install-worker-control.sh
sudo systemctl restart ai6-monitor
```

Helper разрешает dashboard только ограниченные операции над llama worker services; полный root shell monitor-у не выдаётся.

### Важно про профиль 2+2+2

Текущий Qwen3-Coder 30B Q4_K занимает около 18.5 GB и **не помещается в 2×8 GB VRAM**. Поэтому `2+2+2` намеренно не запустится, пока не задана более компактная GGUF-модель.

Настройка:

```bash
sudo nano /etc/ai6-worker-profiles.env
```

Указать:

```text
MODEL_2GPU=/полный/путь/к/меньшей-модели.gguf
CTX_2GPU=32768
```

После этого кнопка `2+2+2` переключит систему на три независимых 2-GPU worker-а. Выбранный профиль сохраняется через systemd enable/disable и переживает reboot.

Вернуться на проверенный Qwen3-Coder 30B профиль можно кнопкой `3+3` или командой:

```bash
sudo /usr/local/sbin/ai6-workerctl profile 33
```
