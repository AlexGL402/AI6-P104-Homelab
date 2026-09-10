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
- доступность llama workers на `8081` и `8082`
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


## Управление llama workers из dashboard

В карточке `llama workers` доступны кнопки **Start / Stop / Restart** для каждого worker-а и переключатели профиля:

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
