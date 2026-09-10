# AI6 Host Monitor — установка

Этот сервис работает **на Ubuntu-хосте**, а не внутри Open Terminal. Поэтому он видит реальные данные `nvidia-smi` по всем 6× P104-100 и одновременно читает CPU/RAM через `psutil`.

## Что показывает

- CPU usage и load average
- CPU temperature, если её отдаёт Linux sensors interface
- RAM usage
- disk usage
- network counters
- все NVIDIA GPU:
  - utilization
  - temperature
  - power draw
  - power limit
  - VRAM used/total
  - fan
  - graphics/memory clocks
- суммарную мощность всех GPU
- max GPU temperature
- доступность llama workers на `8081` и `8082`
- ручное сохранение показания линии `12V` с мультиметра в CSV

## Установка

```bash
cd ~/AI6-P104-Homelab
git pull

sudo apt update
sudo apt install -y python3-venv

cd monitor
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Проверить, что на **хосте** работает NVIDIA:

```bash
nvidia-smi
```

Должны быть видны все 6 P104-100.

## Первый ручной запуск

```bash
cd ~/AI6-P104-Homelab/monitor
.venv/bin/uvicorn ai6_monitor:app --host 0.0.0.0 --port 8090
```

С другого ПК открыть:

```text
http://10.36.1.164:8090/
```

API:

```text
http://10.36.1.164:8090/api/stats
http://10.36.1.164:8090/docs
```

Проверка локально:

```bash
curl http://127.0.0.1:8090/health
curl http://127.0.0.1:8090/api/stats
```

## Установка как systemd service

В репозитории есть `configs/ai6-monitor.service` с путями для пользователя `ai6`.

```bash
sudo mkdir -p /var/lib/ai6-monitor
sudo chown ai6:ai6 /var/lib/ai6-monitor

sudo cp ~/AI6-P104-Homelab/configs/ai6-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai6-monitor
```

Проверить:

```bash
systemctl --no-pager --full status ai6-monitor
curl http://127.0.0.1:8090/health
```

Логи:

```bash
journalctl -u ai6-monitor -f
```

## Тест БП по ступеням нагрузки

На dashboard есть поле `PSU 12V sample`.

Порядок теста:

1. Создать стабильную нагрузку на GPU.
2. Дождаться стабилизации `GPU total power`.
3. Мультиметром измерить +12V на свободном PCIe/Molex разъёме того же БП.
4. Ввести, например, `12.06` в dashboard.
5. В `note` написать `200W`, `400W`, `beep starts` и т.п.
6. Нажать `Save sample`.

Сервис одновременно сохранит:

- timestamp
- измеренные вручную 12V
- текущий суммарный GPU power
- max GPU temperature
- CPU usage
- RAM usage
- note

CSV находится здесь:

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

## Почему не внутри Open Terminal

Open Terminal специально остаётся изолированным coding-container. Сейчас внутри него нет `nvidia-smi`, поэтому отдавать ему GPU runtime только ради мониторинга не требуется.

Схема:

```text
Ubuntu host
  ├─ nvidia-smi -> 6× P104
  ├─ psutil -> CPU/RAM
  └─ AI6 Host Monitor :8090
          |
          +-> dashboard /
          +-> /api/stats
          +-> /api/psu-sample

Open Terminal остаётся отдельным контейнером.
```

## Важно

Порт `8090` сейчас слушает `0.0.0.0`, то есть доступен в LAN. Не пробрасывайте его в интернет без firewall/auth/reverse proxy.
