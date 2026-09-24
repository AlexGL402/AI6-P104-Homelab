# AI6 Installer USB

Это установщик для AI6 Host Monitor поверх Ubuntu Server 24.04.5 LTS.

## Что делает образ

- загружает официальный Ubuntu Server 24.04.5;
- запускает Ubuntu Autoinstall, но **оставляет интерактивными**:
  - выбор сети;
  - выбор диска;
  - имя пользователя/пароль;
- после установки включает SSH;
- на первом старте ставит базовые пакеты и Docker;
- если обнаружена CMP 50HX/CMP-платформа, **не разрешает Ubuntu автоматически поставить случайную актуальную ветку NVIDIA**;
- для CMP требуется зафиксированный рабочий драйвер **NVIDIA 610.43.03 + MIT/GPL Open Kernel Modules**;
- для не-CMP хоста сохраняется установка рекомендованного Ubuntu NVIDIA compute driver;
- после появления `nvidia-smi` клонирует ветку `feature/cmp-tune-web`;
- ставит AI6 Host Monitor на порт `8090`;
- автоматически поднимает Docker-стек Open WebUI на `3000` и Open Terminal на `8000`;
- создаёт локальные секреты `WEBUI_SECRET_KEY` и `OPEN_TERMINAL_API_KEY` в `deploy/.env`, если файла ещё нет;
- сохраняет данные Open WebUI и Open Terminal в Docker named volumes, чтобы конфигурация переживала restart/reboot;
- ставит ограниченные helper-команды для GPU power limit и reboot/poweroff;
- если `cmp-tune` уже установлен, подключает CMP Tune control.

Модели, ForgeMiner, кошельки, API-ключи и другие секреты **не вшиваются** в ISO.

## Сборка ISO

На Ubuntu:

```bash
sudo apt update
sudo apt install -y xorriso curl coreutils
cd ~/AI6-P104-Homelab
git fetch origin
git switch feature/ai6-installer
bash installer/build-ai6-iso.sh
```

Результат:

```text
dist/AI6-Ubuntu-24.04.5-amd64.iso
```

Официальный Ubuntu ISO (~3.8 GB) скачивается автоматически и проверяется по SHA256SUMS.

## Запись на флешку

Сначала посмотреть диски:

```bash
lsblk -o NAME,SIZE,MODEL,TRAN,MOUNTPOINTS
```

Затем:

```bash
sudo bash installer/write-usb.sh /dev/sdX
```

**Внимание:** `/dev/sdX` будет полностью стёрт. Скрипт требует ручного подтверждения.

16 GB флешки достаточно.

## Установка

1. Загрузиться с USB в UEFI/Legacy.
2. Выбрать обычный пункт установки Ubuntu Server.
3. Выбрать LAN/DHCP.
4. В разделе Storage внимательно выбрать системный SSD/NVMe.
5. Создать пользователя (например `ai6`) и пароль.
6. Дождаться установки и перезагрузки.
7. Для CMP installer остановится перед установкой драйвера и попросит поставить зафиксированный AI6 CMP driver **610.43.03**. Это сделано специально, чтобы Ubuntu не выбрала другую ветку вроде 595.
8. Для CMP установить драйвер:

```bash
cd ~
wget https://download.nvidia.com/XFree86/Linux-x86_64/610.43.03/NVIDIA-Linux-x86_64-610.43.03.run
chmod +x NVIDIA-Linux-x86_64-610.43.03.run
sudo ./NVIDIA-Linux-x86_64-610.43.03.run --dkms
```

В интерактивном NVIDIA installer выбрать:

- `MIT/GPL` / Open Kernel Modules;
- DKMS: `Yes`;
- X configuration: `No` для headless AI6.

После reboot снова выполнить:

```bash
cd ~/AI6-P104-Homelab
sudo bash installer/firstboot.sh
```

9. После этого доступны:

```text
AI6 Monitor:   http://IP_МАШИНЫ:8090
Open WebUI:    http://IP_МАШИНЫ:3000
Open Terminal: http://IP_МАШИНЫ:8000
```

Open WebUI использует конфигурацию из `deploy/docker-compose.yml` и постоянный Docker volume `open-webui`. Репозиторий также содержит AI6 helper-функции в `openwebui/functions/`. Аккаунты и база конкретной старой установки Open WebUI в Git не хранятся; для их полного клонирования нужен отдельный backup Docker volume/DB.

Проверка по SSH:

```bash
systemctl status ai6-firstboot --no-pager
systemctl status ai6-monitor --no-pager
nvidia-smi
hostname -I
```

Лог первого запуска:

```bash
sudo journalctl -u ai6-firstboot -b --no-pager
```

## Важное

Установщик не стирает диск без подтверждения в интерфейсе Ubuntu: Storage оставлен интерактивным специально.

Для CMP 50HX/P104 после установки можно отдельно добавить ForgeMiner, модели, vLLM/ComfyUI. Они намеренно не включены в базовый образ, чтобы USB оставался универсальным.


## CMP: почему не Ubuntu 595

AI6 CMP baseline зафиксирован на:

```text
NVIDIA driver/userspace: 610.43.03
Kernel modules: MIT/GPL Open
CMP unlock: Forge CMP или xrip — отдельным шагом после базового драйвера
```

Ubuntu 24.04 может автоматически рекомендовать другую текущую ветку, например 595. Для CMP firstboot теперь специально не запускает `ubuntu-drivers install`, чтобы не смешивать другой KMD/userspace с сохранённым рабочим CMP stack.

После reboot проверить:

```bash
nvidia-smi
modinfo -F version nvidia
modinfo -F license nvidia
modinfo -n nvidia
```

Ожидаемая базовая версия перед установкой CMP unlock: `610.43.03`.
