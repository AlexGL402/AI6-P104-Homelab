# AI6 Installer USB

Это установщик для AI6 Host Monitor поверх Ubuntu Server 24.04.5 LTS.

## Что делает образ

- загружает официальный Ubuntu Server 24.04.5;
- запускает Ubuntu Autoinstall, но **оставляет интерактивными**:
  - выбор сети;
  - выбор диска;
  - имя пользователя/пароль;
- после установки включает SSH;
- на первом старте ставит базовые пакеты, Docker и рекомендованный NVIDIA compute driver;
- после появления `nvidia-smi` клонирует ветку `feature/cmp-tune-web`;
- ставит AI6 Host Monitor на порт `8090`;
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
7. Первый старт может занять несколько минут: ставится NVIDIA driver. Возможна ещё одна автоматическая перезагрузка.
8. После этого открыть:

```text
http://IP_МАШИНЫ:8090
```

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
