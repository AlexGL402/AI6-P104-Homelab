# AI6: клонирование Ubuntu на отдельный SSD

Практическая инструкция по созданию загрузочной копии AI6 на другой SSD без переноса больших моделей, кэшей и виртуальных окружений.

> **Осторожно:** этап разметки полностью стирает целевой диск. Перед запуском обязательно проверьте `lsblk`, модель, размер и serial целевого SSD.

## 1. Проверить исходный и целевой диски

```bash
lsblk -o NAME,SIZE,MODEL,SERIAL,TRAN,MOUNTPOINTS
```

В исходной процедуре AI6 системный диск был Kingston SATA с отдельными EFI, /boot и LVM-разделами, а целевой SSD подключался как `/dev/sdb`.

Ниже предполагается:

```bash
disk=/dev/sdb
dst=/mnt/ai6-clone
```

**Не копируйте эти значения вслепую.** Подставьте фактический целевой диск.

## 2. Разметить SSD: GPT + EFI + ext4

Сначала убедитесь, что целевой диск не смонтирован:

```bash
lsblk "$disk"
```

Затем:

```bash
sudo parted -s "$disk" mklabel gpt
sudo parted -s "$disk" mkpart ESP fat32 1MiB 513MiB
sudo parted -s "$disk" set 1 esp on
sudo parted -s "$disk" mkpart primary ext4 513MiB 100%
sudo partprobe "$disk"
sudo udevadm settle

sudo mkfs.vfat -F32 -n AI6EFI "${disk}1"
sudo mkfs.ext4 -m 1 -L AI6ROOT "${disk}2"

sudo mkdir -p "$dst"
sudo mount "${disk}2" "$dst"
sudo mkdir -p "$dst/boot/efi"
sudo mount "${disk}1" "$dst/boot/efi"
```

Проверить UUID:

```bash
sudo blkid "${disk}1" "${disk}2"
```

Запишите UUID EFI и ROOT — они понадобятся ниже.

## 3. Первый rsync без моделей и тяжёлых окружений

```bash
sudo rsync -aHAXx --numeric-ids --info=progress2 \
  --exclude='/boot/*' \
  --exclude='/dev/*' --exclude='/proc/*' \
  --exclude='/sys/*' --exclude='/run/*' \
  --exclude='/tmp/*' --exclude='/mnt/*' \
  --exclude='/media/*' --exclude='/swap.img' \
  --exclude='/home/ai6/models/*' \
  --exclude='/home/ai6/ComfyUI/models/*' \
  --exclude='/home/ai6/ComfyUI/venv/*' \
  --exclude='/home/ai6/vllm-env/*' \
  --exclude='/home/ai6/vllm028-env/*' \
  --exclude='/home/ai6/.cache/*' \
  --exclude='/opt/nvidia/nsight-*' \
  / "$dst/"
```

Если `rsync` завершился кодом 24 из-за исчезнувших временных файлов, это допустимо для живой системы. Другие ошибки нужно проверить.

Скопировать `/boot`, не перезаписывая смонтированный EFI:

```bash
sudo rsync -aHAXx --numeric-ids --exclude='/efi/*' /boot/ "$dst/boot/"
sync
df -h "$dst"
```

## 4. Финальная синхронизация

Чтобы получить консистентную копию, остановите изменяющиеся службы. На AI6 использовались Docker/containerd и сервисы `ai6-monitor`, `ai6-model-gateway`, `ollama`.

Перед остановкой сохраните список работающих контейнеров:

```bash
mapfile -t containers < <(docker ps -q)
```

Остановите активные AI6-службы и Docker:

```bash
sudo systemctl stop ai6-monitor ai6-model-gateway ollama 2>/dev/null || true
(("${#containers[@]}")) && docker stop -t 60 "${containers[@]}"
sudo systemctl stop docker.socket docker.service containerd.service
```

Повторите `rsync` из шага 3, но добавьте `--delete`, затем:

```bash
sudo rsync -aHAXx --numeric-ids --delete --exclude='/efi/*' /boot/ "$dst/boot/"
sync
```

После синхронизации исходные службы можно запустить обратно:

```bash
sudo systemctl start containerd.service docker.service
(("${#containers[@]}")) && docker start "${containers[@]}"
sudo systemctl start ai6-monitor ai6-model-gateway ollama 2>/dev/null || true
```

## 5. Настроить клон

Подставьте реальные UUID:

```bash
root_uuid='<UUID ext4 раздела>'
efi_uuid='<UUID FAT32 EFI раздела>'
new_hostname='ai6-node'
```

Создайте новый `fstab`:

```bash
sudo cp -a "$dst/etc/fstab" "$dst/root/fstab.original"

sudo tee "$dst/etc/fstab" >/dev/null <<EOF
UUID=$root_uuid / ext4 defaults 0 1
UUID=$efi_uuid /boot/efi vfat umask=0077 0 1
EOF
```

Hostname:

```bash
echo "$new_hostname" | sudo tee "$dst/etc/hostname"
sudo sed -i '/^127\.0\.1\.1[[:space:]]/d' "$dst/etc/hosts"
echo "127.0.1.1 $new_hostname" | sudo tee -a "$dst/etc/hosts"
```

DHCP через systemd-networkd:

```bash
sudo rm -f "$dst/etc/netplan/"*.yaml "$dst/etc/netplan/"*.yml
sudo tee "$dst/etc/netplan/01-clone.yaml" >/dev/null <<'EOF'
network:
  version: 2
  renderer: networkd
  ethernets:
    wired:
      match:
        name: "e*"
      dhcp4: true
      optional: true
EOF

sudo chmod 600 "$dst/etc/netplan/01-clone.yaml"
sudo mkdir -p "$dst/etc/cloud/cloud.cfg.d"
echo 'network: {config: disabled}' | sudo tee "$dst/etc/cloud/cloud.cfg.d/99-clone-network.cfg"
```

Создать уникальный machine-id и SSH host keys:

```bash
sudo rm -f "$dst/etc/machine-id" "$dst/var/lib/dbus/machine-id"
dbus-uuidgen | sudo tee "$dst/etc/machine-id" >/dev/null
sudo ln -s /etc/machine-id "$dst/var/lib/dbus/machine-id"

sudo rm -f "$dst"/etc/ssh/ssh_host_*
sudo chroot "$dst" ssh-keygen -A
```

## 6. Подготовить chroot и загрузчик

```bash
for dir in dev proc sys; do
  sudo mount --rbind "/$dir" "$dst/$dir"
  sudo mount --make-rslave "$dst/$dir"
done
```

Для клона AI6 использовались:

```bash
echo RESUME=none | sudo tee "$dst/etc/initramfs-tools/conf.d/resume"
echo MODULES=most | sudo tee "$dst/etc/initramfs-tools/conf.d/clone-modules"
echo GRUB_DISABLE_OS_PROBER=true | sudo tee "$dst/etc/default/grub.d/99-clone.cfg"

sudo chroot "$dst" update-initramfs -u -k all

sudo chroot "$dst" grub-install \
  --target=x86_64-efi \
  --efi-directory=/boot/efi \
  --bootloader-id=ubuntu \
  --removable --no-nvram

sudo chroot "$dst" update-grub
```

Проверить fallback UEFI loader:

```bash
test -s "$dst/boot/efi/EFI/BOOT/BOOTX64.EFI" && echo "UEFI fallback OK"
```

## 7. Безопасно отключить SSD

Сначала размонтировать bind mounts:

```bash
sudo umount -R "$dst/dev" 2>/dev/null || true
sudo umount -R "$dst/proc" 2>/dev/null || true
sudo umount -R "$dst/sys" 2>/dev/null || true
```

Затем:

```bash
sudo sync
sudo umount "$dst/boot/efi"
sudo umount "$dst"
echo "SSD можно отключить"
```

Не вынимайте SSD, если `umount` завершился ошибкой.

## 8. Первая загрузка клона

В BIOS/UEFI выберите UEFI-загрузку с нового SSD.

Проверить:

```bash
hostname
hostname -I
lsblk -f
nvidia-smi
systemctl is-active ai6-monitor docker
```

Логин `ai6` и его пароль при таком клонировании сохраняются от исходной системы.

## 9. Если после клонирования появляется только `grub>`

В нашем случае разделы определились как:

- `(hd0,gpt1)` — EFI;
- `(hd0,gpt2)` — корневая Ubuntu.

Разовая загрузка:

```text
set root=(hd0,gpt2)
set prefix=(hd0,gpt2)/boot/grub
insmod normal
normal
```

После загрузки Ubuntu **обязательно** закрепить GRUB. Сначала проверить:

```bash
lsblk -o NAME,SIZE,FSTYPE,FSVER,MOUNTPOINTS,PARTUUID
mount | grep -i efi
[ -d /sys/firmware/efi ] && echo UEFI || echo LEGACY
```

Если EFI уже смонтирован в `/boot/efi` и система действительно загружена в UEFI:

```bash
sudo grub-install --target=x86_64-efi \
  --efi-directory=/boot/efi \
  --bootloader-id=ubuntu \
  --recheck

sudo update-grub
sudo efibootmgr -v
```

Успешный результат содержит:

```text
Installation finished. No error reported.
```

После этого в нашем случае появилась постоянная EFI-запись `Ubuntu -> \\EFI\\ubuntu\\shimx64.efi`, и клон начал загружаться самостоятельно.

## Примечание по BTCpro

Клон, перенесённый на ASRock H510 Pro BTC+, был переименован:

```bash
sudo hostnamectl set-hostname BTCpro
sudo sed -i 's/\bai6-node\b/BTCpro/g' /etc/hosts
```

После reboot приглашение стало:

```text
ai6@BTCpro:~$
```

На этой машине GRUB был окончательно восстановлен обычным UEFI `grub-install --recheck` уже после первой ручной загрузки.


## BTCpro: рабочая конфигурация CMP 50HX (2026-09-26)

Проверенная конфигурация BTCpro после переноса клона на ASRock H510 Pro BTC+:

- Ubuntu 24.04.5 LTS, hostname `BTCpro`.
- Рабочее ядро для CMP stack: `6.8.0-139-generic`.
- NVIDIA KMD/UMD: `610.43.03`, CUDA UMD 13.3.
- Две CMP 50HX 10 GB (`10de:1e09`).
- GPU0 `0000:01:00.0`: PCIe **Gen2 x4** (5.0 GT/s x4).
- GPU1 `0000:02:00.0`: PCIe **Gen2 x1** (5.0 GT/s x1).

### Gen2 unlock

На BTCpro используется patched module set из `cmp50hx-unlock`. Активный модуль:

```text
/lib/modules/6.8.0-139-generic/updates/forge-cmp/nvidia.ko
```

Он должен содержать параметр:

```text
cmp50_gen2_adopt
```

Конфигурация:

```text
/etc/modprobe.d/cmp50hx-gen2.conf:
options nvidia cmp50_gen2_adopt=1
```

После изменения модулей/параметров:

```bash
sudo depmod -a 6.8.0-139-generic
sudo update-initramfs -u -k 6.8.0-139-generic
```

Проверка:

```bash
cat /sys/module/nvidia/parameters/cmp50_gen2_adopt
sudo dmesg | grep -Ei 'CMP50_GEN2|ADOPT_|RETRAIN'
```

На проверенном cold boot обе карты получили `ADOPT_PASS`. GPU0 автоматически завершил `RETRAIN_PASS` и поднялся в Gen2 x4. GPU1 получил Gen2 capability, но первая ранняя попытка retrain завершилась `RETRAIN_FAIL`; повторный retrain upstream-порта `0000:00:1c.0` успешно поднял его до Gen2 x1.

Старый `cmp50hx-gen2.service` отключён:

```bash
sudo systemctl disable --now cmp50hx-gen2.service
```

Для поздней автодоводки линка установлен `cmp50hx-gen2-fixup.service`. Он ждёт после загрузки, пропускает карты уже на 5.0 GT/s или выше и делает стандартный PCIe retrain upstream-порта только для CMP 50HX, оставшейся на Gen1. Sysfs сообщает скорость как строку `5.0 GT/s PCIe`, поэтому проверка должна принимать шаблон `"5.0 GT/s"*`, а не только точное `"5.0 GT/s"`.

Проверка итоговых линков:

```bash
for d in 01:00.0 02:00.0; do
    echo "===== $d ====="
    sudo lspci -vv -s "$d" | grep -E 'LnkCap:|LnkSta:'
done
```

Рабочий результат:

```text
01:00.0: LnkCap Speed 5GT/s, LnkSta Speed 5GT/s Width x4
02:00.0: LnkCap Speed 5GT/s, LnkSta Speed 5GT/s Width x1
```

### P-state / idle power

CMP 50HX сама остаётся в P0 при 0% GPU load. На BTCpro наблюдалось около 60–64 W на карту в простое при памяти 7000 MHz и ядре около 2 GHz; power limit выставлен 170 W.

В проекте `cmp50hx-unlock` для этого предусмотрен отдельный `cmp-idle-governor`. Не включать исследовательский `05-cmp50-auto-pstate.patch`: upstream проекта оставляет его вне сборки, потому что он может удерживать карту в P8 и под нагрузкой. Для idle используется governor, который принудительно запрашивает P8 при простое и освобождает состояние при появлении нагрузки.

Установка уже присутствует в клоне; включение:

```bash
sudo systemctl enable --now cmp-idle-governor
```

Проверка:

```bash
systemctl status cmp-idle-governor --no-pager
journalctl -u cmp-idle-governor -n 50 --no-pager
nvidia-smi --query-gpu=index,pstate,clocks.gr,clocks.mem,power.draw,utilization.gpu --format=csv
```

Перед дальнейшими изменениями PCIe или NVIDIA stack сохранить текущую рабочую конфигурацию: Gen2 x4 + Gen2 x1 и рабочий `610.43.03` на ядре `6.8.0-139-generic`.
