# CMP 40HX: PCIe Gen2 x16 на Huanan X79 v2.49

Проверенная Linux-конфигурация для NVIDIA CMP 40HX (TU106) на Huanan X79 v2.49.

## Проверенный результат

Хост:

- Huanan X79 v2.49
- AMI BIOS 4.6.5, 2019-04-25
- Ubuntu Linux
- CMP 40HX 8 GB, TU106
- GPU BDF во время теста: `0000:03:00.0`
- CPU root port: `0000:00:02.0`

До unlock:

```text
GPU:
LnkCap: Speed 2.5GT/s, Width x16
LnkSta: Speed 2.5GT/s, Width x16
LnkCtl2: Target Link Speed: 2.5GT/s

ROOT:
LnkCap: Speed 8GT/s, Width x16
LnkSta: Speed 2.5GT/s, Width x16
LnkCtl2: Target Link Speed: 5GT/s
```

После GPU-side PL0 + TLS Gen2 + retrain root port:

```text
GPU:
LnkCap: Speed 5GT/s, Width x16
LnkSta: Speed 5GT/s, Width x16
LnkCtl2: Target Link Speed: 5GT/s

ROOT:
LnkCap: Speed 8GT/s, Width x16
LnkSta: Speed 5GT/s, Width x16
LnkCtl2: Target Link Speed: 5GT/s

nvidia-smi:
pcie.link.gen.current = 2
pcie.link.gen.max = 2
pcie.link.width.current = 16
pcie.link.width.max = 16
```

Итог: **PCIe 2.0 x16**.

## BIOS

Для CPU PCIe порта/слота с CMP40 выставить:

```text
Target Link Speed = Force to 5.0 GT/s
```

или эквивалентный `PCIe Link Speed = Gen2`.

После изменения BIOS сделать cold boot. На рабочей системе root port уже показывал:

```text
LnkCtl2: Target Link Speed: 5GT/s
```

## GPU-side регистры

Рабочий набор PL0 для TU106:

```text
BAR0+0x8872C = 0x00000006   # XVE_OVR
BAR0+0x8C040 = 0x80085800   # LINK_CONFIG_0
BAR0+0x8841C = 0xE0B42D00   # PRIV_MISC_1
BAR0+0x8C2C0 = 0x068731B3   # CYA_0
```

Перед записью проверяется `BOOT_0`; для тестовой карты:

```text
BOOT_0 = 0x166000A1
```

После записи все четыре регистра дали точный readback.

## Проверенный порядок

1. BIOS root port -> Gen2.
2. Записать четыре GPU-side регистра через BAR0.
3. Выставить `LNKCTL2 TLS=2` на GPU.
4. Выставить `LNKCTL2 TLS=2` на root port.
5. Выполнить Retrain Link на upstream/root port.
6. Проверить `current_link_speed`, `lspci` и `nvidia-smi`.

На проверенной Huanan первый retrain через `00:02.0` сразу поднял линк:

```text
Initial link: 2.5 GT/s PCIe
Retrain #1 via ROOT
link = 5.0 GT/s PCIe
*** GEN2 ACHIEVED ***
```

## Bandwidth

CUDA pinned-memory test после unlock:

| Transfer | 256 MiB | 512 MiB | 1024 MiB |
|---|---:|---:|---:|
| H2D | 5.83 GiB/s | 5.77 GiB/s | 5.78 GiB/s |
| D2H | 6.21 GiB/s | 6.21 GiB/s | 6.18 GiB/s |

Предыдущий локальный результат Gen1 x16 был около 3.12 GB/s в каждую сторону, поэтому фактическая пропускная способность почти удвоилась.

## Автоматизация

В `deploy/cmp40hx-gen2.py` находится версия скрипта, которая автоматически определяет upstream root port по sysfs и не привязана к BDF `00:02.0`.

Systemd unit: `deploy/cmp40hx-gen2.service`.

Установка:

```bash
sudo install -m 0755 deploy/cmp40hx-gen2.py /usr/local/sbin/cmp40hx-gen2
sudo install -m 0644 deploy/cmp40hx-gen2.service /etc/systemd/system/cmp40hx-gen2.service
sudo systemctl daemon-reload
sudo systemctl enable --now cmp40hx-gen2.service
```

Проверка:

```bash
systemctl status cmp40hx-gen2.service --no-pager -l
sudo lspci -s 03:00.0 -vv | grep -E 'LnkCap:|LnkSta:|LnkCtl2:'
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv
```

## Примечание

Этот рецепт меняет PCIe link policy/runtime state. Он не прошивает VBIOS. При cold boot состояние GPU-side возвращается и service должен применить unlock снова.
