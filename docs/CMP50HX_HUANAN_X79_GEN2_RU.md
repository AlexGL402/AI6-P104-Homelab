# CMP 50HX: PCIe Gen2 x4 на Huanan X79 v2.49

Проверенная конфигурация для Huanan X79 v2.49 с модифицированным AMI BIOS (Mod Andrey1986) и NVIDIA CMP 50HX `10de:1e09`.

## Проверенный результат

На тестовой системе удалось получить:

```text
GPU endpoint:
LnkCap:  Speed 5GT/s, Width x16
LnkSta:  Speed 5GT/s, Width x4 (downgraded)
LnkCtl2: Target Link Speed: 5GT/s

CPU root port:
LnkCap:  Speed 8GT/s, Width x16
LnkSta:  Speed 5GT/s, Width x4
LnkCtl2: Target Link Speed: 5GT/s
```

То есть фактический режим — **PCIe 2.0 x4**.

## Железо и ПО

Проверено на:

- Huanan X79, board version `v2.49`
- AMI BIOS `4.6.5`, дата `04/25/2019`
- BIOS serial string: `Mod Andrey1986`
- NVIDIA CMP 50HX 10 GB, PCI ID `10de:1e09`
- Ubuntu 24.04
- kernel `6.8.0-139-generic`
- NVIDIA open kernel module `610.43.03`
- patched `cmp50hx-unlock`
- `cmp50hx-gen2.service`

Upstream project: https://github.com/xrip/cmp50hx-unlock

## Главное: BIOS должен быть принудительно выставлен в Gen2

На этой Huanan X79 с `Auto` CMP 50HX стартовала как Gen1, даже когда CPU root port сам умеет Gen3.

В BIOS нужно выставить:

```text
Target Link Speed = Force to 5.0 GT/s
```

или эквивалентный пункт:

```text
PCIe Link Speed = Gen2
```

Не оставлять `Auto` и не выбирать `2.5 GT/s`.

После этого root port должен показывать:

```text
LnkCtl2: Target Link Speed: 5GT/s
```

Проверка:

```bash
sudo lspci -s 00:03.0 -vv | grep -E 'LnkCap:|LnkSta:|LnkCtl2:'
```

BDF root port зависит от слота; на тестовой системе CMP 50HX была за `00:03.0`, endpoint — `04:00.0`.

## Параметр драйвера

Рабочая конфигурация:

```text
options nvidia cmp50_rebar_size=8 cmp50_gen2_adopt=1
```

Файл:

```bash
/etc/modprobe.d/cmp50hx-unlock.conf
```

Установка:

```bash
echo 'options nvidia cmp50_rebar_size=8 cmp50_gen2_adopt=1' | \
sudo tee /etc/modprobe.d/cmp50hx-unlock.conf

sudo update-initramfs -u -k 6.8.0-139-generic
```

После этого нужен полный cold boot:

```bash
sudo poweroff
```

Обесточить систему на 10–15 секунд и включить снова.

## Что должно быть в dmesg

При корректном BIOS Gen2 ранний драйверный путь показывает, что capability уже Gen2:

```text
CMP50_GEN2: POLICY_PASS phase=post-booter ... CAP=...02 ... LC2=...02 ... PL=00240032
CMP50_GEN2: ALREADY_GEN2 phase=gsp-ready ... CAP=...02 LC2=...02
CMP50_GEN2: ADOPT_PASS mode=normal ...
```

На тестовой Huanan kernel-path успевал вывести:

```text
CMP50_GEN2: RETRAIN_FAIL
```

хотя это не было конечным состоянием. Чуть позже systemd-сервис успешно завершал retrain:

```text
cmp50hx-gen2: 0000:04:00.0: card unlocked (TLS=2), firing retrain via 0000:00:03.0 (link now Gen1)
cmp50hx-gen2: 0000:04:00.0: PASS, link at 5.0 GT/s PCIe
cmp50hx-gen2: all CMP 50HX cards at their target generation
```

Поэтому итог нужно подтверждать не одной строкой `RETRAIN_FAIL`, а текущим `lspci` и журналом сервиса.

## Финальная проверка

Для GPU:

```bash
sudo lspci -s 04:00.0 -vv | grep -E 'LnkCap:|LnkSta:|LnkCtl2:'
```

Для root port:

```bash
sudo lspci -s 00:03.0 -vv | grep -E 'LnkCap:|LnkSta:|LnkCtl2:'
```

Ожидается:

```text
GPU:
LnkCap:  Speed 5GT/s, Width x16
LnkSta:  Speed 5GT/s, Width x4 (downgraded)
LnkCtl2: Target Link Speed: 5GT/s

ROOT:
LnkCap:  Speed 8GT/s, Width x16
LnkSta:  Speed 5GT/s, Width x4
LnkCtl2: Target Link Speed: 5GT/s
```

Проверка сервиса:

```bash
systemctl status cmp50hx-gen2.service --no-pager -l
sudo journalctl -u cmp50hx-gen2.service -b --no-pager | tail -50
```

Для oneshot-сервиса состояние `inactive (dead)` после успешного выполнения нормально, если есть:

```text
status=0/SUCCESS
PASS, link at 5.0 GT/s PCIe
all CMP 50HX cards at their target generation
```

## Почему настройка BIOS обязательна именно здесь

До изменения BIOS наблюдалось:

```text
GPU LnkCap:  2.5GT/s
GPU LnkCtl2: 2.5GT/s
ROOT LnkCtl2: 2.5GT/s
```

Перестановка CMP 50HX между двумя CPU PCIe x16 слотами не помогала: ограничение следовало за картой.

После принудительного `5.0 GT/s` в BIOS:

```text
GPU LnkCap:  5GT/s
GPU LnkCtl2: 5GT/s
ROOT LnkCtl2: 5GT/s
```

и `cmp50hx-gen2.service` смог завершить retrain до Gen2 x4.

## Важные замечания

- Не использовать Link Disable как способ retrain: upstream-аудит `cmp50hx-unlock` отмечает риск потери GPU/Xid на этом пути.
- UEFI/EFI unlock в upstream используется для других ранних unlock-механизмов; текущий PCIe Gen2 path выполняется из Linux.
- Для второй CMP 50HX BIOS-порт второго слота также следует принудительно выставить в `5.0 GT/s / Gen2`.
- `Width x4` для этой CMP 50HX в данной конфигурации ожидаем; этот рецепт меняет скорость Gen1 -> Gen2, а не количество линий.

## Проверка пропускной способности

До Gen2 x4 типичный предел на этой карте — около 0.85 GB/s в одну сторону. Upstream `cmp50hx-unlock` для Gen2 x4 показывает примерно 1.70–1.71 GB/s H2D/D2H.

Локальный результат для Huanan X79 следует записать отдельно после CUDA bandwidth-теста.
