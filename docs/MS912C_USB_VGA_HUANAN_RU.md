# MacroSilicon MS912C USB→VGA на Huanan X79 / Ubuntu 24.04

Проверенная рабочая конфигурация для USB→VGA адаптера MacroSilicon:

```text
VID:PID: 534d:6021
Product: USB Display usb extscreen
chip id: 0x3 (MS912C)
USB: High-Speed 480 Mbit/s
порт видео: VGA (port_type = 2)
```

Тестовая система:

- Huanan X79
- Ubuntu 24.04
- kernel `6.8.0-139-generic`
- адаптер `534d:6021`
- монитор VGA
- NVIDIA CMP 50HX как compute-only GPU
- рабочий framebuffer USB-VGA: `/dev/fb1`

## Что не заработало

Сначала тестировался драйвер:

https://github.com/bambinounos/ms91xx-linux-drm

Он корректно определял MS912C, создавал `fb1`, включал VGA DAC и отправлял управляющие команды, но bulk endpoint 4 зависал.

В usbmon было видно, что передача на EP4 начиналась, но завершалась таймаутом. На разных USB-контроллерах (EHCI/xHCI), с chunk 64 KiB, 16 KiB и одним большим URB результат оставался нестабильным.

Старый драйвер не рекомендуется выгружать через `rmmod usbdisp_drm`: в upstream есть известная проблема teardown/use-after-free. Для переключения драйвера использовалась перезагрузка.

## Рабочий драйвер

Используется:

https://github.com/rhgndf/ms912x

Для kernel 6.8 подошла ветка `kernel-6.6`:

```bash
cd ~
git clone -b kernel-6.6 https://github.com/rhgndf/ms912x.git
cd ~/ms912x
git log -1 --oneline
```

Проверенный commit:

```text
061ba0c backport
```

Установка DKMS:

```bash
sudo dkms install .
```

Проверка:

```bash
dkms status
modinfo ms912x | head
```

Рабочий модуль:

```text
ms912x/0.1, 6.8.0-139-generic, x86_64: installed
```

## Отключение старого usbdisp

Чтобы `usbdisp_*` не перехватывал `534d:6021`:

```bash
sudo tee /etc/modprobe.d/ms912x-test.conf >/dev/null <<'EOF'
blacklist usbdisp_usb
blacklist usbdisp_drm
EOF

sudo update-initramfs -u
sudo reboot
```

После загрузки:

```bash
lsmod | grep -E 'ms912x|usbdisp'
lsusb -t
cat /proc/fb
```

Ожидается:

```text
Driver=ms912x
0 simpledrmdrmfb
1 ms912xdrmfb
```

## Критическая правка для MS912C: VGA mode ID

В upstream `ms912x` для 640×480@60 был указан mode ID `0x40`:

```c
MS912X_MODE(640, 480, 60, 0x40)
```

На нашем MS912C это давало неправильный аппаратный timing. Монитор показывал:

```text
67.5 kHz / 129 Hz
out of range
```

У родного MacroSilicon protocol для этого чипа 640×480 использует VIC/mode `0x01`.

Правка:

```bash
cd ~/ms912x

sed -i \
  's/MS912X_MODE(640, 480, 60, 0x40)/MS912X_MODE(640, 480, 60, 0x01)/' \
  ms912x_drv.c
```

Проверка:

```bash
grep -n '640, 480' ms912x_drv.c
```

Должно быть:

```c
MS912X_MODE(640, 480, 60, 0x01)
```

## Preferred mode 640×480

Если EDID не читается, upstream выбирает 1024×768 preferred. Для этого адаптера рабочим оказался 640×480.

Правка:

```bash
cd ~/ms912x

sed -i \
  's/drm_set_preferred_mode(connector, 1024, 768);/drm_set_preferred_mode(connector, 640, 480);/' \
  ms912x_connector.c
```

Проверка:

```bash
grep -n 'drm_set_preferred_mode' ms912x_connector.c
```

Должно быть:

```c
drm_set_preferred_mode(connector, 640, 480);
```

После обеих правок пересобрать DKMS:

```bash
sudo dkms remove ms912x/0.1 --all
sudo dkms install .
sudo reboot
```

## Проверка framebuffer

После перезагрузки:

```bash
cat /sys/class/drm/card2-VGA-1/modes
sudo fbset -fb /dev/fb1 -i
```

Проверенный рабочий результат:

```text
640x480

mode "640x480"
    geometry 640 480 640 480 32
    rgba 8/16,8/8,8/0,0/0
endmode

Name        : ms912xdrmfb
Size        : 1228800
LineLength  : 2560
```

Номер `card2` может измениться после перестановки устройств.

## Вывод Linux console на USB-VGA

Привязать tty1 к `fb1`:

```bash
sudo con2fbmap 1 1
sudo chvt 1
```

После этого console начинает обновлять USB framebuffer.

Проверка USB bulk через usbmon показала успешные завершения:

```text
S Bo:3:003:4 -115 272 = ...
C Bo:3:003:4 0 272
```

То есть endpoint 4 принимает кадры полностью, без прежних `-110` timeout.

## Мелкий кириллический шрифт

Для 640×480 удобнее маленький Terminus:

```bash
sudo setfont /usr/share/consolefonts/CyrAsia-Terminus12x6.psf.gz
```

При необходимости в `/etc/default/console-setup`:

```text
FONTFACE="Terminus"
FONTSIZE="6x12"
```

Затем:

```bash
sudo setupcon
```

## VGA Auto Adjust

После первого нормального 640×480 изображение было сильно смещено по горизонтали. Это оказался не framebuffer и не шрифт, а аналоговая геометрия VGA.

На мониторе нужно выполнить:

```text
AUTO / Auto Adjust
```

После автонастройки изображение стало нормальным.

## Итоговая рабочая конфигурация

```text
Driver:      rhgndf/ms912x
Branch:      kernel-6.6
Commit:      061ba0c
Kernel:      6.8.0-139-generic
VID:PID:     534d:6021
Chip:        MS912C
Connector:   VGA
Mode:        640x480
Mode/VIC ID: 0x01
Framebuffer: /dev/fb1
Console:     tty1 -> fb1
USB bulk:    OK
```

Ключевые локальные изменения относительно upstream:

1. `640x480@60`: mode ID `0x40 -> 0x01`.
2. preferred mode без EDID: `1024x768 -> 640x480`.
3. старый `usbdisp_usb/usbdisp_drm` заблокирован через modprobe blacklist.
4. после запуска VGA монитор один раз настроен через `AUTO`.

Это конфигурация, которая фактически дала нормальную Linux console через USB→VGA на тестовой Huanan X79.
