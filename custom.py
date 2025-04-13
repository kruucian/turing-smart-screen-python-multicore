#!/usr/bin/env python
"""
turing-smart-screen-python - a Python system monitor for USB-C displays.
(원본 라이센스 및 주석 생략)
"""

import glob
import os
import sys
import atexit
import locale
import platform
import signal
import subprocess
import time
import math
from pathlib import Path
from PIL import Image
from ping3 import ping
import psutil
import library.stats as stats
from library.log import logger
import library.scheduler as scheduler
from library.display import display
from library.sensors.sensors_librehardwaremonitor import Cpu, Gpu, Memory, Disk, Net

if platform.system() == 'Windows':
    import win32api
    import win32con
    import win32gui

try:
    import pystray
except:
    pass

MIN_PYTHON = (3, 9)
if sys.version_info < MIN_PYTHON:
    print("[ERROR] Python %s.%s or later is required." % MIN_PYTHON)
    try:
        sys.exit(0)
    except:
        os._exit(0)

MAIN_DIRECTORY = str(Path(__file__).parent.resolve()) + "/"
gpu_logged = False

disk_io_prev = None
net_io_prev = None
last_update_time = None
disk_sensor_prev = None

def wait_for_empty_queue(timeout: int = 5):
    logger.info("Waiting for all pending request to be sent to display (%ds max)..." % timeout)
    wait_time = 0
    while not scheduler.is_queue_empty() and wait_time < timeout:
        time.sleep(0.1)
        wait_time += 0.1
    logger.debug("(Waited %.1fs)" % wait_time)

def clean_stop(tray_icon=None):
    display.turn_off()
    scheduler.STOPPING = True
    wait_for_empty_queue(5)
    if tray_icon:
        tray_icon.visible = False
    try:
        sys.exit(0)
    except:
        os._exit(0)

def on_signal_caught(signum, frame=None):
    logger.info("Caught signal %d, exiting" % signum)
    clean_stop()

def on_configure_tray(tray_icon, item):
    logger.info("Configure from tray icon")
    subprocess.Popen(f'"{MAIN_DIRECTORY}{glob.glob("configure.*", root_dir=MAIN_DIRECTORY)[0]}"', shell=True)
    clean_stop(tray_icon)

def on_exit_tray(tray_icon, item):
    logger.info("Exit from tray icon")
    clean_stop(tray_icon)

def on_clean_exit(*args):
    logger.info("Program will now exit")
    clean_stop()

if platform.system() == "Windows":
    def on_win32_ctrl_event(event):
        if event in (win32con.CTRL_C_EVENT, win32con.CTRL_BREAK_EVENT, win32con.CTRL_CLOSE_EVENT):
            logger.debug("Caught Windows control event %s, exiting" % event)
            clean_stop()
        return 0

    def on_win32_wm_event(hWnd, msg, wParam, lParam):
        logger.debug("Caught Windows window message event %s" % msg)
        if msg == win32con.WM_POWERBROADCAST:
            if wParam == win32con.PBT_APMSUSPEND:
                logger.info("Computer is going to sleep, display will turn off")
                display.turn_off()
            elif wParam == win32con.PBT_APMRESUMEAUTOMATIC:
                logger.info("Computer is resuming from sleep, display will turn on")
                display.turn_on()
                display.display_static_images()
                display.display_static_text()
        else:
            logger.info("Program will now exit")
            clean_stop()

try:
    tray_icon = pystray.Icon(
        name='Turing System Monitor',
        title='Turing System Monitor',
        icon=Image.open(MAIN_DIRECTORY + "res/icons/monitor-icon-17865/64.png"),
        menu=pystray.Menu(
            pystray.MenuItem(text='Configure', action=on_configure_tray),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(text='Exit', action=on_exit_tray)
        )
    )
    if platform.system() != "Darwin":
        tray_icon.run_detached()
        logger.info("Tray icon has been displayed")
except:
    tray_icon = None
    logger.warning("Tray icon is not supported on your platform")

atexit.register(on_clean_exit)
signal.signal(signal.SIGINT, on_signal_caught)
signal.signal(signal.SIGTERM, on_signal_caught)
if os.name == 'posix':
    signal.signal(signal.SIGQUIT, on_signal_caught)
if platform.system() == "Windows":
    win32api.SetConsoleCtrlHandler(on_win32_ctrl_event, True)

logger.info("Initialize display")
display.initialize_display()
scheduler.QueueHandler()
display.display_static_images()
display.display_static_text()

def safe_height(h):
    """Ensure progress bar height is at least 1."""
    return h if h > 0 else 1

def display_cpu_bars_and_info():
    # per-core util은 기존대로 psutil로 가져옵니다.
    core_utils = psutil.cpu_percent(percpu=True)
    if len(core_utils) < 16:
        core_utils = (core_utils * ((16 // len(core_utils)) + 1))[:16]
    
    bar_width = 25
    bar_height = 45
    spacing = 5
    x_start = 10
    y_first = 10
    y_second = y_first + bar_height + 5
    info_x = x_start

    # 센서 라이브러리의 Cpu 함수를 사용하여 전체 CPU 정보를 가져옵니다.
    # percentage()는 interval (초)를 인자로 받습니다.
    overall_util = Cpu.percentage(1.0)
    freq_mhz = Cpu.frequency()
    # 주파수는 센서에서 MHz로 반환하므로 GHz로 변환합니다.
    freq = freq_mhz / 1000 if not math.isnan(freq_mhz) else 0
    avg_temp = Cpu.temperature()
    fan_speed = Cpu.fan_percent()
    if math.isnan(fan_speed):
        fan_speed = 0

    # 첫번째 행의 8개 수직 진행바를 그림
    for i in range(8):
        x = x_start + i * (bar_width + spacing)
        display.lcd.DisplayProgressBar_Vert(
            x, y_first,
            width=bar_width,
            height=bar_height,
            min_value=0,
            max_value=100,
            value=core_utils[i],
            bar_color=(255, 255, 255),  # changed to white
            bar_outline=False,
            background_image=None,  # transparent background
            background_color=(0,0,0)  # added background_color
        )
    
    # 두번째 행의 8개 수직 진행바를 그림
    for i in range(8):
        x = x_start + i * (bar_width + spacing)
        display.lcd.DisplayProgressBar_Vert(
            x, y_second,
            width=bar_width,
            height=bar_height,
            min_value=0,
            max_value=100,
            value=core_utils[i + 8],
            bar_color=(255, 255, 255),  # changed to white
            bar_outline=False,
            background_image=None,
            background_color=(0,0,0)  # added background_color
        )
    
    # 텍스트 정보 출력
    text1 = "{:.0f}%".format(overall_util)
    text2 = "{:.2f}GHz".format(freq)
    text3 = "{}°C".format(int(avg_temp))
    text4 = "{}rpm".format(int(fan_speed))
    display.lcd.DisplayText(text1, info_x, y_second + bar_height, font_size=12)
    display.lcd.DisplayText(text2, info_x + 60, y_second + bar_height, font_size=12)
    display.lcd.DisplayText(text3, info_x + 140, y_second + bar_height, font_size=12)
    display.lcd.DisplayText(text4, info_x + 200, y_second + bar_height, font_size=12)


def display_gpu_stats():
    x_start_metrics = 260
    y_bar_bottom = 10
    text_spacing = 5
    vbar_width = 50
    vbar_height = 45  # 변경: 그래프 높이를 45로 수정
    spacing = 5

    # 실제 구현체를 직접 사용 (sensors.Gpu 대신 Gpu)
    if Gpu.is_available():
        utilization = Gpu.utilization_domains()
        gpu_3d_load   = utilization.get("GPU Core", 0)
        gpu_copy_load = utilization.get("GPU Memory Controller", 0)
        gpu_ve_load   = utilization.get("GPU Video Engine", 0)
        gpu_bus_load  = utilization.get("GPU Bus", 0)
        gpu_stats_tuple = Gpu.stats()
        if gpu_stats_tuple:
            _, _, used_mem, total_mem, _ = gpu_stats_tuple
            gpu_vram_usage = used_mem / 1024.0
            gpu_vram_total = total_mem / 1024.0
        else:
            gpu_vram_usage = gpu_vram_total = 0
    else:
        gpu_3d_load = gpu_copy_load = gpu_ve_load = gpu_bus_load = 0
        gpu_vram_usage = gpu_vram_total = 0

    for i, (label, value) in enumerate(zip(
            ["3D", "Cpy", "Vid", "Bus"],
            [gpu_3d_load, gpu_copy_load, gpu_ve_load, gpu_bus_load])):
        x = x_start_metrics + i * (vbar_width + spacing)

        display.lcd.DisplayProgressBar_Vert(
            x, y_bar_bottom,
            width=vbar_width, height=vbar_height,
            min_value=0, max_value=100, value=value,
            bar_color=(255, 255, 255),  # white
            bar_outline=False,
            background_image=None,
            background_color=(0,0,0)  # added background_color
        )
        display.lcd.DisplayText(
            f"{label} {int(value)}%",
            x, y_bar_bottom + vbar_height,
            font_size=10
        )
    # Modified VRAM display section:
    available_width = display.lcd.get_width() - x_start_metrics
    vram_percent = int((gpu_vram_usage / 22.5) * 100)
    y_after_vertical = y_bar_bottom + vbar_height + spacing

    display.lcd.DisplayText(
        "VRAM {:.1f}/{:.1f}GB".format(gpu_vram_usage, gpu_vram_total),
        x_start_metrics, y_after_vertical + 15,
        font_size=10
    )
    display.lcd.DisplayProgressBar(
        x_start_metrics, y_after_vertical +30,
        width=available_width-5, height=10,
        min_value=0, max_value=100, value=vram_percent,
        bar_color=(255, 255, 255),  # white
        bar_outline=True, background_image=None,
        background_color=(0,0,0)  # added background_color
    )


def display_memory_stats():
    mem_used = psutil.virtual_memory().used / (1024 ** 3)
    mem_total = psutil.virtual_memory().total / (1024 ** 3)
    x_start = 260
    y_base = 100
    text_spacing = 15

    available_width = display.lcd.get_width() - x_start
    mem_percent = int((mem_used / mem_total) * 100) if mem_total else 1

    display.lcd.DisplayProgressBar(
        x_start, y_base + text_spacing,
        width=available_width-5, height=10,
        min_value=0, max_value=100, value=mem_percent,
        bar_color=(255, 255, 255),  # white
        bar_outline=True, background_image=None,
        background_color=(0,0,0)  # added background_color
    )
    display.lcd.DisplayText(
        "Memory {:.1f}/{:.1f}GB".format(mem_used, mem_total),
        x_start, y_base, font_size=10
    )

import math
import psutil
import platform
import time
# sensors 모듈과 디스크 클래스를 가져옵니다.
from library.sensors import sensors

def display_disk_and_net_stats():
    global disk_io_prev, net_io_prev, last_update_time, disk_sensor_prev

    disk = psutil.disk_usage("C:\\" if platform.system() == "Windows" else "/")
    disk_usage_percent = disk.percent
    disk_used = disk.used / (1024 ** 3)
    disk_total = disk.total / (1024 ** 3)

    current_time = time.monotonic()
    interval = current_time - last_update_time if last_update_time else 1
    last_update_time = current_time

    current_disk_io = psutil.disk_io_counters(perdisk=False)
    if disk_io_prev is not None:
        read_speed = (current_disk_io.read_bytes - disk_io_prev.read_bytes) / (1024**2) / interval
        write_speed = (current_disk_io.write_bytes - disk_io_prev.write_bytes) / (1024**2) / interval
    else:
        read_speed = write_speed = 0
    disk_io_prev = current_disk_io

    # 센서 기반 디스크 활성시간, 평균 응답시간 계산
    try:
        current_busy = sensors.Disk.disk_busy_time()
        current_read_time = sensors.Disk.disk_read_time()
        current_write_time = sensors.Disk.disk_write_time()
        current_read_count = sensors.Disk.disk_read_count()
        current_write_count = sensors.Disk.disk_write_count()
    except Exception:
        current_busy = current_read_time = current_write_time = 0
        current_read_count = current_write_count = 0

    if disk_sensor_prev is not None:
        delta_busy = current_busy - disk_sensor_prev['busy']
        active_time_percent = (delta_busy / (interval * 1000)) * 100

        delta_read_time = current_read_time - disk_sensor_prev['read_time']
        delta_write_time = current_write_time - disk_sensor_prev['write_time']
        delta_ops = ((current_read_count - disk_sensor_prev['read_count']) +
                     (current_write_count - disk_sensor_prev['write_count']))
        if delta_ops > 0:
            avg_response_time = (delta_read_time + delta_write_time) / delta_ops
        else:
            avg_response_time = 0.0
    else:
        active_time_percent = avg_response_time = 0.0

    disk_sensor_prev = {
        'busy': current_busy,
        'read_time': current_read_time,
        'write_time': current_write_time,
        'read_count': current_read_count,
        'write_count': current_write_count
    }

    # 네트워크 업/다운 속도 계산 (Mbps)
    current_net_io = psutil.net_io_counters()
    if net_io_prev is not None:
        # 전송된 바이트를 Mbps로 변환: (bytes * 8) / (10**6)
        net_up = (current_net_io.bytes_sent - net_io_prev.bytes_sent) * 8 / (10**6) / interval
        net_down = (current_net_io.bytes_recv - net_io_prev.bytes_recv) * 8 / (10**6) / interval
    else:
        net_up = net_down = 0.0
    net_io_prev = current_net_io

    x_start = 10
    x_start_bar = 140
    y_base = 180
    progress_bar_width = 140
    progress_bar_height = 15
    text_spacing = 5

    display.lcd.DisplayProgressBar(
        x_start_bar, y_base + text_spacing,
        width=progress_bar_width, height=progress_bar_height,
        min_value=0, max_value=100, value=disk_usage_percent,
        bar_color=(255, 255, 255),  # white
        bar_outline=True, background_image=None,
        background_color=(0,0,0)  # added background_color
    )
    display.lcd.DisplayText(
        "Disk {:.1f}/{:.1f}GB \n({:.1f}%, {:.1f}GB free)".format(
            disk_used, disk_total, disk_usage_percent, disk_total - disk_used),
        x_start, y_base, font_size=10
    )

    y_offset = y_base + progress_bar_height + text_spacing
    display.lcd.DisplayProgressBar(
        x_start, y_offset + text_spacing,
        width=progress_bar_width, height=progress_bar_height,
        min_value=0, max_value=3500, value=read_speed,
        bar_color=(255, 255, 255),  # white
        bar_outline=True, background_image=None,
        background_color=(0,0,0)  # added background_color
    )
    display.lcd.DisplayText(
        "Read: {:.1f}MB/s".format(read_speed),
        x_start, y_offset, font_size=10
    )

    y_offset += progress_bar_height + text_spacing
    display.lcd.DisplayProgressBar(
        x_start_bar, y_offset + text_spacing,
        width=progress_bar_width, height=progress_bar_height,
        min_value=0, max_value=2700, value=write_speed,
        bar_color=(255, 255, 255),  # white
        bar_outline=True, background_image=None,
        background_color=(0,0,0)  # added background_color
    )
    display.lcd.DisplayText(
        "Write: {:.1f}MB/s".format(write_speed),
        x_start, y_offset, font_size=10
    )

    y_offset += progress_bar_height + text_spacing
    display.lcd.DisplayProgressBar(
        x_start_bar, y_offset + text_spacing,
        width=progress_bar_width, height=progress_bar_height,
        min_value=0, max_value=100, value=active_time_percent,
        bar_color=(255, 255, 255),  # white
        bar_outline=True, background_image=None,
        background_color=(0,0,0)  # added background_color
    )
    display.lcd.DisplayText(
        "Active: {:.1f}%".format(active_time_percent),
        x_start, y_offset, font_size=10
    )

    y_offset += progress_bar_height + text_spacing
    display.lcd.DisplayProgressBar(
        x_start_bar, y_offset + text_spacing,
        width=progress_bar_width, height=progress_bar_height,
        min_value=0, max_value=100, value=avg_response_time,
        bar_color=(255, 255, 255),  # white
        bar_outline=True, background_image=None,
        background_color=(0,0,0)  # added background_color
    )
    display.lcd.DisplayText(
        "AvgResp: {:.1f} ms".format(avg_response_time),
        x_start, y_offset, font_size=10
    )

    x_start_net = 300
    bar_width_net = 100

    display.lcd.DisplayProgressBar(
        x_start_net, y_base + text_spacing,
        width=bar_width_net, height=progress_bar_height,
        min_value=0, max_value=1000, value=net_up,
        bar_color=(255, 255, 255),  # white
        bar_outline=True, background_image=None,
        background_color=(0,0,0)  # added background_color
    )
    display.lcd.DisplayText(
        "Up: {:.1f}Mbps".format(net_up),
        x_start_net, y_base, font_size=10
    )

    y_offset = y_base + progress_bar_height + text_spacing
    display.lcd.DisplayProgressBar(
        x_start_net, y_offset + text_spacing,
        width=bar_width_net, height=progress_bar_height,
        min_value=0, max_value=1000, value=net_down,
        bar_color=(255, 255, 255),  # white
        bar_outline=True, background_image=None,
        background_color=(0,0,0)  # added background_color
    )
    display.lcd.DisplayText(
        "Down: {:.1f}Mbps".format(net_down),
        x_start_net, y_offset, font_size=10
    )

    y_offset += progress_bar_height + text_spacing
    ping1 = ping("168.126.63.1", unit="ms") or 0
    ping2 = ping("1.1.1.1", unit="ms") or 0
    display.lcd.DisplayText(
        "KT DNS1 {}ms".format(int(ping1)),
        x_start_net, y_offset, font_size=15
    )
    display.lcd.DisplayText(
        "1.1.1.1 {}ms".format(int(ping2)),
        x_start_net, y_offset + 15, font_size=15
    )

if __name__ == "__main__":
    display_cpu_bars_and_info()
    display_gpu_stats()
    display_memory_stats()
    display_disk_and_net_stats()

    wait_for_empty_queue(10)
    logger.info("Starting system monitoring")

    scheduler.CPUPercentage(); time.sleep(0.25)
    scheduler.CPUFrequency(); time.sleep(0.25)
    scheduler.CPULoad(); time.sleep(0.25)
    scheduler.CPUTemperature(); time.sleep(0.25)
    scheduler.CPUFanSpeed(); time.sleep(0.25)
    if stats.Gpu.is_available():
        scheduler.GpuStats(); time.sleep(0.25)
    scheduler.MemoryStats(); time.sleep(0.25)
    scheduler.DiskStats(); time.sleep(0.25)
    scheduler.NetStats(); time.sleep(0.25)
    scheduler.PingStats(); time.sleep(0.25)

    import threading
    def update_display_loop():
        while True:
            display_cpu_bars_and_info()
            display_gpu_stats()
            display_memory_stats()
            display_disk_and_net_stats()
            wait_for_empty_queue(10)
            time.sleep(5)
    update_thread = threading.Thread(target=update_display_loop, daemon=True)
    update_thread.start()

    while True:
        time.sleep(1)