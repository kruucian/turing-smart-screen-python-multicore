# turing-smart-screen-python - a Python system monitor and library for USB-C displays like Turing Smart Screen or XuanFang
# https://github.com/mathoudebine/turing-smart-screen-python/

# Copyright (C) 2021-2023  Matthieu Houdebine (mathoudebine)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# This file will use LibreHardwareMonitor.dll library to get hardware sensors
# Some metrics are still fetched by psutil when not available on LibreHardwareMonitor
# For Windows platforms only

import ctypes
import ctypes.wintypes as wintypes
import math
import os
import sys
import struct
from statistics import mean
from typing import Tuple

import clr  # Clr is from pythonnet package. Do not install clr package
import psutil
from win32api import *

import library.sensors.sensors as sensors
from library.log import logger

# Import LibreHardwareMonitor dll to Python
lhm_dll = os.getcwd() + '\\external\\LibreHardwareMonitor\\LibreHardwareMonitorLib.dll'
# noinspection PyUnresolvedReferences
clr.AddReference(lhm_dll)
# noinspection PyUnresolvedReferences
clr.AddReference(os.getcwd() + '\\external\\LibreHardwareMonitor\\HidSharp.dll')
# noinspection PyUnresolvedReferences
from LibreHardwareMonitor import Hardware

File_information = GetFileVersionInfo(lhm_dll, "\\")

ms_file_version = File_information['FileVersionMS']
ls_file_version = File_information['FileVersionLS']

logger.debug("Found LibreHardwareMonitorLib %s" % ".".join([str(HIWORD(ms_file_version)), str(LOWORD(ms_file_version)),
                                                            str(HIWORD(ls_file_version)),
                                                            str(LOWORD(ls_file_version))]))

if ctypes.windll.shell32.IsUserAnAdmin() == 0:
    logger.error(
        "Program is not running as administrator. Please run with admin rights or choose another HW_SENSORS option in "
        "config.yaml")
    try:
        sys.exit(0)
    except:
        os._exit(0)

handle = Hardware.Computer()
handle.IsCpuEnabled = True
handle.IsGpuEnabled = True
handle.IsMemoryEnabled = True
handle.IsMotherboardEnabled = True  # For CPU Fan Speed
handle.IsControllerEnabled = True  # For CPU Fan Speed
handle.IsNetworkEnabled = True
handle.IsStorageEnabled = True
handle.IsPsuEnabled = False
handle.Open()
for hardware in handle.Hardware:
    if hardware.HardwareType == Hardware.HardwareType.Cpu:
        logger.info("Found CPU: %s" % hardware.Name)
    elif hardware.HardwareType == Hardware.HardwareType.Memory:
        logger.info("Found Memory: %s" % hardware.Name)
    elif hardware.HardwareType == Hardware.HardwareType.GpuNvidia:
        logger.info("Found Nvidia GPU: %s" % hardware.Name)
    elif hardware.HardwareType == Hardware.HardwareType.GpuAmd:
        logger.info("Found AMD GPU: %s" % hardware.Name)
    elif hardware.HardwareType == Hardware.HardwareType.GpuIntel:
        logger.info("Found Intel GPU: %s" % hardware.Name)
    elif hardware.HardwareType == Hardware.HardwareType.Storage:
        logger.info("Found Storage: %s" % hardware.Name)
    elif hardware.HardwareType == Hardware.HardwareType.Network:
        logger.info("Found Network interface: %s" % hardware.Name)


def get_hw_and_update(hwtype: Hardware.HardwareType, name: str = None) -> Hardware.Hardware:
    for hardware in handle.Hardware:
        if hardware.HardwareType == hwtype:
            if (name and hardware.Name == name) or name is None:
                hardware.Update()
                return hardware
    return None


def get_gpu_name() -> str:
    # Determine which GPU to use, in case there are multiple : try to avoid using discrete GPU for stats
    hw_gpus = []
    for hardware in handle.Hardware:
        if hardware.HardwareType == Hardware.HardwareType.GpuNvidia \
                or hardware.HardwareType == Hardware.HardwareType.GpuAmd \
                or hardware.HardwareType == Hardware.HardwareType.GpuIntel:
            hw_gpus.append(hardware)

    if len(hw_gpus) == 0:
        # No supported GPU found on the system
        logger.warning("No supported GPU found")
        return ""
    elif len(hw_gpus) == 1:
        # Found one supported GPU
        logger.debug("Found one supported GPU: %s" % hw_gpus[0].Name)
        return str(hw_gpus[0].Name)
    else:
        # Found multiple GPUs, try to determine which one to use
        amd_gpus = 0
        intel_gpus = 0
        nvidia_gpus = 0

        gpu_to_use = ""

        # Count GPUs by manufacturer
        for gpu in hw_gpus:
            if gpu.HardwareType == Hardware.HardwareType.GpuAmd:
                amd_gpus += 1
            elif gpu.HardwareType == Hardware.HardwareType.GpuIntel:
                intel_gpus += 1
            elif gpu.HardwareType == Hardware.HardwareType.GpuNvidia:
                nvidia_gpus += 1

        logger.warning(
            "Found %d GPUs on your system (%d AMD / %d Nvidia / %d Intel). Auto identify which GPU to use." % (
                len(hw_gpus), amd_gpus, nvidia_gpus, intel_gpus))

        if nvidia_gpus >= 1:
            # One (or more) Nvidia GPU: use first available for stats
            gpu_to_use = get_hw_and_update(Hardware.HardwareType.GpuNvidia).Name
        elif amd_gpus == 1:
            # No Nvidia GPU, only one AMD GPU: use it
            gpu_to_use = get_hw_and_update(Hardware.HardwareType.GpuAmd).Name
        elif amd_gpus > 1:
            # No Nvidia GPU, several AMD GPUs found: try to use the real GPU but not the APU integrated in CPU
            for gpu in hw_gpus:
                if gpu.HardwareType == Hardware.HardwareType.GpuAmd:
                    for sensor in gpu.Sensors:
                        if sensor.SensorType == Hardware.SensorType.Load and str(sensor.Name).startswith("GPU Core"):
                            # Found load sensor for this GPU: assume it is main GPU and use it for stats
                            gpu_to_use = gpu.Name
        else:
            # No AMD or Nvidia GPU: there are several Intel GPUs, use first available for stats
            gpu_to_use = get_hw_and_update(Hardware.HardwareType.GpuIntel).Name

        if gpu_to_use:
            logger.debug("This GPU will be used for stats: %s" % gpu_to_use)
        else:
            logger.warning("No supported GPU found (no GPU with load sensor)")

        return gpu_to_use


def get_net_interface_and_update(if_name: str) -> Hardware.Hardware:
    for hardware in handle.Hardware:
        if hardware.HardwareType == Hardware.HardwareType.Network and hardware.Name == if_name:
            hardware.Update()
            return hardware

    logger.warning("Network interface '%s' not found. Check names in config.yaml." % if_name)
    return None


class Cpu(sensors.Cpu):
    @staticmethod
    def percentage(interval: float) -> float:
        cpu = get_hw_and_update(Hardware.HardwareType.Cpu)
        for sensor in cpu.Sensors:
            if sensor.SensorType == Hardware.SensorType.Load and str(sensor.Name).startswith(
                    "CPU Total") and sensor.Value is not None:
                return float(sensor.Value)

        logger.error("CPU load cannot be read")
        return math.nan

    @staticmethod
    def frequency() -> float:
        frequencies = []
        cpu = get_hw_and_update(Hardware.HardwareType.Cpu)
        try:
            for sensor in cpu.Sensors:
                if sensor.SensorType == Hardware.SensorType.Clock:
                    # Keep only real core clocks, ignore effective core clocks
                    if "Core #" in str(sensor.Name) and "Effective" not in str(
                            sensor.Name) and sensor.Value is not None:
                        frequencies.append(float(sensor.Value))

            if frequencies:
                # Take mean of all core clock as "CPU clock" (as it is done in Windows Task Manager Performance tab)
                return mean(frequencies)
        except:
            pass

        # Frequencies reading is not supported on this CPU
        return math.nan

    @staticmethod
    def load() -> Tuple[float, float, float]:  # 1 / 5 / 15min avg (%):
        # Get this data from psutil because it is not available from LibreHardwareMonitor
        return psutil.getloadavg()

    @staticmethod
    def temperature() -> float:
        cpu = get_hw_and_update(Hardware.HardwareType.Cpu)
        try:
            # By default, the average temperature of all CPU cores will be used
            for sensor in cpu.Sensors:
                if sensor.SensorType == Hardware.SensorType.Temperature and str(sensor.Name).startswith(
                        "Core Average") and sensor.Value is not None:
                    return float(sensor.Value)
            # If not available, the max core temperature will be used
            for sensor in cpu.Sensors:
                if sensor.SensorType == Hardware.SensorType.Temperature and str(sensor.Name).startswith(
                        "Core Max") and sensor.Value is not None:
                    return float(sensor.Value)
            # If not available, the CPU Package temperature (usually same as max core temperature) will be used
            for sensor in cpu.Sensors:
                if sensor.SensorType == Hardware.SensorType.Temperature and str(sensor.Name).startswith(
                        "CPU Package") and sensor.Value is not None:
                    return float(sensor.Value)
            # Otherwise any sensor named "Core..." will be used
            for sensor in cpu.Sensors:
                if sensor.SensorType == Hardware.SensorType.Temperature and str(sensor.Name).startswith(
                        "Core") and sensor.Value is not None:
                    return float(sensor.Value)
        except:
            pass

        return math.nan

    @staticmethod
    def fan_percent(fan_name: str = None) -> float:
        mb = get_hw_and_update(Hardware.HardwareType.Motherboard)
        try:
            for sh in mb.SubHardware:
                sh.Update()
                for sensor in sh.Sensors:
                    if sensor.SensorType == Hardware.SensorType.Control and "#2" in str(
                            sensor.Name) and sensor.Value is not None:  # Is Motherboard #2 Fan always the CPU Fan ?
                        return float(sensor.Value)
        except:
            pass

        # No Fan Speed sensor for this CPU model
        return math.nan


class Gpu(sensors.Gpu):
    # 수정된 get_gpu_to_use 함수: 이름 비교 없이 GPU를 순서대로 반환
    @classmethod
    def get_gpu_to_use(cls):
        gpu = get_hw_and_update(Hardware.HardwareType.GpuNvidia)
        if gpu:
            logger.debug("Using Nvidia GPU: %s", gpu.Name)
            return gpu
        gpu = get_hw_and_update(Hardware.HardwareType.GpuAmd)
        if gpu:
            logger.debug("Using AMD GPU: %s", gpu.Name)
            return gpu
        gpu = get_hw_and_update(Hardware.HardwareType.GpuIntel)
        if gpu:
            logger.debug("Using Intel GPU: %s", gpu.Name)
            return gpu
        logger.warning("No GPU available for stats")
        return None

    @classmethod
    def stats(cls) -> Tuple[float, float, float, float, float]:
        """
        Returns a tuple:
          (GPU load [%], GPU memory usage [%], GPU used memory (Mb), GPU total memory (Mb), GPU temperature [°C])
        """
        gpu_to_use = cls.get_gpu_to_use()
        if gpu_to_use is None:
            return math.nan, math.nan, math.nan, math.nan, math.nan

        load = math.nan
        used_mem = math.nan
        total_mem = math.nan
        temp = math.nan

        for sensor in gpu_to_use.Sensors:
            if sensor.SensorType == Hardware.SensorType.Load and sensor.Value is not None:
                if str(sensor.Name).startswith("GPU Core"):
                    load = float(sensor.Value)
                elif str(sensor.Name).startswith("D3D 3D") and math.isnan(load):
                    load = float(sensor.Value)
            elif sensor.SensorType == Hardware.SensorType.SmallData and sensor.Value is not None:
                if str(sensor.Name).startswith("GPU Memory Used"):
                    used_mem = float(sensor.Value)
                elif str(sensor.Name).startswith("D3D") and str(sensor.Name).endswith("Memory Used") and math.isnan(used_mem):
                    used_mem = float(sensor.Value)
                elif str(sensor.Name).startswith("GPU Memory Total"):
                    total_mem = float(sensor.Value)
            elif sensor.SensorType == Hardware.SensorType.Temperature and sensor.Value is not None:
                if str(sensor.Name).startswith("GPU Core"):
                    temp = float(sensor.Value)
        memory_percentage = math.nan
        if not math.isnan(used_mem) and not math.isnan(total_mem) and total_mem != 0:
            memory_percentage = used_mem / total_mem * 100.0
        return load, memory_percentage, used_mem, total_mem, temp

    @classmethod
    def utilization_domains(cls) -> dict:
        domains = {
            "GPU Core": math.nan,
            "GPU Memory Controller": math.nan,
            "GPU Video Engine": math.nan,
            "GPU Bus": math.nan,
        }
        gpu_to_use = cls.get_gpu_to_use()
        if gpu_to_use is None:
            return domains

        for sensor in gpu_to_use.Sensors:
            if sensor.SensorType == Hardware.SensorType.Load and sensor.Value is not None:
                sensor_name = str(sensor.Name)
                if sensor_name in domains:
                    domains[sensor_name] = float(sensor.Value)
        return domains

    @classmethod
    def fps(cls) -> int:
        gpu_to_use = cls.get_gpu_to_use()
        if gpu_to_use is None:
            return -1
        try:
            for sensor in gpu_to_use.Sensors:
                if sensor.SensorType == Hardware.SensorType.Factor and "FPS" in str(sensor.Name) and sensor.Value is not None:
                    if int(sensor.Value) > 0:
                        cls.prev_fps = int(sensor.Value)
                    return cls.prev_fps
        except:
            pass
        return -1

    @classmethod
    def fan_percent(cls) -> float:
        gpu_to_use = cls.get_gpu_to_use()
        if gpu_to_use is None:
            return math.nan
        try:
            for sensor in gpu_to_use.Sensors:
                if sensor.SensorType == Hardware.SensorType.Control and sensor.Value is not None:
                    return float(sensor.Value)
        except:
            pass
        return math.nan

    @classmethod
    def frequency(cls) -> float:
        gpu_to_use = cls.get_gpu_to_use()
        if gpu_to_use is None:
            return math.nan
        try:
            for sensor in gpu_to_use.Sensors:
                if sensor.SensorType == Hardware.SensorType.Clock:
                    if "Core" in str(sensor.Name) and "Effective" not in str(sensor.Name) and sensor.Value is not None:
                        return float(sensor.Value)
        except:
            pass
        return math.nan

    @classmethod
    def is_available(cls) -> bool:
        return cls.get_gpu_to_use() is not None


class Memory(sensors.Memory):
    @staticmethod
    def swap_percent() -> float:
        memory = get_hw_and_update(Hardware.HardwareType.Memory)

        virtual_mem_used = math.nan
        mem_used = math.nan
        virtual_mem_available = math.nan
        mem_available = math.nan

        # Get virtual / physical memory stats
        for sensor in memory.Sensors:
            if sensor.SensorType == Hardware.SensorType.Data and str(sensor.Name).startswith(
                    "Virtual Memory Used") and sensor.Value is not None:
                virtual_mem_used = int(sensor.Value)
            elif sensor.SensorType == Hardware.SensorType.Data and str(sensor.Name).startswith(
                    "Memory Used") and sensor.Value is not None:
                mem_used = int(sensor.Value)
            elif sensor.SensorType == Hardware.SensorType.Data and str(sensor.Name).startswith(
                    "Virtual Memory Available") and sensor.Value is not None:
                virtual_mem_available = int(sensor.Value)
            elif sensor.SensorType == Hardware.SensorType.Data and str(sensor.Name).startswith(
                    "Memory Available") and sensor.Value is not None:
                mem_available = int(sensor.Value)

        # Compute swap stats from virtual / physical memory stats
        swap_used = virtual_mem_used - mem_used
        swap_available = virtual_mem_available - mem_available
        swap_total = swap_used + swap_available
        try:
            percent_swap = swap_used / swap_total * 100.0
        except:
            # No swap / pagefile disabled
            percent_swap = 0.0

        return percent_swap

    @staticmethod
    def virtual_percent() -> float:
        memory = get_hw_and_update(Hardware.HardwareType.Memory)
        for sensor in memory.Sensors:
            if sensor.SensorType == Hardware.SensorType.Load and str(sensor.Name).startswith(
                    "Memory") and sensor.Value is not None:
                return float(sensor.Value)

        return math.nan

    @staticmethod
    def virtual_used() -> int:  # In bytes
        memory = get_hw_and_update(Hardware.HardwareType.Memory)
        for sensor in memory.Sensors:
            if sensor.SensorType == Hardware.SensorType.Data and str(sensor.Name).startswith(
                    "Memory Used") and sensor.Value is not None:
                return int(sensor.Value * 1000000000.0)

        return 0

    @staticmethod
    def virtual_free() -> int:  # In bytes
        memory = get_hw_and_update(Hardware.HardwareType.Memory)
        for sensor in memory.Sensors:
            if sensor.SensorType == Hardware.SensorType.Data and str(sensor.Name).startswith(
                    "Memory Available") and sensor.Value is not None:
                return int(sensor.Value * 1000000000.0)

        return 0

# Windows API 상수들
GENERIC_READ = 0x80000000
OPEN_EXISTING = 3

# IOCTL 명령 (실제 값은 LibreHardwareMonitorLib 구현을 참고)
IOCTL_SCSI_PASS_THROUGH_DIRECT = 0x4D014  # 예시 값, 필요시 수정
# 예제에서는 NVMe 건강 정보(온도)를 가져오기 위한 IOCTL 코드 (예시)
IOCTL_NVM_HEALTH = 0x80CCE314  

# SCSI_PASS_THROUGH_DIRECT 구조체 정의 (간단 버전, 실제 구현은 더 복잡할 수 있음)
class SCSI_PASS_THROUGH_DIRECT(ctypes.Structure):
    _fields_ = [
        ("Length", wintypes.USHORT),
        ("ScsiStatus", wintypes.BYTE),
        ("PathId", wintypes.BYTE),
        ("TargetId", wintypes.BYTE),
        ("Lun", wintypes.BYTE),
        ("CdbLength", wintypes.BYTE),
        ("SenseInfoLength", wintypes.BYTE),
        ("DataIn", wintypes.BYTE),
        ("DataTransferLength", wintypes.ULONG),
        ("TimeOutValue", wintypes.ULONG),
        ("DataBuffer", ctypes.c_void_p),
        ("SenseInfoOffset", wintypes.ULONG),
        ("Cdb", ctypes.c_ubyte * 16)
    ]

class Disk(sensors.Disk):
    @staticmethod
    def disk_usage_percent() -> float:
        return psutil.disk_usage("/").percent

    @staticmethod
    def disk_used() -> int:  # In bytes
        return psutil.disk_usage("/").used

    @staticmethod
    def disk_free() -> int:  # In bytes
        return psutil.disk_usage("/").free

    @staticmethod
    def disk_read_count() -> int:
        return psutil.disk_io_counters(perdisk=False).read_count

    @staticmethod
    def disk_write_count() -> int:
        return psutil.disk_io_counters(perdisk=False).write_count

    @staticmethod
    def disk_read_bytes() -> int:
        return psutil.disk_io_counters(perdisk=False).read_bytes

    @staticmethod
    def disk_write_bytes() -> int:
        return psutil.disk_io_counters(perdisk=False).write_bytes

    @staticmethod
    def disk_read_time() -> int:
        return psutil.disk_io_counters(perdisk=False).read_time

    @staticmethod
    def disk_write_time() -> int:
        return psutil.disk_io_counters(perdisk=False).write_time

    @staticmethod
    def disk_busy_time() -> int:
        # busy_time은 Windows에서만 지원됩니다.
        return psutil.disk_io_counters(perdisk=False).busy_time

    @staticmethod
    def get_health_info(device=r"\\.\PhysicalDrive0"):
        """
        NVMe 드라이브에서 온도(temperature) 등 건강 정보를 가져옵니다.
        Windows 환경에서 DeviceIoControl를 사용합니다.
        device: 기본값은 PhysicalDrive0, 필요시 다른 드라이브로 변경.
        """
        # Windows API 함수 참조
        CreateFileW = ctypes.windll.kernel32.CreateFileW
        DeviceIoControl = ctypes.windll.kernel32.DeviceIoControl
        CloseHandle = ctypes.windll.kernel32.CloseHandle

        # 디바이스 열기
        handle = CreateFileW(device,
                            GENERIC_READ,
                            0,
                            None,
                            OPEN_EXISTING,
                            0,
                            None)
        if handle == wintypes.HANDLE(-1).value:
            return {"temperature": math.nan, "status": f"Failed to open device '{device}'."}

        # SCSI_PASS_THROUGH_DIRECT를 이용한 IOCTL을 위한 버퍼 준비
        buffer_size = 512
        data_buffer = (ctypes.c_ubyte * buffer_size)()
        sptd = SCSI_PASS_THROUGH_DIRECT()
        sptd.Length = ctypes.sizeof(SCSI_PASS_THROUGH_DIRECT)
        sptd.ScsiStatus = 0
        sptd.PathId = 0
        sptd.TargetId = 0
        sptd.Lun = 0
        sptd.CdbLength = 16
        sptd.SenseInfoLength = 0
        sptd.DataIn = 1  # Data from device
        sptd.DataTransferLength = buffer_size
        sptd.TimeOutValue = 2
        sptd.DataBuffer = ctypes.cast(data_buffer, ctypes.c_void_p)
        sptd.SenseInfoOffset = 0
        # 구성한 SCSI CDB: 아래 CDB는 예시이며 실제 NVMe 건강 정보 명령에 맞게 수정 필요
        cdb = [0xB5, 0xFE, 0x00, 0x05, 0x00] + [0x00] * 11
        for i in range(16):
            sptd.Cdb[i] = cdb[i]

        # IOCTL 호출에 사용할 출력 버퍼 (SCSI_PASS_THROUGH_DIRECT와 data_buffer가 포함된 경우)
        out_buffer = (ctypes.c_ubyte * (ctypes.sizeof(SCSI_PASS_THROUGH_DIRECT) + buffer_size))()
        # Copy sptd 구조체 into out_buffer
        ctypes.memmove(out_buffer, ctypes.byref(sptd), ctypes.sizeof(sptd))

        bytes_returned = wintypes.DWORD(0)
        ret = DeviceIoControl(handle,
                            IOCTL_NVM_HEALTH,
                            out_buffer,
                            len(out_buffer),
                            out_buffer,
                            len(out_buffer),
                            ctypes.byref(bytes_returned),
                            None)
        if not ret:
            CloseHandle(handle)
            return {"temperature": math.nan, "status": "DeviceIoControl failed."}

        # 온도 정보 추출: 예시로 out_buffer의 오프셋 위치 6~10을 사용 (빅 엔디안)
        # (실제 위치 및 처리 방식은 NVMe 드라이브 데이터 구조에 따라 달라짐)
        raw_temp = struct.unpack('>I', bytes(out_buffer[ctypes.sizeof(SCSI_PASS_THROUGH_DIRECT) + 6:
                                                        ctypes.sizeof(SCSI_PASS_THROUGH_DIRECT) + 10]))[0]
        # 온도 처리: 아래 처리는 예시 (해당 온도 값 변환식은 NVMe 스펙에 맞게 수정 필요)
        temperature = (raw_temp & 0xFFF0) >> 4

        CloseHandle(handle)
        return {"temperature": temperature, "status": "OK"}

class Net(sensors.Net):
    @staticmethod
    def stats(if_name, interval) -> Tuple[
        int, int, int, int]:  # up rate (B/s), uploaded (B), dl rate (B/s), downloaded (B)

        upload_rate = 0
        uploaded = 0
        download_rate = 0
        downloaded = 0

        if (if_name != ""):
            net_if = get_net_interface_and_update(if_name)
            if net_if is not None:
                for sensor in net_if.Sensors:
                    if sensor.SensorType == Hardware.SensorType.Data and str(sensor.Name).startswith(
                            "Data Uploaded") and sensor.Value is not None:
                        uploaded = int(sensor.Value * 1000000000.0)
                    elif sensor.SensorType == Hardware.SensorType.Data and str(sensor.Name).startswith(
                            "Data Downloaded") and sensor.Value is not None:
                        downloaded = int(sensor.Value * 1000000000.0)
                    elif sensor.SensorType == Hardware.SensorType.Throughput and str(sensor.Name).startswith(
                            "Upload Speed") and sensor.Value is not None:
                        upload_rate = int(sensor.Value)
                    elif sensor.SensorType == Hardware.SensorType.Throughput and str(sensor.Name).startswith(
                            "Download Speed") and sensor.Value is not None:
                        download_rate = int(sensor.Value)

        return upload_rate, uploaded, download_rate, downloaded
