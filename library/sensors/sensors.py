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

# This file defines all supported hardware in virtual classes and their abstract methods to access sensors
# To be overriden by child sensors classes

from abc import ABC, abstractmethod
from typing import Tuple
import math


class Cpu(ABC):
    @staticmethod
    @abstractmethod
    def percentage(interval: float) -> float:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def frequency() -> float:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load() -> Tuple[float, float, float]:  # 1/5/15 min 평균 (%)
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def temperature() -> float:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def fan_percent(fan_name: str = None) -> float:
        raise NotImplementedError


class Gpu(ABC):
    @staticmethod
    @abstractmethod
    def stats() -> Tuple[float, float, float, float, float]:
        """
        반환 튜플:
          (GPU 로드 [%], GPU 메모리 상대 사용률 [%], GPU 실제 사용 메모리 (Mb),
           GPU 총 메모리 (Mb), GPU 온도 [°C])
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def fps() -> int:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def fan_percent() -> float:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def frequency() -> float:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        raise NotImplementedError

    # 기본 구현: 센서 이름을 부분 일치로 검색하여 각 도메인 값을 채움
    @staticmethod
    def utilization_domains() -> dict:
        """
        아래 도메인에 대해 GPU 부하를 반환합니다.
        
        - "GPU Core"
        - "GPU Memory Controller"
        - "GPU Video Engine"
        - "GPU Bus"

        기본 구현에서는 센서 이름에 해당 문자열이 포함되는지 여부를
        확인하도록 되어 있으며, 구체적인 센서 클래스에서 재정의할 수 있습니다.
        """
        domains = {
            "GPU Core": math.nan,
            "GPU Memory Controller": math.nan,
            "GPU Video Engine": math.nan,
            "GPU Bus": math.nan,
        }
        # 구체적인 센서 구현에서는 자체 센서들을 순회하며
        # 센서 이름에 "GPU Core" 등이 포함되어 있으면 해당 값을 업데이트하세요.
        return domains


class Memory(ABC):
    @staticmethod
    @abstractmethod
    def swap_percent() -> float:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def virtual_percent() -> float:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def virtual_used() -> int:  # 바이트 단위
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def virtual_free() -> int:  # 바이트 단위
        raise NotImplementedError


class Disk(ABC):
    @staticmethod
    @abstractmethod
    def disk_usage_percent() -> float:
        """
        디스크 사용률(%)을 반환합니다.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def disk_used() -> int:
        """
        사용 중인 디스크 용량(바이트 단위)을 반환합니다.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def disk_free() -> int:
        """
        사용 가능한 디스크 용량(바이트 단위)을 반환합니다.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def disk_read_count() -> int:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def disk_write_count() -> int:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def disk_read_bytes() -> int:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def disk_write_bytes() -> int:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def disk_read_time() -> int:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def disk_write_time() -> int:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def disk_busy_time() -> int:
        """
        지원되지 않는 경우 0을 반환합니다.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def disk_temperature() -> float:
        """
        디스크 온도를 섭씨(°C)로 반환합니다.
        지원되지 않으면 math.nan을 반환합니다.
        """
        raise NotImplementedError


class Net(ABC):
    @staticmethod
    @abstractmethod
    def stats(if_name, interval) -> Tuple[int, int, int, int]:
        """
        반환 튜플:
          (업로드 속도 (B/s), 업로드 총량 (B), 다운로드 속도 (B/s), 다운로드 총량 (B))
        """
        raise NotImplementedError
