"""Whole-machine CPU/RAM sampler backed by psutil.

cpu_percent(interval=None) is non-blocking and reports usage since the previous
call, so the runner's 1/s tick doubles as the sampling window; the constructor
makes a priming call so the very first real sample is meaningful.
"""

from __future__ import annotations

import re
import time

import psutil

from ..model import SysSample


def _short_cpu_name(raw: str) -> str:
    """注册表里是完整营销名(如 "AMD Ryzen 7 9700X 8-Core Processor"),
    精简成 "Ryzen 7 9700X" / "i9-9700X" 这种短名;认不出来就截断兜底。"""
    s = re.sub(r"\(R\)|\(TM\)|\(C\)", " ", raw)
    s = re.sub(r"\s+", " ", s).strip()
    toks = s.split()
    amd = "AMD" in raw.upper()
    model_re = re.compile(r"\d{4,5}[A-Za-z]{0,3}$")
    for i, tok in enumerate(toks):
        if model_re.search(tok):
            tier = ""
            if i > 0 and re.fullmatch(r"[3579]|R\d|i[3579]", toks[i - 1]):
                tier = toks[i - 1] + " "
            return ("Ryzen " if amd else "") + tier + tok
    return s[:24]


def _read_cpu_name() -> str:
    try:
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        try:
            raw, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        finally:
            winreg.CloseKey(key)
        return _short_cpu_name(str(raw))
    except Exception:  # noqa: BLE001
        try:
            import platform

            return _short_cpu_name(platform.processor()) or "CPU"
        except Exception:  # noqa: BLE001
            return "CPU"


class SysSampler:
    def __init__(self) -> None:
        self.available = False
        self.cpu_name = _read_cpu_name()
        try:
            psutil.cpu_percent(interval=None)
            psutil.virtual_memory()
            self.available = True
        except Exception:  # noqa: BLE001
            self.available = False

    def sample(self) -> SysSample | None:
        try:
            cpu = float(psutil.cpu_percent(interval=None))
            vm = psutil.virtual_memory()
            total_gb = vm.total / 1024 ** 3
            used_gb = (vm.total - vm.available) / 1024 ** 3
            return SysSample(
                cpu_name=self.cpu_name,
                cpu_pct=cpu,
                mem_pct=float(vm.percent),
                mem_used_gb=used_gb,
                mem_total_gb=total_gb,
                ts=time.time(),
            )
        except Exception:  # noqa: BLE001
            return None
