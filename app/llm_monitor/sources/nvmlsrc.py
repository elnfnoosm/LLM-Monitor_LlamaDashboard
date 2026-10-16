"""NVML device sampler with probe-once capability gating.

Fields the driver/GPU don't support (NVMLError_NotSupported / InvalidArgument)
are marked unsupported for the session and render as placeholders - they must
NOT count as source failures (spec: 能力门控的字段)."""

from __future__ import annotations

import time

from ..model import GpuSample

try:
    import pynvml  # provided by nvidia-ml-py
    _HAVE_PYNVML = True
except ImportError:  # pragma: no cover
    pynvml = None  # type: ignore
    _HAVE_PYNVML = False

# clocks event reason bit -> 人话
_THROTTLE_LABELS = [
    (getattr(pynvml, "NVML_CLOCKS_EVENT_REASON_HW_SLOWDOWN", 1 << 5), "HW慢降频"),
    (getattr(pynvml, "NVML_CLOCKS_EVENT_REASON_SW_THERMAL_SLOWDOWN", 1 << 4), "热控降频"),
    (getattr(pynvml, "NVML_CLOCKS_EVENT_REASON_HW_THERMAL_SLOWDOWN", 1 << 6), "热HW降频"),
    (getattr(pynvml, "NVML_CLOCKS_EVENT_REASON_SW_POWER_CAP", 1 << 3), "功率受限"),
    (getattr(pynvml, "NVML_CLOCKS_EVENT_REASON_APPLICATIONS_CLOCKS_SETTING", 1 << 2), "应用时钟"),
    (getattr(pynvml, "NVML_CLOCKS_EVENT_REASON_SYNC_BOOST", 1 << 8), "同步加速"),
    (getattr(pynvml, "NVML_CLOCKS_EVENT_REASON_GPU_IDLE", 1 << 0), "空闲"),
]


def _is_unsupported(exc: Exception) -> bool:
    if _HAVE_PYNVML and isinstance(exc, pynvml.NVMLError):
        return str(exc) in ("Not Supported", "Invalid Argument") or "Not Supported" in str(exc) \
            or "Invalid Argument" in str(exc)
    name = type(exc).__name__
    return any(s in name for s in ("NotSupported", "Unsupported", "InvalidArgument"))


class NvmlSampler:
    def __init__(self) -> None:
        self.available = False
        self.unsupported: set[str] = set()
        self.device_name = ""
        self._h = None

    def init(self) -> bool:
        if not _HAVE_PYNVML:
            return False
        try:
            pynvml.nvmlInit()
            self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.device_name = _as_str(pynvml.nvmlDeviceGetName(self._h))
            self.available = True
            return True
        except Exception:
            self.available = False
            return False

    def _get(self, field: str, fn, *a):
        """Call a getter; unsupported => remember & return None (no failure raised)."""
        if field in self.unsupported:
            return None
        try:
            return fn(*a)
        except Exception as e:  # noqa: BLE001
            if _is_unsupported(e):
                self.unsupported.add(field)
                return None
            raise

    def sample(self) -> GpuSample:
        if not self.available:
            raise RuntimeError("nvml unavailable")
        h = self._h
        s = GpuSample(ts=time.time(), name=self.device_name)
        try:
            mi = self._get("mem", pynvml.nvmlDeviceGetMemoryInfo, h)
            if mi is not None:
                s.mem_used_mib = mi.used / (1024 * 1024)
                s.mem_total_mib = mi.total / (1024 * 1024)
            ur = self._get("util", pynvml.nvmlDeviceGetUtilizationRates, h)
            if ur is not None:
                s.util_gpu, s.util_mem = ur.gpu, ur.memory
            pw = self._get("power", pynvml.nvmlDeviceGetPowerUsage, h)
            if pw is not None:
                s.power_w = pw / 1000.0
            pl = self._get("power_limit", pynvml.nvmlDeviceGetEnforcedPowerLimit, h)
            if pl is not None:
                s.power_limit_w = pl / 1000.0
            s.clk_core = self._get("clk_core", pynvml.nvmlDeviceGetClockInfo, h, pynvml.NVML_CLOCK_GRAPHICS)
            s.clk_mem = self._get("clk_mem", pynvml.nvmlDeviceGetClockInfo, h, pynvml.NVML_CLOCK_MEM)
            ps = self._get("pstate", pynvml.nvmlDeviceGetPerformanceState, h)
            if ps is not None:
                s.pstate = str(ps)
            fan = self._get("fan", pynvml.nvmlDeviceGetFanSpeed, h)
            if fan is not None:
                s.fan_pct = int(fan)
            s.temp_c = self._get("temp", pynvml.nvmlDeviceGetTemperature,
                                 h, pynvml.NVML_TEMPERATURE_GPU)
            thr = self._get("slowdown", pynvml.nvmlDeviceGetTemperatureThreshold,
                            h, pynvml.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN)
            if thr is not None:
                s.slowdown_c = int(thr)
            s.throttle = self._throttle_text(h)
        except Exception:
            raise  # genuine sample failure -> counted by caller
        return s

    def _throttle_text(self, h) -> str | None:
        reasons = None
        try:
            reasons = pynvml.nvmlDeviceGetCurrentClocksEventReasons(h)
        except Exception as e:  # noqa: BLE001
            if not _is_unsupported(e):
                try:
                    reasons = pynvml.nvmlDeviceGetPreviousClocksEventReasons(h)
                except Exception:  # noqa: BLE001
                    reasons = None
        if reasons is None:
            self.unsupported.add("throttle")
            return None
        if reasons == 0:
            return None
        labels = [lab for bit, lab in _THROTTLE_LABELS if bit and reasons & bit]
        return " / ".join(labels) if labels else f"0x{reasons:x}"

    def extra_temps(self) -> dict[str, float]:
        """Best-effort hotspot / memory / zone temps via thermal sensors.
        Anything the driver won't give is simply absent from the dict."""
        out: dict[str, float] = {}
        if not self.available:
            return out
        h = self._h
        try:
            sensors = pynvml.nvmlDeviceGetThermalSettings(h, 0)
            for i, s in enumerate(sensors):
                if 0 < i and s.currentTemp and s.currentTemp > 0:
                    out[f"区{chr(ord('A') + i - 1)}"] = float(s.currentTemp)
        except Exception as e:  # noqa: BLE001
            if not _is_unsupported(e):
                self.unsupported.add("zones")
        try:
            mem_t = pynvml.nvmlDeviceGetTemperature(h, 1)  # NVML_TEMPERATURE_MEMORY
            if mem_t is not None and mem_t >= 0:
                out["显存"] = float(mem_t)
        except Exception as e:  # noqa: BLE001
            if not _is_unsupported(e):
                self.unsupported.add("mem_temp")
        # Hotspot: probe the v2 sensor API (NVML_TEMPERATURE_HOTSPOT == 2 on newer drivers)
        if "热点" not in out and "hotspot" not in self.unsupported:
            try:
                r = pynvml.nvmlDeviceGetTemperatureV(h, 2)
                t = getattr(r, "temperature", None)
                if t is None and isinstance(r, tuple):
                    t = r[-1]
                if t is not None and t >= 0:
                    out["热点"] = float(t)
            except Exception as e:  # noqa: BLE001
                if not _is_unsupported(e):
                    self.unsupported.add("hotspot")
        return out


def _as_str(v) -> str:
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)
