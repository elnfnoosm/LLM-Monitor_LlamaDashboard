"""Per-source health: success/failure counters, data age, health vector.
A source is "ok" while its last success is fresher than a stale window."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class SourceStat:
    name: str
    successes: int = 0
    failures: int = 0
    last_ok: float | None = None
    absent: bool = False   # source structurally unavailable (e.g. no server / no NVML): not a failure
    stale_s: float | None = None   # per-source window override (e.g. slow sources)

    def record_ok(self, now: float | None = None) -> None:
        self.successes += 1
        self.last_ok = now or time.time()
        self.absent = False

    def record_fail(self) -> None:
        self.failures += 1

    def age(self, now: float | None = None) -> float | None:
        if self.last_ok is None:
            return None
        return (now or time.time()) - self.last_ok

    def healthy(self, stale_after_s: float, now: float | None = None) -> bool:
        if self.absent:
            return True   # degraded-by-design counts as ok (placeholder shown)
        age = self.age(now)
        limit = self.stale_s if self.stale_s is not None else stale_after_s
        return age is not None and age <= limit


class HealthBoard:
    SOURCE_ORDER = ("proc", "http", "log", "nvml")

    def __init__(self, stale_after_s: float = 3.0) -> None:
        self.stale_after_s = stale_after_s
        self.sources: dict[str, SourceStat] = {n: SourceStat(n) for n in self.SOURCE_ORDER}

    def get(self, name: str) -> SourceStat:
        return self.sources[name]

    def vector(self, now: float | None = None) -> list[int]:
        return [1 if self.sources[n].healthy(self.stale_after_s, now) else 0
                for n in self.SOURCE_ORDER]

    def totals(self) -> tuple[int, int]:
        s = f = 0
        for st in self.sources.values():
            s += st.successes
            f += st.failures
        return s, f

    def data_age(self, now: float | None = None) -> float | None:
        ages = [st.age(now) for st in self.sources.values() if st.age(now) is not None]
        return min(ages) if ages else None

    def reset_session(self) -> None:
        for st in self.sources.values():
            st.successes = st.failures = 0
            st.last_ok = None
