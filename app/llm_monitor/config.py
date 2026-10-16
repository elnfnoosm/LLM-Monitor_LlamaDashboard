"""Runtime settings. No device-specific defaults anywhere: starts roots come
from the CLI and/or the per-machine state file only."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .state import State


@dataclass
class Settings:
    starts_roots: list[Path] = field(default_factory=list)   # CLI ∪ state, deduped
    state: State = field(default_factory=State)
    state_notice: str | None = None
    sample_interval: float = 1.0        # /slots + nvml cadence (s)
    props_interval: float = 30.0        # static facts refresh (s)
    proc_interval: float = 5.0          # process rediscovery fallback (s)
    stale_ticks: int = 3                # ticks without success => source not ok
    dump_path: Path | None = None       # optional jsonl dump of request rows
    host: str = "127.0.0.1"             # probe host; server may bind 0.0.0.0
