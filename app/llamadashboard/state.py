"""Per-machine remembered discoveries (starts roots + extra profile dirs).

Kept OUT of the repo and OUT of code defaults so the project stays portable:
on a fresh device the state file simply does not exist and the picker offers
the directory browser instead of stale foreign paths."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExtraProfile:
    dir: str
    bat: str = "run.bat"


@dataclass
class State:
    starts_roots: list[str] = field(default_factory=list)
    extra_profiles: list[ExtraProfile] = field(default_factory=list)
    last_browse_dir: str | None = None
    # runstats 缓存:{日志绝对路径 -> 该分卷的本模型段统计(见 sources/runstats.py)}
    runstats: dict = field(default_factory=dict)

    def add_extra(self, dirpath: Path, bat: Path | None = None) -> bool:
        """Remember a recipe dir; returns False when already known (dedupe)."""
        key = str(Path(dirpath).resolve())
        for e in self.extra_profiles:
            if str(Path(e.dir).resolve()) == key:
                return False
        self.extra_profiles.append(ExtraProfile(dir=key, bat=(bat or Path(dirpath) / "run.bat").name))
        return True

    def remove_extra(self, dirpath: Path) -> bool:
        """Forget a remembered recipe dir; returns False when it wasn't there."""
        key = str(Path(dirpath).resolve())
        keep = [e for e in self.extra_profiles if str(Path(e.dir).resolve()) != key]
        removed = len(keep) != len(self.extra_profiles)
        self.extra_profiles = keep
        return removed

    def add_root(self, root: Path) -> bool:
        key = str(Path(root).resolve())
        if key in {str(Path(r).resolve()) for r in self.starts_roots}:
            return False
        self.starts_roots.append(key)
        return True


def state_path() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "llamadashboard" / "state.json"


def load(path: Path | None = None) -> tuple[State, str | None]:
    """Returns (state, notice). Corrupt file => backup + empty state + notice."""
    p = path or state_path()
    if not p.is_file():
        return State(), None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("state root is not an object")
        raw_rs = raw.get("runstats", {})
        runstats: dict = {}
        if isinstance(raw_rs, dict):
            for k, v in raw_rs.items():
                if isinstance(k, str) and isinstance(v, dict) \
                        and isinstance(v.get("size"), int) and isinstance(v.get("mtime"), (int, float)):
                    runstats[k] = v
        return State(
            starts_roots=[str(x) for x in raw.get("starts_roots", []) if isinstance(x, str)],
            extra_profiles=[ExtraProfile(**e) for e in raw.get("extra_profiles", [])
                            if isinstance(e, dict) and isinstance(e.get("dir"), str)],
            last_browse_dir=raw.get("last_browse_dir") if isinstance(raw.get("last_browse_dir"), str) else None,
            runstats=runstats,
        ), None
    except (ValueError, TypeError, OSError) as exc:
        backup = p.with_name(f".broken-{time.strftime('%Y%m%d-%H%M%S')}")
        try:
            p.replace(backup)
        except OSError:
            backup = p
        return State(), f"状态文件损坏({exc.__class__.__name__}),已备份为 {backup.name} 并按空状态继续"


def save(state: State, path: Path | None = None) -> None:
    p = path or state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(
        {"starts_roots": state.starts_roots,
         "extra_profiles": [{"dir": e.dir, "bat": e.bat} for e in state.extra_profiles],
         "last_browse_dir": state.last_browse_dir,
         "runstats": state.runstats},
        ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)
