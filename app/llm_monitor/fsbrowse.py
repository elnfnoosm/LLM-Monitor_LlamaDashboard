"""Read-only directory browser logic: discover launch recipes (run.bat dirs or
standalone .bat files) anywhere on the machine. No writes, ever."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

import psutil

PROFILE, BAT, DIR, UP = "profile", "bat", "dir", "up"


@dataclass
class Entry:
    kind: str          # PROFILE | BAT | DIR
    path: Path
    label: str

    @property
    def recipe_dir(self) -> Path:
        """PROFILE -> itself; BAT -> its parent dir."""
        return self.path if self.kind == PROFILE else self.path.parent

    @property
    def recipe_bat(self) -> Path:
        return self.path if self.kind == BAT else self.path / "run.bat"


def _hidden(p: Path) -> bool:
    if p.name.startswith("."):
        return True
    if os.name == "nt":
        try:
            return bool(p.stat().st_file_attributes & stat.FILE_ATTRIBUTE_HIDDEN)
        except OSError:
            return False
    return False


def list_drives() -> list[Path]:
    out: list[Path] = []
    try:
        parts = psutil.disk_partitions(all=False)
    except Exception:  # noqa: BLE001
        return out
    for d in parts:
        try:
            p = Path(d.mountpoint)
            if p not in out:
                out.append(p)
        except Exception:  # noqa: BLE001
            continue
    return out


def list_entries(path: Path) -> tuple[list[Entry], str | None]:
    """One level: profile dirs first, then loose bats, then plain dirs."""
    try:
        children = sorted(path.iterdir(), key=lambda p: p.name.lower())
    except (PermissionError, OSError) as exc:
        return [], f"无法读取 {path}: {exc.__class__.__name__}"
    prof, bats, dirs = [], [], []
    for c in children:
        try:
            if _hidden(c):
                continue
            if c.is_dir():
                if (c / "run.bat").is_file():
                    prof.append(Entry(PROFILE, c, c.name))
                else:
                    dirs.append(Entry(DIR, c, c.name))
            elif c.suffix.lower() == ".bat":
                bats.append(Entry(BAT, c, c.name))
        except OSError:
            continue
    return prof + bats + dirs, None


def resolve_paste(text: str) -> tuple[Path | None, str | None]:
    p = Path(text.strip()).expanduser()
    if not p.is_absolute():
        return None, "需要绝对路径"
    if not p.exists():
        return None, f"路径不存在: {p}"
    if not p.is_dir():
        return None, f"不是目录: {p}"
    return p, None
