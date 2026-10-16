"""Profile layer: scan starts/*/run.bat recipes (opaque for control), merge in
remembered extras from the state file, and resolve which profile owns the
running server (light `-m` read for matching only, fallback: most recently
active run log, else attach mode)."""

from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path

from ..model import EXTRA_SOURCE, Profile, ServerFacts
from ..state import ExtraProfile

# run.ps1 launches the exe with  -m "D:\models\x.gguf"  (backtick-continued lines);
# a recipe may also wrap a launcher script (-Model "…gguf" / -Model '…gguf').
_MODEL_RE = re.compile(r'(?:-m|-model|--model)(?:=|\s)\s*["\']([^"\']+\.gguf)["\']',
                       re.IGNORECASE)

LOG_GLOB = "server*.log"
_STAMP_RE = re.compile(r"^server-(\d{8}-\d{6})\.log$")


def pick_active_log(profile_dir: Path) -> Path | None:
    """Chronological pick: stamped files sort by name (server-<stamp>.log);
    legacy un-stamped server.log wins only when no stamped file exists.
    Late flushes to an older run log never pull the follower back.

    目录里没有 server*.log 时,按 LM Studio 的布局递归找:
    server-logs\\<YYYY-MM>\\<YYYY-MM-DD>.<n>.log —— 同一天可能多次重启产生
    .1/.2...,跨天还会换月份子目录,所以统一按修改时间取最新(跟踪器每次轮询
    都会重选,换文件/跨天自动跟上)。"""
    logs = [p for p in profile_dir.glob(LOG_GLOB) if p.is_file()]
    if logs:
        stamped = [p for p in logs if _STAMP_RE.match(p.name)]
        if stamped:
            return max(stamped, key=lambda p: p.name)
        return max(logs, key=lambda p: p.stat().st_mtime)
    deep = [p for p in profile_dir.rglob("*.log") if p.is_file()]
    if deep:
        return max(deep, key=lambda p: (p.stat().st_mtime, p.name))
    return None


def scan(starts_root: Path) -> list[Profile]:
    profiles: list[Profile] = []
    if not starts_root.is_dir():
        return profiles
    for child in sorted(starts_root.iterdir()):
        bat = child / "run.bat"
        if not child.is_dir() or not bat.is_file():
            continue
        model = None
        ps1 = child / "run.ps1"
        if ps1.is_file():
            try:
                m = _MODEL_RE.search(ps1.read_text(encoding="utf-8-sig", errors="replace"))
                model = m.group(1) if m else None
            except OSError:
                model = None
        profiles.append(Profile(name=child.name, dir=child, bat=bat, model_path=model))
    return profiles


def collect(roots: list[Path], extras: list[ExtraProfile] | None = None) -> list[Profile]:
    """Merge starts-root scans with remembered extra recipe dirs; dedupe by
    resolved dir; disambiguate same-name profiles via their source."""
    out: dict[str, Profile] = {}
    for root in roots:
        for p in scan(root):
            key = str(p.dir.resolve())
            if key not in out:
                p.source = root.name or str(root)
                out[key] = p
    for e in extras or []:
        d = Path(e.dir)
        key = str(d.resolve())
        if key in out:
            continue
        bat = d / e.bat
        if not bat.is_file():
            continue                      # stale remembered entry
        model = None
        ps1 = d / "run.ps1"
        if ps1.is_file():
            try:
                m = _MODEL_RE.search(ps1.read_text(encoding="utf-8-sig", errors="replace"))
                model = m.group(1) if m else None
            except OSError:
                model = None
        out[key] = Profile(name=d.name, dir=d, bat=bat, model_path=model, source=EXTRA_SOURCE)
    counts = Counter(p.name for p in out.values())
    for p in out.values():
        p.display = f"{p.name} @ {p.source}" if counts[p.name] > 1 and p.source else p.name
    return sorted(out.values(), key=lambda p: p.display)


def newest_run_log(profile_dir: Path) -> Path | None:
    logs = [p for p in profile_dir.glob(LOG_GLOB) if p.is_file()]
    if not logs:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime)


def _norm(path: str) -> str:
    return path.replace("/", "\\").lower().rstrip()


def match_profile(
    profiles: list[Profile],
    facts: ServerFacts | None,
    now: float | None = None,
    active_window_s: float = 90.0,
) -> Profile | None:
    """1) match by -m (ties broken by freshest run log: the recipe that actually
    launched is the one writing logs), 2) fallback to the dir with a recently
    active run log, 3) None (attach mode)."""
    if facts is not None and facts.model_path:
        want = _norm(facts.model_path)
        hits = [p for p in profiles if p.model_path and _norm(p.model_path) == want]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            def log_mtime(p: Profile) -> float:
                log = newest_run_log(p.dir)
                return log.stat().st_mtime if log else 0.0
            return max(hits, key=log_mtime)
    now = now or time.time()
    best: tuple[float, Profile] | None = None
    for p in profiles:
        log = newest_run_log(p.dir)
        if log is None:
            continue
        st = log.stat()
        age = now - st.st_mtime
        if age <= active_window_s and st.st_size > 0:
            if best is None or st.st_mtime > best[0]:
                best = (st.st_mtime, p)
    return best[1] if best else None
