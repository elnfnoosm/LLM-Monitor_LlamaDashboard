"""Process discovery: everything about the running server comes from the live
process command line (zero-modification contract; port/api-key/model follow
whatever the user's recipe actually launched).

Several llama.cpp builds ship different exe names (plain `llama-server.exe` and
the KV-memory variant `llama-kvmem-server.exe`); any of them is monitored the
same way - whichever is running is the one we attach to.

Strata has no llama-server.exe: its server is `python.exe serve/server.py
--config <json> --port N`, which spawns engine\\strata.exe as a child.  We
recognise that cmdline shape and read the config json it names (port/api key/
context/log path/model_name) - the same zero-config idea, one indirection in."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import psutil

from ..model import ServerFacts

PROC_NAMES = ("llama-server.exe", "llama-kvmem-server.exe")
PROC_NAME = PROC_NAMES[0]        # backwards-compatible alias

# Strata's service process is the python interpreter running serve/server.py
_PYTHON_NAMES = ("python.exe", "pythonw.exe")
_STRATA_SCRIPT = os.path.join("serve", "server.py")   # ".../serve/server.py"

# flags that consume the next token as their value
_VALUE_FLAGS = {
    "-m", "--model", "--port", "--api-key", "-c", "--ctx-size", "--ctx-len",
    "--parallel", "-np", "--alias", "-ma", "--host", "-h", "--spec-type",
    "--model-content", "--lora", "--adapter",
    # Strata's serve/server.py flags
    "--config", "--engine",
}


def is_server_proc(name: str | None) -> bool:
    """Case-insensitive match against the known server exe names."""
    return (name or "").lower() in PROC_NAMES


def _is_strata_cmdline(cmd: list[str]) -> bool:
    """True for `python ... serve/server.py ...` (path separators both ways)."""
    for tok in cmd:
        t = tok.replace("/", os.sep).lower()
        if t.endswith(_STRATA_SCRIPT):
            return True
    return False


def short_names() -> tuple[str, ...]:
    """Known exe names without the suffix, for hints (llama-server / llama-kvmem-server)."""
    return tuple(n[:-4] if n.lower().endswith(".exe") else n for n in PROC_NAMES)


def proc_label(exe: str | None) -> str:
    """Short build label for the header, e.g. 'llama-kvmem-server'."""
    if not exe:
        return "—"
    name = exe.replace("\\", "/").rsplit("/", 1)[-1]
    return name[:-4] if name.lower().endswith(".exe") else (name or exe)


def parse_cmdline(pid: int, exe: str, cmdline: list[str]) -> ServerFacts:
    """Extract value flags from an argv list; supports `--flag value` and `--flag=value`."""
    args: dict[str, str] = {}
    i = 0
    while i < len(cmdline):
        tok = cmdline[i]
        if tok.startswith("-"):
            if "=" in tok:
                key, _, val = tok.partition("=")
                args.setdefault(key, val)
            elif tok in _VALUE_FLAGS and i + 1 < len(cmdline):
                args.setdefault(tok, cmdline[i + 1])
                i += 1
        i += 1
    return ServerFacts(pid=pid, exe=exe, cmdline=list(cmdline), args=args)


def _config_arg(config_args: list[str], name: str) -> str | None:
    """Read `--name value` (also `--name=value`) from the engine argv stored in
    Strata's config json."""
    for i, tok in enumerate(config_args):
        t = str(tok)
        if t == name and i + 1 < len(config_args):
            return str(config_args[i + 1])
        if t.startswith(name + "="):
            return t.split("=", 1)[1]
    return None


def parse_strata(pid: int, exe: str, cmd: list[str]) -> ServerFacts:
    """Build facts for a running `python serve/server.py` process.

    Port comes from --port on the cmdline (the json's "port" is the fallback);
    api key, context size, log path and model name live in the --config json
    (strata-iq3_s.json style).  Missing/unreadable config degrades gracefully -
    the port alone is enough to attach."""
    base = parse_cmdline(pid, exe, cmd)
    args = dict(base.args)
    log_path: str | None = None
    label: str | None = None
    cfg_path = args.get("--config")
    cfg: dict = {}
    if cfg_path:
        try:
            with open(cfg_path, "r", encoding="utf-8-sig") as fh:
                cfg = json.load(fh)
        except (OSError, ValueError):
            cfg = {}
    if isinstance(cfg, dict):
        if not args.get("--port"):
            port = cfg.get("port")
            if port is not None:
                args["--port"] = str(port)
        key = cfg.get("api_key", "")
        if isinstance(key, list):          # Strata also accepts ["k1", "k2"]
            key = key[0] if key else ""
        if key and not args.get("--api-key"):
            args["--api-key"] = str(key)
        eng_args = cfg.get("args")
        if isinstance(eng_args, list):
            ctx = _config_arg(eng_args, "--max-context")
            if ctx and not args.get("-c"):
                args["-c"] = ctx
        name = cfg.get("model_name")
        if isinstance(name, str) and name:
            label = name
            args.setdefault("--alias", name)
        log = cfg.get("log")
        if isinstance(log, str) and log:
            log_path = log
    return ServerFacts(pid=pid, exe=exe, cmdline=list(cmd), args=args,
                       kind="strata", log_path=log_path, label=label)


def _lmstudio_logs_dir(exe: str) -> str | None:
    """LM Studio 自带的 llama-server(exe 位于 <home>\\.lmstudio\\extensions\\
    backends\\<后端>\\llama-server.exe)把每次请求的结算日志写到:
    <home>\\.lmstudio\\server-logs\\<YYYY-MM>\\<YYYY-MM-DD>.<n>.log
    找到这个目录,日志跟踪器就能读到请求历史(格式与原版 llama.cpp 一致)。"""
    try:
        for parent in Path(exe).resolve().parents:
            if parent.name == ".lmstudio":
                logs = parent / "server-logs"
                if logs.is_dir():
                    return str(logs)
                break
    except OSError:
        pass
    fallback = Path.home() / ".lmstudio" / "server-logs"   # 自定义安装位置的兜底
    return str(fallback) if fallback.is_dir() else None


def _enrich_lmstudio(facts: ServerFacts) -> ServerFacts:
    """给 LM Studio 启动的 llama-server 补上它的日志目录(其他来源不动)。"""
    if facts.log_path:
        return facts
    logs = _lmstudio_logs_dir(facts.exe)
    return replace(facts, log_path=logs) if logs else facts


def discover() -> list[ServerFacts]:
    """All running server processes, any known backend (v1 monitors one at a
    time: caller takes the first, i.e. the lowest pid)."""
    out: list[ServerFacts] = []
    strata_pids: set[int] = set()
    for p in psutil.process_iter(["name", "pid", "ppid", "exe", "cmdline"]):
        try:
            name = (p.info["name"] or "").lower()
            cmd = p.info["cmdline"] or []
            if not cmd:
                continue
            if name in PROC_NAMES:
                facts = parse_cmdline(p.info["pid"], p.info["exe"] or cmd[0], cmd)
                out.append(_enrich_lmstudio(facts))
            elif name in _PYTHON_NAMES and _is_strata_cmdline(cmd):
                # Strata respawns itself as a child with the same cmdline; only
                # the root service process owns the listening port.
                if p.info["ppid"] in strata_pids:
                    continue
                strata_pids.add(p.info["pid"])
                out.append(parse_strata(p.info["pid"], p.info["exe"] or cmd[0], cmd))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    out = [f for f in out if not (f.kind == "strata" and _strata_parent_running(f.pid, strata_pids))]
    out.sort(key=lambda f: f.pid)
    return out


def _strata_parent_running(pid: int, strata_pids: set[int]) -> bool:
    """True if a (grand)parent of pid is another Strata service process."""
    try:
        parent = psutil.Process(pid).parent()
        return parent is not None and parent.pid in strata_pids
    except psutil.Error:
        return False
