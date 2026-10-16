"""Control plane. Stop = terminate the discovered PID. Restart/switch = run the
user's own run.bat verbatim in a NEW console (its taskkill semantics keep the
single-instance world; we never bypass the recipe by spawning the exe directly).
Attach-mode fallback: re-execute the discovered command line."""

from __future__ import annotations

import subprocess
import sys
import time

import psutil

from .model import Profile, ServerFacts


class ControlWindow:
    """While active, a missing server is an expected transition, not an error."""

    def __init__(self) -> None:
        self.until = 0.0

    def open(self, seconds: float = 180.0) -> None:
        self.until = time.time() + seconds

    @property
    def active(self) -> bool:
        return time.time() < self.until


def stop_server(facts: ServerFacts, timeout_s: float = 6.0) -> bool:
    try:
        p = psutil.Process(facts.pid)
    except psutil.NoSuchProcess:
        return True
    p.terminate()
    try:
        p.wait(timeout=timeout_s)
        return True
    except psutil.TimeoutExpired:
        p.kill()
        return True


def launch_bat(profile: Profile) -> None:
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
    subprocess.Popen(  # noqa: S603
        ["cmd", "/c", str(profile.bat)],
        cwd=str(profile.dir),
        creationflags=flags,
        close_fds=True,
    )


def relaunch_cmdline(facts: ServerFacts) -> None:
    """Attach-mode restart: replay the exact discovered argv in a new console."""
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
    exe = facts.exe or (facts.cmdline[0] if facts.cmdline else None)
    if not exe:
        raise RuntimeError("no executable to replay")
    args = facts.cmdline[1:] if len(facts.cmdline) > 1 else []
    subprocess.Popen([str(exe), *args], creationflags=flags, close_fds=True)  # noqa: S603


def restart(facts: ServerFacts, profile: Profile | None) -> str:
    """Returns the strategy used."""
    if profile is not None:
        launch_bat(profile)
        return "bat"
    relaunch_cmdline(facts)
    return "cmdline"
