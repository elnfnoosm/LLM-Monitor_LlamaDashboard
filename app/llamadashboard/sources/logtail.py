"""Run-log tailer: follows the newest server-<stamp>.log inside the resolved
profile dir; tolerates rotation (a new stamp file appears per run) and
truncation (offset > size => re-read from 0).

It can also bind one fixed log file (Strata: the path is named in the running
config json, e.g. D:\\Strata\\Strata-main\\strata-iq3_s.log)."""

from __future__ import annotations

from pathlib import Path

from .profiles import pick_active_log


class LogTailer:
    def __init__(self, profile_dir: Path | None = None) -> None:
        self._dir: Path | None = None
        self._file: Path | None = None
        self._path: Path | None = None
        self._offset = 0
        self._partial = ""
        self.set_profile(profile_dir)

    def set_profile(self, profile_dir: Path | None) -> None:
        if profile_dir != self._dir:
            self._dir = profile_dir
            self._file = None
            self._path = None
            self._offset = 0
            self._partial = ""

    def set_file(self, path: Path | None) -> None:
        """Follow one fixed file instead of scanning a profile directory."""
        if path != self._file:
            self._file = path
            self._dir = None
            self._path = None
            self._offset = 0
            self._partial = ""

    def bind(self, target: Path | None) -> None:
        """Directory -> newest server-*.log follower; file -> fixed follower."""
        if target is None:
            self.set_profile(None)
        elif target.is_dir():
            self.set_profile(target)
        else:
            self.set_file(target)

    @property
    def has_target(self) -> bool:
        return self._dir is not None or self._file is not None

    @property
    def path(self) -> Path | None:
        return self._path

    def _candidate(self) -> Path | None:
        if self._file is not None:
            return self._file
        if self._dir is None:
            return None
        return pick_active_log(self._dir)

    def poll(self) -> list[str]:
        """Return complete lines produced since last call. Raises OSError->caller counts."""
        cand = self._candidate()
        if cand is None:
            return []
        if cand != self._path:
            # stamp-name order is chronological: switching back never happens
            self._path, self._offset, self._partial = cand, 0, ""

        path = self._path
        assert path is not None
        if not path.exists():
            return []
        size = path.stat().st_size
        if size < self._offset:          # truncated
            self._offset = 0
            self._partial = ""
        if size == self._offset:
            return []
        with path.open("rb") as fh:
            fh.seek(self._offset)
            chunk = fh.read(size - self._offset)
            self._offset = fh.tell()
        text = self._partial + chunk.decode("utf-8", errors="replace")
        lines = text.splitlines(keepends=False)
        # last element is partial unless chunk ended with newline
        self._partial = lines[-1] if lines and not chunk.endswith(b"\n") else ""
        if self._partial:
            lines = lines[:-1]
        return [ln for ln in lines if ln.strip()]
