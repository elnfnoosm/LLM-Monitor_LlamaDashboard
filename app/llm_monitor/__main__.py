from __future__ import annotations

import argparse
from pathlib import Path

from . import state as state_mod
from .app import DashboardApp
from .config import Settings


def parse_args(argv: list[str] | None = None) -> Settings:
    ap = argparse.ArgumentParser(prog="llm-monitor",
                                 description="llama.cpp 运行状况 TUI 监控台(与启动 bat 配合且解耦)")
    ap.add_argument("--starts-root", action="append", default=None,
                    help="profile 根目录,可重复;不传则用本机记住的根(首次为空,界面里浏览添加)")
    ap.add_argument("--interval", type=float, default=1.0, help="采样间隔秒(默认 1)")
    ap.add_argument("--stale", type=int, default=3, help="连续多少个间隔无成功视为非健康")
    ap.add_argument("--dump", default=None, help="请求结算 jsonl 落盘路径(可选)")
    a = ap.parse_args(argv)
    st, notice = state_mod.load()
    roots: list[Path] = []
    for r in (a.starts_root or []) + st.starts_roots:
        p = Path(r)
        if p not in roots:
            roots.append(p)
    return Settings(
        starts_roots=roots,
        state=st,
        state_notice=notice,
        sample_interval=a.interval,
        stale_ticks=a.stale,
        dump_path=Path(a.dump) if a.dump else None,
    )


def main(argv: list[str] | None = None) -> int:
    settings = parse_args(argv)
    DashboardApp(settings).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
