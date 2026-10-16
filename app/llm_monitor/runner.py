"""Sampling orchestrator: one tick advances all four sources with independent
health accounting, plus re-attachment after control transitions."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from . import control, state as state_mod
from .config import Settings
from .health import HealthBoard
from .model import PromptProgress, RequestRow
from .sources import httpsrc, profiles, proc, runstats
from .sources.cpusrc import SysSampler
from .sources.logparse import LineParser
from .sources.logtail import LogTailer
from .sources.nvmlsrc import NvmlSampler
from .store import SessionStore


class Runner:
    def __init__(self, settings: Settings, store: SessionStore, health: HealthBoard) -> None:
        self.settings = settings
        self.store = store
        self.health = health
        self.tailer = LogTailer(None)
        # 本模型周期累计:缓存放在 state.runstats,初次扫描在线程里跑
        self.runtotals = runstats.RunTotals(settings.state.runstats)
        self._rt_bind_seq = 0          # 绑定代号:过期的异步扫描结果作废
        self._last_rt_save = 0.0
        self.parser = LineParser()
        self.nvml = NvmlSampler()
        self.sys_sampler = SysSampler()
        self.window = control.ControlWindow()
        self._http: httpsrc.HttpSource | None = None
        self._http_pid: int | None = None
        self._last_proc = 0.0
        self._last_props = 0.0
        self._parse_fail_base = 0
        self.paused = False
        self._pause_since: float | None = None
        self._paused_total = 0.0
        # proc samples every proc_interval, not every tick: widen its stale window
        self.health.get("proc").stale_s = settings.proc_interval * 3 + 2
        if self.nvml.init():
            self.health.get("nvml").record_ok(self.now())
        else:
            self.health.get("nvml").absent = True

    # virtual clock: frozen while paused so "数据 Xs 前" doesn't drift (spec 7.2)
    def now(self) -> float:
        if self._pause_since is not None:
            return self._pause_since - self._paused_total
        return time.time() - self._paused_total

    def toggle_pause(self) -> bool:
        if self.paused:
            self._paused_total += time.time() - (self._pause_since or time.time())
            self._pause_since = None
            self.paused = False
        else:
            self._pause_since = time.time()
            self.paused = True
        return self.paused

    # ---------------- one sample tick ----------------

    async def tick(self) -> None:
        now = self.now()
        if now - self._last_proc >= self.settings.proc_interval or self._http is None:
            self._rediscover(now)
            self._last_proc = now
        await self._sample_http(now)
        self._sample_log(now)
        self._sample_runstats(now)
        self._sample_nvml(now)
        self._sample_sys(now)
        self.store.set_mtp_evidence(self.parser.mtp_enabled)

    # ---------------- proc / attach ----------------

    def _rediscover(self, now: float) -> None:
        try:
            found = proc.discover()
            self.health.get("proc").record_ok(now)
        except Exception:  # noqa: BLE001
            self.health.get("proc").record_fail()
            return
        facts = found[0] if found else None
        prof = None
        if facts is not None:
            try:
                known = profiles.collect(self.settings.starts_roots,
                                         self.settings.state.extra_profiles)
                prof = profiles.match_profile(known, facts, now)
            except OSError:
                prof = None
        changed = (self.store.facts is None) != (facts is None) or (
            facts is not None and (self.store.facts is None or self.store.facts.pid != facts.pid))
        if changed:
            self.store.attach(facts, prof)
            if prof is not None:
                rt_target = Path(prof.dir)
                self.tailer.bind(rt_target)
            elif facts is not None and facts.log_path:
                # Strata attach mode: its log path is named in the config json
                rt_target = Path(facts.log_path)
                self.tailer.bind(rt_target)
            else:
                rt_target = None
                self.tailer.bind(None)
            self._start_rt_bind(rt_target)
            self.parser.reset_records()
            self._parse_fail_base = self.parser.failures
            if self._http is not None:
                _spawn_close(self._http)
                self._http = None
                self._http_pid = None
            if facts is not None:
                self._http = httpsrc.HttpSource(facts)
                self._http_pid = facts.pid

    # ---------------- http ----------------

    async def _sample_http(self, now: float) -> None:
        src = self.health.get("http")
        if self._http is None:
            src.absent = True
            return
        st = self.store
        try:
            if st.dialect is None:
                d = await self._http.probe_dialect()
                if d.props_ok or d.slots_ok:
                    st.dialect = d
                    src.record_ok(now)
                    src.absent = False
                else:
                    # server still booting (not listening yet): keep dialect unset,
                    # retry next tick; show as absent rather than failure-storm
                    src.absent = True
            elif st.dialect.slots_ok:
                raw = await self._http.slots()
                st.on_slots(raw, now)
                src.record_ok(now)
                src.absent = False
            else:
                # /slots disabled on this server: keep liveness via cheap /health (1/5 ticks)
                if int(now) % 5 == 0:
                    await self._http.health()
                src.record_ok(now)
                src.absent = False
        except Exception:  # noqa: BLE001
            src.record_fail()
            if not self.window.active and self.store.facts is not None:
                # server may have died: force re-discovery next tick
                self._last_proc = 0.0
        if now - self._last_props >= self.settings.props_interval:
            self._last_props = now
            try:
                props = await self._http.props()
                self.store.on_props(props)
            except Exception:  # noqa: BLE001
                pass  # props is low-frequency; slots carries health counting
            if st.dialect is not None and st.dialect.avg_tps_source == "metrics":
                try:
                    out_tps, in_tps = await self._http.avg_tps()
                    st.on_server_avg(out_tps, in_tps, now)
                except Exception:  # noqa: BLE001
                    pass      # 取不到就留给面板走会话均值降级

    # ---------------- log ----------------

    def _sample_log(self, now: float) -> None:
        src = self.health.get("log")
        if not self.tailer.has_target:
            src.absent = True
            return
        src.absent = False
        try:
            lines = self.tailer.poll()
            src.record_ok(now)
        except OSError:
            src.record_fail()
            return
        rows: list[RequestRow] = []
        for ln in lines:
            row = self.parser.feed(ln, now)
            if row is not None:
                rows.append(row)
            pp = self._prompt_progress(ln)
            if pp is not None:
                self.store.on_prompt_progress(pp, now)
        for row in rows:
            self.store.on_request_row(row)
        extra_fail = self.parser.failures - self._parse_fail_base
        for _ in range(max(0, extra_fail)):
            src.record_fail()
        self._parse_fail_base = self.parser.failures

    def _prompt_progress(self, ln: str) -> PromptProgress | None:
        """分别调用各方言的实时 prefill 进度提取器(先落 kvmem,原版 llama 待实测补上)。
        各方言行格式不同,谁匹配谁生效;不依赖 dialect 判定(实时行在 prompt 阶段就发)。"""
        return (self.parser.kvmem_prompt_progress(ln)
                or self.parser.vanilla_prompt_progress(ln))

    # ---------------- run totals(本模型周期累计) ----------------

    def _start_rt_bind(self, target: Path | None) -> None:
        """换了监控目标:重扫日志分卷。无缓存的首次扫描可能数秒,放后台线程。"""
        self._rt_bind_seq += 1
        seq = self._rt_bind_seq
        if target is None:
            self.runtotals.bind(None)
            return

        async def _job() -> None:
            try:
                await asyncio.to_thread(self.runtotals.bind, target)
            except Exception:  # noqa: BLE001  扫描失败不致命:累计区回落历史口径
                return
            if seq == self._rt_bind_seq:       # 扫描期间又换了目标则作废
                self.store.set_run_totals(self.runtotals.totals)
                self._save_rt()

        try:
            asyncio.get_running_loop().create_task(_job())
        except RuntimeError:
            self.runtotals.bind(target)
            self.store.set_run_totals(self.runtotals.totals)

    def _sample_runstats(self, now: float) -> None:
        if not self.runtotals.ready or not self.runtotals.has_target:
            return
        try:
            t = self.runtotals.poll()
        except OSError:
            return
        self.store.set_run_totals(t)
        if self.runtotals.dirty and now - self._last_rt_save >= 30:
            self._save_rt(now)

    def _save_rt(self, now: float | None = None) -> None:
        if not self.runtotals.dirty:
            return
        try:
            state_mod.save(self.settings.state)
            self.runtotals.mark_saved()
            self._last_rt_save = now if now is not None else self.now()
        except OSError:
            pass

    def save_runstats_now(self) -> None:
        """应用退出时兜底落盘一次。"""
        self._save_rt()

    # ---------------- nvml ----------------

    def _sample_nvml(self, now: float) -> None:
        src = self.health.get("nvml")
        if not self.nvml.available:
            src.absent = True
            return
        try:
            s = self.nvml.sample()
            self.store.on_gpu(s, self.nvml.extra_temps())
            src.record_ok(now)
        except Exception:  # noqa: BLE001
            src.record_fail()

    # ---------------- sys (whole-machine CPU/RAM) ----------------

    def _sample_sys(self, now: float) -> None:
        # psutil 是本机只读数据,不进健康灯;采样失败(极少见)时保留上次值
        if not self.sys_sampler.available:
            return
        s = self.sys_sampler.sample()
        if s is not None:
            self.store.on_sys(s)

    # ---------------- control ----------------

    async def do_stop(self) -> None:
        if self.store.facts is None:
            return
        self.window.open(60)
        control.stop_server(self.store.facts)
        self._last_proc = 0.0

    async def do_restart(self) -> None:
        facts, prof = self.store.facts, self.store.profile
        if facts is None:
            return
        self.window.open(180)
        if prof is not None:
            control.restart(facts, prof)
        else:
            control.stop_server(facts)
            control.relaunch_cmdline(facts)
        self._last_proc = 0.0

    async def do_switch(self, profile_dir: Path) -> None:
        self.window.open(180)
        prof = profiles.Profile(profile_dir.name, profile_dir, profile_dir / "run.bat")
        control.launch_bat(prof)
        self._last_proc = 0.0

    async def do_switch_profile(self, prof) -> None:
        """Switch to an explicitly chosen recipe (e.g. from the dir browser)."""
        self.window.open(180)
        control.launch_bat(prof)
        self._last_proc = 0.0


def _spawn_close(http: httpsrc.HttpSource) -> None:
    import asyncio

    async def _c() -> None:
        try:
            await http.aclose()
        except Exception:  # noqa: BLE001
            pass

    try:
        asyncio.get_running_loop().create_task(_c())
    except RuntimeError:
        pass
