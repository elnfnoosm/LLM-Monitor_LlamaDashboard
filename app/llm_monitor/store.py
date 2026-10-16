"""Session store: live slot state + rates (from /slots deltas), authoritative
history (from log), session aggregates (本次监控 semantics), optional jsonl dump."""

from __future__ import annotations

import json
import time
from collections import deque

from .config import Settings
from .health import HealthBoard
from .model import GpuSample, Profile, PromptProgress, RequestRow, ServerFacts, SlotLive, SysSample
from .sources.httpsrc import ServerDialect


class SessionStore:
    def __init__(self, settings: Settings, health: HealthBoard) -> None:
        self.settings = settings
        self.health = health
        # attach state
        self.facts: ServerFacts | None = None
        self.profile: Profile | None = None
        self.dialect: ServerDialect | None = None
        self.server_boot_ts: float | None = None
        # live
        self.slots: dict[int, SlotLive] = {}
        self.live_out_tps: float | None = None
        self.live_in_tps: float | None = None
        self.gpu: GpuSample | None = None
        self.gpu_extra: dict[str, float] = {}
        self.sys: SysSample | None = None
        self.show_thermal_detail = False
        # avg speed as reported by the server itself (see sources/httpsrc.py);
        # None = not exposed by this build -> render falls back to session mean
        self.server_avg_out_tps: float | None = None
        self.server_avg_in_tps: float | None = None
        self.server_avg_ts: float | None = None
        # session aggregates
        self.history: deque[RequestRow] = deque(maxlen=1000)   # newest first
        # 本模型周期累计(由日志全分卷汇总,见 sources/runstats.py);
        # None=尚未扫完,累计区临时回落到 history(1000 条封顶的旧口径)
        self.run_totals: dict | None = None
        self.mtp_enabled: bool | None = None
        self._in_prev: dict[tuple[int, int], list] = {}     # (slot,task)->[proc, t, rate]
        self._out_prev: dict[tuple[int, int], list] = {}    # (slot,task)->[gen, t, rate]
        # slot -> frozen prompt t/s of the last prompt phase; kept on purpose when
        # the request ends so the panel can still show what that prompt achieved
        self._prompt_rate_done: dict[int, float] = {}
        self._fresh_max: dict[tuple[int, int], int] = {}    # monotone fresh_prompt
        self._prev_counts: dict[tuple[int, int], tuple[int, int]] = {}   # (n_prompt_tokens, n_decoded)
        # llama-server appends generated tokens to n_prompt_tokens; llama-kvmem-server
        # does not (verified on both - see tests/fixtures/README.md). Detected live so
        # no build sniffing is needed anywhere else.
        self.prompt_includes_generation = True
        self._first_kv_snap: dict[int, tuple[int, int]] = {}   # task -> (prompt, cache)
        # slot -> (progress 0..1, rate), from the log's `prompt processing` line (kvmem)
        self._prompt_progress: dict[int, tuple[float, float | None]] = {}
        self._prompt_started: dict[int, float] = {}            # slot -> prefill start ts
        self._prompt_task: dict[int, int] = {}                # slot -> task id the clock belongs to
        self._dump_fh = None

    # ---------------- attach / detach ----------------

    def attach(self, facts: ServerFacts | None, profile: Profile | None) -> None:
        changed = (facts is None) != (self.facts is None) or \
                  (facts is not None and self.facts is not None and facts.pid != self.facts.pid)
        self.facts = facts
        self.profile = profile
        if changed:
            self.dialect = None
            self.slots.clear()
            self.run_totals = None
            self._in_prev.clear()
            self._out_prev.clear()
            self._prompt_rate_done.clear()
            self._prompt_progress.clear()
            self._prompt_started.clear()
            self._prompt_task.clear()
            self._fresh_max.clear()
            self._prev_counts.clear()
            self.prompt_includes_generation = True
            self.server_avg_out_tps = None
            self.server_avg_in_tps = None
            self.server_avg_ts = None
            self.live_out_tps = self.live_in_tps = None
            if facts is not None:
                self.server_boot_ts = _proc_create_time(facts.pid)
            else:
                self.server_boot_ts = None

    @property
    def attached(self) -> bool:
        return self.facts is not None

    @property
    def attach_mode(self) -> bool:
        return self.attached and self.profile is None

    def uptime_s(self, now: float | None = None) -> float | None:
        if self.server_boot_ts is None:
            return None
        return (now or time.time()) - self.server_boot_ts

    # ---------------- samples in ----------------

    def on_props(self, props: dict) -> None:
        if self.dialect is not None:
            self.dialect.build_info = props.get("build_info", self.dialect.build_info)
            self.dialect.model_alias = props.get("model_alias", self.dialect.model_alias)
            self.dialect.total_slots = props.get("total_slots", self.dialect.total_slots)

    def on_slots(self, raw: list[dict], now: float | None = None) -> None:
        now = now or time.time()
        new_slots: dict[int, SlotLive] = {}
        for d in raw:
            try:
                nt = (d.get("next_token") or [{}])[0]
                sl = SlotLive(
                    id=int(d.get("id", -1)), n_ctx=int(d.get("n_ctx", 0)),
                    is_processing=bool(d.get("is_processing")),
                    id_task=int(d.get("id_task", -1)),
                    n_prompt_tokens=int(d.get("n_prompt_tokens", 0)),
                    n_prompt_tokens_processed=int(d.get("n_prompt_tokens_processed", 0)),
                    n_prompt_tokens_cache=int(d.get("n_prompt_tokens_cache", 0)),
                    n_decoded=int(nt.get("n_decoded", 0)),
                    n_remain=int(nt.get("n_remain", -1)),
                    speculative=bool(d.get("speculative")),
                    ts=now,
                )
            except (TypeError, ValueError):
                continue
            key = (sl.id, sl.id_task)
            if not sl.is_processing:
                sl.phase = "idle"
            else:
                # NOTE (llama.cpp quirk, verified against the webui): on llama-server
                # n_prompt_tokens *keeps growing during decode* because generated
                # tokens are appended to the slot prompt, so this turn's own prompt is
                # total minus what we generated. llama-kvmem-server does not append
                # them (tests/fixtures/README.md), hence the build flag. The tokens
                # that must be evaluated are that minus explicit cache hits. Field
                # updates lag each other by up to a decode window, so fresh_prompt is
                # monotone (max) within a task.
                base = (sl.n_prompt_tokens - sl.n_decoded if self.prompt_includes_generation
                        else sl.n_prompt_tokens)
                computed = max(base, sl.n_prompt_tokens_processed)
                fm = self._fresh_max.get(key)
                if fm is None or computed > fm:
                    self._fresh_max[key] = computed
                    fm = computed if fm is None else max(fm, computed)
                computed = max(computed, fm or 0)
                sl.fresh_prompt = computed
                sl.eval_total = max(1, computed - sl.n_prompt_tokens_cache)
                done = (sl.n_prompt_tokens_processed >= sl.eval_total
                        and sl.n_prompt_tokens_processed >= 64)   # ignore first-tick ramp
                sl.phase = "decode" if (sl.n_decoded > 0 or done) else "prompt"
            # rates = single-tick instantaneous values: Δcount ÷ Δwall between
            # the two most recent samples. The window moves every tick (so a
            # stalled tick keeps the previous value instead of decaying) but the
            # value is only recomputed when the counter actually moved.
            if sl.is_processing:
                g = self._out_prev.get(key)
                if g is None:
                    self._out_prev[key] = [sl.n_decoded, now, None]
                else:
                    dg, dt = sl.n_decoded - g[0], now - g[1]
                    if dg > 0 and dt > 0:
                        g[0], g[1], g[2] = sl.n_decoded, now, dg / dt
                    elif dg < 0:                       # counters rebased: restart
                        g[0], g[1] = sl.n_decoded, now
                    else:
                        g[1] = now                     # stalled tick: keep value
                p = self._in_prev.get(key)
                proc = sl.n_prompt_tokens_processed
                if p is None:
                    self._in_prev[key] = [proc, now, None, now]     # proc, t_last, rate, t_start
                else:
                    dp = proc - p[0]
                    # builds that only publish the prompt count when prefill ends
                    # (kvmem) show one single jump: attribute it to the whole time we
                    # have been watching this task instead of to one sampling interval.
                    dt = (now - p[3]) if p[0] == 0 else (now - p[1])
                    if dp > 0 and dt > 0:
                        p[0], p[1], p[2] = proc, now, dp / dt
                    elif dp < 0:
                        p[0], p[1], p[3] = proc, now, now
                    else:
                        p[1] = now
                if sl.phase == "prompt":
                    sl.rate_in = self._in_prev[key][2]
                    self._prompt_rate_done.pop(sl.id, None)   # new prompt: live again
                    if self._prompt_task.get(sl.id) != sl.id_task:
                        # new request: restart the prefill clock so the elapsed
                        # fallback measures *this* prompt, not the previous one
                        self._prompt_task[sl.id] = sl.id_task
                        self._prompt_started[sl.id] = now
                        # drop the previous prompt's % (only here, not every tick:
                        # kvmem emits the progress line every ~3.5s while /slots is
                        # polled at 1Hz, so a per-tick pop would blank the row
                        # between lines and make it flicker to the elapsed fallback)
                        self._prompt_progress.pop(sl.id, None)
                elif sl.phase == "decode":
                    # flip tick: the increment seen here is still prompt progress
                    frozen = self._in_prev[key][2]
                    if frozen is not None:
                        self._prompt_rate_done[sl.id] = frozen
                    sl.rate_in = self._prompt_rate_done.get(sl.id)
                    sl.rate_out = self._out_prev[key][2]
                    self._in_prev.pop(key, None)
            if sl.is_processing:
                pc = self._prev_counts.get(key)
                if pc is not None and sl.n_decoded > pc[1] and sl.n_prompt_tokens == pc[0]:
                    self.prompt_includes_generation = False    # kvmem: prompt count is stable
                self._prev_counts[key] = (sl.n_prompt_tokens, sl.n_decoded)
            new_slots[sl.id] = sl
            if sl.id_task >= 0 and sl.fresh_prompt > 0:
                prev = self._first_kv_snap.get(sl.id_task)
                cache = max(sl.n_prompt_tokens_cache, prev[1] if prev else 0)
                self._first_kv_snap[sl.id_task] = (max(prev[0] if prev else 0, sl.fresh_prompt), cache)
        # prune anchors of finished tasks
        active = {(s.id, s.id_task) for s in new_slots.values() if s.is_processing}
        # NOTE: _prompt_rate_done is keyed by slot and kept on purpose - the frozen
        # prompt speed stays readable after the request is gone.
        for store_ in (self._in_prev, self._out_prev, self._fresh_max, self._prev_counts):
            for k in list(store_):
                if k not in active:
                    del store_[k]
        self.slots = new_slots
        cur = self.current_slot()
        self.live_in_tps = cur.rate_in if cur else None
        self.live_out_tps = cur.rate_out if cur else None

    def prompt_rate(self) -> float | None:
        """输入速度:prompt 阶段为实时值,阶段结束后为冻结值(不回落到上一请求)。"""
        if self.live_in_tps is not None:
            return self.live_in_tps
        if self._prompt_rate_done:
            return list(self._prompt_rate_done.values())[-1]
        return None

    def on_prompt_progress(self, pp: PromptProgress, now: float | None = None) -> None:
        """log 的 `prompt processing` 行(kvmem 实时):按槽融合,槽号缺省归当前槽。"""
        now = now or time.time()
        sid = pp.slot
        if sid is None:
            cur = self.current_slot()
            if cur is None:
                return
            sid = cur.id
        self._prompt_progress[sid] = (pp.progress, pp.rate)
        self._prompt_started.setdefault(sid, now)

    def prompt_progress_for(self, slot_id: int) -> tuple[float, float | None] | None:
        return self._prompt_progress.get(slot_id)

    def prompt_elapsed_s(self, slot_id: int) -> float | None:
        start = self._prompt_started.get(slot_id)
        if start is None:
            return None
        cur = self.current_slot()
        now = cur.ts if cur is not None else time.time()
        return max(0.0, now - start)

    def on_server_avg(self, out_tps: float | None, in_tps: float | None,
                      now: float | None = None) -> None:
        """服务端自报的 avg speed(与自带 web 界面同源):/metrics 或日志结算块。"""
        self.server_avg_out_tps = out_tps
        self.server_avg_in_tps = in_tps
        self.server_avg_ts = now if now is not None else time.time()

    def on_gpu(self, sample: GpuSample | None, extra: dict[str, float]) -> None:
        self.gpu = sample
        self.gpu_extra = extra

    def on_sys(self, sample: SysSample | None) -> None:
        if sample is not None:
            self.sys = sample

    # ---------------- history (log authoritative) ----------------

    def on_request_row(self, row: RequestRow) -> None:
        snap = self._first_kv_snap.get(row.task)
        if snap is not None and row.kv_prompt_tokens is None:
            row.kv_prompt_tokens, row.kv_cache_tokens = snap[0], snap[1]
        self.history.appendleft(row)
        # 该 build 没有 avg speed 端点(如 kvmem):服务端每请求的速率就是 web 界面上的
        # 同一个数,由日志结算块提供
        if self.dialect is None or self.dialect.avg_tps_source in (None, "log"):
            self.on_server_avg(row.out_tps if row.out_tps > 0 else None,
                               row.in_tps if row.in_tps > 0 else None, row.wall_ts)
        if row.mtp_accept is not None:
            self.mtp_enabled = True
        if self.settings.dump_path is not None:
            try:
                if self._dump_fh is None:
                    self.settings.dump_path.parent.mkdir(parents=True, exist_ok=True)
                    self._dump_fh = self.settings.dump_path.open("a", encoding="utf-8")
                self._dump_fh.write(json.dumps(row_to_dict(row), ensure_ascii=False) + "\n")
                self._dump_fh.flush()
            except OSError:
                pass

    def set_mtp_evidence(self, enabled: bool | None) -> None:
        if enabled is not None:
            self.mtp_enabled = enabled

    # ---------------- aggregates ----------------

    def set_run_totals(self, t: dict | None) -> None:
        self.run_totals = t

    def totals(self) -> dict:
        rt = self.run_totals
        if rt is not None:
            n_req = int(rt.get("requests", 0))
            gen = int(rt.get("generated", 0))
            in_tok = int(rt.get("input_tokens", 0))
        else:
            n_req = len(self.history)
            gen = sum(r.output_tokens for r in self.history)
            in_tok = sum(r.input_tokens for r in self.history)
        kv_c = sum(r.kv_cache_tokens or 0 for r in self.history)
        kv_p = sum(r.kv_prompt_tokens or 0 for r in self.history)
        out_secs = sum(r.total_ms / 1000 for r in self.history if r.out_tps > 0 and r.output_tokens)
        gen_secs = sum(r.output_tokens / r.out_tps for r in self.history if r.out_tps > 0)
        prompt_secs = sum(r.input_tokens / r.in_tps for r in self.history if r.in_tps > 0)
        return {
            "requests": n_req,
            "generated": gen,
            "input_tokens": in_tok,
            "kv_cum": (kv_c / kv_p) if kv_p else None,
            "avg_out_tps": (gen / gen_secs) if gen_secs else None,
            "avg_in_tps": (in_tok / prompt_secs) if prompt_secs else None,
            "wall_gen_secs": out_secs,
        }

    def current_slot(self) -> SlotLive | None:
        proc = [s for s in self.slots.values() if s.is_processing]
        if proc:
            return sorted(proc, key=lambda s: s.id)[0]
        return None

    def reset_session(self) -> None:
        self.history.clear()
        self._first_kv_snap.clear()
        self.mtp_enabled = None
        self.health.reset_session()

    def last_request(self) -> RequestRow | None:
        return self.history[0] if self.history else None


def row_to_dict(r: RequestRow) -> dict:
    return {
        "wall_ts": r.wall_ts, "slot": r.slot, "task": r.task,
        "input": r.input_tokens, "output": r.output_tokens,
        "in_tps": round(r.in_tps, 2), "out_tps": round(r.out_tps, 2),
        "total_ms": round(r.total_ms, 1),
        "mtp_accept": r.mtp_accept, "mtp_accepted": r.mtp_accepted, "mtp_generated": r.mtp_generated,
        "kv_cache": r.kv_cache_tokens, "kv_prompt": r.kv_prompt_tokens,
    }


def _proc_create_time(pid: int) -> float | None:
    try:
        import psutil
        return psutil.Process(pid).create_time()
    except Exception:  # noqa: BLE001
        return None
