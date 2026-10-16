"""Tolerant parser registry for llama-server run logs.

Extractors recognize per-request settlement blocks; family keyword present but
extraction failed => counted as a parse failure (visible in source health).
Unknown lines are silently ignored (not failures)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..model import PromptProgress, RequestRow

_TS_PREFIX = r"(?:\d+\.)*\d+\.\d+\.\d+\.\d+\s+\w\s+"  # e.g. "0.41.912.708 I "

_RE_LAUNCH = re.compile(
    r"slot\s+launch_slot_:\s+id\s+(\d+)\s*\|\s*task\s+(\d+)\s*\|\s*processing task"
)
_RE_RELEASE = re.compile(
    r"slot\s+release:\s*id\s+(\d+)\s*\|\s*task\s+(\d+)\s*\|\s*stop processing:\s*n_tokens\s*=\s*(\d+)"
)
_RE_PROMPT = re.compile(
    r"slot\s+print_timing:\s+id\s+(\d+)\s*\|\s*task\s+(\d+)\s*\|\s*prompt eval time\s*=\s*([\d.,]+)\s*ms\s*/\s*([\d.,]+)\s*tokens"
    r"(?:.*?,\s*([\d.,]+)\s*tokens per second)?"
)
_RE_EVAL = re.compile(
    r"slot\s+print_timing:\s+id\s+(\d+)\s*\|\s*task\s+(\d+)\s*\|\s*eval time\s*=\s*([\d.,]+)\s*ms\s*/\s*([\d.,]+)\s*tokens"
    r"(?:.*?,\s*([\d.,]+)\s*tokens per second)?"
)
_RE_TOTAL = re.compile(
    r"slot\s+print_timing:\s+id\s+(\d+)\s*\|\s*task\s+(\d+)\s*\|\s*total time\s*=\s*([\d.,]+)\s*ms\s*/\s*([\d.,]+)\s*tokens"
)
_RE_DRAFT = re.compile(
    r"slot\s+print_timing:\s+id\s+(\d+)\s*\|\s*task\s+(\d+)\s*\|\s*draft acceptance\s*=\s*([\d.,]+)"
    r"\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated\)"
)

# ---- llama-kvmem-server: leaner settlement block (verified on a live run) ----
# no "print_timing: id | task |" prefix, tps inline, explicit "cache = N", and no
# release line at all - the block is closed by "total time".
_RE_KV_LAUNCH = re.compile(
    r"slot\s+processing task,\s*n_prompt\s*=\s*(\d+),\s*n_predict\s*=\s*(-?\d+)"
)
_RE_KV_PROMPT = re.compile(
    r"slot\s+prompt eval time\s*=\s*([\d.,]+)\s*ms\s*/\s*([\d.,]+)\s*tokens"
    r"(?:\s*\(\s*([\d.,]+)\s*tokens per second\))?(?:,\s*cache\s*=\s*(\d+))?"
)
_RE_KV_EVAL = re.compile(
    r"slot\s+eval time\s*=\s*([\d.,]+)\s*ms\s*/\s*([\d.,]+)\s*tokens"
    r"(?:\s*\(\s*([\d.,]+)\s*tokens per second\))?"
)
_RE_KV_TOTAL = re.compile(
    r"slot\s+total time\s*=\s*([\d.,]+)\s*ms\s*/\s*([\d.,]+)\s*tokens"
)
# kvmem 实时 prefill 行(实测格式):`slot   prompt processing, n_tokens =  61952,
# progress = 0.59, t = 83.05 s / 746.00 tokens per second`。不带槽号/任务号。
_RE_KV_PROMPT_PROG = re.compile(
    r"slot\s+prompt processing,\s*n_tokens\s*=\s*(\d+),\s*progress\s*=\s*([\d.]+)"
    r"(?:.*?/\s*([\d.]+)\s*tokens per second)?"
)

# ---- Strata: one self-contained settlement line per request ----
# e.g. "strata serve: prompt 34 tokens = 0 reused + 34 read in 247 ms
#       (137.4 tok/s), 128 generated in 1227 ms (104.3 tok/s),
#       drafts accepted 90 of 113, 1 checkpoints"
# prompt = reused (conversation-cache hits) + read (evaluated this request).
_RE_STRATA_SETTLE = re.compile(
    r"strata serve:\s*prompt\s+(\d+)\s+tokens\s*=\s*(\d+)\s+reused\s*\+\s*(\d+)\s+read"
    r"\s+in\s+([\d.]+)\s+ms"
    r"(?:\s*\(\s*([\d.]+)\s*tok/s\s*\))?"
    r"\s*,\s*(\d+)\s+generated\s+in\s+([\d.]+)\s+ms"
    r"(?:\s*\(\s*([\d.]+)\s*tok/s\s*\))?"
    r"(?:\s*,\s*drafts accepted\s+(\d+)\s+of\s+(\d+))?"
)
_STRATA_KEY = "strata serve: prompt"

_RE_MTP_INIT = re.compile(r"creating MTP draft context")
_RE_INIT = re.compile(r"llama_server:\s*initializing")
# ---- kvmem: MTP/spec statistics (needs --verbosity 4; prints once per request,
# right before the settlement block, and the counters are cumulative) ----
# e.g. "spec common_specu: statistics        draft-mtp: #calls(b,g,a) =    1     90
#       90, #gen drafts =     90, #acc drafts =    56, #gen tokens =    270,
#       #acc tokens =   130, ..."
_RE_SPEC_STATS = re.compile(
    r"spec\s+\S+:\s+statistics\s+\S+:.*?"
    r"#gen tokens\s*=\s*(\d+),\s*#acc tokens\s*=\s*(\d+)"
)
_RE_SLOTS = re.compile(r"n_slots\s*=\s*(\d+),\s*n_ctx_slot\s*=\s*(\d+)")
_RE_MODEL = re.compile(r"loading model\s+'([^']+)'")
_RE_LISTEN = re.compile(r"listening on\s+(http://\S+)")

# (keyword marking extractor family, compiled extractor, kind). The kvmem patterns
# come first: they are anchored on "slot <name> =" and therefore cannot swallow a
# llama-server line, which carries the "print_timing: id | task |" prefix.
_BODY_EXTRACTORS: list[tuple[str, re.Pattern, str]] = [
    ("prompt eval time", _RE_KV_PROMPT, "kv_prompt"),
    ("prompt eval time", _RE_PROMPT, "prompt"),
    ("eval time", _RE_KV_EVAL, "kv_eval"),
    ("eval time", _RE_EVAL, "eval"),
    ("total time", _RE_KV_TOTAL, "kv_total"),
    ("total time", _RE_TOTAL, "total"),
    ("draft acceptance", _RE_DRAFT, "draft"),
]
_FAMILY_KEYS = sorted({k for k, _, _ in _BODY_EXTRACTORS})


def _f(m: re.Match, i: int) -> float:
    return float(m.group(i).replace(",", "")) if m.group(i) else 0.0


@dataclass
class _Record:
    slot: int
    task: int
    prompt_ms: float = 0.0
    prompt_tok: int = 0
    prompt_tps: float = 0.0
    eval_ms: float = 0.0
    eval_tok: int = 0
    eval_tps: float = 0.0
    total_ms: float = 0.0
    mtp: float | None = None
    mtp_accepted: int | None = None
    mtp_generated: int | None = None
    # kvmem: full prompt (incl. cache hits) and the explicitly reported cache
    n_prompt_full: int = 0
    cache_tok: int | None = None


@dataclass
class ParseResult:
    rows: list[RequestRow] = field(default_factory=list)
    failures: int = 0

class LineParser:
    def __init__(self) -> None:
        self._records: dict[tuple[int, int], _Record] = {}
        self._released: set[tuple[int, int]] = set()
        self.mtp_enabled: bool | None = None   # None until evidence
        self.startup: dict[str, object] = {}
        self.failures = 0
        # kvmem prints no slot/task ids: single slot, synthetic increasing task id
        self._kv_task = 0
        self._kv_key: tuple[int, int] | None = None
        # Strata prints no slot/task ids either: single sequence, synthetic task id
        self._strata_task = 0
        # cumulative spec counters of the last statistics line (gen tokens, acc tokens)
        self._spec_cum: tuple[int, int] | None = None
        # (accept rate, accepted, generated) waiting for a record to attach to
        self._spec_pending: tuple[float, int, int] | None = None

    def feed(self, line: str, now: float) -> RequestRow | None:
        """Feed one log line; return a completed RequestRow on release, else None."""
        # spec statistics carry no "slot" token: they must be seen before the
        # slot pre-screen below, and never count as a parse failure.
        if m := _RE_SPEC_STATS.search(line):
            self._spec_stats(int(m.group(1)), int(m.group(2)))
            return None
        if "initializing" in line and _RE_INIT.search(line):
            self.reset_records()
            self.startup = {}
            return None
        # Strata's settlement line has no "slot" token: handle it before the
        # llama.cpp pre-screen below.
        if _STRATA_KEY in line:
            return self._strata_settlement(line, now)
        # cheap family pre-screen
        if "slot" not in line:
            self._startup(line)
            return None
        row: RequestRow | None = None
        if m := _RE_LAUNCH.search(line):
            key = (int(m.group(1)), int(m.group(2)))
            self._records[key] = _Record(*key)
            self._released.discard(key)
        elif m := _RE_KV_LAUNCH.search(line):
            self._kv_task += 1
            key = (0, self._kv_task)                    # kvmem: single slot, no ids printed
            rec = _Record(*key, n_prompt_full=int(m.group(1)))
            if self._spec_pending is not None:            # stats seen before the block opened
                rec.mtp, rec.mtp_accepted, rec.mtp_generated = self._spec_pending
                self._spec_pending = None
            self._records[key] = rec
            self._released.discard(key)
            self._kv_key = key
        elif m := _RE_RELEASE.search(line):
            key = (int(m.group(1)), int(m.group(2)))
            rec = self._records.pop(key, None)
            if rec is not None and key not in self._released:
                self._released.add(key)
                cap = int(m.group(3))               # prompt(full) + generated at release
                full_prompt = cap - rec.eval_tok if cap >= rec.eval_tok else rec.prompt_tok
                derived_cache = max(0, full_prompt - rec.prompt_tok)
                row = RequestRow(
                    wall_ts=now, slot=key[0], task=key[1],
                    input_tokens=full_prompt, output_tokens=rec.eval_tok,
                    in_tps=rec.prompt_tps, out_tps=rec.eval_tps,
                    total_ms=rec.total_ms,
                    mtp_accept=rec.mtp, mtp_accepted=rec.mtp_accepted,
                    mtp_generated=rec.mtp_generated,
                    kv_prompt_tokens=full_prompt if full_prompt else None,
                    kv_cache_tokens=derived_cache if full_prompt else None,
                )
                if row.input_tokens == 0 and row.output_tokens == 0:
                    row = None  # task released without timing data (cancel/error): drop silently
        else:
            row = self._bodies(line, now)
        self._startup(line)
        return row

    def kvmem_prompt_progress(self, line: str) -> PromptProgress | None:
        """kvmem 实时 prefill 行(实测格式,不带槽号,slot 归 None 由 store 归槽)。

        实测(21 行真机样本):该行约每 3.5s 一行,远慢于 1Hz 的 /slots 采样;
        progress 是相对固定总量的真实分数(该样本总量 ~103424 tok,每 2048 tok
        推进 0.02);rate = n_tokens / t 是累计均值,不是瞬时速度。"""
        m = _RE_KV_PROMPT_PROG.search(line)
        if not m:
            return None
        return PromptProgress(
            slot=None,
            n_tokens=int(m.group(1)),
            progress=float(m.group(2)),
            rate=float(m.group(3)) if m.group(3) else None,
        )

    def vanilla_prompt_progress(self, line: str) -> PromptProgress | None:
        """原版 llama-server 的 prompt processing 行:格式待实测(tasks 1.1),先不落。"""
        return None

    def _bodies(self, line: str, now: float) -> RequestRow | None:
        for key_word, rx, kind in _BODY_EXTRACTORS:
            if key_word not in line:
                continue
            m = rx.search(line)
            if m is None:
                continue                    # try the other dialect's pattern
            return self._body_match(kind, m, now)
        # family keyword present but no extractor could read it
        if any(k in line for k in _FAMILY_KEYS):
            self.failures += 1
        return None

    def _body_match(self, kind: str, m: re.Match, now: float) -> RequestRow | None:
        if kind.startswith("kv_"):
            rec = self._records.get(self._kv_key) if self._kv_key else None
            if rec is None:
                return None        # mid-stream attach: no record; not a failure
            if kind == "kv_prompt":
                rec.prompt_ms, rec.prompt_tok, rec.prompt_tps = _f(m, 1), int(_f(m, 2)), _f(m, 3)
                rec.cache_tok = int(m.group(4)) if m.group(4) else None
            elif kind == "kv_eval":
                rec.eval_ms, rec.eval_tok, rec.eval_tps = _f(m, 1), int(_f(m, 2)), _f(m, 3)
            else:                             # kv_total closes the block (no release line)
                rec.total_ms = _f(m, 1)
                return self._finish_kv(rec, now)
            return None
        key = (int(m.group(1)), int(m.group(2)))
        rec = self._records.get(key)
        if rec is None:
            return None  # mid-stream attach: no record to attach to; not a failure
        g = m.groups()
        if kind == "prompt":
            rec.prompt_ms, rec.prompt_tok, rec.prompt_tps = _f(m, 3), int(_f(m, 4)), _f(m, 5)
        elif kind == "eval":
            rec.eval_ms, rec.eval_tok, rec.eval_tps = _f(m, 3), int(_f(m, 4)), _f(m, 5)
        elif kind == "total":
            rec.total_ms = _f(m, 3)
        elif kind == "draft":
            rec.mtp = _f(m, 3)
            rec.mtp_accepted, rec.mtp_generated = int(g[3]), int(g[4])
            self.mtp_enabled = True
        return None

    def _finish_kv(self, rec: _Record, now: float) -> RequestRow | None:
        key = (rec.slot, rec.task)
        if key in self._released:
            return None
        self._released.add(key)
        self._records.pop(key, None)
        self._kv_key = None
        full = rec.n_prompt_full or rec.prompt_tok
        # kvmem reports the cache hits outright; other builds still derive them
        cache = rec.cache_tok if rec.cache_tok is not None else max(0, full - rec.prompt_tok)
        return RequestRow(
            wall_ts=now, slot=key[0], task=key[1],
            input_tokens=full, output_tokens=rec.eval_tok,
            in_tps=rec.prompt_tps, out_tps=rec.eval_tps,
            total_ms=rec.total_ms,
            mtp_accept=rec.mtp, mtp_accepted=rec.mtp_accepted,
            mtp_generated=rec.mtp_generated,
            kv_prompt_tokens=full or None, kv_cache_tokens=cache if full else None,
        )

    def _strata_settlement(self, line: str, now: float) -> RequestRow | None:
        """One Strata settlement line -> one finished RequestRow (single slot,
        synthetic task id).  Keyword present but unparseable counts as a failure,
        matching the registry's family-failure convention."""
        m = _RE_STRATA_SETTLE.search(line)
        if m is None:
            self.failures += 1
            return None
        g = m.groups()
        full, reused = int(g[0]), int(g[1])
        prompt_ms, eval_ms = float(g[3]), float(g[6])
        out_tok = int(g[5])
        in_tps = float(g[4]) if g[4] else 0.0
        out_tps = float(g[7]) if g[7] else 0.0
        mtp_accepted = int(g[8]) if g[8] is not None else None
        mtp_generated = int(g[9]) if g[9] is not None else None
        if mtp_accepted is not None and mtp_generated:
            mtp_accept: float | None = mtp_accepted / mtp_generated
            self.mtp_enabled = True
        else:
            mtp_accept = None
        if full == 0 and out_tok == 0:
            return None                       # cancelled/empty request: drop silently
        self._strata_task += 1
        return RequestRow(
            wall_ts=now, slot=0, task=self._strata_task,
            input_tokens=full, output_tokens=out_tok,
            in_tps=in_tps, out_tps=out_tps,
            total_ms=prompt_ms + eval_ms,
            mtp_accept=mtp_accept, mtp_accepted=mtp_accepted, mtp_generated=mtp_generated,
            kv_prompt_tokens=full or None, kv_cache_tokens=reused if full else None,
        )

    def _spec_stats(self, gen_tok: int, acc_tok: int) -> None:
        """kvmem spec statistics line: cumulative counters -> per-request delta."""
        self.mtp_enabled = True
        prev, self._spec_cum = self._spec_cum, (gen_tok, acc_tok)
        if prev is not None and (gen_tok < prev[0] or acc_tok < prev[1]):
            prev = None                       # counters rebased (server restarted)
        gen = gen_tok - (prev[0] if prev else 0)
        acc = acc_tok - (prev[1] if prev else 0)
        if gen <= 0:
            return
        val = (acc / gen, acc, gen)
        rec = self._records.get(self._kv_key) if self._kv_key else None
        if rec is not None:
            rec.mtp, rec.mtp_accepted, rec.mtp_generated = val
        else:
            self._spec_pending = val          # seen outside a request: next one takes it

    def _startup(self, line: str) -> None:
        if _RE_MTP_INIT.search(line):
            self.mtp_enabled = True
        if m := _RE_SLOTS.search(line):
            self.startup["n_slots"], self.startup["n_ctx_slot"] = int(m.group(1)), int(m.group(2))
        if m := _RE_MODEL.search(line):
            self.startup["model"] = m.group(1)
        if m := _RE_LISTEN.search(line):
            self.startup["listen"] = m.group(1)

    def reset_records(self) -> None:
        self._records.clear()
        self._released.clear()
        self._kv_key = None        # task counter keeps increasing: ids stay unique
        self._spec_cum = None      # new server: cumulative spec counters restart at 0
        self._spec_pending = None
