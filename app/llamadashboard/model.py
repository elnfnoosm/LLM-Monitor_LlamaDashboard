"""Data model shared by sources, store and UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _flag(args: dict[str, str], *names: str) -> str | None:
    for n in names:
        if n in args:
            return args[n]
    return None


@dataclass
class PromptProgress:
    """Real-time prefill progress from the log's `prompt processing` line.

    kvmem prints this line during prompt (prefill) - unlike the HTTP
    `n_prompt_tokens_processed`, which only moves on the settlement frame for that
    dialect. `slot` is None when the line carries no id (kvmem) so the store can
    attribute it to the active slot."""

    slot: int | None = None
    n_tokens: int = 0
    progress: float = 0.0        # 0..1
    rate: float | None = None    # tokens per second, as reported


EXTRA_SOURCE = "记住的"       # source 标记:来自本机状态文件的 extra_profiles(可删除)


@dataclass
class ServerFacts:
    """Facts derived from the live process command line (never from config files)."""

    pid: int
    exe: str
    cmdline: list[str]
    args: dict[str, str] = field(default_factory=dict)
    # "llama": llama.cpp server exe; "strata": Strata's python serve/server.py
    # (its engine exe is a child; connection facts come from its --config json)
    kind: str = "llama"
    log_path: str | None = None     # Strata: per-run log named inside its config json
    label: str | None = None        # Strata: config's model_name, for the header

    @property
    def port(self) -> int | None:
        v = _flag(self.args, "--port")
        return int(v) if v and v.isdigit() else None

    @property
    def api_key(self) -> str | None:
        return _flag(self.args, "--api-key")

    @property
    def model_path(self) -> str | None:
        return _flag(self.args, "-m", "--model")

    @property
    def alias(self) -> str | None:
        return _flag(self.args, "--alias", "-ma")

    @property
    def n_ctx(self) -> int | None:
        v = _flag(self.args, "-c", "--ctx-size", "--ctx-len")
        return int(v) if v and v.lstrip("-").isdigit() else None

    @property
    def n_parallel(self) -> int | None:
        v = _flag(self.args, "--parallel", "-np")
        return int(v) if v and v.isdigit() else None

    @property
    def host(self) -> str:
        return _flag(self.args, "--host", "-h") or "127.0.0.1"

    @property
    def mtp(self) -> bool:
        # --spec-type draft-mtp (any explicit spec-type counts as speculative decoding)
        return (_flag(self.args, "--spec-type") or "") != ""


@dataclass
class Profile:
    name: str
    dir: Path
    bat: Path
    model_path: str | None = None  # read from the recipe only for matching, never for control
    source: str | None = None      # starts-root name / "记住的" for state extras
    display: str = ""              # disambiguated label for pickers


@dataclass
class RequestRow:
    """One settled request (authoritative source: server log)."""

    wall_ts: float            # when the dashboard observed completion
    slot: int
    task: int
    input_tokens: int = 0
    output_tokens: int = 0
    in_tps: float = 0.0
    out_tps: float = 0.0
    total_ms: float = 0.0
    mtp_accept: float | None = None   # None => MTP not enabled / absent in line
    mtp_accepted: int | None = None
    mtp_generated: int | None = None
    kv_cache_tokens: int | None = None  # from /slots first-sight snapshot
    kv_prompt_tokens: int | None = None

    @property
    def kv_ratio(self) -> float | None:
        if self.kv_cache_tokens is not None and self.kv_prompt_tokens:
            return self.kv_cache_tokens / self.kv_prompt_tokens
        return None


@dataclass
class SlotLive:
    id: int
    n_ctx: int
    is_processing: bool
    id_task: int = -1
    n_prompt_tokens: int = 0
    n_prompt_tokens_processed: int = 0
    n_prompt_tokens_cache: int = 0
    n_decoded: int = 0
    n_remain: int = -1
    speculative: bool = False
    phase: str = "idle"        # prompt | decode | idle
    fresh_prompt: int = 0      # this-turn prompt tokens (excludes appended generation)
    eval_total: int = 0        # tokens that must be computed (fresh_prompt - cached)
    rate_in: float | None = None   # window-averaged prompt t/s (this request)
    rate_out: float | None = None  # window-averaged decode t/s (this request)
    ts: float = 0.0


@dataclass
class GpuSample:
    name: str = ""
    mem_used_mib: float | None = None
    mem_total_mib: float | None = None
    util_gpu: int | None = None
    util_mem: int | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    clk_core: int | None = None
    clk_mem: int | None = None
    pstate: str | None = None
    fan_pct: int | None = None
    temp_c: int | None = None
    slowdown_c: int | None = None
    throttle: str | None = None      # human readable, e.g. 功率受限
    ts: float = 0.0


@dataclass
class SysSample:
    """整机 CPU/内存占用(psutil),与具体被监控进程无关。"""
    cpu_name: str = ""                # 简短型号,如 Ryzen 7 9700X / i9-9700X
    cpu_pct: float | None = None        # 全机 CPU 使用率 0..100
    mem_pct: float | None = None        # 内存使用率 0..100
    mem_used_gb: float | None = None    # 已用 = 总量 - 可用(含缓存可回收部分)
    mem_total_gb: float | None = None
    ts: float = 0.0
