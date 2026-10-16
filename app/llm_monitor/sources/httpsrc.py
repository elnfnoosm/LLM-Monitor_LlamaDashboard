"""HTTP observation channel: /health (public), /props, /slots (bearer from proc
discovery). Attach-time probing yields a ServerDialect so version drift (b11223
vs master assumptions) is absorbed here and nowhere else.

Strata dialect: its /health answers ``"service": "strata"`` and its /slots view
is leaner than llama.cpp's (no n_decoded / no cache counters on the live slot).
The richer GET /metrics JSON carries the live request (state, prompt/generated
counts, prompt-read progress), so for Strata we synthesise llama.cpp-shaped
slots from that JSON here - every layer above keeps consuming one shape.
Server-wide avg speed comes from /metrics/prometheus, which Strata emits under
llamacpp:* gauge names (its own naming, noted below)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from ..model import ServerFacts

EXPECTED_SLOT_KEYS = {
    "id", "n_ctx", "is_processing", "id_task",
    "n_prompt_tokens", "n_prompt_tokens_processed", "n_prompt_tokens_cache",
    "next_token",
}


# Prometheus names llama.cpp exposes when --metrics is on; there is no global
# "avg speed" endpoint otherwise (verified: llama-kvmem-server has no /metrics at
# all - its web UI reads the per-request timings of the streaming response).
METRIC_OUT_KEYS = ("llamacpp:tokens_predicted_second", "tokens_predicted_second",
                   # Strata /metrics/prometheus: cumulative average gauges
                   "llamacpp:predicted_tokens_seconds")
METRIC_IN_KEYS = ("llamacpp:prompt_tokens_second", "prompt_tokens_second",
                  "llamacpp:prompt_tokens_seconds")


def parse_metrics(text: str) -> tuple[float | None, float | None]:
    """Prometheus 文本 -> (输出 avg t/s, 输入 avg t/s);取不到为 None。"""
    out: float | None = None
    inp: float | None = None
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        # metric names may carry labels ("name{k=v} 1") or not ("name 1",
        # which is how Strata's exposition is formatted)
        name = parts[0].split("{", 1)[0].strip()
        try:
            value = float(parts[-1])
        except ValueError:
            continue
        if out is None and name in METRIC_OUT_KEYS:
            out = value
        elif inp is None and name in METRIC_IN_KEYS:
            inp = value
    return out, inp


@dataclass
class ServerDialect:
    kind: str = "llama"               # "llama" | "strata" (from /health.service)
    props_ok: bool = False
    slots_ok: bool = False
    build_info: str | None = None
    model_alias: str | None = None
    model_path: str | None = None
    total_slots: int | None = None
    endpoint_slots: bool | None = None
    endpoint_metrics: bool | None = None
    default_n_ctx: int | None = None
    slot_keys: set[str] = field(default_factory=set)
    # where the server's own avg speed comes from: "metrics" (Prometheus) or "log"
    # (per-request settlement block, the same source the built-in web UI reads).
    avg_tps_source: str | None = None
    metrics_ok: bool | None = None

    @property
    def missing_slot_keys(self) -> set[str]:
        return EXPECTED_SLOT_KEYS - self.slot_keys


class HttpSource:
    def __init__(self, facts: ServerFacts, client: httpx.AsyncClient | None = None) -> None:
        self.facts = facts
        self.strata = facts.kind == "strata"
        self._client = client or httpx.AsyncClient(timeout=2.5)
        port = facts.port or 8080
        # server may bind 0.0.0.0; always probe via loopback
        self.base_url = f"http://127.0.0.1:{port}"
        headers = {}
        if facts.api_key:
            headers["Authorization"] = f"Bearer {facts.api_key}"
        self._headers = headers
        # Strata live-slot synthesis state: synthetic task id bumps on every
        # idle->busy edge, so the store's per-task rate anchors stay unique.
        self._st_task = 0
        self._st_busy = False

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict[str, Any]:
        r = await self._client.get(self.base_url + "/health")  # public endpoint
        r.raise_for_status()
        return r.json()

    async def props(self) -> dict[str, Any]:
        r = await self._client.get(self.base_url + "/props", headers=self._headers)
        r.raise_for_status()
        return r.json()

    async def slots(self) -> list[dict[str, Any]]:
        if self.strata:
            return await self._strata_slots()
        r = await self._client.get(self.base_url + "/slots", headers=self._headers)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []

    async def metrics(self) -> str:
        # Strata's plain /metrics is JSON (its Monitor tab); the Prometheus text
        # exposition lives at /metrics/prometheus.
        path = "/metrics/prometheus" if self.strata else "/metrics"
        r = await self._client.get(self.base_url + path, headers=self._headers)
        r.raise_for_status()
        return r.text

    async def metrics_json(self) -> dict[str, Any]:
        """Strata GET /metrics without a Prometheus Accept header -> its JSON view."""
        r = await self._client.get(self.base_url + "/metrics", headers=self._headers)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, dict) else {}

    async def avg_tps(self) -> tuple[float | None, float | None]:
        """服务端自报的 avg speed(仅 /metrics 可用时);否则 (None, None)。"""
        return parse_metrics(await self.metrics())

    async def _strata_slots(self) -> list[dict[str, Any]]:
        """Map Strata's /metrics JSON live view onto one llama.cpp-shaped slot.

        - reading:  n_prompt_tokens = full prompt size, processed = prompt_read
                    progress (0..total while the engine streams PP lines);
        - generating: prompt count keeps the generated tokens appended (the
                    llama-server convention), next_token.n_decoded = generated;
        - idle/unloaded: [] - no active slot, exactly like an idle llama-server.
        """
        m = await self.metrics_json()
        live = m.get("live") if isinstance(m.get("live"), dict) else {}
        state = live.get("state")
        busy = state in ("reading", "generating")
        if not busy:
            self._st_busy = False
            return []
        if not self._st_busy:
            self._st_task += 1
            self._st_busy = True
        try:
            ctx = int((m.get("engine") or {}).get("max_context") or 0)
            prompt = int(live.get("prompt_tokens") or 0)
            generated = int(live.get("generated") or 0)
        except (TypeError, ValueError):
            return []
        max_tokens = live.get("max_tokens")
        if state == "reading":
            processed = 0
            try:
                pr, total = live.get("prompt_read"), live.get("prompt_total")
                if isinstance(pr, int) and isinstance(total, int) and total > 0:
                    processed = max(0, min(pr, total))
            except TypeError:
                processed = 0
            n_prompt, n_decoded, remain = prompt, 0, -1
        else:
            processed = prompt
            n_prompt = prompt + generated
            n_decoded = generated
            remain = int(max_tokens) - generated if isinstance(max_tokens, int) else -1
        return [{
            "id": 0,
            "n_ctx": ctx,
            "is_processing": True,
            "id_task": self._st_task,
            "n_prompt_tokens": n_prompt,
            "n_prompt_tokens_processed": processed,
            "n_prompt_tokens_cache": 0,      # Strata exposes reuse per request in the log, not live
            "next_token": [{"n_decoded": n_decoded, "n_remain": remain}],
        }]

    async def probe_dialect(self) -> ServerDialect:
        d = ServerDialect(kind="strata" if self.strata else "llama")
        try:
            h = await self.health()
            if isinstance(h, dict) and h.get("service") == "strata":
                d.kind = "strata"
                self.strata = True
        except (httpx.HTTPError, ValueError):
            pass
        try:
            p = await self.props()
            d.props_ok = True
            d.build_info = p.get("build_info")
            d.model_alias = p.get("model_alias")
            d.model_path = p.get("model_path")
            d.total_slots = p.get("total_slots")
            d.endpoint_slots = p.get("endpoint_slots")
            d.endpoint_metrics = p.get("endpoint_metrics")
            dgs = p.get("default_generation_settings") or {}
            if isinstance(dgs.get("n_ctx"), int):
                d.default_n_ctx = dgs["n_ctx"]
        except (httpx.HTTPError, ValueError):
            pass
        if d.kind == "strata":
            # /slots is served (synthesised from /metrics here) and the
            # Prometheus exposition is always present on Strata.
            d.slots_ok = True
            d.endpoint_slots = True
            d.endpoint_metrics = True
            try:
                out, inp = parse_metrics(await self.metrics())
                d.metrics_ok = out is not None or inp is not None
                if d.metrics_ok:
                    d.avg_tps_source = "metrics"
            except (httpx.HTTPError, ValueError):
                d.metrics_ok = False
            if d.avg_tps_source is None:
                d.avg_tps_source = "log"     # Strata's settlement line carries per-request t/s
            return d
        try:
            s = await self.slots()
            if isinstance(s, list):
                d.slots_ok = True
                if s and isinstance(s[0], dict):
                    d.slot_keys = set(s[0].keys())
        except (httpx.HTTPError, ValueError):
            d.slots_ok = False
        if d.endpoint_metrics:
            try:
                out, inp = parse_metrics(await self.metrics())
                d.metrics_ok = out is not None or inp is not None
                if d.metrics_ok:
                    d.avg_tps_source = "metrics"
            except (httpx.HTTPError, ValueError):
                d.metrics_ok = False
        if d.avg_tps_source is None and (d.props_ok or d.slots_ok):
            d.avg_tps_source = "log"        # 与 web 界面同源,但按每请求落到日志结算块
        return d
