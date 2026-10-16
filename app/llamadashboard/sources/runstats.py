"""本模型周期累计(run totals)
================================

口径:从服务日志里【最后一次加载模型】开始,到当前为止的全部已结算请求。
本地服务换模型时日志会写一行  load_model: loading model '<路径>' ,
它就是"本模型运行期间"的天然起点;起点之前的记录全部不计,换模型后三个
累计自动从零重新开始。

为什么按日志而不是内存计数:
- 日志是权威记录,监控窗口关闭/重启后从日志重算,数字不丢(关着监控时
  发的请求也照算,因为服务自己一直在写日志);
- LM Studio 按"日期 + 单个 10MB"切分卷、Strata 可能单文件,本模块把同一
  来源的所有分卷按文件名时间序串起来,跨天、跨卷连续累计;
- 与下方历史列表(只留最近 1000 条)完全脱钩,累计不会在 1000 处封顶。

缓存:每个分卷的统计结果(文件大小/修改时间 + 段内计数)写入 state.json,
下次打开时没变化的分卷直接跳过,只全量扫一次正在写的那个文件。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .logparse import LineParser
from .profiles import LOG_GLOB

# srv  load_model: loading model 'D:\\...\\xxx.gguf'
_LOAD_RE = re.compile(r"load_model: loading model '([^']+)'")


@dataclass
class _Seg:
    """单个日志分卷的扫描状态。seg_* 只统计本文件最后一次换模型标记【之后】
    的结算行(文件里没有标记时就是整个文件)。"""
    size: int = 0
    mtime: float = 0.0
    seg_req: int = 0
    seg_in: int = 0
    seg_gen: int = 0
    has_load: bool = False
    model: str | None = None
    offset: int = 0           # 已完整消费到的字节位置(增量跟随用)
    partial: bytes = b""      # 末尾还没写完的半行
    parser: LineParser = field(default_factory=LineParser)
    dirty: bool = False


def enumerate_logs(target: Path) -> list[Path]:
    """按来源列出全部分卷(时间序):
    单文件(Strata)→ 它自己;profile 目录 → 顶层 server*.log;
    LM Studio 目录 → 递归 *.log(月份子目录),按相对路径排序(即时间序)。"""
    if target.is_file():
        return [target]
    if not target.is_dir():
        return []
    top = sorted((p for p in target.glob(LOG_GLOB) if p.is_file()),
                 key=lambda p: p.name)
    if top:
        return top
    return sorted((p for p in target.rglob("*.log") if p.is_file()),
                  key=lambda p: str(p.relative_to(target)))


class RunTotals:
    def __init__(self, cache: dict | None = None) -> None:
        # cache: {绝对路径: {size,mtime,seg_req,seg_in,seg_gen,has_load,model}}
        self._cache = cache if cache is not None else {}
        self._target: Path | None = None
        self._files: list[Path] = []
        self._segs: dict[Path, _Seg] = {}
        self.ready = False
        self.dirty = False
        self.totals = {"requests": 0, "input_tokens": 0, "generated": 0,
                       "model": None}

    @property
    def has_target(self) -> bool:
        return self._target is not None

    # ---------------- 绑定与初次扫描 ----------------

    def bind(self, target: Path | None) -> None:
        """重新绑定来源并完成全量扫描(首次无缓存时可能耗时,应在线程里调用)。"""
        self.ready = False
        self._target = target
        self._files = enumerate_logs(target) if target is not None else []
        self._segs = {}
        for p in self._files:
            seg = self._restore_or_scan(p)
            self._segs[p] = seg
        self._recompute()
        self.ready = True

    def _restore_or_scan(self, p: Path, full: bool = False) -> _Seg:
        st = p.stat()
        c = self._cache.get(str(p.resolve()))
        seg = _Seg(size=st.st_size, mtime=st.st_mtime)
        if (not full and isinstance(c, dict)
                and c.get("size") == st.st_size and c.get("mtime") == st.st_mtime):
            seg.seg_req = int(c.get("seg_req", 0))
            seg.seg_in = int(c.get("seg_in", 0))
            seg.seg_gen = int(c.get("seg_gen", 0))
            seg.has_load = bool(c.get("has_load", False))
            m = c.get("model")
            seg.model = m if isinstance(m, str) else None
            seg.offset, seg.partial = st.st_size, _tail_partial(p, st.st_size)
            return seg
        self._scan_file(seg, p, 0, b"")
        seg.dirty = True
        self._persist(p, seg)
        return seg

    def _scan_file(self, seg: _Seg, p: Path, start: int, partial: bytes) -> None:
        """从 start 字节起读全段并解析;若 start==0 则段计数先清零(整文件重扫)。
        seg.offset 是"已读到文件的哪个位置"(读游标),seg.partial 是游标之前
        末尾那行还没写完的内容,下次新字节接在它后面拼成完整行。"""
        if start == 0:
            seg.seg_req = seg.seg_in = seg.seg_gen = 0
            seg.has_load = False
            seg.model = None
        with p.open("rb") as fh:
            fh.seek(start)
            new_bytes = fh.read()
        lines, seg.partial = _join_lines(partial, new_bytes)
        seg.offset = start + len(new_bytes)
        for ln in lines:
            self._consume_line(seg, ln)

    def _consume_line(self, seg: _Seg, ln: str) -> None:
        m = _LOAD_RE.search(ln)
        if m:
            # 新模型起点:段计数归零,同时清掉旧模型的 launch/release 残留记录
            seg.has_load = True
            seg.model = Path(m.group(1).replace("\\", "/")).name
            seg.seg_req = seg.seg_in = seg.seg_gen = 0
            seg.parser.reset_records()
            seg.dirty = True
            return
        row = seg.parser.feed(ln, 0.0)
        if row is not None:
            seg.seg_req += 1
            seg.seg_in += row.input_tokens or 0
            seg.seg_gen += row.output_tokens or 0
            seg.dirty = True

    # ---------------- 运行中增量跟随 ----------------

    def poll(self) -> dict:
        """每秒级调用:发现新分卷、跟进当前分卷增长,返回最新累计。未就绪时返回零值。"""
        if not self.ready or self._target is None:
            return self.totals
        # 新分卷出现(跨天/10MB 滚动):把新文件纳入,先整卷扫一遍
        cur_files = enumerate_logs(self._target)
        for p in cur_files:
            if p not in self._segs:
                self._segs[p] = self._restore_or_scan(p)
        if cur_files != self._files:
            self._files = cur_files
        active = self._files[-1] if self._files else None
        if active is not None:
            try:
                st_now = active.stat()
                size = st_now.st_size
            except OSError:
                st_now, size = None, 0
            seg = self._segs[active]
            if size < seg.offset:                 # 文件被截断/重写
                seg = self._restore_or_scan(active, full=True)
                self._segs[active] = seg
            elif size > seg.offset:
                try:
                    with active.open("rb") as fh:
                        fh.seek(seg.offset)
                        chunk = fh.read(size - seg.offset)
                except OSError:
                    chunk = b""
                if chunk:
                    lines, seg.partial = _join_lines(seg.partial, chunk)
                    seg.offset += len(chunk)
                    if st_now is not None:
                        seg.size, seg.mtime = size, st_now.st_mtime
                    for ln in lines:
                        self._consume_line(seg, ln)
                    if seg.dirty:
                        self._persist(active, seg)
        self._recompute()
        return self.totals

    # ---------------- 汇总 ----------------

    def _recompute(self) -> None:
        ordered = [self._segs[p] for p in self._files if p in self._segs]
        boundary = -1
        model = None
        for i, s in enumerate(ordered):
            if s.has_load:
                boundary, model = i, s.model
        if boundary < 0:                         # 日志里没有换模型标记:全部计入
            boundary = 0
        chosen = ordered[boundary:]
        self.totals = {
            "requests": sum(s.seg_req for s in chosen),
            "input_tokens": sum(s.seg_in for s in chosen),
            "generated": sum(s.seg_gen for s in chosen),
            "model": model,
        }

    def mark_saved(self) -> None:
        """state.json 落盘后调用:清掉待保存标记(各分卷的 seg.dirty 也一并清)。"""
        self.dirty = False
        for s in self._segs.values():
            s.dirty = False

    def _persist(self, p: Path, seg: _Seg) -> None:
        self._cache[str(p.resolve())] = {
            "size": seg.size, "mtime": seg.mtime,
            "seg_req": seg.seg_req, "seg_in": seg.seg_in, "seg_gen": seg.seg_gen,
            "has_load": seg.has_load, "model": seg.model,
        }
        self.dirty = True
        seg.dirty = False


def _tail_partial(p: Path, size: int) -> bytes:
    """文件未变(走缓存)但可能还在写:取出末尾未完成的半行,便于和后续字节拼接。
    只回读最多 64KB;极端情况下末尾是超长行(请求体 JSON)无换行就放弃拼接。"""
    if size <= 0:
        return b""
    back = min(size, 65536)
    with p.open("rb") as fh:
        fh.seek(size - back)
        tail = fh.read()
    nl = tail.rfind(b"\n")
    return tail[nl + 1:] if nl >= 0 else b""


def _join_lines(carry: bytes, new_bytes: bytes) -> tuple[list[str], bytes]:
    """上次的半行 carry + 本次新字节 → (完整行列表, 新的末尾半行)。"""
    if not new_bytes:
        return [], carry
    buf = carry + new_bytes
    nl = buf.rfind(b"\n")
    if nl < 0:
        return [], buf
    lines = buf[:nl].decode("utf-8", errors="replace").splitlines()
    return [ln for ln in lines if ln.strip()], buf[nl + 1:]
