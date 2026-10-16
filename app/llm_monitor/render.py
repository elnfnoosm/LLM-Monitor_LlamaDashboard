"""Pure renderable builders (Rich) shared by the TUI - no Textual imports here,
so layout content can be unit-tested off-screen."""

from __future__ import annotations

import time

from rich.console import Console, Group
from rich.table import Table
from rich.text import Text, cell_len

from .health import HealthBoard
from .model import SlotLive
from .sources.proc import short_names
from .store import SessionStore

BAR_W = 22
# 一个带 round 边框 + `padding: 0 1` 的面板在内容宽度之外占的额外列数
PANEL_CHROME_W = 4


def content_width(renderable) -> int:
    """渲染体的自然宽度(按终端列计,CJK 记 2 列)。"""
    c = Console(width=200, force_terminal=False)
    with c.capture() as cap:
        c.print(renderable)
    return max((cell_len(ln) for ln in cap.get().splitlines()), default=0)


def fmt_uptime(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _num(v: float | None, unit: str = "", nd: int = 1) -> str:
    if v is None:
        return "—"
    return f"{v:.{nd}f}{unit}"


def _pct(v: float | None) -> str:
    return f"{v * 100:.1f}%" if v is not None else "—"


def _bar(pct: float | None, color: str, width: int = BAR_W) -> Text:
    if pct is None:
        return Text("—")
    n = max(0, min(width, round(pct * width)))
    return Text("█" * n + "░" * (width - n), style=color)


def _pct_color(p: float) -> str:
    return "red" if p >= 0.9 else ("yellow" if p >= 0.7 else "green")


# ---- 实时框数字定宽(面板宽度在待机/空闲/生成中必须完全一致,不能随数字位数跳) ----
_TOK_W = 6   # token 数统一 6 位宽(覆盖到 999999,含 131072 上下文);待机也按此预留


def _dw(w: int) -> str:
    """无值占位:'—' 前面补空格凑足 w 个显示列。破折号在终端按 1 列算,
    宽度必须用 cell_len 实测,不能想当然按 2 列。"""
    return " " * max(0, w - cell_len("—")) + "—"


def _tok(v: int | None) -> str:
    """token 数:定宽右对齐;无值时 '—' 占据同样的显示宽度。"""
    return f"{v:{_TOK_W}d}" if v is not None else _dw(_TOK_W)


def _tps(v: float | None) -> str:
    """速度:统一 '9999.9 t/s' 宽(6 位数字 + 后缀),无值时等宽占位。"""
    return f"{v:6.1f} t/s" if v is not None else _dw(6) + " t/s"


def _remain(v: int | None) -> str:
    """剩余 token:数字定宽;'不限'(2 个汉字=4 格)补空格到同样的 6 格。"""
    if v is not None and v < 0:
        return " " * (_TOK_W - cell_len("不限")) + "不限"
    return _tok(v)


def _avg_label(server_val: float | None, session_val: float | None) -> str:
    """运行均值:优先服务端自报值,取不到时降级为会话均值并标注来源。"""
    if server_val is not None:
        return f"运行均值 {_num(server_val, '', 1)}"
    return f"运行均值 {_num(session_val, '', 1)} (会话)"


def header_text(store: SessionStore) -> Text:
    t = Text()
    f, d = store.facts, store.dialect
    if f is None:
        t.append(f"待机:未发现 server 进程({' / '.join(short_names())} / Strata 的 serve\\server.py)")
        return t
    model = ""
    if f.model_path:
        model = f.model_path.replace("\\", "/").rsplit("/", 1)[-1]
        model = model[:-5] if model.endswith(".gguf") else model
    elif getattr(f, "label", None):
        model = f.label
    ctx = f.n_ctx or (d.default_n_ctx if d else None)
    if ctx is None and store.slots:
        ctx = next(iter(store.slots.values())).n_ctx
    build = d.build_info if d else "—"
    t.append("模型 ", style="dim")
    t.append(model or (d.model_alias if d else "") or "—", style="bold")
    t.append(f"  ctx ", style="dim"); t.append(f"{ctx if ctx else '—'}")
    t.append("  运行 ", style="dim"); t.append(fmt_uptime(store.uptime_s()), style="bold")
    t.append("  build ", style="dim"); t.append(build or "—")
    if store.attach_mode:
        t.append("  [attach]", style="magenta")
    return t


def metrics_text(store: SessionStore) -> Table:
    tb = Table.grid(padding=(0, 3))
    tot = store.totals()
    cur = store.current_slot()
    last = store.last_request()
    out_tps = store.live_out_tps or (last.out_tps if last else None)
    in_tps = store.prompt_rate()          # 实时/冻结值,不回落到上一请求
    # KV: 累计 + 本次(当前请求优先)
    kv_cum = tot["kv_cum"]
    kv_cur: float | None = None
    if cur is not None and cur.fresh_prompt:
        kv_cur = cur.n_prompt_tokens_cache / cur.fresh_prompt
    elif (last := store.last_request()) is not None and last.kv_ratio is not None:
        kv_cur = last.kv_ratio
    mtp_val: float | None = None
    gen_w = 0
    num = 0.0
    for r in store.history:
        if r.mtp_accept is not None and r.mtp_generated:
            num += r.mtp_accept * r.mtp_generated
            gen_w += r.mtp_generated
    mtp_val = num / gen_w if gen_w else None
    mtp_on = (bool(store.facts and store.facts.mtp)
              or (cur is not None and cur.speculative)
              or bool(store.mtp_enabled))

    cells: list[tuple[str, Text, Text]] = []
    v = Text(_num(out_tps, " t/s"), style="bold green") if out_tps else Text("—")
    cells.append(("输出速度", v, Text(_avg_label(store.server_avg_out_tps, tot["avg_out_tps"]),
                                     style="dim")))
    v = Text(_num(in_tps, " t/s"), style="bold cyan") if in_tps else Text("—")
    cells.append(("输入速度", v, Text(_avg_label(store.server_avg_in_tps, tot["avg_in_tps"]),
                                     style="dim")))
    cells.append(("KV 命中率", Text(_pct(kv_cum), style="bold magenta"),
                  Text(f"本次 {_pct(kv_cur)}", style="dim")))
    # MTP:有数据显示百分比;启用但日志没给统计(如 verbosity 不够)时明确区分
    if mtp_val is None:
        mtp_sub = "启用·无统计" if mtp_on else "未启用"
    else:
        mtp_sub = "本次监控"
    cells.append(("MTP 接受率",
                  Text(_pct(mtp_val), style="bold yellow") if mtp_val is not None else Text("—"),
                  Text(mtp_sub, style="dim")))
    cells.append(("累计请求", Text(f"{tot['requests']}", style="bold"), Text("本模型周期", style="dim")))
    cells.append(("累计输入", Text(f"{tot['input_tokens']}", style="bold cyan"), Text("输入 token", style="dim")))
    cells.append(("累计生成", Text(f"{tot['generated']}", style="bold blue"), Text("输出 token", style="dim")))

    tb.add_row(*[lbl for lbl, _, _ in cells], style="dim")
    tb.add_row(*[v for _, v, _ in cells])
    tb.add_row(*[s for _, _, s in cells])
    return tb


HW_BAR_W = 10        # 硬件框左列 GPU 的短进度条;右列 CPU/内存用彩色百分比节省宽度


def gpu_text(store: SessionStore, detail: bool, inner_width: int | None = None):
    """硬件面板:左列 GPU,右列整机 CPU/内存;左右各 4/2 行,默认与两个实时框等高。

    inner_width 给定时(框被拉宽),右列整块向右靠:在右列内容前补空格,
    使右列结束于 inner_width 内缘;CPU 型号行用实测列位置对齐,自动跟随。
    None 时按自然宽度渲染(用于测量面板最小宽度)。
    """
    g = store.gpu
    sy = store.sys
    gpu_name = (Text(f"{g.name}" + (f"  {g.mem_total_mib / 1024:.0f}GB" if g.mem_total_mib else ""),
                     style="bold", no_wrap=True)
                if g is not None else Text("GPU 不可用（仅显示 CPU/内存）", style="dim"))
    cpu_name = sy.cpu_name if sy is not None and sy.cpu_name else ""
    bar_w = HW_BAR_W       # 极窄窗口时会自动缩短给右列腾位置

    def hbar(p: float | None) -> Text:
        return _bar(p, _pct_color(p), bar_w) if p is not None else _bar(None, "green", bar_w)

    def sys_cell(kind: str, extra_pad: int = 0, mem_gb: bool = True) -> Text:
        """右列:开头 1 个空格(加表格留白共 3 个空格)与左半拉开距离,extra_pad
        为框变宽时额外补的前导空格;再显示标签 + 彩色百分比 + 内存 GB。
        no_wrap 保证窄窗下绝不折行;窄到放不下时由调用方传 mem_gb=False 省略 GB。"""
        t = Text(" " * (1 + extra_pad)
                 + (kind if cell_len(kind) >= 4 else kind + " ")   # "CPU"补1格,与"内存"等宽
                 + " ", style="dim", no_wrap=True)
        if sy is None:
            t.append("—")
            return t
        if kind == "CPU":
            pct, suffix = sy.cpu_pct, ""
        else:
            pct = sy.mem_pct
            if mem_gb and sy.mem_used_gb is not None and sy.mem_total_gb:
                w = len(f"{sy.mem_total_gb:.0f}")   # 数字按本机总量位数定宽右对齐
                suffix = f" {sy.mem_used_gb:{w}.0f}/{sy.mem_total_gb:.0f}GB"
            else:
                suffix = ""
        if pct is None:
            t.append("—")
        else:
            t.append(f"{pct:3.0f}%", style=f"bold {_pct_color(pct / 100)}")
        t.append(suffix, style="dim")
        return t

    def build(extra_pad: int, mem_gb: bool = True, gpu_gb: bool = True) -> Table:
        tb = Table.grid(padding=(0, 1))

        def row(label: str, bar: Text, value: Text, right: Text | None = None):
            tb.add_row(Text(label, style="dim", no_wrap=True), bar, value, right or Text(""))

        # 行1:显存 | CPU(百分比在前,GB 在后,百分比与内存一样带绿/黄/红颜色)
        if g is not None and g.mem_total_mib and g.mem_used_mib is not None:
            p = g.mem_used_mib / g.mem_total_mib
            vm = Text(no_wrap=True)   # 注意:不能给整体 Text 设底色,否则 append 的 dim 会和红色叠加变暗红
            vm.append(f"{p * 100:3.0f}%", style=f"bold {_pct_color(p)}")
            if gpu_gb:
                used_gb, total_gb = g.mem_used_mib / 1024, g.mem_total_mib / 1024
                w = len(f"{total_gb:.1f}")
                vm.append(f" {used_gb:{w}.1f}/{total_gb:.1f}GB", style="dim")
            row("显存", hbar(p), vm, sys_cell("CPU", extra_pad, mem_gb))
        else:
            row("显存", _bar(None, "green", bar_w), Text("—"), sys_cell("CPU", extra_pad, mem_gb))
        # 行2:带宽 | 内存
        if g is not None and g.util_mem is not None:
            row("带宽", hbar(g.util_mem / 100), Text(f"{g.util_mem:3.0f}%"),
                sys_cell("内存", extra_pad, mem_gb))
        else:
            row("带宽", _bar(None, "green", bar_w), Text("—"),
                sys_cell("内存", extra_pad, mem_gb))
        # 行3:功率(右半留白)
        if g is not None and g.power_w is not None:
            p = g.power_w / g.power_limit_w if g.power_limit_w else None
            if g.power_limit_w:
                w = len(f"{g.power_limit_w:.0f}")   # 按功率上限位数定宽
                pw_text = Text(f"{g.power_w:{w}.0f}/{g.power_limit_w:.0f}W")
            else:
                pw_text = Text(f"{g.power_w:3.0f} W")
            row("功率", hbar(p), pw_text)
        else:
            row("功率", _bar(None, "green", bar_w), Text("—"))
        # 行4:温度(精简成 GPU 温度 + 风扇;显存/热点温度挪到 h 详情)
        if g is not None and g.temp_c is not None:
            t1 = Text(f"GPU {g.temp_c:3.0f} C")
        else:
            t1 = Text("GPU   — C")
        if g is not None and g.fan_pct is not None:
            t2 = Text(f"风扇 {g.fan_pct:3.0f}%")
        else:
            t2 = Text("风扇   — %")
        row("温度", t1, t2)
        return tb

    def build_detail() -> list[Text]:
        """h 键展开的详情行(频率 / 显存·热点温度 / 分区温度)。

        必须作为独立文字行挂在表格【外面】:它们若放进上面的 Table.grid,长文字
        会把第 2 列(进度条列,宽 10)撑到 15+,硬件框随之变宽(55→60,分区多时
        可达 89),把两个实时框挤到下一行。独立行不参与列宽计算,视觉位置不变:
        标签与"显存/带宽"对齐,内容从进度条起点开始。分区行封顶 51 格,与普通
        模式框宽 55 留有余量。
        """
        if not detail:
            return []
        rows: list[Text] = []
        if g is not None:
            core = f"{g.clk_core:4.0f}" if g.clk_core is not None else "   —"
            mem_clk = f"{g.clk_mem:4.0f}" if g.clk_mem is not None else "   —"
            t = Text()
            t.append("频率 ", style="dim")
            t.append(f"核心 {core} MHz   显存 {mem_clk} MHz")
            rows.append(t)
        extra = store.gpu_extra
        t = Text()
        t.append("温度 ", style="dim")
        t.append(f"显存 {extra['显存']:.0f} C" if "显存" in extra else "显存 —")
        t.append(f"   热点 {extra['热点']:.0f} C" if "热点" in extra else "   热点 —")
        rows.append(t)
        parts = [f"{k} {v:.0f} C" for k, v in extra.items() if k.startswith("区")]
        zones = ""
        for p in parts:                    # 逐个拼,超出 46 格就收尾(含标签整行≤51)
            cand = p if not zones else zones + "    " + p
            if cell_len(cand) <= 46:
                zones = cand
            else:
                zones += " …"
                break
        t = Text("分区 " + (zones or "驱动不提供(—)"), style="dim")
        rows.append(t)
        return rows

    detail_rows = build_detail()

    # 自适应(inner_width 给定时):
    # 宽屏时右列(CPU/内存、CPU 型号)整块贴着框右内缘;窗口缩窄时右列随右缘平滑
    # 左移,直到它和左边显卡内容排满 —— 此刻正好是顶部两个硬件名字(NVIDIA / Ryzen)
    # 快接近的位置,这时才省略内存 GB;右列随即停住不回跳(不会凭空多出一块空白)。
    # 继续变窄才依次缩短进度条、显存只留百分比。任何情况下两个名字都同一行,
    # 实在放不下时宁可被边框裁掉也不换行。
    tb = build(0)
    extra_pad = 0
    mem_gb = True
    gpu_gb = True
    if inner_width is not None:
        import io

        def base_x_of(tbl) -> int:
            buf = io.StringIO()
            Console(file=buf, force_terminal=False, width=4096).print(tbl)
            first = buf.getvalue().splitlines()[0]
            idx = first.find("CPU")
            if idx < 0:
                return cell_len(first.rstrip())
            return cell_len(first[:idx])   # 行首有"显存"等中文,必须按显示宽度算

        base_x = base_x_of(tb)
        if sy is None:
            r_full = r_min = max(cell_len("CPU  —"), cell_len("内存 —"))
        else:
            cpu_w = cell_len(f"CPU  {sy.cpu_pct:3.0f}%") if sy.cpu_pct is not None else cell_len("CPU  —")
            mem_pct_w = cell_len(f"内存 {sy.mem_pct:3.0f}%") if sy.mem_pct is not None else cell_len("内存 —")
            if sy.mem_used_gb is not None and sy.mem_total_gb:
                gw = len(f"{sy.mem_total_gb:.0f}")
                mem_gb_w = cell_len(f" {sy.mem_used_gb:{gw}.0f}/{sy.mem_total_gb:.0f}GB")
            else:
                mem_gb_w = 0
            r_full = max(cpu_w, mem_pct_w + mem_gb_w)   # 右列完整宽(含 GB)
            r_min = max(cpu_w, mem_pct_w)               # 右列最简宽(只留百分比)

        def shrink_bar(need: int) -> None:
            # 排满仍放不下:逐格缩短进度条(右列起点随之左移),最短保留 4 格
            nonlocal bar_w, tb, base_x
            while bar_w > 4 and base_x + need > inner_width:
                bar_w -= 1
                tb = build(0, mem_gb, gpu_gb)
                base_x = base_x_of(tb)

        # 两列排满(两个硬件名字接近)仍放不下完整右列 → 这才省略内存 GB
        if base_x + r_full > inner_width:
            mem_gb = False
            tb = build(0, mem_gb, gpu_gb)
            base_x = base_x_of(tb)
        shrink_bar(r_full if mem_gb else r_min)
        # 极窄窗口:显存数字也只留百分比,再继续缩进度条
        if base_x + (r_full if mem_gb else r_min) > inner_width:
            gpu_gb = False
            tb = build(0, mem_gb, gpu_gb)
            base_x = base_x_of(tb)
            shrink_bar(r_min)
        # 右列位置:能靠右就靠右;排满后停在 base_x,省略 GB 后也不回跳
        col_x = max(base_x, inner_width - r_full)
        extra_pad = col_x - base_x
        tb = build(extra_pad, mem_gb, gpu_gb)

    # 名称行:GPU 名在左,CPU 型号与下方右列的 "CPU/内存" 文字左对齐(列位置实测)。
    # 两个名字始终在同一行;窄到极限时超出部分由边框裁掉,绝不换行。
    header = gpu_name
    if cpu_name:
        import io
        buf = io.StringIO()
        Console(file=buf, force_terminal=False, width=4096).print(tb)
        first = buf.getvalue().splitlines()[0]
        idx = first.find("CPU")
        cpu_x = cell_len(first.rstrip()) if idx < 0 else cell_len(first[:idx])
        g_name_w = content_width(gpu_name)
        cpu_name_w = cell_len(cpu_name)
        # 正常与下方 CPU/内存 列对齐;窄到极限时允许名字再靠近(最小间距 2),
        # 让两个名字恰好排满 = 面板最小宽度;再窄就裁切,也不换行。
        pad = max(2, cpu_x - g_name_w)
        if inner_width is not None:
            pad = max(2, min(pad, inner_width - g_name_w - cpu_name_w))
        header = gpu_name.copy()
        header.append(" " * pad + cpu_name, style="bold")
    return Group(header, tb, *detail_rows)


def _ctx_total(store: SessionStore, cur: SlotLive | None) -> int | None:
    """上下文总长:当前槽 → 命令行 -c → 任一槽快照。"""
    if cur is not None and cur.n_ctx:
        return cur.n_ctx
    if store.facts is not None and store.facts.n_ctx:
        return store.facts.n_ctx
    for s in store.slots.values():
        if s.n_ctx:
            return s.n_ctx
    return None


def _peak_text(peak: int | None, ctx: int | None) -> str:
    """上下文峰值(紧凑等宽行):'峰值 131.1k/131.1k 100%'。
    所有数字定宽,无值时 '—' 等宽占位,任何状态下本行长度一致。"""
    pk = f"{peak / 1000:5.1f}k" if peak is not None else _dw(5) + "k"
    cx = f"{ctx / 1000:5.1f}k" if ctx else _dw(5) + "k"
    pct = (f"{peak / ctx * 100:3.0f}%" if (peak is not None and ctx)
           else " " * (4 - cell_len("—%")) + "—%")
    return f"峰值 {pk}/{cx} {pct}"


def live_prompt_text(store: SessionStore) -> Table:
    """实时 prompt 面板:阶段、处理进度、输入速度、本次(空闲时上次)输入与缓存。

    四个字段常显(默认就是最全的字段集):当前取不到实时值时回落到最近一次
    有效值(冻结的 prompt 速率 / 上一请求的输入与缓存),从未有值时显示 `—`,
    面板不会因阶段切换而少行。
    """
    cur = store.current_slot()
    last = store.last_request()
    tb = Table.grid(padding=(0, 1))
    tb.add_row(Text("prompt", style="bold cyan"))
    if store.facts is None:                       # 待机:字段仍常显,值全是等宽占位
        tb.add_row(Text("阶段 待机", style="dim"))
        tb.add_row(Text(f"处理输入 {_tok(None)}/{_tok(None)}", style="dim"))
        tb.add_row(Text("输入速度 ") + Text(_tps(None), style="bold cyan"))
        tb.add_row(Text(f"输入 {_tok(None)} 缓存 {_tok(None)}", style="dim"))
        return tb
    rate = store.prompt_rate()                    # prompt 中=实时值,之后=冻结值
    if cur is None:                               # 空闲:没有进行中的进度,回落到上次值
        speed = rate if rate is not None else (last.in_tps if last else None)
        tb.add_row(Text("阶段 空闲", style="bold green"))
        tb.add_row(Text(f"处理输入 {_tok(None)}/{_tok(None)}", style="dim"))
        tb.add_row(Text("输入速度 ") + Text(_tps(speed), style="bold cyan"))
        shown = last.input_tokens if last else None
        cache = last.kv_cache_tokens if last else None
        tb.add_row(Text(f"上次 {_tok(shown)} 缓存 {_tok(cache)}", style="dim"))
        return tb
    phase = "处理中" if cur.phase == "prompt" else "已完成"
    tb.add_row(Text(f"阶段 {phase}"))
    if cur.phase == "prompt":
        prog = store.prompt_progress_for(cur.id)
        if prog is not None:                       # (a) log 实时进度
            tb.add_row(Text(f"处理输入 {prog[0] * 100:4.0f}%"))
        else:                                      # (b) 该版本无实时行,回落到已用时间
            el = store.prompt_elapsed_s(cur.id)
            tb.add_row(Text(f"处理输入 {el:4.0f}s" if el is not None
                            else f"处理输入 {_dw(4)}"))
    else:                                          # (c) 结算后
        tb.add_row(Text(f"处理输入 {_tok(cur.n_prompt_tokens_processed)}/"
                        f"{_tok(cur.eval_total)}"))
    tb.add_row(Text("输入速度 ") + Text(_tps(rate), style="bold cyan"))
    tb.add_row(Text(f"输入 {_tok(cur.fresh_prompt or cur.n_prompt_tokens)} "
                    f"缓存 {_tok(cur.n_prompt_tokens_cache)}"))
    return tb


def live_decode_text(store: SessionStore) -> Table:
    """实时 decode 面板:阶段、已生成/剩余、输出速度、上下文峰值(紧凑等宽,
    同 prompt 面板的常显规则)。"""
    cur = store.current_slot()
    last = store.last_request()
    tb = Table.grid(padding=(0, 1))
    tb.add_row(Text("decode", style="bold green"))
    if store.facts is None:
        tb.add_row(Text("阶段 待机", style="dim"))
        tb.add_row(Text(f"生成 {_tok(None)} 剩余 {_tok(None)}", style="dim"))
        tb.add_row(Text("输出速度 ") + Text(_tps(None), style="bold green"))
        tb.add_row(Text(_peak_text(None, None), style="dim"))
        return tb
    if cur is None:
        gen = last.output_tokens if last else None
        speed = last.out_tps if last else None
        peak = (last.input_tokens + last.output_tokens) if last else None
        tb.add_row(Text("阶段 空闲", style="bold green"))
        tb.add_row(Text(f"生成 {_tok(gen)} 剩余 {_tok(None)}", style="dim"))
        tb.add_row(Text("输出速度 ") + Text(_tps(speed), style="bold green"))
        tb.add_row(Text(_peak_text(peak, _ctx_total(store, cur)), style="dim"))
        return tb
    tb.add_row(Text("解码中" if cur.phase == "decode" else "等待生成"))
    tb.add_row(Text(f"生成 {_tok(cur.n_decoded)} 剩余 {_remain(cur.n_remain)}"))
    rate = cur.rate_out if cur.rate_out is not None else store.live_out_tps
    tb.add_row(Text("输出速度 ") + Text(_tps(rate), style="bold green"))
    peak = cur.n_prompt_tokens              # already includes appended generation
    tb.add_row(Text(_peak_text(peak, _ctx_total(store, cur))))
    return tb


def panel_width(renderable) -> int:
    """带框面板需要的最小宽度:文字内容宽度 + 边框与内边距。"""
    return content_width(renderable) + PANEL_CHROME_W


def history_panel_width(store: SessionStore, top_widths: list[int], stacked: bool,
                        available: int, max_rows: int = 10) -> int:
    """请求历史面板宽度:先满足内容最小宽度,再为对齐适当变宽。

    - 三框同行:目标 = 三框宽度之和(与上方行等宽);
    - 三框堆叠:目标 = 四个面板中最宽者(与上方框同宽);
    - 内容本身更宽时以内容为准;最终不超过终端可用宽度。
    """
    need = panel_width(history_text(store, max_rows=max_rows))
    target = max(top_widths + [need]) if stacked else max(sum(top_widths), need)
    return min(target, available) if available > 0 else target


def row_fits(panels, available: int) -> bool:
    """三框(各带边框与内边距)能否在同一行里完整排下。"""
    return sum(content_width(p) + PANEL_CHROME_W for p in panels) <= available


def history_text(store: SessionStore, max_rows: int = 10) -> Table:
    """请求历史表:列宽由表头与可见行内容自适应(数字右对齐),列间只留必要 padding。

    不写死每列宽度:固定宽度会为最坏情况预留空间,数值位数少时列内剩余宽度
    全变成左侧空白(实测「槽」与「输入」之间会出现 8 列空档)。
    """
    tb = Table(box=None, pad_edge=False, expand=False)
    tb.title = None
    for col in ("时间", "槽", "输入", "输出", "输入t/s", "输出t/s", "KV", "MTP", "耗时ms"):
        tb.add_column(col, style="dim", justify="right")
    for r in list(store.history)[:max_rows]:
        kv = f"{r.kv_ratio * 100:.1f}%" if r.kv_ratio is not None else "—"
        mtp = f"{r.mtp_accept * 100:.1f}%" if r.mtp_accept is not None else "—"
        tb.add_row(
            time.strftime("%H:%M:%S", time.localtime(r.wall_ts)),
            str(r.slot), f"{r.input_tokens}", f"{r.output_tokens}",
            f"{r.in_tps:.1f}", f"{r.out_tps:.1f}", kv, mtp, f"{r.total_ms:.0f}",
        )
    if not store.history:
        tb.add_row("—", "", "", "", "", "", "", "", "")
    return tb


def status_text(store: SessionStore, health: HealthBoard, interval: float,
                paused: bool, now: float | None = None) -> Text:
    age = health.data_age(now)
    s, f = health.totals()
    vec = "/".join(str(v) for v in health.vector(now))
    t = Text()
    t.append(f"数据 {age:.0f}s 前" if age is not None else "数据 尚无", style="dim")
    t.append(f"   采样成功 {s} / 失败 {f}")
    t.append(f"   采样 {interval:.0f}s   src {vec}", style="dim")
    if paused:
        t.append("   [已暂停 p 恢复]", style="bold yellow")
    if store.facts is None:
        t.append("   [待机]", style="cyan")
    return t


HINT = "q 退出  m 换模型  s 停止  r 重启  l 日志  p 暂停  c 清历史  h 温度"
