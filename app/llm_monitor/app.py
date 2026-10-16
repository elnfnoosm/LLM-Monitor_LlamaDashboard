"""Textual app: 1Hz main screen, key bindings, modal overlays (log / model
picker / confirm-with-default-cancel)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView, RichLog, Static

from . import fsbrowse
from . import render
from . import state as state_mod
from .config import Settings
from .health import HealthBoard
from .model import EXTRA_SOURCE, Profile
from .runner import Runner
from .sources import profiles, proc
from .store import SessionStore


class ConfirmScreen(ModalScreen[bool]):
    """Default focus = cancel: any key other than Enter/y dismisses with False."""

    CSS = """
    ConfirmScreen { align: center middle; }
    #cf-box { width: auto; min-width: 46; padding: 1 2; border: thick $warning; background: $surface; }
    #cf-hint { color: $text 60%; }
    """

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="cf-box"):
            yield Label(f"[bold]{self.message}[/bold]")
            yield Label("Enter/y 确认 · 其他任意键取消", id="cf-hint")

    def on_key(self, event) -> None:
        # confirm keys pass; ANY other key (incl. Esc) cancels — default focus = cancel
        event.stop()
        event.prevent_default()
        if event.key in ("enter", "y"):
            self.dismiss(True)
        else:
            self.dismiss(False)


BROWSE = "__browse__"
BACK = "__back__"          # 目录浏览器最上层再"上一级" → 回到模型选择器
FORGET = "__forget__"      # 选择器里删除记住的条目:dismiss((FORGET, profile))


class ModelPickerScreen(ModalScreen[object]):
    """Known profiles (running marker + source disambiguation + [记住] tag on
    state-remembered entries) plus a 'browse for launch scripts…' action; usable
    with zero known profiles. Delete/d on a [记住] entry requests forgetting it."""

    CSS = """
    ModelPickerScreen { align: center middle; }
    #mp-box { width: 70; height: auto; max-height: 24; border: thick $primary; background: $surface; padding: 1 2; }
    #mp-hint { height: 1; color: $text 60%; }
    """

    HINT = "Enter 切换 · Delete/d 删除 [记住] 条目 · Esc 取消"

    def __init__(self, profiles_list: list, current_name: str | None) -> None:
        super().__init__()
        self._profiles = profiles_list
        self.current_name = current_name
        self._picks: list = []

    def compose(self) -> ComposeResult:
        with Vertical(id="mp-box"):
            yield ListView(id="mp-list")
            yield Static(self.HINT, id="mp-hint")

    def on_mount(self) -> None:
        self._rebuild()
        self.query_one(ListView).focus()

    def _rebuild(self) -> None:
        lv = self.query_one(ListView)
        lv.clear()
        self._picks = []
        for p in self._profiles:
            mark = "  [运行中]" if self.current_name and p.name == self.current_name else ""
            tag = "  [记住]" if p.source == EXTRA_SOURCE else ""
            lv.append(ListItem(Label(f"{p.display}{mark}{tag}"), id=f"mp-{len(self._picks)}"))
            self._picks.append(p)
        lv.append(ListItem(Label("[bold cyan]浏览启动脚本目录…[/bold cyan]"),
                           id=f"mp-{len(self._picks)}"))
        self._picks.append(BROWSE)
        lv.index = 0 if self._picks else None

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        try:
            idx = int(str(event.item.id).rsplit("-", 1)[1])
            self.dismiss(self._picks[idx])
        except (ValueError, IndexError):
            self.dismiss(None)

    def on_key(self, event) -> None:
        if event.key in ("up", "down", "enter", "pageup", "pagedown", "home", "end"):
            return
        if event.key in ("delete", "d"):
            event.stop()
            event.prevent_default()
            self._forget_highlighted()
            return
        self.dismiss(None)

    def _forget_highlighted(self) -> None:
        """Delete/d: only state-remembered entries can be forgotten."""
        lv = self.query_one(ListView)
        idx = lv.index
        if idx is None or not (0 <= idx < len(self._picks)):
            return
        picked = self._picks[idx]
        if picked is BROWSE:
            self.notify("该项不是记住的条目", severity="warning")
            return
        if picked.source != EXTRA_SOURCE:
            self.notify("只能删除记住的条目:扫描到的 profile 不受影响", severity="warning")
            return
        self.dismiss((FORGET, picked))


class DirBrowserScreen(ModalScreen):
    """Read-only keyboard browser: drives -> dirs -> [PROFILE]/[BAT] entries.
    Each level lists a `[上级] ..` entry first (Enter / Backspace / ← 返回上一级;
    at the drive list it returns to the model picker). Selecting a recipe entry
    dismisses with the Entry; Esc cancels."""

    CSS = """
    DirBrowserScreen { align: center middle; }
    #db-box { width: 80%; height: 80%; border: thick $secondary; background: $surface; padding: 0 1; }
    #db-path { height: 1; color: $text 80%; }
    #db-err { height: 1; color: $warning; }
    """

    TAGS = {fsbrowse.PROFILE: "PROFILE", fsbrowse.BAT: "启动脚本", fsbrowse.DIR: "",
            fsbrowse.UP: "上级"}

    def __init__(self, initial: Path | None) -> None:
        super().__init__()
        self._cwd: Path | None = None
        self._entries: list = []
        self._initial = initial
        self._paste: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="db-box"):
            yield Static(id="db-path")
            yield ListView(id="db-list")
            yield Static(id="db-err")

    def on_mount(self) -> None:
        init = self._initial if (self._initial and self._initial.is_dir()) else None
        self.navigate(init)

    # ---------- navigation ----------

    def navigate(self, path: Path | None) -> None:
        self._paste = None
        if path is None:
            self._cwd = None
            drives = [fsbrowse.Entry(fsbrowse.DIR, d, str(d)) for d in fsbrowse.list_drives()]
            self._entries = [fsbrowse.Entry(fsbrowse.UP, Path(), "返回模型选择器")] + drives
            head = "驱动器 (Enter 进入 · ../Backspace 返回 · Esc 取消 · g 粘贴路径)"
            err = None
        else:
            entries, err = fsbrowse.list_entries(path)
            self._cwd = path
            self._entries = [fsbrowse.Entry(fsbrowse.UP, path, "..")] + entries
            head = f"{path}  (Enter 进入 · ../Backspace 上级 · Esc 取消 · g 粘贴路径)"
        lv = self.query_one(ListView)
        lv.clear()
        self._seq = getattr(self, "_seq", 0) + 1      # unique ids per navigation round
        for i, e in enumerate(self._entries):
            tag = self.TAGS[e.kind]
            label = f"[{tag}] {e.label}" if tag else e.label
            lv.append(ListItem(Label(label), id=f"e{self._seq}-{i}"))
        self.query_one("#db-path", Static).update(head)
        self.query_one("#db-err", Static).update(err or "")
        lv.focus()
        lv.index = 0 if self._entries else None      # clear() 会清空高亮;首项([上级] ..)立即可 Enter

    def _go_up(self) -> None:
        """上一级:目录 → 父目录;已在驱动器列表 → 回到模型选择器。"""
        if self._cwd is None:
            self.dismiss(BACK)
            return
        parent = self._cwd.parent
        self.navigate(None if parent == self._cwd else parent)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        try:
            idx = int(str(event.item.id).rsplit("-", 1)[1])
            e = self._entries[idx]
        except (ValueError, IndexError):
            return
        if e.kind == fsbrowse.UP:
            self._go_up()
        elif e.kind == fsbrowse.DIR:
            self.navigate(e.path)
        else:
            self.dismiss(e)

    def on_key(self, event) -> None:
        if self._paste is not None:
            if event.key == "escape":
                self._paste = None
                self.navigate(self._cwd)
            elif event.key == "enter":
                p, err = fsbrowse.resolve_paste(self._paste)
                if p is not None:
                    self.navigate(p)
                else:
                    self.query_one("#db-err", Static).update(err or "")
                    self._paste = None
            elif event.key == "backspace":
                self._paste = self._paste[:-1]
                self.query_one("#db-path", Static).update("粘贴路径: " + self._paste)
            elif event.is_printable:
                self._paste += event.character or ""
                self.query_one("#db-path", Static).update("粘贴路径: " + self._paste)
            event.stop()
            event.prevent_default()
            return
        if event.key in ("backspace", "left"):
            event.stop()
            event.prevent_default()
            self._go_up()
        elif event.key == "g":
            event.stop()
            event.prevent_default()
            self._paste = ""
            self.query_one("#db-path", Static).update("粘贴路径: ")
        elif event.key == "escape":
            event.stop()
            event.prevent_default()
            self.dismiss(None)


class LogOverlayScreen(ModalScreen[None]):
    """Tails the newest run log of the resolved profile; any close key returns."""

    CSS = """
    LogOverlayScreen { align: center middle; }
    #lo-box { width: 90%; height: 90%; border: thick $secondary; background: $surface; }
    RichLog { width: 1fr; height: 1fr; }
    """

    BINDINGS = [
        Binding("escape", "close", "关闭"),
        Binding("q", "close"),
        Binding("l", "close"),
    ]

    def __init__(self, store: SessionStore) -> None:
        super().__init__()
        self.store = store
        self._path: Path | None = None
        self._watch_dir: Path | None = None   # LM Studio 等目录模式:每秒重选最新日志
        self._offset = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="lo-box"):
            yield RichLog(highlight=False, markup=False, wrap=False)

    def _resolve_path(self) -> Path | None:
        """按监控来源定位当前日志文件:
        1) 配置脚本启动的 llama.cpp:profile 目录(server*.log);
        2) attach 且 log_path 是具体文件(Strata);
        3) attach 且 log_path 是目录(LM Studio 的 server-logs):按月份/日期
           布局取最新文件。"""
        if self.store.profile is not None:
            return profiles.pick_active_log(self.store.profile.dir)
        if self.store.facts is not None and self.store.facts.log_path:
            cand = Path(self.store.facts.log_path)
            if cand.is_file():
                return cand
            if cand.is_dir():
                self._watch_dir = cand
                return profiles.pick_active_log(cand)
        return None

    def _show_tail(self) -> None:
        """把当前文件末尾最多 400 行灌进窗口,并把读取位置指到文件尾。"""
        log = self.query_one(RichLog)
        try:
            data = self._path.read_bytes() if self._path is not None else b""
        except OSError:
            data = b""
        self._offset = len(data)
        for ln in data.decode("utf-8", errors="replace").splitlines()[-400:]:
            log.write(Text(ln, style="dim"))

    def on_mount(self) -> None:
        log = self.query_one(RichLog)
        self._path = self._resolve_path()
        if self._path is None:
            log.write(Text("日志不可用(attach 模式:未发现该 server 的 profile 目录或日志文件)", style="yellow"))
            return
        self._show_tail()
        self.set_interval(1.0, self._follow)

    def _follow(self) -> None:
        # 目录模式:每秒重选最新日志;LM Studio 重启/跨天换文件时自动切过去
        if self._watch_dir is not None:
            newest = profiles.pick_active_log(self._watch_dir)
            if newest is not None and newest != self._path:
                self._path = newest
                log = self.query_one(RichLog)
                log.write(Text(f"── 切换到日志文件: {newest.name} ──", style="cyan"))
                self._show_tail()
                return
        if self._path is None:
            return
        try:
            size = self._path.stat().st_size
            if size < self._offset:
                self._offset = 0
            with self._path.open("rb") as fh:
                fh.seek(self._offset)
                chunk = fh.read()
                self._offset = fh.tell()
        except OSError:
            return
        if chunk:
            log = self.query_one(RichLog)
            for ln in chunk.decode("utf-8", errors="replace").splitlines():
                if ln.strip():
                    log.write(Text(ln))

    def action_close(self) -> None:
        self.dismiss(None)


class RowLayout(Horizontal):
    """上三框的容器:自身尺寸变化(终端 resize)时立刻重算四个面板宽度。

    App 不是 Widget,DashboardApp.on_resize 实测不会被触发(内联宽度会一直保持
    旧值,框的右半被终端遮住),所以把钩子放在真正会收到 Resize 的容器上。
    """

    def on_resize(self) -> None:
        app = self.app
        if isinstance(app, DashboardApp):
            app._sync_row_layout()


class DashboardApp(App[None]):
    TITLE = "LLM Monitor"
    CSS = """
    #topbox { height: auto; border: round #9e9e9e; padding: 0 1; }
    #header { height: auto; }
    #metrics { height: auto; }
    #row { height: auto; }
    /* 同行时硬件框弹性吃满剩余宽度;两个实时框由 JS 按内容固定宽度 */
    #gpu { width: 1fr; min-width: 0; height: auto; border: round $primary; padding: 0 1; }
    #live-prompt { width: auto; max-width: 100%; height: auto; border: round $success; padding: 0 1; }
    #live-decode { width: auto; max-width: 100%; height: auto; border: round $success; padding: 0 1; }
    #row.stacked { layout: vertical; }        /* 放不下三框时上下排列 */
    #row.stacked #gpu, #row.stacked #live-prompt, #row.stacked #live-decode { width: 100%; }
    /* 历史表与顶部信息框始终撑满整行 */
    #history { width: 100%; height: auto; max-height: 16; border: round $warning; padding: 0 1; }
    #hint { height: 1; color: $text 70%; padding: 0 1; }
    #status { height: 1; padding: 0 1; }
    """

    BINDINGS = [
        Binding("q", "quit", "退出"),
        Binding("m", "switch_model", "换模型"),
        Binding("s", "stop", "停止"),
        Binding("r", "restart", "重启"),
        Binding("l", "log", "日志"),
        Binding("p", "pause", "暂停"),
        Binding("c", "clear_history", "清历史"),
        Binding("h", "thermal", "温度"),
    ]

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.health = HealthBoard(stale_after_s=settings.sample_interval * settings.stale_ticks)
        self.store = SessionStore(settings, self.health)
        self.runner = Runner(settings, self.store, self.health)

    def compose(self) -> ComposeResult:
        with Vertical(id="topbox"):
            yield Static(id="header")
            yield Static(id="metrics")
        with RowLayout(id="row"):
            gpu = Static(id="gpu", classes="GpuPanel")
            gpu.border_title = "硬件 · GPU/CPU"
            yield gpu
            lp = Static(id="live-prompt", classes="LivePanel")
            lp.border_title = "实时 · prompt"
            yield lp
            ld = Static(id="live-decode", classes="LivePanel")
            ld.border_title = "实时 · decode"
            yield ld
        hist = Static(id="history")
        hist.border_title = "请求历史 (新→旧)"
        yield hist
        yield Static(render.HINT, id="hint")
        yield Static(id="status")

    def on_mount(self) -> None:
        if self.settings.state_notice:
            self.notify(self.settings.state_notice, severity="warning", timeout=8)
        self._refresh()
        self.set_interval(self.settings.sample_interval, self._tick)

    def on_unmount(self) -> None:
        # 退出时把本模型累计缓存兜底落盘(正常每 30s 也会存一次)
        try:
            self.runner.save_runstats_now()
        except Exception:  # noqa: BLE001
            pass

    def all_profiles(self) -> list:
        return profiles.collect(self.settings.starts_roots, self.settings.state.extra_profiles)

    async def _tick(self) -> None:
        if self.runner.paused:
            return
        await self.runner.tick()
        self._refresh()

    def _refresh(self) -> None:
        st = self.store
        now = self.runner.now()
        # 终端窗口标题固定不变(监控对象在面板内容里已能看出);且标题栏只能用英文
        want_title = "LLM Monitor"
        if self.title != want_title:
            self.title = want_title
        # 顶部信息框的框标题(嵌在边框线上),随监控对象变化
        topbox = self.query_one("#topbox", Vertical)
        if st.facts is None:
            box_title = "llama.cpp / Strata 监控"
        elif getattr(st.facts, "kind", "llama") == "strata":
            box_title = "Strata 监控"
        else:
            box_title = "llama.cpp 监控"
        topbox.border_title = Text(box_title, style="bold cyan")
        self.query_one("#header", Static).update(render.header_text(st))
        self.query_one("#metrics", Static).update(render.metrics_text(st))
        gpu = self.query_one("#gpu", Static)
        gpu.border_title = "硬件 · GPU/CPU" + (" (正在加载模型…)" if self.runner.window.active and st.facts is None else "")
        cur = st.current_slot()
        if cur is None:                          # 无活动槽:两个框都空闲
            pb = db = " (空闲)"
        elif cur.phase == "prompt":              # 还在消化输入:prompt 在处理,decode 尚未开始
            pb, db = " (处理)", " (空闲)"
        else:                                    # 已在出字:输入阶段完成
            pb, db = " (完成)", " (解码)"
        # 注:状态词必须≤2个汉字——实时框只有27列宽,3个字标题会被边框截成"已完…"
        self.query_one("#live-prompt", Static).border_title = "实时 · prompt" + pb
        self.query_one("#live-decode", Static).border_title = "实时 · decode" + db
        self.query_one("#live-prompt", Static).update(render.live_prompt_text(st))
        self.query_one("#live-decode", Static).update(render.live_decode_text(st))
        self._sync_row_layout()
        self.query_one("#history", Static).update(render.history_text(st, max_rows=10))
        self.query_one("#status", Static).update(
            render.status_text(st, self.health, self.settings.sample_interval, self.runner.paused, now))

    def _sync_row_layout(self) -> None:
        """宽度规则:三行框全部撑满屏幕,并随终端宽度自适应。

        - 宽屏同行:两个实时框保持各自内容的固定宽度,硬件框用 1fr 吃掉全部
          剩余宽度;硬件框内右列(CPU/内存、CPU 型号)整块贴框右内缘;
        - 窄屏放不下:三框上下堆叠,全部 100% 宽,硬件框右列同样靠右;
        - 顶部信息框(CSS 默认全宽)与历史框始终满宽。
        """
        st = self.store
        gpu_min = render.panel_width(render.gpu_text(st, st.show_thermal_detail))
        lp_w = render.panel_width(render.live_prompt_text(st))
        ld_w = render.panel_width(render.live_decode_text(st))
        available = self.size.width
        stacked = gpu_min + lp_w + ld_w > available
        self.query_one("#row").set_class(stacked, "stacked")
        gpu_wgt = self.query_one("#gpu", Static)
        lp_wgt = self.query_one("#live-prompt", Static)
        ld_wgt = self.query_one("#live-decode", Static)
        if stacked:
            for wgt in (gpu_wgt, lp_wgt, ld_wgt):
                wgt.styles.width = "100%"
        else:
            gpu_wgt.styles.width = "1fr"
            lp_wgt.styles.width = lp_w
            ld_wgt.styles.width = ld_w
        # 硬件框内容要按"真实框内宽"渲染:宽度 1fr/100% 要等本轮布局完成后才知道,
        # 且内容超高出现滚动条时还会再少一列,所以推迟到刷新后读 content_size。
        self.call_after_refresh(self._render_gpu_panel)
        self.query_one("#history").styles.width = "100%"

    def _render_gpu_panel(self) -> None:
        """按硬件框当前真实内容区宽度重绘(右列靠右、窄屏降级都在 render 里处理)。"""
        gpu_wgt = self.query_one("#gpu", Static)
        inner = max(0, gpu_wgt.content_size.width)
        st = self.store
        gpu_wgt.update(render.gpu_text(st, st.show_thermal_detail, inner))

    # 终端尺寸变化时的重算钩子在 RowLayout.on_resize(App 不是 Widget,不会收到 Resize)

    # ------------- actions -------------

    def _confirm(self, message: str, then: Callable[[], None]) -> None:
        def cb(ok: bool) -> None:
            if ok:
                then()

        self.push_screen(ConfirmScreen(message), cb)

    async def action_quit(self) -> None:
        self.exit()          # spec: exiting must not touch the server

    def action_stop(self) -> None:
        if self.store.facts is None:
            self.notify(f"没有在跑的 server 进程({' / '.join(proc.short_names())})", severity="warning")
            return
        self._confirm("停止当前 server?(模型将卸载,重新加载需 1-2 分钟)",
                      lambda: self._run_async(self.runner.do_stop()))

    def action_restart(self) -> None:
        if self.store.facts is None:
            self.notify(f"没有在跑的 server 进程({' / '.join(proc.short_names())}),可用 m 启动 profile",
                        severity="warning")
            return
        mode = "运行 run.bat" if self.store.profile else "重放发现的命令行(attach 模式)"
        self._confirm(f"重启 server?{mode}", lambda: self._run_async(self.runner.do_restart()))

    def action_switch_model(self) -> None:
        current = self.store.profile.name if self.store.profile else None
        self.push_screen(ModelPickerScreen(self.all_profiles(), current), self._on_profile_picked)

    def _on_profile_picked(self, picked) -> None:
        if picked is None:
            return
        if isinstance(picked, tuple) and len(picked) == 2 and picked[0] == FORGET:
            self._forget_profile(picked[1])
            return
        if picked == BROWSE:
            last = self.settings.state.last_browse_dir
            init = Path(last) if last else None
            self.push_screen(DirBrowserScreen(init), self._on_browse_pick)
            return
        prof = picked
        msg = f"切换到 profile「{prof.display}」?将执行其 {prof.bat.name}(自带先杀后启)"
        self._confirm(msg, lambda: self._run_async(self.runner.do_switch_profile(prof)))

    def _on_browse_pick(self, entry) -> None:
        if entry is None:
            return
        if entry == BACK:
            self.action_switch_model()      # 回到模型选择器,不写状态、不动进程
            return
        recipe, bat = entry.recipe_dir, entry.recipe_bat
        msg = f"记住并切换到「{recipe.name}」?将写入本机状态并执行 {bat.name}"

        def go() -> None:
            stt = self.settings.state
            stt.add_extra(recipe, bat)
            stt.last_browse_dir = str(entry.path.parent)
            try:
                state_mod.save(stt)
            except OSError as exc:
                self.notify(f"状态写入失败: {exc}", severity="error")
            prof = Profile(name=recipe.name, dir=recipe, bat=bat,
                           source=EXTRA_SOURCE, display=recipe.name)
            self._run_async(self.runner.do_switch_profile(prof))

        self._confirm(msg, go)

    def _forget_profile(self, prof) -> None:
        """忘记一个记住的配方:仅改状态文件,不动目录、不动 server。"""
        name = prof.display or prof.name
        msg = f"删除记住的「{name}」?仅从本机状态移除,不删除目录、不影响正在跑的 server"

        def go() -> None:
            stt = self.settings.state
            if not stt.remove_extra(prof.dir):
                self.notify("状态里已没有该条目", severity="warning")
                return
            try:
                state_mod.save(stt)
            except OSError as exc:
                self.notify(f"状态写入失败: {exc}", severity="error")
                return
            self.notify(f"已忘记「{prof.name}」(目录与 server 未受影响)")
            self.action_switch_model()      # 重开列表,已不含该条目

        self._confirm(msg, go)

    def action_log(self) -> None:
        self.push_screen(LogOverlayScreen(self.store))

    def action_pause(self) -> None:
        paused = self.runner.toggle_pause()
        if not paused:
            self._refresh()
        else:
            self.query_one("#status", Static).update(
                render.status_text(self.store, self.health, self.settings.sample_interval, True,
                                   self.runner.now()))

    def action_clear_history(self) -> None:
        self.store.reset_session()
        self.runner._parse_fail_base = self.runner.parser.failures  # noqa: SLF001
        self.notify("已清空本次监控统计(日志文件未受影响)")
        self._refresh()

    def action_thermal(self) -> None:
        self.store.show_thermal_detail = not self.store.show_thermal_detail
        self._refresh()

    def _run_async(self, coro) -> None:
        self.run_worker(coro, exclusive=True)
