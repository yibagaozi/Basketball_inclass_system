"""Long-running safety for the live CV process.

Three independent pieces, all used by LiveEngine:

- AsyncLineWriter / install_async_stdio: stdout/stderr go through a bounded queue and
  a writer thread. When the reader of our pipe (edge) stops reading, print() used to
  block the calling thread for good (the finalize thread prints every 5 s, so actions
  stopped). Now lines are dropped and counted instead; inference and WS events go on.
- Watchdog: heartbeats from the finalize loop, "busy" marks around per-frame inference,
  and liveness of the worker threads. On a stall it writes every thread's stack to
  <live dir>/stall-*.txt (a file, not stdout, which may be the thing that is stuck),
  emits a [cv-event] stall, and after exit_after_s exits the process so edge restarts it.
- Windows: keeps the system awake while live inference runs (SetThreadExecutionState),
  so the default power plan does not put the host to sleep in the middle of a class.
"""

from __future__ import annotations

import io
import os
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

_MAX_ITEMS = 5000
_MAX_CHARS = 4 * 1024 * 1024


class AsyncLineWriter(io.TextIOBase):
    """Text stream whose write() never blocks: a writer thread forwards to ``target``."""

    def __init__(self, target: Any, *, name: str = "stdout",
                 max_items: int = _MAX_ITEMS, max_chars: int = _MAX_CHARS):
        super().__init__()
        self._target = target
        self.name = name
        self._max_items = max(1, int(max_items))
        self._max_chars = max(1, int(max_chars))
        self._q: deque[str] = deque()
        self._chars = 0
        self._cond = threading.Condition()
        self.dropped = 0
        self._dropped_unreported = 0
        self.write_errors = 0
        self._closed = False
        self._thread = threading.Thread(target=self._run, name=f"async-{name}", daemon=True)
        self._thread.start()

    # -- TextIOBase API --
    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        if not s:
            return 0
        with self._cond:
            if len(self._q) >= self._max_items or self._chars + len(s) > self._max_chars:
                self.dropped += 1
                self._dropped_unreported += 1
                return len(s)
            self._q.append(s)
            self._chars += len(s)
            self._cond.notify()
        return len(s)

    def flush(self) -> None:  # never blocks; the writer flushes after each burst
        return None

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        return self._target.fileno()

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return getattr(self._target, "encoding", None) or "utf-8"

    @property
    def errors(self) -> Optional[str]:  # type: ignore[override]
        return getattr(self._target, "errors", None)

    @property
    def target(self) -> Any:
        return self._target

    # -- writer thread --
    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._q and not self._closed:
                    self._cond.wait(0.5)
                if self._closed and not self._q:
                    return
                batch = list(self._q)
                self._q.clear()
                self._chars = 0
                lost, self._dropped_unreported = self._dropped_unreported, 0
            try:
                if lost:
                    self._target.write(f"  [live] {self.name} 输出曾阻塞，丢弃 {lost} 段日志（累计 {self.dropped}）\n")
                self._target.write("".join(batch))
                self._target.flush()
            except Exception:
                self.write_errors += 1
                time.sleep(1.0)

    def drain(self, timeout: float = 2.0) -> bool:
        """Wait until queued text has been handed to the target (tests / shutdown)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._cond:
                if not self._q:
                    return True
            time.sleep(0.01)
        return False

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify()
        self._thread.join(timeout=2.0)
        super().close()


def install_async_stdio(*, force: bool = False) -> bool:
    """Route sys.stdout/sys.stderr through AsyncLineWriter (only when piped, e.g. by edge).

    Interactive consoles are left alone unless ``force``. Idempotent.
    """
    done = False
    for attr in ("stdout", "stderr"):
        cur = getattr(sys, attr)
        if cur is None or isinstance(cur, AsyncLineWriter):
            continue
        try:
            tty = cur.isatty()
        except Exception:
            tty = False
        if tty and not force:
            continue
        setattr(sys, attr, AsyncLineWriter(cur, name=attr))
        done = True
    return done


def stdio_stats() -> dict[str, int]:
    out: dict[str, int] = {}
    for attr in ("stdout", "stderr"):
        w = getattr(sys, attr)
        if isinstance(w, AsyncLineWriter):
            out[f"{attr}_dropped"] = w.dropped
            out[f"{attr}_write_errors"] = w.write_errors
    return out


# ------------------------------------------------------------------ sleep prevention

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def keep_system_awake(on: bool) -> bool:
    """Windows only: ask the OS not to sleep while this thread lives (display may still turn off)."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        flags = _ES_CONTINUOUS | (_ES_SYSTEM_REQUIRED if on else 0)
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))  # type: ignore[attr-defined]
    except Exception:
        return False


# ------------------------------------------------------------------ watchdog

def dump_all_stacks(path: Path, reason: str, extra: Optional[dict[str, Any]] = None) -> Optional[Path]:
    """Write every Python thread's current stack to ``path``; returns the path or None."""
    try:
        frames = sys._current_frames()
        names = {t.ident: (t.name, t.daemon) for t in threading.enumerate()}
        lines = [f"# live stall dump {datetime.now().isoformat(timespec='seconds')}", f"# reason: {reason}"]
        for k, v in (extra or {}).items():
            lines.append(f"# {k}: {v}")
        for ident, frame in frames.items():
            name, daemon = names.get(ident, ("?", False))
            lines.append(f"\n--- thread {name} (ident={ident}{', daemon' if daemon else ''}) ---")
            lines.extend(x.rstrip("\n") for x in traceback.format_stack(frame))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
    except Exception:
        return None


class Watchdog:
    """Detects a dead/stuck finalize loop or inference call; dumps stacks; optionally exits.

    - ``beat(name)``: call at the top of every loop iteration (finalize).
    - ``busy(name)`` / ``idle(name)``: around one unit of work (one frame of inference).
    - ``watch_thread(name, thread)``: the thread must stay alive until ``stop()``.
    """

    def __init__(
        self,
        out_dir: Optional[Path],
        *,
        emit: Optional[Callable[..., None]] = None,
        enabled: bool = True,
        stall_s: float = 60.0,
        exit_after_s: float = 180.0,
        interval_s: float = 5.0,
        prevent_sleep: bool = True,
        clock: Callable[[], float] = time.monotonic,
        exit_fn: Callable[[int], None] = os._exit,
        log: Callable[[str], None] = lambda m: print(m, flush=True),
    ):
        self.out_dir = Path(out_dir) if out_dir else None
        self.enabled = bool(enabled)
        self.stall_s = max(5.0, float(stall_s))
        self.exit_after_s = max(0.0, float(exit_after_s or 0.0))
        self.interval_s = max(0.2, float(interval_s))
        self.prevent_sleep = bool(prevent_sleep)
        self._emit = emit or (lambda *a, **k: None)
        self._clock = clock
        self._exit = exit_fn
        self._log = log
        self._lock = threading.Lock()
        self._beats: dict[str, float] = {}
        self._busy: dict[str, float] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stall_since: Optional[float] = None
        self._reported: Optional[tuple[str, ...]] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.dumps: list[Path] = []
        self.awake = False

    # -- producers (hot paths: one dict write) --
    def beat(self, name: str) -> None:
        self._beats[name] = self._clock()

    def busy(self, name: str) -> None:
        self._busy[name] = self._clock()

    def idle(self, name: str) -> None:
        self._busy.pop(name, None)

    def watch_thread(self, name: str, thread: threading.Thread) -> None:
        with self._lock:
            self._threads[name] = thread

    # -- checks --
    def check(self) -> list[str]:
        now = self._clock()
        problems = []
        with self._lock:
            threads = dict(self._threads)
        for name, t in sorted(threads.items()):
            if not t.is_alive():
                problems.append(f"thread_dead:{name}")
        for name, t0 in sorted(self._beats.items()):
            if now - t0 > self.stall_s:
                problems.append(f"no_heartbeat:{name}:{now - t0:.0f}s")
        for name, t0 in sorted(list(self._busy.items())):
            if now - t0 > self.stall_s:
                problems.append(f"busy:{name}:{now - t0:.0f}s")
        return problems

    def tick(self) -> list[str]:
        """One watchdog round (the thread calls this every interval_s)."""
        if self._stop.is_set():
            return []
        problems = self.check()
        now = self._clock()
        if not problems:
            if self._stall_since is not None:
                self._log(f"  [live] watchdog: 已恢复（{self._reported}）")
                self._emit("stall_recovered", force=True, problems=list(self._reported or ()),
                           stalled_s=round(now - self._stall_since, 1))
            self._stall_since = None
            self._reported = None
            return problems
        kinds = tuple(p.rsplit(":", 1)[0] if p.startswith(("no_heartbeat", "busy")) else p for p in problems)
        if self._stall_since is None:
            self._stall_since = now
        if kinds != self._reported:
            self._reported = kinds
            path = self._dump("stall", problems)
            self._log(f"  [live] watchdog: 检测到卡住/线程退出 {problems}；线程堆栈已写入 {path}")
            self._emit("stall", force=True, problems=problems, dump=str(path) if path else None,
                       exit_after_s=self.exit_after_s or None)
        stalled = now - self._stall_since
        if self.exit_after_s and stalled >= self.exit_after_s:
            path = self._dump("exit", problems)
            self._log(f"  [live] watchdog: 卡住已 {stalled:.0f}s，退出进程以便 edge 自动重启；堆栈 {path}")
            self._emit("watchdog_exit", force=True, problems=problems, stalled_s=round(stalled, 1),
                       dump=str(path) if path else None)
            self._flush_stdio()
            self._exit(70)
        return problems

    def _dump(self, tag: str, problems: list[str]) -> Optional[Path]:
        if self.out_dir is None:
            return None
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = dump_all_stacks(self.out_dir / f"{tag}-{stamp}.txt", ", ".join(problems), stdio_stats())
        if path is not None:
            self.dumps.append(path)
        return path

    @staticmethod
    def _flush_stdio() -> None:
        for attr in ("stdout", "stderr"):
            w = getattr(sys, attr)
            if isinstance(w, AsyncLineWriter):
                w.drain(1.0)

    # -- thread --
    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="live-watchdog", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        # SetThreadExecutionState is per thread: this thread lives as long as the engine
        if self.prevent_sleep:
            self.awake = keep_system_awake(True)
        try:
            while not self._stop.wait(self.interval_s):
                try:
                    self.tick()
                except Exception as exc:  # the watchdog itself must not die
                    self._log(f"  [live] watchdog error: {type(exc).__name__}: {exc}")
        finally:
            if self.awake:
                keep_system_awake(False)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
