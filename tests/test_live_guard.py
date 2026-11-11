"""Long-running safety: non-blocking stdout, watchdog, never-raising writers, finalize loop survival."""

from __future__ import annotations

import io
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from src.streaming.live_guard import AsyncLineWriter, Watchdog, dump_all_stacks, install_async_stdio, keep_system_awake


class _BlockingTarget(io.StringIO):
    """A pipe nobody reads: write() blocks until released."""

    def __init__(self):
        super().__init__()
        self.release = threading.Event()

    def write(self, s):
        self.release.wait()
        return super().write(s)


def test_async_writer_never_blocks_and_counts_drops():
    target = _BlockingTarget()
    w = AsyncLineWriter(target, name="stdout", max_items=50)
    t0 = time.monotonic()
    for i in range(500):
        w.write(f"line {i}\n")
        w.flush()
    assert time.monotonic() - t0 < 0.5  # producers never wait for the reader
    assert w.dropped > 0
    target.release.set()
    assert w.drain(2.0)
    time.sleep(0.05)
    w.write("after\n")
    assert w.drain(2.0)
    time.sleep(0.05)
    out = target.getvalue()
    assert "line 0\n" in out and "after\n" in out
    assert "输出曾阻塞" in out  # the reader learns how much was lost
    w.close()


def test_async_writer_survives_target_errors():
    class Broken(io.StringIO):
        def write(self, s):
            raise OSError(22, "Invalid argument")

    w = AsyncLineWriter(Broken())
    w.write("x\n")
    deadline = time.monotonic() + 3
    while w.write_errors == 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert w.write_errors >= 1
    assert w.write("still fine\n") == len("still fine\n")
    w.close()


def test_install_async_stdio_skips_tty_and_is_idempotent(monkeypatch):
    class Tty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdout", Tty())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    install_async_stdio()
    assert not isinstance(sys.stdout, AsyncLineWriter)  # interactive console untouched
    assert isinstance(sys.stderr, AsyncLineWriter)
    first = sys.stderr
    install_async_stdio()
    assert sys.stderr is first


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_watchdog_heartbeat_busy_and_recovery(tmp_path):
    clk, events, exits = _Clock(), [], []
    wd = Watchdog(tmp_path, emit=lambda name, **kw: events.append((name, kw)), stall_s=60,
                  exit_after_s=0, clock=clk, exit_fn=exits.append, log=lambda m: None)
    wd.beat("finalize")
    wd.busy("infer:cam_01")
    clk.t += 30
    assert wd.tick() == []
    clk.t += 40  # 70 s without a beat, inference call busy 70 s
    problems = wd.tick()
    assert any(p.startswith("no_heartbeat:finalize") for p in problems)
    assert any(p.startswith("busy:infer:cam_01") for p in problems)
    assert events[-1][0] == "stall" and events[-1][1]["dump"]
    dump = Path(events[-1][1]["dump"])
    assert dump.parent == tmp_path and "--- thread" in dump.read_text(encoding="utf-8")
    n = len(events)
    clk.t += 5
    wd.tick()
    assert len(events) == n  # same problem: reported once
    wd.beat("finalize")
    wd.idle("infer:cam_01")
    assert wd.tick() == []
    assert events[-1][0] == "stall_recovered"
    assert exits == []


def test_watchdog_dead_thread_exits_after_threshold(tmp_path):
    clk, events, exits = _Clock(), [], []
    wd = Watchdog(tmp_path, emit=lambda name, **kw: events.append(name), stall_s=60, exit_after_s=120,
                  clock=clk, exit_fn=exits.append, log=lambda m: None)
    t = threading.Thread(target=lambda: None)
    t.start()
    t.join()
    wd.watch_thread("finalize", t)
    assert wd.tick() == ["thread_dead:finalize"]
    clk.t += 119
    wd.tick()
    assert exits == []
    clk.t += 2
    wd.tick()
    assert exits == [70] and events[-1] == "watchdog_exit"


def test_watchdog_thread_runs_and_stops(tmp_path):
    wd = Watchdog(tmp_path, interval_s=0.2, prevent_sleep=True, log=lambda m: None)
    wd.start()
    time.sleep(0.3)
    wd.stop()
    assert wd._thread is not None and not wd._thread.is_alive()
    if sys.platform != "win32":
        assert keep_system_awake(True) is False  # no-op off Windows


def test_dump_all_stacks_lists_threads(tmp_path):
    ev = threading.Event()
    t = threading.Thread(target=ev.wait, name="stuck-worker", daemon=True)
    t.start()
    path = dump_all_stacks(tmp_path / "d.txt", "test", {"k": "v"})
    ev.set()
    text = path.read_text(encoding="utf-8")
    assert "stuck-worker" in text and "# k: v" in text and "wait" in text


# ------------------------------------------------------------------ writers never raise

def test_edge_ws_publish_survives_jsonl_failure_and_numpy(tmp_path):
    from src.streaming.edge_ws_client import EdgeWsClient

    bad = tmp_path / "events.jsonl"
    bad.mkdir()  # opening a directory for append fails like a full / read-only disk
    cli = EdgeWsClient("ws://127.0.0.1:9/x", jsonl_path=bad)
    cli.publish({"event": "action_finalized", "made": np.bool_(True), "conf": np.float32(0.5),
                 "idx": np.int64(3), "arr": np.arange(3)})
    assert cli.jsonl_failed == 1 and cli.encode_failed == 0
    assert cli.dropped == 1  # not connected: dropped as before, but no exception

    ok = EdgeWsClient("ws://127.0.0.1:9/x", jsonl_path=tmp_path / "ok.jsonl")
    ok.publish({"event": "timeline_gap", "v": np.float32(1.5), "b": np.bool_(False)})
    row = json.loads((tmp_path / "ok.jsonl").read_text(encoding="utf-8"))
    assert row["v"] == 1.5 and row["b"] is False


def test_metrics_write_failure_does_not_raise(tmp_path, capsys):
    from src.streaming.live_metrics import LiveMetrics

    lm = LiveMetrics("s", tmp_path)
    lm.jsonl_path = tmp_path  # a directory: every append fails
    lm.event("live_start", force=True)
    lm.event("live_start", force=True)  # within the 60 s back-off: not retried
    assert lm.write_failed == 1
    assert lm.maybe_emit(force=True) is True
    assert "metrics.jsonl 写入失败" in capsys.readouterr().out
    lm.close()


# ------------------------------------------------------------------ engine

class _Hub:
    def __init__(self, fail_first=0):
        self.sent = []
        self.fail_first = fail_first

    def publish(self, ev):
        if self.fail_first:
            self.fail_first -= 1
            raise OSError(28, "No space left on device")
        self.sent.append(ev)


def _engine(monkeypatch, tmp_path, hub, cfg=None):
    import src.streaming.live_engine as le

    monkeypatch.setattr("src.config.DATA", tmp_path)
    return le, le.LiveEngine("live_g", {"cam_01": "rtsp://127.0.0.1:9/a"}, {}, hub=hub,
                             cfg=cfg if cfg is not None else {"observability": {"stdout": False}})


def _fill_pose_ring(eng, n=40, t0=1_000_000.0):
    for i in range(n):
        eng.rings["cam_03"].push(t0 + i * 100.0, i, {"camera_id": "cam_03", "persons": []})
    return t0 + (n - 1) * 100.0


def _cand(sid, t):
    return {"student_id": sid, "action_type": "jump_shot", "start_ms": t - 1500, "end_ms": t,
            "release_ms": t - 300, "made": None, "phases": [], "identity": None, "confidence": 0.9}


def test_one_failing_publish_does_not_stop_the_others(monkeypatch, tmp_path):
    hub = _Hub(fail_first=1)
    le, eng = _engine(monkeypatch, tmp_path, hub)
    last = _fill_pose_ring(eng)
    cands = [_cand("stu_01", last - 2000), _cand("stu_02", last - 2000)]
    monkeypatch.setattr(le, "detect_live_actions", lambda *a, **k: list(cands))
    eng._finalize_once()
    assert [m["student_id"] for m in hub.sent] == ["stu_02"]
    assert eng._errors["emit_error"] == 1
    eng._finalize_once()  # the failed one counts as emitted: not re-sent every round
    assert len(hub.sent) == 1


def test_finalize_thread_survives_exceptions(monkeypatch, tmp_path):
    le, eng = _engine(monkeypatch, tmp_path, _Hub())
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise OSError(28, "No space left on device")

    monkeypatch.setattr(eng, "_finalize_once", flaky)
    th = threading.Thread(target=eng._finalize_loop, daemon=True)
    th.start()
    deadline = time.monotonic() + 5
    while len(calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    eng._stop.set()
    th.join(3)
    assert len(calls) >= 2 and eng._errors["finalize_error"] == 1
    assert "finalize" in eng.watchdog._beats


def test_watchdog_wired_into_start_and_stop(monkeypatch, tmp_path):
    le, eng = _engine(monkeypatch, tmp_path, _Hub())
    monkeypatch.setattr(eng.perception, "preload", lambda ball=True: [])
    monkeypatch.setattr(eng.capture, "start", lambda: None)
    monkeypatch.setattr(le, "install_async_stdio", lambda **k: False)
    eng.start()
    assert {"finalize", "infer:cam_01", "infer:cam_03"} <= set(eng.watchdog._threads)
    assert eng.watchdog._thread is not None and eng.watchdog._thread.is_alive()
    assert eng.watchdog.check() == []
    eng.stop()
    assert not eng.watchdog._thread.is_alive()


def test_watchdog_config(monkeypatch, tmp_path):
    _, eng = _engine(monkeypatch, tmp_path, _Hub(), cfg={
        "observability": {"enabled": False, "async_stdout": False},
        "watchdog": {"enabled": False, "stall_s": 90, "exit_after_s": 0, "prevent_sleep": False},
    })
    wd = eng.watchdog
    assert (wd.enabled, wd.stall_s, wd.exit_after_s, wd.prevent_sleep) == (False, 90.0, 0.0, False)
    assert eng._async_stdout is False
    wd.start()
    assert wd._thread is None  # disabled: no thread
    _, eng2 = _engine(monkeypatch, tmp_path, _Hub(), cfg={})
    assert (eng2.watchdog.enabled, eng2.watchdog.stall_s, eng2.watchdog.exit_after_s) == (True, 60.0, 180.0)
    assert eng2._async_stdout is True
