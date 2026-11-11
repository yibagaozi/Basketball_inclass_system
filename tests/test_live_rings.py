"""Live rings are read by the finalize thread while inference threads push."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from src.streaming.fast_path import TimestampRingBuffer
from src.streaming.live_action import detect_live_actions
from src.streaming.live_angles import angles_for_window


def _person(sid: str, t: float, x: float) -> dict:
    k = np.zeros((133, 3), np.float32)
    k[:, 0] = x
    k[:, 1] = 300.0
    k[10, 1] = 300.0 - 80.0 * max(0.0, np.sin(t / 700.0))  # wrist rises and falls
    k[:, 2] = 0.9
    return {"student_id": sid, "score": 1.0, "keypoints": k.tolist()}


def _payload(t: float) -> dict:
    return {"persons": [_person(f"stu_{p:02d}", t, 200.0 * p) for p in range(4)]}


def _fill(ring: TimestampRingBuffer, n: int, step_ms: float = 40.0) -> None:
    for i in range(n):
        ring.push(i * step_ms, i, _payload(i * step_ms))


def _run_with_pusher(rings: list[TimestampRingBuffer], start: int, fn, rounds: int) -> list[BaseException]:
    stop = threading.Event()

    def pusher(ring: TimestampRingBuffer) -> None:
        i = start
        while not stop.is_set():
            ring.push(i * 40.0, i, _payload(i * 40.0))
            i += 1

    threads = [threading.Thread(target=pusher, args=(r,), daemon=True) for r in rings]
    for t in threads:
        t.start()
    errors: list[BaseException] = []
    try:
        for _ in range(rounds):
            try:
                fn()
            except RuntimeError as exc:  # "deque mutated during iteration"
                errors.append(exc)
    finally:
        stop.set()
        for t in threads:
            t.join(5)
    return errors


def test_detect_while_pushing_never_sees_a_mutating_ring():
    pose, ball = TimestampRingBuffer(capacity_ms=8_000), TimestampRingBuffer(capacity_ms=8_000)
    _fill(pose, 200)
    for i in range(200):
        ball.push(i * 40.0, i, {"ball": {"center": [500.0, 100.0]}, "hoop": {"center": [500.0, 80.0]}})
    errors = _run_with_pusher([pose], 200, lambda: detect_live_actions(pose, ball, min_seq=20), rounds=4)
    assert errors == []


def test_angles_while_pushing_never_sees_a_mutating_ring():
    rings = {c: TimestampRingBuffer(capacity_ms=60_000) for c in ("cam_01", "cam_02", "cam_03")}
    for r in rings.values():
        _fill(r, 200)
    errors = _run_with_pusher(
        list(rings.values()), 200,
        lambda: angles_for_window(rings, 2000.0, 7000.0, student_id="stu_01",
                                  calib_dir=Path("/nonexistent_calib")),
        rounds=4,
    )
    assert errors == []


def test_snapshot_and_latest():
    ring = TimestampRingBuffer(capacity_ms=100.0)
    assert ring.latest() is None and ring.snapshot() == []
    for i in range(5):
        ring.push(i * 40.0, i, {"i": i})
    snap = ring.snapshot()
    assert [x.frame_idx for x in snap] == [2, 3, 4]  # trimmed to capacity_ms
    ring.push(200.0, 5, {"i": 5})
    assert [x.frame_idx for x in snap] == [2, 3, 4]  # snapshot is a copy
    assert ring.latest().frame_idx == 5
    assert [x.frame_idx for x in ring.window(160.0, 200.0)] == [4, 5]
