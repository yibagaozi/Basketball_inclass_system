"""Nearest-frame lookup, angle calibration handling, worker wakeup, file pacing."""

from __future__ import annotations

import json
import threading
import time

import cv2
import numpy as np
import pytest

import src.streaming.live_angles as la
from src.acquisition.rtsp import LiveFrame, RtspCameraWorker
from src.streaming.fast_path import RingFrame, TimestampRingBuffer


def _linear(items, t, slop=80.0):
    if not items:
        return None
    best = min(items, key=lambda x: abs(x.timestamp_ms - t))
    return None if abs(best.timestamp_ms - t) > slop else best


@pytest.mark.parametrize("dupes", [False, True])
def test_nearest_ring_frame_equals_linear_scan(dupes):
    rng = np.random.default_rng(0)
    ts = np.sort(rng.uniform(0, 4000, 200)).round(0)
    if dupes:
        ts[20:26] = ts[20]
    items = [RingFrame(float(t), i, {}) for i, t in enumerate(ts)]
    for t in list(rng.uniform(-200, 4200, 500)) + list((ts[:-1] + ts[1:]) / 2):
        assert la.nearest_ring_frame(items, float(t)) is _linear(items, float(t))
    shuffled = items[:]
    shuffled[3], shuffled[4] = shuffled[4], shuffled[3]  # non-monotonic → linear fallback
    for t in (10.0, float(ts[3]), 999.0):
        assert la.nearest_ring_frame(shuffled, t) is _linear(shuffled, t)


def _rings():
    rings = {}
    for cam in ("cam_01", "cam_02", "cam_03"):
        r = TimestampRingBuffer()
        for i in range(10):
            k = np.zeros((133, 3))
            k[:, 2] = 0.9
            r.push(100.0 + 33.0 * i, i, {"persons": [{"student_id": "stu_00", "keypoints": k}]})
        rings[cam] = r
    return rings


def test_no_session_calibration_means_null_angles(monkeypatch):
    def boom(d):
        raise AssertionError("must not fall back to the offline v2 calibration")

    monkeypatch.setattr(la, "load_camera_calibration", boom)
    rows = la.angles_for_window(_rings(), 100.0, 300.0, student_id="stu_00", calib_dir=None)
    assert rows and all(v is None for r in rows for k, v in r.items() if k != "t_ms")


def test_calibration_loaded_once_until_files_change(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(la, "load_camera_calibration", lambda d: calls.append(d) or {})
    (tmp_path / "cameras.json").write_text(json.dumps({"solved": {"cameras": {}}}))
    for _ in range(3):
        la.angles_for_window(_rings(), 100.0, 300.0, student_id="stu_00", calib_dir=tmp_path)
    assert len(calls) == 1
    time.sleep(0.01)
    (tmp_path / "cam_01.json").write_text("{}")  # recalibrated
    la.angles_for_window(_rings(), 100.0, 300.0, student_id="stu_00", calib_dir=tmp_path)
    assert len(calls) == 2


def test_worker_wakes_on_frame_and_stops(monkeypatch, tmp_path):
    import src.streaming.live_engine as le

    monkeypatch.setattr("src.config.DATA", tmp_path)

    class _Hub:
        def publish(self, ev):
            pass

    eng = le.LiveEngine("live_w", {"cam_01": "rtsp://127.0.0.1:9/a"}, {}, hub=_Hub())
    seen = []
    eng.perception.process = lambda fr: seen.append((fr.frame_idx, time.monotonic()))
    th = threading.Thread(target=eng._infer_loop, args=("cam_01",), daemon=True)
    th.start()
    time.sleep(0.05)
    t0 = time.monotonic()
    eng.capture.workers["cam_01"].on_frame(LiveFrame("cam_01", 7, 0, 0, 0, np.zeros((2, 2, 3), np.uint8)))
    deadline = time.monotonic() + 1.0
    while not seen and time.monotonic() < deadline:
        time.sleep(0.005)
    assert seen and seen[0][0] == 7 and seen[0][1] - t0 < 0.1
    eng._stop.set()
    with eng._latest_lock:
        for c in eng._frame_ready.values():
            c.notify_all()
    th.join(1.0)
    assert not th.is_alive()


def _video(path, n=30, fps=30.0):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), i, np.uint8))
    w.release()


@pytest.mark.parametrize("pace", [True, False])
def test_file_source_is_paced_to_its_fps(tmp_path, pace):
    _video(tmp_path / "v.avi")
    got = []
    w = RtspCameraWorker("cam_01", str(tmp_path / "v.avi"), 0.0, loop_file=False,
                         queue_max=0, on_frame=got.append, pace_files=pace)
    w.start()
    time.sleep(0.5)
    n_half = len(got)
    w.stop()
    if pace:
        assert 8 <= n_half <= 22  # ~15 frames in 0.5 s at 30 fps
    else:
        assert n_half == 30
