"""Opt-in live switches (all off by default = 2.2.0 behaviour)."""

from __future__ import annotations

import threading
import time

import numpy as np

import src.streaming.clock as clk
import src.streaming.live_engine as le
import src.streaming.live_perception as lp
from src.acquisition.rtsp import LiveFrame
from src.identity.embedders import StubBodyEmbedder, StubFaceEmbedder
from src.identity.enrollment import EnrollmentGallery
from src.streaming.fast_path import TimestampRingBuffer
from src.streaming.live_action import detect_live_actions


class _Hub:
    def publish(self, ev):
        pass


def _engine(monkeypatch, tmp_path, cfg=None):
    monkeypatch.setattr("src.config.DATA", tmp_path)
    return le.LiveEngine("live_o", {"cam_01": "rtsp://127.0.0.1:9/a"}, {}, hub=_Hub(), cfg=cfg)


def test_defaults_keep_2_2_0_behaviour(monkeypatch, tmp_path):
    eng = _engine(monkeypatch, tmp_path, cfg={})
    assert eng.window_pre_ms is None and eng.window_post_ms is None
    assert all(v == 0.0 for v in eng.min_interval_s.values())
    assert all(w.clock == "wall" for w in eng.capture.workers.values())
    assert eng.perception._hoop_mode == "first"


def test_monotonic_clock_ignores_system_clock_steps(monkeypatch):
    a_wall, a_mono = clk.capture_now_ms("wall"), clk.capture_now_ms("monotonic")
    assert abs(a_wall - a_mono) < 5_000  # same epoch-ms scale
    real = time.time
    monkeypatch.setattr(clk.time, "time", lambda: real() + 3600.0)  # clock stepped +1 h
    assert clk.capture_now_ms("wall") - a_wall > 3_500_000
    assert clk.capture_now_ms("monotonic") - a_mono < 1_000


def test_min_interval_caps_camera_rate(monkeypatch, tmp_path):
    eng = _engine(monkeypatch, tmp_path, cfg={"inference": {"min_interval_ms": {"cam_01": 200}}})
    seen = []
    eng.perception.process = lambda fr: seen.append(fr.frame_idx)
    th = threading.Thread(target=eng._infer_loop, args=("cam_01",), daemon=True)
    th.start()
    push = eng.capture.workers["cam_01"].on_frame
    end = time.monotonic() + 0.65
    i = 0
    while time.monotonic() < end:
        i += 1
        push(LiveFrame("cam_01", i, 0, 0, 0, np.zeros((2, 2, 3), np.uint8)))
        time.sleep(0.005)
    eng._stop.set()
    th.join(1.0)
    assert 2 <= len(seen) <= 5  # ~every 200 ms, always the newest frame
    assert seen == sorted(seen)


def _shot_ring(fps: float):
    ring = TimestampRingBuffer()
    n = int(4.0 * fps)
    for i in range(n):
        t = 1000.0 + i * 1000.0 / fps
        k = np.zeros((133, 3))
        k[:, 2] = 0.9
        k[0, 1], k[5, 1], k[6, 1], k[9, 1] = 220.0, 250.0, 250.0, 300.0
        k[10, 1] = 280.0 - (80.0 if abs(i - n // 2) <= max(1, int(0.15 * fps)) else 0.0) - i * 0.1
        ring.push(t, i + 1, {"persons": [{"student_id": "stu_01", "score": 9.0, "keypoints": k}]})
    return ring


def _span(fps, **kw):
    cands = detect_live_actions(_shot_ring(fps), TimestampRingBuffer(), student_ids=["stu_01"], min_seq=10, **kw)
    shots = [c for c in cands if c.get("release_ms") is not None]
    assert shots
    return shots[0]["end_ms"] - shots[0]["start_ms"]


def test_window_by_ms_spans_time_not_samples():
    # samples (default): 40 before / 25 after the peak → duration depends on processed fps
    assert _span(8.0) > 3500.0 and _span(25.0) < 2700.0
    # ms: ~pose_pre_ms + pose_post_ms at any fps (within one frame step at each end)
    for fps in (8.0, 25.0):
        span = _span(fps, pre_ms=1800.0, post_ms=900.0)
        assert 2700.0 - 2 * 1000.0 / fps <= span <= 2700.0


def _perception_with_hoops(monkeypatch, tmp_path, mode, centers):
    monkeypatch.setattr("src.config.DATA", tmp_path)
    gallery = EnrollmentGallery("hoop", face_embedder=StubFaceEmbedder(), body_embedder=StubBodyEmbedder())
    p = lp.LivePerception("hoop", {"cam_04": TimestampRingBuffer()}, gallery=gallery,
                          hoop_lock=mode, hoop_lock_samples=5)
    it = iter(centers)

    class _Ball:
        def detect(self, bgr):
            c = next(it)
            return {"ball": [], "hoop": [{"center": c, "bbox": (c[0] - 30, c[1] - 20, 60, 40), "confidence": 0.9}]}

    p._ball = _Ball()
    img = np.zeros((10, 10, 3), np.uint8)
    return [p._ball_payload(img)["hoop"]["center"] for _ in centers]


def test_hoop_lock_first_vs_median(monkeypatch, tmp_path):
    centers = [(100, 900)] + [(960 + d, 150 - d) for d in (0, 2, -1, 1, 3, 0, 1)]  # first one is a false positive
    first = _perception_with_hoops(monkeypatch, tmp_path, "first", centers)
    assert first[-1] == [100, 900] or tuple(first[-1]) == (100, 900)  # 2.2.0: stuck on the outlier
    median = _perception_with_hoops(monkeypatch, tmp_path, "median", centers)
    assert abs(median[-1][0] - 960) <= 3 and abs(median[-1][1] - 150) <= 3
