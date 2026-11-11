"""Live model loading: once, thread-safe, fail fast instead of silent stubs."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import src.identity.perception as perc
from src.identity.embedders import StubBodyEmbedder, StubFaceEmbedder
from src.identity.enrollment import EnrollmentGallery
from src.streaming.fast_path import TimestampRingBuffer
from src.streaming.live_perception import LivePerception


@pytest.fixture
def fresh_backends(monkeypatch):
    monkeypatch.setattr(perc, "_yolo_pose_backend", None)
    monkeypatch.setattr(perc, "_rtmlib_backend", None)
    monkeypatch.setattr(perc, "BACKEND_ERRORS", {})
    import src.streaming.live_perception as lp
    monkeypatch.setattr(lp, "BACKEND_ERRORS", perc.BACKEND_ERRORS)


class _FakeYolo:
    def __init__(self):
        self.loaded = 0
        self.calls = 0

    def _ensure_model(self):
        self.loaded += 1

    def detect_persons(self, frame, score_thr=None, best_only=False):
        self.calls += 1
        return []


def _perception(tmp_path, monkeypatch) -> LivePerception:
    import src.streaming.live_perception as lp
    monkeypatch.setattr(lp, "create_face_embedder", StubFaceEmbedder)
    monkeypatch.setattr(lp, "create_body_embedder", StubBodyEmbedder)
    monkeypatch.setattr("src.config.DATA", tmp_path)
    gallery = EnrollmentGallery("live_t", face_embedder=StubFaceEmbedder(), body_embedder=StubBodyEmbedder())
    return LivePerception("live_t", {c: TimestampRingBuffer() for c in ("cam_01", "cam_04")}, gallery=gallery)


def test_backend_created_once_under_concurrency(fresh_backends, monkeypatch):
    calls = []

    def slow_create(*a, **k):
        calls.append(1)
        time.sleep(0.05)
        return _FakeYolo()

    monkeypatch.setattr("src.perception.yolo_pose_detector.create_yolo_pose_detector", slow_create)
    got = []
    threads = [threading.Thread(target=lambda: got.append(perc._get_yolo_pose())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1
    assert len({id(x) for x in got}) == 1


def test_missing_rtmlib_is_cached_and_reported_once(fresh_backends, monkeypatch, capsys):
    calls = []

    def missing(*a, **k):
        calls.append(1)
        raise FileNotFoundError("RTMLib models missing: ['models/pose/rtmw_l/end2end.onnx']")

    monkeypatch.setattr("src.perception.rtmlib_backend.create_rtmlib_perception", missing)
    for _ in range(50):  # before: re-checked on every call (per track per frame)
        assert perc._get_rtmlib() is None
    assert len(calls) == 1
    assert capsys.readouterr().out.count("WARNING: rtmlib") == 1


def test_offline_keeps_stub_live_refuses_it(fresh_backends, monkeypatch):
    monkeypatch.setattr(perc, "_get_yolo_pose", lambda: None)
    monkeypatch.setattr(perc, "_get_rtmlib", lambda: None)
    frame = np.zeros((360, 640, 3), np.uint8)
    assert len(perc._detect_persons(frame, 0.5)) == 1  # offline dev path unchanged
    with pytest.raises(RuntimeError):
        perc._detect_persons(frame, 0.5, allow_stub=False)
    kpts, _ = perc._estimate_pose_with_fallback(frame, [100, 50, 200, 300], [], 0.45)
    assert kpts is not None  # stub skeleton offline
    kpts, _ = perc._estimate_pose_with_fallback(frame, [100, 50, 200, 300], [], 0.45, allow_stub=False)
    assert kpts is None


def test_preload_fails_fast_without_person_detector(fresh_backends, monkeypatch, tmp_path):
    import src.streaming.live_perception as lp
    monkeypatch.setattr(lp, "_get_yolo_pose", lambda: None)
    monkeypatch.setattr(lp, "_get_rtmlib", lambda: None)
    p = _perception(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="人员检测"):
        p.preload(ball=False)


def test_preload_warms_up_and_reports_optional_models(fresh_backends, monkeypatch, tmp_path, capsys):
    import src.streaming.live_perception as lp
    yolo = _FakeYolo()
    monkeypatch.setattr(lp, "_get_yolo_pose", lambda: yolo)
    monkeypatch.setattr(lp, "_get_rtmlib", lambda: None)

    class _NoBall:
        def __init__(self):
            raise FileNotFoundError("Ball/hoop YOLO model not found: models/detection/yolo_ball/x.pt")

    monkeypatch.setattr("src.shot.yolo_detector.YoloBallHoopDetector", _NoBall)
    p = _perception(tmp_path, monkeypatch)
    warnings = p.preload(ball=True)
    assert yolo.loaded == 1 and yolo.calls == 1  # loaded + warmed up before any frame
    joined = " ".join(warnings)
    assert "RTMW" in joined and "OSNet" in joined and "篮球" in joined
    assert "WARNING" in capsys.readouterr().out
    # no per-frame reload attempts afterwards
    frame = np.zeros((360, 640, 3), np.uint8)
    for _ in range(5):
        assert p._ball_payload(frame) == {"camera_id": "cam_04", "ball": None, "hoop": None}
    assert p._ball is None
