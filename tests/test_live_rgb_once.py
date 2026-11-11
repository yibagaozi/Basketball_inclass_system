"""RTMW input: one RGB conversion per frame, identical to the old per-person copy."""

from __future__ import annotations

import cv2
import numpy as np

import src.streaming.live_perception as lp
from src.identity.embedders import StubBodyEmbedder, StubFaceEmbedder
from src.identity.enrollment import EnrollmentGallery
from src.perception.rtmlib_backend import RTMLibPerception
from src.streaming.fast_path import TimestampRingBuffer


def test_cvtcolor_equals_reversed_copy():
    img = (np.random.default_rng(0).random((1080, 1920, 3)) * 255).astype(np.uint8)
    assert np.array_equal(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), img[:, :, ::-1].copy())


def test_estimate_pose133_uses_given_rgb():
    seen = []
    rt = RTMLibPerception.__new__(RTMLibPerception)
    rt._use_custom_pose = True
    rt.pose = lambda rgb, bboxes: (seen.append(rgb) or np.zeros((1, 133, 2)), np.zeros((1, 133)))
    img = (np.random.default_rng(1).random((120, 160, 3)) * 255).astype(np.uint8)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rt.estimate_pose133(img, [10, 10, 60, 100], frame_rgb=rgb)
    rt.estimate_pose133(img, [10, 10, 60, 100])  # old call style still works
    assert seen[0] is rgb and np.array_equal(seen[1], rgb)


def test_live_converts_once_per_frame(monkeypatch, tmp_path):
    monkeypatch.setattr("src.config.DATA", tmp_path)
    dets = [{"bbox": [100.0 + 200 * i, 100.0, 220.0 + 200 * i, 460.0], "score": 0.9, "keypoints": None} for i in range(4)]
    monkeypatch.setattr(lp, "_detect_persons", lambda *a, **k: [dict(d) for d in dets])
    monkeypatch.setattr(lp, "_get_rtmlib", lambda: object())
    conversions = []
    real = cv2.cvtColor
    monkeypatch.setattr(lp.cv2, "cvtColor",
                        lambda img, code: (code == cv2.COLOR_BGR2RGB and conversions.append(1)) or real(img, code))
    got = []

    def fake_pose(frame, bbox, detections, kpt_thr, *, allow_stub=True, frame_rgb=None):
        got.append(frame_rgb)
        return np.zeros((133, 3)), "rtmw"

    monkeypatch.setattr(lp, "_estimate_pose_with_fallback", fake_pose)
    gallery = EnrollmentGallery("rgb", face_embedder=StubFaceEmbedder(), body_embedder=StubBodyEmbedder())
    p = lp.LivePerception("rgb", {"cam_01": TimestampRingBuffer()}, gallery=gallery)
    p._face, p._body = StubFaceEmbedder(), StubBodyEmbedder()
    img = (np.random.default_rng(2).random((720, 1280, 3)) * 255).astype(np.uint8)
    p._pose_payload("cam_01", img)
    assert len(got) == 4 and len(conversions) == 1
    assert all(g is got[0] for g in got)
    assert np.array_equal(got[0], img[:, :, ::-1])
