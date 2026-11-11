"""Live pose only for tracks detected in this frame (offline rule: skip age > 0)."""

from __future__ import annotations

import numpy as np

import src.streaming.live_perception as lp
from src.identity.embedders import StubBodyEmbedder, StubFaceEmbedder
from src.identity.enrollment import EnrollmentGallery
from src.streaming.fast_path import TimestampRingBuffer


def test_coasting_track_gets_no_pose(monkeypatch, tmp_path):
    monkeypatch.setattr("src.config.DATA", tmp_path)
    frames = [
        [{"bbox": [100.0, 100.0, 220.0, 460.0], "score": 0.9, "keypoints": None},
         {"bbox": [900.0, 100.0, 1020.0, 460.0], "score": 0.9, "keypoints": None}],
        [{"bbox": [104.0, 100.0, 224.0, 460.0], "score": 0.9, "keypoints": None}],  # 2nd person gone
    ]
    it = iter(frames)
    monkeypatch.setattr(lp, "_detect_persons", lambda *a, **k: next(it))
    posed: list[list[float]] = []

    def fake_pose(frame, bbox, detections, kpt_thr, **kw):
        posed.append(list(bbox))
        return np.zeros((133, 3), np.float32), "rtmw"

    monkeypatch.setattr(lp, "_estimate_pose_with_fallback", fake_pose)
    gallery = EnrollmentGallery("live_c", face_embedder=StubFaceEmbedder(), body_embedder=StubBodyEmbedder())
    p = lp.LivePerception("live_c", {"cam_01": TimestampRingBuffer()}, gallery=gallery)
    p._face, p._body = StubFaceEmbedder(), StubBodyEmbedder()
    img = np.full((720, 1280, 3), 90, np.uint8)

    first = p._pose_payload("cam_01", img)
    assert len(first["persons"]) == 2
    posed.clear()
    second = p._pose_payload("cam_01", img)
    assert len(second["persons"]) == 1  # the vanished person's track is coasting
    assert posed == [[104.0, 100.0, 224.0, 460.0]]
