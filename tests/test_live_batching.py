"""Body ReID per frame in one batch; YOLO results copied to host once."""

from __future__ import annotations

import numpy as np
import pytest

import src.streaming.live_perception as lp
from src.identity.embedders import StubBodyEmbedder, StubFaceEmbedder
from src.identity.enrollment import EnrollmentGallery
from src.streaming.fast_path import TimestampRingBuffer

torch = pytest.importorskip("torch")


def _tiny_osnet():
    from src.identity.body_reid import OSNetBodyEmbedder

    torch.manual_seed(0)
    e = OSNetBodyEmbedder.__new__(OSNetBodyEmbedder)
    e.device = "cpu"
    e._torch = torch
    e.model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, stride=2), torch.nn.ReLU(), torch.nn.BatchNorm2d(8),
        torch.nn.AdaptiveAvgPool2d((4, 2)), torch.nn.Flatten(), torch.nn.Linear(64, 32),
    ).eval()
    return e


def test_osnet_batch_matches_single():
    e = _tiny_osnet()
    img = (np.random.default_rng(0).random((480, 640, 3)) * 255).astype(np.uint8)
    boxes = [[30 * i, 20, 30 * i + 90, 400] for i in range(7)] + [[600, 400, 700, 900]]  # last one clipped
    single = [e.embed(img, b) for b in boxes]
    batch = e.embed_batch(img, boxes, chunk=3)  # chunking keeps order
    assert len(batch) == len(boxes)
    for a, b in zip(single, batch):
        assert a.shape == b.shape and a.dtype == b.dtype == np.float32
        assert float(np.abs(a - b).max()) < 1e-5


def test_default_embed_batch_is_the_loop():
    e = StubBodyEmbedder()
    img = (np.random.default_rng(1).random((200, 300, 3)) * 255).astype(np.uint8)
    boxes = [[10, 10, 100, 190], [120, 5, 250, 195]]
    assert all(np.array_equal(a, b) for a, b in zip(e.embed_batch(img, boxes), [e.embed(img, b) for b in boxes]))


class _CountingBody(StubBodyEmbedder):
    def __init__(self):
        self.single = 0
        self.batches = 0

    def embed(self, image_bgr, bbox):
        self.single += 1
        return super().embed(image_bgr, bbox)

    def embed_batch(self, image_bgr, bboxes):
        self.batches += 1
        return [StubBodyEmbedder.embed(self, image_bgr, b) for b in bboxes]


@pytest.mark.parametrize("batch_on", [True, False])
def test_live_uses_one_batch_per_frame(monkeypatch, tmp_path, batch_on):
    monkeypatch.setattr("src.config.DATA", tmp_path)
    dets = [{"bbox": [50.0 + 150 * i, 80.0, 160.0 + 150 * i, 420.0], "score": 0.9, "keypoints": None} for i in range(5)]
    monkeypatch.setattr(lp, "_detect_persons", lambda *a, **k: [dict(d) for d in dets])
    monkeypatch.setattr(lp, "_estimate_pose_with_fallback", lambda *a, **k: (np.zeros((133, 3)), "rtmw"))
    monkeypatch.setattr(lp, "_get_rtmlib", lambda: None)
    cfg = dict(lp._identity_cfg())
    cfg["live_body_batch"] = batch_on
    monkeypatch.setattr(lp, "_identity_cfg", lambda: cfg)
    gallery = EnrollmentGallery("bat", face_embedder=StubFaceEmbedder(), body_embedder=StubBodyEmbedder())
    p = lp.LivePerception("bat", {"cam_01": TimestampRingBuffer()}, gallery=gallery)
    body = _CountingBody()
    p._face, p._body = StubFaceEmbedder(), body
    img = (np.random.default_rng(2).random((600, 900, 3)) * 255).astype(np.uint8)
    out = p._pose_payload("cam_01", img)
    assert len(out["persons"]) == 5
    assert (body.batches, body.single) == ((1, 0) if batch_on else (0, 5))


def test_to_numpy_handles_tensors_and_arrays():
    from src.perception.yolo_pose_detector import _to_numpy

    t = torch.tensor([[1.5, 2.25], [3.0, 4.0]], dtype=torch.float32)
    assert np.array_equal(_to_numpy(t), t.numpy())
    assert np.array_equal(_to_numpy(np.ones(3)), np.ones(3))
