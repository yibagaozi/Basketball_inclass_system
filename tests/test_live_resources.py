"""Gallery embedders load lazily; live capture keeps no unread frame queue."""

from __future__ import annotations

import numpy as np
import pytest

import src.identity.enrollment as enr
from src.acquisition.rtsp import LiveFrame, RtspCameraWorker
from src.identity.embedders import StubBodyEmbedder, StubFaceEmbedder
from src.identity.enrollment import EnrollmentGallery


@pytest.fixture
def counting_factories(monkeypatch, tmp_path):
    monkeypatch.setattr("src.config.DATA", tmp_path)
    made = {"face": 0, "body": 0}

    def face():
        made["face"] += 1
        return StubFaceEmbedder()

    def body():
        made["body"] += 1
        return StubBodyEmbedder()

    monkeypatch.setattr(enr, "create_face_embedder", face)
    monkeypatch.setattr(enr, "create_body_embedder", body)
    return made


def test_matching_only_gallery_loads_no_models(counting_factories, tmp_path):
    d = tmp_path / "enrollment" / "s1" / "stu_00"
    d.mkdir(parents=True)
    np.save(d / "body_000.npy", np.ones(4, np.float32))
    g = EnrollmentGallery("s1")
    assert g.list_students() == ["stu_00"]
    assert len(g.load_student("stu_00")["body"]) == 1
    assert counting_factories == {"face": 0, "body": 0}


def test_embedders_created_once_on_first_add(counting_factories):
    g = EnrollmentGallery("s2")
    img = np.full((200, 200, 3), 120, np.uint8)
    g.add_body_sample("stu_00", img, [10, 10, 100, 190])
    g.add_body_sample("stu_00", img, [12, 10, 102, 190])
    g.add_face_sample("stu_00", img, [30, 10, 80, 60])
    assert counting_factories == {"face": 1, "body": 1}


def test_passed_embedders_are_used(counting_factories):
    fe, be = StubFaceEmbedder(), StubBodyEmbedder()
    g = EnrollmentGallery("s3", face_embedder=fe, body_embedder=be)
    assert g.face_embedder is fe and g.body_embedder is be
    assert counting_factories == {"face": 0, "body": 0}


def test_new_built_gallery_keeps_explicit_none(counting_factories):
    from src.identity.global_registry import create_face_lazy_gallery

    g = create_face_lazy_gallery("s4")
    assert g.face_embedder is None and g.body_embedder is None
    assert counting_factories == {"face": 0, "body": 0}


def _frame(i: int) -> LiveFrame:
    return LiveFrame("cam_01", i, 0.0, 0.0, 0.0, np.zeros((4, 4, 3), np.uint8))


def test_queue_max_zero_keeps_no_frames():
    seen = []
    w = RtspCameraWorker("cam_01", "rtsp://127.0.0.1:9/x", 0.0, queue_max=0, on_frame=seen.append)
    for i in range(100):
        w._push(_frame(i))
    assert len(w.queue) == 0 and len(seen) == 100


def test_queue_max_positive_unchanged():
    w = RtspCameraWorker("cam_01", "rtsp://127.0.0.1:9/x", 0.0, queue_max=8)
    for i in range(20):
        w._push(_frame(i))
    assert [f.frame_idx for f in w.queue] == list(range(12, 20))


def test_live_engine_builds_without_loading_models_or_queues(counting_factories, monkeypatch):
    import src.streaming.live_engine as le
    import src.streaming.live_perception as lp

    def boom():
        raise AssertionError("models must not load in LiveEngine.__init__")

    monkeypatch.setattr(lp, "create_face_embedder", boom)
    monkeypatch.setattr(lp, "create_body_embedder", boom)

    class _Hub:
        def publish(self, ev):
            pass

    eng = le.LiveEngine("live_r", {"cam_01": "rtsp://127.0.0.1:9/a"}, {}, hub=_Hub())
    assert all(w.queue_max == 0 for w in eng.capture.workers.values())
    assert counting_factories == {"face": 0, "body": 0}
