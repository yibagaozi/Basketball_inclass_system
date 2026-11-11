"""Opt-in live performance switches: inference.feature_interval / inference.rtmw_cameras."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import src.identity.perception as perc
import src.streaming.live_perception as lp
from src.identity.embedders import StubBodyEmbedder
from src.identity.enrollment import EnrollmentGallery
from src.streaming.fast_path import TimestampRingBuffer


class _Face:
    def __init__(self):
        self.calls = 0
        self.rng = np.random.default_rng(0)

    def embed(self, bgr, bbox=None):
        self.calls += 1
        v = self.rng.standard_normal(512).astype(np.float32)
        return v / np.linalg.norm(v)


class _Tracker:
    """Maps detection j → track id ids[j]; records what it was given."""

    def __init__(self, conf="high"):
        self.ids = None
        self.conf = conf
        self.seen = []

    def update(self, dets):
        self.seen.append(dets)
        ids = self.ids or list(range(1, len(dets) + 1))
        return [SimpleNamespace(track_id=ids[j], bbox=d["bbox"], student_id=f"stu_{ids[j]:02d}",
                                identity_confidence=self.conf, age=0, hits=5)
                for j, d in enumerate(dets)]


def _setup(monkeypatch, tmp_path, dets, *, conf="high", **kw):
    monkeypatch.setattr("src.config.DATA", tmp_path)
    monkeypatch.setattr(lp, "_detect_persons", lambda *a, **k: [dict(d) for d in dets])
    monkeypatch.setattr(lp, "_get_rtmlib", lambda: None)
    colors = []
    monkeypatch.setattr(lp, "extract_clothing_color",
                        lambda bgr, bbox, keypoints=None: colors.append(1) or np.ones(8, np.float32))
    monkeypatch.setattr(lp, "_estimate_pose_with_fallback", lambda *a, **k: (None, "x"))
    gallery = EnrollmentGallery("perf", face_embedder=_Face(), body_embedder=StubBodyEmbedder())
    p = lp.LivePerception("perf", {"cam_01": TimestampRingBuffer()}, gallery=gallery, **kw)
    p._face, p._body = _Face(), StubBodyEmbedder()
    trk = _Tracker(conf)
    p._trackers["cam_01"] = trk
    return p, trk, colors


_IMG = np.zeros((720, 1280, 3), np.uint8)
_APART = [{"bbox": [100.0, 100.0, 220.0, 460.0], "score": 0.9},
          {"bbox": [600.0, 100.0, 720.0, 460.0], "score": 0.9}]


def test_default_computes_every_frame(monkeypatch, tmp_path):
    p, _trk, colors = _setup(monkeypatch, tmp_path, _APART)
    for _ in range(5):
        p._pose_payload("cam_01", _IMG)
    assert p._face.calls == 10 and len(colors) == 10


def test_confirmed_tracks_refresh_every_n(monkeypatch, tmp_path):
    p, trk, colors = _setup(monkeypatch, tmp_path, _APART, feature_interval=3)
    for _ in range(7):
        p._pose_payload("cam_01", _IMG)
    # computed on frames 1, 4, 7; reused on 2, 3, 5, 6
    assert p._face.calls == 6 and len(colors) == 6
    f1, f2, f4 = trk.seen[0][0]["face_emb"], trk.seen[1][0]["face_emb"], trk.seen[3][0]["face_emb"]
    assert f2 is f1 and f4 is not f1  # tracker sees the last computed values
    assert all(d["body_emb"] is not None for frame in trk.seen for d in frame)  # body every frame


def test_unconfirmed_tracks_never_reuse(monkeypatch, tmp_path):
    p, _trk, _colors = _setup(monkeypatch, tmp_path, _APART, conf="medium", feature_interval=5)
    for _ in range(4):
        p._pose_payload("cam_01", _IMG)
    assert p._face.calls == 8


def test_overlapping_people_compute_fresh(monkeypatch, tmp_path):
    close = [{"bbox": [100.0, 100.0, 220.0, 460.0], "score": 0.9},
             {"bbox": [150.0, 100.0, 270.0, 460.0], "score": 0.9}]  # IoU ~0.41
    p, _trk, _colors = _setup(monkeypatch, tmp_path, close, feature_interval=5)
    for _ in range(4):
        p._pose_payload("cam_01", _IMG)
    assert p._face.calls == 8


def test_reassigned_detection_drops_cache(monkeypatch, tmp_path):
    p, trk, _colors = _setup(monkeypatch, tmp_path, _APART, feature_interval=5)
    p._pose_payload("cam_01", _IMG)  # computes 2, tracks 1, 2
    trk.ids = [2, 1]  # tracker swaps: reused features landed on the other track
    p._pose_payload("cam_01", _IMG)  # reuses 2 (plan was made before the tracker ran)
    assert p._face.calls == 2
    trk.ids = None
    p._pose_payload("cam_01", _IMG)  # cache dropped → fresh
    assert p._face.calls == 4


def test_rtmw_cameras(monkeypatch, tmp_path):
    p, _trk, _colors = _setup(monkeypatch, tmp_path, _APART, rtmw_cameras="cam_03")
    p._trackers["cam_03"] = _Tracker()
    monkeypatch.setattr(lp, "_get_rtmlib", lambda: object())
    calls = []
    monkeypatch.setattr(lp, "_estimate_pose_with_fallback",
                        lambda *a, **k: calls.append((k.get("use_rtmw", True), k.get("frame_rgb") is not None))
                        or (None, "x"))
    conversions = []
    real = cv2.cvtColor
    monkeypatch.setattr(lp.cv2, "cvtColor",
                        lambda img, code: (code == cv2.COLOR_BGR2RGB and conversions.append(1)) or real(img, code))
    p._pose_payload("cam_01", _IMG)
    assert calls == [(False, False)] * 2 and not conversions
    calls.clear()
    p._pose_payload("cam_03", _IMG)
    assert calls == [(True, True)] * 2 and len(conversions) == 1


def test_pose_fallback_without_rtmw(monkeypatch):
    def boom():
        raise AssertionError("RTMW must not be touched")

    monkeypatch.setattr(perc, "_get_rtmlib", boom)
    kpts, reason = perc._estimate_pose_with_fallback(_IMG, [10, 10, 60, 100], [], 0.3,
                                                    allow_stub=False, use_rtmw=False)
    assert kpts is None and reason.startswith("rtmw_off")


@pytest.mark.parametrize("cfg,interval,cams", [
    ({}, 1, None),
    ({"inference": {"feature_interval": 4, "rtmw_cameras": "cam_03"}}, 4, {"cam_03"}),
    ({"inference": {"rtmw_cameras": ["cam_03", "cam_02"]}}, 1, {"cam_02", "cam_03"}),
])
def test_engine_config(monkeypatch, tmp_path, cfg, interval, cams):
    import src.streaming.live_engine as le

    monkeypatch.setattr("src.config.DATA", tmp_path)

    class _Hub:
        def publish(self, ev):
            pass

    eng = le.LiveEngine("live_p", {"cam_01": "rtsp://127.0.0.1:9/a"}, {}, hub=_Hub(),
                        cfg={**cfg, "observability": {"enabled": False}})
    assert eng.perception._feat_every == interval and eng.perception._rtmw_cams == cams
