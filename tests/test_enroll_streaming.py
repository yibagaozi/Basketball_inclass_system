"""Streaming live-enrollment scan vs batch scan (stub detector/embedders, no models)."""

from __future__ import annotations

import numpy as np
import pytest

import src.identity.sequential_enroll as se
from src.identity.embedders import StubBodyEmbedder, StubFaceEmbedder

H, W = 360, 640


def _keypoints(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """COCO-17 with confident face + level shoulders (frontal)."""
    k = np.zeros((17, 3), np.float32)
    cx, w, h = (x1 + x2) / 2, x2 - x1, y2 - y1
    offsets = {0: (0, .08), 1: (-.04, .06), 2: (.04, .06), 3: (-.08, .07), 4: (.08, .07),
               5: (-.2, .22), 6: (.2, .22)}
    for i, (dx, dy) in offsets.items():
        k[i] = (cx + dx * w, y1 + dy * h, 0.9)
    return k


def _fake_detect(backend, kind, frame, score_thr):
    v = int(frame[0, 0, 0])  # frame content encodes whether / where a person is
    if v == 0:
        return []
    x1, y1 = 200.0 + v % 7, 40.0
    x2, y2 = x1 + 160, y1 + 300
    return [{"bbox": [x1, y1, x2, y2], "score": 0.9, "keypoints": _keypoints(x1, y1, x2, y2)}]


def _frames(n: int = 60) -> list[tuple[int, float, np.ndarray]]:
    # blocks of 10 empty frames, then 10 frames with a person
    return [(i, i / 7.5, np.full((H, W, 3), (i // 10) % 2 * (1 + i), np.uint8)) for i in range(n)]


@pytest.fixture
def stub_models(monkeypatch):
    monkeypatch.setattr(se, "_load_yolo", lambda: (None, "fake"))
    monkeypatch.setattr(se, "_detect_persons_pose", _fake_detect)
    monkeypatch.setattr(se, "create_body_embedder", StubBodyEmbedder)
    monkeypatch.setattr(se, "create_face_embedder", StubFaceEmbedder)


def test_streaming_scan_matches_batch(stub_models):
    frames = _frames()
    batch = se.scan_frontal_candidates_from_frames(frames, **se.live_scan_kwargs())

    scan = se.StreamingFrontalScan(**se.live_scan_kwargs())
    for f in frames:
        scan.submit(*f)
    stream = scan.finish()

    assert len(stream) == len(batch) == 30
    for a, b in zip(batch, stream):
        assert (a.frame_idx, a.bbox, a.frontal_score, a.quality) == (b.frame_idx, b.bbox, b.frontal_score, b.quality)
        assert np.array_equal(a.body_emb, b.body_emb)
        assert np.array_equal(a.color_desc, b.color_desc)
        assert b.frame_bgr is frames[b.frame_idx][2]  # kept by reference, not copied
    assert scan.submitted == len(frames)
    assert isinstance(scan.body_embedder, StubBodyEmbedder)
    assert isinstance(scan.face_embedder, StubFaceEmbedder)


def test_streaming_scan_reraises_worker_error(stub_models, monkeypatch):
    def broken_detector(backend, kind, frame, score_thr):
        raise RuntimeError("detector failed")

    monkeypatch.setattr(se, "_detect_persons_pose", broken_detector)
    scan = se.StreamingFrontalScan(**se.live_scan_kwargs())
    for f in _frames(5):
        scan.submit(*f)
    with pytest.raises(RuntimeError, match="detector failed"):
        scan.finish()


def _write_video(path, n_frames: int, fps: float = 15.0) -> None:
    import cv2

    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for i in range(n_frames):
        w.write(np.full((48, 64, 3), i * 5 % 255, np.uint8))
    w.release()


def test_video_sampling_follows_video_timeline(tmp_path):
    from src.streaming.live_identity import collect_video_frames

    video = tmp_path / "enroll.avi"
    _write_video(video, 45)  # 3 s at 15 fps
    got: list[tuple[int, float]] = []
    n = collect_video_frames(video, on_frame=lambda i, t, fr: got.append((i, t)), sample_hz=8.0)
    # 15 fps sampled at 8 Hz keeps every 2nd frame (same rule as the live sampler)
    assert n == len(got) == 23
    assert [i for i, _ in got] == list(range(0, 45, 2))
    assert got[1][1] == pytest.approx(2 / 15)

    got.clear()
    collect_video_frames(video, on_frame=lambda i, t, fr: got.append((i, t)), sample_hz=8.0, max_seconds=1.0)
    assert max(t for _, t in got) < 1.0


def test_video_missing_file_raises(tmp_path):
    from src.streaming.live_identity import collect_video_frames

    with pytest.raises(RuntimeError):
        collect_video_frames(tmp_path / "missing.mp4", on_frame=lambda *a: None)
