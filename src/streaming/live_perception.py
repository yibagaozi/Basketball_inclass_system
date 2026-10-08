"""Per-frame live perception into TimestampRingBuffer (no raw mp4)."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any

import numpy as np

from src.acquisition.rtsp import LiveFrame
from src.cameras.registry import camera_runs_pose2d
from src.identity.clothing_color import extract_clothing_color
from src.identity.embedders import create_body_embedder, create_face_embedder
from src.identity.enrollment import EnrollmentGallery
from src.identity.perception import (
    _create_tracker,
    _detect_persons,
    _estimate_face_bbox,
    _estimate_pose_with_fallback,
    _person_detect_score_threshold,
    _person_kpt_score_threshold,
    compute_alpha,
)
from src.streaming.fast_path import TimestampRingBuffer

# 分段耗时统计：LIVE_PROF=1 打开，每 5 秒打印一次各阶段平均耗时与人数
_PROF_ON = os.environ.get("LIVE_PROF", "") not in ("", "0", "false", "False")
_PROF_SUM: dict[str, float] = defaultdict(float)
_PROF_CNT: dict[str, int] = defaultdict(int)
_PROF_LAST = [0.0]


class _Stage:
    """with _Stage("face"): ... —— 关掉时开销可忽略。"""

    __slots__ = ("name", "t0")

    def __init__(self, name: str):
        self.name = name
        self.t0 = 0.0

    def __enter__(self):
        if _PROF_ON:
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if not _PROF_ON:
            return False
        _PROF_SUM[self.name] += (time.perf_counter() - self.t0) * 1000.0
        _PROF_CNT[self.name] += 1
        return False


def _prof_note(name: str, value: float) -> None:
    if _PROF_ON:
        _PROF_SUM[name] += value
        _PROF_CNT[name] += 1


def _prof_report() -> None:
    """每 5 秒打印一次；单位毫秒,persons 为每帧平均人数。"""
    if not _PROF_ON:
        return
    now = time.perf_counter()
    if now - _PROF_LAST[0] < 5.0:
        return
    _PROF_LAST[0] = now
    parts = []
    for k in sorted(_PROF_SUM):
        n = max(_PROF_CNT[k], 1)
        parts.append(f"{k}={_PROF_SUM[k] / n:.1f}")
    print("  [prof] " + "  ".join(parts) + "  (ms, persons 为计数)", flush=True)
    _PROF_SUM.clear()
    _PROF_CNT.clear()


class LivePerception:
    """YOLO pose + tracker on cam_01–03; ball/hoop on cam_04."""

    def __init__(
        self,
        session_id: str,
        rings: dict[str, TimestampRingBuffer],
        *,
        gallery: EnrollmentGallery | None = None,
        global_map: dict[str, str] | None = None,
    ):
        self.session_id = session_id
        self.rings = rings
        self.gallery = gallery or EnrollmentGallery(session_id)
        self.global_map = dict(global_map or {})
        self._trackers: dict[str, Any] = {}
        self._face = None
        self._body = None
        self._ball = None
        self._hoop_lock: dict[str, Any] | None = None

        for sid, data in (
            (s, self.gallery.load_student(s)) for s in self.gallery.list_students()
        ):
            gid = (data.get("meta") or {}).get("global_id")
            if gid:
                self.global_map.setdefault(sid, str(gid))

    def _tracker(self, camera_id: str):
        if camera_id not in self._trackers:
            self._trackers[camera_id] = _create_tracker(self.gallery)
        return self._trackers[camera_id]

    def _ensure_id_models(self) -> None:
        if self._face is None:
            self._face = create_face_embedder()
        if self._body is None:
            self._body = create_body_embedder()

    def _ensure_ball(self) -> None:
        if self._ball is None:
            from src.shot.yolo_detector import YoloBallHoopDetector
            self._ball = YoloBallHoopDetector()

    def process(self, frame: LiveFrame) -> None:
        cam = frame.camera_id
        ring = self.rings.get(cam)
        if ring is None:
            return
        if camera_runs_pose2d(cam):
            with _Stage("0_total_pose"):
                payload = self._pose_payload(cam, frame.bgr)
        else:
            with _Stage("0_total_ball"):
                payload = self._ball_payload(frame.bgr)
        ring.push(frame.common_ms, frame.frame_idx, payload)
        _prof_report()

    def _pose_payload(self, camera_id: str, bgr: np.ndarray) -> dict[str, Any]:
        det_thr = _person_detect_score_threshold()
        kpt_thr = _person_kpt_score_threshold()
        with _Stage("1_detect"):
            detections = _detect_persons(bgr, det_thr)
        self._ensure_id_models()
        _prof_note("persons", float(len(detections)))
        det_for_track: list[dict] = []
        for d in detections:
            bbox = list(map(float, d["bbox"]))
            face_bbox = _estimate_face_bbox(bbox, bgr.shape)
            with _Stage("2_face_emb"):
                face_emb = self._face.embed(bgr, face_bbox) if self._face else None
            with _Stage("3_body_emb"):
                body_emb = self._body.embed(bgr, bbox) if self._body else None
            with _Stage("4_color"):
                color = extract_clothing_color(bgr, bbox, keypoints=d.get("keypoints"))
            alpha = compute_alpha(face_bbox, 1.0 if face_emb is not None else 0.0)
            det_for_track.append({
                "bbox": bbox,
                "face_emb": face_emb,
                "body_emb": body_emb,
                "color_desc": color,
                "alpha": alpha,
                "score": float(d.get("score") or 0.5),
                "keypoints": d.get("keypoints"),
            })
        with _Stage("5_tracker"):
            tracks = self._tracker(camera_id).update(det_for_track)
        persons: list[dict[str, Any]] = []
        for t in tracks:
            with _Stage("6_pose"):
                kpts, src = _estimate_pose_with_fallback(
                    bgr, t.bbox, detections, kpt_thr,
                )
            sid = t.student_id
            gid = self.global_map.get(sid) if sid else None
            persons.append({
                "student_id": sid,
                "global_id": gid,
                "identity_confidence": t.identity_confidence,
                "identity_source": "face_gallery" if gid else "session_gallery",
                "score": float(t.hits),
                "bbox": list(t.bbox),
                "keypoints": None if kpts is None else kpts.tolist(),
                "pose_source": src,
            })
        return {"camera_id": camera_id, "persons": persons}

    def _ball_payload(self, bgr: np.ndarray) -> dict[str, Any]:
        try:
            self._ensure_ball()
            det = self._ball.detect(bgr)
        except Exception:
            return {"camera_id": "cam_04", "ball": None, "hoop": None}
        balls = det.get("ball") or []
        hoops = det.get("hoop") or []
        ball = None
        if balls:
            b = balls[0]
            ball = {
                "center": list(b["center"]),
                "bbox": list(b.get("bbox") or []),
                "confidence": float(b.get("confidence") or 0.0),
            }
        hoop = None
        if hoops:
            h = hoops[0]
            hoop = {
                "center": list(h["center"]),
                "bbox": list(h.get("bbox") or []),
                "confidence": float(h.get("confidence") or 0.0),
            }
            if self._hoop_lock is None and hoop["center"]:
                self._hoop_lock = hoop
        if self._hoop_lock is not None:
            hoop = self._hoop_lock
        return {"camera_id": "cam_04", "ball": ball, "hoop": hoop}
