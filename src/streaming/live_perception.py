"""Per-frame live perception into TimestampRingBuffer (no raw mp4)."""

from __future__ import annotations

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
            payload = self._pose_payload(cam, frame.bgr)
        else:
            payload = self._ball_payload(frame.bgr)
        ring.push(frame.common_ms, frame.frame_idx, payload)

    def _pose_payload(self, camera_id: str, bgr: np.ndarray) -> dict[str, Any]:
        det_thr = _person_detect_score_threshold()
        kpt_thr = _person_kpt_score_threshold()
        detections = _detect_persons(bgr, det_thr)
        self._ensure_id_models()
        det_for_track: list[dict] = []
        for d in detections:
            bbox = list(map(float, d["bbox"]))
            face_bbox = _estimate_face_bbox(bbox, bgr.shape)
            face_emb = self._face.embed(bgr, face_bbox) if self._face else None
            body_emb = self._body.embed(bgr, bbox) if self._body else None
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
        tracks = self._tracker(camera_id).update(det_for_track)
        persons: list[dict[str, Any]] = []
        for t in tracks:
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
