"""Per-frame live perception into TimestampRingBuffer (no raw mp4)."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any

import cv2
import numpy as np

from src.acquisition.rtsp import LiveFrame
from src.cameras.registry import camera_runs_pose2d
from src.identity.clothing_color import extract_clothing_color
from src.identity.embedders import create_body_embedder, create_face_embedder
from src.identity.enrollment import EnrollmentGallery
from src.identity.perception import (
    BACKEND_ERRORS,
    _create_tracker,
    _identity_cfg,
    _detect_persons,
    _estimate_face_bbox,
    _estimate_pose_with_fallback,
    _get_rtmlib,
    _get_yolo_pose,
    _perception_cfg,
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
        self._ball_error: str | None = None  # set once: no per-frame reload attempts
        self._ball_runtime_errors = 0
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

    def preload(self, *, ball: bool = True) -> list[str]:
        """Load and warm up every live model once, in order, before frames arrive.

        Before, YOLO-Pose / RTMW / ball YOLO were created lazily by 3-4 camera
        threads on their first frames at the same time (duplicate loads, summed
        VRAM peaks), and any failure silently fell back for the rest of the run,
        down to a fixed fake person per frame.

        Raises RuntimeError when no real person detector is available.
        Returns warnings for optional models (RTMW, face/body ReID, ball).
        """
        from src.identity.embedders import StubBodyEmbedder, StubFaceEmbedder

        warnings: list[str] = []
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        box = [560.0, 160.0, 720.0, 600.0]

        yolo = _get_yolo_pose() if _perception_cfg().get("person_detector", "yolo_pose") == "yolo_pose" else None
        rtm = _get_rtmlib()
        if yolo is not None:
            yolo._ensure_model()  # raises with the real reason (missing file, CUDA, ...)
            yolo.detect_persons(dummy, score_thr=0.5)  # warm up CUDA kernels
        elif rtm is None or getattr(rtm, "detector", None) is None:
            raise RuntimeError(
                "实时推理没有可用的人员检测模型（YOLO-Pose 与 RTMLib YOLOX 都不可用）："
                f"{BACKEND_ERRORS or 'see configs/models.yaml'}"
            )
        if rtm is None:
            warnings.append(f"RTMW-133 未加载（{BACKEND_ERRORS.get('rtmlib', '?')}），姿态回退为 YOLO-Pose 17 点")
        else:
            rtm.estimate_pose133(dummy, box)

        self._ensure_id_models()
        if isinstance(self._face, StubFaceEmbedder):
            warnings.append("人脸模型（InsightFace）未加载，使用占位嵌入器，人脸不参与身份匹配")
        if isinstance(self._body, StubBodyEmbedder):
            warnings.append("身体 ReID 模型（OSNet）未加载，使用占位嵌入器，身份匹配结果不可用")

        if ball:
            try:
                self._ensure_ball()
                self._ball._ensure_model()
                self._ball.detect(dummy)
            except Exception as exc:
                self._ball_error = f"{type(exc).__name__}: {exc}"
                warnings.append(f"篮球/篮筐模型不可用（{self._ball_error}），ball/hoop 为空、made 为 null")

        for w in warnings:
            print(f"  [live] WARNING: {w}", flush=True)
        return warnings

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
            detections = _detect_persons(bgr, det_thr, allow_stub=False)
        self._ensure_id_models()
        _prof_note("persons", float(len(detections)))
        det_for_track: list[dict] = []
        body_embs: list | None = None
        if self._body and len(detections) > 1 and _identity_cfg().get("live_body_batch", True):
            # one ReID forward pass per frame instead of one per person
            with _Stage("3_body_emb_batch"):
                body_embs = self._body.embed_batch(bgr, [list(map(float, d["bbox"])) for d in detections])
        for j, d in enumerate(detections):
            bbox = list(map(float, d["bbox"]))
            face_bbox = _estimate_face_bbox(bbox, bgr.shape)
            with _Stage("2_face_emb"):
                face_emb = self._face.embed(bgr, face_bbox) if self._face else None
            if body_embs is not None:
                body_emb = body_embs[j]
            else:
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
        # RTMW wants RGB: convert this frame once, not once per person (was a
        # full 1080p copy + channel flip per track)
        frame_rgb = None
        if _get_rtmlib() is not None and any(t.age == 0 for t in tracks):
            frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for t in tracks:
            # Coasting track (no detection this frame): its bbox is stale, so a pose
            # there would belong to whoever stands in that spot now. Same rule as
            # the offline path (run_perception_on_video skips age > 0).
            if t.age > 0:
                continue
            with _Stage("6_pose"):
                kpts, src = _estimate_pose_with_fallback(
                    bgr, t.bbox, detections, kpt_thr, allow_stub=False, frame_rgb=frame_rgb,
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
                # ndarray, not .tolist(): 133x3 nested lists were ~20 KB and 500+ Python
                # objects per person in the 60 s rings (memory + multi-100 ms GC pauses).
                # Readers copy before use (live_action._kpts133, live_angles._kpts_arr).
                "keypoints": None if kpts is None else np.asarray(kpts),
                "pose_source": src,
            })
        return {"camera_id": camera_id, "persons": persons}

    def _ball_payload(self, bgr: np.ndarray) -> dict[str, Any]:
        if self._ball_error is not None:
            return {"camera_id": "cam_04", "ball": None, "hoop": None}
        try:
            self._ensure_ball()
            det = self._ball.detect(bgr)
        except Exception as exc:
            self._ball_runtime_errors += 1
            if self._ball_runtime_errors in (1, 10, 100) or self._ball_runtime_errors % 1000 == 0:
                print(f"  [live] cam_04 ball detect failed ({self._ball_runtime_errors}x): "
                      f"{type(exc).__name__}: {exc}", flush=True)
            if isinstance(exc, (FileNotFoundError, ImportError)):
                self._ball_error = f"{type(exc).__name__}: {exc}"  # permanent: stop retrying
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
