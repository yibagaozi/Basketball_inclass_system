"""3D joint angles for a live action window via multi-view triangulation."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.action.halpe2h36m import wholebody133_to_h36m
from src.pose.angles import compute_h36m_angles, pack_angle_row
from src.pose.triangulate import load_camera_calibration, triangulate_skeleton17
from src.streaming.fast_path import RingFrame, TimestampRingBuffer


def _kpts_arr(person: dict[str, Any]) -> np.ndarray | None:
    k = person.get("keypoints")
    if k is None:
        return None
    arr = np.asarray(k, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 17:
        return None
    if arr.shape[1] == 2:
        out = np.zeros((arr.shape[0], 3), dtype=np.float64)
        out[:, :2] = arr
        out[:, 2] = 1.0
        arr = out
    if arr.shape[0] >= 133:
        return wholebody133_to_h36m(arr)
    if arr.shape[0] >= 17:
        # already H36M-ish / COCO-17 — convert via 133 pad
        from src.action.halpe2h36m import coco17_to_wholebody133

        return wholebody133_to_h36m(coco17_to_wholebody133(arr[:17]))
    return None


def _person_for_student(payload: dict[str, Any], student_id: str | None) -> dict[str, Any] | None:
    persons = payload.get("persons") or []
    if student_id:
        for p in persons:
            if p.get("student_id") == student_id:
                return p
    if not persons:
        return None
    return max(persons, key=lambda p: float(p.get("score") or 0.0))


def nearest_ring_frame(ring: TimestampRingBuffer, t_ms: float, slop_ms: float = 80.0) -> RingFrame | None:
    items = list(ring._items)
    if not items:
        return None
    best = min(items, key=lambda x: abs(x.timestamp_ms - t_ms))
    if abs(best.timestamp_ms - t_ms) > slop_ms:
        return None
    return best


def angles_for_window(
    pose_rings: dict[str, TimestampRingBuffer],
    start_ms: float,
    end_ms: float,
    *,
    student_id: str | None,
    calib_dir: Any | None,
    shooting_hand: str | None = None,
    pose_cams: tuple[str, ...] = ("cam_01", "cam_02", "cam_03"),
    sample_step_ms: float = 33.0,
) -> list[dict[str, Any]]:
    """Triangulate target student in [start_ms, end_ms]; missing angles are null."""
    cameras = load_camera_calibration(calib_dir)
    usable_cams = [c for c in pose_cams if c in cameras]
    anchor = pose_rings.get("cam_03") or next(iter(pose_rings.values()), None)
    if anchor is None:
        return []
    window = anchor.window(start_ms, end_ms)
    if not window:
        t = start_ms
        out = []
        while t <= end_ms:
            out.append(pack_angle_row(t, {}))
            t += sample_step_ms
        return out

    rows: list[dict[str, Any]] = []
    last_t = None
    for fr in window:
        t = float(fr.timestamp_ms)
        if last_t is not None and (t - last_t) < sample_step_ms * 0.5:
            continue
        last_t = t
        kpts_by_cam: dict[str, np.ndarray] = {}
        for cam in pose_cams:
            ring = pose_rings.get(cam)
            if ring is None:
                continue
            hit = fr if cam == "cam_03" else nearest_ring_frame(ring, t)
            if hit is None:
                continue
            person = _person_for_student(hit.payload, student_id)
            if person is None:
                continue
            k = _kpts_arr(person)
            if k is not None:
                kpts_by_cam[cam] = k
        if len(usable_cams) >= 2 and len([c for c in kpts_by_cam if c in cameras]) >= 2:
            xyz, _conf = triangulate_skeleton17(kpts_by_cam, cameras)
            ang = compute_h36m_angles(xyz, shooting_hand=shooting_hand)
        else:
            ang = {}
        rows.append(pack_angle_row(t, ang))
    return rows
