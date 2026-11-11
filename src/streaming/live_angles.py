"""3D joint angles for a live action window via multi-view triangulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.action.halpe2h36m import wholebody133_to_h36m
from src.pose.angles import compute_h36m_angles, pack_angle_row
from src.pose.triangulate import load_camera_calibration, triangulate_skeleton17
from src.streaming.fast_path import RingFrame, TimestampRingBuffer, is_sorted, nearest_index_first, ring_items


def _kpts_arr(person: dict[str, Any]) -> np.ndarray | None:
    k = person.get("keypoints")
    if k is None:
        return None
    arr = np.array(k, dtype=np.float64)  # copy: never hand ring data to the math below
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
    """The named student in this view, or None (never another person: triangulating
    different people across cameras gives meaningless angles)."""
    persons = payload.get("persons") or []
    if student_id:
        for p in persons:
            if p.get("student_id") == student_id:
                return p
        return None
    if not persons:
        return None
    return max(persons, key=lambda p: float(p.get("score") or 0.0))


def _nearest(items: list[RingFrame], ts: list[float] | None, t_ms: float, slop_ms: float) -> RingFrame | None:
    if not items:
        return None
    if ts is not None:
        best = items[nearest_index_first(ts, t_ms)]
    else:  # timestamps not monotonic: original linear scan
        best = min(items, key=lambda x: abs(x.timestamp_ms - t_ms))
    if abs(best.timestamp_ms - t_ms) > slop_ms:
        return None
    return best


def _sorted_ts(items: list[RingFrame]) -> list[float] | None:
    ts = [x.timestamp_ms for x in items]
    return ts if is_sorted(ts) else None


def nearest_ring_frame(
    ring: TimestampRingBuffer | list[RingFrame], t_ms: float, slop_ms: float = 80.0,
) -> RingFrame | None:
    items = ring_items(ring)
    return _nearest(items, _sorted_ts(items), t_ms, slop_ms)


_CALIB_CACHE: dict[str, tuple[tuple, dict]] = {}


def _calibration(calib_dir: Any | None) -> dict[str, dict[str, Any]]:
    """Cameras of this session's calibration, reloaded only when its files change.

    ``None`` means the session has no calibration: no cameras, angles null (as
    run_live_ws announces). Before, load_camera_calibration(None) silently fell
    back to the offline data/calibration/v2_4cam_zoned of another setup.
    """
    if calib_dir is None:
        return {}
    d = Path(calib_dir)
    try:
        sig = tuple(sorted(
            (f.name, f.stat().st_mtime_ns, f.stat().st_size)
            for f in d.iterdir() if f.name == "cameras.json" or (f.name.startswith("cam_") and f.suffix == ".json")
        ))
    except OSError:
        sig = ()
    hit = _CALIB_CACHE.get(str(d))
    if hit is not None and hit[0] == sig:
        return hit[1]
    cams = load_camera_calibration(d) if sig else {}
    _CALIB_CACHE[str(d)] = (sig, cams)
    return cams


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
    cameras = _calibration(calib_dir)
    usable_cams = [c for c in pose_cams if c in cameras]
    anchor = pose_rings.get("cam_03") or next(iter(pose_rings.values()), None)
    if anchor is None:
        return []
    window = anchor.window(start_ms, end_ms)
    # one consistent snapshot per camera for the whole window (rings keep growing)
    snaps = {cam: ring_items(pose_rings[cam]) for cam in pose_cams if pose_rings.get(cam) is not None}
    snap_ts = {cam: _sorted_ts(items) for cam, items in snaps.items()}
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
            items = snaps.get(cam)
            if items is None:
                continue
            hit = fr if cam == "cam_03" else _nearest(items, snap_ts[cam], t, 80.0)
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
