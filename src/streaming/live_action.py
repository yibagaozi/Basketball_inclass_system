"""Detect actions from live pose/ball ring buffers (no session mp4)."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.action.detect import _shooting_release_candidates, classify_release_action
from src.action.multicam_release import _phases_for_action
from src.action.pose_only import detect_pose_only_segments
from src.action.registry import is_shooting_action
from src.streaming.fast_path import TimestampRingBuffer
from src.types import ActionClip, ActionPhase


def _frame_ms_map(seq_meta: list[tuple[int, float]]) -> dict[int, float]:
    return {int(f): float(t) for f, t in seq_meta}


def _ms(frame: int, fmap: dict[int, float], fallback: float | None = None) -> float | None:
    if frame in fmap:
        return fmap[frame]
    if not fmap:
        return fallback
    nearest = min(fmap, key=lambda k: abs(k - frame))
    return fmap[nearest]


def _phases_ms(clip: ActionClip, fmap: dict[int, float]) -> list[dict[str, Any]]:
    out = []
    for ph in clip.phases:
        out.append({
            "name": ph.name,
            "start_ms": _ms(int(ph.start), fmap, clip.start_ms),
            "end_ms": _ms(int(ph.end), fmap, clip.end_ms),
        })
    return out


def sequences_from_pose_ring(
    ring: TimestampRingBuffer,
    student_id: str | None = None,
) -> tuple[list[tuple[int, np.ndarray]], list[tuple[int, float]], dict[str, Any]]:
    """Build (frame, kpts133) seq on the ring's camera clock.

    Returns seq, (frame, common_ms) list, and last identity snapshot.
    """
    seq: list[tuple[int, np.ndarray]] = []
    meta: list[tuple[int, float]] = []
    ident: dict[str, Any] = {}
    for i, item in enumerate(ring._items):
        persons = item.payload.get("persons") or []
        chosen = None
        if student_id:
            for p in persons:
                if p.get("student_id") == student_id:
                    chosen = p
                    break
        if chosen is None and persons:
            chosen = max(persons, key=lambda p: float(p.get("score") or 0.0))
        if chosen is None:
            continue
        k = chosen.get("keypoints")
        if k is None:
            continue
        arr = np.asarray(k, dtype=np.float32)
        if arr.ndim != 2:
            continue
        if arr.shape[1] == 2:
            pad = np.zeros((arr.shape[0], 3), dtype=np.float32)
            pad[:, :2] = arr
            pad[:, 2] = 1.0
            arr = pad
        if arr.shape[0] < 17:
            continue
        if arr.shape[0] < 133:
            from src.action.halpe2h36m import coco17_to_wholebody133
            arr = coco17_to_wholebody133(arr[:17])
        fid = int(item.frame_idx if item.frame_idx else i)
        seq.append((fid, arr))
        meta.append((fid, float(item.timestamp_ms)))
        ident = {
            "student_id": chosen.get("student_id"),
            "global_id": chosen.get("global_id"),
            "confidence": chosen.get("identity_confidence"),
            "source": chosen.get("identity_source") or "live_tracker",
        }
    return seq, meta, ident


def list_student_ids(ring: TimestampRingBuffer) -> list[str]:
    found: set[str] = set()
    for item in ring._items:
        for p in item.payload.get("persons") or []:
            sid = p.get("student_id")
            if sid:
                found.add(str(sid))
    return sorted(found)


def ball_by_pose_frame(
    ball_ring: TimestampRingBuffer,
    pose_meta: list[tuple[int, float]],
    slop_ms: float = 80.0,
) -> dict[int, dict]:
    if not pose_meta:
        return {}
    out: dict[int, dict] = {}
    for item in ball_ring._items:
        ball = item.payload.get("ball")
        if not ball or not ball.get("center"):
            continue
        t = float(item.timestamp_ms)
        fid, pt = min(pose_meta, key=lambda x: abs(x[1] - t))
        if abs(pt - t) > slop_ms:
            continue
        out[int(fid)] = {
            "center": list(ball["center"]),
            "bbox": ball.get("bbox"),
            "confidence": float(ball.get("confidence") or 0.0),
        }
    return out


def latest_hoop_xy(ball_ring: TimestampRingBuffer) -> tuple[float, float] | None:
    hoop = None
    for item in reversed(ball_ring._items):
        h = item.payload.get("hoop")
        if h and h.get("center"):
            hoop = h
            break
    if hoop is None:
        return None
    c = hoop["center"]
    return float(c[0]), float(c[1])


def infer_made(
    ball_ring: TimestampRingBuffer,
    release_ms: float | None,
    *,
    window_ms: float = 1800.0,
) -> bool | None:
    if release_ms is None:
        return None
    hoop = None
    for item in ball_ring._items:
        h = item.payload.get("hoop")
        if h and h.get("center"):
            hoop = h
    if hoop is None:
        return None
    hx, hy = float(hoop["center"][0]), float(hoop["center"][1])
    bb = hoop.get("bbox") or [0, 0, 80, 60]
    if len(bb) >= 4 and float(bb[3]) > float(bb[1]) and float(bb[2]) > 20:
        hh = abs(float(bb[3]) - float(bb[1]))
    else:
        hh = float(bb[3] if len(bb) >= 4 else 60.0)
    hoop_bot = hy + 0.55 * max(hh, 20.0)
    near_x = max(40.0, 0.9 * max(hh, 20.0))
    samples = []
    for item in ball_ring._items:
        ball = item.payload.get("ball")
        t = float(item.timestamp_ms)
        if ball and ball.get("center") and release_ms - 200 <= t <= release_ms + window_ms:
            samples.append((t, float(ball["center"][0]), float(ball["center"][1])))
    if len(samples) < 3:
        return None
    above = [s for s in samples if s[2] < hy and abs(s[1] - hx) <= near_x * 1.4]
    below = [s for s in samples if s[2] > hoop_bot and abs(s[1] - hx) <= near_x]
    if above and below:
        return True
    if above and any(abs(s[1] - hx) > near_x * 1.6 for s in samples if s[0] > release_ms):
        return False
    return None


def _clip_to_candidate(
    clip: ActionClip,
    fmap: dict[int, float],
    ident: dict[str, Any],
    made: bool | None,
) -> dict[str, Any]:
    start_ms = _ms(clip.start_frame, fmap)
    end_ms = _ms(clip.end_frame, fmap)
    release_ms = None
    for ph in clip.phases:
        if ph.name == "release":
            release_ms = _ms(int(ph.start), fmap)
            break
    if is_shooting_action(clip.action_type) is False:
        release_ms = None
        made = None
    return {
        "student_id": ident.get("student_id") or clip.student_id,
        "global_id": ident.get("global_id"),
        "identity": ident,
        "action_type": clip.action_type,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "release_ms": release_ms,
        "made": made,
        "confidence": float(clip.confidence),
        "phases": _phases_ms(clip, fmap),
        "shooting_hand": (clip.metadata or {}).get("shooting_hand"),
        "metadata": dict(clip.metadata or {}),
    }


def detect_live_actions(
    pose_ring: TimestampRingBuffer,
    ball_ring: TimestampRingBuffer,
    *,
    student_ids: list[str] | None = None,
    min_seq: int = 28,
) -> list[dict[str, Any]]:
    """cam_03 pose + cam_04 ball → action candidates (not yet WS-emitted)."""
    ids = student_ids or list_student_ids(pose_ring)
    if not ids:
        ids = [None]  # type: ignore[list-item]
    hoop_xy = latest_hoop_xy(ball_ring)
    out: list[dict[str, Any]] = []
    for sid in ids:
        seq, meta, ident = sequences_from_pose_ring(pose_ring, sid)
        if len(seq) < min_seq:
            continue
        fmap = _frame_ms_map(meta)
        ball_map = ball_by_pose_frame(ball_ring, meta)
        try:
            peaks = _shooting_release_candidates(seq, min_peak_distance=40, ball_by_frame=ball_map)
        except Exception:
            peaks = []
        frames = [f for f, _ in seq]
        used_spans: list[tuple[int, int]] = []
        for peak_idx, conf, shooting_hand, hand_meta in peaks:
            release = frames[peak_idx]
            try:
                atype, cls_meta = classify_release_action(seq, release, hoop_xy=hoop_xy)
            except Exception:
                atype, cls_meta = "free_throw", {}
            if not is_shooting_action(atype):
                atype = "free_throw"
            start = frames[max(0, peak_idx - 40)]
            end = frames[min(len(frames) - 1, peak_idx + 25)]
            phases = _phases_for_action(atype, start, release, end)
            clip = ActionClip(
                action_type=atype,
                start_frame=start,
                end_frame=end,
                phases=phases,
                confidence=float(conf),
                student_id=sid,
                metadata={
                    "shooting_hand": shooting_hand,
                    "shooting_hand_meta": hand_meta,
                    "action_classify": cls_meta,
                    "detector": "live_release",
                },
            )
            made = infer_made(ball_ring, _ms(release, fmap))
            out.append(_clip_to_candidate(clip, fmap, ident if ident.get("student_id") else {
                **ident, "student_id": sid,
            }, made))
            used_spans.append((start, end))

        try:
            pose_clips = detect_pose_only_segments(seq, ball_by_frame=ball_map)
        except Exception:
            pose_clips = []
        for clip in pose_clips:
            overlap = False
            for a, b in used_spans:
                inter = min(clip.end_frame, b) - max(clip.start_frame, a)
                if inter > 8:
                    overlap = True
                    break
            if overlap:
                continue
            out.append(_clip_to_candidate(clip, fmap, ident if ident.get("student_id") else {
                **ident, "student_id": sid,
            }, None))
    return out


def is_new_action(
    cand: dict[str, Any],
    emitted: list[dict[str, Any]],
    *,
    min_gap_ms: float = 1600.0,
) -> bool:
    key_t = cand.get("release_ms")
    if key_t is None:
        key_t = cand.get("start_ms")
    if key_t is None:
        return True
    sid = cand.get("student_id")
    atype = cand.get("action_type")
    for prev in emitted:
        if prev.get("student_id") != sid or prev.get("action_type") != atype:
            continue
        pt = prev.get("release_ms")
        if pt is None:
            pt = prev.get("start_ms")
        if pt is None:
            continue
        if abs(float(key_t) - float(pt)) < min_gap_ms:
            return False
    return True
