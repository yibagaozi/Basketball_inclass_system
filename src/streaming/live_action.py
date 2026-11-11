"""Detect actions from live pose/ball ring buffers (no session mp4)."""

from __future__ import annotations

from bisect import bisect_left
from typing import Any, Callable

import numpy as np

from src.action.detect import _shooting_release_candidates, classify_release_action
from src.action.multicam_release import _phases_for_action
from src.action.pose_only import WindowMemo, detect_pose_only_segments
from src.action.registry import is_shooting_action
from src.streaming.fast_path import (
    RingFrame,
    TimestampRingBuffer,
    is_sorted,
    nearest_index_first,
    ring_items,
)
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
    ring: TimestampRingBuffer | list[RingFrame],
    student_id: str | None = None,
) -> tuple[list[tuple[int, np.ndarray]], list[tuple[int, float]], dict[str, Any]]:
    """Build (frame, kpts133) seq on the ring's camera clock.

    With ``student_id`` only that student's frames are used: frames where the
    student is absent are skipped, never filled with another person's skeleton
    (that mixed people into one sequence and reported actions under whoever
    was chosen last). Without ``student_id`` (empty gallery) the highest-score
    person per frame is used, as before.

    Returns seq, (frame, common_ms) list, and last identity snapshot.
    """
    seq: list[tuple[int, np.ndarray]] = []
    meta: list[tuple[int, float]] = []
    ident: dict[str, Any] = {}
    for i, item in enumerate(ring_items(ring)):
        persons = item.payload.get("persons") or []
        chosen = None
        if student_id:
            for p in persons:
                if p.get("student_id") == student_id:
                    chosen = p
                    break
        elif persons:
            chosen = max(persons, key=lambda p: float(p.get("score") or 0.0))
        if chosen is None:
            continue
        arr = _kpts133(chosen.get("keypoints"))
        if arr is None:
            continue
        fid = int(item.frame_idx if item.frame_idx else i)
        seq.append((fid, arr))
        meta.append((fid, float(item.timestamp_ms)))
        ident = _ident(chosen)
    return seq, meta, ident


def _kpts133(k: Any) -> np.ndarray | None:
    """Keypoints → fresh (133, 3) float32 array, or None when unusable.

    Always a copy: rings hold ndarray keypoints, and the detectors must never
    write into ring data that later passes read again.
    """
    if k is None:
        return None
    arr = np.array(k, dtype=np.float32)
    if arr.ndim != 2:
        return None
    if arr.shape[1] == 2:
        pad = np.zeros((arr.shape[0], 3), dtype=np.float32)
        pad[:, :2] = arr
        pad[:, 2] = 1.0
        arr = pad
    if arr.shape[0] < 17:
        return None
    if arr.shape[0] < 133:
        from src.action.halpe2h36m import coco17_to_wholebody133
        arr = coco17_to_wholebody133(arr[:17])
    return arr


def _ident(p: dict[str, Any]) -> dict[str, Any]:
    return {
        "student_id": p.get("student_id"),
        "global_id": p.get("global_id"),
        "confidence": p.get("identity_confidence"),
        "source": p.get("identity_source") or "live_tracker",
    }


def _usable_raw(k: Any) -> Any:
    """Cheap pre-check equivalent to ``_kpts133(k) is not None``; returns what to
    convert later (ndarrays are only shape-checked, legacy lists are converted)."""
    if k is None:
        return None
    if isinstance(k, np.ndarray):
        return k if (k.ndim == 2 and k.shape[0] >= 17) else None
    return _kpts133(k)


def _student_frames(
    ring: TimestampRingBuffer | list[RingFrame],
    student_ids: list[str],
) -> dict[str, tuple[list[tuple[int, Any]], list[tuple[int, float]], dict[str, Any]]]:
    out: dict[str, tuple[list, list, dict]] = {sid: ([], [], {}) for sid in student_ids if sid}
    for i, item in enumerate(ring_items(ring)):
        seen: set[str] = set()
        for p in item.payload.get("persons") or []:
            sid = p.get("student_id")
            if sid not in out or sid in seen:
                continue
            seen.add(sid)  # first occurrence only, as the per-student scan does
            raw = _usable_raw(p.get("keypoints"))
            if raw is None:
                continue
            fid = int(item.frame_idx if item.frame_idx else i)
            seq, meta, _ = out[sid]
            seq.append((fid, raw))
            meta.append((fid, float(item.timestamp_ms)))
            out[sid] = (seq, meta, _ident(p))
    return out


def _materialize(seq: list[tuple[int, Any]]) -> list[tuple[int, np.ndarray]]:
    return [(f, _kpts133(k)) for f, k in seq]


def sequences_by_student(
    ring: TimestampRingBuffer | list[RingFrame],
    student_ids: list[str],
) -> dict[str, tuple[list[tuple[int, np.ndarray]], list[tuple[int, float]], dict[str, Any]]]:
    """``sequences_from_pose_ring(ring, sid)`` for every named sid in one pass over
    the ring (was: one full ring scan per student per detection round)."""
    return {sid: (_materialize(seq), meta, ident)
            for sid, (seq, meta, ident) in _student_frames(ring, student_ids).items()}


def list_student_ids(ring: TimestampRingBuffer | list[RingFrame]) -> list[str]:
    found: set[str] = set()
    for item in ring_items(ring):
        for p in item.payload.get("persons") or []:
            sid = p.get("student_id")
            if sid:
                found.add(str(sid))
    return sorted(found)


def ball_by_pose_frame(
    ball_ring: TimestampRingBuffer | list[RingFrame],
    pose_meta: list[tuple[int, float]],
    slop_ms: float = 80.0,
) -> dict[int, dict]:
    if not pose_meta:
        return {}
    ts = [pt for _, pt in pose_meta]
    sorted_ts = is_sorted(ts)
    out: dict[int, dict] = {}
    for item in ring_items(ball_ring):
        ball = item.payload.get("ball")
        if not ball or not ball.get("center"):
            continue
        t = float(item.timestamp_ms)
        if sorted_ts:
            fid, pt = pose_meta[nearest_index_first(ts, t)]
        else:  # clock went backwards: keep the original linear scan
            fid, pt = min(pose_meta, key=lambda x: abs(x[1] - t))
        if abs(pt - t) > slop_ms:
            continue
        out[int(fid)] = {
            "center": list(ball["center"]),
            "bbox": ball.get("bbox"),
            "confidence": float(ball.get("confidence") or 0.0),
        }
    return out


def latest_hoop_xy(ball_ring: TimestampRingBuffer | list[RingFrame]) -> tuple[float, float] | None:
    hoop = None
    for item in reversed(ring_items(ball_ring)):
        h = item.payload.get("hoop")
        if h and h.get("center"):
            hoop = h
            break
    if hoop is None:
        return None
    c = hoop["center"]
    return float(c[0]), float(c[1])


def infer_made(
    ball_ring: TimestampRingBuffer | list[RingFrame],
    release_ms: float | None,
    *,
    window_ms: float = 1800.0,
) -> bool | None:
    if release_ms is None:
        return None
    items = ring_items(ball_ring)
    hoop = None
    for item in items:
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
    for item in items:
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


class ActionDetectCache:
    """Per-student memo of the two expensive detectors.

    ``_shooting_release_candidates`` and ``detect_pose_only_segments`` are pure
    functions of (student sequence, ball map). Ring payloads are never modified
    after push, so the sequence is identified by its (frame_idx, timestamp_ms)
    list; a student whose frames and ball samples are unchanged since the last
    round gets the stored result instead of a recomputation.
    """

    def __init__(self) -> None:
        self._memo: dict[Any, tuple[tuple, dict, Any]] = {}
        self._windows: dict[Any, WindowMemo] = {}  # per-student pose-only window memo
        self._anchor: dict[Any, int] = {}  # per-student first frame of the window grid
        self.hits = 0
        self.misses = 0

    def windows(self, sid: Any) -> WindowMemo:
        memo = self._windows.get(sid)
        if memo is None:
            memo = self._windows[sid] = WindowMemo()
        return memo

    def lookup(self, sid: Any, meta: list[tuple[int, float]], ball_map: dict, compute: Callable[[], Any]) -> Any:
        key = tuple(meta)
        hit = self._memo.get(sid)
        if hit is not None and hit[0] == key and hit[1] == ball_map:
            self.hits += 1
            return hit[2]
        self.misses += 1
        val = compute()
        self._memo[sid] = (key, ball_map, val)
        return val

    def retain(self, sids: list[Any]) -> None:
        keep = set(sids)
        for d in (self._memo, self._windows, self._anchor):
            for sid in list(d):
                if sid not in keep:
                    del d[sid]


def _run_detectors(
    seq: list[tuple[int, np.ndarray]], ball_map: dict, memo: WindowMemo | None = None,
) -> tuple[list, list]:
    try:
        peaks = _shooting_release_candidates(seq, min_peak_distance=40, ball_by_frame=ball_map)
    except Exception:
        peaks = []
    try:
        pose_clips = detect_pose_only_segments(seq, ball_by_frame=ball_map, window_memo=memo)
    except Exception:
        pose_clips = []
    if memo is not None:
        memo.end_round()
    return peaks, pose_clips


_POSE_ONLY_STEP = 8  # detect_pose_only_segments default step


def _tail_start(
    meta: list[tuple[int, float]],
    window_ms: float,
    min_items: int,
    anchor_fid: int | None,
) -> int:
    """Start index of a student's detection window.

    Time-based (last ``window_ms``, at least ``min_items`` samples), but moved
    only forward in whole pose-only steps from the previous start, so the
    sliding-window grid stays on the same frames from round to round (windows
    are reused via WindowMemo and results do not jitter as old frames drop out).
    """
    n = len(meta)
    last = meta[-1][1]
    want = 0
    while want < n and meta[want][1] < last - window_ms:
        want += 1
    want = min(want, max(0, n - int(min_items)))
    if anchor_fid is None:
        return want
    fids = [f for f, _ in meta]
    a = bisect_left(fids, anchor_fid)
    if a >= n or fids[a] != anchor_fid:
        return want  # anchor dropped out of the ring: re-anchor
    if a < want:
        a += _POSE_ONLY_STEP * -(-(want - a) // _POSE_ONLY_STEP)
        if a >= n:
            return want
    return a


def detect_live_actions(
    pose_ring: TimestampRingBuffer | list[RingFrame],
    ball_ring: TimestampRingBuffer | list[RingFrame],
    *,
    student_ids: list[str] | None = None,
    min_seq: int = 28,
    cache: ActionDetectCache | None = None,
    window_ms: float | None = None,
    min_items: int = 150,
) -> list[dict[str, Any]]:
    """cam_03 pose + cam_04 ball → action candidates (not yet WS-emitted).

    Rings are snapshotted once so every student sees the same consistent data
    while inference threads keep pushing. Sequences for all students are built
    in one pass over the ring; pass a ``cache`` (the live engine keeps one) to
    skip re-running the detectors for students whose data did not change.

    ``window_ms``: only each student's most recent ``window_ms`` (at least
    ``min_items`` samples) is searched — actions older than that were reported
    in earlier rounds. None / 0 searches the whole ring (2.2.0 behaviour).
    Requires monotonic frame_idx (live capture) when used with a cache.
    """
    pose_ring = ring_items(pose_ring)
    ball_ring = ring_items(ball_ring)
    ids = student_ids or list_student_ids(pose_ring)
    if not ids:
        ids = [None]  # type: ignore[list-item]
    named = _student_frames(pose_ring, [sid for sid in ids if sid])  # keypoints converted after the window cut
    if cache is not None:
        cache.retain(ids)
    hoop_xy = latest_hoop_xy(ball_ring)
    out: list[dict[str, Any]] = []
    for sid in ids:
        seq, meta, ident = named[sid] if sid else sequences_from_pose_ring(pose_ring, sid)
        if window_ms and seq:
            a = _tail_start(meta, float(window_ms), min_items,
                            cache._anchor.get(sid) if cache is not None else None)
            seq, meta = seq[a:], meta[a:]
            if cache is not None and seq:
                cache._anchor[sid] = seq[0][0]
        if sid:
            seq = _materialize(seq)
        if len(seq) < min_seq:
            continue
        fmap = _frame_ms_map(meta)
        ball_map = ball_by_pose_frame(ball_ring, meta)
        if cache is not None:
            memo = cache.windows(sid)
            peaks, pose_clips = cache.lookup(sid, meta, ball_map, lambda: _run_detectors(seq, ball_map, memo))
        else:
            peaks, pose_clips = _run_detectors(seq, ball_map)
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
