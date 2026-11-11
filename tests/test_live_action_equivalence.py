"""Faster live action detection must equal the frozen reference where it claims to."""

from __future__ import annotations

import numpy as np
import pytest

import tests._live_action_reference as ref
from src.action.pose_only import WindowMemo, detect_pose_only_segments
from src.streaming.fast_path import TimestampRingBuffer
from src.streaming.live_action import (
    ActionDetectCache,
    ball_by_pose_frame,
    detect_live_actions,
    sequences_by_student,
    sequences_from_pose_ring,
)

PERSONS, STUDENTS = 6, 9


def _payload(rng, base, t, as_list):
    persons = []
    for p in range(PERSONS):
        if rng.random() < 0.12:
            continue  # student briefly not detected
        k = np.zeros((133, 3), np.float64)
        k[:, :2] = base + [p * 60, 0] + rng.normal(0, 4, (133, 2))
        k[9:11, 1] -= 120 * max(0.0, np.sin(t / 900.0 + p))
        k[13:17, 0] += 40 * np.sin(t / 400.0 + p)  # some leg motion for pose-only windows
        k[:, 2] = 0.9
        persons.append({"student_id": f"stu_{(p + int(t // 9000)) % STUDENTS:02d}", "global_id": None,
                        "identity_confidence": "high", "score": float(rng.uniform(1, 20)),
                        "keypoints": k.tolist() if as_list else k})
    return {"persons": persons}


def _rings(seed, n, as_list, fps=12.0):
    rng = np.random.default_rng(seed)
    base = np.random.default_rng(99).uniform(0, 1, (133, 2)) * [120, 400] + [600, 300]
    pose, ball = TimestampRingBuffer(capacity_ms=30_000), TimestampRingBuffer(capacity_ms=30_000)
    state = {"i": 0}

    def push(k):
        for _ in range(k):
            i = state["i"]
            t = i * 1000.0 / fps
            pose.push(t, i + 1, _payload(rng, base, t, as_list))
            ball.push(t + 5, i + 1, {"ball": ({"center": [700 + 4 * (i % 40), 260.0 - 3 * (i % 25)],
                                               "bbox": [0, 0, 20, 20], "confidence": 0.8} if i % 3 else None),
                                      "hoop": {"center": [960.0, 150.0], "bbox": [930, 130, 990, 170], "confidence": 0.9}})
            state["i"] += 1

    push(n)
    return pose, ball, push


def test_full_ring_equals_reference_over_rounds():
    """window off: candidates identical to the reference, round after round, with the
    cache and window memo in use, as the ring fills and then slides."""
    pose_l, ball_l, push_l = _rings(1, 240, as_list=True)
    pose_a, ball_a, push_a = _rings(1, 240, as_list=False)
    cache = ActionDetectCache()
    total = 0
    for _ in range(6):
        want = ref.detect_live_actions(pose_l, ball_l)
        got = detect_live_actions(pose_a, ball_a, cache=cache)
        assert got == want
        assert detect_live_actions(pose_l, ball_l) == want  # list keypoints, no cache
        total += len(want)
        push_l(30)
        push_a(30)
    assert total > 0
    assert cache.hits + cache.misses > 0


def test_window_memo_is_exact():
    pose, ball, push = _rings(2, 300, as_list=False)
    memo = WindowMemo()
    for _ in range(5):
        for sid in ("stu_00", "stu_03"):
            seq, meta, _ = sequences_from_pose_ring(pose, sid)
            bmap = ball_by_pose_frame(ball, meta)
            plain = detect_pose_only_segments(seq, ball_by_frame=bmap)
            memod = detect_pose_only_segments(seq, ball_by_frame=bmap, window_memo=memo)
            assert [(c.action_type, c.start_frame, c.end_frame, c.confidence, c.metadata) for c in plain] == \
                   [(c.action_type, c.start_frame, c.end_frame, c.confidence, c.metadata) for c in memod]
        memo.end_round()
        push(25)
    assert memo.hits > 0


def test_sequences_by_student_equals_per_student_scan():
    pose, _, _ = _rings(3, 150, as_list=False)
    ids = [f"stu_{i:02d}" for i in range(STUDENTS)]
    allseq = sequences_by_student(pose, ids)
    for sid in ids:
        seq, meta, ident = sequences_from_pose_ring(pose, sid)
        s2, m2, i2 = allseq[sid]
        assert m2 == meta and i2 == ident
        assert all(f1 == f2 and np.array_equal(a, b) for (f1, a), (f2, b) in zip(seq, s2))


@pytest.mark.parametrize("dupes", [False, True])
def test_ball_matching_bisect_equals_linear_scan(dupes):
    rng = np.random.default_rng(5)
    ts = np.sort(rng.uniform(0, 5000, 300)).round(1)
    if dupes:
        ts[50:55] = ts[50]
    meta = [(i + 1, float(t)) for i, t in enumerate(ts)]
    ball = TimestampRingBuffer(capacity_ms=1e9)
    for j, t in enumerate(np.concatenate([rng.uniform(-100, 5100, 400), ts[::7], (ts[:-1] + ts[1:])[::5] / 2])):
        ball.push(float(t), j, {"ball": {"center": [float(j), 1.0], "bbox": None, "confidence": 0.5}})
    assert ball_by_pose_frame(ball, meta) == ref.ball_by_pose_frame(ball, meta)
    shuffled = meta[:]
    shuffled[10], shuffled[11] = shuffled[11], shuffled[10]  # clock jump → linear fallback
    assert ball_by_pose_frame(ball, shuffled) == ref.ball_by_pose_frame(ball, shuffled)


def _shot_ring():
    """stu_01 shoots twice, 20 s apart; plenty of idle frames in between."""
    ring, ball = TimestampRingBuffer(capacity_ms=60_000), TimestampRingBuffer(capacity_ms=60_000)
    for i in range(700):
        t = i * 40.0
        k = np.zeros((133, 3), np.float64)
        k[:, 0] = 900.0
        k[:, 2] = 0.9
        k[0, 1], k[5, 1], k[6, 1], k[9, 1] = 220.0, 250.0, 250.0, 300.0
        peak = (100 <= i % 500 <= 110)
        k[10, 1] = 280.0 - (80.0 if peak else 0.0) - (i % 500) * 0.05
        ring.push(t, i + 1, {"persons": [{"student_id": "stu_01", "score": 9.0, "keypoints": k}]})
    return ring, ball


def test_window_mode_reports_recent_shot_only():
    ring, ball = _shot_ring()  # shots near i=100 (4 s) and i=600 (24 s); ring ends at 28 s
    cache = ActionDetectCache()
    full = detect_live_actions(ring, ball, student_ids=["stu_01"], min_seq=20)
    win = detect_live_actions(ring, ball, student_ids=["stu_01"], min_seq=20, cache=cache, window_ms=15_000)
    rel = lambda cs: sorted(round(c["release_ms"]) for c in cs if c.get("release_ms") is not None)
    assert rel(full)[-1] == rel(win)[-1]  # the recent shot is found either way
    assert all(r >= 28_000 - 15_000 - 40 * 150 for r in rel(win))  # older one not re-searched


def test_ring_keypoints_are_never_modified():
    pose, ball, _ = _rings(6, 200, as_list=False)
    before = [p["keypoints"].copy() for it in pose.snapshot() for p in it.payload["persons"]]
    seqs = sequences_by_student(pose, ["stu_00", "stu_01"])
    for seq, _, _ in seqs.values():
        for _, k in seq:
            k += 1000.0  # a careless detector writing into its input
    detect_live_actions(pose, ball, cache=ActionDetectCache(), window_ms=10_000)
    after = [p["keypoints"] for it in pose.snapshot() for p in it.payload["persons"]]
    assert all(np.array_equal(a, b) for a, b in zip(before, after))
