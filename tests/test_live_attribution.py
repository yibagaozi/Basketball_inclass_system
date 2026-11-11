"""A named student's action/angle data must never be filled with another person."""

from __future__ import annotations

import numpy as np

from src.streaming.fast_path import TimestampRingBuffer
from src.streaming.live_action import detect_live_actions, sequences_from_pose_ring
from src.streaming.live_angles import _person_for_student


def _skeleton(x: float, wrist_raise: float = 0.0) -> np.ndarray:
    k = np.zeros((133, 3), np.float32)
    k[:, 0] = x
    k[:, 2] = 0.9
    k[0, 1] = 220.0  # nose
    k[5, 1] = k[6, 1] = 250.0  # shoulders
    k[9, 1] = 300.0
    k[10, 1] = 280.0 - wrist_raise  # right wrist
    return k


def _ring(n: int = 91) -> TimestampRingBuffer:
    """stu_01 shoots (wrist peak mid-sequence, higher score); stu_00 stands still
    and is only visible in every 3rd frame (including the last one)."""
    ring = TimestampRingBuffer()
    for i in range(n):
        raise_px = 80.0 if 45 <= i <= 55 else 0.0
        persons = [{"student_id": "stu_01", "score": 50.0,
                    "keypoints": _skeleton(900.0, raise_px + i * 0.2).tolist()}]
        if i % 3 == 0:
            persons.append({"student_id": "stu_00", "score": 5.0,
                            "keypoints": _skeleton(300.0).tolist()})
        ring.push(1000.0 + i * 33.0, i + 1, {"persons": persons})
    return ring


def test_sequence_holds_only_the_named_student():
    ring = _ring()
    seq, meta, ident = sequences_from_pose_ring(ring, "stu_00")
    assert len(seq) == 31  # every 3rd of 91 frames
    assert all(float(k[0, 0]) == 300.0 for _, k in seq)  # stu_00's skeleton only
    assert ident["student_id"] == "stu_00"
    assert [f for f, _ in seq] == [f for f, _ in meta]


def test_unnamed_sequence_still_uses_best_person():
    seq, _, ident = sequences_from_pose_ring(_ring(), None)
    assert len(seq) == 91 and ident["student_id"] == "stu_01"  # highest score per frame


def test_shot_is_not_reported_under_a_bystander():
    ring = _ring()
    cands = detect_live_actions(ring, TimestampRingBuffer(), student_ids=["stu_00", "stu_01"], min_seq=20)
    # one shot, reported once, under the shooter (before: duplicated under the bystander)
    assert [c["student_id"] for c in cands] == ["stu_01"]


def test_angles_skip_views_without_the_student():
    payload = {"persons": [{"student_id": "stu_01", "score": 9.0}]}
    assert _person_for_student(payload, "stu_00") is None
    assert _person_for_student(payload, "stu_01")["student_id"] == "stu_01"
    assert _person_for_student(payload, None)["student_id"] == "stu_01"
