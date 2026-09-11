"""Unit tests for v2.2.0 live RTSP + WebSocket path (no real cameras)."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cameras.event_sync import apply_offset, invert_offset
from src.pose.angles import ANGLE_KEYS, pack_angle_row
from src.streaming.clock import (
    CAMS,
    common_ms,
    is_compatible_sync_doc,
    local_ms,
    normalize_offsets,
    save_sync_doc,
)
from src.streaming.events import (
    EVENT_ACTION,
    EVENT_GAP,
    build_action_event,
    build_gap_event,
    phase_names_for,
)
from src.streaming.fast_path import TimestampRingBuffer
from src.streaming.live_action import detect_live_actions, infer_made, is_new_action
from src.streaming.live_angles import angles_for_window


def test_common_ms_offset_convention():
    # cam_01 local clock is 350ms ahead of anchor → offset +350
    assert apply_offset(5350, 350) == 5000
    assert invert_offset(5000, 350) == 5350
    assert common_ms(5350, 350) == 5000
    assert local_ms(5000, 350) == 5350
    offs = normalize_offsets({"cam_01": 120, "cam_02": -40, "cam_03": 99, "cam_04": 10})
    assert offs["cam_03"] == 0.0
    assert offs["cam_01"] == 120.0


def test_sync_json_compatible_schema(tmp_path: Path | None = None):
    d = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
    path = d / "sync.json"
    save_sync_doc(path, {
        "anchor_camera": "cam_03",
        "camera_time_offsets_ms": {"cam_01": 80.0, "cam_02": -15.0, "cam_03": 7.0, "cam_04": 40.0},
        "source": "test",
    })
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["anchor_camera"] == "cam_03"
    assert set(doc["camera_time_offsets_ms"]) >= set(CAMS)
    assert doc["camera_time_offsets_ms"]["cam_03"] == 0.0
    assert is_compatible_sync_doc(doc)


def test_timeline_gap_event():
    ev = build_gap_event(
        session_id="live_x",
        start_ms=1000.0,
        end_ms=2500.0,
        cameras=["cam_01", "cam_04"],
        reason="disconnect",
    )
    assert ev["event"] == EVENT_GAP
    assert ev["start_ms"] == 1000.0
    assert ev["end_ms"] == 2500.0
    assert ev["cameras"] == ["cam_01", "cam_04"]
    assert "student_id" not in ev or ev.get("student_id") is None


def test_action_event_nullable_fields():
    msg = build_action_event(
        session_id="s",
        student_id=None,
        action_type="pass",
        start_ms=10.0,
        end_ms=400.0,
        release_ms=200.0,
        made=True,
        phases=[{"name": "load", "start_ms": 10.0, "end_ms": None}],
        angles=None,
        global_id=None,
    )
    assert msg["event"] == EVENT_ACTION
    assert msg["student_id"] is None
    assert msg["global_id"] is None
    assert msg["release_ms"] is None  # pass family
    assert msg["made"] is None
    assert msg["angles"] == []
    for k in ("student_id", "release_ms", "made", "global_id"):
        assert k in msg
    shot = build_action_event(
        session_id="s",
        student_id="stu_00",
        action_type="jump_shot",
        start_ms=1.0,
        end_ms=2.0,
        release_ms=1.5,
        made=None,
        phases=[],
        angles=[{"t_ms": 1.0}],
        global_id="stu_global_03",
    )
    assert shot["release_ms"] == 1.5
    assert shot["made"] is None
    assert shot["global_id"] == "stu_global_03"
    row = shot["angles"][0]
    assert row["t_ms"] == 1.0
    for k in ANGLE_KEYS:
        assert k in row
        assert row[k] is None


def test_phase_names_locked():
    assert phase_names_for("free_throw") == ("load", "set", "release", "follow_through")
    assert phase_names_for("jump_shot") == ("load", "takeoff", "release", "follow_through")
    assert phase_names_for("layup") == ("approach", "gather", "takeoff", "release", "finish")
    assert phase_names_for("pass") == ("load", "action", "recover")
    assert phase_names_for("triple_threat") == ("load", "action", "recover")


def test_angle_schema_nan_to_null():
    row = pack_angle_row(12.0, {"right_elbow": 90.0, "left_knee": float("nan")})
    assert row["t_ms"] == 12.0
    assert row["right_elbow"] == 90.0
    assert row["left_knee"] is None
    assert row["shooting_wrist"] is None
    assert set(row) == {"t_ms", *ANGLE_KEYS}


def test_reconnect_keeps_startup_offsets():
    from src.acquisition.rtsp import RtspCameraWorker

    w = RtspCameraWorker("cam_01", "rtsp://127.0.0.1:9/x", offset_ms=275.0, session_id="s")
    assert w.offset_ms == 275.0
    w.stats.reconnects = 3
    assert w.offset_ms == 275.0  # never re-estimated


def test_gap_from_stall_logic():
    """Worker emits gap when wall delta exceeds gap_ms (unit, no decode)."""
    from src.acquisition.rtsp import RtspCameraWorker

    seen: list[dict] = []
    w = RtspCameraWorker(
        "cam_02", "file:///dev/null", 40.0, session_id="live_g",
        gap_ms=900.0, on_gap=seen.append,
    )
    w._last_ok_common = 1000.0
    w._emit_gap(1000.0, 2200.0, "disconnect")
    assert len(seen) == 1
    assert seen[0]["event"] == EVENT_GAP
    assert seen[0]["cameras"] == ["cam_02"]
    assert seen[0]["start_ms"] == 1000.0
    assert seen[0]["end_ms"] == 2200.0


def _kpts_raised_wrist(n: int = 80) -> list[tuple[int, np.ndarray, float]]:
    """Synthetic 133-kpt with a wrist peak around mid sequence."""
    out = []
    for i in range(n):
        k = np.zeros((133, 3), dtype=np.float32)
        k[:, 2] = 0.9
        k[0, 1] = 220
        k[5, 1] = 250
        k[6, 1] = 250
        # wrist y dips (raise) around i=50
        k[10, 1] = 280 - (80 if 45 <= i <= 55 else 0) - i * 0.2
        k[9, 1] = 300
        t = 1000.0 + i * 33.0
        out.append((i + 1, k, t))
    return out


def test_live_detect_from_rings_and_dedup():
    pose = TimestampRingBuffer(capacity_ms=60_000)
    ball = TimestampRingBuffer(capacity_ms=60_000)
    for fid, k, t in _kpts_raised_wrist():
        pose.push(t, fid, {
            "camera_id": "cam_03",
            "persons": [{
                "student_id": "stu_00",
                "global_id": "stu_global_01",
                "identity_confidence": "high",
                "score": 1.0,
                "keypoints": k.tolist(),
            }],
        })
        # hoop + ball above hoop near the peak
        cy = 400.0 if fid < 48 or fid > 58 else 120.0
        ball.push(t, fid, {
            "camera_id": "cam_04",
            "ball": {"center": [960.0, cy], "bbox": [940, cy - 10, 40, 40], "confidence": 0.8},
            "hoop": {"center": [960.0, 200.0], "bbox": [900, 160, 120, 80], "confidence": 0.9},
        })
    cands = detect_live_actions(pose, ball, student_ids=["stu_00"], min_seq=20)
    assert isinstance(cands, list)
    emitted: list[dict] = []
    for c in cands:
        if is_new_action(c, emitted, min_gap_ms=1600):
            emitted.append(c)
    # second pass must dedup
    for c in cands:
        assert is_new_action(c, emitted, min_gap_ms=1600) is False or c not in emitted


def test_angles_window_nulls_without_calib():
    pose = TimestampRingBuffer(capacity_ms=10_000)
    for i in range(5):
        t = 100.0 + i * 33.0
        k = np.zeros((133, 3), dtype=np.float32)
        k[:, 2] = 0.9
        pose.push(t, i, {
            "persons": [{"student_id": "stu_00", "keypoints": k.tolist(), "score": 1.0}],
        })
    rows = angles_for_window(
        {"cam_03": pose, "cam_01": pose, "cam_02": pose},
        100.0, 250.0,
        student_id="stu_00",
        calib_dir=Path("/tmp/no_such_calib_dir_live"),
    )
    assert rows
    for row in rows:
        assert "t_ms" in row
        for k in ANGLE_KEYS:
            assert k in row
            assert row[k] is None


def test_infer_made_null_without_ball():
    ring = TimestampRingBuffer()
    assert infer_made(ring, 1000.0) is None


def test_ws_hub_jsonl_and_optional_socket(tmp_path: Path | None = None):
    d = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
    jsonl = d / "events.jsonl"
    from src.streaming.ws_hub import WsHub, websockets

    hub = WsHub("127.0.0.1", 18765, jsonl_path=jsonl)
    msg = build_action_event(
        session_id="s", student_id=None, action_type="layup",
        start_ms=None, end_ms=None, release_ms=None, made=None,
    )
    if websockets is None:
        hub.publish(msg)
        lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
        assert json.loads(lines[0])["action_type"] == "layup"
        return
    try:
        hub.start_background()
        hub.publish(msg)
        time.sleep(0.15)
        lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
        parsed = json.loads(lines[0])
        assert parsed["event"] == EVENT_ACTION
        assert parsed["student_id"] is None
        assert parsed["start_ms"] is None
    finally:
        hub.stop()


def test_live_yaml_and_rtsp_placeholders():
    from src.streaming.live_config import rtsp_urls, websocket_bind

    host, port = websocket_bind()
    assert host == "127.0.0.1"
    assert port == 8765
    urls = rtsp_urls()
    for c in CAMS:
        assert c in urls
        assert "rtsp://" in urls[c]


def test_run_live_help_mentions_subcommands():
    import os
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_live_ws.py"), "-h"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 0
    out = r.stdout + r.stderr
    for word in ("sync", "calibrate", "enroll", "run"):
        assert word in out


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"ok  {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
            raise
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
