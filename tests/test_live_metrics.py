"""Observability anchors: [cv-metrics] / [cv-event] lines, metrics.jsonl, action_id."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from src.acquisition.rtsp import LiveFrame
from src.streaming.events import build_action_event
from src.streaming.live_metrics import EVENT_PREFIX, METRICS_PREFIX, LiveMetrics, redact_url


def _lines(capsys, prefix):
    out = capsys.readouterr().out.splitlines()
    return [json.loads(l[len(prefix):]) for l in out if l.startswith(prefix)]


def _by_name(batch, name, **dims):
    return [m["value"] for m in batch["metrics"]
            if m["name"] == name and all(m["dims"].get(k) == v for k, v in dims.items())]


def test_metrics_batch_and_jsonl(tmp_path, capsys):
    lm = LiveMetrics("s1", tmp_path, interval_s=10)
    for _ in range(10):
        lm.captured("cam_03")
    for lat in (40.0, 60.0, 200.0):
        lm.processed("cam_03", lat)
    persons = [
        {"student_id": "stu_01", "identity_confidence": "high", "pose_source": "rtmw133"},
        {"student_id": None, "identity_confidence": "low", "pose_source": "yolo17"},
    ]
    lm.pose_frame("cam_03", persons=persons, detections=3, stage_ms={"detect": 8.0, "pose": 12.0},
                  switches=1, duplicates=0)
    lm.pose_frame("cam_03", persons=persons + [persons[0]], detections=3, stage_ms={"detect": 10.0},
                  duplicates=1)
    lm.ball_frame("cam_04", ball_seen=True)
    lm.ball_frame("cam_04", ball_seen=False)
    lm.detect_pass(0.2, hits=3, misses=1)
    lm.action("jump_shot", 900.0, attributed=True, made=True, has_angles=True)
    lm.action("jump_shot", 1100.0, attributed=False, made=None)
    assert lm.maybe_emit() is False  # interval not reached
    assert lm.maybe_emit(force=True) is True
    (batch,) = _lines(capsys, METRICS_PREFIX)
    assert batch["schema"] == 1 and batch["session_id"] == "s1"
    assert _by_name(batch, "cv.live.drop.ratio", camera="cam_03") == [0.7]
    assert _by_name(batch, "cv.live.latency.p50.ms", camera="cam_03") == [60.0]
    assert _by_name(batch, "cv.live.persons.avg", camera="cam_03") == [2.5]
    # 3 of 5 person-frames carry a student id
    assert _by_name(batch, "cv.live.identified.ratio", camera="cam_03") == [0.6]
    assert _by_name(batch, "cv.live.identity.ratio", camera="cam_03", conf="none") == [0.4]
    assert _by_name(batch, "cv.live.identity.duplicate.ratio", camera="cam_03") == [0.5]
    assert _by_name(batch, "cv.live.identity.switches", camera="cam_03") == [1]
    assert _by_name(batch, "cv.live.stage.ms", camera="cam_03", stage="detect") == [9.0]
    # unknown / failure pose sources collapse into "none" (bounded dimension values)
    assert _by_name(batch, "cv.live.pose.ratio", camera="cam_03", source="none") == [1.0]
    assert _by_name(batch, "cv.live.ball.seen.ratio", camera="cam_04") == [0.5]
    assert _by_name(batch, "cv.live.students.visible") == [1]
    assert _by_name(batch, "cv.live.detect.cache.hit.ratio") == [0.75]
    assert _by_name(batch, "cv.live.actions", action_type="jump_shot") == [2]
    assert _by_name(batch, "cv.live.actions.unattributed.ratio") == [0.5]
    assert _by_name(batch, "cv.live.actions.made.ratio", made="null") == [0.5]
    assert _by_name(batch, "cv.live.action.emit_latency.ms", stat="max") == [1100.0]
    rows = [json.loads(l) for l in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["kind"] == "metrics" and rows[0]["metrics"] == batch["metrics"]
    # counters reset per interval
    lm.maybe_emit(force=True)
    (batch2,) = _lines(capsys, METRICS_PREFIX)
    assert not _by_name(batch2, "cv.live.process.fps", camera="cam_03")
    lm.close()


def test_events_rate_limit_and_force(tmp_path, capsys):
    lm = LiveMetrics("s1", tmp_path, max_events_per_interval=2)
    for i in range(5):
        lm.event("identity_switch", track_id=i)
    lm.event("action_emitted", force=True, action_id="x-1")
    events = _lines(capsys, EVENT_PREFIX)
    assert [e["event"] for e in events] == ["identity_switch", "identity_switch", "action_emitted"]
    lm.maybe_emit(force=True)
    (batch,) = _lines(capsys, METRICS_PREFIX)
    assert _by_name(batch, "cv.live.events.suppressed") == [3]
    lm.event("identity_switch", track_id=9)  # budget renewed
    assert len(_lines(capsys, EVENT_PREFIX)) == 1
    lm.close()


def test_disabled_is_silent(tmp_path, capsys):
    lm = LiveMetrics("s1", tmp_path, enabled=False)
    lm.captured("cam_01")
    lm.event("live_start", force=True)
    assert lm.maybe_emit(force=True) is False
    lm.close()
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "metrics.jsonl").exists()


def test_numpy_values_serialize(tmp_path, capsys):
    lm = LiveMetrics("s1", tmp_path, jsonl=False)
    lm.event("hoop_locked", force=True, center=np.array([1.5, 2.0]), conf=np.float32(0.5))
    (ev,) = _lines(capsys, EVENT_PREFIX)
    assert ev["center"] == [1.5, 2.0] and ev["conf"] == 0.5
    lm.close()


def test_redact_url():
    assert redact_url("rtsp://admin:pw@10.0.0.5:554/Streaming/101") == "rtsp://***@10.0.0.5:554/Streaming/101"
    assert redact_url("rtsp://10.0.0.5/x") == "rtsp://10.0.0.5/x"
    assert redact_url("C:/videos/cam_01.mp4") == "C:/videos/cam_01.mp4"


def test_action_id_optional():
    kw = dict(session_id="s", student_id="stu_01", action_type="jump_shot", start_ms=1.0, end_ms=2.0)
    assert "action_id" not in build_action_event(**kw)
    assert build_action_event(**kw, action_id="abc-00001")["action_id"] == "abc-00001"


class _Hub:
    def __init__(self):
        self.sent = []

    def publish(self, ev):
        self.sent.append(ev)


def test_identity_transitions(monkeypatch, tmp_path, capsys):
    import src.streaming.live_perception as lp

    monkeypatch.setattr("src.config.DATA", tmp_path)
    lm = LiveMetrics("s1", tmp_path)
    per = lp.LivePerception("s1", {}, metrics=lm)

    def tr(tid, sid):
        return SimpleNamespace(track_id=tid, student_id=sid, identity_confidence="medium")

    def frame(tracks):
        persons = [{"student_id": t.student_id} for t in tracks]
        per._note_frame("cam_01", tracks, persons, len(tracks), 1000.0, {"detect": 1.0})

    frame([tr(1, None), tr(2, "stu_02")])
    frame([tr(1, "stu_01"), tr(2, "stu_03")])  # 1 acquired, 2 switched
    frame([tr(1, None), tr(2, "stu_03"), tr(3, "stu_03")])  # 1 lost, duplicate stu_03
    events = _lines(capsys, EVENT_PREFIX)
    assert [(e["track_id"], e["from_student"], e["to_student"]) for e in events] == [(2, "stu_02", "stu_03")]
    lm.maybe_emit(force=True)
    (batch,) = _lines(capsys, METRICS_PREFIX)
    assert _by_name(batch, "cv.live.identity.acquired", camera="cam_01") == [1]
    assert _by_name(batch, "cv.live.identity.lost", camera="cam_01") == [1]
    assert _by_name(batch, "cv.live.identity.switches", camera="cam_01") == [1]
    assert _by_name(batch, "cv.live.identity.duplicate.ratio", camera="cam_01") == [round(1 / 3, 4)]
    lm.close()


def test_engine_anchors(monkeypatch, tmp_path, capsys):
    import src.streaming.live_engine as le

    monkeypatch.setattr("src.config.DATA", tmp_path)
    hub = _Hub()
    eng = le.LiveEngine("live_m", {"cam_01": "rtsp://u:p@127.0.0.1:9/a"}, {}, hub=hub, cfg={})
    assert eng.metrics.jsonl_path == tmp_path / "outputs" / "live" / "live_m" / "metrics.jsonl"
    # capture_start once per camera, frames counted
    for i in (1, 2):
        eng.capture.workers["cam_01"].on_frame(LiveFrame("cam_01", i, 5.0, 5.0, 5.0, np.zeros((4, 6, 3), np.uint8)))
    msg = build_action_event(session_id="live_m", student_id="stu_01", action_type="jump_shot",
                             start_ms=1.0, end_ms=2.0, made=True, action_id="r-00001",
                             angles=[{"t_ms": 1.5, "right_elbow": 90.0}])
    eng._note_action("r-00001", msg, {"metadata": {"detector": "live_release"}})
    eng.stop()  # live_stop + final metrics batch
    events = _lines(capsys, EVENT_PREFIX)
    names = [e["event"] for e in events]
    assert names == ["capture_start", "action_emitted", "live_stop"]
    cs, act = events[0], events[1]
    assert cs["camera"] == "cam_01" and cs["frame_idx"] == 1 and cs["width"] == 6
    assert act["action_id"] == "r-00001" and act["detector"] == "live_release"
    assert act["angle_rows_valid"] == 1 and act["emit_latency_ms"] > 0
    rows = (tmp_path / "outputs" / "live" / "live_m" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    kinds = [json.loads(r)["kind"] for r in rows]
    assert kinds.count("event") == 3 and kinds[-1] == "metrics"


def test_engine_observability_off(monkeypatch, tmp_path, capsys):
    import src.streaming.live_engine as le

    monkeypatch.setattr("src.config.DATA", tmp_path)
    eng = le.LiveEngine("live_n", {"cam_01": "rtsp://127.0.0.1:9/a"}, {}, hub=_Hub(),
                        cfg={"observability": {"enabled": False}})
    eng.capture.workers["cam_01"].on_frame(LiveFrame("cam_01", 1, 0, 0, 0, np.zeros((2, 2, 3), np.uint8)))
    eng._note_action("r-1", {"end_ms": 1.0}, {})
    eng.stop()
    out = capsys.readouterr().out
    assert METRICS_PREFIX not in out and EVENT_PREFIX not in out
    assert not (tmp_path / "outputs" / "live" / "live_n" / "metrics.jsonl").exists()


def test_ws_state_events(monkeypatch, tmp_path, capsys):
    import src.streaming.live_engine as le

    monkeypatch.setattr("src.config.DATA", tmp_path)

    class _Ws(_Hub):
        url = "ws://127.0.0.1:8081/internal/cv/stream"
        connects = 0
        dropped = 0
        send_failed = 0
        _ws = None
        _seq = 0

    hub = _Ws()
    eng = le.LiveEngine("live_ws", {"cam_01": "rtsp://127.0.0.1:9/a"}, {}, hub=hub, cfg={})
    eng._poll_ws()  # never connected: nothing
    hub.connects, hub._ws = 1, object()
    eng._poll_ws()
    eng._poll_ws()  # unchanged
    hub._ws = None
    eng._poll_ws()
    hub.connects, hub._ws = 2, object()
    eng._poll_ws()
    names = [e["event"] for e in _lines(capsys, EVENT_PREFIX)]
    assert names == ["ws_connected", "ws_disconnected", "ws_connected"]
    eng.metrics.maybe_emit(force=True)
    (batch,) = _lines(capsys, METRICS_PREFIX)
    assert _by_name(batch, "cv.live.ws.connected") == [1]
    assert _by_name(batch, "cv.live.ring.items", camera="cam_03") == [0]
    eng.metrics.close()


def test_models_ready_starts_first_interval(monkeypatch, tmp_path, capsys):
    import time

    import src.streaming.live_engine as le

    monkeypatch.setattr("src.config.DATA", tmp_path)
    eng = le.LiveEngine("live_s", {"cam_01": "rtsp://127.0.0.1:9/a"}, {}, hub=_Hub(), cfg={})

    def slow_preload(ball=True):
        time.sleep(0.3)
        return ["RTMW-133 未加载"]

    monkeypatch.setattr(eng.perception, "preload", slow_preload)
    monkeypatch.setattr(eng.capture, "start", lambda: None)
    eng.start()
    eng.stop()
    events = _lines(capsys, EVENT_PREFIX)
    names = [e["event"] for e in events]
    assert names[:3] == ["live_start", "model_warning", "models_ready"]
    assert events[2]["load_s"] >= 0.3 and events[2]["warnings"] == 1
    rows = [json.loads(r) for r in (tmp_path / "outputs" / "live" / "live_s" / "metrics.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    assert [r for r in rows if r["kind"] == "metrics"][-1]["interval_s"] < 0.3  # load time excluded
