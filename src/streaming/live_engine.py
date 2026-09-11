"""Always-on 4-cam live engine: RTSP → rings → finalize → one WS JSON.

Does not write sessions/.../raw/*.mp4. Does not build a dashboard.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from src.acquisition.rtsp import LiveFrame, MultiRtspCapture
from src.identity.enrollment import EnrollmentGallery
from src.streaming.clock import CAMS, load_sync_doc, normalize_offsets
from src.streaming.events import build_action_event
from src.streaming.fast_path import TimestampRingBuffer
from src.streaming.live_action import detect_live_actions, is_new_action
from src.streaming.live_angles import angles_for_window
from src.streaming.live_config import load_live_config
from src.streaming.live_perception import LivePerception
from src.streaming.ws_hub import WsHub


class LiveEngine:
    def __init__(
        self,
        session_id: str,
        urls: dict[str, str],
        offsets_ms: dict[str, float],
        *,
        hub: WsHub,
        calib_dir: Path | None = None,
        gallery_session: str | None = None,
        cfg: dict[str, Any] | None = None,
    ):
        self.session_id = session_id
        self.cfg = cfg or load_live_config()
        cap_cfg = self.cfg.get("capture") or {}
        ring_cfg = self.cfg.get("ring") or {}
        fin = self.cfg.get("finalize") or {}
        self.offsets_ms = normalize_offsets(offsets_ms)
        self.calib_dir = Path(calib_dir) if calib_dir else None
        self.post_delay_ms = float(fin.get("post_delay_ms") or 400.0)
        self.min_repeat_gap_ms = float(fin.get("min_repeat_gap_ms") or 1600.0)
        capacity = float(ring_cfg.get("capacity_ms") or 60_000.0)
        self.rings = {c: TimestampRingBuffer(capacity_ms=capacity) for c in CAMS}
        gid_session = gallery_session or session_id
        gallery = EnrollmentGallery(gid_session)
        global_map: dict[str, str] = {}
        for sid in gallery.list_students():
            gid = (gallery.load_student(sid).get("meta") or {}).get("global_id")
            if gid:
                global_map[sid] = str(gid)
        self.perception = LivePerception(
            gid_session, self.rings, gallery=gallery, global_map=global_map,
        )
        self.hub = hub
        self._emitted: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()

        def on_gap(ev: dict[str, Any]) -> None:
            ev["session_id"] = self.session_id
            self.hub.publish(ev)

        def on_frame(fr: LiveFrame) -> None:
            if self._stop.is_set():
                return
            try:
                self.perception.process(fr)
            except Exception as exc:
                print(f"  [live] {fr.camera_id} perception: {exc}", flush=True)

        self.capture = MultiRtspCapture(
            urls,
            self.offsets_ms,
            session_id=session_id,
            gap_ms=float(cap_cfg.get("gap_ms") or 900.0),
            reconnect_sec=float(cap_cfg.get("reconnect_sec") or 1.5),
            ffmpeg_tcp=bool(cap_cfg.get("ffmpeg_tcp", True)),
            queue_max=int(cap_cfg.get("queue_max_frames") or 60),
            on_gap=on_gap,
            on_frame=on_frame,
        )

    def _finalize_loop(self) -> None:
        pose_ring = self.rings["cam_03"]
        ball_ring = self.rings["cam_04"]
        while not self._stop.is_set():
            time.sleep(0.35)
            if len(pose_ring) < 20:
                continue
            try:
                cands = detect_live_actions(pose_ring, ball_ring)
            except Exception as exc:
                print(f"  [live] detect: {exc}", flush=True)
                continue
            now_ms = None
            if pose_ring._items:
                now_ms = pose_ring._items[-1].timestamp_ms
            for cand in cands:
                end_ms = cand.get("end_ms")
                if now_ms is not None and end_ms is not None:
                    if now_ms < float(end_ms) + self.post_delay_ms:
                        continue
                if not is_new_action(cand, self._emitted, min_gap_ms=self.min_repeat_gap_ms):
                    continue
                start = cand.get("start_ms")
                end = cand.get("end_ms")
                angles: list[dict[str, Any]] = []
                if start is not None and end is not None:
                    try:
                        angles = angles_for_window(
                            self.rings,
                            float(start),
                            float(end),
                            student_id=cand.get("student_id"),
                            calib_dir=self.calib_dir,
                            shooting_hand=cand.get("shooting_hand"),
                        )
                    except Exception as exc:
                        print(f"  [live] angles: {exc}", flush=True)
                        angles = []
                msg = build_action_event(
                    session_id=self.session_id,
                    student_id=cand.get("student_id"),
                    action_type=cand.get("action_type"),
                    start_ms=start,
                    end_ms=end,
                    release_ms=cand.get("release_ms"),
                    made=cand.get("made"),
                    phases=cand.get("phases"),
                    angles=angles,
                    identity=cand.get("identity"),
                    global_id=cand.get("global_id"),
                    confidence=cand.get("confidence"),
                    extra={"source": "live_engine"},
                )
                self._emitted.append(cand)
                self.hub.publish(msg)

    def start(self) -> None:
        self.capture.start()
        self._fin_thread = threading.Thread(
            target=self._finalize_loop, name="live-finalize", daemon=True,
        )
        self._fin_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.capture.stop()


def load_offsets_file(path: Path) -> dict[str, float]:
    doc = load_sync_doc(path)
    anchor = str(doc.get("anchor_camera") or "cam_03")
    return normalize_offsets(doc.get("camera_time_offsets_ms"), anchor_camera=anchor)
