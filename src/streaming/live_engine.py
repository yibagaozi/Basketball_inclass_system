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
from src.streaming.live_action import ActionDetectCache, detect_live_actions, is_new_action
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
        self._urls = dict(urls)
        self.cfg = cfg or load_live_config()
        cap_cfg = self.cfg.get("capture") or {}
        ring_cfg = self.cfg.get("ring") or {}
        fin = self.cfg.get("finalize") or {}
        self.offsets_ms = normalize_offsets(offsets_ms)
        self.calib_dir = Path(calib_dir) if calib_dir else None
        self.post_delay_ms = float(fin.get("post_delay_ms") or 400.0)
        self.min_repeat_gap_ms = float(fin.get("min_repeat_gap_ms") or 1600.0)
        # Search only each student's recent window (actions older than that were
        # reported in earlier rounds); finalize.detect_window_ms: 0 = whole ring (2.2.0).
        win = fin.get("detect_window_ms")
        self.detect_window_ms = 15_000.0 if win is None else float(win)
        self.detect_min_items = int(fin.get("detect_min_items") or 150)
        self._detect_cache = ActionDetectCache()
        self._last_detect_s = 0.0
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
        # Capture threads only hand off the newest frame; inference runs elsewhere.
        self._latest: dict[str, LiveFrame | None] = {c: None for c in CAMS}
        self._latest_lock = threading.Lock()
        # Per-camera wakeup (was: each idle worker polling every 3 ms)
        self._frame_ready = {c: threading.Condition(self._latest_lock) for c in CAMS}
        self._processed = {c: 0 for c in CAMS}
        self._dropped = {c: 0 for c in CAMS}
        self._infer_threads: list[threading.Thread] = []
        self._last_report = 0.0

        def on_gap(ev: dict[str, Any]) -> None:
            ev["session_id"] = self.session_id
            self.hub.publish(ev)

        def on_frame(fr: LiveFrame) -> None:
            """Capture-thread hot path: keep only the newest frame, never block."""
            if self._stop.is_set():
                return
            with self._latest_lock:
                if self._latest[fr.camera_id] is not None:
                    self._dropped[fr.camera_id] += 1
                self._latest[fr.camera_id] = fr
                cond = self._frame_ready.get(fr.camera_id)
                if cond is not None:
                    cond.notify()

        self.capture = MultiRtspCapture(
            urls,
            self.offsets_ms,
            session_id=session_id,
            gap_ms=float(cap_cfg.get("gap_ms") or 900.0),
            reconnect_sec=float(cap_cfg.get("reconnect_sec") or 1.5),
            ffmpeg_tcp=bool(cap_cfg.get("ffmpeg_tcp", True)),
            # Frames go to _latest via on_frame; nothing reads the per-camera queue
            queue_max=0,
            on_gap=on_gap,
            on_frame=on_frame,
        )

    def _infer_loop(self, camera_id: str) -> None:
        """One worker per camera: always take the newest frame, skip the backlog."""
        cond = self._frame_ready[camera_id]
        while not self._stop.is_set():
            with cond:
                if self._latest[camera_id] is None:
                    cond.wait(timeout=0.2)
                fr = self._latest[camera_id]
                self._latest[camera_id] = None
            if fr is None:
                continue
            try:
                self.perception.process(fr)
                self._processed[camera_id] += 1
            except Exception as exc:
                print(f"  [live] {camera_id} perception: {exc}", flush=True)

    def _finalize_loop(self) -> None:
        pose_ring = self.rings["cam_03"]
        ball_ring = self.rings["cam_04"]
        while not self._stop.is_set():
            # A slow round must not hold the GIL back to back with the next one:
            # wait at least as long as the last round took (capped at 2 s).
            time.sleep(max(0.35, min(self._last_detect_s, 2.0)))
            if time.time() - self._last_report >= 5.0:
                self._last_report = time.time()
                print(
                    "  [live] " + "  ".join(
                        f"{c} ring={len(self.rings[c])} ok={self._processed[c]}"
                        f" drop={self._dropped[c]}"
                        for c in CAMS
                    ),
                    flush=True,
                )
            if len(pose_ring) < 20:
                continue
            # Inference threads keep pushing: work on one snapshot per pass
            pose_items = pose_ring.snapshot()
            ball_items = ball_ring.snapshot()
            t0 = time.perf_counter()
            try:
                cands = detect_live_actions(
                    pose_items, ball_items,
                    cache=self._detect_cache,
                    window_ms=self.detect_window_ms or None,
                    min_items=self.detect_min_items,
                )
            except Exception as exc:
                print(f"  [live] detect: {exc}", flush=True)
                continue
            finally:
                self._last_detect_s = time.perf_counter() - t0
            now_ms = pose_items[-1].timestamp_ms if pose_items else None
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
        # Load + warm up every model here, not on the first frames of 4 threads at once.
        # No usable person detector → fail fast (edge sees the CV process exit).
        try:
            self.perception.preload(ball=bool(self._urls.get("cam_04")))
        except Exception as exc:
            print(f"  [live] 模型加载失败，直播推理未启动: {type(exc).__name__}: {exc}", flush=True)
            raise
        for cam in CAMS:
            t = threading.Thread(
                target=self._infer_loop, args=(cam,),
                name=f"live-infer-{cam}", daemon=True,
            )
            t.start()
            self._infer_threads.append(t)
        self.capture.start()
        self._fin_thread = threading.Thread(
            target=self._finalize_loop, name="live-finalize", daemon=True,
        )
        self._fin_thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._latest_lock:
            for cond in self._frame_ready.values():
                cond.notify_all()
        self.capture.stop()


def load_offsets_file(path: Path) -> dict[str, float]:
    doc = load_sync_doc(path)
    anchor = str(doc.get("anchor_camera") or "cam_03")
    return normalize_offsets(doc.get("camera_time_offsets_ms"), anchor_camera=anchor)
