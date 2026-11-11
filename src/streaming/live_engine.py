"""Always-on 4-cam live engine: RTSP → rings → finalize → one WS JSON.

Does not write sessions/.../raw/*.mp4. Does not build a dashboard.
"""

from __future__ import annotations

import threading
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from src.acquisition.rtsp import LiveFrame, MultiRtspCapture
from src.identity.enrollment import EnrollmentGallery
from src.streaming.clock import CAMS, capture_now_ms, common_ms, load_sync_doc, normalize_offsets
from src.streaming.events import build_action_event
from src.streaming.fast_path import TimestampRingBuffer
from src.streaming.live_action import ActionDetectCache, detect_live_actions, is_new_action
from src.streaming.live_angles import angles_for_window
from src.streaming.live_config import load_live_config
from src.streaming.live_guard import Watchdog, install_async_stdio, stdio_stats
from src.streaming.live_metrics import LiveMetrics, redact_url
from src.streaming.live_perception import LivePerception
from src.streaming.ws_hub import WsHub

_REPO = Path(__file__).resolve().parents[2]


def _version() -> str | None:
    try:
        return (_REPO / "VERSION").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


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
        # Opt-in (default off = 2.2.0 behaviour):
        # finalize.window_by_ms: release actions span pose_pre_ms / pose_post_ms in time
        self.window_pre_ms = self.window_post_ms = None
        if fin.get("window_by_ms"):
            self.window_pre_ms = float(fin.get("pose_pre_ms") or 1800.0)
            self.window_post_ms = float(fin.get("pose_post_ms") or 900.0)
        # inference.min_interval_ms: {cam: ms} or one number — cap a camera's rate
        # (e.g. let cam_03, which drives action detection, get more GPU/GIL time)
        inf_cfg = self.cfg.get("inference") or {}
        mi = inf_cfg.get("min_interval_ms") or {}
        self.min_interval_s = {
            c: float((mi.get(c, 0) if isinstance(mi, dict) else mi) or 0) / 1000.0 for c in CAMS
        }
        capacity = float(ring_cfg.get("capacity_ms") or 60_000.0)
        self.rings = {c: TimestampRingBuffer(capacity_ms=capacity) for c in CAMS}
        self.clock = str(cap_cfg.get("clock") or "wall")
        self.hub = hub
        # observability: [cv-metrics] / [cv-event] stdout lines + metrics.jsonl
        # (docs/实时推理观测与精度评估.md). observability.enabled: false turns all of it off.
        obs = self.cfg.get("observability") or {}
        obs_dir = obs.get("dir")
        if obs_dir:
            obs_dir = Path(obs_dir)
        elif getattr(hub, "jsonl_path", None):
            obs_dir = Path(hub.jsonl_path).parent  # next to events.jsonl
        else:
            import src.config as _config

            obs_dir = Path(_config.DATA) / "outputs" / "live" / session_id
        self.metrics = LiveMetrics(
            session_id, obs_dir,
            enabled=bool(obs.get("enabled", True)),
            interval_s=float(obs.get("interval_s") or 10.0),
            stdout=bool(obs.get("stdout", True)),
            jsonl=bool(obs.get("jsonl", True)),
            max_events_per_interval=int(obs.get("max_events_per_interval") or 200),
        )
        self.metrics.add_gauges(self._gauges)
        # Long-running safety (docs/实时推理长时间运行排查.md): non-blocking stdout, watchdog
        # with stack dumps + exit-for-restart on a stall, Windows keep-awake. watchdog.enabled: false = off.
        self._async_stdout = bool(obs.get("async_stdout", True))
        wd = self.cfg.get("watchdog") or {}
        self.watchdog = Watchdog(
            obs_dir,
            emit=self.metrics.event,
            enabled=bool(wd.get("enabled", True)),
            stall_s=float(wd.get("stall_s") or 60.0),
            exit_after_s=float(wd.get("exit_after_s", 180.0) or 0.0),
            prevent_sleep=bool(wd.get("prevent_sleep", True)),
        )
        self._errors: Counter = Counter()
        self._error_logged: Counter = Counter()
        self._run_tag = format(int(time.time()), "x")  # action_id prefix, unique per CV run
        self._action_seq = 0
        self._ws_seen: tuple[int, bool] | None = None
        self._first_frame: set[str] = set()
        self._cache_seen = (0, 0)
        gid_session = gallery_session or session_id
        gallery = EnrollmentGallery(gid_session)
        global_map: dict[str, str] = {}
        for sid in gallery.list_students():
            gid = (gallery.load_student(sid).get("meta") or {}).get("global_id")
            if gid:
                global_map[sid] = str(gid)
        self._gallery_session = gid_session
        self._gallery_size = len(gallery.list_students())
        ball_cfg = self.cfg.get("ball") or {}
        self.perception = LivePerception(
            gid_session, self.rings, gallery=gallery, global_map=global_map,
            hoop_lock=str(ball_cfg.get("hoop_lock") or "first"),
            hoop_lock_samples=int(ball_cfg.get("hoop_lock_samples") or 15),
            metrics=self.metrics,
            # opt-ins, default off: see docs/实时推理观测与精度评估.md
            feature_interval=int(inf_cfg.get("feature_interval") or 1),
            rtmw_cameras=inf_cfg.get("rtmw_cameras"),
        )
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
            self.metrics.event(
                "timeline_gap", force=True, cameras=ev.get("cameras"), reason=ev.get("reason"),
                start_ms=ev.get("start_ms"), end_ms=ev.get("end_ms"),
            )

        def on_frame(fr: LiveFrame) -> None:
            """Capture-thread hot path: keep only the newest frame, never block."""
            if self._stop.is_set():
                return
            self.metrics.captured(fr.camera_id)
            if fr.camera_id not in self._first_frame:
                # Anchor for offline evaluation: event times (common_ms) ↔ source time.
                self._first_frame.add(fr.camera_id)
                self.metrics.event(
                    "capture_start", force=True, camera=fr.camera_id, source=fr.source,
                    frame_idx=fr.frame_idx, common_ms=fr.common_ms, wall_ms=fr.wall_ms,
                    offset_ms=self.offsets_ms.get(fr.camera_id), clock=self.clock,
                    width=int(fr.bgr.shape[1]), height=int(fr.bgr.shape[0]),
                )
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
            # capture.clock: monotonic (opt-in) — timestamps immune to system clock steps
            clock=self.clock,
        )

    def _gauges(self) -> list[dict[str, Any]]:
        """Evaluated at each [cv-metrics] emit."""
        out = [
            {"name": "cv.live.ring.items", "value": len(self.rings[c]), "unit": "count", "dims": {"camera": c}}
            for c in CAMS
        ]
        hub = self.hub
        if hasattr(hub, "connects"):  # EdgeWsClient
            out += [
                {"name": "cv.live.ws.connected", "value": int(getattr(hub, "_ws", None) is not None),
                 "unit": "bool", "dims": {}},
                {"name": "cv.live.ws.published.total", "value": int(getattr(hub, "_seq", 0)), "unit": "count",
                 "dims": {}},
                {"name": "cv.live.ws.dropped.total", "value": int(hub.dropped), "unit": "count", "dims": {}},
                {"name": "cv.live.ws.send_failed.total", "value": int(hub.send_failed), "unit": "count",
                 "dims": {}},
                {"name": "cv.live.ws.jsonl_failed.total", "value": int(getattr(hub, "jsonl_failed", 0)),
                 "unit": "count", "dims": {}},
            ]
        for kind, n in sorted(self._errors.items()):
            out.append({"name": "cv.live.errors.total", "value": n, "unit": "count", "dims": {"kind": kind}})
        for key, n in sorted(stdio_stats().items()):
            out.append({"name": "cv.live.stdio.total", "value": n, "unit": "count", "dims": {"kind": key}})
        return out

    def _log_error(self, kind: str, exc: BaseException, **fields: Any) -> None:
        """Count + throttled log (full traceback for the first 3, then every 100th) + event."""
        self._errors[kind] += 1
        n = self._errors[kind]
        try:
            if n <= 3 or n % 100 == 0:
                tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                print(f"  [live] {kind} 第 {n} 次: {type(exc).__name__}: {exc}\n{tb}", flush=True)
            self.metrics.event(kind, error=f"{type(exc).__name__}: {exc}", count=n, **fields)
        except Exception:
            pass

    def _poll_ws(self) -> None:
        hub = self.hub
        if not hasattr(hub, "connects"):
            return
        state = (int(hub.connects), getattr(hub, "_ws", None) is not None)
        if state == self._ws_seen:
            return
        prev, self._ws_seen = self._ws_seen, state
        if state[1] and (prev is None or state[0] != prev[0] or not prev[1]):
            self.metrics.event("ws_connected", force=True, connects=state[0], url=redact_url(str(hub.url)))
        elif not state[1] and prev is not None and prev[1]:
            self.metrics.event("ws_disconnected", force=True, connects=state[0],
                               dropped=int(hub.dropped), send_failed=int(hub.send_failed))

    def _infer_loop(self, camera_id: str) -> None:
        """One worker per camera: always take the newest frame, skip the backlog."""
        cond = self._frame_ready[camera_id]
        interval = self.min_interval_s.get(camera_id, 0.0)
        next_ok = 0.0
        while not self._stop.is_set():
            if interval:
                wait = next_ok - time.monotonic()
                if wait > 0 and self._stop.wait(wait):
                    break
                next_ok = time.monotonic() + interval
            with cond:
                if self._latest[camera_id] is None:
                    cond.wait(timeout=0.2)
                fr = self._latest[camera_id]
                self._latest[camera_id] = None
            if fr is None:
                continue
            self.watchdog.busy(f"infer:{camera_id}")
            try:
                self.perception.process(fr)
                self._processed[camera_id] += 1
                self.metrics.processed(camera_id, capture_now_ms(self.clock) - fr.wall_ms)
            except Exception as exc:
                try:
                    print(f"  [live] {camera_id} perception: {exc}", flush=True)
                    self.metrics.event("perception_error", camera=camera_id, error=f"{type(exc).__name__}: {exc}")
                except Exception:
                    pass
            finally:
                self.watchdog.idle(f"infer:{camera_id}")

    def _finalize_loop(self) -> None:
        """Detection + emission thread. Any exception is logged and the loop goes on:
        before, one failing write (e.g. events.jsonl on a full disk) ended this thread
        silently and no action or [live] line came out for the rest of the class."""
        while not self._stop.is_set():
            # A slow round must not hold the GIL back to back with the next one:
            # wait at least as long as the last round took (capped at 2 s).
            time.sleep(max(0.35, min(self._last_detect_s, 2.0)))
            self.watchdog.beat("finalize")
            try:
                self._finalize_once()
            except Exception as exc:
                self._log_error("finalize_error", exc)
                time.sleep(1.0)

    def _finalize_once(self) -> None:
        pose_ring = self.rings["cam_03"]
        ball_ring = self.rings["cam_04"]
        self._poll_ws()
        self.metrics.maybe_emit()
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
            return
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
                pre_ms=self.window_pre_ms,
                post_ms=self.window_post_ms,
            )
        except Exception as exc:
            self._log_error("detect_error", exc)
            return
        finally:
            self._last_detect_s = time.perf_counter() - t0
            hits, misses = self._detect_cache.hits, self._detect_cache.misses
            self.metrics.detect_pass(
                self._last_detect_s,
                hits=hits - self._cache_seen[0], misses=misses - self._cache_seen[1],
            )
            self._cache_seen = (hits, misses)
        now_ms = pose_items[-1].timestamp_ms if pose_items else None
        for cand in cands:
            try:
                self._emit_candidate(cand, now_ms)
            except Exception as exc:
                # one bad candidate must not cost the others (or the thread)
                self._log_error("emit_error", exc, action_type=cand.get("action_type"),
                                student_id=cand.get("student_id"))

    def _emit_candidate(self, cand: dict[str, Any], now_ms: float | None) -> None:
        end_ms = cand.get("end_ms")
        if now_ms is not None and end_ms is not None:
            if now_ms < float(end_ms) + self.post_delay_ms:
                return
        if not is_new_action(cand, self._emitted, min_gap_ms=self.min_repeat_gap_ms):
            return
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
        self._action_seq += 1
        action_id = f"{self._run_tag}-{self._action_seq:05d}"
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
            action_id=action_id,
        )
        self._emitted.append(cand)
        if len(self._emitted) > 2000:  # dedupe only looks min_repeat_gap_ms back; bound a full day
            del self._emitted[:1000]
        self.hub.publish(msg)
        self._note_action(action_id, msg, cand)

    def _note_action(self, action_id: str, msg: dict[str, Any], cand: dict[str, Any]) -> None:
        """[cv-event] action_emitted: the per-action anchor (join key: action_id)."""
        if not self.metrics.enabled:
            return
        end = msg.get("end_ms")
        latency = None
        if end is not None:
            now_c = common_ms(capture_now_ms(self.clock), self.offsets_ms.get("cam_03", 0.0))
            latency = now_c - float(end)
        angles = msg.get("angles") or []
        ident = msg.get("identity") or {}
        with_angles = sum(1 for r in angles if any(v is not None for k, v in r.items() if k != "t_ms"))
        self.metrics.action(
            msg.get("action_type"), latency, attributed=bool(msg.get("student_id")),
            made=msg.get("made"), has_angles=with_angles > 0,
        )
        self.metrics.event(
            "action_emitted", force=True, action_id=action_id,
            student_id=msg.get("student_id"), global_id=msg.get("global_id"),
            action_type=msg.get("action_type"),
            start_ms=msg.get("start_ms"), end_ms=end, release_ms=msg.get("release_ms"),
            made=msg.get("made"), confidence=msg.get("confidence"),
            identity_confidence=ident.get("confidence"),
            angle_rows=len(angles), angle_rows_valid=with_angles,
            emit_latency_ms=None if latency is None else round(latency, 1),
            detector=(cand.get("metadata") or {}).get("detector"),
        )

    def start(self) -> None:
        # Piped stdout (edge) must never block inference / finalize: see live_guard
        if self._async_stdout:
            install_async_stdio()
        # Load + warm up every model here, not on the first frames of 4 threads at once.
        # No usable person detector → fail fast (edge sees the CV process exit).
        self.metrics.event(
            "live_start", force=True, version=_version(),
            cameras={c: redact_url(u) for c, u in self._urls.items() if u},
            offsets_ms=self.offsets_ms, clock=self.clock,
            gallery_session=self._gallery_session, gallery_students=self._gallery_size,
            calib_dir=str(self.calib_dir) if self.calib_dir else None,
            detect_window_ms=self.detect_window_ms, post_delay_ms=self.post_delay_ms,
            feature_interval=self.perception._feat_every,
            rtmw_cameras=sorted(self.perception._rtmw_cams) if self.perception._rtmw_cams is not None else None,
            interval_s=self.metrics.interval_s,
            watchdog={"enabled": self.watchdog.enabled, "stall_s": self.watchdog.stall_s,
                      "exit_after_s": self.watchdog.exit_after_s, "prevent_sleep": self.watchdog.prevent_sleep},
            async_stdout=self._async_stdout,
        )
        t_load = time.monotonic()
        try:
            warnings = self.perception.preload(ball=bool(self._urls.get("cam_04")))
        except Exception as exc:
            print(f"  [live] 模型加载失败，直播推理未启动: {type(exc).__name__}: {exc}", flush=True)
            self.metrics.event("model_error", force=True, error=f"{type(exc).__name__}: {exc}")
            raise
        for w in warnings or []:
            self.metrics.event("model_warning", force=True, message=w)
        self.metrics.event("models_ready", force=True, load_s=round(time.monotonic() - t_load, 2),
                           warnings=len(warnings or []))
        self.metrics.restart_interval()
        for cam in CAMS:
            t = threading.Thread(
                target=self._infer_loop, args=(cam,),
                name=f"live-infer-{cam}", daemon=True,
            )
            t.start()
            self._infer_threads.append(t)
            self.watchdog.watch_thread(f"infer:{cam}", t)
        self.capture.start()
        self._fin_thread = threading.Thread(
            target=self._finalize_loop, name="live-finalize", daemon=True,
        )
        self._fin_thread.start()
        self.watchdog.watch_thread("finalize", self._fin_thread)
        self.watchdog.beat("finalize")
        self.watchdog.start()

    def stop(self) -> None:
        first = not self._stop.is_set()
        self.watchdog.stop()  # threads end below: not a stall
        self._stop.set()
        with self._latest_lock:
            for cond in self._frame_ready.values():
                cond.notify_all()
        self.capture.stop()
        if first:
            self.metrics.event(
                "live_stop", force=True, actions=self._action_seq,
                processed=dict(self._processed), dropped=dict(self._dropped),
            )
            self.metrics.close()


def load_offsets_file(path: Path) -> dict[str, float]:
    doc = load_sync_doc(path)
    anchor = str(doc.get("anchor_camera") or "cam_03")
    return normalize_offsets(doc.get("camera_time_offsets_ms"), anchor_camera=anchor)
