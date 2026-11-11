"""Live observability anchors: periodic metrics + discrete events.

Two stable line formats on stdout. edge logs every CV line as ``[cv] <line>``; an
edge that knows the prefixes turns them into telemetry metric / log events:

  [cv-metrics] {"schema": 1, "ts_ms": ..., "session_id": ..., "interval_s": 10.0,
                "metrics": [{"name": "cv.live.process.fps", "value": 7.9, "unit": "fps",
                             "dims": {"camera": "cam_03"}}, ...]}
  [cv-event]   {"schema": 1, "ts_ms": ..., "session_id": ..., "event": "action_emitted", ...}

The same objects go to ``data/outputs/live/<session>/metrics.jsonl`` (one per line,
with ``"kind": "metrics" | "event"``) for offline analysis and accuracy reports.
Metric names, units, dims and event fields are documented in docs/实时推理观测与精度评估.md;
treat them as an interface (add fields, do not rename).
"""

from __future__ import annotations

import gc
import json
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

METRICS_PREFIX = "[cv-metrics] "
EVENT_PREFIX = "[cv-event] "
SCHEMA = 1
_LATENCY_KEEP = 4000  # latency samples kept per camera per interval
_POSE_SOURCES = ("rtmw", "yolo_pose", "stub")


def redact_url(url: str) -> str:
    """rtsp://user:pass@host/x → rtsp://***@host/x (credentials never reach logs)."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "***"
    if parts.username or parts.password:
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        return urlunsplit((parts.scheme, f"***@{host}", parts.path, parts.query, parts.fragment))
    return url


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return float(s[k])


class _CamStats:
    __slots__ = ("captured", "processed", "frames", "persons", "detections", "identified", "conf", "pose",
                 "switches", "acquired", "lost", "dup_frames", "reused", "latency", "stage", "ball_frames",
                 "ball_seen", "sids")

    def __init__(self) -> None:
        self.captured = 0
        self.processed = 0
        self.frames = 0  # pose frames with stats
        self.persons = 0
        self.detections = 0
        self.identified = 0
        self.conf: Counter = Counter()
        self.pose: Counter = Counter()
        self.switches = 0
        self.acquired = 0
        self.lost = 0
        self.dup_frames = 0
        self.reused = 0
        self.latency: list[float] = []
        self.stage: dict[str, float] = defaultdict(float)
        self.ball_frames = 0
        self.ball_seen = 0
        self.sids: set[str] = set()


class LiveMetrics:
    """Thread-safe collector; ``maybe_emit()`` is called from the finalize loop."""

    def __init__(
        self,
        session_id: str,
        out_dir: Path | None = None,
        *,
        enabled: bool = True,
        interval_s: float = 10.0,
        stdout: bool = True,
        jsonl: bool = True,
        max_events_per_interval: int = 200,
        clock: Callable[[], float] | None = None,
    ):
        self.session_id = session_id
        self.enabled = bool(enabled)
        self.interval_s = max(1.0, float(interval_s))
        self.stdout = bool(stdout)
        self.jsonl_path = (Path(out_dir) / "metrics.jsonl") if (out_dir and jsonl) else None
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_events = int(max_events_per_interval)
        self._now_ms = clock or (lambda: time.time() * 1000.0)
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self.write_failed = 0
        self._jsonl_retry_at = 0.0
        self._cams: dict[str, _CamStats] = defaultdict(_CamStats)
        self._detect: list[float] = []
        self._cache_hits = 0
        self._cache_misses = 0
        self._actions: Counter = Counter()
        self._action_flags: Counter = Counter()
        self._action_latency: list[float] = []
        self._events_in_interval = 0
        self._suppressed = 0
        self._gauges: list[Callable[[], list[dict[str, Any]]]] = []
        self._gc_start: float | None = None
        self._gc_pauses: list[float] = []
        self._t_last = time.monotonic()
        if self.enabled:
            gc.callbacks.append(self._on_gc)

    # ---- output ----

    def _write(self, prefix: str, kind: str, obj: dict[str, Any]) -> None:
        """Never raises (callers are the finalize / capture / inference threads).

        A failing metrics.jsonl (e.g. full disk) is retried at most once a minute.
        """
        try:
            line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=_jsonable)
        except Exception:
            self.write_failed += 1
            return
        with self._io_lock:
            if self.stdout:
                try:
                    print(prefix + line, flush=True)
                except Exception:
                    self.write_failed += 1
            if self.jsonl_path is not None and time.monotonic() >= self._jsonl_retry_at:
                try:
                    with self.jsonl_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"kind": kind, **obj}, ensure_ascii=False, default=_jsonable) + "\n")
                except Exception as exc:
                    self.write_failed += 1
                    self._jsonl_retry_at = time.monotonic() + 60.0
                    try:
                        print(f"  [live] metrics.jsonl 写入失败(磁盘满?),60 s 后重试: {exc}", flush=True)
                    except Exception:
                        pass

    def event(self, name: str, *, force: bool = False, **fields: Any) -> None:
        """Discrete anchor. Rate-limited per interval (``force`` bypasses: start/stop/actions)."""
        if not self.enabled:
            return
        with self._lock:
            if not force and self._events_in_interval >= self.max_events:
                self._suppressed += 1
                return
            self._events_in_interval += 1
        self._write(EVENT_PREFIX, "event", {
            "schema": SCHEMA, "ts_ms": round(self._now_ms(), 1), "session_id": self.session_id,
            "event": name, **fields,
        })

    def add_gauges(self, fn: Callable[[], list[dict[str, Any]]]) -> None:
        """``fn`` returns metric dicts evaluated at each emit (ring sizes, ws state, ...)."""
        self._gauges.append(fn)

    # ---- recording (hot paths: one lock, a few additions) ----

    def captured(self, camera: str) -> None:
        if self.enabled:
            with self._lock:
                self._cams[camera].captured += 1

    def processed(self, camera: str, latency_ms: float | None) -> None:
        if not self.enabled:
            return
        with self._lock:
            st = self._cams[camera]
            st.processed += 1
            if latency_ms is not None and len(st.latency) < _LATENCY_KEEP:
                st.latency.append(float(latency_ms))

    def pose_frame(
        self,
        camera: str,
        *,
        persons: list[dict[str, Any]],
        detections: int,
        stage_ms: dict[str, float],
        switches: int = 0,
        acquired: int = 0,
        lost: int = 0,
        duplicates: int = 0,
        reused: int = 0,
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            st = self._cams[camera]
            st.frames += 1
            st.persons += len(persons)
            st.detections += int(detections)
            st.dup_frames += int(duplicates > 0)
            st.reused += int(reused)
            for p in persons:
                sid = p.get("student_id")
                if sid:
                    st.identified += 1
                    st.sids.add(str(sid))
                st.conf[str(p.get("identity_confidence") or "low") if sid else "none"] += 1
                src = p.get("pose_source")
                # failures carry reason strings (e.g. "no_rtmw|yolo_reject:...") → one bucket
                st.pose[src if src in _POSE_SOURCES else "none"] += 1
            st.switches += switches
            st.acquired += acquired
            st.lost += lost
            for k, v in stage_ms.items():
                st.stage[k] += float(v)

    def ball_frame(self, camera: str, *, ball_seen: bool) -> None:
        if self.enabled:
            with self._lock:
                st = self._cams[camera]
                st.ball_frames += 1
                st.ball_seen += int(bool(ball_seen))

    def detect_pass(self, seconds: float, *, hits: int = 0, misses: int = 0) -> None:
        if self.enabled:
            with self._lock:
                self._detect.append(float(seconds) * 1000.0)
                self._cache_hits += hits
                self._cache_misses += misses

    def action(
        self,
        action_type: str | None,
        emit_latency_ms: float | None,
        *,
        attributed: bool = True,
        made: bool | None = None,
        has_angles: bool = False,
    ) -> None:
        if self.enabled:
            with self._lock:
                self._actions[str(action_type or "unknown")] += 1
                self._action_flags["unattributed"] += int(not attributed)
                self._action_flags["with_angles"] += int(bool(has_angles))
                self._action_flags["made_" + {True: "true", False: "false", None: "null"}[made]] += 1
                if emit_latency_ms is not None:
                    self._action_latency.append(float(emit_latency_ms))

    def _on_gc(self, phase: str, info: dict[str, Any]) -> None:
        if info.get("generation") != 2:
            return
        if phase == "start":
            self._gc_start = time.perf_counter()
        elif self._gc_start is not None:
            self._gc_pauses.append((time.perf_counter() - self._gc_start) * 1000.0)
            self._gc_start = None

    def restart_interval(self) -> None:
        """Start the first interval now (after model loading), so per-second /
        per-minute rates do not include the load time."""
        self._t_last = time.monotonic()

    # ---- periodic emit ----

    def maybe_emit(self, *, force: bool = False) -> bool:
        if not self.enabled:
            return False
        now = time.monotonic()
        dt = now - self._t_last
        if not force and dt < self.interval_s:
            return False
        self._t_last = now
        with self._lock:
            cams, self._cams = self._cams, defaultdict(_CamStats)
            detect, self._detect = self._detect, []
            hits, misses = self._cache_hits, self._cache_misses
            self._cache_hits = self._cache_misses = 0
            actions, self._actions = self._actions, Counter()
            aflags, self._action_flags = self._action_flags, Counter()
            alat, self._action_latency = self._action_latency, []
            suppressed, self._suppressed = self._suppressed, 0
            self._events_in_interval = 0
        gcp, self._gc_pauses = self._gc_pauses, []
        dt = max(dt, 1e-6)
        m: list[dict[str, Any]] = []

        def add(name: str, value: float | None, unit: str, **dims: Any) -> None:
            if value is not None:
                m.append({"name": name, "value": round(float(value), 4), "unit": unit, "dims": dims})

        visible: set[str] = set()
        for cam in sorted(cams):
            st = cams[cam]
            add("cv.live.capture.fps", st.captured / dt, "fps", camera=cam)
            add("cv.live.process.fps", st.processed / dt, "fps", camera=cam)
            if st.captured:
                add("cv.live.drop.ratio", max(0.0, 1.0 - st.processed / st.captured), "ratio", camera=cam)
            add("cv.live.latency.p50.ms", _pct(st.latency, 0.50), "ms", camera=cam)
            add("cv.live.latency.p95.ms", _pct(st.latency, 0.95), "ms", camera=cam)
            if st.frames:
                add("cv.live.persons.avg", st.persons / st.frames, "count", camera=cam)
                add("cv.live.detections.avg", st.detections / st.frames, "count", camera=cam)
                if st.detections:
                    # inference.feature_interval: share of detections that reused face/colour
                    add("cv.live.features.reuse.ratio", st.reused / st.detections, "ratio", camera=cam)
                for stage, total in sorted(st.stage.items()):
                    add("cv.live.stage.ms", total / st.frames, "ms", camera=cam, stage=stage)
                add("cv.live.identity.switches", st.switches, "count", camera=cam)
                add("cv.live.identity.acquired", st.acquired, "count", camera=cam)
                add("cv.live.identity.lost", st.lost, "count", camera=cam)
                # same student on two tracks in one frame: a certain identity error
                add("cv.live.identity.duplicate.ratio", st.dup_frames / st.frames, "ratio", camera=cam)
            if st.persons:
                add("cv.live.identified.ratio", st.identified / st.persons, "ratio", camera=cam)
                for conf, n in sorted(st.conf.items()):
                    add("cv.live.identity.ratio", n / st.persons, "ratio", camera=cam, conf=conf)
                for src, n in sorted(st.pose.items()):
                    add("cv.live.pose.ratio", n / st.persons, "ratio", camera=cam, source=src)
            if st.ball_frames:
                add("cv.live.ball.seen.ratio", st.ball_seen / st.ball_frames, "ratio", camera=cam)
            visible |= st.sids
        add("cv.live.students.visible", len(visible), "count")
        if detect:
            add("cv.live.detect.pass.ms", sum(detect) / len(detect), "ms", stat="avg")
            add("cv.live.detect.pass.ms", max(detect), "ms", stat="max")
        if hits + misses:
            add("cv.live.detect.cache.hit.ratio", hits / (hits + misses), "ratio")
        for atype, n in sorted(actions.items()):
            add("cv.live.actions", n, "count", action_type=atype)
        n_act = sum(actions.values())
        if n_act:
            add("cv.live.actions.unattributed.ratio", aflags["unattributed"] / n_act, "ratio")
            add("cv.live.actions.angles.ratio", aflags["with_angles"] / n_act, "ratio")
            for res in ("true", "false", "null"):
                add("cv.live.actions.made.ratio", aflags["made_" + res] / n_act, "ratio", made=res)
        if alat:
            add("cv.live.action.emit_latency.ms", sum(alat) / len(alat), "ms", stat="avg")
            add("cv.live.action.emit_latency.ms", max(alat), "ms", stat="max")
        add("cv.live.gc.gen2.count", len(gcp), "count")
        if gcp:
            add("cv.live.gc.gen2.pause.ms", max(gcp), "ms", stat="max")
        if suppressed:
            add("cv.live.events.suppressed", suppressed, "count")
        if self.write_failed:
            add("cv.live.output.write_failed.total", self.write_failed, "count")
        try:
            import psutil  # ultralytics dependency; optional here

            add("cv.live.rss.mb", psutil.Process().memory_info().rss / 2**20, "MB")
        except Exception:
            pass
        for fn in self._gauges:
            try:
                m.extend(fn())
            except Exception:
                pass
        self._write(METRICS_PREFIX, "metrics", {
            "schema": SCHEMA, "ts_ms": round(self._now_ms(), 1), "session_id": self.session_id,
            "interval_s": round(dt, 3), "metrics": m,
        })
        return True

    def close(self) -> None:
        if not self.enabled:
            return
        self.maybe_emit(force=True)
        try:
            gc.callbacks.remove(self._on_gc)
        except ValueError:
            pass


def _jsonable(o: Any) -> Any:
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)
