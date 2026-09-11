"""Four-camera RTSP (or file) capture: wall-clock stamps, reconnect, gap events.

Live path does **not** write sessions/.../raw/*.mp4.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse

import cv2
import numpy as np

from src.streaming.clock import CAMS, common_ms as to_common
from src.streaming.events import build_gap_event

GapCallback = Callable[[dict[str, Any]], None]
FrameCallback = Callable[["LiveFrame"], None]


@dataclass
class LiveFrame:
    camera_id: str
    frame_idx: int
    wall_ms: float
    local_ms: float
    common_ms: float
    bgr: np.ndarray
    source: str = "rtsp"


@dataclass
class CaptureStats:
    frames: int = 0
    reconnects: int = 0
    gaps: int = 0
    last_common_ms: float | None = None
    last_error: str | None = None


def _is_rtsp(url: str) -> bool:
    return urlparse(url).scheme in {"rtsp", "rtsps"}


def open_video_source(url: str, *, ffmpeg_tcp: bool = True, timeout_sec: float = 8.0):
    """Open RTSP (FFmpeg) or a local file / mpegts URL. Does not record."""
    if not url:
        raise ValueError("empty video url")
    if _is_rtsp(url):
        if ffmpeg_tcp:
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        # Some builds honor CAP_PROP_OPEN_TIMEOUT_MSEC
        try:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout_sec * 1000))
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(timeout_sec * 1000))
        except Exception:
            pass
        return cap
    path = url[7:] if url.startswith("file://") else url
    return cv2.VideoCapture(path)


def grab_one_frame(url: str, *, tries: int = 12, ffmpeg_tcp: bool = True) -> np.ndarray | None:
    cap = open_video_source(url, ffmpeg_tcp=ffmpeg_tcp)
    if not cap.isOpened():
        cap.release()
        return None
    frame = None
    for _ in range(max(1, tries)):
        ok, fr = cap.read()
        if ok and fr is not None and fr.size:
            frame = fr
            break
    cap.release()
    return frame


class RtspCameraWorker:
    """One camera: decode every frame, stamp local/common ms, reconnect, emit gaps."""

    def __init__(
        self,
        camera_id: str,
        url: str,
        offset_ms: float,
        *,
        session_id: str = "",
        gap_ms: float = 900.0,
        reconnect_sec: float = 1.5,
        ffmpeg_tcp: bool = True,
        loop_file: bool = True,
        on_frame: FrameCallback | None = None,
        on_gap: GapCallback | None = None,
        queue: deque[LiveFrame] | None = None,
        queue_max: int = 60,
    ):
        self.camera_id = camera_id
        self.url = url
        self.offset_ms = float(offset_ms)
        self.session_id = session_id
        self.gap_ms = float(gap_ms)
        self.reconnect_sec = float(reconnect_sec)
        self.ffmpeg_tcp = ffmpeg_tcp
        self.loop_file = loop_file
        self.on_frame = on_frame
        self.on_gap = on_gap
        self.queue = queue if queue is not None else deque(maxlen=queue_max)
        self.queue_max = queue_max
        self.stats = CaptureStats()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_idx = 0
        self._last_ok_common: float | None = None
        self._last_ok_wall: float | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"rtsp-{self.camera_id}", daemon=True,
        )
        self._thread.start()

    def stop(self, join: bool = True) -> None:
        self._stop.set()
        if join and self._thread is not None:
            self._thread.join(timeout=4.0)

    def _emit_gap(self, start_ms: float, end_ms: float, reason: str) -> None:
        if end_ms <= start_ms:
            return
        self.stats.gaps += 1
        if self.on_gap is None:
            return
        self.on_gap(build_gap_event(
            session_id=self.session_id,
            start_ms=start_ms,
            end_ms=end_ms,
            cameras=[self.camera_id],
            reason=reason,
        ))

    def _push(self, item: LiveFrame) -> None:
        q = self.queue
        if self.queue_max > 0 and len(q) >= self.queue_max:
            try:
                q.popleft()
            except IndexError:
                pass
        q.append(item)
        if self.on_frame is not None:
            self.on_frame(item)

    def _run(self) -> None:
        while not self._stop.is_set():
            cap = None
            try:
                cap = open_video_source(self.url, ffmpeg_tcp=self.ffmpeg_tcp)
                if not cap.isOpened():
                    self.stats.last_error = f"open_failed:{self.url}"
                    time.sleep(self.reconnect_sec)
                    continue
                file_loop = self.loop_file and not _is_rtsp(self.url)
                while not self._stop.is_set():
                    ok, fr = cap.read()
                    wall = time.time() * 1000.0
                    if not ok or fr is None:
                        if self._last_ok_common is not None:
                            now_c = to_common(wall, self.offset_ms)
                            self._emit_gap(self._last_ok_common, now_c, "disconnect")
                        if file_loop:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        break
                    local = wall
                    common = to_common(local, self.offset_ms)
                    if (
                        self._last_ok_wall is not None
                        and (wall - self._last_ok_wall) > self.gap_ms
                    ):
                        self._emit_gap(float(self._last_ok_common or common), common, "stall")
                    self._last_ok_wall = wall
                    self._last_ok_common = common
                    self._frame_idx += 1
                    self.stats.frames += 1
                    self.stats.last_common_ms = common
                    self._push(LiveFrame(
                        camera_id=self.camera_id,
                        frame_idx=self._frame_idx,
                        wall_ms=wall,
                        local_ms=local,
                        common_ms=common,
                        bgr=fr,
                        source="rtsp" if _is_rtsp(self.url) else "file",
                    ))
            except Exception as exc:
                self.stats.last_error = str(exc)
            finally:
                if cap is not None:
                    cap.release()
            if self._stop.is_set():
                break
            self.stats.reconnects += 1
            time.sleep(self.reconnect_sec)


class MultiRtspCapture:
    """Start/stop four workers sharing startup offsets (never re-estimated)."""

    def __init__(
        self,
        urls: dict[str, str],
        offsets_ms: dict[str, float],
        *,
        session_id: str = "",
        gap_ms: float = 900.0,
        reconnect_sec: float = 1.5,
        ffmpeg_tcp: bool = True,
        queue_max: int = 60,
        on_gap: GapCallback | None = None,
        on_frame: FrameCallback | None = None,
    ):
        self.offsets_ms = dict(offsets_ms)
        self.workers: dict[str, RtspCameraWorker] = {}
        self.queues: dict[str, deque[LiveFrame]] = {}
        for cam in CAMS:
            url = urls.get(cam) or ""
            q: deque[LiveFrame] = deque(maxlen=queue_max)
            self.queues[cam] = q
            self.workers[cam] = RtspCameraWorker(
                cam, url, float(self.offsets_ms.get(cam, 0.0)),
                session_id=session_id,
                gap_ms=gap_ms,
                reconnect_sec=reconnect_sec,
                ffmpeg_tcp=ffmpeg_tcp,
                on_frame=on_frame,
                on_gap=on_gap,
                queue=q,
                queue_max=queue_max,
            )

    def start(self) -> None:
        for w in self.workers.values():
            w.start()

    def stop(self) -> None:
        for w in self.workers.values():
            w.stop(join=False)
        for w in self.workers.values():
            w.stop(join=True)
