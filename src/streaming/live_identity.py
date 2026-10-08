"""Sequential enrollment from the RTSP preview (no mp4) or from a local video file."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import cv2

from src.acquisition.rtsp import open_video_source
from src.config import data_path
from src.identity.sequential_enroll import (
    StreamingFrontalScan,
    enroll_sequential_from_samples,
    live_scan_kwargs,
)
from src.streaming.display import has_display


def collect_enroll_frames(
    url: str,
    *,
    seconds: float = 45.0,
    sample_hz: float = 8.0,
    preview: bool = True,
    window_title: str = "live enroll — 正面顺序走过镜头，Q 结束",
    on_frame: Callable[[int, float, Any], None] | None = None,
) -> list[tuple[int, float, Any]]:
    """Sample frames from the enrollment camera for ``seconds``.

    With ``on_frame``, each sampled frame is handed over as it arrives and not
    accumulated (the returned list stays empty).
    """
    cap = open_video_source(url)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开注册机位：{url}")
    show = preview and has_display()
    if preview and not show:
        print("无 DISPLAY：以无预览方式采集注册帧。", flush=True)
    frames: list[tuple[int, float, Any]] = []
    n_kept = 0
    t0 = time.time()
    last_keep = -1e9
    idx = 0
    interval = 1.0 / max(sample_hz, 0.5)
    try:
        while True:
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            now = time.time()
            elapsed = now - t0
            if elapsed - last_keep >= interval:
                item = (idx, elapsed, fr.copy())
                if on_frame is not None:
                    on_frame(*item)
                else:
                    frames.append(item)
                n_kept += 1
                last_keep = elapsed
            idx += 1
            if show:
                vis = fr.copy()
                cv2.putText(
                    vis, f"t={elapsed:.1f}s  n={n_kept}  Q=quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
                )
                cv2.imshow(window_title, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
            if elapsed >= seconds:
                break
    finally:
        cap.release()
        if show:
            cv2.destroyWindow(window_title)
    return frames


def collect_video_frames(
    path: str | Path,
    *,
    on_frame: Callable[[int, float, Any], None],
    sample_hz: float = 8.0,
    max_seconds: float | None = None,
    realtime: bool = False,
    preview: bool = False,
    window_title: str = "video enroll — Q 结束",
) -> int:
    """Sample a local video on its own timeline (not the wall clock).

    Same keep rule as the live sampler, with ``t = frame_index / fps``; frames in
    between are only grabbed. ``realtime`` paces reading at the native fps, like
    a camera. Returns the number of frames handed to ``on_frame``.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not 0.0 < fps < 1000.0:
        fps = 25.0
    show = preview and has_display()
    interval = 1.0 / max(sample_hz, 0.5)
    last_keep = -1e9
    idx = n_kept = 0
    wall0 = time.time()
    try:
        while True:
            t = idx / fps
            if max_seconds is not None and t >= max_seconds:
                break
            if not cap.grab():
                break
            if realtime:
                delay = wall0 + t - time.time()
                if delay > 0:
                    time.sleep(delay)
            if t - last_keep >= interval:
                ok, fr = cap.retrieve()
                if ok and fr is not None:
                    on_frame(idx, t, fr.copy())
                    n_kept += 1
                    last_keep = t
                    if show:
                        vis = fr.copy()
                        cv2.putText(
                            vis, f"t={t:.1f}s  n={n_kept}  Q=quit",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
                        )
                        cv2.imshow(window_title, vis)
                        if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
                            break
            idx += 1
    finally:
        cap.release()
        if show:
            cv2.destroyWindow(window_title)
    return n_kept


def enroll_live(
    session_id: str,
    url: str,
    *,
    seconds: float = 45.0,
    sample_hz: float = 8.0,
    preview: bool = True,
    expected_persons: int | None = None,
    preview_dir: Path | None = None,
) -> list[str]:
    # Frames are scored on a worker thread during capture (models load in the
    # background); only the backlog remains once capture stops.
    scan = StreamingFrontalScan(**live_scan_kwargs())
    try:
        collect_enroll_frames(
            url, seconds=seconds, sample_hz=sample_hz, preview=preview,
            on_frame=scan.submit,
        )
    except BaseException:
        scan.close()
        raise
    return _finish_enrollment(scan, session_id, expected_persons, preview_dir)


def enroll_from_video(
    session_id: str,
    path: str | Path,
    *,
    sample_hz: float = 8.0,
    max_seconds: float | None = None,
    realtime: bool = False,
    preview: bool = False,
    expected_persons: int | None = None,
    preview_dir: Path | None = None,
) -> list[str]:
    """Same enrollment as :func:`enroll_live`, fed from a local video file."""
    # Faster than real time, decoding must wait for the scorer rather than
    # queue the whole video in memory; in real-time mode it behaves like a camera.
    scan = StreamingFrontalScan(max_queue=0 if realtime else 32, **live_scan_kwargs())
    try:
        collect_video_frames(
            path, on_frame=scan.submit, sample_hz=sample_hz, max_seconds=max_seconds,
            realtime=realtime, preview=preview,
        )
    except BaseException:
        scan.close()
        raise
    return _finish_enrollment(scan, session_id, expected_persons, preview_dir)


def _finish_enrollment(
    scan: StreamingFrontalScan,
    session_id: str,
    expected_persons: int | None,
    preview_dir: Path | None,
) -> list[str]:
    t_capture = time.time()
    print(f"  [live-enroll] capture done: {scan.submitted} frames", flush=True)
    if not scan.submitted:
        scan.close()
        print("  [live-enroll] 未采到帧", flush=True)
        return []
    samples = scan.finish()
    print(
        f"  [live-enroll] scan done {time.time() - t_capture:.1f}s after capture",
        flush=True,
    )
    ids = enroll_sequential_from_samples(
        session_id,
        samples,
        expected_persons=expected_persons,
        preview_dir=preview_dir,
        face_embedder=scan.face_embedder,
        body_embedder=scan.body_embedder,
    )
    print(
        f"  [live-enroll] students={ids} gallery={data_path('enrollment', session_id)} "
        f"({time.time() - t_capture:.1f}s after capture)",
        flush=True,
    )
    return ids
