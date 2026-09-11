"""Live sequential enrollment from RTSP preview (no mp4)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2

from src.acquisition.rtsp import open_video_source
from src.identity.enrollment import EnrollmentGallery
from src.identity.sequential_enroll import enroll_sequential_from_frames
from src.streaming.display import has_display


def collect_enroll_frames(
    url: str,
    *,
    seconds: float = 45.0,
    sample_hz: float = 8.0,
    preview: bool = True,
    window_title: str = "live enroll — 正面顺序走过镜头，Q 结束",
) -> list[tuple[int, float, Any]]:
    cap = open_video_source(url)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开注册机位：{url}")
    show = preview and has_display()
    if preview and not show:
        print("无 DISPLAY：以无预览方式采集注册帧。", flush=True)
    frames: list[tuple[int, float, Any]] = []
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
                frames.append((idx, elapsed, fr.copy()))
                last_keep = elapsed
            idx += 1
            if show:
                vis = fr.copy()
                cv2.putText(
                    vis, f"t={elapsed:.1f}s  n={len(frames)}  Q=quit",
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
    frames = collect_enroll_frames(
        url, seconds=seconds, sample_hz=sample_hz, preview=preview,
    )
    if not frames:
        print("  [live-enroll] 未采到帧", flush=True)
        return []
    ids = enroll_sequential_from_frames(
        session_id,
        frames,
        expected_persons=expected_persons,
        preview_dir=preview_dir,
    )
    gallery = EnrollmentGallery(session_id)
    print(f"  [live-enroll] students={ids} gallery={gallery.root}", flush=True)
    return ids
