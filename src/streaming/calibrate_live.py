"""Grab live stills and run existing court annotate + solve (new dir, not v2/v3)."""

from __future__ import annotations

from pathlib import Path

import cv2

from src.acquisition.rtsp import grab_one_frame
from src.calibration.annotate import annotate_camera_gui
from src.calibration.solve import export_calibration, solve_all_cameras
from src.streaming.clock import CAMS
from src.streaming.display import require_display


def grab_live_calibration_frames(
    urls: dict[str, str],
    frames_dir: Path,
    *,
    include_cam04: bool = True,
) -> dict[str, Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    cams = CAMS if include_cam04 else ("cam_01", "cam_02", "cam_03")
    for cam in cams:
        url = urls.get(cam) or ""
        fr = grab_one_frame(url)
        if fr is None:
            print(f"  skip {cam}: 无法从 {url} 抽帧", flush=True)
            continue
        dst = frames_dir / f"{cam}.jpg"
        cv2.imwrite(str(dst), fr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        written[cam] = dst
        print(f"  {cam} → {dst}", flush=True)
    return written


def annotate_and_solve(
    frames_dir: Path,
    ann_path: Path,
    out_dir: Path,
) -> Path:
    require_display("球场标定标注 GUI")
    for cam in ("cam_01", "cam_02", "cam_03"):
        img = frames_dir / f"{cam}.jpg"
        if not img.exists():
            print(f"skip annotate {cam}: missing {img}", flush=True)
            continue
        annotate_camera_gui(img, camera_id=cam, annotations_path=ann_path)
    from src.calibration.annotate import load_annotations

    doc = load_annotations(ann_path)
    solved = solve_all_cameras(doc)
    return export_calibration(solved, out_dir, annotations=doc)
