"""Load live.yaml + cameras.yaml RTSP URLs."""

from __future__ import annotations

from typing import Any

from src.cameras.registry import get_camera, get_camera_ids, get_enrollment_camera
from src.config import load_yaml
from src.streaming.clock import CAMS, DEFAULT_ANCHOR, normalize_offsets


def load_live_config() -> dict[str, Any]:
    try:
        doc = load_yaml("live.yaml") or {}
    except FileNotFoundError:
        doc = {}
    return doc


def websocket_bind(cfg: dict[str, Any] | None = None) -> tuple[str, int]:
    cfg = cfg or load_live_config()
    ws = cfg.get("websocket") or {}
    host = str(ws.get("host") or "127.0.0.1")
    port = int(ws.get("port") or 8765)
    return host, port


def rtsp_urls(cfg: dict[str, Any] | None = None) -> dict[str, str]:
    """Per-cam URL: live.yaml rtsp.* overrides cameras.yaml rtsp_url."""
    cfg = cfg or load_live_config()
    live_map = dict(cfg.get("rtsp") or {})
    out: dict[str, str] = {}
    for cam in get_camera_ids() or list(CAMS):
        url = live_map.get(cam)
        if not url:
            url = get_camera(cam).get("rtsp_url")
        out[cam] = str(url or "")
    return out


def default_offsets(cfg: dict[str, Any] | None = None) -> dict[str, float]:
    cfg = cfg or load_live_config()
    clock = cfg.get("clock") or {}
    anchor = str(clock.get("anchor_camera") or DEFAULT_ANCHOR)
    return normalize_offsets(clock.get("camera_time_offsets_ms"), anchor_camera=anchor)


def enroll_camera(cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or load_live_config()
    ident = cfg.get("identity") or {}
    return str(ident.get("enroll_camera") or get_enrollment_camera())
