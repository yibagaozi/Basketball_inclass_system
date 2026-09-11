"""Live clock: wall-clock local_ms ↔ common (anchor) timeline.

Convention (same as offline group sync / event_sync):
  common_ms = local_ms - offset_ms
  local_ms  = common_ms + offset_ms
  Anchor camera offset is always 0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.cameras.event_sync import apply_offset, invert_offset

CAMS = ("cam_01", "cam_02", "cam_03", "cam_04")
DEFAULT_ANCHOR = "cam_03"


def normalize_offsets(
    offsets: dict[str, Any] | None,
    *,
    anchor_camera: str = DEFAULT_ANCHOR,
) -> dict[str, float]:
    """Fill cam_01–04; force anchor to 0."""
    raw = dict(offsets or {})
    out = {c: float(raw.get(c, 0.0) or 0.0) for c in CAMS}
    out[str(anchor_camera)] = 0.0
    return out


def common_ms(local_ms: float, offset_ms: float) -> float:
    return apply_offset(local_ms, offset_ms)


def local_ms(common: float, offset_ms: float) -> float:
    return invert_offset(common, offset_ms)


def load_sync_doc(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_sync_doc(path: Path, doc: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anchor = str(doc.get("anchor_camera") or DEFAULT_ANCHOR)
    offs = normalize_offsets(doc.get("camera_time_offsets_ms"), anchor_camera=anchor)
    payload = {
        "anchor_camera": anchor,
        "camera_time_offsets_ms": offs,
        "offset_convention": (
            "common_ms = local_ms - offset_ms; "
            "local_ms = common_ms + offset_ms; "
            "anchor offset is 0"
        ),
    }
    for k, v in doc.items():
        if k not in payload:
            payload[k] = v
    payload["anchor_camera"] = anchor
    payload["camera_time_offsets_ms"] = offs
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def is_compatible_sync_doc(doc: dict[str, Any]) -> bool:
    offs = doc.get("camera_time_offsets_ms")
    if not isinstance(offs, dict):
        return False
    if str(doc.get("anchor_camera") or "") != DEFAULT_ANCHOR:
        return False
    if abs(float(offs.get(DEFAULT_ANCHOR, 1e9))) > 1e-6:
        return False
    return all(c in offs for c in CAMS)
