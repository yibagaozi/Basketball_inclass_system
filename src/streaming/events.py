"""WebSocket JSON events for live 2.2.0 (action finalize + timeline_gap)."""

from __future__ import annotations

from typing import Any

from src.action.registry import POSE_ONLY_ACTION_TYPES, is_shooting_action, normalize_action_type
from src.pose.angles import ANGLE_KEYS, pack_angle_row

EVENT_ACTION = "action_finalized"
EVENT_GAP = "timeline_gap"

PHASE_NAMES = {
    "free_throw": ("load", "set", "release", "follow_through"),
    "jump_shot": ("load", "takeoff", "release", "follow_through"),
    "layup": ("approach", "gather", "takeoff", "release", "finish"),
    "pass": ("load", "action", "recover"),
    "triple_threat": ("load", "action", "recover"),
}


def _nullish(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    return value


def phase_names_for(action_type: str | None) -> tuple[str, ...]:
    return PHASE_NAMES.get(normalize_action_type(action_type), ())


def normalize_phases(
    action_type: str | None,
    phases: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Keep name/start_ms/end_ms; missing times → null."""
    out: list[dict[str, Any]] = []
    for ph in phases or []:
        out.append({
            "name": ph.get("name"),
            "start_ms": _nullish(ph.get("start_ms")),
            "end_ms": _nullish(ph.get("end_ms")),
        })
    return out


def empty_angle_row(t_ms: float) -> dict[str, float | None]:
    return pack_angle_row(t_ms, {})


def build_identity_block(
    student_id: str | None,
    global_id: str | None = None,
    *,
    confidence: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    return {
        "student_id": student_id,
        "global_id": global_id,
        "confidence": confidence,
        "source": source,
    }


def build_action_event(
    *,
    session_id: str,
    student_id: str | None,
    action_type: str | None,
    start_ms: float | None,
    end_ms: float | None,
    release_ms: float | None = None,
    made: bool | None = None,
    phases: list[dict[str, Any]] | None = None,
    angles: list[dict[str, Any]] | None = None,
    identity: dict[str, Any] | None = None,
    global_id: str | None = None,
    confidence: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One WS message at action finalize. Nullable fields stay JSON null."""
    atype = normalize_action_type(action_type) if action_type else None
    if atype == "unknown":
        atype = action_type
    if atype in POSE_ONLY_ACTION_TYPES:
        release_ms = None
        made = None
    elif atype and not is_shooting_action(atype):
        pass

    ident = identity or build_identity_block(student_id, global_id)
    gid = ident.get("global_id") if ident else global_id

    packed_angles: list[dict[str, Any]] = []
    for row in angles or []:
        t = row.get("t_ms")
        if t is None:
            continue
        packed_angles.append(pack_angle_row(float(t), row))

    msg: dict[str, Any] = {
        "event": EVENT_ACTION,
        "session_id": session_id,
        "student_id": student_id,
        "global_id": gid,
        "action_type": atype,
        "start_ms": _nullish(start_ms),
        "end_ms": _nullish(end_ms),
        "release_ms": _nullish(release_ms),
        "made": made,
        "confidence": _nullish(confidence),
        "phases": normalize_phases(atype, phases),
        "identity": ident,
        "angles": packed_angles,
        "angle_keys": list(ANGLE_KEYS),
    }
    if extra:
        msg["metadata"] = extra
    return msg


def build_gap_event(
    *,
    session_id: str,
    start_ms: float,
    end_ms: float,
    cameras: list[str],
    reason: str = "disconnect",
) -> dict[str, Any]:
    return {
        "event": EVENT_GAP,
        "session_id": session_id,
        "start_ms": float(start_ms),
        "end_ms": float(end_ms),
        "cameras": list(cameras),
        "reason": reason,
    }
