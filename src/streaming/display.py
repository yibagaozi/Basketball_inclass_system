"""GUI availability check (sync / calibrate annotate)."""

from __future__ import annotations

import os
import sys


def has_display() -> bool:
    if sys.platform in {"win32", "darwin"}:
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def require_display(purpose: str) -> None:
    if has_display():
        return
    raise SystemExit(
        f"错误：{purpose} 需要图形界面，但当前没有 DISPLAY / WAYLAND_DISPLAY。"
        "请在有桌面的会话中运行，或用 SSH -Y 转发 X11。"
    )
