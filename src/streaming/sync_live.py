"""Live 4-cam sync GUI: capture a short buffer from RTSP, then nudge offsets.

Saves compatible json:
  {"anchor_camera":"cam_03","camera_time_offsets_ms":{...}}
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.acquisition.rtsp import LiveFrame, MultiRtspCapture
from src.streaming.clock import CAMS, DEFAULT_ANCHOR, normalize_offsets, save_sync_doc
from src.streaming.display import require_display


def capture_sync_buffers(
    urls: dict[str, str],
    *,
    seconds: float = 8.0,
    max_width: int = 640,
    gap_ms: float = 900.0,
) -> dict[str, list[tuple[float, np.ndarray]]]:
    """Record wall-clock local_ms + downscaled BGR for each camera (no mp4)."""
    buf: dict[str, list[tuple[float, np.ndarray]]] = defaultdict(list)

    def on_frame(fr: LiveFrame) -> None:
        img = fr.bgr
        h, w = img.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            img = cv2.resize(img, (max_width, int(h * scale)))
        buf[fr.camera_id].append((float(fr.local_ms), img))

    cap = MultiRtspCapture(
        urls,
        {c: 0.0 for c in CAMS},
        gap_ms=gap_ms,
        on_frame=on_frame,
        queue_max=8,
    )
    cap.start()
    t0 = time.time()
    try:
        while time.time() - t0 < seconds:
            time.sleep(0.05)
    finally:
        cap.stop()
    return {c: buf.get(c, []) for c in CAMS}


class _BufferSource:
    def __init__(self, cam_id: str, items: list[tuple[float, np.ndarray]]):
        self.cam_id = cam_id
        if not items:
            raise RuntimeError(f"{cam_id} 同步缓冲为空（拉流失败？）")
        items = sorted(items, key=lambda x: x[0])
        self.t0 = float(items[0][0])
        self.times = np.array([t - self.t0 for t, _ in items], dtype=np.float64)
        self.frames = [im for _, im in items]
        dts = np.diff(self.times)
        med = float(np.median(dts)) if len(dts) else 33.0
        self.fps = 1000.0 / max(med, 1.0)
        self.duration_ms = float(self.times[-1]) if len(self.times) else 0.0

    def read_at_local_rel(self, local_rel_ms: float) -> np.ndarray | None:
        if not len(self.times):
            return None
        i = int(np.clip(np.searchsorted(self.times, local_rel_ms), 0, len(self.times) - 1))
        if i > 0 and abs(self.times[i - 1] - local_rel_ms) < abs(self.times[i] - local_rel_ms):
            i -= 1
        return self.frames[i]


def run_live_sync_gui(
    urls: dict[str, str],
    out_path: Path,
    *,
    buffer_sec: float = 8.0,
    anchor: str = DEFAULT_ANCHOR,
    existing: dict[str, Any] | None = None,
) -> Path:
    require_display("直播四路同步 GUI")
    print(f"采集 {buffer_sec:.0f}s 四路预览缓冲（不录像）…", flush=True)
    raw = capture_sync_buffers(urls, seconds=buffer_sec)
    missing = [c for c in CAMS if not raw.get(c)]
    if missing:
        raise SystemExit(f"同步采集失败，无画面：{missing}。请检查 RTSP / ffmpeg 推流。")

    import tkinter as tk
    from tkinter import messagebox, ttk
    from PIL import Image, ImageTk

    sources = {c: _BufferSource(c, raw[c]) for c in CAMS}
    # Shift all to a shared origin: min t0 across cams, store offset of each t0 vs min
    t0_min = min(sources[c].t0 for c in CAMS)
    capture_shift = {c: sources[c].t0 - t0_min for c in CAMS}
    # local_rel on source already starts at 0 for that cam's first frame.
    # True local_ms_rel_to_session = capture_shift[c] + local_rel
    # Operator offset is additional constant Δt on top of capture_shift.

    offs = normalize_offsets(
        (existing or {}).get("camera_time_offsets_ms"),
        anchor_camera=anchor,
    )
    PANEL_W, PANEL_H = 640, 360

    class LiveSyncGUI:
        def __init__(self) -> None:
            self.offsets_ms = dict(offs)
            self.anchor = anchor
            self.selected = anchor
            self.dirty = False
            self.playing = False
            self._tick_job = None
            self._photo: dict[str, Any] = {}
            # common timeline: after offsets, overlapping coverage
            self.common_max_ms = min(
                max(0.0, sources[c].duration_ms + capture_shift[c] - self.offsets_ms[c])
                for c in CAMS
            )
            self.common_ms = 0.0
            self.root = tk.Tk()
            self.root.title(f"直播同步 — 锚点 {anchor}（不写 raw mp4）")
            self.root.protocol("WM_DELETE_WINDOW", self._quit)
            self._build()
            self._refresh()

        def _build(self) -> None:
            top = ttk.Frame(self.root, padding=6)
            top.pack(fill=tk.BOTH, expand=True)
            grid = ttk.Frame(top)
            grid.pack(fill=tk.BOTH, expand=True)
            self.labels: dict[str, tk.Label] = {}
            self.offset_vars: dict[str, tk.DoubleVar] = {}
            self.offset_labels: dict[str, ttk.Label] = {}
            for i, cam in enumerate(CAMS):
                cell = ttk.LabelFrame(grid, text=cam)
                cell.grid(row=i // 2, column=i % 2, sticky="nsew", padx=4, pady=4)
                lab = tk.Label(cell, bg="#222")
                lab.pack()
                self.labels[cam] = lab
                var = tk.DoubleVar(value=self.offsets_ms[cam])
                self.offset_vars[cam] = var
                row = ttk.Frame(cell)
                row.pack(fill=tk.X, pady=2)
                ttk.Label(row, text="offset ms").pack(side=tk.LEFT)
                state = "disabled" if cam == self.anchor else "normal"
                ttk.Scale(
                    row, from_=-5000, to=5000, variable=var, orient=tk.HORIZONTAL,
                    command=lambda _v, c=cam: self._on_off(c), state=state,
                ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
                ol = ttk.Label(row, width=8)
                ol.pack(side=tk.RIGHT)
                self.offset_labels[cam] = ol
                ol.configure(text=f"{self.offsets_ms[cam]:+.0f}")
            grid.columnconfigure(0, weight=1)
            grid.columnconfigure(1, weight=1)
            bot = ttk.Frame(top)
            bot.pack(fill=tk.X, pady=6)
            self.time_var = tk.DoubleVar(value=0.0)
            ttk.Label(bot, text="common t").pack(side=tk.LEFT)
            self.time_scale = ttk.Scale(
                bot, from_=0, to=max(1.0, self.common_max_ms),
                variable=self.time_var, orient=tk.HORIZONTAL,
                command=lambda _v: self._on_time(),
            )
            self.time_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
            self.time_label = ttk.Label(bot, width=18)
            self.time_label.pack(side=tk.RIGHT)
            ctrl = ttk.Frame(top)
            ctrl.pack(fill=tk.X)
            ttk.Button(ctrl, text="▶ Space", command=self._toggle).pack(side=tk.LEFT, padx=2)
            ttk.Button(ctrl, text="Save (S)", command=self._save).pack(side=tk.RIGHT)
            ttk.Button(ctrl, text="Quit", command=self._quit).pack(side=tk.RIGHT, padx=4)
            self.status = ttk.Label(
                top,
                text="对齐同一瞬间。offset>0 表示该机画面相对锚点更晚。锚点 cam_03 固定 0。",
                wraplength=1200,
            )
            self.status.pack(fill=tk.X, pady=4)
            r = self.root
            r.bind("<space>", lambda _e: self._toggle())
            r.bind("<Left>", lambda _e: self._step(-1))
            r.bind("<Right>", lambda _e: self._step(1))
            r.bind("<s>", lambda _e: self._save())
            r.bind("<S>", lambda _e: self._save())
            r.bind("<q>", lambda _e: self._quit())
            r.bind("<Escape>", lambda _e: self._quit())
            for i, cam in enumerate(CAMS, 1):
                r.bind(str(i), lambda _e, c=cam: self._sel(c))

        def _sel(self, cam: str) -> None:
            self.selected = cam

        def _on_off(self, cam: str) -> None:
            if cam == self.anchor:
                self.offset_vars[cam].set(0.0)
                return
            self.offsets_ms[cam] = float(self.offset_vars[cam].get())
            self.offset_labels[cam].configure(text=f"{self.offsets_ms[cam]:+.0f}")
            self.dirty = True
            self._refresh()

        def _on_time(self) -> None:
            self.common_ms = float(self.time_var.get())
            if not self.playing:
                self._refresh()

        def _toggle(self) -> None:
            self.playing = not self.playing
            if self.playing:
                self._tick()

        def _tick(self) -> None:
            if not self.playing:
                return
            self.common_ms = min(self.common_max_ms, self.common_ms + 33.0)
            self.time_var.set(self.common_ms)
            self._refresh()
            if self.common_ms >= self.common_max_ms - 1:
                self.playing = False
                return
            self._tick_job = self.root.after(33, self._tick)

        def _step(self, frames: int) -> None:
            self.common_ms = float(np.clip(self.common_ms + frames * 33.0, 0, self.common_max_ms))
            self.time_var.set(self.common_ms)
            self._refresh()

        def _refresh(self) -> None:
            for cam in CAMS:
                # local_rel = common + offset - capture_shift
                local_rel = self.common_ms + self.offsets_ms[cam] - capture_shift[cam]
                fr = sources[cam].read_at_local_rel(local_rel)
                if fr is None:
                    rgb = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
                else:
                    rgb = cv2.cvtColor(cv2.resize(fr, (PANEL_W, PANEL_H)), cv2.COLOR_BGR2RGB)
                if cam == self.anchor:
                    cv2.rectangle(rgb, (2, 2), (PANEL_W - 3, PANEL_H - 3), (40, 200, 80), 3)
                cv2.putText(
                    rgb, f"{cam} off={self.offsets_ms[cam]:+.0f}ms",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                )
                img = Image.fromarray(rgb)
                photo = ImageTk.PhotoImage(image=img)
                self._photo[cam] = photo
                self.labels[cam].configure(image=photo)
            self.time_label.configure(text=f"{self.common_ms/1000:6.2f}s")
            dirty = " *未保存*" if self.dirty else ""
            self.status.configure(
                text=f"common={self.common_ms:.0f}ms offsets={self.offsets_ms}{dirty}"
            )

        def _save(self) -> None:
            self.offsets_ms[self.anchor] = 0.0
            doc = {
                "anchor_camera": self.anchor,
                "camera_time_offsets_ms": {c: float(self.offsets_ms[c]) for c in CAMS},
                "source": "live_sync_gui",
            }
            save_sync_doc(out_path, doc)
            self.dirty = False
            self.status.configure(text=f"已保存 → {out_path}")
            messagebox.showinfo("Saved", f"Offsets written to:\n{out_path}")

        def _quit(self) -> None:
            if self.dirty and not messagebox.askyesno("Unsaved", "有未保存的偏移，仍要退出吗？"):
                return
            if self._tick_job is not None:
                self.root.after_cancel(self._tick_job)
            self.root.destroy()

        def run(self) -> None:
            self.root.mainloop()

    LiveSyncGUI().run()
    if not out_path.exists():
        # operator quit without save — still write current zeros so later steps have a file
        save_sync_doc(out_path, {
            "anchor_camera": anchor,
            "camera_time_offsets_ms": offs,
            "source": "live_sync_gui_unsaved_defaults",
        })
        print(f"未手动保存，写入默认偏移 → {out_path}", flush=True)
    return out_path
