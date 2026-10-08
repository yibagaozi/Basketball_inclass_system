#!/usr/bin/env python3
"""2.2.0 四路 RTSP 直播推理 + WebSocket JSON（独立进程，默认 127.0.0.1:8765）。

子命令
------
  sync       开课人工同步 GUI → camera_time_offsets_ms（锚点 cam_03=0）
  calibrate  从四路各抽一帧 → 标注+solve（本堂目录，默认不覆盖 v2/v3）
  enroll     注册机位直播预览，正面顺序注册（不落 mp4）
  run        拉流推理，动作 finalize 时广播一条 JSON；断流重连沿用启动 offsets
  setup      sync → calibrate → enroll（需 DISPLAY）

不写 sessions/.../raw/*.mp4，流结束后不生成 dashboard。
事件可追加 data/outputs/live/{session}/events.jsonl。

离线批处理请继续用 scripts/run_v*_testset.py / pipelines/run_session.py。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ROOT as SRC_ROOT  # noqa: E402
from src.streaming.clock import DEFAULT_ANCHOR, load_sync_doc, save_sync_doc  # noqa: E402
from src.streaming.live_config import (  # noqa: E402
    default_offsets,
    enroll_camera,
    load_live_config,
    rtsp_urls,
    websocket_bind,
)
from src.streaming.edge_ws_client import EdgeWsClient


def _session_id(args: argparse.Namespace) -> str:
    if args.session:
        return str(args.session)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"live_{stamp}_{uuid.uuid4().hex[:6]}"


def _live_dir(session_id: str) -> Path:
    d = SRC_ROOT / "data" / "outputs" / "live" / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _urls(args: argparse.Namespace) -> dict[str, str]:
    urls = rtsp_urls()
    if args.rtsp_json:
        extra = json.loads(Path(args.rtsp_json).read_text(encoding="utf-8"))
        urls.update({str(k): str(v) for k, v in extra.items()})
    for cam in ("cam_01", "cam_02", "cam_03", "cam_04"):
        val = getattr(args, cam, None)
        if val:
            urls[cam] = val
    missing = [c for c, u in urls.items() if not u]
    if missing:
        raise SystemExit(f"缺少 RTSP URL：{missing}。写入 configs/live.yaml 或传 --cam_XX")
    return urls


def _sync_path(session_id: str, args: argparse.Namespace) -> Path:
    if args.sync_json:
        return Path(args.sync_json)
    return _live_dir(session_id) / "sync.json"


def _calib_dir(session_id: str, args: argparse.Namespace) -> Path:
    if args.calib_dir:
        return Path(args.calib_dir)
    return SRC_ROOT / "data" / "calibration" / f"live_{session_id}"


def cmd_sync(args: argparse.Namespace) -> None:
    from src.streaming.sync_live import run_live_sync_gui

    sid = _session_id(args)
    out = _sync_path(sid, args)
    existing = load_sync_doc(out) if out.exists() else None
    run_live_sync_gui(
        _urls(args),
        out,
        buffer_sec=float(args.buffer_sec),
        anchor=args.anchor,
        existing=existing,
    )
    print(f"sync → {out}", flush=True)


def cmd_calibrate(args: argparse.Namespace) -> None:
    from src.streaming.calibrate_live import annotate_and_solve, grab_live_calibration_frames

    sid = _session_id(args)
    calib = _calib_dir(sid, args)
    frames = calib / "frames"
    print("从直播抽标定帧（不录像）…", flush=True)
    written = grab_live_calibration_frames(_urls(args), frames)
    if len(written) < 3:
        raise SystemExit("标定至少需要 cam_01–03 各一帧。")
    if args.grab_only:
        print(f"frames → {frames}", flush=True)
        return
    ann = calib / "annotations.json"
    out = annotate_and_solve(frames, ann, calib)
    print(f"calibration → {out}", flush=True)


def cmd_enroll(args: argparse.Namespace) -> None:
    from src.streaming.live_identity import enroll_live

    sid = _session_id(args)
    urls = _urls(args)
    cam = args.enroll_camera or enroll_camera()
    url = urls.get(cam) or ""
    preview_dir = _live_dir(sid) / "enroll_preview"
    ids = enroll_live(
        sid,
        url,
        seconds=float(args.seconds),
        sample_hz=float(args.sample_hz),
        preview=not args.no_preview,
        expected_persons=args.expected_persons,
        preview_dir=preview_dir,
    )
    meta = _live_dir(sid) / "enrollment.json"
    meta.write_text(
        json.dumps({"session_id": sid, "student_ids": ids, "enroll_camera": cam}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"enroll → {ids}  {meta}", flush=True)


def cmd_run(args: argparse.Namespace) -> None:
    from src.streaming.live_engine import LiveEngine, load_offsets_file
    from src.streaming.ws_hub import WsHub

    sid = _session_id(args)
    live_dir = _live_dir(sid)
    sync_p = _sync_path(sid, args)
    if sync_p.exists():
        offsets = load_offsets_file(sync_p)
        print(f"offsets ← {sync_p}  {offsets}", flush=True)
    else:
        offsets = default_offsets()
        save_sync_doc(sync_p, {
            "anchor_camera": DEFAULT_ANCHOR,
            "camera_time_offsets_ms": offsets,
            "source": "run_defaults",
        })
        print(f"无同步文件，使用默认全 0 偏移 → {sync_p}", flush=True)

    calib = _calib_dir(sid, args)
    if not (calib / "cameras.json").exists() and not list(calib.glob("cam_*.json")):
        print(f"警告：未找到本堂标定 {calib}，angles[] 将为 null。", flush=True)
        calib_dir = None
    else:
        calib_dir = calib

    host, port = websocket_bind()
    if args.host:
        host = args.host
    if args.port:
        port = int(args.port)
    jsonl = live_dir / "events.jsonl"
    edge_url = "ws://127.0.0.1:8081/internal/cv/stream"
    hub = EdgeWsClient(edge_url, jsonl_path=jsonl)
    # hub = WsHub(host, port, jsonl_path=jsonl)
    hub.start_background()
    print(f"edge {edge_url} jsonl {jsonl}, flush=true")
    # print(f"WebSocket ws://{host}:{port}  jsonl={jsonl}", flush=True)

    engine = LiveEngine(
        sid,
        _urls(args),
        offsets,
        hub=hub,
        calib_dir=calib_dir,
        gallery_session=args.gallery_session or sid,
    )
    engine.start()
    print("直播推理已启动（Ctrl+C 停止；不生成 dashboard）。", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("停止…", flush=True)
    finally:
        engine.stop()


def cmd_setup(args: argparse.Namespace) -> None:
    cmd_sync(args)
    cmd_calibrate(args)
    cmd_enroll(args)
    sid = _session_id(args)
    print(
        "\n下一步：\n"
        f"  PYTHONPATH=. python scripts/run_live_ws.py run --session {sid}\n",
        flush=True,
    )


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--session", default=None, help="session id（默认 live_时间戳）")
    p.add_argument("--sync-json", type=Path, default=None)
    p.add_argument("--calib-dir", type=Path, default=None)
    p.add_argument("--rtsp-json", type=Path, default=None, help='{"cam_01":"rtsp://..."}')
    p.add_argument("--cam_01")
    p.add_argument("--cam_02")
    p.add_argument("--cam_03")
    p.add_argument("--cam_04")
    p.add_argument("--anchor", default=DEFAULT_ANCHOR)


def main() -> None:
    cfg = load_live_config()
    p = argparse.ArgumentParser(description="四路 RTSP 直播 + WebSocket JSON（v2.2.0）")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sync", help="人工同步 GUI")
    _add_common(s)
    s.add_argument("--buffer-sec", type=float, default=8.0)
    s.set_defaults(func=cmd_sync)

    s = sub.add_parser("calibrate", help="抽帧 + 标定 GUI + solve")
    _add_common(s)
    s.add_argument("--grab-only", action="store_true")
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("enroll", help="直播正面顺序注册")
    _add_common(s)
    s.add_argument("--enroll-camera", default=None)
    s.add_argument("--seconds", type=float, default=45.0)
    s.add_argument("--sample-hz", type=float, default=float((cfg.get("identity") or {}).get("enroll_sample_hz") or 8))
    s.add_argument("--expected-persons", type=int, default=None)
    s.add_argument("--no-preview", action="store_true")
    s.set_defaults(func=cmd_enroll)

    s = sub.add_parser("run", help="推理 + WebSocket")
    _add_common(s)
    s.add_argument("--host", default=None, help="默认 127.0.0.1")
    s.add_argument("--port", type=int, default=None, help="默认 8765")
    s.add_argument("--gallery-session", default=None, help="复用已有 enrollment session id")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("setup", help="sync + calibrate + enroll")
    _add_common(s)
    s.add_argument("--buffer-sec", type=float, default=8.0)
    s.add_argument("--enroll-camera", default=None)
    s.add_argument("--seconds", type=float, default=45.0)
    s.add_argument("--sample-hz", type=float, default=8.0)
    s.add_argument("--expected-persons", type=int, default=None)
    s.add_argument("--no-preview", action="store_true")
    s.add_argument("--grab-only", action="store_true")
    s.set_defaults(func=cmd_setup)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
