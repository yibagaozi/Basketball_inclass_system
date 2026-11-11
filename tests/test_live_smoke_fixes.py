"""Fixes found by the end-to-end smoke run (file sources + fake edge)."""

from __future__ import annotations

import asyncio
import time

import cv2
import numpy as np
import pytest

from src.acquisition.rtsp import RtspCameraWorker

websockets = pytest.importorskip("websockets")


def test_looping_file_emits_no_gap(tmp_path):
    path = tmp_path / "short.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 50.0, (32, 24))
    for i in range(10):
        w.write(np.full((24, 32, 3), i, np.uint8))
    w.release()
    gaps, frames = [], []
    worker = RtspCameraWorker("cam_01", str(path), 0.0, loop_file=True, queue_max=0,
                              on_frame=frames.append, on_gap=gaps.append)
    worker.start()
    time.sleep(0.7)  # ~35 frames at 50 fps → the 10-frame file loops several times
    worker.stop()
    assert len(frames) > 15
    assert gaps == []


class _FakeWs:
    def __init__(self, exc):
        self.exc = exc

    async def send(self, text):
        raise self.exc


def _closed_exc():
    from websockets.exceptions import ConnectionClosed

    try:
        from websockets.frames import Close
        return ConnectionClosed(Close(1006, ""), None)
    except Exception:  # websockets < 10 signature
        return ConnectionClosed(1006, "")


def test_send_error_on_open_connection_keeps_it():
    from src.streaming.edge_ws_client import EdgeWsClient

    cli = EdgeWsClient("ws://127.0.0.1:9/x")
    ws = _FakeWs(ValueError("bad frame"))
    cli._ws = ws
    asyncio.run(cli._send(ws, "{}"))
    assert cli._ws is ws and cli.send_failed == 1  # later events still go out

    closed = _FakeWs(_closed_exc())
    cli._ws = closed
    asyncio.run(cli._send(closed, "{}"))
    assert cli._ws is None and cli.send_failed == 2  # reconnect path


def test_stop_does_not_wait_out_reconnect_backoff(tmp_path):
    worker = RtspCameraWorker("cam_01", str(tmp_path / "missing.avi"), 0.0, reconnect_sec=30.0, queue_max=0)
    worker.start()
    time.sleep(0.3)  # open failed → in the 30 s backoff
    t0 = time.monotonic()
    worker.stop()
    assert time.monotonic() - t0 < 1.0 and not worker._thread.is_alive()
