"""EdgeWsClient: survives edge restarts; never blocks or raises when edge is down."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time

import pytest

websockets = pytest.importorskip("websockets")

from src.streaming.edge_ws_client import EdgeWsClient  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Edge:
    """Minimal stand-in for edge's /internal/cv/stream (records envelopes)."""

    def __init__(self, port: int):
        self.port = port
        self.got: list[dict] = []

    def start(self) -> "_Edge":
        ready = threading.Event()

        def run() -> None:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            async def main() -> None:
                self.stop_evt = asyncio.Event()

                async def handler(ws, *_path):  # websockets <10.1 passes path too
                    async for msg in ws:
                        self.got.append(json.loads(msg))

                async with websockets.serve(handler, "127.0.0.1", self.port):
                    ready.set()
                    await self.stop_evt.wait()

            self.loop.run_until_complete(main())

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        assert ready.wait(5)
        return self

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.stop_evt.set)
        self.thread.join(5)


def _wait(cond, timeout: float = 6.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


def test_reconnects_after_edge_restart(tmp_path):
    port = _free_port()
    edge = _Edge(port).start()
    cli = EdgeWsClient(f"ws://127.0.0.1:{port}/internal/cv/stream",
                       jsonl_path=tmp_path / "events.jsonl", reconnect_sec=0.2)
    cli.start_background()
    try:
        assert _wait(lambda: cli._ws is not None)
        cli.publish({"event": "action_finalized", "n": 1})
        assert _wait(lambda: len(edge.got) == 1)
        assert edge.got[0]["type"] == "actionFinalized"
        assert edge.got[0]["seq"] == 1
        assert edge.got[0]["payload"] == {"event": "action_finalized", "n": 1}

        edge.stop()  # edge restarts
        assert _wait(lambda: cli._ws is None)
        edge = _Edge(port).start()
        assert _wait(lambda: cli.connects >= 2 and cli._ws is not None)

        cli.publish({"event": "timeline_gap", "n": 2})
        assert _wait(lambda: len(edge.got) == 1)
        assert edge.got[0]["type"] == "timelineGap"
        assert edge.got[0]["seq"] == 2
    finally:
        cli.stop()
        edge.stop()
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["n"] for x in lines] == [1, 2]


def test_publish_without_edge_drops_quietly(tmp_path):
    cli = EdgeWsClient(f"ws://127.0.0.1:{_free_port()}/x",
                       jsonl_path=tmp_path / "e.jsonl", reconnect_sec=0.1)
    cli.start_background()
    try:
        for i in range(3):
            cli.publish({"event": "action_finalized", "n": i})
        assert cli.dropped == 3
    finally:
        cli.stop()
    assert len((tmp_path / "e.jsonl").read_text(encoding="utf-8").splitlines()) == 3


def test_seq_is_unique_across_threads():
    cli = EdgeWsClient("ws://127.0.0.1:9/x")  # never started: publish only numbers + drops

    def burst() -> None:
        for _ in range(500):
            cli.publish({"event": "timeline_gap"})

    threads = [threading.Thread(target=burst) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert cli._seq == 2000
