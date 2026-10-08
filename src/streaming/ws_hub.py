"""Independent JSON WebSocket hub (default 127.0.0.1:8765). Not teacher_ui."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError:  # pragma: no cover
    websockets = None  # type: ignore
    WebSocketServerProtocol = Any  # type: ignore


class JsonlSink:
    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, obj: dict[str, Any]) -> None:
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


class WsHub:
    """Broadcast dicts as JSON text to all connected clients; append jsonl."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8081,
        *,
        jsonl_path: Path | None = None,
    ):
        self.host = host
        self.port = int(port)
        self.sink = JsonlSink(jsonl_path)
        self._clients: set[Any] = set()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server = None
        self._ready = threading.Event()

    def publish(self, event: dict[str, Any]) -> None:
        self.sink.append(event)
        payload = json.dumps(event, ensure_ascii=False)
        loop = self._loop
        if loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), loop)

    async def _broadcast(self, payload: str) -> None:
        with self._lock:
            clients = list(self._clients)
        dead = []
        for ws in clients:
            try:
                await ws.send(payload)
            except Exception:
                dead.append(ws)
        if dead:
            with self._lock:
                for ws in dead:
                    self._clients.discard(ws)

    async def _handler(self, ws: WebSocketServerProtocol, _path: str | None = None) -> None:
        with self._lock:
            self._clients.add(ws)
        try:
            await ws.wait_closed()
        finally:
            with self._lock:
                self._clients.discard(ws)

    async def _serve(self) -> None:
        if websockets is None:
            raise RuntimeError("需要 websockets 包：pip install websockets")
        self._server = await websockets.serve(
            self._handler,
            self.host,
            self.port,
            reuse_address=True,
        )
        self._ready.set()
        await asyncio.Future()

    def start_background(self) -> None:
        if websockets is None:
            raise RuntimeError("需要 websockets 包：pip install websockets")

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            try:
                loop.run_until_complete(self._serve())
            except Exception:
                self._ready.set()
                raise
            finally:
                loop.close()

        self._thread = threading.Thread(target=_run, name="live-ws", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError(f"WebSocket 未能在 {self.host}:{self.port} 启动")

    def stop(self) -> None:
        loop = self._loop
        if loop is not None and self._server is not None:
            def _close() -> None:
                self._server.close()
            loop.call_soon_threadsafe(_close)
