# src/streaming/edge_ws_client.py
#
# 直播 WS 输出:由“算法起 WS 服务(WsHub)等人连”改为“算法作客户端连 edge 推事件”。
# 与 WsHub 完全同接口(start_background / publish / stop),LiveEngine 一行都不用改。
#
# - 连接:ws://127.0.0.1:8080/internal/cv/stream(edge 的本机内部通道;算法与 edge 同机)
# - 每条事件包一层 edge 信封:{"type","seq","ts","payload"},payload = 你现在
#   build_action_event / build_gap_event 产出的 dict 原样(字段一个都不改名,edge 认 snake_case)。
# - edge 未起/断开时自动重连;未连上就丢该条(实时数据可丢)。
import asyncio
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import websockets  # 已是 v2.2.0 依赖

# 算法内部事件名 → edge 信封 type
TYPE_MAP = {"action_finalized": "actionFinalized", "timeline_gap": "timelineGap"}


class EdgeWsClient:
    """与 WsHub 同接口,但作为客户端连 edge 的 /internal/cv/stream 上推事件。"""

    def __init__(
        self,
        url: str = "ws://127.0.0.1:8081/internal/cv/stream",
        *,
        jsonl_path: Optional[Path] = None,
        reconnect_sec: float = 2.0,
    ):
        self.url = url
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.reconnect_sec = reconnect_sec
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ws = None
        self._seq = 0
        self._running = False

    def start_background(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_loop())

    async def _connect_loop(self) -> None:
        while self._running:
            try:
                async with websockets.connect(self.url, max_size=None) as ws:
                    self._ws = ws
                    # 只上行,不需要读;保持连接直到断开
                    while self._running:
                        await asyncio.sleep(0.5)
            except Exception:  # edge 未起/断开 → 重连
                self._ws = None
                await asyncio.sleep(self.reconnect_sec)

    def publish(self, event: dict[str, Any]) -> None:
        # 可选:和原来一样落本地 jsonl
        if self.jsonl_path:
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._seq += 1
        frame = {
            "type": TYPE_MAP.get(event.get("event"), event.get("event")),
            "seq": self._seq,
            "ts": datetime.now(timezone.utc).isoformat(),
            "payload": event,  # ← 原事件 dict 原样,字段不改名
        }
        loop, ws = self._loop, self._ws
        if loop is None or ws is None:
            return  # 未连上就丢这条(edge 断线;实时数据可丢)
        asyncio.run_coroutine_threadsafe(
            ws.send(json.dumps(frame, ensure_ascii=False)), loop
        )

    def stop(self) -> None:
        self._running = False
