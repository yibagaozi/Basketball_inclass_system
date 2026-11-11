# src/streaming/edge_ws_client.py
#
# 直播 WS 输出:由“算法起 WS 服务(WsHub)等人连”改为“算法作客户端连 edge 推事件”。
# 与 WsHub 完全同接口(start_background / publish / stop),LiveEngine 一行都不用改。
#
# - 连接:ws://127.0.0.1:8081/internal/cv/stream(edge 的本机内部通道;算法与 edge 同机)
# - 每条事件包一层 edge 信封:{"type","seq","ts","payload"},payload = 你现在
#   build_action_event / build_gap_event 产出的 dict 原样(字段一个都不改名,edge 认 snake_case)。
# - edge 未起/断开时自动重连;未连上就丢该条(实时数据可丢),jsonl 照常落盘。
#   连接关闭由 wait_closed() 感知:websockets 旧版(legacy,<14)与新版 asyncio 实现都有该方法。
import asyncio
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import websockets  # 已是 v2.2.0 依赖
import websockets.exceptions

# 算法内部事件名 → edge 信封 type
TYPE_MAP = {"action_finalized": "actionFinalized", "timeline_gap": "timelineGap"}

_LOG_EVERY_SEC = 10.0


def _jsonable(o: Any) -> Any:
    """json.dumps default: numpy 标量/数组、set 等转成 JSON 可表示的值(原来会抛 TypeError)。"""
    if hasattr(o, "item") and callable(o.item) and getattr(o, "ndim", 1) == 0:
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


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
        self._stop_evt: Optional[asyncio.Event] = None
        self._seq = 0
        self._seq_lock = threading.Lock()  # publish 来自取流线程(gap)与 finalize 线程
        self._jsonl_lock = threading.Lock()
        self._running = False
        self.connects = 0  # 成功连上的次数(含重连)
        self.dropped = 0  # 未连上而丢弃的条数
        self.send_failed = 0
        self.jsonl_failed = 0  # events.jsonl 写失败(如磁盘满);不影响推送
        self.encode_failed = 0
        self._last_log = 0.0

    def start_background(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name="edge-ws", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_loop())

    def _log_throttled(self, msg: str) -> None:
        now = time.monotonic()
        if now - self._last_log >= _LOG_EVERY_SEC:
            self._last_log = now
            print(f"  [edge-ws] {msg}", flush=True)

    async def _connect_loop(self) -> None:
        self._stop_evt = asyncio.Event()
        while self._running:
            try:
                async with websockets.connect(self.url, max_size=None) as ws:
                    self._ws = ws
                    self.connects += 1
                    print(f"  [edge-ws] 已连接 {self.url}（第 {self.connects} 次）", flush=True)
                    # 只上行;等到连接关闭(edge 重启/断网)或 stop() 再出来,然后重连
                    closed = asyncio.ensure_future(ws.wait_closed())
                    stopped = asyncio.ensure_future(self._stop_evt.wait())
                    await asyncio.wait({closed, stopped}, return_when=asyncio.FIRST_COMPLETED)
                    for t in (closed, stopped):
                        if not t.done():
                            t.cancel()
                    if self._running:
                        print(f"  [edge-ws] 连接已断开，{self.reconnect_sec:g}s 后重连", flush=True)
            except Exception as exc:  # edge 未起/拒绝 → 重连
                self._log_throttled(f"连接 {self.url} 失败: {type(exc).__name__}: {exc}")
            finally:
                self._ws = None
            if self._running:
                try:
                    await asyncio.wait_for(self._stop_evt.wait(), timeout=self.reconnect_sec)
                except asyncio.TimeoutError:
                    pass

    async def _send(self, ws, text: str) -> None:
        try:
            await ws.send(text)
        except websockets.exceptions.ConnectionClosed as exc:
            self.send_failed += 1
            if self._ws is ws:
                self._ws = None  # 连接已关闭：后续直接丢弃，等 _connect_loop 重连
            self._log_throttled(f"发送失败（连接已关闭，累计 {self.send_failed} 条）: {exc}")
        except Exception as exc:
            # 连接仍在（例如单条消息异常）：只记这一条，不影响后续事件
            self.send_failed += 1
            self._log_throttled(f"发送失败（累计 {self.send_failed} 条）: {type(exc).__name__}: {exc}")

    def publish(self, event: dict[str, Any]) -> None:
        """从不抛异常:调用方是 finalize 线程,这里抛出会让整条动作输出停掉。"""
        try:
            line = json.dumps(event, ensure_ascii=False, default=_jsonable) + "\n"
        except Exception as exc:
            self.encode_failed += 1
            self._log_throttled(f"事件序列化失败(累计 {self.encode_failed} 条): {type(exc).__name__}: {exc}")
            return
        # 可选:和原来一样落本地 jsonl;写失败(磁盘满等)只计数,照常推送 edge
        if self.jsonl_path:
            try:
                with self._jsonl_lock:
                    with self.jsonl_path.open("a", encoding="utf-8") as f:
                        f.write(line)
            except OSError as exc:
                self.jsonl_failed += 1
                self._log_throttled(f"events.jsonl 写入失败(累计 {self.jsonl_failed} 次,磁盘满?): {exc}")
        with self._seq_lock:
            self._seq += 1
            seq = self._seq
        frame = {
            "type": TYPE_MAP.get(event.get("event"), event.get("event")),
            "seq": seq,
            "ts": datetime.now(timezone.utc).isoformat(),
            "payload": event,  # ← 原事件 dict 原样,字段不改名
        }
        loop, ws = self._loop, self._ws
        if loop is None or ws is None or loop.is_closed():
            self.dropped += 1  # 未连上就丢这条(edge 断线;实时数据可丢)
            self._log_throttled(f"未连接 edge，已丢弃 {self.dropped} 条（jsonl 已记录）")
            return
        coro = self._send(ws, json.dumps(frame, ensure_ascii=False, default=_jsonable))
        try:
            asyncio.run_coroutine_threadsafe(coro, loop)
        except RuntimeError:  # loop 已关闭
            coro.close()
            self.dropped += 1

    def stop(self) -> None:
        self._running = False
        loop, evt = self._loop, self._stop_evt
        if loop is not None and evt is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(evt.set)
            except RuntimeError:
                pass
