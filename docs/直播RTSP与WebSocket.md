# 2.2.0 四路 RTSP 直播推理 + WebSocket JSON

> 离线 mkv 批处理（`run_v*_testset.py` / `run_session.py`）不变。本页只描述 **新入口**。

## 1. 行为约定

| 项 | 约定 |
|----|------|
| 输入 | 同时 4 路 RTSP → cam_01–04（职责与现网一致：01/02 姿态+身份，03 动作主时钟，04 球/筐） |
| 抽帧/录像 | **直播不抽帧、不写** `sessions/.../raw/*.mp4` |
| 时钟 | `common_ms = local_ms - offset_ms`；锚点 **cam_03 offset=0**；对齐优先于延迟 |
| 启动 | 先人工同步 GUI，再抽帧标定；开播前正面顺序注册 |
| 身份 | 人脸 ↔ 全局 registry（跨课次）；衣着/身体 ReID **只当堂** |
| 推送 | 独立进程 WebSocket，默认 **`127.0.0.1:8765`**，只推 JSON，不推视频 |
| 节奏 | **动作 finalize 时一条消息**，内含该动作 `[start_ms,end_ms]` 的 `angles[]` |
| 断流 | 自动重连，**继续用启动 offsets**，并发 `timeline_gap` |
| 收尾 | **不**自动 dashboard；事件可 append `data/outputs/live/{session}/events.jsonl` |

## 2. 启动命令

占位 URL 在 [`configs/live.yaml`](../configs/live.yaml) / [`configs/cameras.yaml`](../configs/cameras.yaml) 的 `rtsp_url`。现场改成真实地址。

```bash
conda activate basketball_classroom
export PYTHONPATH=.

# 开课（需 DISPLAY）：同步 → 标定 → 注册
python scripts/run_live_ws.py setup --session live_demo

# 或分步
python scripts/run_live_ws.py sync --session live_demo
python scripts/run_live_ws.py calibrate --session live_demo
python scripts/run_live_ws.py enroll --session live_demo --seconds 45

# 推理 + WS（无 DISPLAY 也可）
python scripts/run_live_ws.py run --session live_demo
```

无 `DISPLAY` 时，`sync` / `calibrate` 的标注 GUI 会打印明确错误并退出。`enroll` 可加 `--no-preview` 无界面采集。

同步 json 兼容现网：

```json
{
  "anchor_camera": "cam_03",
  "camera_time_offsets_ms": {
    "cam_01": 80.0,
    "cam_02": -15.0,
    "cam_03": 0.0,
    "cam_04": 40.0
  }
}
```

默认写入 `data/outputs/live/{session}/sync.json`。标定写入 `data/calibration/live_{session}/`（不覆盖 `v2_4cam_zoned`）。

客户端示例：

```bash
python -c "import asyncio,websockets
async def m():
    async with websockets.connect('ws://127.0.0.1:8765') as ws:
        print(await ws.recv())
asyncio.run(m())"
```

## 3. WebSocket 消息示例

### 动作 finalize（投篮族）

```json
{
  "event": "action_finalized",
  "session_id": "live_demo",
  "student_id": "stu_00",
  "global_id": "stu_global_03",
  "action_type": "jump_shot",
  "start_ms": 12040.0,
  "end_ms": 14880.0,
  "release_ms": 13600.0,
  "made": true,
  "confidence": 0.71,
  "phases": [
    {"name": "load", "start_ms": 12040.0, "end_ms": 12800.0},
    {"name": "takeoff", "start_ms": 12800.0, "end_ms": 13540.0},
    {"name": "release", "start_ms": 13540.0, "end_ms": 13680.0},
    {"name": "follow_through", "start_ms": 13680.0, "end_ms": 14880.0}
  ],
  "identity": {
    "student_id": "stu_00",
    "global_id": "stu_global_03",
    "confidence": "high",
    "source": "face_gallery"
  },
  "angles": [
    {
      "t_ms": 12040.0,
      "right_elbow": 92.4,
      "left_elbow": 148.1,
      "right_knee": 121.0,
      "left_knee": 118.2,
      "right_wrist": 171.3,
      "shooting_elbow": 92.4,
      "shooting_wrist": 171.3
    }
  ],
  "angle_keys": ["right_elbow", "left_elbow", "right_knee", "left_knee", "right_wrist", "shooting_elbow", "shooting_wrist"]
}
```

阶段名：`free_throw` = load/set/release/follow_through；`jump_shot` = load/takeoff/release/follow_through；`layup` = approach/gather/takeoff/release/finish；`pass` / `triple_threat` = load/action/recover。后两者 `release_ms`、`made` 为 `null`。缺测关节角为 `null`。

### 断流空洞

```json
{
  "event": "timeline_gap",
  "session_id": "live_demo",
  "start_ms": 50120.0,
  "end_ms": 52880.0,
  "cameras": ["cam_01"],
  "reason": "disconnect"
}
```

## 4. 用 ffmpeg 模拟四路 RTSP（无真机验收）

仓库测试 mkv 被 gitignore；本地若有 `data/test_data_v3/{g}-{c}.mkv`，可先起一个 RTSP 中继（[MediaMTX](https://github.com/bluenviron/mediamtx)）：

```bash
# 终端 0：RTSP 服务（默认 8554）
# docker run --rm -p 8554:8554 bluenviron/mediamtx:latest
# 或下载 mediamtx 二进制后直接运行

# 终端 1–4：把现有 mkv 循环推上去（路径按本机测试集改）
ffmpeg -re -stream_loop -1 -i data/test_data_v3/1-1.mkv -an \
  -c:v libx264 -preset ultrafast -tune zerolatency -g 30 \
  -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/cam_01

ffmpeg -re -stream_loop -1 -i data/test_data_v3/1-2.mkv -an \
  -c:v libx264 -preset ultrafast -tune zerolatency -g 30 \
  -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/cam_02

ffmpeg -re -stream_loop -1 -i data/test_data_v3/1-3.mkv -an \
  -c:v libx264 -preset ultrafast -tune zerolatency -g 30 \
  -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/cam_03

ffmpeg -re -stream_loop -1 -i data/test_data_v3/1-4.mkv -an \
  -c:v libx264 -preset ultrafast -tune zerolatency -g 30 \
  -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/cam_04
```

然后 `configs/live.yaml` 中的占位 URL 即可直接 `run`。也可用 `--cam_01 file://...mkv` 走文件循环（开发用，仍不写 raw mp4）。

## 5. 单测（不依赖摄像机）

```bash
PYTHONPATH=. python tests/test_live_ws.py
```

覆盖：offset 约定、sync json、`timeline_gap`、动作消息可空字段、`ANGLE_KEYS`、断流不改 Δt、WS jsonl。

## 6. 已知限制

- 直播动作检测是环形缓冲上的近实时路径，召回/NMS 细节弱于离线 `detect_actions_auto` 全会话后处理
- 无本堂标定或单视可见时，`angles[]` 各关节为 `null`
- 进球 `made` 为 cam_04 球相对筐的简化几何，弱于离线遮挡否决全链路
- 默认同机 YOLO 串行感知，四路 1080p 可能跟不上 30fps（对齐优先，队列有界）
- 不提供实时 dashboard / 评分 HTML / 视频预览推送
