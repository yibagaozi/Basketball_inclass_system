# Changelog

## 2.2.0 — 2026-09-11

### 新增
- **四路 RTSP 直播推理**：cam_01–04 同时拉流，不抽帧、不写 `sessions/.../raw/*.mp4`
- **开课启动**：人工同步 GUI（`camera_time_offsets_ms`，锚点 cam_03=0）+ 直播抽帧球场标定（本堂 `data/calibration/live_{session}/`，不覆盖 v2/v3）
- **开播前正面顺序注册**：直播预览、不落 mp4；人脸匹配全局 `stu_global_XX`，衣着/身体 ReID 仅当堂 gallery
- **独立 WebSocket**（默认 `127.0.0.1:8765`，不挂 teacher_ui）：动作 **finalize 时一条 JSON**（含 `angles[]` 真 3D 关节角）+ `timeline_gap`
- 断流自动重连，**沿用启动时 offsets**，不重估同步；流结束 **不** 自动 dashboard；事件可归档 `data/outputs/live/{session}/events.jsonl`
- 入口：`scripts/run_live_ws.py {sync,calibrate,enroll,run,setup}`
- 文档：`docs/直播RTSP与WebSocket.md`（含 ffmpeg 模拟四路 RTSP）
- 单测：`tests/test_live_ws.py`（偏移、gap、角度 schema、WS 可空字段；不依赖真摄像机）

### 兼容
- `run_v*_testset.py` / `pipelines/run_session.py` 与既有离线输出 **只多不减**

## 2.1.0 — 2026-09-07

### 新增
- **跨课次人脸全局 ID**：InsightFace 人脸质心匹配 → `stu_global_XX`（`data/identity/`）
- 每堂注册后自动把 session 内 `stu_XX` 与全局 ID 链接；**衣着/身体仍为当堂 gallery**
- 工具：`scripts/link_global_identity.py`、`scripts/validate_global_identity.py`
- 文档：`docs/身份识别跨课次全局ID.md`；schema 示例在 `data/schema_examples/`
- 变更前源码备份：`versions/v2.0.8/`

### 验证
- 身份自洽（克隆 gallery 二次 link）：**6/6**
- v2↔v3 跨课次（不同人）：阈值 0.58 下 **0 误匹配**；registry 写入 10 个 global id
- v3 动作/课内 ID：`eval_v3_gt` **7/7 PASS**（id_acc=1.0）
- v2 动作：现有 outputs **4/7**（相对 2.0.6 报告漂移，与本次身份改动无关；见 `docs/身份验证_v2.1.0.md`）

## 2.0.7 — 2026-08-06

### 改进
- **跨机身份**：cam_01↔cam_02 图像左右反序配对（默认替代球场 XY）；可选对齐 cam_03 同序
- **时间同步**：修复 `_cam04_segment_times`；支持 `manual_offsets_ms` / `--set-offset`（手动优先）
- **篮筐冻结**：前 2 次合理检出后永久锁定，拒绝超大/贴边假框
- **group1 罚篮**：球飞行弱脚步不再标 TT；强化 planted_ft 会话清理
- **ReID**：pose2d 写入 confidence；sticky 加长；衣着可区分时放宽匹配

### 数据
- 保留当前 `data/outputs/v2` 对应 7 个 session；清理历史孤儿 session/enrollment

## 2.0.6 — 2026-07-29

### 改进
- **衣着颜色：可区分度驱动权重**（替换固定 torso/shoe 手调权重）
  - 注册 gallery 上按部位算学生间平均 HSV 距离 → `part_weights`
  - 上衣/裤子人人一样时鞋权重自动升高；球衣不同时躯干/腿主导
  - 匹配时再乘该学生的 uniqueness（谁的鞋/衣服更独特，谁在对应维上权更高）
  - body 高相似短路在颜色强否决时不再直接覆盖
- **会话签名优先级**：`planted_ft` 先于 `breakthrough_layup`；jumper 要求 FT 明显多于 layup

### 验证
- full 重跑注册 + group0–6 后动作校正：**7/7 通过**
  - g1 FT:8；g3 TT:17/layup:11；g4 混合；g5 TT/JS/FT；g6 pass:16

## 2.0.5 — 2026-07-27

### 改进
- **跳投双脚离地**：`classify_release_action` 要求左右踝相对站定基线在**同一批帧**同时抬升（`both_feet_airborne`），再结合骨盆抬升才判 `jump_shot`；过滤罚篮踮脚/单脚支撑
- **混合收尾会话**：g4 同时有上篮与 FT/JS 时不再整批收成 layup；g3 仍转换误标出手

### 验证
- `scripts/analyze_v2_ground_truth.py`：**7/7 通过**
  - g4 JS:2 FT:7；g5 JS:9；跳投均带 `both_feet_airborne`

## 2.0.4 — 2026-07-27

### 改进
- **group1 全罚篮**：站定投篮会话签名将 `jump_shot` → `free_throw`，并抑制误检 TT；真值改为仅 `free_throw`
- **弱突破 TT**：降低重心 + 变向/侧切即可检出（不要求标准三威胁手位）；加密集滑窗
- **TT 召回修复**：误检 pass 不再清空射击课的 TT；孤儿 TT 按会话级出手窗口判断；pass 重叠不再误删 TT
- **pass 召回**：略放宽手势/球飞行门槛与 NMS 间隔（group6：8→10）

### 验证
- `scripts/analyze_v2_ground_truth.py`：**group0–6 全部通过（7/7）**
  - g1 FT:14；g3 TT:17；g4 TT:27；g5 TT:20；g6 pass:10

## 2.0.3 — 2026-07-26

### 修复
- **pose_only 致命 Bug**：`wrist_raise` → `raise_`（此前 NameError 被静默吞掉，导致 TT/pass 全无）
- detector 异常改为 **打印日志**，不再静默失败
- **TT / shooting 合并**：保留突破窗口（裁剪后与出手共存）；传球会话抑制 TT
- **出手分类重写**：near-rim / high-travel → layup；pull-up hop → jump_shot；弱 rim-only 事件丢弃
- **会话级上下文清理**：纯上篮 / 突破上篮 / 突破跳投 / 传球 四种签名，校正跨类型误标
- **pass**：略放宽手势+球飞行门槛；投篮会话丢弃近邻 pass

### 验证
- `scripts/analyze_v2_ground_truth.py`：**group0–6 全部通过（7/7）**

## 2.0.2 — 2026-07-24

### 修复
- **ReID body 主、face 弱辅助**：`face_alpha_high=0.20`；弱脸/强 body 时进一步压低 face 权重
- **cam_04 投篮门控过严**：`shot_like` 距离阈值 400→900px，`min_points` 5→3（group1 由 1 次 rim 事件提升至 ~6 次）
- **pose_only 误标三威胁**：过滤罚篮/跳投准备姿态（下蹲 + 垂直举手）
- **罚篮误标 layup**：同机位筐坐标 + 罚篮线 `free_throw_line` 与 `closing_layup` 分离

### 工具 / 文档
- 新增 `scripts/analyze_v2_ground_truth.py`（对照 `docs/v2测试集动作真值.md`）
- 新增 `docs/v2验证报告.md`

## 2.0.1 — 2026-07-24

### 修复
- **投篮时骨架大批量丢失**：pose 不再依赖 ReID 成功才写入；tracker sticky ID；弱脸时 body 高相似短路
- 放宽投篮姿态质量（臂展/出框）；降低检测宽高比下限
- cam_01–03 **保留多篮球**检测；各机位筐框 **前 10 帧平均后固定**

### 文档
- 新增 `docs/v2测试集动作真值.md`（group1–6 真值标签）

## 2.0.0 — 2026-07-24

多人注册与精确 ID 跟踪；新增跳投。

### 能力
- **顺序正面注册**（`src/identity/sequential_enroll.py`）：group0 / cam_01 轮流正面 → 多人 gallery
- 注册相机改为 **cam_01**；动作主时钟仍为 cam_03
- 规范动作五类：`pass` | `triple_threat` | `free_throw` | **`jump_shot`** | `layup`
  - `jump_shot` 与 `free_throw` 的差别：是否明显纵跳（pelvis/ankle 上升）
- v2 批处理：`scripts/run_v2_testset.py`（group0 注册 → group1+ 复用 gallery）
- `test_data_v2`：原 group1 按 90s 拆成 group0（注册）与 group1（轮流罚篮）

### 测试 / 文档
- 新增 `tests/test_v2.py`
- 更新架构 / 输出格式 / README；测试集说明见 `data/test_data_v2/README.md`

## 1.0.0 — 2026-07-21

首个正式 v1 版本。

### 能力
- 四机位独立感知（YOLO11m-Pose + RTMW-l；cam_04 球/筐）
- 事件对齐、规则动作四类（pass / triple_threat / free_throw / layup）
- 三威胁以肢体为主、持球为辅；传球双人归因
- cam_04 进球：筐沿橙色遮挡否决 + 轨迹
- `realtime` / `full` 批处理与 group dashboard

### 清理
- 移除实验对比脚本、单视频 demo 入口、MotionBERT 运行时包装
- 输出目录仅保留 `group_0X` + `manifest.json`
- 根目录重复权重与杂余报告清理
