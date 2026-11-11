# Changelog

## 2.2.0+enroll.1.live.3（重新整理）— 2026-11-11

原补丁 0009–0029 重新整理为三个补丁，在 2.2.0+enroll.1（0001–0005）上安装；不依赖 0006–0008（标定修复，只改标定相关文件，可先可后）：

- live-01：实时推理可靠性与性能（原 0010–0020）
- live-02：长时间运行稳定性、观测锚点、默认关闭的开关、文档与版本（原 0021–0025、0027–0029 与文档 0009、0022、0028）
- live-03（可选，可暂缓）：录像回放与精度评估工具（原 0026）

装完 live-01 + live-02 与按原顺序装完 0009–0029 的代码相同（只少评估工具）。

## 2.2.0+enroll.1.live.3 — 2026-11-04（增量补丁 0029）

见 `docs/实时推理长时间运行排查.md`。修复“运行一段时间后不再输出”的几条永久停住路径：

- finalize 循环任何异常都不再结束线程（原来一次写 events.jsonl 失败，如磁盘满，就会让本节课之后不再有动作和 `[live]` 日志）
- events.jsonl / metrics.jsonl 写失败只计数，推送 edge 不受影响；numpy 值可序列化
- stdout/stderr 被管道读取时异步写出：edge 读不动时丢弃日志并计数，不再卡住推理
- 看门狗：卡住或线程退出时写线程堆栈文件、输出 `stall` 事件，持续 180 s 退出进程由 edge 自动重启（`watchdog.*`）
- Windows：直播推理运行期间禁止系统睡眠

## 2.2.0+enroll.1.live.2 — 2026-10-15（增量补丁 0023–0028）

见 `docs/实时推理观测与精度评估.md`。

### 观测（调用方打 log 的锚点）
- 每 10 s 一行 `[cv-metrics] {json}`：各机位帧率/丢帧/延迟、已识别比例、身份置信度分布、ID 切换、姿态来源、各阶段耗时；
  动作数/未归属/命中分布/发出延迟；推送通道状态；GC、内存
- 离散事件 `[cv-event] {json}`：live_start、models_ready、model_warning、capture_start、hoop_locked、identity_switch、
  timeline_gap、ws_connected/ws_disconnected、action_emitted、live_stop、*_error
- 同样内容写 `data/outputs/live/<课程id>/metrics.jsonl`；配置 `observability.*`（默认开启，`enabled: false` 关闭）
- 动作消息新增 `action_id`（与 action_emitted 事件一致），edge 可按它串联日志

### 精度评估
- `scripts/replay_live.py`：录像按实时速度走直播引擎，输出 events.jsonl / metrics.jsonl
- `scripts/eval_live_events.py`：对真值（v3 真值文档或 CSV）计算召回/精确率/类型混淆/时间误差/归属准确率/命中判定/角度覆盖率；
  无真值时汇总代理指标

### 性能（可选，默认关闭）
- `inference.feature_interval`：已确认身份的轨迹每 N 帧刷新人脸与衣着特征（CPU 冒烟：特征阶段 −61%，处理帧率约 ×2）
- `inference.rtmw_cameras`：只在指定机位跑 RTMW

### 修复
- 本地测试文件循环播放时不再发出虚假的 timeline_gap
- 单条消息发送异常不再断开推送通道（只有连接关闭才重连）
- 停止时不再等待取流线程的重连间隔
- 指标批次从模型就绪后开始计时

## 2.2.0+enroll.1.live.1 — 2026-10-09（增量补丁 0010–0022）

实时推理（`run_live_ws.py run`）按 `docs/实时推理审查.md` 修复。

### 可靠性
- edge 重启后自动重连推送通道（原来连接关闭后事件全部静默丢失）
- 环形缓冲区线程安全，动作检测/角度按快照读取（原来 `deque mutated during iteration` 使整轮检测被跳过）
- 动作与角度只用该学生自己的骨架，事件按真实学生归属（**结果变化**）
- 未匹配（漂移）轨迹不再估计姿态，与离线一致（**结果变化**）
- 启动时预加载并预热全部模型；无人员检测器即报错退出，不再降级为占位假人；可选模型缺失有明确提示
- 无本堂标定时角度为 null，不再静默使用离线 v2 标定（**结果变化**）

### 性能
- 跟踪器比对注册库矩阵化：20 人/帧 1034 → 31 ms/帧/机位，决策不变
- 动作检测：每轮约 5 s → 0.19 s（单遍序列、二分配对、按学生缓存、pose-only 滑窗记忆、15 s 检测窗口）
- 关键点以数组存入缓冲区：内存约 1/10，完整垃圾回收 645 → 26 ms
- RTMW 的 RGB 每帧转换一次；身体 ReID 按帧批量；YOLO 结果每帧一次拷回 CPU
- gallery 嵌入模型按需创建；直播不再保留无人读取的帧队列（约 1.5 GB）；推理线程按需唤醒

### 新配置（默认值保持原行为，除检测窗口外）
- `finalize.detect_window_ms`（15000，0 = 整环）、`finalize.detect_min_items`（150）
- 可选：`finalize.window_by_ms`、`capture.clock: monotonic`、`inference.min_interval_ms`、`ball.hoop_lock: median`、
  `identity.live_body_batch`（cameras.yaml，默认 true）

## 2.2.0+enroll.1 — 2026-10-08（增量包）

### 性能（注册）
- YOLO-Pose 只加载一次（2.2.0 每帧重载）；直播注册改为边采边算，采集结束只剩积压
- 无候选帧即时释放、候选帧不再复制；InsightFace / OSNet 每次注册只加载一套；人脸只跑检测 + 识别
- 全局注册表比对：每边只准备一次 + 矩阵乘，决策不变
- 分阶段耗时日志：models ready / capture done / scan done / gallery / registry

### 新增
- `run_live_ws.py enroll --video <本地视频>`（`--realtime` 模拟直播）：与直播注册同一流程与产物
- `scripts/bench_enroll.py`：注册压测（各环节耗时、端到端、GPU 利用率/显存按阶段）
- 文档：`docs/注册加速与压测.md`；单测：`tests/test_enroll_streaming.py`

### 兼容
- 产物格式不变；同一组帧上与 2.2.0 结果逐位一致；`enrollment.json` 仅视频注册时多 `source`/`video`
- `enroll --seconds` 不传时：直播仍为 45 秒，视频处理整段

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
