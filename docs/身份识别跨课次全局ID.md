# 身份识别：跨课次全局 ID（v2.1.0）

> 2026-09-07 · 人脸（InsightFace buffalo_l）负责跨课堂对应；衣着/身体特征仍为**当堂 session** 外观库。

## 设计要点

| 层级 | 主键 | 特征 | 作用域 |
|------|------|------|--------|
| 全局 | `stu_global_XX` | 人脸 embedding（ArcFace 512-D） | 跨课次 / 跨学期 |
| 课次 | `stu_XX`（注册顺序 / 并排左→右） | face + body(OSNet) + clothing HSV | **单次课** `data/enrollment/<session>/` |

跟踪与动作评测仍使用 session 内 `stu_XX`（保持 v2/v3 真值双射不变）。`meta.json` 与 session 链接 JSON 记录 `global_id` 对应关系。

**不做的事**：不把真值写进算法；不用衣着做跨天匹配（换装即失效）。

## 数据路径与 Schema

```
data/identity/
  global_registry.json          # 全局学生索引（无大数组；样本在 faces/）
  faces/stu_global_XX/face_*.npy
  sessions/<session_id>.json    # 本课 local↔global 映射 + 衣着样本计数

data/enrollment/<session_id>/stu_XX/
  face_*.npy / body_*.npy / color_*.npy / meta.json
  # meta.global_id / meta.global_match
```

示例（可入库）：

- `data/schema_examples/global_registry.example.json`
- `data/schema_examples/session_identity_link.example.json`

### `global_registry.json`（摘要）

```json
{
  "version": 1,
  "next_index": 6,
  "id_prefix": "stu_global",
  "students": [
    {
      "global_id": "stu_global_00",
      "sessions": ["<uuid>", "..."],
      "local_ids": {"<uuid>": "stu_00"},
      "n_face_samples": 8
    }
  ]
}
```

### `sessions/<session_id>.json`（摘要）

```json
{
  "schema": "session_identity_link/v1",
  "session_id": "<uuid>",
  "mappings": [
    {
      "local_id": "stu_00",
      "global_id": "stu_global_03",
      "match_score": 0.71,
      "is_new": false,
      "has_clothing": true
    }
  ]
}
```

## 算法（注册时）

1. 仍按 v2 顺序正面 / v3 并排正面写 session gallery（face/body/color）。
2. 人脸提取：在**人体/头肩框裁剪**内跑 InsightFace（避免全帧误选旁人清晰脸）。
3. 多样本先做 **consensus 过滤**（剔除与 leave-one-out 质心过远的离群样本），再算 L2 归一化质心。
4. 与全局库匹配分 = `centroid_weight * 质心余弦 + (1-centroid_weight) * mutual_best_mean`。
5. 若 `score ≥ face_match_threshold`（默认 **0.52**）且相对第二名 `margin ≥ ambiguity_margin`（默认 **0.05**）→ 复用该 `stu_global_XX`，并可追加人脸样本。
6. 否则分配新的 `stu_global_{next_index:02d}`。
7. 写入 `meta.global_id` 与 `data/identity/sessions/<session>.json`。

配置：`configs/cameras.yaml` → `identity.global_registry`。

已有 enrollment 若人脸被全帧旁人污染，可重提后再 link：

```bash
PYTHONPATH=. python scripts/reextract_enrollment_faces.py \
  --manifest data/outputs/v2/gallery_manifest.json \
  --manifest data/outputs/v3/gallery_manifest.json
PYTHONPATH=. python scripts/link_global_identity.py --reset-registry \
  --from-manifest data/outputs/v2/gallery_manifest.json
PYTHONPATH=. python scripts/link_global_identity.py \
  --from-manifest data/outputs/v3/gallery_manifest.json
```

## 与上一版差异（相对 `versions/v2.0.8`）

| 项 | v2.0.8 及以前 | v2.1.0 |
|----|---------------|--------|
| 跨课次对应 | 无（仅批跑内 gallery 拷贝） | 人脸全局注册表 |
| 衣着 | session gallery | 不变（session-scoped） |
| 跟踪主键 | `stu_XX` | 仍为 `stu_XX`（+ meta 挂 global） |
| 备份 | — | `versions/v2.0.8/` 变更前快照 |

## 下一堂课如何注册

```bash
conda activate basketball_classroom
./scripts/setup_gpu_env.sh   # 如需
cd /path/to/Basketball_inclass_system

# 正常批跑即可：group0 注册结束后会自动 link 全局库
PYTHONPATH=. python scripts/run_v3_testset.py --groups 0 --mode full
# 或 v2：
PYTHONPATH=. python scripts/run_v2_testset.py --groups 0 --mode full

# 仅把已有 enrollment 挂到全局库（不重跑视频）
PYTHONPATH=. python scripts/link_global_identity.py \
  --from-manifest data/outputs/v3/gallery_manifest.json

# 自洽性 / 跨课次重叠 / 重复脸诊断 → 写入 identity 结果目录
PYTHONPATH=. python scripts/validate_global_identity.py \
  --manifest-a data/outputs/v2/gallery_manifest.json \
  --manifest-b data/outputs/v3/gallery_manifest.json \
  --dump-identity-dirs \
  --out data/outputs/identity/validation_summary.json
# 产物：data/outputs/identity/、data/outputs/v2/identity/、data/outputs/v3/identity/
```

同一批学生换球衣再上课：人脸应命中已有 `stu_global_*`，当日 `body/color` 写入新 session 的 `stu_XX` 目录。

## 代码入口

- `src/identity/global_registry.py` — 注册表 / 匹配 / link
- `src/identity/embedders.py` — InsightFace 人体框裁剪提取
- `src/identity/sequential_enroll.py` / `lineup_enroll.py` — 注册末尾自动 link
- `scripts/reextract_enrollment_faces.py` — 已有 gallery 人脸重提
- `scripts/link_global_identity.py` / `validate_global_identity.py`
