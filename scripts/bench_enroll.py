#!/usr/bin/env python3
"""注册（人脸采集）压测：各环节耗时 + GPU 使用情况。

数据源（二选一）
  --video PATH      本地视频（推荐真实课堂注册录像，如 30 人、2 分钟）
  --synthetic N     合成 N 名学生依次走近镜头（无录像时用；人物取自 ultralytics 自带示例图，
                    先写成 data/bench/synthetic_N.mp4 再走同一条视频路径）

阶段
  env    torch / CUDA / 显卡架构是否被支持、onnxruntime、驱动
  load   YOLO-Pose / OSNet / InsightFace 加载耗时；InsightFace 各模型实际生效的 provider
  micro  单帧耗时 p50/p95：YOLO（缓存）、旧版每帧重载（--compare-reload）、OSNet、衣着颜色、人脸
  e2e    完整注册（与 run_live_ws.py enroll --video 同一流程）：边采边算 + 聚类 + 写库 + 全局注册表；
         --realtime 按原帧率回放模拟直播，统计“采集结束后还要等多久”

GPU：后台 nvidia-smi 每 --gpu-interval 秒采样（利用率/显存/功耗/温度，驱动支持时含编解码器），
按阶段汇总；另记 torch 峰值显存（不含 onnxruntime）。
输出：控制台汇总 + data/bench/<run_id>/report.json + gpu_samples.csv

不污染正式数据：注册写 data/enrollment/bench_<run_id>_<n>（结束删除，--keep 保留），
全局注册表用正式库的拷贝（--registry empty 用空库）。

示例
  python scripts/bench_enroll.py --video D:/videos/enroll_30p.mp4 --expected-persons 30
  python scripts/bench_enroll.py --video D:/videos/enroll_30p.mp4 --realtime
  python scripts/bench_enroll.py --synthetic 30 --realtime --repeat 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.config import DATA, data_path  # noqa: E402

# --------------------------------------------------------------------------- GPU sampling

_GPU_FIELDS_FULL = [
    "utilization.gpu", "utilization.memory", "utilization.encoder", "utilization.decoder",
    "memory.used", "memory.total", "power.draw", "temperature.gpu",
]
_GPU_FIELDS_BASIC = ["utilization.gpu", "memory.used", "memory.total", "power.draw", "temperature.gpu"]


def _find_nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    for cand in (r"C:\Windows\System32\nvidia-smi.exe",
                 r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"):
        if os.path.isfile(cand):
            return cand
    return None


def _num(s: str) -> float | None:
    try:
        return float(s.strip())
    except ValueError:
        return None  # "[N/A]", "[Not Supported]"


class GpuSampler:
    """Samples one GPU every ``interval`` s (nvidia-smi loop, pynvml fallback), tagged by phase."""

    def __init__(self, interval: float = 0.5, index: int = 0):
        self.interval = interval
        self.index = index
        self.phase = "idle"
        self.fields: list[str] = []
        self.samples: list[dict[str, Any]] = []
        self.backend = "none"
        self.note = ""
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._t0 = time.time()

    def start(self) -> None:
        exe = _find_nvidia_smi()
        if exe:
            for fields in (_GPU_FIELDS_FULL, _GPU_FIELDS_BASIC):
                probe = subprocess.run(
                    [exe, "-i", str(self.index), f"--query-gpu={','.join(fields)}",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=15,
                )
                if probe.returncode == 0 and probe.stdout.strip():
                    self.fields = fields
                    break
            if self.fields:
                self._proc = subprocess.Popen(
                    [exe, "-i", str(self.index), f"--query-gpu={','.join(self.fields)}",
                     "--format=csv,noheader,nounits", f"--loop-ms={max(100, int(self.interval * 1000))}"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
                )
                self.backend = "nvidia-smi"
                threading.Thread(target=self._read_smi, daemon=True).start()
                return
            self.note = "nvidia-smi 查询失败"
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
        except Exception:
            self.note = self.note or "未找到 nvidia-smi / pynvml，跳过 GPU 采样"
            return
        self.fields = _GPU_FIELDS_BASIC
        self.backend = "pynvml"
        threading.Thread(target=self._poll_nvml, args=(pynvml, handle), daemon=True).start()

    def _add(self, values: list[float | None]) -> None:
        row = {"t": round(time.time() - self._t0, 2), "phase": self.phase}
        row.update(dict(zip(self.fields, values)))
        self.samples.append(row)

    def _read_smi(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            parts = line.strip().split(",")
            if len(parts) == len(self.fields):
                self._add([_num(p) for p in parts])

    def _poll_nvml(self, pynvml, handle) -> None:
        while not self._stop.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                self._add([util.gpu, mem.used / 2**20, mem.total / 2**20, power, temp])
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def summary(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for phase in dict.fromkeys(s["phase"] for s in self.samples):
            rows = [s for s in self.samples if s["phase"] == phase]

            def col(name: str) -> list[float]:
                return [r[name] for r in rows if r.get(name) is not None]

            util, mem, power = col("utilization.gpu"), col("memory.used"), col("power.draw")
            enc, dec = col("utilization.encoder"), col("utilization.decoder")
            out[phase] = {
                "samples": len(rows),
                "util_mean": round(float(np.mean(util)), 1) if util else None,
                "util_max": max(util) if util else None,
                "mem_used_max_mb": max(mem) if mem else None,
                "power_mean_w": round(float(np.mean(power)), 1) if power else None,
                "encoder_mean": round(float(np.mean(enc)), 1) if enc else None,
                "decoder_mean": round(float(np.mean(dec)), 1) if dec else None,
            }
        return out


# --------------------------------------------------------------------------- helpers


def _stats_ms(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs) * 1000.0
    return {"n": len(xs), "p50": round(float(np.percentile(a, 50)), 1),
            "p95": round(float(np.percentile(a, 95)), 1), "mean": round(float(a.mean()), 1)}


def _peak_rss_gb() -> float | None:
    try:
        import psutil

        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", None)  # Windows
        if peak:
            return round(peak / 1e9, 2)
    except Exception:
        pass
    try:
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)  # Linux: KiB
    except Exception:
        return None


def _torch_peak_mb(reset: bool = False) -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        if reset:
            torch.cuda.reset_peak_memory_stats()
            return None
        return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except Exception:
        return None


def _mmss(s: float | None) -> str:
    return "-" if s is None else f"{s:.1f}s"


def _pad(value: Any, width: int) -> str:
    """Left-align for terminals: CJK characters take two columns."""
    import unicodedata

    text = "-" if value is None else str(value)
    used = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(1, width - used)


# --------------------------------------------------------------------------- env


def env_report() -> dict[str, Any]:
    import src

    info: dict[str, Any] = {
        "algo_version": getattr(src, "__version__", None),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "opencv": cv2.__version__,
    }
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    try:
        import torch

        t: dict[str, Any] = {
            "version": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            arch = f"sm_{major}{minor}"
            arch_list = torch.cuda.get_arch_list()
            t.update(device=torch.cuda.get_device_name(0), arch=arch, arch_list=arch_list,
                     arch_supported=arch in arch_list)
            try:
                x = torch.randn(1, 3, 64, 64, device="cuda")
                w = torch.randn(8, 3, 3, 3, device="cuda")
                float(torch.nn.functional.conv2d(x, w).sum())
                t["kernel_test"] = "ok"
            except Exception as e:  # e.g. "no kernel image is available"
                t["kernel_test"] = f"FAIL: {e}"
        info["torch"] = t
    except Exception as e:
        info["torch"] = {"error": str(e)}
    try:
        import onnxruntime as ort

        info["onnxruntime"] = {"version": ort.__version__, "available_providers": ort.get_available_providers()}
    except Exception as e:
        info["onnxruntime"] = {"error": str(e)}
    for mod in ("ultralytics", "insightface"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    exe = _find_nvidia_smi()
    if exe:
        try:
            q = subprocess.run([exe, "--query-gpu=name,driver_version,memory.total",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
            info["nvidia_smi"] = q.stdout.strip().splitlines()
        except Exception as e:
            info["nvidia_smi"] = f"error: {e}"
    return info


# --------------------------------------------------------------------------- synthetic source

# Person boxes in ultralytics/assets (bus.jpg x3, zidane.jpg x1), full-body and mostly frontal.
_SYNTH_PEOPLE = [
    ("bus.jpg", (51.1, 397.3, 243.9, 905.8)),
    ("bus.jpg", (223.1, 406.2, 345.3, 860.7)),
    ("bus.jpg", (668.7, 387.4, 810.0, 889.1)),
    ("zidane.jpg", (746.0, 39.1, 1145.9, 709.9)),
]


def make_synthetic_video(n_students: int, out: Path, *, fps: float = 15.0) -> Path:
    """N students walk up one by one (3 s each, 1.6 s gap); flips/hue make 4 people look like N."""
    if out.exists():
        return out
    import ultralytics

    assets = Path(ultralytics.__file__).parent / "assets"
    images = {name: cv2.imread(str(assets / name)) for name in {p for p, _ in _SYNTH_PEOPLE}}
    if any(v is None for v in images.values()):
        raise SystemExit(f"找不到 ultralytics 示例图（{assets}），请改用 --video")
    width, height, present, gap = 1920, 1080, 3.0, 1.6
    period = int(round((present + gap) * fps))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    blank = np.full((height, width, 3), 128, np.uint8)
    for i in range(n_students * period):
        k, j = divmod(i, period)
        if j >= present * fps:
            writer.write(blank)
            continue
        name, (x1, y1, x2, y2) = _SYNTH_PEOPLE[k % len(_SYNTH_PEOPLE)]
        prog = j / (present * fps)
        zoom = 0.85 + 0.20 * prog  # walking closer
        crop_h = (y2 - y1) / (0.75 * zoom)
        crop_w = crop_h * width / height
        x0 = 0.5 * (x1 + x2) + 0.02 * np.sin(2 * np.pi * prog) * crop_w - crop_w / 2
        y0 = 0.5 * (y1 + y2) - crop_h / 2
        s = height / crop_h
        m = np.float32([[s, 0, -s * x0], [0, s, -s * y0]])
        fr = cv2.warpAffine(images[name], m, (width, height), borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(128, 128, 128))
        if (k // 4) % 2:
            fr = np.ascontiguousarray(fr[:, ::-1])
        hue = ((k // 8) * 40) % 180
        if hue:
            hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int32) + hue) % 180).astype(np.uint8)
            fr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        writer.write(fr)
    writer.release()
    tmp.replace(out)
    return out


# --------------------------------------------------------------------------- phases


def load_models() -> tuple[dict[str, Any], dict[str, Any]]:
    import src.identity.sequential_enroll as se
    from src.identity.embedders import create_body_embedder, create_face_embedder

    res: dict[str, Any] = {}
    t = time.perf_counter()
    model = se._get_raw_pose_model()
    res["yolo_load_s"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    model.predict(np.zeros((1080, 1920, 3), np.uint8), verbose=False)
    res["yolo_warmup_s"] = round(time.perf_counter() - t, 2)
    res["yolo_device"] = str(getattr(model, "device", "?"))

    t = time.perf_counter()
    body = create_body_embedder()
    res["osnet_load_s"] = round(time.perf_counter() - t, 2)
    res["osnet"] = f"{type(body).__name__} ({getattr(body, 'device', '-')})"

    t = time.perf_counter()
    face = create_face_embedder()
    res["face_load_s"] = round(time.perf_counter() - t, 2)
    res["face"] = type(face).__name__
    app = getattr(face, "_app", None)
    providers = {}
    for task, m in (getattr(app, "models", None) or {}).items():
        sess = getattr(m, "session", None)
        try:
            providers[task] = sess.get_providers()[0] if sess is not None else "?"
        except Exception:
            providers[task] = "?"
    res["face_providers"] = providers
    backend, kind = se._load_yolo()
    return res, {"backend": backend, "kind": kind, "body": body, "face": face}


class _Enough(Exception):
    pass


def micro(video: Path, models: dict[str, Any], args) -> dict[str, Any]:
    import src.identity.sequential_enroll as se
    from src.identity.clothing_color import extract_clothing_color
    from src.identity.perception import _estimate_face_bbox
    from src.streaming.live_identity import collect_video_frames

    t_yolo, t_body, t_color, t_face, hits = [], [], [], [], 0
    person_frames: list[np.ndarray] = []

    def on_frame(i: int, t: float, frame: np.ndarray) -> None:
        nonlocal hits
        t0 = time.perf_counter()
        dets = se._detect_persons_pose(models["backend"], models["kind"], frame, 0.35)
        t_yolo.append(time.perf_counter() - t0)
        if dets:
            d = max(dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
            t0 = time.perf_counter()
            models["body"].embed(frame, d["bbox"])
            t_body.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            extract_clothing_color(frame, d["bbox"], keypoints=d.get("keypoints"))
            t_color.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            hits += models["face"].embed(frame, _estimate_face_bbox(d["bbox"], frame.shape)) is not None
            t_face.append(time.perf_counter() - t0)
            if len(person_frames) < max(1, args.compare_reload):
                person_frames.append(frame)
        if len(t_body) >= args.micro_frames or len(t_yolo) >= 4 * args.micro_frames:
            raise _Enough

    try:
        collect_video_frames(video, on_frame=on_frame, sample_hz=args.sample_hz)
    except _Enough:
        pass
    res: dict[str, Any] = {
        "yolo_cached": _stats_ms(t_yolo), "osnet": _stats_ms(t_body),
        "clothing_color": _stats_ms(t_color), "face": _stats_ms(t_face),
        "face_hit_rate": round(hits / len(t_face), 2) if t_face else None,
    }
    if args.compare_reload > 0 and person_frames:
        from ultralytics import YOLO

        from src.config import load_models_config, model_path

        path = str(model_path(load_models_config()["yolo_pose"]["path"]))
        t_old = []
        for k in range(args.compare_reload):
            frame = person_frames[k % len(person_frames)]
            t0 = time.perf_counter()
            YOLO(path).predict(frame, verbose=False, conf=0.35)  # v2.2.0 behaviour: new model per frame
            t_old.append(time.perf_counter() - t0)
        res["yolo_reload_per_frame_v220"] = _stats_ms(t_old)
    # Scoring capacity: every sampled frame runs YOLO; frames with a person also OSNet + color.
    if t_yolo:
        share = len(t_body) / len(t_yolo)
        per_frame = np.median(t_yolo) + share * (np.median(t_body or [0]) + np.median(t_color or [0]))
        res["scan_capacity_fps"] = round(1.0 / per_frame, 1)
        res["realtime_margin"] = round(res["scan_capacity_fps"] / min(args.sample_hz, _video_info(video)[0]), 2)
    return res


def _video_info(video: Path) -> tuple[float, int]:
    """(fps, frame count) with the same fps fallback as collect_video_frames."""
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return (fps if 0 < fps < 1000 else 25.0), n


def _timed(fn: Callable, acc: dict[str, float], key: str) -> Callable:
    def wrapper(*a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            acc[key] = acc.get(key, 0.0) + time.perf_counter() - t0
            acc[key + "_calls"] = acc.get(key + "_calls", 0) + 1
    return wrapper


def e2e(video: Path, args, run_id: str, k: int, sampler: GpuSampler, out_dir: Path) -> dict[str, Any]:
    import src.identity.global_registry as gr
    import src.identity.sequential_enroll as se
    from src.streaming.live_identity import collect_video_frames

    session = f"bench_{run_id}_{k}"
    acc: dict[str, float] = {}
    orig_cluster, orig_link = se.cluster_sequential_enrollments, gr.link_session_enrollment_to_global
    se.cluster_sequential_enrollments = _timed(orig_cluster, acc, "cluster")
    gr.link_session_enrollment_to_global = _timed(orig_link, acc, "registry")
    _torch_peak_mb(reset=True)
    try:
        scan = se.StreamingFrontalScan(max_queue=0 if args.realtime else 32, **se.live_scan_kwargs())
        sampler.phase = f"e2e#{k}:capture"
        t0 = time.time()
        try:
            n = collect_video_frames(video, on_frame=scan.submit, sample_hz=args.sample_hz,
                                     max_seconds=args.max_seconds, realtime=args.realtime)
        except BaseException:
            scan.close()
            raise
        t_cap = time.time()
        sampler.phase = f"e2e#{k}:scan_tail"
        samples = scan.finish()
        t_scan = time.time()
        sampler.phase = f"e2e#{k}:gallery"
        face = scan.face_embedder
        if face is not None:
            face.embed = _timed(face.embed, acc, "face")  # instance attribute shadows the method
        ids = se.enroll_sequential_from_samples(
            session, samples, expected_persons=args.expected_persons,
            preview_dir=out_dir / f"preview_{k}",
            face_embedder=face, body_embedder=scan.body_embedder,
        )
        t_end = time.time()
    finally:
        se.cluster_sequential_enrollments, gr.link_session_enrollment_to_global = orig_cluster, orig_link
        sampler.phase = "idle"
    face_calls = int(acc.get("face_calls", 0))
    scan_s = t_scan - t0 - (scan.models_ready_s or 0.0)
    return {
        "session": session,
        "frames": n,
        "candidates": len(samples),
        "persons": len(ids),
        "models_ready_s": None if scan.models_ready_s is None else round(scan.models_ready_s, 1),
        "capture_s": round(t_cap - t0, 1),
        "scan_tail_s": round(t_scan - t_cap, 1),
        "scan_fps": round(n / scan_s, 1) if scan_s > 0 else None,
        "gallery_registry_s": round(t_end - t_scan, 1),
        "cluster_s": round(acc.get("cluster", 0.0), 2),
        "face_calls": face_calls,
        "face_s": round(acc.get("face", 0.0), 1),
        "registry_s": round(acc.get("registry", 0.0), 2),
        "tail_after_capture_s": round(t_end - t_cap, 1),
        "total_s": round(t_end - t0, 1),
        "peak_rss_gb": _peak_rss_gb(),
        "torch_peak_mb": _torch_peak_mb(),
    }


def isolate_registry(run_id: str, mode: str) -> dict[str, Any]:
    """Point the global face registry at a bench copy (or an empty one) for this process."""
    import src.identity.global_registry as gr

    real = gr.get_global_registry_config()
    rel = f"bench/{run_id}/identity"
    cfg = dict(real, registry_relpath=f"{rel}/global_registry.json",
               faces_relpath=f"{rel}/faces", sessions_relpath=f"{rel}/sessions")
    n_ids = 0
    if mode == "copy":
        src_json = data_path(*str(real["registry_relpath"]).split("/"))
        src_faces = data_path(*str(real["faces_relpath"]).split("/"))
        if src_json.exists():
            dst = data_path(*cfg["registry_relpath"].split("/"))
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_json, dst)
            n_ids = len(json.loads(src_json.read_text(encoding="utf-8")).get("students", []))
        if src_faces.exists():
            shutil.copytree(src_faces, data_path(*cfg["faces_relpath"].split("/")), dirs_exist_ok=True)
    gr.get_global_registry_config = lambda: dict(cfg)
    return {"mode": mode, "global_ids": n_ids}


# --------------------------------------------------------------------------- report


def print_summary(rep: dict[str, Any]) -> None:
    env, load, mic = rep["env"], rep.get("load") or {}, rep.get("micro") or {}
    t = env.get("torch", {})
    print("\n==== 环境 ====")
    print(f"算法版本 {env.get('algo_version')}  python {env['python']}  CPU {env['cpu_count']} 核  内存 {env.get('ram_gb')} GB")
    if t.get("cuda_available"):
        print(f"torch {t['version']}  CUDA {t['cuda']}  GPU {t['device']} ({t['arch']})  "
              f"架构在 torch 编译列表中: {'是' if t['arch_supported'] else '否'}  CUDA 内核测试: {t['kernel_test']}")
    else:
        print(f"torch {t.get('version')}  CUDA 不可用（YOLO/OSNet 将在 CPU 上运行）")
    ort = env.get("onnxruntime", {})
    print(f"onnxruntime {ort.get('version')}  可用 providers {ort.get('available_providers')}")
    if load:
        print("\n==== 模型加载 ====")
        print(f"YOLO-Pose {load['yolo_load_s']}s + 预热 {load['yolo_warmup_s']}s ({load['yolo_device']}) | "
              f"OSNet {load['osnet_load_s']}s {load['osnet']} | InsightFace {load['face_load_s']}s {load['face']}")
        prov = load.get("face_providers") or {}
        print("InsightFace 实际 provider: " + (", ".join(f"{k}={v}" for k, v in prov.items()) or "-"))
        if any("CPU" in v for v in prov.values()):
            print("  !! 人脸模型在 CPU 上运行：检查 onnxruntime-gpu 与 CUDA/cuDNN DLL")
        if "Stub" in load["osnet"] or load["face"].startswith("Stub"):
            print("  !! 使用了 Stub 嵌入器：模型文件缺失，结果不代表真实性能")
    if mic:
        print("\n==== 单帧耗时 ms（p50 / p95）====")
        for key, label in (("yolo_cached", "YOLO(缓存)"), ("yolo_reload_per_frame_v220", "YOLO 每帧重载(2.2.0 旧逻辑)"),
                           ("osnet", "OSNet"), ("clothing_color", "衣着颜色"), ("face", "人脸")):
            s = mic.get(key)
            if s and s.get("n"):
                print(f"  {_pad(label, 28)}{s['p50']:>7} / {_pad(s['p95'], 8)}(n={s['n']})")
        if mic.get("face_hit_rate") is not None:
            print(f"  人脸检出率 {mic['face_hit_rate']:.0%}")
        if mic.get("scan_capacity_fps"):
            verdict = ("能实时跟上采样，采集结束后基本无积压" if mic["realtime_margin"] >= 1.2
                       else "跟不上采样，采集结束后会有积压")
            print(f"  打分能力约 {mic['scan_capacity_fps']} 帧/秒（采样的 {mic['realtime_margin']} 倍）→ {verdict}")
    for k, r in enumerate(rep.get("e2e") or []):
        print(f"\n==== 端到端 #{k + 1}（{'实时回放' if rep['args']['realtime'] else '尽快处理'}）====")
        print(f"  采样帧 {r['frames']}  候选 {r['candidates']} → {r['persons']} 人  模型就绪 {_mmss(r['models_ready_s'])}")
        print(f"  采集 {_mmss(r['capture_s'])}  打分积压 {_mmss(r['scan_tail_s'])}  写库+注册表 {_mmss(r['gallery_registry_s'])}"
              f"（人脸 {r['face_calls']} 次 {_mmss(r['face_s'])}，注册表 {_mmss(r['registry_s'])}）")
        rate = "" if rep["args"]["realtime"] else f"  平均打分 {r['scan_fps']} 帧/秒"
        print(f"  采集结束后还需 {_mmss(r['tail_after_capture_s'])}  总计 {_mmss(r['total_s'])}{rate}")
        torch_mem = "-" if r["torch_peak_mb"] is None else f"{r['torch_peak_mb']} MB"
        print(f"  峰值内存 {r['peak_rss_gb']} GB  torch 峰值显存 {torch_mem}")
    gpu = rep.get("gpu") or {}
    print(f"\n==== GPU（{gpu.get('backend')}{'，' + gpu['note'] if gpu.get('note') else ''}）====")
    phases = gpu.get("phases") or {}
    if phases:
        print("  " + _pad("阶段", 20) + _pad("利用率 均/峰 %", 16) + _pad("显存峰值 MB", 13)
              + _pad("功耗 W", 9) + "编码器 %")
        for name, s in phases.items():
            if name == "done":
                continue
            util = f"{s['util_mean']}/{s['util_max']}" if s["util_mean"] is not None else None
            print("  " + _pad(name, 20) + _pad(util, 16) + _pad(s["mem_used_max_mb"], 13)
                  + _pad(s["power_mean_w"], 9) + ("-" if s["encoder_mean"] is None else str(s["encoder_mean"])))
    print(f"\n报告：{rep['report_path']}")


def main() -> None:
    from src.streaming.live_config import load_live_config

    cfg = load_live_config()
    p = argparse.ArgumentParser(description="注册压测：性能 + GPU 使用情况")
    src_grp = p.add_mutually_exclusive_group(required=True)
    src_grp.add_argument("--video", type=Path, help="本地注册视频")
    src_grp.add_argument("--synthetic", type=int, metavar="N", help="合成 N 名学生的注册视频")
    p.add_argument("--sample-hz", type=float, default=float((cfg.get("identity") or {}).get("enroll_sample_hz") or 8))
    p.add_argument("--max-seconds", type=float, default=None, help="只处理视频前 N 秒")
    p.add_argument("--realtime", action="store_true", help="按原帧率回放，模拟直播采集")
    p.add_argument("--expected-persons", type=int, default=None)
    p.add_argument("--repeat", type=int, default=1, help="端到端重复次数")
    p.add_argument("--micro-frames", type=int, default=30, help="单帧耗时统计用的有人帧数")
    p.add_argument("--compare-reload", type=int, default=10, help="测 2.2.0 每帧重载 YOLO 的帧数（0 跳过）")
    p.add_argument("--skip-micro", action="store_true")
    p.add_argument("--skip-e2e", action="store_true")
    p.add_argument("--gpu-interval", type=float, default=0.5)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--registry", choices=("copy", "empty"), default="copy",
                   help="全局注册表：copy=正式库拷贝（贴近现场规模），empty=空库")
    p.add_argument("--keep", action="store_true", help="保留本次注册产物（gallery、注册表拷贝、缩略图）")
    args = p.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = DATA / "bench" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        print(f"生成合成注册视频（{args.synthetic} 人）…", flush=True)
        video = make_synthetic_video(args.synthetic, DATA / "bench" / f"synthetic_{args.synthetic}.mp4")
    else:
        video = args.video
        if not video.is_file():
            raise SystemExit(f"找不到视频文件：{video}")
    fps, n_frames = _video_info(video)
    rep: dict[str, Any] = {
        "run_id": run_id, "video": str(video), "video_fps": fps, "video_frames": n_frames,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }

    sampler = GpuSampler(args.gpu_interval, args.gpu_index)
    sampler.start()
    try:
        time.sleep(max(1.0, 2 * args.gpu_interval))  # idle baseline (e.g. NVENC from recording)
        rep["env"] = env_report()
        rep["registry"] = isolate_registry(run_id, args.registry)
        sampler.phase = "load"
        print("加载模型…", flush=True)
        rep["load"], models = load_models()
        if not args.skip_micro:
            sampler.phase = "micro"
            print("单帧耗时…", flush=True)
            rep["micro"] = micro(video, models, args)
        del models
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        rep["e2e"] = []
        if not args.skip_e2e:
            for k in range(1, max(1, args.repeat) + 1):
                print(f"端到端 #{k}…", flush=True)
                rep["e2e"].append(e2e(video, args, run_id, k, sampler, out_dir))
    finally:
        sampler.phase = "done"
        time.sleep(args.gpu_interval)
        sampler.stop()
        rep["gpu"] = {"backend": sampler.backend, "note": sampler.note, "fields": sampler.fields,
                      "phases": sampler.summary()}
        with open(out_dir / "gpu_samples.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["t", "phase", *sampler.fields])
            w.writeheader()
            w.writerows(sampler.samples)
        rep["report_path"] = str(out_dir / "report.json")
        (out_dir / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.keep:
            for r in rep.get("e2e") or []:
                shutil.rmtree(data_path("enrollment", r["session"]), ignore_errors=True)
            shutil.rmtree(out_dir / "identity", ignore_errors=True)
            for d in out_dir.glob("preview_*"):
                shutil.rmtree(d, ignore_errors=True)
    print_summary(rep)


if __name__ == "__main__":
    main()
