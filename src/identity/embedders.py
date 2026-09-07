"""Pluggable embedding backends with stub fallback for dev without GPU models."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np


def _deterministic_embed(seed: str, dim: int) -> np.ndarray:
    h = hashlib.sha256(seed.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-8
    return v


class FaceEmbedder(ABC):
    @abstractmethod
    def embed(self, image_bgr: np.ndarray, bbox: list[float] | None = None) -> np.ndarray | None:
        ...


class BodyEmbedder(ABC):
    @abstractmethod
    def embed(self, image_bgr: np.ndarray, bbox: list[float]) -> np.ndarray:
        ...


class StubFaceEmbedder(FaceEmbedder):
    """Dev stub; replace with InsightFace ArcFace in production."""

    dim = 512

    def embed(self, image_bgr: np.ndarray, bbox: list[float] | None = None) -> np.ndarray | None:
        if bbox is None:
            return None
        x1, y1, x2, y2 = map(int, bbox)
        if x2 <= x1 or y2 <= y1:
            return None
        patch = image_bgr[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        key = f"face_{patch.mean():.4f}_{patch.shape}"
        return _deterministic_embed(key, self.dim)


class StubBodyEmbedder(BodyEmbedder):
    """Dev stub; replace with CLIP-ReID in production."""

    dim = 768

    def embed(self, image_bgr: np.ndarray, bbox: list[float]) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        patch = image_bgr[y1:y2, x1:x2]
        key = f"body_{patch.mean():.4f}_{patch.shape}"
        return _deterministic_embed(key, self.dim)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def try_load_insightface(use_gpu: bool = True):
    try:
        from insightface.app import FaceAnalysis  # type: ignore

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                providers = ["CPUExecutionProvider"]
        except ImportError:
            providers = ["CPUExecutionProvider"]

        app = FaceAnalysis(name="buffalo_l", providers=providers)
        ctx_id = 0 if use_gpu and providers[0] == "CUDAExecutionProvider" else -1
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        return app
    except Exception:
        return None


class InsightFaceEmbedder(FaceEmbedder):
    def __init__(self, app=None):
        self._app = app or try_load_insightface()
        if self._app is None:
            raise RuntimeError("InsightFace not available")

    def embed(self, image_bgr: np.ndarray, bbox: list[float] | None = None) -> np.ndarray | None:
        """Embed face; when bbox is given, detect inside a padded crop (not full frame).

        Full-frame detection often picks a clearer bystander when the subject is
        small/shadowed — that contaminated cross-session matching.
        """
        crop = image_bgr
        offset_x, offset_y = 0, 0
        if bbox is not None:
            h, w = image_bgr.shape[:2]
            x1, y1, x2, y2 = [float(v) for v in bbox]
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            pad = 0.15
            x1 = max(0, int(x1 - pad * bw))
            y1 = max(0, int(y1 - pad * bh))
            x2 = min(w, int(x2 + pad * bw))
            y2 = min(h, int(y2 + pad * bh))
            if x2 <= x1 or y2 <= y1:
                return None
            crop = image_bgr[y1:y2, x1:x2]
            # Upscale tiny crops so the detector can fire reliably.
            ch, cw = crop.shape[:2]
            if max(ch, cw) < 320:
                scale = 320.0 / max(ch, cw)
                crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)))

        faces = self._app.get(crop)
        if not faces:
            return None
        if bbox is not None:
            # Prefer highest-det face whose center lies in the upper portion of the crop.
            scored = []
            ch = float(crop.shape[0])
            for f in faces:
                ds = float(f.det_score)
                fcy = (float(f.bbox[1]) + float(f.bbox[3])) / 2.0
                if fcy > 0.80 * ch:
                    continue
                scored.append((ds, f))
            if scored:
                scored.sort(key=lambda t: t[0], reverse=True)
                return scored[0][1].normed_embedding.astype(np.float32)
        return max(faces, key=lambda f: float(f.det_score)).normed_embedding.astype(np.float32)


def create_face_embedder(prefer_insightface: bool = True) -> FaceEmbedder:
    if prefer_insightface:
        try:
            from src.identity.backends import get_runtime_config
            rt = get_runtime_config()
            return InsightFaceEmbedder(try_load_insightface(use_gpu=rt["use_cuda"]))
        except Exception:
            try:
                return InsightFaceEmbedder()
            except Exception:
                pass
    return StubFaceEmbedder()


def create_body_embedder() -> BodyEmbedder:
    try:
        from src.identity.backends import get_runtime_config
        from src.identity.body_reid import OSNetBodyEmbedder

        rt = get_runtime_config()
        if rt.get("reid_path") and Path(rt["reid_path"]).exists():
            device = "cuda" if rt["use_cuda"] else "cpu"
            return OSNetBodyEmbedder(rt["reid_path"], device=device)
    except Exception:
        pass
    return StubBodyEmbedder()
