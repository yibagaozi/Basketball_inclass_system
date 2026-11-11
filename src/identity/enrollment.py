"""Enrollment gallery management."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import data_path
from src.identity.embedders import BodyEmbedder, FaceEmbedder, create_body_embedder, create_face_embedder


_UNSET = object()


class EnrollmentGallery:
    """Per-session gallery (npy samples on disk).

    The embedders are only needed to *add* samples; matching (live tracker,
    offline perception) only reads npy files. They are therefore created on
    first use: before, every gallery opened for matching loaded its own
    InsightFace + OSNet that nobody used, on top of the caller's own copies.
    """

    def __init__(
        self,
        session_id: str,
        face_embedder: FaceEmbedder | None = None,
        body_embedder: BodyEmbedder | None = None,
    ):
        self.session_id = session_id
        self.root = data_path("enrollment", session_id)
        self._face_embedder = face_embedder if face_embedder is not None else _UNSET
        self._body_embedder = body_embedder if body_embedder is not None else _UNSET
        self._cache: dict[str, dict] = {}

    # Galleries built via __new__ (global_registry.create_face_lazy_gallery) assign
    # these attributes directly; an explicit None stays None, as before.
    @property
    def face_embedder(self) -> FaceEmbedder:
        fe = getattr(self, "_face_embedder", _UNSET)
        if fe is _UNSET:
            fe = self._face_embedder = create_face_embedder()
        return fe

    @face_embedder.setter
    def face_embedder(self, value: FaceEmbedder | None) -> None:
        self._face_embedder = value

    @property
    def body_embedder(self) -> BodyEmbedder:
        be = getattr(self, "_body_embedder", _UNSET)
        if be is _UNSET:
            be = self._body_embedder = create_body_embedder()
        return be

    @body_embedder.setter
    def body_embedder(self, value: BodyEmbedder | None) -> None:
        self._body_embedder = value

    def student_dir(self, student_id: str) -> Path:
        d = self.root / student_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def add_face_sample(self, student_id: str, image_bgr, face_bbox: list[float]) -> bool:
        emb = self.face_embedder.embed(image_bgr, face_bbox)
        if emb is None:
            return False
        d = self.student_dir(student_id)
        idx = len(list(d.glob("face_*.npy")))
        np.save(d / f"face_{idx:03d}.npy", emb)
        self._cache.pop(student_id, None)
        return True

    def add_body_sample(self, student_id: str, image_bgr, body_bbox: list[float]) -> bool:
        emb = self.body_embedder.embed(image_bgr, body_bbox)
        d = self.student_dir(student_id)
        idx = len(list(d.glob("body_*.npy")))
        np.save(d / f"body_{idx:03d}.npy", emb)
        self._cache.pop(student_id, None)
        return True

    def add_clothing_color_sample(
        self,
        student_id: str,
        image_bgr,
        body_bbox: list[float],
        keypoints=None,
    ) -> bool:
        from src.identity.clothing_color import extract_clothing_color

        desc = extract_clothing_color(image_bgr, body_bbox, keypoints=keypoints)
        d = self.student_dir(student_id)
        idx = len(list(d.glob("color_*.npy")))
        np.save(d / f"color_{idx:03d}.npy", desc)
        self._cache.pop(student_id, None)
        return True

    def load_student(self, student_id: str) -> dict:
        if student_id in self._cache:
            return self._cache[student_id]
        d = self.root / student_id
        faces = [np.load(p) for p in sorted(d.glob("face_*.npy"))]
        bodies = [np.load(p) for p in sorted(d.glob("body_*.npy"))]
        colors = [np.load(p) for p in sorted(d.glob("color_*.npy"))]
        meta_path = d / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        data = {"face": faces, "body": bodies, "color": colors, "meta": meta}
        self._cache[student_id] = data
        return data

    def list_students(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def save_meta(self, student_id: str, display_name: str, extra: dict | None = None) -> None:
        d = self.student_dir(student_id)
        meta = {"display_name": display_name, **(extra or {})}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
