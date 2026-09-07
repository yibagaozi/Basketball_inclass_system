"""Cross-session global student identity via face embeddings.

Session-local galleries keep stu_XX + clothing/body for that day's kit.
Persistent correspondence across classes uses InsightFace face embeddings
stored under data/identity/.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.config import ROOT, data_path, load_yaml
from src.identity.embedders import cosine_sim
from src.identity.enrollment import EnrollmentGallery


REGISTRY_VERSION = 1
DEFAULT_ID_PREFIX = "stu_global"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(x)) + 1e-8
    return (x / n).astype(np.float32)


def _centroid(embs: list[np.ndarray]) -> np.ndarray | None:
    if not embs:
        return None
    stacked = np.stack([_l2_normalize(e) for e in embs], axis=0)
    return _l2_normalize(stacked.mean(axis=0))


def consensus_filter_faces(
    embs: list[np.ndarray],
    *,
    min_sim: float = 0.40,
    min_keep: int = 2,
) -> list[np.ndarray]:
    """Drop outlier face samples that disagree with the leave-one-out centroid."""
    if len(embs) <= 2:
        return [_l2_normalize(e) for e in embs]
    norms = [_l2_normalize(e) for e in embs]
    scores: list[tuple[float, int]] = []
    for i, e in enumerate(norms):
        others = [norms[j] for j in range(len(norms)) if j != i]
        c = _centroid(others)
        scores.append((float(cosine_sim(e, c)) if c is not None else 0.0, i))
    kept = [norms[i] for s, i in scores if s >= min_sim]
    if len(kept) < min_keep:
        scores.sort(key=lambda t: t[0], reverse=True)
        kept = [norms[i] for _, i in scores[: max(min_keep, 1)]]
    return kept


def mutual_best_mean(query: list[np.ndarray], gallery: list[np.ndarray]) -> float:
    """Mean of each side's best cross-sample cosine (symmetric)."""
    if not query or not gallery:
        return 0.0
    q = [_l2_normalize(e) for e in query]
    g = [_l2_normalize(e) for e in gallery]
    mba = float(np.mean([max(cosine_sim(a, b) for b in g) for a in q]))
    mbb = float(np.mean([max(cosine_sim(a, b) for a in q) for b in g]))
    return 0.5 * (mba + mbb)


def face_match_score(
    query: list[np.ndarray],
    gallery: list[np.ndarray],
    *,
    centroid_weight: float = 0.5,
) -> float:
    """Blend centroid cosine with mutual best-mean for multi-sample robustness."""
    q = consensus_filter_faces(query)
    g = consensus_filter_faces(gallery)
    cq, cg = _centroid(q), _centroid(g)
    if cq is None or cg is None:
        return 0.0
    c_sim = float(cosine_sim(cq, cg))
    m_sim = mutual_best_mean(q, g)
    w = float(np.clip(centroid_weight, 0.0, 1.0))
    return w * c_sim + (1.0 - w) * m_sim


def get_global_registry_config() -> dict[str, Any]:
    try:
        cam = load_yaml("cameras.yaml")
        cfg = dict((cam.get("identity") or {}).get("global_registry") or {})
    except Exception:
        cfg = {}
    cfg.setdefault("enabled", True)
    cfg.setdefault("registry_relpath", "identity/global_registry.json")
    cfg.setdefault("faces_relpath", "identity/faces")
    cfg.setdefault("sessions_relpath", "identity/sessions")
    cfg.setdefault("id_prefix", DEFAULT_ID_PREFIX)
    # Classroom cams are distant; after crop+consensus, same-person ~0.55–0.65,
    # hard negatives usually <0.45. Keep a margin against lookalikes.
    cfg.setdefault("face_match_threshold", 0.52)
    cfg.setdefault("ambiguity_margin", 0.05)
    cfg.setdefault("centroid_weight", 0.6)
    cfg.setdefault("consensus_min_sim", 0.40)
    cfg.setdefault("update_on_match", True)
    cfg.setdefault("max_face_samples_per_student", 24)
    return cfg


def registry_json_path(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or get_global_registry_config()
    return data_path(*str(cfg["registry_relpath"]).split("/"))


def faces_root(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or get_global_registry_config()
    p = data_path(*str(cfg["faces_relpath"]).split("/"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def session_link_path(session_id: str, cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or get_global_registry_config()
    root = data_path(*str(cfg["sessions_relpath"]).split("/"))
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{session_id}.json"


@dataclass
class MatchResult:
    global_id: str
    score: float
    margin: float
    is_new: bool
    second_best_id: str | None = None
    second_best_score: float = 0.0


class GlobalFaceRegistry:
    """Persistent face gallery keyed by stu_global_XX."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg or get_global_registry_config()
        self.path = registry_json_path(self.cfg)
        self.faces_dir = faces_root(self.cfg)
        self._data: dict[str, Any] = {
            "version": REGISTRY_VERSION,
            "updated_at": None,
            "next_index": 0,
            "id_prefix": str(self.cfg.get("id_prefix", DEFAULT_ID_PREFIX)),
            "students": [],
        }
        self._centroids: dict[str, np.ndarray] = {}
        self._sample_bank: dict[str, list[np.ndarray]] = {}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        self._centroids = {}
        self._sample_bank = {}
        for stu in self._data.get("students", []):
            gid = stu["global_id"]
            embs = self._load_face_embs(gid)
            filtered = consensus_filter_faces(
                embs, min_sim=float(self.cfg.get("consensus_min_sim", 0.40))
            )
            self._sample_bank[gid] = filtered
            c = _centroid(filtered)
            if c is not None:
                self._centroids[gid] = c

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data["updated_at"] = _utc_now()
        self._data["version"] = REGISTRY_VERSION
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _rel_to_root(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(ROOT.resolve()))
        except ValueError:
            return str(path)

    def list_global_ids(self) -> list[str]:
        return [s["global_id"] for s in self._data.get("students", [])]

    def get_student(self, global_id: str) -> dict[str, Any] | None:
        for s in self._data.get("students", []):
            if s["global_id"] == global_id:
                return s
        return None

    def _student_face_dir(self, global_id: str) -> Path:
        d = self.faces_dir / global_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_face_embs(self, global_id: str) -> list[np.ndarray]:
        d = self.faces_dir / global_id
        if not d.exists():
            return []
        return [np.load(p) for p in sorted(d.glob("face_*.npy"))]

    def _append_face_samples(
        self,
        global_id: str,
        embs: list[np.ndarray],
        *,
        max_samples: int | None = None,
    ) -> int:
        max_n = int(max_samples or self.cfg.get("max_face_samples_per_student", 24))
        d = self._student_face_dir(global_id)
        existing = sorted(d.glob("face_*.npy"))
        added = 0
        for emb in embs:
            if len(existing) + added >= max_n:
                break
            idx = len(existing) + added
            np.save(d / f"face_{idx:03d}.npy", _l2_normalize(emb))
            added += 1
        # Refresh centroid
        all_embs = self._load_face_embs(global_id)
        filtered = consensus_filter_faces(
            all_embs, min_sim=float(self.cfg.get("consensus_min_sim", 0.40))
        )
        self._sample_bank[global_id] = filtered
        c = _centroid(filtered)
        if c is not None:
            self._centroids[global_id] = c
        return added

    def match_face(
        self,
        query_embs: list[np.ndarray] | np.ndarray,
        *,
        threshold: float | None = None,
        ambiguity_margin: float | None = None,
    ) -> MatchResult | None:
        """Match query face(s) to registry. Returns None if query empty."""
        if isinstance(query_embs, np.ndarray):
            embs = [query_embs]
        else:
            embs = list(query_embs)
        q_filt = consensus_filter_faces(
            embs, min_sim=float(self.cfg.get("consensus_min_sim", 0.40))
        )
        if not q_filt:
            return None

        thr = float(threshold if threshold is not None else self.cfg["face_match_threshold"])
        margin = float(
            ambiguity_margin if ambiguity_margin is not None else self.cfg["ambiguity_margin"]
        )
        cw = float(self.cfg.get("centroid_weight", 0.5))

        scores: list[tuple[str, float]] = []
        for gid, gal in self._sample_bank.items():
            if not gal:
                continue
            scores.append((gid, face_match_score(q_filt, gal, centroid_weight=cw)))
        scores.sort(key=lambda x: x[1], reverse=True)

        if not scores:
            return None  # empty registry → caller allocates

        best_id, best = scores[0]
        second_id, second = (scores[1] if len(scores) > 1 else (None, 0.0))
        gap = float(best - second)
        if best >= thr and gap >= margin:
            return MatchResult(
                global_id=best_id,
                score=float(best),
                margin=gap,
                is_new=False,
                second_best_id=second_id,
                second_best_score=float(second),
            )
        return MatchResult(
            global_id="",  # signal: no confident match
            score=float(best),
            margin=gap,
            is_new=True,
            second_best_id=best_id,
            second_best_score=float(best),
        )

    def allocate_new(
        self,
        face_embs: list[np.ndarray],
        *,
        session_id: str | None = None,
        display_name: str | None = None,
        local_id: str | None = None,
    ) -> str:
        prefix = str(self._data.get("id_prefix") or self.cfg.get("id_prefix", DEFAULT_ID_PREFIX))
        idx = int(self._data.get("next_index", 0))
        gid = f"{prefix}_{idx:02d}"
        while self.get_student(gid) is not None:
            idx += 1
            gid = f"{prefix}_{idx:02d}"
        self._data["next_index"] = idx + 1
        now = _utc_now()
        record = {
            "global_id": gid,
            "display_name": display_name or gid,
            "created_at": now,
            "first_seen_session": session_id,
            "sessions": [session_id] if session_id else [],
            "local_ids": {session_id: local_id} if session_id and local_id else {},
            "n_face_samples": 0,
        }
        self._data.setdefault("students", []).append(record)
        n = self._append_face_samples(gid, face_embs)
        record["n_face_samples"] = len(self._load_face_embs(gid))
        if n == 0 and not face_embs:
            # still register id so enroll meta can link; centroid empty until faces arrive
            pass
        self.save()
        return gid

    def link_existing(
        self,
        global_id: str,
        face_embs: list[np.ndarray],
        *,
        session_id: str | None = None,
        local_id: str | None = None,
        update_faces: bool | None = None,
    ) -> None:
        stu = self.get_student(global_id)
        if stu is None:
            raise KeyError(global_id)
        do_update = self.cfg.get("update_on_match", True) if update_faces is None else update_faces
        if do_update and face_embs:
            self._append_face_samples(global_id, face_embs)
        if session_id:
            sessions = list(stu.get("sessions") or [])
            if session_id not in sessions:
                sessions.append(session_id)
            stu["sessions"] = sessions
            locs = dict(stu.get("local_ids") or {})
            if local_id:
                locs[session_id] = local_id
            stu["local_ids"] = locs
        stu["n_face_samples"] = len(self._load_face_embs(global_id))
        self.save()

    def resolve_or_create(
        self,
        face_embs: list[np.ndarray],
        *,
        session_id: str | None = None,
        local_id: str | None = None,
        display_name: str | None = None,
    ) -> MatchResult:
        """Match to existing global id or allocate a new one."""
        usable = _centroid(list(face_embs)) is not None
        matched = self.match_face(face_embs) if usable else None

        # Empty registry → match_face returns None even when query faces exist.
        if matched is None:
            gid = self.allocate_new(
                list(face_embs) if usable else [],
                session_id=session_id,
                display_name=display_name,
                local_id=local_id,
            )
            return MatchResult(global_id=gid, score=0.0, margin=0.0, is_new=True)

        if matched.is_new or not matched.global_id:
            gid = self.allocate_new(
                list(face_embs) if usable else [],
                session_id=session_id,
                display_name=display_name,
                local_id=local_id,
            )
            return MatchResult(
                global_id=gid,
                score=matched.score,
                margin=matched.margin,
                is_new=True,
                second_best_id=matched.second_best_id,
                second_best_score=matched.second_best_score,
            )

        self.link_existing(
            matched.global_id,
            list(face_embs) if usable else [],
            session_id=session_id,
            local_id=local_id,
        )
        return matched


def create_face_lazy_gallery(session_id: str) -> EnrollmentGallery:
    """Load enrollment npy without spinning up InsightFace/OSNet (link/validate)."""
    g = EnrollmentGallery.__new__(EnrollmentGallery)
    g.session_id = session_id
    g.root = data_path("enrollment", session_id)
    g.face_embedder = None  # type: ignore
    g.body_embedder = None  # type: ignore
    g._cache = {}
    return g


def link_session_enrollment_to_global(
    session_id: str,
    student_ids: list[str] | None = None,
    *,
    cfg: dict[str, Any] | None = None,
    gallery: EnrollmentGallery | None = None,
) -> dict[str, Any]:
    """After session gallery write: bind each local stu_XX to a global face id.

    - Face → persistent stu_global_XX (cross-class)
    - Body/color remain session-scoped under data/enrollment/<session>/<stu_XX>/
    """
    cfg = cfg or get_global_registry_config()
    if not cfg.get("enabled", True):
        return {"enabled": False, "session_id": session_id, "mappings": []}

    gallery = gallery or create_face_lazy_gallery(session_id)
    ids = student_ids if student_ids is not None else gallery.list_students()
    registry = GlobalFaceRegistry(cfg)

    mappings: list[dict[str, Any]] = []
    for local_id in ids:
        data = gallery.load_student(local_id)
        faces = list(data.get("face") or [])
        result = registry.resolve_or_create(
            faces,
            session_id=session_id,
            local_id=local_id,
            display_name=str((data.get("meta") or {}).get("display_name") or local_id),
        )
        meta = dict(data.get("meta") or {})
        meta["global_id"] = result.global_id
        meta["global_match"] = {
            "score": round(float(result.score), 4),
            "margin": round(float(result.margin), 4),
            "is_new": bool(result.is_new),
            "second_best_id": result.second_best_id,
            "second_best_score": round(float(result.second_best_score), 4),
        }
        gallery.save_meta(local_id, meta.get("display_name", local_id), meta)

        mappings.append({
            "local_id": local_id,
            "global_id": result.global_id,
            "match_score": round(float(result.score), 4),
            "match_margin": round(float(result.margin), 4),
            "is_new": bool(result.is_new),
            "n_face_samples": len(faces),
            "n_body_samples": len(data.get("body") or []),
            "n_color_samples": len(data.get("color") or []),
            "has_clothing": bool(data.get("color")),
        })

    link_doc = {
        "schema": "session_identity_link/v1",
        "session_id": session_id,
        "updated_at": _utc_now(),
        "registry_path": GlobalFaceRegistry._rel_to_root(registry.path),
        "enrollment_root": GlobalFaceRegistry._rel_to_root(gallery.root),
        "mappings": mappings,
        "notes": (
            "local_id (stu_XX) is session-scoped for tracking/clothing; "
            "global_id is face-stable across classes."
        ),
    }
    out = session_link_path(session_id, cfg)
    out.write_text(json.dumps(link_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    link_doc["link_path"] = str(out)
    return link_doc
