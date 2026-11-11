"""IoU + appearance tracker (body ReID / optional face / clothing color)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.identity.embedders import cosine_sim
from src.identity.enrollment import EnrollmentGallery
from src.identity.clothing_color import (
    N_PARTS,
    GalleryColorPrior,
    _as_parts,
    build_gallery_color_prior,
    clothing_color_match_batch,
    clothing_color_match_info,
    relative_color_scores,
)
from src.config import load_yaml

# Gallery folders are re-listed at most this often (was: on every match call).
_GALLERY_RELIST_SEC = 2.0


def _identity_cfg() -> dict:
    return load_yaml("cameras.yaml").get("identity", {})


def _clothing_color_weight() -> float:
    return float(_identity_cfg().get("clothing_color_weight", 0.40))


def _stack_rows(rows: list[np.ndarray]) -> np.ndarray | None:
    """(n, d) float32 stack, or None when sample shapes differ (use the scalar path)."""
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    flat = [np.asarray(r).reshape(-1) for r in rows]
    if len({f.shape for f in flat}) != 1:
        return None
    return np.stack(flat).astype(np.float32, copy=False)


class _GalleryIndex:
    """Gallery samples stacked per kind, built once per gallery state.

    Matching a detection against it is a few matrix ops instead of ~240 Python
    cosine / clothing-color calls per detection per frame.
    """

    def __init__(self, sids: list[str], datas: list[dict], prior: GalleryColorPrior | None):
        self.sids = sids
        self.datas = datas  # kept alive: identity of each cached dict is the staleness key
        self.prior = prior
        self.regular = True
        self.face, self.face_norm, self.face_starts, self.face_counts = self._embed_block(datas, "face")
        self.body, self.body_norm, self.body_starts, self.body_counts = self._embed_block(datas, "body")
        colors, ok, weights, counts = [], [], [], []
        for sid, data in zip(sids, datas):
            samples = data.get("color") or []
            counts.append(len(samples))
            w = prior.weights_for(sid) if prior is not None else None
            w = np.ones(N_PARTS, np.float32) if w is None else np.asarray(w, np.float32).reshape(-1)
            if w.size != N_PARTS:
                w = np.ones(N_PARTS, np.float32)
            for c in samples:
                parts = _as_parts(c)
                ok.append(parts is not None)
                colors.append(parts if parts is not None else np.zeros((N_PARTS, 3), np.float32))
                weights.append(w)
        self.color = np.stack(colors) if colors else np.zeros((0, N_PARTS, 3), np.float32)
        self.color_ok = np.asarray(ok, dtype=bool)
        self.color_w = np.stack(weights) if weights else np.zeros((0, N_PARTS), np.float32)
        self.color_counts = np.asarray(counts, dtype=np.int64)
        self.color_starts = np.concatenate([[0], np.cumsum(self.color_counts)[:-1]]).astype(np.int64)

    def _embed_block(self, datas: list[dict], kind: str):
        rows, counts = [], []
        for data in datas:
            samples = data.get(kind) or []
            counts.append(len(samples))
            rows.extend(samples)
        mat = _stack_rows(rows)
        if mat is None:
            self.regular = False
            mat = np.zeros((0, 0), np.float32)
        # per-sample norms exactly as cosine_sim computes them
        norms = np.asarray([np.linalg.norm(r) for r in rows], dtype=np.float32) if mat.size else np.zeros(0, np.float32)
        counts_arr = np.asarray(counts, dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(counts_arr)[:-1]]).astype(np.int64)
        return mat, norms, starts, counts_arr

    def is_current(self, sids: list[str], datas: list[dict], prior: GalleryColorPrior | None) -> bool:
        return (
            prior is self.prior
            and sids == self.sids
            and all(a is b for a, b in zip(datas, self.datas))
        )

    @staticmethod
    def _seg_max(values: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
        out = np.zeros(len(counts), dtype=np.float64)
        has = counts > 0
        if values.size and has.any():
            out[has] = np.maximum.reduceat(values, starts[has])
        return out

    def _cos_max(self, q: np.ndarray, mat: np.ndarray, norms: np.ndarray,
                 starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
        if mat.shape[0] == 0:
            return np.zeros(len(counts), dtype=np.float64)
        dots = mat @ q
        cos = dots / (np.linalg.norm(q) * norms + 1e-8)
        return self._seg_max(cos, starts, counts)

    def stats(self, face_emb, body_emb, color_desc, use_color: bool) -> list:
        """Per student, in gallery order: (sid, has_face_sims, fs, has_body_sims, bs,
        cs_abs | None, shoe_decisive) — what the scalar pass 1 computes."""
        n = len(self.sids)
        has_f = (self.face_counts > 0) if face_emb is not None else np.zeros(n, bool)
        has_b = (self.body_counts > 0) if body_emb is not None else np.zeros(n, bool)
        fs = self._cos_max(np.asarray(face_emb), self.face, self.face_norm, self.face_starts,
                           self.face_counts) if has_f.any() else np.zeros(n)
        bs = self._cos_max(np.asarray(body_emb), self.body, self.body_norm, self.body_starts,
                           self.body_counts) if has_b.any() else np.zeros(n)
        sims = decs = None
        if use_color and self.color.shape[0]:
            sims, decs = clothing_color_match_batch(color_desc, self.color, self.color_ok, self.color_w)
        out = []
        for i, sid in enumerate(self.sids):
            cs_abs, shoe = None, False
            if sims is not None and self.color_counts[i] > 0 and (has_f[i] or has_b[i]):
                lo = int(self.color_starts[i])
                seg = sims[lo:lo + int(self.color_counts[i])]
                j = int(np.argmax(seg))  # first max, like max(infos, key=similarity)
                cs_abs = float(seg[j])
                shoe = bool(decs[lo + j])
            out.append((sid, bool(has_f[i]), float(fs[i]) if has_f[i] else 0.0,
                        bool(has_b[i]), float(bs[i]) if has_b[i] else 0.0, cs_abs, shoe))
        return out


def _ema_vec(prev: np.ndarray | None, new: np.ndarray | None, alpha: float) -> np.ndarray | None:
    """StrongSORT-style EMA; alpha=1 disables smoothing (raw replace)."""
    if new is None:
        return prev
    n = np.asarray(new, dtype=np.float32)
    if prev is None or alpha >= 0.999:
        return n.copy()
    a = float(np.clip(alpha, 0.05, 0.95))
    out = (1.0 - a) * np.asarray(prev, dtype=np.float32) + a * n
    if out.ndim == 1 and out.size > 8:
        out = out / (float(np.linalg.norm(out)) + 1e-8)
    return out


@dataclass
class Track:
    track_id: int
    bbox: list[float]
    student_id: str | None = None
    face_emb: np.ndarray | None = None
    body_emb: np.ndarray | None = None
    color_desc: np.ndarray | None = None
    hits: int = 0
    age: int = 0
    face_sim: float = 0.0
    body_sim: float = 0.0
    alpha: float = 0.0
    identity_confidence: str = "high"
    gallery_cost: float = 1.0
    # Sticky identity: keep last matched student_id across brief ReID failures
    sticky_student_id: str | None = None
    sticky_ttl: int = 0
    # Consecutive "better" conflicting gallery hits required before overwrite
    sticky_switch_streak: int = 0


@dataclass
class FaceBodyTracker:
    gallery: EnrollmentGallery
    iou_weight: float = 0.45
    id_weight: float = 0.55
    match_threshold: float = 0.55
    max_age: int = 30
    _tracks: list[Track] = field(default_factory=list)
    _next_id: int = 1
    _color_prior: GalleryColorPrior | None = field(default=None, repr=False)
    _index: Any = field(default=None, repr=False)
    _sids: list = field(default_factory=list, repr=False)
    _sids_at: float = field(default=-1e9, repr=False)

    def _ensure_color_prior(self) -> GalleryColorPrior | None:
        """Build once: part weights that maximize inter-person clothing separation."""
        if self._color_prior is not None:
            return self._color_prior
        student_colors: dict[str, list] = {}
        for sid in self.gallery.list_students():
            colors = self.gallery.load_student(sid).get("color") or []
            if colors:
                student_colors[sid] = list(colors)
        if len(student_colors) < 2:
            self._color_prior = None
            return None
        self._color_prior = build_gallery_color_prior(student_colors)
        return self._color_prior

    def _iou(self, a: list[float], b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter + 1e-8)

    def _gallery_index(self, prior: GalleryColorPrior | None) -> _GalleryIndex:
        now = time.monotonic()
        if now - self._sids_at >= _GALLERY_RELIST_SEC:
            self._sids = self.gallery.list_students()
            self._sids_at = now
        datas = [self.gallery.load_student(sid) for sid in self._sids]  # cached dicts
        idx = self._index
        if idx is None or not idx.is_current(self._sids, datas, prior):
            idx = self._index = _GalleryIndex(list(self._sids), datas, prior)
        return idx

    def _pass1_reference(self, face_emb, body_emb, color_desc, use_color: bool, prior) -> list:
        """Original per-sample pass 1 (scalar cosine / clothing calls). Fallback for
        galleries whose samples differ in shape, and the reference in tests."""
        out = []
        for sid in self.gallery.list_students():
            data = self.gallery.load_student(sid)
            face_sims = [cosine_sim(face_emb, f) for f in data["face"]] if face_emb is not None and data["face"] else []
            body_sims = [cosine_sim(body_emb, b) for b in data["body"]] if body_emb is not None and data["body"] else []
            fs = max(face_sims) if face_sims else 0.0
            bs = max(body_sims) if body_sims else 0.0
            cs_abs, shoe = None, False
            gallery_colors = data.get("color") or []
            if (face_sims or body_sims) and use_color and gallery_colors:
                part_w = prior.weights_for(sid) if prior is not None else None
                infos = [
                    clothing_color_match_info(color_desc, c, part_weights=part_w)
                    for c in gallery_colors
                ]
                best_info = max(infos, key=lambda x: float(x.get("similarity") or 0.0))
                cs_abs = float(best_info.get("similarity") or 0.0)
                shoe = bool(best_info and best_info.get("shoe_decisive"))
            out.append((sid, bool(face_sims), fs, bool(body_sims), bs, cs_abs, shoe))
        return out

    def _match_gallery(
        self,
        face_emb: np.ndarray | None,
        body_emb: np.ndarray | None,
        alpha: float,
        color_desc: np.ndarray | None = None,
        *,
        _reference: bool = False,
    ) -> tuple[str | None, float, float, str, float]:
        """Return (student_id, face_sim, body_sim, conf, cost)."""
        best_sid, best_cost = None, 1.0
        best_face, best_body = 0.0, 0.0
        second_cost = 1.0
        idcfg = _identity_cfg()
        match_mode = str(idcfg.get("match_mode", "body_color"))
        color_w = _clothing_color_weight() * float(
            idcfg.get("tracklet_color_weight_scale", 1.0)
        )
        # body_color: ignore face (g08 two-person winner)
        if match_mode == "body_color":
            face_emb = None
            alpha = 0.0
        elif match_mode == "face_body":
            color_desc = None
            color_w = 0.0
        prior = self._ensure_color_prior()
        use_relative = bool(idcfg.get("clothing_color_relative", True))
        rel_temp = float(idcfg.get("clothing_color_rel_temperature", 0.12))
        color_sep = 0.0
        if prior is not None and prior.part_weights is not None:
            color_sep = float(np.max(prior.part_weights) - np.min(prior.part_weights))

        # Pass 1: appearance base + absolute clothing sims (for relative transform)
        use_color = color_desc is not None and color_w > 0
        index = None if _reference else self._gallery_index(prior)
        if index is not None and index.regular:
            stats = index.stats(face_emb, body_emb, color_desc, use_color)
        else:
            stats = self._pass1_reference(face_emb, body_emb, color_desc, use_color, prior)
        cand: dict[str, dict] = {}
        abs_color: dict[str, float] = {}
        shoe_decisive_any = False
        for sid, has_face, fs, has_body, bs, cs_abs, shoe in stats:
            if not has_face and not has_body:
                continue
            a = float(alpha) if has_face else 0.0
            if match_mode == "face_body_color" and has_body:
                face_cap = float(idcfg.get("face_alpha_high", 0.20))
                if bs >= 0.70:
                    a = min(a, min(face_cap, 0.12))
                elif bs >= 0.62:
                    a = min(a, min(face_cap, 0.18))
                elif bs >= 0.55:
                    a = min(a, face_cap)
                if fs < 0.45:
                    a = min(a, 0.10)
                elif fs < 0.55:
                    a = min(a, 0.15)
            base_cost = a * (1 - fs) + (1 - a) * (1 - bs)
            if cs_abs is not None:
                abs_color[sid] = cs_abs
                if shoe:
                    shoe_decisive_any = True
            cand[sid] = {
                "fs": fs, "bs": bs, "a": a, "base": base_cost,
                "face_sims": has_face, "cs_abs": cs_abs,
                "shoe_decisive": bool(cs_abs is not None and shoe),
            }

        rel_color = (
            relative_color_scores(abs_color, temperature=rel_temp)
            if (use_relative and len(abs_color) >= 2)
            else {k: float(v) for k, v in abs_color.items()}
        )

        # Pass 2: fuse costs with (relative) clothing scores.
        # When top body matches are nearly tied (e.g. stu_01↔stu_03 ~0.80),
        # do NOT short-circuit on body alone — let clothing/shoes break the tie.
        body_scores = sorted((float(c["bs"]) for c in cand.values()), reverse=True)
        body_ambiguous = (
            len(body_scores) >= 2 and (body_scores[0] - body_scores[1]) < 0.085
        )
        for sid, c in cand.items():
            cost = float(c["base"])
            cs = rel_color.get(sid) if sid in rel_color else c["cs_abs"]
            if cs is not None and color_w > 0:
                cw = color_w
                # Larger gallery color separation / shoe veto → trust color more
                if color_sep >= 0.12 and float(cs) < 0.70:
                    cw = min(0.62, color_w + 0.12 + 0.20 * min(color_sep, 0.35))
                elif c.get("shoe_decisive") and float(cs) < 0.72:
                    cw = min(0.60, color_w + 0.12)
                # Relative mode: if winner margin is clear, lean harder on color
                if use_relative and len(rel_color) >= 2:
                    ranked = sorted(rel_color.values(), reverse=True)
                    if ranked[0] - ranked[1] >= 0.12:
                        cw = min(0.65, cw + 0.08)
                if body_ambiguous:
                    cw = min(0.72, max(cw, color_w + 0.18))
                cost = (1.0 - cw) * cost + cw * (1.0 - float(cs))
            bs, fs, a = float(c["bs"]), float(c["fs"]), float(c["a"])
            if (
                bs >= 0.72
                and not body_ambiguous
                and (not c["face_sims"] or fs >= 0.25 or a <= 0.25)
            ):
                body_cost = (1.0 - bs) * 0.85
                if cs is not None and float(cs) < 0.48:
                    cost = min(cost, 0.55 * cost + 0.45 * body_cost)
                elif cs is not None and color_sep >= 0.15 and float(cs) < 0.65:
                    cost = min(cost, 0.70 * cost + 0.30 * body_cost)
                else:
                    cost = min(cost, body_cost)
            if cost < best_cost:
                second_cost = best_cost
                best_cost, best_sid, best_face, best_body = cost, sid, fs, bs
            elif cost < second_cost:
                second_cost = cost

        conf = "high" if best_cost < 0.30 else ("medium" if best_cost < 0.45 else "low")
        thr = float(self.match_threshold)
        if prior is not None and color_sep >= 0.15:
            thr = min(0.62, thr + 0.05)
        if shoe_decisive_any:
            thr = min(0.64, thr + 0.02)
        if best_sid is None or best_cost > thr:
            return None, best_face, best_body, "low", best_cost
        margin = float(idcfg.get("ambiguity_margin", 0.04))
        if prior is not None:
            uniq = float(np.max(prior.uniqueness.get(best_sid, [0.0])))
            if uniq >= 0.85:
                margin = max(0.02, margin - 0.02)
        if second_cost - best_cost < margin and second_cost < 0.9:
            if not (best_body >= 0.68 and best_body - 0.08 >= (1.0 - second_cost)):
                return None, best_face, best_body, "low", best_cost
        if best_face < 0.25 and best_body < 0.38 and best_cost > 0.40:
            return None, best_face, best_body, "low", best_cost
        return best_sid, best_face, best_body, conf, best_cost

    def _apply_identity(
        self,
        track: Track,
        sid: str | None,
        fs: float,
        bs: float,
        conf: str,
        cost: float,
        alpha: float,
        sticky_ttl: int | None = None,
    ) -> None:
        """Assign gallery match or sticky last-known student_id.

        High/medium sticky IDs are hard to overwrite: a conflicting gallery hit
        must be clearly better (lower cost + high conf). Prevents brief body/color
        confusion during release from flipping track identity.
        """
        track.face_sim, track.body_sim = fs, bs
        track.alpha = alpha
        idcfg = _identity_cfg()
        protect = set(idcfg.get("sticky_protect_conf") or ["high", "medium", "sticky"])
        cost_margin = float(idcfg.get("sticky_overwrite_cost_margin", 0.12))
        if sticky_ttl is None:
            sticky_ttl = int(idcfg.get("sticky_ttl_frames", 240))

        if sid:
            prev = track.sticky_student_id or track.student_id
            prev_conf = track.identity_confidence or "low"
            need_streak = int(idcfg.get("sticky_switch_streak", 3))
            if (
                prev
                and prev != sid
                and prev_conf in protect
                and track.sticky_ttl > 0
            ):
                # Keep sticky unless new match is high-conf and clearly cheaper
                better = (
                    conf == "high"
                    and cost + cost_margin < float(track.gallery_cost)
                )
                if better:
                    track.sticky_switch_streak = int(track.sticky_switch_streak) + 1
                else:
                    track.sticky_switch_streak = 0
                if not better or track.sticky_switch_streak < need_streak:
                    track.student_id = prev
                    track.sticky_student_id = prev
                    track.sticky_ttl = max(track.sticky_ttl - 1, sticky_ttl // 3)
                    track.identity_confidence = "sticky"
                    # Keep prior gallery_cost (do not adopt the rejected match cost)
                    return
            track.gallery_cost = cost
            track.student_id = sid
            track.sticky_student_id = sid
            track.sticky_ttl = sticky_ttl
            track.sticky_switch_streak = 0
            track.identity_confidence = conf or "medium"
            return

        track.gallery_cost = cost
        track.sticky_switch_streak = 0
        # Sticky: keep identity across brief ReID failures (shooting / occlusion)
        if track.sticky_student_id and track.sticky_ttl > 0:
            track.student_id = track.sticky_student_id
            track.sticky_ttl -= 1
            track.identity_confidence = "sticky"
            return
        track.student_id = None
        track.identity_confidence = "low"

    def _enforce_exclusive_student_ids(self) -> None:
        """At most one live track may own each student_id (best gallery_cost wins)."""
        best_for_sid: dict[str, Track] = {}
        for t in self._tracks:
            if not t.student_id:
                continue
            prev = best_for_sid.get(t.student_id)
            if prev is None or t.gallery_cost < prev.gallery_cost - 1e-6:
                best_for_sid[t.student_id] = t
            elif abs(t.gallery_cost - prev.gallery_cost) < 1e-6 and t.hits > prev.hits:
                best_for_sid[t.student_id] = t
        winners = {id(t) for t in best_for_sid.values()}
        for t in self._tracks:
            if t.student_id and id(t) not in winners:
                t.student_id = None
                t.identity_confidence = "low"

    def update(
        self,
        detections: list[dict],
    ) -> list[Track]:
        """
        detections: [{bbox, face_emb?, body_emb?, alpha, color_desc?}]
        """
        for t in self._tracks:
            t.age += 1

        unmatched_dets = list(range(len(detections)))
        matched: list[tuple[Track, int]] = []
        idcfg = _identity_cfg()
        ema_a = float(idcfg.get("tracklet_ema_alpha", 0.30))

        for track in list(self._tracks):
            best_j, best_score = -1, -1.0
            for j in unmatched_dets:
                det = detections[j]
                iou = self._iou(track.bbox, det["bbox"])
                alpha = det.get("alpha", 0.0)
                fs, bs = 0.0, 0.0
                if track.face_emb is not None and det.get("face_emb") is not None:
                    fs = cosine_sim(track.face_emb, det["face_emb"])
                if track.body_emb is not None and det.get("body_emb") is not None:
                    bs = cosine_sim(track.body_emb, det["body_emb"])
                id_sim = alpha * fs + (1 - alpha) * bs
                score = self.iou_weight * iou + self.id_weight * id_sim
                # Prefer spatial continuity when embeddings are weak
                if iou > 0.35:
                    score = max(score, 0.3 * iou + 0.7 * score)
                if score > best_score:
                    best_score, best_j = score, j
            if best_j >= 0 and best_score > 0.22:
                det = detections[best_j]
                track.bbox = det["bbox"]
                track.hits += 1
                track.age = 0
                if det.get("face_emb") is not None:
                    track.face_emb = _ema_vec(track.face_emb, det["face_emb"], ema_a)
                if det.get("body_emb") is not None:
                    track.body_emb = _ema_vec(track.body_emb, det["body_emb"], ema_a)
                if det.get("color_desc") is not None:
                    track.color_desc = _ema_vec(
                        track.color_desc,
                        np.asarray(det["color_desc"], dtype=np.float32),
                        ema_a,
                    )
                color_e = track.color_desc if track.color_desc is not None else det.get("color_desc")
                sid, fs, bs, conf, cost = self._match_gallery(
                    track.face_emb, track.body_emb, det.get("alpha", 0.0),
                    color_desc=color_e,
                )
                self._apply_identity(
                    track, sid, fs, bs, conf, cost, det.get("alpha", 0.0),
                )
                matched.append((track, best_j))
                unmatched_dets.remove(best_j)

        for j in unmatched_dets:
            det = detections[j]
            t = Track(
                track_id=self._next_id,
                bbox=det["bbox"],
                face_emb=det.get("face_emb"),
                body_emb=det.get("body_emb"),
                color_desc=(
                    np.asarray(det["color_desc"], dtype=np.float32)
                    if det.get("color_desc") is not None else None
                ),
                hits=1,
                gallery_cost=1.0,
            )
            sid, fs, bs, conf, cost = self._match_gallery(
                det.get("face_emb"), det.get("body_emb"), det.get("alpha", 0.0),
                color_desc=det.get("color_desc"),
            )
            self._apply_identity(
                t, sid, fs, bs, conf, cost, det.get("alpha", 0.0),
            )
            self._next_id += 1
            self._tracks.append(t)

        self._tracks = [t for t in self._tracks if t.age <= self.max_age]
        self._enforce_exclusive_student_ids()
        # After exclusive, refresh sticky for winners (do not promote displaced IDs)
        default_ttl = int(_identity_cfg().get("sticky_ttl_frames", 240))
        for t in self._tracks:
            if t.student_id:
                t.sticky_student_id = t.student_id
                if t.identity_confidence != "sticky":
                    t.sticky_ttl = max(t.sticky_ttl, default_ttl)
                if not t.identity_confidence:
                    t.identity_confidence = "medium"
            elif t.sticky_ttl <= 0:
                t.sticky_student_id = None
        return self._tracks
