"""Frozen copy of FaceBodyTracker._match_gallery as of 2.2.0 (cd1c380), used as the
reference in test_tracker_equivalence.py. Do not edit: it defines "same result"."""

from __future__ import annotations

import numpy as np

from src.identity.clothing_color import clothing_color_match_info, relative_color_scores
from src.identity.embedders import cosine_sim
from src.identity.tracker import _clothing_color_weight, _identity_cfg


def original_match_gallery(
    self,
    face_emb: np.ndarray | None,
    body_emb: np.ndarray | None,
    alpha: float,
    color_desc: np.ndarray | None = None,
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
    cand: dict[str, dict] = {}
    abs_color: dict[str, float] = {}
    shoe_decisive_any = False
    for sid in self.gallery.list_students():
        data = self.gallery.load_student(sid)
        face_sims = [cosine_sim(face_emb, f) for f in data["face"]] if face_emb is not None and data["face"] else []
        body_sims = [cosine_sim(body_emb, b) for b in data["body"]] if body_emb is not None and data["body"] else []
        fs = max(face_sims) if face_sims else 0.0
        bs = max(body_sims) if body_sims else 0.0
        if not face_sims and not body_sims:
            continue
        a = float(alpha) if face_sims else 0.0
        if match_mode == "face_body_color" and body_sims:
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
        cs_abs = None
        best_info = None
        gallery_colors = data.get("color") or []
        if color_desc is not None and gallery_colors and color_w > 0:
            part_w = prior.weights_for(sid) if prior is not None else None
            infos = [
                clothing_color_match_info(color_desc, c, part_weights=part_w)
                for c in gallery_colors
            ]
            best_info = max(infos, key=lambda x: float(x.get("similarity") or 0.0))
            cs_abs = float(best_info.get("similarity") or 0.0)
            abs_color[sid] = cs_abs
            if best_info.get("shoe_decisive"):
                shoe_decisive_any = True
        cand[sid] = {
            "fs": fs, "bs": bs, "a": a, "base": base_cost,
            "face_sims": bool(face_sims), "cs_abs": cs_abs,
            "shoe_decisive": bool(best_info and best_info.get("shoe_decisive")),
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
