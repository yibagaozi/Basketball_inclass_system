#!/usr/bin/env python3
"""Re-extract enrollment face embeddings with body-crop InsightFace.

Fixes the common failure mode where full-frame face detection latches onto a
clearer bystander when the enrolled subject is small/shadowed.

Also avoids within-session face collisions: later locals will not keep samples
whose face is too similar to an earlier local's face centroid (enrollment
duplicate / group-photo contamination).

Usage:
  PYTHONPATH=. python scripts/reextract_enrollment_faces.py \\
    --manifest data/outputs/v2/gallery_manifest.json \\
    --manifest data/outputs/v3/gallery_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import data_path  # noqa: E402
from src.identity.embedders import cosine_sim, create_body_embedder, create_face_embedder  # noqa: E402
from src.identity.global_registry import (  # noqa: E402
    _centroid,
    _l2_normalize,
    consensus_filter_faces,
    create_face_lazy_gallery,
)
from src.identity.lineup_enroll import (  # noqa: E402
    _detect_persons_pose,
    _frame_lineup_candidates,
    _load_yolo,
)


def _upper_body_bbox(bbox: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [x1, y1, x2, y1 + 0.55 * (y2 - y1)]


def _embed_face(face_app, frame: np.ndarray, body_bbox: list[float], min_det: float = 0.50):
    """Return (emb, det_score, face_area) or None."""
    emb = face_app.embed(frame, _upper_body_bbox(body_bbox))
    if emb is None:
        return None
    # Re-run detector on crop for quality stats (det/area).
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _upper_body_bbox(body_bbox)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    pad = 0.15
    cx1 = max(0, int(x1 - pad * bw))
    cy1 = max(0, int(y1 - pad * bh))
    cx2 = min(w, int(x2 + pad * bw))
    cy2 = min(h, int(y2 + pad * bh))
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    scale = 1.0
    if max(crop.shape[:2]) < 320:
        scale = 320.0 / max(crop.shape[:2])
        crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
    faces = face_app._app.get(crop) if hasattr(face_app, "_app") else []
    if not faces:
        return (_l2_normalize(emb), 0.6, 800.0)
    f = max(faces, key=lambda x: float(x.det_score))
    ds = float(f.det_score)
    if ds < min_det:
        return None
    area = float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) / (scale * scale))
    return (_l2_normalize(emb), ds, area)


def _replace_faces(session_id: str, local_id: str, embs: list[np.ndarray]) -> int:
    d = data_path("enrollment", session_id, local_id)
    for p in d.glob("face_*.npy"):
        p.unlink()
    for i, e in enumerate(embs):
        np.save(d / f"face_{i:03d}.npy", _l2_normalize(e))
    # drop cache if any
    return len(embs)


def reextract_sequential(
    session_id: str,
    *,
    face,
    body,
    backend,
    kind: str,
    prior_face_thr: float = 0.48,
    min_face_area: float = 400.0,
    max_samples: int = 6,
) -> dict[str, int]:
    gallery = create_face_lazy_gallery(session_id)
    prior_cents: list[np.ndarray] = []
    counts: dict[str, int] = {}

    for local_id in sorted(gallery.list_students()):
        meta = gallery.load_student(local_id).get("meta") or {}
        bc = _centroid(list(gallery.load_student(local_id).get("body") or []))
        video = meta.get("source")
        if not video:
            counts[local_id] = 0
            continue
        cap = cv2.VideoCapture(str(video))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        t0, t1 = float(meta.get("t0", 0)), float(meta.get("t_end", 0))
        f0 = int(max(0, (t0 - 1.0) * fps))
        f1 = int((t1 + 1.0) * fps)
        step = max(1, int(0.25 * fps))
        frames = list(range(f0, f1 + 1, step))
        bf = int(meta.get("best_frame") or 0)
        if bf:
            frames.append(bf)

        cands: list[tuple] = []
        for fi in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok:
                continue
            dets = _detect_persons_pose(backend, kind, fr, 0.35)
            ranked: list[tuple] = []
            for d in dets:
                be = _l2_normalize(body.embed(fr, d["bbox"]))
                bs = float(cosine_sim(be, bc)) if bc is not None else 0.0
                got = _embed_face(face, fr, d["bbox"])
                if got is None:
                    continue
                emb, ds, area = got
                prior_pen = max(
                    (float(cosine_sim(emb, pc)) for pc in prior_cents),
                    default=0.0,
                )
                ranked.append((bs, prior_pen, emb, ds, area, fi))
            ranked.sort(key=lambda x: -x[0])
            chosen = None
            for item in ranked:
                if item[1] >= prior_face_thr:
                    continue
                if item[4] < min_face_area:
                    continue
                chosen = item
                break
            if chosen is not None:
                cands.append(chosen)
        cap.release()

        cands.sort(key=lambda x: (-x[0], -x[4]))
        selected: list[tuple] = []
        used_fi: list[int] = []
        for c in cands:
            if len(selected) >= max_samples + 4:
                break
            if any(abs(c[5] - u) < int(0.2 * fps) for u in used_fi):
                continue
            selected.append(c)
            used_fi.append(c[5])

        # quality rank then consensus
        selected.sort(key=lambda x: -(x[3] * np.log1p(x[4])))
        embs = [c[2] for c in selected]
        embs = consensus_filter_faces(embs)[:max_samples]
        n = _replace_faces(session_id, local_id, embs)
        counts[local_id] = n
        if embs:
            prior_cents.append(_centroid(embs))  # type: ignore[arg-type]
        print(f"  {local_id}: wrote {n} faces (candidates={len(cands)})")
    return counts


def reextract_lineup(
    session_id: str,
    *,
    face,
    backend,
    kind: str,
    min_face_area: float = 400.0,
    max_samples: int = 6,
) -> dict[str, int]:
    gallery = create_face_lazy_gallery(session_id)
    ids = sorted(gallery.list_students())
    if not ids:
        return {}
    meta0 = gallery.load_student(ids[0]).get("meta") or {}
    video = meta0.get("source")
    fi = int(meta0.get("frame_idx") or 0)
    if not video:
        return {}
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    expected = len(ids)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, frame0 = cap.read()
    if not ok:
        cap.release()
        return {}
    dets = _detect_persons_pose(backend, kind, frame0, 0.35)
    cands = _frame_lineup_candidates(frame0, dets, min_frontal=0.32)
    cands = sorted(cands, key=lambda x: x["cx"])[:expected]
    offsets = [0] + [int(round(df * fps)) for df in (-0.5, 0.5, -1.0, 1.0, 1.5, 2.0)]
    counts: dict[str, int] = {}
    for si, ref in enumerate(cands):
        local_id = ids[si] if si < len(ids) else f"stu_{si:02d}"
        items: list[tuple] = []
        for off in offsets:
            idx = max(0, fi + off)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            if not ok:
                continue
            dets = _detect_persons_pose(backend, kind, fr, 0.35)
            best, bestd = None, 1e9
            for d in dets:
                cx = 0.5 * (d["bbox"][0] + d["bbox"][2]) / fr.shape[1]
                dd = abs(cx - float(ref["cx"]))
                if dd < bestd:
                    bestd, best = dd, d
            if best is None or bestd > 0.10:
                continue
            got = _embed_face(face, fr, best["bbox"])
            if got is None or got[2] < min_face_area:
                continue
            items.append(got)
        items.sort(key=lambda x: -(x[1] * np.log1p(x[2])))
        embs = consensus_filter_faces([x[0] for x in items])[:max_samples]
        n = _replace_faces(session_id, local_id, embs)
        counts[local_id] = n
        print(f"  {local_id}: wrote {n} faces")
    # any leftover ids without a slot keep old faces untouched only if no cands
    for lid in ids[len(cands) :]:
        counts.setdefault(lid, 0)
    cap.release()
    return counts


def _detect_mode(session_id: str) -> str:
    g = create_face_lazy_gallery(session_id)
    ids = g.list_students()
    if not ids:
        return "unknown"
    mode = str((g.load_student(ids[0]).get("meta") or {}).get("enroll_mode") or "")
    if "lineup" in mode:
        return "lineup"
    return "sequential"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", action="append", default=[])
    ap.add_argument("--manifest", type=Path, action="append", default=[])
    ap.add_argument("--prior-face-thr", type=float, default=0.48)
    args = ap.parse_args()

    sessions: list[str] = list(args.session)
    for m in args.manifest:
        sessions.append(json.loads(m.read_text(encoding="utf-8"))["session_id"])
    if not sessions:
        ap.error("need --session or --manifest")

    face = create_face_embedder()
    body = create_body_embedder()
    backend, kind = _load_yolo()
    print(f"face={type(face).__name__} body={type(body).__name__} det={kind}")

    for sid in sessions:
        mode = _detect_mode(sid)
        print(f"\n=== {sid[:8]}… mode={mode} ===")
        if mode == "lineup":
            reextract_lineup(sid, face=face, backend=backend, kind=kind)
        else:
            reextract_sequential(
                sid,
                face=face,
                body=body,
                backend=backend,
                kind=kind,
                prior_face_thr=args.prior_face_thr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
