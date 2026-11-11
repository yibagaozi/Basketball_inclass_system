"""Vectorized gallery matching must give the same decisions as the 2.2.0 scalar code."""

from __future__ import annotations

import types

import numpy as np
import pytest

import src.identity.tracker as trk
import tests._tracker_reference as ref
from src.identity.clothing_color import N_PARTS, _as_parts, clothing_color_match_batch, clothing_color_match_info
from src.identity.embedders import StubBodyEmbedder, StubFaceEmbedder
from src.identity.enrollment import EnrollmentGallery

TOL = 1e-6  # float32 matrix-vector vs per-sample dot rounding (~5e-8)


def _unit(rng, d=512, base=None, noise=0.0):
    v = rng.standard_normal(d).astype(np.float32) if base is None else (base + noise * rng.standard_normal(d)).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _color(rng, base=None, jitter=12.0):
    if base is None:
        c = np.stack([rng.uniform(0, 180, N_PARTS), rng.uniform(0, 255, N_PARTS), rng.uniform(0, 255, N_PARTS)], 1)
    else:
        c = base + rng.normal(0, jitter, base.shape)
        c[:, 0] = np.mod(c[:, 0], 180.0)
        c[:, 1:] = np.clip(c[:, 1:], 0, 255)
    c = c.astype(np.float32)
    c[rng.random(N_PARTS) < 0.15] = 0.0  # missing parts
    return c


def _make_gallery(tmp_path, monkeypatch, rng, n=30, name="eq"):
    monkeypatch.setattr("src.config.DATA", tmp_path)
    root = tmp_path / "enrollment" / name
    protos = {}
    for i in range(n):
        sid = f"stu_{i:02d}"
        d = root / sid
        d.mkdir(parents=True)
        fb, bb, cb = _unit(rng), _unit(rng), _color(rng)
        protos[sid] = (fb, bb, cb)
        for k in range(int(rng.integers(0, 9))):  # some students have no face samples
            np.save(d / f"face_{k:03d}.npy", _unit(rng, base=fb, noise=0.05))
        for k in range(int(rng.integers(1, 9))):
            np.save(d / f"body_{k:03d}.npy", _unit(rng, base=bb, noise=0.04))
        for k in range(int(rng.integers(0, 9))):
            c = _color(rng, cb)
            if k == 0 and i % 7 == 0:
                c = c[:5]  # legacy (5,3) sample without shoe row
            if k == 1 and i % 11 == 0:
                c = c.reshape(-1)[:7]  # malformed sample → _as_parts None
            np.save(d / f"color_{k:03d}.npy", c)
    g = EnrollmentGallery(name, face_embedder=StubFaceEmbedder(), body_embedder=StubBodyEmbedder())
    return g, protos


def _query(rng, protos):
    sid = list(protos)[int(rng.integers(len(protos)))]
    fb, bb, cb = protos[sid]
    r = rng.random()
    face = None if r < 0.2 else _unit(rng, base=fb if rng.random() < 0.8 else None, noise=0.08)
    body = None if 0.2 <= r < 0.3 else _unit(rng, base=bb if rng.random() < 0.85 else None, noise=0.07)
    rc = rng.random()
    color = None if rc < 0.15 else (np.arange(7, dtype=np.float32) if rc < 0.2 else _color(rng, cb if rng.random() < 0.8 else None))
    alpha = float(rng.choice([0.0, 0.1, 0.2]))
    return face, body, alpha, color


def _set_mode(monkeypatch, mode):
    base = dict(trk._identity_cfg())
    base["match_mode"] = mode
    monkeypatch.setattr(trk, "_identity_cfg", lambda: base)
    monkeypatch.setattr(ref, "_identity_cfg", lambda: base)


def test_color_batch_is_bit_identical():
    rng = np.random.default_rng(3)
    for _ in range(300):
        q = _color(rng) if rng.random() > 0.05 else np.zeros((N_PARTS, 3), np.float32)
        samples = [_color(rng, q if rng.random() < 0.6 else None, jitter=float(rng.uniform(1, 40))) for _ in range(12)]
        samples[0] = samples[0][:5]  # legacy
        samples.append(np.arange(7, dtype=np.float32))  # malformed
        w = rng.random(N_PARTS).astype(np.float32)
        w = None if rng.random() < 0.2 else w / w.sum()
        parts = [_as_parts(s) for s in samples]
        G = np.stack([p if p is not None else np.zeros((N_PARTS, 3), np.float32) for p in parts])
        ok = np.array([p is not None for p in parts])
        W = np.tile(np.ones(N_PARTS, np.float32) if w is None else w, (len(samples), 1))
        sims, decs = clothing_color_match_batch(q, G, ok, W)
        for m, s in enumerate(samples):
            info = clothing_color_match_info(q, s, part_weights=w)
            assert sims[m] == info["similarity"]  # exact
            assert bool(decs[m]) == info["shoe_decisive"]


@pytest.mark.parametrize("mode", ["face_body_color", "body_color", "face_body"])
def test_match_gallery_same_decisions(tmp_path, monkeypatch, mode):
    rng = np.random.default_rng({"face_body_color": 1, "body_color": 2, "face_body": 4}[mode])
    _set_mode(monkeypatch, mode)
    g, protos = _make_gallery(tmp_path, monkeypatch, rng)
    tracker = trk.FaceBodyTracker(gallery=g, match_threshold=0.58)
    max_d, n_id = 0.0, 0
    for _ in range(1500):
        face, body, alpha, color = _query(rng, protos)
        new = tracker._match_gallery(face, body, alpha, color_desc=color)
        old = ref.original_match_gallery(tracker, face, body, alpha, color_desc=color)
        assert (new[0], new[3]) == (old[0], old[3])
        for a, b in zip((new[1], new[2], new[4]), (old[1], old[2], old[4])):
            max_d = max(max_d, abs(a - b))
        n_id += new[0] is not None
    assert max_d < TOL
    assert n_id > 300  # the test exercises real matches, not only rejections


def test_tracker_runs_identically(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    g, protos = _make_gallery(tmp_path, monkeypatch, rng)
    new_t = trk.FaceBodyTracker(gallery=g, match_threshold=0.58)
    old_t = trk.FaceBodyTracker(gallery=g, match_threshold=0.58)
    old_t._match_gallery = types.MethodType(ref.original_match_gallery, old_t)
    sids = list(protos)[:12]
    for f in range(40):
        dets = []
        for j, sid in enumerate(sids):
            if rng.random() < 0.1:
                continue  # occlusion
            fb, bb, cb = protos[sid]
            x = 120.0 * j + rng.normal(0, 3)
            dets.append({"bbox": [x, 100.0, x + 100.0, 400.0],
                         "face_emb": _unit(rng, base=fb, noise=0.1) if rng.random() < 0.6 else None,
                         "body_emb": _unit(rng, base=bb, noise=0.08),
                         "color_desc": _color(rng, cb, jitter=8.0), "alpha": 0.2, "score": 0.9})
        a = new_t.update([dict(d) for d in dets])
        b = old_t.update([dict(d) for d in dets])
        assert [(t.track_id, t.student_id, t.identity_confidence) for t in a] == \
               [(t.track_id, t.student_id, t.identity_confidence) for t in b]
        assert max((abs(x.gallery_cost - y.gallery_cost) for x, y in zip(a, b)), default=0.0) < TOL


def test_index_follows_gallery_changes(tmp_path, monkeypatch):
    rng = np.random.default_rng(11)
    g, protos = _make_gallery(tmp_path, monkeypatch, rng, n=6, name="chg")
    tracker = trk.FaceBodyTracker(gallery=g, match_threshold=0.58)
    face, body, alpha, color = _query(rng, protos)
    tracker._match_gallery(face, body, alpha, color_desc=color)
    first = tracker._index
    img = (rng.random((300, 200, 3)) * 255).astype(np.uint8)

    class _Body512:
        def embed(self, image, bbox):
            return _unit(rng, base=protos["stu_00"][1], noise=0.02)

    g.body_embedder = _Body512()
    g.add_body_sample("stu_00", img, [10, 10, 150, 290])  # invalidates the cached student
    new = tracker._match_gallery(face, body, alpha, color_desc=color)
    assert tracker._index is not first
    old = ref.original_match_gallery(tracker, face, body, alpha, color_desc=color)
    assert (new[0], new[3]) == (old[0], old[3]) and abs(new[4] - old[4]) < TOL


def test_mixed_sample_shapes_fall_back_to_scalar_path(tmp_path, monkeypatch):
    rng = np.random.default_rng(13)
    g, protos = _make_gallery(tmp_path, monkeypatch, rng, n=5, name="mix")
    d = tmp_path / "enrollment" / "mix" / "stu_01"
    np.save(d / "body_099.npy", _unit(rng, d=768))  # e.g. an old stub sample
    tracker = trk.FaceBodyTracker(gallery=g, match_threshold=0.58)
    tracker._index = None
    g._cache.clear()
    face, body, alpha, color = _query(rng, protos)
    with pytest.raises(ValueError):  # the scalar path fails the same way it always did
        tracker._match_gallery(None, _unit(rng), 0.0, color_desc=None)
    assert tracker._index is not None and not tracker._index.regular
