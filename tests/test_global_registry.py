"""Unit tests for cross-session global face registry (no InsightFace required)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from src.identity.global_registry import GlobalFaceRegistry, _l2_normalize


def _emb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _l2_normalize(rng.standard_normal(512).astype(np.float32))


def test_global_registry_match_and_ambiguity(tmp_path, monkeypatch):
    # Redirect data/identity into tmp via config override (no cameras.yaml edit)
    cfg = {
        "enabled": True,
        "registry_relpath": "identity/_unittest/global_registry.json",
        "faces_relpath": "identity/_unittest/faces",
        "sessions_relpath": "identity/_unittest/sessions",
        "id_prefix": "stu_global",
        "face_match_threshold": 0.48,
        "ambiguity_margin": 0.06,
        "centroid_weight": 0.5,
        "consensus_min_sim": 0.40,
        "update_on_match": True,
        "max_face_samples_per_student": 24,
    }
    # Point data_path root indirectly: GlobalFaceRegistry uses data_path which writes under repo data/
    # Clean leftover then run; finally wipe.
    from src.config import data_path

    root = data_path("identity", "_unittest")
    if root.exists():
        shutil.rmtree(root)

    reg = GlobalFaceRegistry(cfg)
    a = [_emb(1), _emb(1)]  # same seed → identical
    # slight noise around a
    a_var = [_l2_normalize(_emb(1) + 0.02 * _emb(99))]
    b = [_emb(2)]

    gid_a = reg.allocate_new(a, session_id="s1", local_id="stu_00")
    gid_b = reg.allocate_new(b, session_id="s1", local_id="stu_01")
    assert gid_a != gid_b

    m = reg.match_face(a_var)
    assert m is not None and not m.is_new and m.global_id == gid_a

    # Ambiguous: midpoint-ish query between two far vectors should be new
    mid = _l2_normalize(0.5 * _emb(1) + 0.5 * _emb(2))
    m2 = reg.match_face([mid], threshold=0.95, ambiguity_margin=0.2)
    assert m2 is not None and (m2.is_new or not m2.global_id)

    # resolve rematch
    r = reg.resolve_or_create(a_var, session_id="s2", local_id="stu_03")
    assert r.global_id == gid_a and not r.is_new
    stu = reg.get_student(gid_a)
    assert "s2" in stu["sessions"]
    assert stu["local_ids"]["s2"] == "stu_03"

    shutil.rmtree(root)
