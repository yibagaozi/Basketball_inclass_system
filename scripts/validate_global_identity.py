#!/usr/bin/env python3
"""Validate cross-session face global-ID correspondence (no GT hardcoding).

Checks:
1) Linking the same enrollment twice recovers the same global_ids (self-consistency).
2) Optional: compare two session galleries (e.g. v2 vs v3 enroll) and report overlaps.
3) Duplicate / collision diagnostics (pairwise face sim above threshold between different globals;
   one local→multiple globals; multiple locals→same global incorrectly).

Usage:
  PYTHONPATH=. python scripts/validate_global_identity.py \\
    --manifest-a data/outputs/v2/gallery_manifest.json \\
    --manifest-b data/outputs/v3/gallery_manifest.json \\
    --dump-identity-dirs
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import data_path  # noqa: E402
from src.identity.global_registry import (  # noqa: E402
    GlobalFaceRegistry,
    create_face_lazy_gallery,
    get_global_registry_config,
    link_session_enrollment_to_global,
    registry_json_path,
    session_link_path,
    _centroid,
)
from src.identity.embedders import cosine_sim  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_enrollment(src_session: str, dst_session: str) -> None:
    src = data_path("enrollment", src_session)
    dst = data_path("enrollment", dst_session)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _mapping_dict(link: dict) -> dict[str, str]:
    return {m["local_id"]: m["global_id"] for m in link.get("mappings") or []}


def _face_centroids_for_session(session_id: str) -> dict[str, np.ndarray]:
    g = create_face_lazy_gallery(session_id)
    out: dict[str, np.ndarray] = {}
    for lid in g.list_students():
        c = _centroid(list(g.load_student(lid).get("face") or []))
        if c is not None:
            out[lid] = c
    return out


def pairwise_sim_matrix(
    cents_a: dict[str, np.ndarray],
    cents_b: dict[str, np.ndarray],
    *,
    row_label: str = "a",
    col_label: str = "b",
) -> dict:
    rows = sorted(cents_a.keys())
    cols = sorted(cents_b.keys())
    matrix: list[list[float]] = []
    flat: list[dict] = []
    for ra in rows:
        row_vals: list[float] = []
        for cb in cols:
            s = float(cosine_sim(cents_a[ra], cents_b[cb]))
            row_vals.append(round(s, 4))
            flat.append({
                f"{row_label}_id": ra,
                f"{col_label}_id": cb,
                "face_sim": round(s, 4),
            })
        matrix.append(row_vals)
    flat_sorted = sorted(flat, key=lambda x: -x["face_sim"])
    best_per_row = []
    for ra in rows:
        best = max(
            (x for x in flat if x[f"{row_label}_id"] == ra),
            key=lambda x: x["face_sim"],
            default=None,
        )
        if best:
            best_per_row.append({
                f"{row_label}_id": ra,
                f"best_{col_label}_id": best[f"{col_label}_id"],
                "face_sim": best["face_sim"],
            })
    return {
        "rows": rows,
        "cols": cols,
        "matrix": matrix,
        "pairs_sorted": flat_sorted,
        "best_per_row": best_per_row,
        "max_sim": flat_sorted[0]["face_sim"] if flat_sorted else None,
    }


def self_consistency(session_id: str) -> dict:
    """Fresh registry: link session, then link a cloned session → expect same globals by face."""
    cfg = get_global_registry_config()
    with tempfile.TemporaryDirectory(prefix="gid_val_") as td:
        _ = td  # isolate via dedicated data/identity/_tmp_validate
        tmp_root = data_path("identity", "_tmp_validate")
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        local_cfg = dict(cfg)
        local_cfg["registry_relpath"] = "identity/_tmp_validate/global_registry.json"
        local_cfg["faces_relpath"] = "identity/_tmp_validate/faces"
        local_cfg["sessions_relpath"] = "identity/_tmp_validate/sessions"

        link1 = link_session_enrollment_to_global(session_id, cfg=local_cfg)
        map1 = _mapping_dict(link1)

        clone_id = f"validate_clone_{session_id[:8]}"
        _copy_enrollment(session_id, clone_id)
        try:
            link2 = link_session_enrollment_to_global(clone_id, cfg=local_cfg)
            map2 = _mapping_dict(link2)
        finally:
            clone_dir = data_path("enrollment", clone_id)
            if clone_dir.exists():
                shutil.rmtree(clone_dir)

        recovered = 0
        total = 0
        details = []
        for lid, gid1 in map1.items():
            total += 1
            gid2 = map2.get(lid)
            ok = gid2 == gid1
            if ok:
                recovered += 1
            details.append({"local_id": lid, "first": gid1, "second": gid2, "ok": ok})

        if tmp_root.exists():
            shutil.rmtree(tmp_root)

        return {
            "session_id": session_id,
            "n": total,
            "recovered": recovered,
            "acc": (recovered / total) if total else 0.0,
            "pass": total > 0 and recovered == total,
            "details": details,
            "new_on_first": sum(1 for m in link1.get("mappings") or [] if m.get("is_new")),
            "new_on_second": sum(1 for m in link2.get("mappings") or [] if m.get("is_new")),
        }


def cross_session_overlap(session_a: str, session_b: str) -> dict:
    """Link A then B on isolated registry; report which B locals matched existing globals."""
    tmp_root = data_path("identity", "_tmp_validate_cross")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    cfg = get_global_registry_config()
    local_cfg = dict(cfg)
    local_cfg["registry_relpath"] = "identity/_tmp_validate_cross/global_registry.json"
    local_cfg["faces_relpath"] = "identity/_tmp_validate_cross/faces"
    local_cfg["sessions_relpath"] = "identity/_tmp_validate_cross/sessions"

    link_a = link_session_enrollment_to_global(session_a, cfg=local_cfg)
    link_b = link_session_enrollment_to_global(session_b, cfg=local_cfg)

    cents_a = _face_centroids_for_session(session_a)
    cents_b = _face_centroids_for_session(session_b)
    pw = pairwise_sim_matrix(cents_a, cents_b, row_label="a_local", col_label="b_local")

    thr = float(local_cfg.get("face_match_threshold", 0.58))
    above_thr = [p for p in pw["pairs_sorted"] if p["face_sim"] >= thr]

    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    return {
        "session_a": session_a,
        "session_b": session_b,
        "face_match_threshold": thr,
        "a_mappings": link_a.get("mappings"),
        "b_mappings": link_b.get("mappings"),
        "b_matched_existing": [
            m for m in (link_b.get("mappings") or []) if not m.get("is_new")
        ],
        "n_b_matched_existing": len([
            m for m in (link_b.get("mappings") or []) if not m.get("is_new")
        ]),
        "n_b": len(link_b.get("mappings") or []),
        "pairwise_best": [
            {
                "a_local": r["a_local_id"],
                "best_b_local": r["best_b_local_id"],
                "face_sim": r["face_sim"],
            }
            for r in pw["best_per_row"]
        ],
        "pairwise_matrix": {
            "rows": pw["rows"],
            "cols": pw["cols"],
            "matrix": pw["matrix"],
            "max_sim": pw["max_sim"],
            "pairs_above_threshold": above_thr,
            "top_pairs": pw["pairs_sorted"][:12],
        },
    }


def detect_duplicates(
    *,
    session_a: str | None,
    session_b: str | None,
    cross: dict | None,
    threshold: float | None = None,
) -> dict:
    """Report duplicate / collision findings.

    Categories:
    - same_face_multiple_globals: N/A at link time unless production mapping is inconsistent
    - multiple_locals_same_global: within one session, >1 local maps to same global
    - pairwise_different_globals_above_thr: face sim ≥ threshold between distinct global centroids
    - cross_session_false_match: B locals that reused A's globals when expected distinct people
    """
    cfg = get_global_registry_config()
    thr = float(threshold if threshold is not None else cfg.get("face_match_threshold", 0.58))

    findings: dict = {
        "threshold": thr,
        "multiple_locals_same_global": [],
        "same_face_multiple_globals": [],
        "pairwise_different_globals_above_thr": [],
        "cross_session_false_matches": [],
        "within_session_local_pairs_above_thr": [],
        "summary": "none",
    }

    # Production session link collisions
    for sid in (session_a, session_b):
        if not sid:
            continue
        lp = session_link_path(sid, cfg)
        if not lp.exists():
            continue
        link = json.loads(lp.read_text(encoding="utf-8"))
        by_g: dict[str, list[str]] = {}
        by_l: dict[str, list[str]] = {}
        for m in link.get("mappings") or []:
            lid, gid = m["local_id"], m["global_id"]
            by_g.setdefault(gid, []).append(lid)
            by_l.setdefault(lid, []).append(gid)
        for gid, lids in by_g.items():
            if len(lids) > 1:
                findings["multiple_locals_same_global"].append({
                    "session_id": sid,
                    "global_id": gid,
                    "local_ids": lids,
                })
        for lid, gids in by_l.items():
            uniq = sorted(set(gids))
            if len(uniq) > 1:
                findings["same_face_multiple_globals"].append({
                    "session_id": sid,
                    "local_id": lid,
                    "global_ids": uniq,
                })

        # Within-session: different locals with face sim ≥ thr (would be near-duplicates)
        cents = _face_centroids_for_session(sid)
        lids = sorted(cents.keys())
        for i, la in enumerate(lids):
            for lb in lids[i + 1 :]:
                s = float(cosine_sim(cents[la], cents[lb]))
                if s >= thr:
                    findings["within_session_local_pairs_above_thr"].append({
                        "session_id": sid,
                        "local_a": la,
                        "local_b": lb,
                        "face_sim": round(s, 4),
                    })

    # Production global registry: distinct globals with high face similarity
    reg_path = registry_json_path(cfg)
    if reg_path.exists():
        reg = GlobalFaceRegistry(cfg)
        gids = sorted(reg._centroids.keys())
        for i, ga in enumerate(gids):
            for gb in gids[i + 1 :]:
                s = float(cosine_sim(reg._centroids[ga], reg._centroids[gb]))
                if s >= thr:
                    findings["pairwise_different_globals_above_thr"].append({
                        "global_a": ga,
                        "global_b": gb,
                        "face_sim": round(s, 4),
                    })

    if cross is not None:
        # Informational: B locals that reused an existing global (may be true overlap).
        findings["cross_session_reused_globals"] = list(cross.get("b_matched_existing") or [])
        findings["cross_session_false_matches"] = findings["cross_session_reused_globals"]  # alias
        for p in (cross.get("pairwise_matrix") or {}).get("pairs_above_threshold") or []:
            findings.setdefault("cross_session_pairs_above_thr", []).append(p)

    # Hard issues only: collisions that break uniqueness within a session / registry bookkeeping.
    # Cross-session reused globals and near-threshold pairs are reported but not counted as failures
    # (classroom sets may legitimately share one person across days).
    hard_keys = (
        "multiple_locals_same_global",
        "same_face_multiple_globals",
        "within_session_local_pairs_above_thr",
    )
    soft_keys = (
        "pairwise_different_globals_above_thr",
        "cross_session_reused_globals",
    )
    n_hard = sum(len(findings[k]) for k in hard_keys)
    n_soft = sum(len(findings.get(k) or []) for k in soft_keys) + len(
        findings.get("cross_session_pairs_above_thr") or []
    )
    findings["n_hard_issues"] = n_hard
    findings["n_soft_notes"] = n_soft
    findings["n_issues"] = n_hard  # backward-compatible: hard only
    findings["summary"] = "duplicates_found" if n_hard else ("notes_only" if n_soft else "none")
    return findings


def production_snapshot(session_a: str | None, session_b: str | None) -> dict:
    cfg = get_global_registry_config()
    reg_path = registry_json_path(cfg)
    snap: dict = {
        "registry_path": str(reg_path.relative_to(ROOT)) if reg_path.exists() else None,
        "updated_at": None,
        "n_globals": 0,
        "next_index": None,
        "sessions": {},
    }
    if reg_path.exists():
        data = json.loads(reg_path.read_text(encoding="utf-8"))
        snap["updated_at"] = data.get("updated_at")
        snap["n_globals"] = len(data.get("students") or [])
        snap["next_index"] = data.get("next_index")
        snap["students"] = [
            {
                "global_id": s["global_id"],
                "local_ids": s.get("local_ids"),
                "sessions": s.get("sessions"),
                "n_face_samples": s.get("n_face_samples"),
            }
            for s in (data.get("students") or [])
        ]

    for label, sid in (("a", session_a), ("b", session_b)):
        if not sid:
            continue
        lp = session_link_path(sid, cfg)
        entry: dict = {"session_id": sid, "link_path": None, "mappings": []}
        if lp.exists():
            link = json.loads(lp.read_text(encoding="utf-8"))
            entry["link_path"] = str(lp.relative_to(ROOT))
            entry["mappings"] = link.get("mappings") or []
            entry["n_mapped"] = len(entry["mappings"])
            entry["local_to_global"] = {
                m["local_id"]: m["global_id"] for m in entry["mappings"]
            }
        snap["sessions"][label] = entry
    return snap


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _readme_block(title: str, lines: list[str]) -> str:
    body = "\n".join(f"- {ln}" if not ln.startswith("#") else ln for ln in lines)
    return f"# {title}\n\n生成时间（UTC）：{_utc_now()}\n\n{body}\n"


def dump_identity_dirs(
    report: dict,
    *,
    label_a: str,
    label_b: str | None,
    manifest_a: Path | None,
    manifest_b: Path | None,
) -> dict[str, str]:
    """Write validation artifacts under data/outputs/{v2,v3,identity}/identity/."""
    out_paths: dict[str, str] = {}
    thr = float(
        (report.get("cross_session") or {}).get("face_match_threshold")
        or get_global_registry_config().get("face_match_threshold", 0.58)
    )
    dup = report.get("duplicates") or {}
    prod = report.get("production") or {}

    # Cross-set folder
    cross_dir = data_path("outputs", "identity")
    cross_dir.mkdir(parents=True, exist_ok=True)
    cross_json = cross_dir / "validation_summary.json"
    _write_json(cross_json, report)
    out_paths["cross_summary"] = str(cross_json.relative_to(ROOT))

    cs = report.get("cross_session") or {}
    n_hit = cs.get("n_b_matched_existing", len(cs.get("b_matched_existing") or []))
    n_b = cs.get("n_b", len(cs.get("b_mappings") or []))
    max_sim = (cs.get("pairwise_matrix") or {}).get("max_sim")
    sc_a = report.get("self_consistency") or {}
    sc_b = report.get("self_consistency_b") or {}

    cross_readme = _readme_block(
        "跨课次身份识别验证结果（v2 ↔ v3）",
        [
            f"阈值 face_match_threshold = **{thr}**",
            f"自洽性 {label_a}: "
            f"**{sc_a.get('recovered', '?')}/{sc_a.get('n', '?')}** pass={sc_a.get('pass')}",
            (
                f"自洽性 {label_b}: "
                f"**{sc_b.get('recovered', '?')}/{sc_b.get('n', '?')}** pass={sc_b.get('pass')}"
                if label_b and sc_b
                else f"自洽性 {label_b}: （未跑）"
            ),
            f"跨课次复用已有 global（B→A）: **{n_hit}/{n_b}**",
            f"v2↔v3 人脸质心最大相似度: **{max_sim}**",
            f"重复脸 / 碰撞检测: **{dup.get('summary', 'n/a')}**"
            f"（hard={dup.get('n_hard_issues', dup.get('n_issues', 0))}，"
            f"soft_notes={dup.get('n_soft_notes', 0)}）",
            f"生产库 globals: **{prod.get('n_globals')}**（next_index={prod.get('next_index')}）",
            f"完整 JSON: `{cross_json.relative_to(ROOT)}`",
            f"manifest A: `{manifest_a}`" if manifest_a else "manifest A: n/a",
            f"manifest B: `{manifest_b}`" if manifest_b else "manifest B: n/a",
        ],
    )
    (cross_dir / "README.md").write_text(cross_readme, encoding="utf-8")
    out_paths["cross_readme"] = str((cross_dir / "README.md").relative_to(ROOT))

    if cs.get("pairwise_matrix"):
        pw_path = cross_dir / "v2_v3_pairwise_matrix.json"
        _write_json(pw_path, cs["pairwise_matrix"])
        out_paths["pairwise"] = str(pw_path.relative_to(ROOT))

    dup_path = cross_dir / "duplicate_findings.json"
    _write_json(dup_path, dup)
    out_paths["duplicates"] = str(dup_path.relative_to(ROOT))

    # Per-set folders
    per_set = [(label_a, "a", "self_consistency", "a_mappings")]
    if label_b:
        per_set.append((label_b, "b", "self_consistency_b", "b_mappings"))
    for label, sid_key, sc_key, map_key in per_set:
        d = data_path("outputs", label, "identity")
        d.mkdir(parents=True, exist_ok=True)
        sess = (prod.get("sessions") or {}).get(sid_key) or {}
        mappings = sess.get("mappings") or (cs.get(map_key) if cs else None) or []
        local_payload = {
            "label": label,
            "session_id": sess.get("session_id"),
            "generated_at": _utc_now(),
            "self_consistency": report.get(sc_key),
            "local_to_global": sess.get("local_to_global")
            or {m["local_id"]: m["global_id"] for m in mappings},
            "mappings": mappings,
            "duplicates_related": {
                "summary": dup.get("summary"),
                "multiple_locals_same_global": [
                    x
                    for x in (dup.get("multiple_locals_same_global") or [])
                    if x.get("session_id") == sess.get("session_id")
                ],
                "within_session_local_pairs_above_thr": [
                    x
                    for x in (dup.get("within_session_local_pairs_above_thr") or [])
                    if x.get("session_id") == sess.get("session_id")
                ],
            },
            "cross_ref": {
                "n_b_matched_existing": n_hit if label_b else None,
                "max_cross_sim": max_sim,
                "cross_summary": out_paths["cross_summary"],
            },
        }
        jp = d / "validation_summary.json"
        _write_json(jp, local_payload)
        out_paths[f"{label}_summary"] = str(jp.relative_to(ROOT))

        mp = d / "local_global_mapping.json"
        _write_json(
            mp,
            {
                "session_id": sess.get("session_id"),
                "local_to_global": local_payload["local_to_global"],
                "mappings": mappings,
            },
        )
        out_paths[f"{label}_mapping"] = str(mp.relative_to(ROOT))

        sc = report.get(sc_key) or {}
        readme = _readme_block(
            f"{label} 身份识别注册验证",
            [
                f"session_id: `{sess.get('session_id')}`",
                f"local→global 映射人数: **{len(local_payload['local_to_global'])}**",
                f"自洽性（二次 link）: "
                f"**{sc.get('recovered', '?')}/{sc.get('n', '?')}** pass={sc.get('pass')}",
                f"本 session 内重复脸对（sim≥{thr}）: "
                f"**{len(local_payload['duplicates_related']['within_session_local_pairs_above_thr'])}**",
                f"多 local 共用同一 global: "
                f"**{len(local_payload['duplicates_related']['multiple_locals_same_global'])}**",
                f"跨课次总览: `{out_paths['cross_summary']}`",
            ],
        )
        (d / "README.md").write_text(readme, encoding="utf-8")
        out_paths[f"{label}_readme"] = str((d / "README.md").relative_to(ROOT))

    return out_paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session-a")
    ap.add_argument("--session-b")
    ap.add_argument("--manifest-a", type=Path)
    ap.add_argument("--manifest-b", type=Path)
    ap.add_argument("--label-a", default="v2", help="output folder name under data/outputs/")
    ap.add_argument("--label-b", default="v3")
    ap.add_argument("--skip-self", action="store_true")
    ap.add_argument("--self-both", action="store_true", help="also run self-consistency on session B")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--dump-identity-dirs",
        action="store_true",
        help="write data/outputs/{label}/identity/ and data/outputs/identity/",
    )
    args = ap.parse_args()

    if args.manifest_a:
        args.session_a = _load_manifest(args.manifest_a)["session_id"]
    if args.manifest_b:
        args.session_b = _load_manifest(args.manifest_b)["session_id"]
    if not args.session_a:
        ap.error("need --session-a or --manifest-a")

    # Default: when both sessions present, self-check both
    if args.session_b and not args.skip_self:
        args.self_both = True

    report: dict = {
        "generated_at": _utc_now(),
        "self_consistency": None,
        "self_consistency_b": None,
        "cross_session": None,
        "duplicates": None,
        "production": None,
    }
    if not args.skip_self:
        report["self_consistency"] = self_consistency(args.session_a)
        print(
            f"self-consistency A: "
            f"{report['self_consistency']['recovered']}/{report['self_consistency']['n']} "
            f"pass={report['self_consistency']['pass']}"
        )
        if args.self_both and args.session_b:
            report["self_consistency_b"] = self_consistency(args.session_b)
            print(
                f"self-consistency B: "
                f"{report['self_consistency_b']['recovered']}/{report['self_consistency_b']['n']} "
                f"pass={report['self_consistency_b']['pass']}"
            )

    if args.session_b:
        report["cross_session"] = cross_session_overlap(args.session_a, args.session_b)
        n_hit = report["cross_session"]["n_b_matched_existing"]
        n_b = report["cross_session"]["n_b"]
        print(f"cross-session B matched existing globals: {n_hit}/{n_b}")
        for row in report["cross_session"]["pairwise_best"]:
            print(f"  {row['a_local']} best→ {row['best_b_local']} sim={row['face_sim']}")
        max_sim = report["cross_session"]["pairwise_matrix"]["max_sim"]
        print(f"  pairwise max sim = {max_sim}")

    report["production"] = production_snapshot(args.session_a, args.session_b)
    report["duplicates"] = detect_duplicates(
        session_a=args.session_a,
        session_b=args.session_b,
        cross=report.get("cross_session"),
    )
    print(
        f"duplicates: {report['duplicates']['summary']} "
        f"(n_issues={report['duplicates']['n_issues']})"
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"wrote {args.out}")

    if args.dump_identity_dirs:
        paths = dump_identity_dirs(
            report,
            label_a=args.label_a,
            label_b=args.label_b if args.session_b else None,
            manifest_a=args.manifest_a,
            manifest_b=args.manifest_b,
        )
        print("dumped identity dirs:")
        for k, v in paths.items():
            print(f"  {k}: {v}")

    ok = True
    if report["self_consistency"] is not None:
        ok = bool(report["self_consistency"]["pass"])
    if report.get("self_consistency_b") is not None:
        ok = ok and bool(report["self_consistency_b"]["pass"])
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
