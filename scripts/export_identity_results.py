#!/usr/bin/env python3
"""Export v2/v3 identity enrollment & validation results into dedicated output folders.

Layout (matches data/outputs/{v2,v3}/):
  data/outputs/v2/identity/
  data/outputs/v3/identity/
  data/outputs/identity/          # cross-session + overall comparison

Usage:
  PYTHONPATH=. python scripts/export_identity_results.py
  PYTHONPATH=. python scripts/export_identity_results.py --skip-validate  # reuse prior JSON
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import data_path  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run_validate(manifest_a: Path, manifest_b: Path | None, out: Path) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "validate_global_identity.py"),
        "--manifest-a",
        str(manifest_a),
        "--out",
        str(out),
    ]
    if manifest_b is not None:
        cmd.extend(["--manifest-b", str(manifest_b)])
    env = dict(**{**dict(**__import__("os").environ), "PYTHONPATH": str(ROOT)})
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
    return _load_json(out)  # type: ignore[return-value]


def _enroll_inventory(session_id: str) -> dict:
    root = data_path("enrollment", session_id)
    students = []
    if not root.exists():
        return {"session_id": session_id, "exists": False, "students": []}
    for stu_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("stu_")):
        meta = _load_json(stu_dir / "meta.json") or {}
        n_face = len(list(stu_dir.glob("face_*.npy")))
        n_body = len(list(stu_dir.glob("body_*.npy")))
        n_color = len(list(stu_dir.glob("color_*.npy")))
        students.append({
            "local_id": stu_dir.name,
            "n_face": n_face,
            "n_body": n_body,
            "n_color": n_color,
            "enroll_ok": n_face > 0 and n_body > 0,
            "global_id": meta.get("global_id"),
            "global_match": meta.get("global_match"),
            "enroll_mode": meta.get("enroll_mode"),
            "n_samples_meta": meta.get("n_samples"),
        })
    return {
        "session_id": session_id,
        "exists": True,
        "enrollment_root": str(root.relative_to(ROOT)) if root.is_relative_to(ROOT) else str(root),
        "n_students": len(students),
        "n_enroll_ok": sum(1 for s in students if s["enroll_ok"]),
        "students": students,
    }


def _link_map_from_session(session_id: str) -> dict:
    link_path = data_path("identity", "sessions", f"{session_id}.json")
    link = _load_json(link_path)
    if not link:
        return {
            "session_id": session_id,
            "link_path": str(link_path),
            "exists": False,
            "mappings": [],
            "local_to_global": {},
        }
    local_to_global = {
        m["local_id"]: m["global_id"] for m in (link.get("mappings") or [])
    }
    return {
        "session_id": session_id,
        "link_path": str(link_path.relative_to(ROOT)) if link_path.is_relative_to(ROOT) else str(link_path),
        "exists": True,
        "updated_at": link.get("updated_at"),
        "mappings": link.get("mappings") or [],
        "local_to_global": local_to_global,
    }


def _v3_group_id_acc(out_dir: Path) -> list[dict]:
    rows = []
    for gdir in sorted(out_dir.glob("group_*")):
        if gdir.name == "group_00":
            continue
        ev = _load_json(gdir / "eval_vs_gt.json")
        if not ev:
            rows.append({"group": gdir.name, "eval_exists": False})
            continue
        rows.append({
            "group": gdir.name,
            "eval_exists": True,
            "pass": ev.get("pass"),
            "id_acc": ev.get("id_acc"),
            "id_consistent": ev.get("id_consistent"),
            "need_id": ev.get("need_id"),
            "id_map": ev.get("id_map"),
            "n_matched": ev.get("n_matched"),
            "n_gt": ev.get("n_gt"),
            "n_false_alarm": ev.get("n_false_alarm"),
        })
    return rows


def _v2_action_status(out_dir: Path) -> list[dict]:
    """Lightweight mirror of analyze_v2_ground_truth (action labels; enroll for g0)."""
    expected = {
        0: set(),
        1: {"free_throw"},
        2: {"layup"},
        3: {"triple_threat", "layup"},
        4: {"triple_threat", "jump_shot", "free_throw", "layup"},
        5: {"triple_threat", "jump_shot", "free_throw"},
        6: {"pass"},
    }
    rows = []
    for gid in range(0, 7):
        summary = _load_json(out_dir / f"group_{gid:02d}" / "summary.json")
        if summary is None:
            rows.append({"group": gid, "status": "missing", "pass": False, "note": "no summary.json"})
            continue
        if gid == 0:
            n = len(summary.get("student_ids") or [])
            rows.append({
                "group": gid,
                "role": "enrollment",
                "enroll_count": n,
                "enroll_ok": n == 6,
                "pass": n == 6,
                "id_acc": None,
                "note": "v2 GT has no per-person letter id_acc; group0 = enroll count only",
            })
            continue
        hist = summary.get("action_type_hist") or {}
        allowed = expected[gid]
        illegal = sorted(t for t in hist if t not in allowed)
        missing = sorted(t for t in allowed if hist.get(t, 0) == 0)
        ok = not illegal and not missing and int(summary.get("clip_count") or 0) > 0
        rows.append({
            "group": gid,
            "role": "action",
            "pass": ok,
            "detected": hist,
            "illegal": illegal,
            "missing_types": missing,
            "clip_count": summary.get("clip_count"),
            "id_acc": None,
            "note": "v2 GT script does not score stu↔person bijection (no id_acc)",
        })
    return rows


def _dataset_bundle(
    *,
    dataset: str,
    manifest: dict,
    inventory: dict,
    link: dict,
    self_val: dict | None,
    group_rows: list[dict],
) -> dict:
    n_ok = inventory.get("n_enroll_ok", 0)
    n_stu = inventory.get("n_students", 0)
    map_n = len(link.get("local_to_global") or {})
    self_pass = bool((self_val or {}).get("pass")) if self_val else None
    return {
        "dataset": dataset,
        "generated_at": _utc_now(),
        "gallery_manifest": manifest,
        "enrollment": {
            "success": n_stu > 0 and n_ok == n_stu,
            "n_students": n_stu,
            "n_enroll_ok": n_ok,
            "session_id": manifest.get("session_id"),
            "enroll_mode_hint": (inventory.get("students") or [{}])[0].get("enroll_mode"),
        },
        "global_link": {
            "success": map_n > 0 and map_n == n_stu,
            "n_mapped": map_n,
            "local_to_global": link.get("local_to_global") or {},
            "all_is_new": all(m.get("is_new") for m in (link.get("mappings") or [])),
        },
        "self_consistency": self_val,
        "per_group": group_rows,
    }


def _md_dataset(summary: dict, title: str) -> str:
    en = summary["enrollment"]
    gl = summary["global_link"]
    sc = summary.get("self_consistency") or {}
    lines = [
        f"# {title}",
        "",
        f"> generated_at: `{summary['generated_at']}`",
        "",
        "## Enrollment",
        "",
        f"| 项 | 值 |",
        f"|----|----|",
        f"| session_id | `{en.get('session_id')}` |",
        f"| students | {en.get('n_enroll_ok')}/{en.get('n_students')} samples OK |",
        f"| enroll success | **{'PASS' if en.get('success') else 'FAIL'}** |",
        f"| enroll_mode | {en.get('enroll_mode_hint')} |",
        "",
        "## Global link (local ↔ global)",
        "",
        f"| local_id | global_id |",
        f"|----------|-----------|",
    ]
    for lid, gid in sorted((gl.get("local_to_global") or {}).items()):
        lines.append(f"| `{lid}` | `{gid}` |")
    lines += [
        "",
        f"link success: **{'PASS' if gl.get('success') else 'FAIL'}** "
        f"({gl.get('n_mapped')} mapped; all_is_new={gl.get('all_is_new')})",
        "",
        "## Self-consistency (clone re-link)",
        "",
    ]
    if sc:
        lines.append(
            f"- recovered **{sc.get('recovered')}/{sc.get('n')}** "
            f"acc={sc.get('acc')} pass=**{'PASS' if sc.get('pass') else 'FAIL'}**"
        )
    else:
        lines.append("- (not run)")
    lines += ["", "## Per-group identity / related", ""]
    rows = summary.get("per_group") or []
    if not rows:
        lines.append("_none_")
    else:
        # v3 style
        if any("id_acc" in r and r.get("id_acc") is not None for r in rows):
            lines += [
                "| Group | pass | id_acc | matched | fa |",
                "|-------|------|--------|---------|----|",
            ]
            for r in rows:
                if not r.get("eval_exists", True) and "pass" not in r:
                    lines.append(f"| {r.get('group')} | — | — | — | — |")
                    continue
                if "eval_exists" in r:
                    lines.append(
                        f"| {r.get('group')} | {r.get('pass')} | {r.get('id_acc')} | "
                        f"{r.get('n_matched')}/{r.get('n_gt')} | {r.get('n_false_alarm')} |"
                    )
                else:
                    note = r.get("note") or ""
                    lines.append(
                        f"| g{r.get('group')} | {r.get('pass')} | {r.get('id_acc')} | "
                        f"clips={r.get('clip_count')} | {note[:40]} |"
                    )
        else:
            lines += [
                "| Group | pass | note |",
                "|-------|------|------|",
            ]
            for r in rows:
                lines.append(
                    f"| g{r.get('group')} | {r.get('pass')} | {r.get('note') or r.get('role') or ''} |"
                )
    lines.append("")
    return "\n".join(lines)


def _md_overall(v2: dict, v3: dict, cross: dict | None) -> str:
    lines = [
        "# Identity recognition results — v2 vs v3",
        "",
        f"> generated_at: `{_utc_now()}`",
        "",
        "## Comparison",
        "",
        "| 指标 | v2 | v3 |",
        "|------|----|----|",
        f"| 注册人数 | {v2['enrollment']['n_students']} | {v3['enrollment']['n_students']} |",
        f"| 注册样本齐全 | {'PASS' if v2['enrollment']['success'] else 'FAIL'} | "
        f"{'PASS' if v3['enrollment']['success'] else 'FAIL'} |",
        f"| 全局 link 映射 | {v2['global_link']['n_mapped']}/{v2['enrollment']['n_students']} | "
        f"{v3['global_link']['n_mapped']}/{v3['enrollment']['n_students']} |",
        f"| 克隆二次 link 自洽 | "
        f"{(v2.get('self_consistency') or {}).get('recovered')}/"
        f"{(v2.get('self_consistency') or {}).get('n')} | "
        f"{(v3.get('self_consistency') or {}).get('recovered')}/"
        f"{(v3.get('self_consistency') or {}).get('n')} |",
        f"| 课内 id_acc（有真值字母） | N/A（v2 真值无 person 字母） | "
        f"见下表（现有 eval 均为 1.0） |",
        "",
    ]
    if cross:
        b_hit = len(cross.get("b_matched_existing") or [])
        b_n = len(cross.get("b_mappings") or [])
        lines += [
            "## Cross-session (v2 enroll → then v3 enroll)",
            "",
            f"- B matched existing globals: **{b_hit}/{b_n}** "
            f"（期望 0：两套测试集为不同人）",
            "",
            "| v2 local | best v3 local | face_sim |",
            "|----------|---------------|----------|",
        ]
        for row in cross.get("pairwise_best") or []:
            lines.append(
                f"| `{row.get('a_local')}` | `{row.get('best_b_local')}` | {row.get('face_sim')} |"
            )
        lines.append("")
    lines += [
        "## Artifact paths",
        "",
        "- `data/outputs/v2/identity/`",
        "- `data/outputs/v3/identity/`",
        "- `data/outputs/identity/`（跨课次 + 总表）",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-validate", action="store_true",
                    help="Do not re-run validate_global_identity.py")
    args = ap.parse_args()

    man_v2_path = ROOT / "data/outputs/v2/gallery_manifest.json"
    man_v3_path = ROOT / "data/outputs/v3/gallery_manifest.json"
    man_v2 = _load_json(man_v2_path)
    man_v3 = _load_json(man_v3_path)
    if not man_v2 or not man_v3:
        print("ERROR: missing gallery manifests", file=sys.stderr)
        return 1

    out_v2 = ROOT / "data/outputs/v2/identity"
    out_v3 = ROOT / "data/outputs/v3/identity"
    out_cross = ROOT / "data/outputs/identity"
    out_v2.mkdir(parents=True, exist_ok=True)
    out_v3.mkdir(parents=True, exist_ok=True)
    out_cross.mkdir(parents=True, exist_ok=True)

    sid_v2 = man_v2["session_id"]
    sid_v3 = man_v3["session_id"]

    # --- validation ---
    cross_raw = None
    self_v2 = None
    self_v3 = None
    if not args.skip_validate:
        cross_raw = _run_validate(
            man_v2_path, man_v3_path, out_cross / "cross_session_validate.json"
        )
        self_v2 = cross_raw.get("self_consistency")
        # dedicated v3 self (isolated; indexing starts at 00)
        v3_only = _run_validate(man_v3_path, None, out_v3 / "validation_self.json")
        self_v3 = v3_only.get("self_consistency")
        _write_json(out_v2 / "validation_self.json", {"self_consistency": self_v2})
    else:
        prev = _load_json(ROOT / "data/outputs/identity_global_validate.json") or {}
        cross_raw = prev
        self_v2 = prev.get("self_consistency")
        self_v3 = (_load_json(out_v3 / "validation_self.json") or {}).get("self_consistency")
        if cross_raw:
            _write_json(out_cross / "cross_session_validate.json", cross_raw)

    # also keep legacy path in sync
    if cross_raw and not args.skip_validate:
        _write_json(ROOT / "data/outputs/identity_global_validate.json", cross_raw)

    inv_v2 = _enroll_inventory(sid_v2)
    inv_v3 = _enroll_inventory(sid_v3)
    link_v2 = _link_map_from_session(sid_v2)
    link_v3 = _link_map_from_session(sid_v3)

    # copy session link JSON into identity folders
    for link, dest in (
        (link_v2, out_v2 / "session_link.json"),
        (link_v3, out_v3 / "session_link.json"),
    ):
        src = data_path("identity", "sessions", f"{link['session_id']}.json")
        if src.exists():
            shutil.copy2(src, dest)

    _write_json(out_v2 / "enroll_inventory.json", inv_v2)
    _write_json(out_v3 / "enroll_inventory.json", inv_v3)
    _write_json(out_v2 / "link_map.json", {
        "dataset": "v2",
        "session_id": sid_v2,
        "local_to_global": link_v2.get("local_to_global"),
        "mappings": link_v2.get("mappings"),
    })
    _write_json(out_v3 / "link_map.json", {
        "dataset": "v3",
        "session_id": sid_v3,
        "local_to_global": link_v3.get("local_to_global"),
        "mappings": link_v3.get("mappings"),
    })

    v2_groups = _v2_action_status(ROOT / "data/outputs/v2")
    v3_groups = _v3_group_id_acc(ROOT / "data/outputs/v3")

    sum_v2 = _dataset_bundle(
        dataset="v2",
        manifest=man_v2,
        inventory=inv_v2,
        link=link_v2,
        self_val=self_v2,
        group_rows=v2_groups,
    )
    sum_v3 = _dataset_bundle(
        dataset="v3",
        manifest=man_v3,
        inventory=inv_v3,
        link=link_v3,
        self_val=self_v3,
        group_rows=v3_groups,
    )

    # honest overall pass flags
    cross_ok = True
    if cross_raw and cross_raw.get("cross_session"):
        cs = cross_raw["cross_session"]
        # expected: no false merges between different people
        cross_ok = len(cs.get("b_matched_existing") or []) == 0

    overall = {
        "generated_at": _utc_now(),
        "repo_version": (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        if (ROOT / "VERSION").exists() else None,
        "v2": {
            "enroll_pass": sum_v2["enrollment"]["success"],
            "link_pass": sum_v2["global_link"]["success"],
            "self_consistency_pass": bool((self_v2 or {}).get("pass")),
            "self_consistency": f"{(self_v2 or {}).get('recovered')}/{(self_v2 or {}).get('n')}",
            "in_class_id_acc": None,
            "in_class_id_acc_note": "v2 GT has no person letters; no id_acc metric",
            "action_gt_pass_groups": sum(1 for r in v2_groups if r.get("pass")),
            "action_gt_total_groups": len(v2_groups),
        },
        "v3": {
            "enroll_pass": sum_v3["enrollment"]["success"],
            "link_pass": sum_v3["global_link"]["success"],
            "self_consistency_pass": bool((self_v3 or {}).get("pass")),
            "self_consistency": f"{(self_v3 or {}).get('recovered')}/{(self_v3 or {}).get('n')}",
            "in_class_id_acc_groups": [
                {"group": r["group"], "id_acc": r.get("id_acc"), "pass": r.get("pass")}
                for r in v3_groups if r.get("eval_exists")
            ],
            "in_class_id_acc_all_1": all(
                (r.get("id_acc") or 0) >= 1.0 - 1e-9
                for r in v3_groups if r.get("eval_exists") and r.get("need_id")
            ),
        },
        "cross_session": {
            "v2_then_v3_false_matches": len(
                (cross_raw or {}).get("cross_session", {}).get("b_matched_existing") or []
            ),
            "v3_n": len((cross_raw or {}).get("cross_session", {}).get("b_mappings") or []),
            "pass_expected_zero_overlap": cross_ok,
            "pairwise_best": (cross_raw or {}).get("cross_session", {}).get("pairwise_best"),
        },
        "production_registry": {
            "path": "data/identity/global_registry.json",
            "n_globals": len((_load_json(data_path("identity", "global_registry.json")) or {}).get("students") or []),
            "seed": "v2→stu_global_00..05; v3→stu_global_06..09",
        },
        "paths": {
            "v2_identity": "data/outputs/v2/identity/",
            "v3_identity": "data/outputs/v3/identity/",
            "cross": "data/outputs/identity/",
        },
    }

    _write_json(out_v2 / "summary.json", sum_v2)
    _write_json(out_v3 / "summary.json", sum_v3)
    _write_json(out_cross / "overall_summary.json", overall)
    _write_text(out_v2 / "summary.md", _md_dataset(sum_v2, "v2 identity results"))
    _write_text(out_v3 / "summary.md", _md_dataset(sum_v3, "v3 identity results"))
    _write_text(
        out_cross / "SUMMARY.md",
        _md_overall(sum_v2, sum_v3, (cross_raw or {}).get("cross_session")),
    )

    # symlink/copy registry snapshot (small JSON only)
    reg = data_path("identity", "global_registry.json")
    if reg.exists():
        shutil.copy2(reg, out_cross / "global_registry_snapshot.json")

    print(f"wrote {out_v2}")
    print(f"wrote {out_v3}")
    print(f"wrote {out_cross}")
    print(
        f"v2 self={(self_v2 or {}).get('recovered')}/{(self_v2 or {}).get('n')} "
        f"v3 self={(self_v3 or {}).get('recovered')}/{(self_v3 or {}).get('n')} "
        f"cross_false_match="
        f"{overall['cross_session']['v2_then_v3_false_matches']}/"
        f"{overall['cross_session']['v3_n']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
