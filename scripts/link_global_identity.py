#!/usr/bin/env python3
"""Link an existing session enrollment gallery to the global face registry.

Usage:
  PYTHONPATH=. python scripts/link_global_identity.py --session <uuid>
  PYTHONPATH=. python scripts/link_global_identity.py --from-manifest data/outputs/v2/gallery_manifest.json
  PYTHONPATH=. python scripts/link_global_identity.py --reset-registry   # wipe registry then link
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import data_path  # noqa: E402
from src.identity.global_registry import (  # noqa: E402
    get_global_registry_config,
    link_session_enrollment_to_global,
    registry_json_path,
    faces_root,
    session_link_path,
)


def _reset_registry() -> None:
    cfg = get_global_registry_config()
    rp = registry_json_path(cfg)
    if rp.exists():
        rp.unlink()
    fr = faces_root(cfg)
    if fr.exists():
        shutil.rmtree(fr)
    sr = data_path(*str(cfg["sessions_relpath"]).split("/"))
    if sr.exists():
        shutil.rmtree(sr)
    print(f"reset registry under data/{cfg['registry_relpath'].split('/')[0]}/")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", help="session UUID under data/enrollment/")
    ap.add_argument("--from-manifest", type=Path, help="gallery_manifest.json with session_id")
    ap.add_argument("--reset-registry", action="store_true", help="wipe global registry first")
    ap.add_argument("--student-ids", nargs="*", default=None)
    args = ap.parse_args()

    if args.reset_registry:
        _reset_registry()

    session_id = args.session
    if args.from_manifest:
        man = json.loads(Path(args.from_manifest).read_text(encoding="utf-8"))
        session_id = man["session_id"]
        if args.student_ids is None:
            args.student_ids = man.get("student_ids")

    if not session_id:
        ap.error("need --session or --from-manifest")

    enroll = data_path("enrollment", session_id)
    if not enroll.exists():
        print(f"ERROR: missing enrollment dir {enroll}", file=sys.stderr)
        return 1

    link = link_session_enrollment_to_global(session_id, args.student_ids)
    print(json.dumps({
        "session_id": session_id,
        "link_path": str(session_link_path(session_id)),
        "mappings": link.get("mappings"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
