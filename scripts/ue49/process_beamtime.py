"""Process a whole UE49 beamtime folder with its transcribed logbook.

    python scripts/ue49/process_beamtime.py "<.../UE49 BESSY II/June 2025>" \
        --dead "<Procedure files>/deadpixel_*.tif" --norm "<Procedure files>/NormImage_*.tif"

The beamtime folder must contain logbook_sessions.csv and logbook_conditions.csv.
Outputs go to <day>/tensorspec_peem/ (clusters/, spectra/, index.csv); the
beamtime folder gets tensorspec_peem_index.csv, _catalog.csv and _logbook_check.csv.
--plan only prints the plan; --days limits processing; existing outputs are skipped
unless --redo.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tensorspec.core.peem.ue49 import batch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--dead")
    ap.add_argument("--norm")
    ap.add_argument("--days", nargs="*")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--overview-only", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    if a.overview_only:
        overview(root)
        return 0
    if a.plan:
        sessions = batch.read_sessions(root / "logbook_sessions.csv")
        conds = batch.read_conditions(root / "logbook_conditions.csv")
        p = batch.plan(root, sessions, conds)
        print(f"{len(p['catalog'])} scans; {len(p['clusters'])} clusters "
              f"({sum(len(c.units) for c in p['clusters'])} units); {len(p['spectra'])} spectra jobs")
        print("unit kinds:", Counter(u.kind for c in p["clusters"] for u in c.units))
        print("spectra kinds:", Counter(j.kind for j in p["spectra"]))
        for c in p["clusters"]:
            print(f"  {c.day} {c.cid:28s} units={len(c.units)} kinds={sorted({u.kind for u in c.units})} "
                  f"left={[s.label for s in c.unassigned]} | {batch._cond_text(c.condition)}")
        for j in p["spectra"]:
            if j.kind != "XAS":
                print(f"  {j.day} {j.sid}")
        issues = batch.validate(p, conds)
        print(f"{len(issues)} logbook/data issues")
        for i in issues[:40]:
            print("   ", i)
        return 0
    preview = None
    if not a.no_preview:
        from tensorspec.plotting.ue49_preview import Preview

        preview = Preview()
    st = batch.Settings(dead_map=a.dead, norm_image=a.norm)
    res = batch.run(root, st, days=a.days, skip_existing=not a.redo, preview=preview,
                    log=lambda m: print(m, flush=True))
    print(Counter(str(r.get("status", "")).split(":")[0] for r in res["rows"]))
    if preview is not None:
        overview(root)
    return 0


def overview(root: Path) -> None:
    """Per-session contact sheets from the index written by batch.run."""
    import csv

    from tensorspec.plotting.ue49_preview import session_overview

    with open(root / "tensorspec_peem_index.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    # rows of skipped (existing) clusters carry no condition; recover them from cluster.json
    import json
    for r in rows:
        if r.get("status", "").startswith("ok"):
            r["status"] = "ok"
        if r.get("type") == "cluster" and r.get("status") == "existing":
            meta = json.loads((root / r["path"] / "cluster.json").read_text())
            c = meta["condition"]
            r.update({"session": c["session"], "status": "ok", "T_K_measured": meta.get("temperature_K_measured"),
                      "log_ccs_mA": c["ccs_mA"], "log_pulse_before": c["pulse_before"],
                      "log_rotation_deg": c["rotation_deg"]})
    for session in sorted({r["session"] for r in rows if r.get("session")}):
        n = session_overview(root, session, rows, root / f"tensorspec_peem_overview_{session}.png")
        print(f"overview {session}: {n} clusters", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
