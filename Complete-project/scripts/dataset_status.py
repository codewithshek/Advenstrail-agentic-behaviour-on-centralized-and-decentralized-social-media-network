#!/usr/bin/env python
"""
Print which datasets are wired into the pipeline and which are not, with reasons.

This is the "what are we actually training on?" question, answered from the config
and the drop-zone rather than from memory. Run it before writing up results.

    python scripts/dataset_status.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ml" / "src"))

import pandas as pd
from aegis import config as acfg
from aegis import local_store as ls


def main() -> int:
    settings = acfg.load_config(quiet=True)
    text_specs = settings.text_datasets or {}
    graph_specs = settings.graph_datasets or {}
    all_specs = {**text_specs, **graph_specs}

    scans = ls.scan_all(all_specs, repo_root=settings.paths.root, extract=False)

    rows = []
    for name, spec in all_specs.items():
        spec = spec or {}
        scan = scans.get(name)
        rows.append({
            "dataset": name,
            "branch": "text" if name in text_specs else "graph",
            "era": spec.get("era", ""),
            "used": "YES" if spec.get("enabled", True) else "no",
            "on_disk": "yes" if (scan and scan.available) else "no",
            "files": scan.n_files if scan else 0,
            "archives": len(scan.archives) if scan else 0,
            # Read-in-place corpora deliberately leave the 232/466 MB
            # containers unexpanded. Omitting archive_bytes made them appear
            # to be 531-byte datasets because only info.json was counted.
            "size": (
                ls._human(scan.total_bytes + scan.archive_bytes)
                if scan else "0 B"
            ),
            "reason_not_used": spec.get("unavailable_reason", ""),
        })

    frame = pd.DataFrame(rows).sort_values(
        ["used", "branch", "dataset"], ascending=[False, True, True]
    )
    pd.set_option("display.width", 200)
    print(frame.to_string(index=False))

    used = frame[frame["used"] == "YES"]
    print(f"\n{len(used)} of {len(frame)} datasets in use "
          f"({(used['branch'] == 'text').sum()} text, {(used['branch'] == 'graph').sum()} graph)")
    print("plus the generated 2026 agentic campaign (synthetic_agents.py), which is not a "
          "download and so has no drop-zone entry.")

    unused = frame[frame["used"] != "YES"]
    if len(unused):
        print("\nNot used:")
        for _, row in unused.iterrows():
            present = "present but unusable" if row["on_disk"] == "yes" else "not on disk"
            print(f"  {row['dataset']:<30} {present:<22} {row['reason_not_used']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
