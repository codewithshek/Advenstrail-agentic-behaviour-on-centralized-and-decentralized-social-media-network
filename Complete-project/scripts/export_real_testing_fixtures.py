#!/usr/bin/env python
"""
Export 4 Centralized and Decentralized Social Network Evaluation Fixtures.

Generates and exports 4 deterministic, distinct benchmarks for testing:
1. centralized_x_hashtag_flood.json / .csv
2. centralized_ig_shill_ring.json / .csv
3. decentralized_mastodon_bridge_attack.json / .csv
4. decentralized_a2a_protocol_spoof.json / .csv

Outputs are written to:
- Complete-project/data/synthetic/ (Parquet)
- Complete-project/datasets/fixtures/ (JSON and CSV)
- Testing-files/ (JSON and CSV, replacing legacy real_network_* fixtures)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ml" / "src"))

from aegis.config import load_config
from aegis.synthetic_agents import generate_campaign

FIXTURES_DIR = REPO_ROOT / "datasets" / "fixtures"
TESTING_FILES_DIR = REPO_ROOT.parent / "Testing-files"
SYNTHETIC_DATA_DIR = REPO_ROOT / "data" / "synthetic"

SCENARIOS_META = [
    {
        "id": "centralized_x_hashtag_flood",
        "scenario": "centralized_x_hashtag_flood",
        "name": "Centralized: X Hashtag Flood",
        "architecture": "centralized",
        "attack_mode": "Synchronized Hashtag Flood",
        "seed": 101,
        "n_agents": 8,
        "n_turns": 12,
    },
    {
        "id": "centralized_ig_shill_ring",
        "scenario": "centralized_ig_shill_ring",
        "name": "Centralized: Instagram Hub Shill Ring",
        "architecture": "centralized",
        "attack_mode": "Algorithmic Retweet Ring",
        "seed": 202,
        "n_agents": 8,
        "n_turns": 12,
    },
    {
        "id": "decentralized_mastodon_bridge_attack",
        "scenario": "decentralized_mastodon_bridge_attack",
        "name": "Decentralized: Mastodon Cross-Instance Bridge",
        "architecture": "decentralized_federated",
        "attack_mode": "Cross-Instance Sybil Bridge Attack",
        "seed": 303,
        "n_agents": 8,
        "n_turns": 12,
    },
    {
        "id": "decentralized_a2a_protocol_spoof",
        "scenario": "decentralized_a2a_protocol_spoof",
        "name": "Decentralized: A2A Protocol Spoof",
        "architecture": "decentralized_federated",
        "attack_mode": "A2A/MCP Protocol Discovery Spoof",
        "seed": 404,
        "n_agents": 8,
        "n_turns": 12,
    },
]


def clear_old_testing_files(directory: Path) -> List[Path]:
    """Remove obsolete real_network_* fixture files from testing directory."""
    removed = []
    if directory.exists():
        for old_file in directory.glob("real_network_*"):
            if old_file.is_file():
                old_file.unlink()
                removed.append(old_file)
    return removed


def export_csv(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    posts: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    """Export unified denormalized CSV file with row_type identifier."""
    fieldnames = [
        "row_type",
        "user_id",
        "screen_name",
        "label",
        "followers_count",
        "following_count",
        "statuses_count",
        "account_age_days",
        "verified",
        "description",
        "instance_domain",
        "source",
        "target",
        "relation",
        "is_cross_instance",
        "post_id",
        "text",
        "created_at",
        "hashtags",
        "mentions",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for node in nodes:
            writer.writerow({
                "row_type": "node",
                "user_id": node.get("user_id"),
                "screen_name": node.get("screen_name"),
                "label": node.get("label"),
                "followers_count": node.get("followers_count"),
                "following_count": node.get("following_count"),
                "statuses_count": node.get("statuses_count"),
                "account_age_days": node.get("account_age_days"),
                "verified": node.get("verified"),
                "description": node.get("description"),
                "instance_domain": node.get("instance_domain"),
            })

        for edge in edges:
            writer.writerow({
                "row_type": "edge",
                "source": edge.get("source"),
                "target": edge.get("target"),
                "relation": edge.get("relation"),
                "is_cross_instance": edge.get("is_cross_instance", False),
            })

        for post in posts:
            writer.writerow({
                "row_type": "post",
                "post_id": post.get("post_id"),
                "user_id": post.get("user_id"),
                "text": post.get("text"),
                "created_at": post.get("created_at"),
                "hashtags": " ".join(post.get("hashtags") or []),
                "mentions": " ".join(post.get("mentions") or []),
            })


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    TESTING_FILES_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    removed = clear_old_testing_files(TESTING_FILES_DIR)
    print(f"Cleared {len(removed)} old real_network_* fixture(s) in {TESTING_FILES_DIR}")

    settings = load_config(quiet=True)

    for item in SCENARIOS_META:
        fid = item["id"]
        scenario = item["scenario"]
        seed = item["seed"]
        arch = item["architecture"]
        attack_mode = item["attack_mode"]

        print(f"\nGenerating fixture: {fid} (scenario={scenario}, seed={seed})...")
        res = generate_campaign(
            settings,
            scenario=scenario,
            seed=seed,
            n_agents=item["n_agents"],
            campaign_turns=item["n_turns"],
            persist=False,
        )

        nodes_df = res.graph.nodes.copy()
        edges_df = res.graph.edges.copy()
        posts_df = res.posts.copy()

        # Save Parquet tables
        parquet_dir = SYNTHETIC_DATA_DIR / fid
        parquet_dir.mkdir(parents=True, exist_ok=True)
        nodes_df.to_parquet(parquet_dir / "nodes.parquet", index=False)
        edges_df.to_parquet(parquet_dir / "edges.parquet", index=False)
        posts_df.to_parquet(parquet_dir / "posts.parquet", index=False)

        # Build clean JSON dictionaries
        nodes_records = nodes_df.to_dict(orient="records")
        edges_records = edges_df.to_dict(orient="records")
        # Format created_at to ISO string
        posts_records = []
        for p in posts_df.to_dict(orient="records"):
            ca = p.get("created_at")
            if hasattr(ca, "isoformat"):
                p["created_at"] = ca.isoformat()
            else:
                p["created_at"] = str(ca)
            posts_records.append(p)

        metadata = {
            "fixture_id": fid,
            "name": item["name"],
            "network_architecture": arch,
            "architecture_attack_profile": attack_mode,
            "seed": seed,
            "counts": {
                "nodes": len(nodes_records),
                "edges": len(edges_records),
                "posts": len(posts_records),
                "agents": item["n_agents"],
                "instance_domains": sorted(list({n.get("instance_domain") for n in nodes_records})),
                "cross_instance_edges": sum(bool(e.get("is_cross_instance")) for e in edges_records),
            },
        }

        payload = {
            "metadata": metadata,
            "nodes": nodes_records,
            "edges": edges_records,
            "posts": posts_records,
        }

        # Write to datasets/fixtures/
        json_path_1 = FIXTURES_DIR / f"{fid}.json"
        csv_path_1 = FIXTURES_DIR / f"{fid}.csv"
        with open(json_path_1, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        export_csv(nodes_records, edges_records, posts_records, csv_path_1)

        # Write to Testing-files/
        json_path_2 = TESTING_FILES_DIR / f"{fid}.json"
        csv_path_2 = TESTING_FILES_DIR / f"{fid}.csv"
        with open(json_path_2, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        export_csv(nodes_records, edges_records, posts_records, csv_path_2)

        print(
            f"  -> Saved {fid}.json ({len(nodes_records)} nodes, {len(edges_records)} edges, {len(posts_records)} posts) "
            f"[domains: {metadata['counts']['instance_domains']}, cross-instance: {metadata['counts']['cross_instance_edges']}]"
        )

    print("\nAll 4 Centralized and Decentralized fixtures successfully exported!")


if __name__ == "__main__":
    main()
