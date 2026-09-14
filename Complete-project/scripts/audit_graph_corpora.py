"""Audit real graph corpora for coverage, leakage, and baseline separability.

Examples:
    python scripts/audit_graph_corpora.py
    python scripts/audit_graph_corpora.py caverlee_2011 --features

``--features`` is intentionally opt-in: extracting behavioral features from
Caverlee's 1.7M retained posts takes roughly ten minutes on a desktop CPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml" / "src"))

from aegis.config import load_config
from aegis.dataset_loaders import load_graph_dataset
from aegis.graph_features import build_features

DEFAULT_CORPORA = ("caverlee_2011", "cresci_2015", "cresci_2017_full")


def audit(name: str, *, include_features: bool) -> dict[str, object]:
    import networkx as nx

    settings = load_config()
    started = time.perf_counter()
    bundle = load_graph_dataset(name, settings)
    nodes, edges, posts = bundle.nodes, bundle.edges, bundle.posts

    graph = nx.Graph()
    graph.add_nodes_from(
        (uid, {"label": int(label)})
        for uid, label in zip(nodes["user_id"], nodes["label"])
    )
    graph.add_edges_from(zip(edges["source"], edges["target"]))
    assortativity = (
        nx.attribute_assortativity_coefficient(graph, "label")
        if graph.number_of_edges()
        else np.nan
    )

    numeric_ids = pd.to_numeric(nodes["user_id"], errors="coerce")
    id_auc = (
        roc_auc_score(nodes["label"], numeric_ids.fillna(0))
        if numeric_ids.notna().mean() > 0.9
        else np.nan
    )

    result: dict[str, object] = {
        "dataset": name,
        "nodes": len(nodes),
        "edges": len(edges),
        "posts": len(posts),
        "human": int((nodes["label"] == 0).sum()),
        "bot": int((nodes["label"] == 1).sum()),
        "label_assortativity": round(float(assortativity), 4),
        "numeric_id_auc": round(float(id_auc), 4),
        "load_seconds": round(time.perf_counter() - started, 1),
    }

    if include_features:
        feature_bundle = build_features(bundle, settings)
        X, y = feature_bundle.X, feature_bundle.y
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=settings.seed, stratify=y
        )
        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=settings.seed,
            n_jobs=-1,
        ).fit(X_train, y_train)
        result["iid_rf_auc"] = round(
            float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])), 4
        )
        result["total_seconds"] = round(time.perf_counter() - started, 1)

    if "group" in nodes:
        print(f"\n{name} groups:")
        print(nodes.groupby("group")["label"].agg(["size", "mean"]).to_string())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=list(DEFAULT_CORPORA))
    parser.add_argument(
        "--features",
        action="store_true",
        help="also build all graph features and fit an i.i.d. RF diagnostic",
    )
    args = parser.parse_args()

    rows = [audit(name, include_features=args.features) for name in args.datasets]
    print("\nAudit summary (high AUC is not generalization when leakage columns are high):")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
