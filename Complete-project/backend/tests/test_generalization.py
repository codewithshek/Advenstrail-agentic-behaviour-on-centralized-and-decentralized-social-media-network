from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import LeaveOneGroupOut
from torch_geometric.nn import SAGEConv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))

from aegis import config as acfg
from aegis import graph_features as gf
from aegis.dataset_loaders import GraphBundle
from aegis.synthetic_agents import BACKEND_OFFLINE, generate_campaign


def _bundle() -> GraphBundle:
    nodes = pd.DataFrame(
        [
            {
                "user_id": "c1::a",
                "label": 1,
                "split": "train",
                "account_age_days": 2,
                "followers_count": 1000,
                "following_count": 1,
                "campaign_id": "c1",
            },
            {
                "user_id": "c1::b",
                "label": 0,
                "split": "test",
                "account_age_days": 9000,
                "followers_count": 1,
                "following_count": 1000,
                "campaign_id": "c1",
            },
        ]
    )
    edges = pd.DataFrame(
        [{"source": "c1::a", "target": "c1::b", "relation": "mentions"}]
    )
    posts = pd.DataFrame(
        [
            {
                "post_id": "p1",
                "user_id": "c1::a",
                "text": "shared campaign text",
                "created_at": pd.Timestamp("2026-01-01T00:00:00Z"),
                "hashtags": ["topic"],
                "mentions": ["c1::b"],
            },
            {
                "post_id": "p2",
                "user_id": "c1::b",
                "text": "ordinary reply",
                "created_at": pd.Timestamp("2026-01-01T02:00:00Z"),
                "hashtags": [],
                "mentions": [],
            },
        ]
    )
    return GraphBundle(
        name="c1",
        nodes=nodes,
        edges=edges,
        posts=posts,
        provenance="generated",
        era="frontier_2026",
    )


def test_transfer_feature_contract_excludes_profile_shortcuts() -> None:
    settings = acfg.load_config(quiet=True)
    features = gf.build_transfer_features(_bundle(), settings)
    assert features.feature_columns == gf.TRANSFER_FEATURE_COLUMNS
    assert tuple(settings.graph_model["features"]) == gf.TRANSFER_FEATURE_COLUMNS
    assert "account_age_days" not in features.feature_columns
    assert "followers_to_following" not in features.feature_columns
    assert np.isfinite(features.X).all()
    assert np.max(features.features[["in_degree", "out_degree"]].to_numpy()) <= 1.0


def test_undirected_edge_dropout_keeps_reverse_pairs_consistent() -> None:
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    torch.manual_seed(7)
    dropped = gf.dropout_undirected_edges(edge_index, p=0.5, training=True)
    pairs = {tuple(pair) for pair in dropped.t().tolist()}
    assert all((target, source) in pairs for source, target in pairs)
    assert gf.dropout_undirected_edges(edge_index, p=0.5, training=False).equal(
        edge_index
    )


def test_cpu_graphsage_forward_uses_transfer_feature_width() -> None:
    features = gf.build_transfer_features(_bundle(), acfg.load_config(quiet=True))
    data = gf.to_pyg_data(features, undirected=True)
    layer = SAGEConv(len(gf.TRANSFER_FEATURE_COLUMNS), 8)
    output = layer(data.x, data.edge_index)
    assert output.shape == (2, 8)
    assert torch.isfinite(output).all()


def test_campaign_id_is_stable_and_persisted_on_all_graph_tables() -> None:
    settings = acfg.load_config(quiet=True)
    result = generate_campaign(
        settings,
        backend=BACKEND_OFFLINE,
        scenario="product_shill",
        seed=909,
        persist=False,
    )
    assert result.meta["campaign_id"] == "product_shill__seed_909"
    for frame in (result.graph.nodes, result.graph.edges, result.posts):
        assert frame["campaign_id"].nunique() == 1
        assert frame["campaign_id"].iloc[0] == result.meta["campaign_id"]


def test_leave_one_campaign_out_has_no_group_overlap() -> None:
    groups = np.repeat(["c1", "c2", "c3", "c4"], 3)
    y = np.tile([0, 1, 1], 4)
    X = np.arange(len(y) * 2).reshape(len(y), 2)
    for train, test in LeaveOneGroupOut().split(X, y, groups):
        assert set(groups[train]).isdisjoint(set(groups[test]))
        assert len(set(groups[test])) == 1


def test_notebooks_and_dashboard_encode_generalization_contract() -> None:
    graph_source = (ROOT / "ml/notebooks/_src/03_graph_coordination_model.py").read_text(
        encoding="utf-8"
    )
    fusion_source = (ROOT / "ml/notebooks/_src/04_hybrid_fusion_model.py").read_text(
        encoding="utf-8"
    )
    dashboard = (ROOT / "frontend/dashboard.html").read_text(encoding="utf-8")
    assert "campaign_bank_nodes.parquet" in graph_source
    assert "dropout_undirected_edges" in graph_source
    assert "LeaveOneGroupOut" in fusion_source
    assert (
        '.on("pointerenter", (event, current) => paintFocus(current.id))'
        in dashboard
    )
    assert '.on("pointermove", positionTooltip)' not in dashboard
    assert "tooltip.innerHTML = tooltipMarkup(nodesById.get(nodeId))" in dashboard

    for number in ("02", "03", "04"):
        matches = list((ROOT / "ml/notebooks").glob(f"{number}_*.ipynb"))
        assert len(matches) == 1
        json.loads(matches[0].read_text(encoding="utf-8"))
