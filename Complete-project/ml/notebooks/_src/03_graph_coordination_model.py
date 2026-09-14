# %% [markdown]
# # 03 · Strictly Inductive GraphSAGE
#
# Every campaign is a separate graph. Validation and test campaign edges never
# participate in training message passing, and scaling is fit on training
# campaigns only.

# %%
from __future__ import annotations

import copy
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_here = Path.cwd()
for _candidate in (_here, *_here.parents):
    if (_candidate / "ml" / "src" / "aegis").is_dir():
        sys.path.insert(0, str(_candidate / "ml" / "src"))
        break
else:
    raise RuntimeError("Launch Jupyter from the repository root.")

from aegis import config as acfg
from aegis import graph_features as gf
from aegis import io_utils as iou
from aegis import metrics as amx
from aegis.dataset_loaders import GraphBundle

settings = acfg.load_config()
acfg.set_seed(settings.seed)
GCFG = settings.graph_model
DEVICE = acfg.resolve_device(settings.device)
FEATURE_COLUMNS = tuple(gf.TRANSFER_FEATURE_COLUMNS)
EDGE_DROPOUT = float(GCFG.get("edge_dropout", 0.30))
INPUT_DROPOUT = float(GCFG.get("input_dropout", 0.30))
HIDDEN_DROPOUT = float(GCFG.get("dropout", 0.40))
print(f"device={DEVICE} features={len(FEATURE_COLUMNS)} transform=transfer_v1")

# %% [markdown]
# ## Load and split whole campaign graphs

# %%
processed = settings.paths.processed
nodes = iou.load_frame(processed / "campaign_bank_nodes.parquet")
edges = iou.load_frame(processed / "campaign_bank_edges.parquet")
posts = iou.load_frame(processed / "campaign_bank_posts.parquet")

required = {"campaign_id", "scenario"}
for name, frame in {"nodes": nodes, "edges": edges, "posts": posts}.items():
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{name} is missing {sorted(missing)}; rerun notebook 01.")

campaign_ids = sorted(nodes["campaign_id"].dropna().astype(str).unique())
if len(campaign_ids) < 4:
    raise RuntimeError("Strict campaign validation requires at least four campaigns.")
rng = np.random.default_rng(settings.seed)
rng.shuffle(campaign_ids)
n_test = max(2 if len(campaign_ids) >= 6 else 1, int(round(len(campaign_ids) * 0.25)))
n_val = max(1, int(round(len(campaign_ids) * 0.20)))
test_ids = campaign_ids[:n_test]
val_ids = campaign_ids[n_test:n_test + n_val]
train_ids = campaign_ids[n_test + n_val:]
if not train_ids:
    raise RuntimeError("No training campaigns remain after group holdout.")

campaign_split = {
    **{campaign_id: "train" for campaign_id in train_ids},
    **{campaign_id: "validation" for campaign_id in val_ids},
    **{campaign_id: "test" for campaign_id in test_ids},
}
print(pd.Series(campaign_split).value_counts().to_string())
print(json.dumps(campaign_split, indent=2))


def campaign_bundle(campaign_id: str) -> GraphBundle:
    n = nodes[nodes["campaign_id"].astype(str).eq(campaign_id)].copy()
    e = edges[edges["campaign_id"].astype(str).eq(campaign_id)].copy()
    p = posts[posts["campaign_id"].astype(str).eq(campaign_id)].copy()
    n["split"] = "test"
    return GraphBundle(
        nodes=n,
        edges=e,
        posts=p,
        name=campaign_id,
        provenance="generated",
        era="frontier_2026",
    )


bundles = {campaign_id: campaign_bundle(campaign_id) for campaign_id in campaign_ids}
feature_bundles = {
    campaign_id: gf.build_transfer_features(bundle, settings)
    for campaign_id, bundle in bundles.items()
}

# Explicit isolation gate: a user id and an edge endpoint belong to one group.
for campaign_id, bundle in bundles.items():
    known = set(bundle.nodes["user_id"].astype(str))
    assert set(bundle.edges["source"].astype(str)).issubset(known)
    assert set(bundle.edges["target"].astype(str)).issubset(known)
assert len(set(train_ids) & set(val_ids + test_ids)) == 0

# %% [markdown]
# ## Training-only scaling and disconnected PyG batches

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

train_matrix = np.concatenate([
    feature_bundles[campaign_id].X for campaign_id in train_ids
])
train_labels = np.concatenate([
    feature_bundles[campaign_id].y for campaign_id in train_ids
])
scaler = StandardScaler().fit(train_matrix)

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Batch
    from torch_geometric.nn import SAGEConv
except ImportError as exc:
    raise ImportError("Notebook 03 requires torch and torch-geometric.") from exc


def pyg(campaign_id: str):
    feature_bundle = feature_bundles[campaign_id]
    scaled = scaler.transform(feature_bundle.X).astype(np.float32)
    return gf.to_pyg_data(feature_bundle, scaled=scaled, undirected=True)


pyg_graphs = {campaign_id: pyg(campaign_id) for campaign_id in campaign_ids}
train_batch = Batch.from_data_list([pyg_graphs[c] for c in train_ids]).to(DEVICE)

classes = np.array(sorted(np.unique(train_labels)))
weights = compute_class_weight("balanced", classes=classes, y=train_labels)
class_weight = torch.ones(2, dtype=torch.float, device=DEVICE)
for label, weight in zip(classes, weights):
    class_weight[int(label)] = float(weight)
print(f"train-only class weights={class_weight.cpu().tolist()}")

# %% [markdown]
# ## GraphSAGE with feature and undirected-consistent edge dropout

# %%
class InductiveGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int, layers: int):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for index in range(layers):
            self.convs.append(SAGEConv(in_channels if index == 0 else hidden, hidden))
            self.norms.append(torch.nn.BatchNorm1d(hidden))
        self.head = torch.nn.Linear(hidden, 2)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=INPUT_DROPOUT, training=self.training)
        edge_index = gf.dropout_undirected_edges(
            edge_index, p=EDGE_DROPOUT, training=self.training
        )
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=HIDDEN_DROPOUT, training=self.training)
        return self.head(x)


model = InductiveGraphSAGE(
    len(FEATURE_COLUMNS),
    int(GCFG.get("hidden_channels", 128)),
    int(GCFG.get("num_layers", 2)),
).to(DEVICE)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=float(GCFG.get("learning_rate", 5e-3)),
    weight_decay=float(GCFG.get("weight_decay", 5e-4)),
)


def probabilities(campaign_id: str) -> np.ndarray:
    graph = pyg_graphs[campaign_id].to(DEVICE)
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(graph.x, graph.edge_index), dim=-1)[:, 1].cpu().numpy()


def reports(ids, threshold: float):
    result = {}
    for campaign_id in ids:
        y = feature_bundles[campaign_id].y
        result[campaign_id] = amx.evaluate(
            y, probabilities(campaign_id), threshold=threshold
        )
    return result


EPOCHS = int(GCFG.get("epochs", 200))
PATIENCE = int(GCFG.get("early_stopping_patience", 30))
best_macro_f1, best_state, stale = -1.0, None, 0
history = []
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(train_batch.x, train_batch.edge_index)
    loss = F.cross_entropy(logits, train_batch.y, weight=class_weight)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    optimizer.step()

    validation = reports(val_ids, threshold=0.5)
    macro_f1 = float(np.mean([report.f1 for report in validation.values()]))
    history.append({"epoch": epoch, "loss": float(loss.item()), "val_macro_f1": macro_f1})
    if macro_f1 > best_macro_f1 + 1e-4:
        best_macro_f1 = macro_f1
        best_state = copy.deepcopy(model.state_dict())
        stale = 0
    else:
        stale += 1
    if epoch == 1 or epoch % 10 == 0:
        print(f"epoch={epoch:03d} loss={loss.item():.4f} val_macro_f1={macro_f1:.4f}")
    if stale >= PATIENCE:
        print(f"early stopping at epoch {epoch}")
        break

if best_state is None:
    raise RuntimeError("Training produced no valid checkpoint.")
model.load_state_dict(best_state)
print(f"trained in {(time.time() - t0):.1f}s; best val macro F1={best_macro_f1:.4f}")

# %% [markdown]
# ## Validation-only threshold and unseen campaign evaluation

# %%
validation_y = np.concatenate([feature_bundles[c].y for c in val_ids])
validation_scores = np.concatenate([probabilities(c) for c in val_ids])
threshold, validation_report = amx.tune_threshold(
    validation_y, validation_scores, objective="f1"
)
test_reports = reports(test_ids, threshold)
test_y = np.concatenate([feature_bundles[c].y for c in test_ids])
test_scores = np.concatenate([probabilities(c) for c in test_ids])
test_micro = amx.evaluate(test_y, test_scores, threshold=threshold)
test_macro_f1 = float(np.mean([report.f1 for report in test_reports.values()]))
worst_campaign = min(test_reports, key=lambda key: test_reports[key].recall)
print(f"threshold={threshold:.3f} validation={validation_report}")
print(f"unseen test micro={test_micro}")
print(f"test macro F1={test_macro_f1:.4f}")
print(f"worst recall={test_reports[worst_campaign].recall:.4f} ({worst_campaign})")

# Features-only baseline is fit on exactly the same training campaigns.
rf = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced_subsample",
    min_samples_leaf=3,
    max_features="sqrt",
    random_state=settings.seed,
    n_jobs=-1,
).fit(scaler.transform(train_matrix), train_labels)
rf_scores = rf.predict_proba(
    np.concatenate([
        scaler.transform(feature_bundles[campaign_id].X) for campaign_id in test_ids
    ])
)[:, 1]
rf_report = amx.evaluate(test_y, rf_scores, threshold=0.5)
print(f"unseen campaign RF baseline={rf_report}")

# %% [markdown]
# ## External Cresci diagnostic (never used for model selection)

# %%
external_report = None
external_nodes_path = processed / "graph_nodes.parquet"
if external_nodes_path.exists():
    external_nodes = iou.load_frame(external_nodes_path)
    external_edges = iou.load_frame(processed / "graph_edges.parquet")
    external_posts = iou.load_frame(processed / "graph_posts.parquet")
    external_nodes["split"] = "test"
    external_bundle = GraphBundle(
        nodes=external_nodes,
        edges=external_edges,
        posts=external_posts,
        name="cresci_2017_external",
        provenance="real",
        era="legacy",
    )
    external_features = gf.build_transfer_features(external_bundle, settings)
    external_data = gf.to_pyg_data(
        external_features,
        scaled=scaler.transform(external_features.X).astype(np.float32),
        undirected=True,
    ).to(DEVICE)
    model.eval()
    with torch.no_grad():
        external_scores = torch.softmax(
            model(external_data.x, external_data.edge_index), dim=-1
        )[:, 1].cpu().numpy()
    external_report = amx.evaluate(
        external_features.y, external_scores, threshold=threshold
    )
    print(f"Cresci external diagnostic={external_report}")

# %% [markdown]
# ## Persist inference-compatible artifacts and campaign predictions

# %%
out_dir = settings.paths.graph_model
out_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(scaler, out_dir / "feature_scaler.joblib")
joblib.dump(rf, out_dir / "rf_features.joblib")
torch.save(
    {
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "architecture": "sage",
        "in_channels": len(FEATURE_COLUMNS),
        "hidden": int(GCFG.get("hidden_channels", 128)),
        "num_layers": int(GCFG.get("num_layers", 2)),
        "dropout": HIDDEN_DROPOUT,
        "input_dropout": INPUT_DROPOUT,
        "edge_dropout": EDGE_DROPOUT,
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_transform": "transfer_v1",
        "threshold": float(threshold),
    },
    out_dir / "swarm_gnn.pt",
)

score_frames = []
for campaign_id in campaign_ids:
    frame = feature_bundles[campaign_id].features.loc[
        :, ["user_id", "label", "campaign_id", "scenario"]
    ].copy()
    frame["campaign_split"] = campaign_split[campaign_id]
    frame["graph_score"] = probabilities(campaign_id)
    score_frames.append(frame)
campaign_scores = pd.concat(score_frames, ignore_index=True)
iou.save_frame(campaign_scores, processed / "graph_scores_campaigns.parquet")

metrics = {
    "protocol": "strict_inductive_campaign_holdout",
    "feature_transform": "transfer_v1",
    "feature_columns": list(FEATURE_COLUMNS),
    "campaign_split": campaign_split,
    "threshold": float(threshold),
    "validation": validation_report.to_dict(),
    "campaign_transfer": {
        "graphsage": test_micro.to_dict(),
        "macro_f1": test_macro_f1,
        "worst_campaign": worst_campaign,
        "worst_campaign_metrics": test_reports[worst_campaign].to_dict(),
        "per_campaign": {
            campaign_id: report.to_dict()
            for campaign_id, report in test_reports.items()
        },
    },
    "features_only_baseline": rf_report.to_dict(),
    "cresci_external_diagnostic": (
        external_report.to_dict() if external_report is not None else None
    ),
    "regularization": {
        "input_dropout": INPUT_DROPOUT,
        "hidden_dropout": HIDDEN_DROPOUT,
        "edge_dropout": EDGE_DROPOUT,
    },
}
iou.save_json(metrics, out_dir / "graph_metrics.json")
print(f"artifacts -> {out_dir}")
