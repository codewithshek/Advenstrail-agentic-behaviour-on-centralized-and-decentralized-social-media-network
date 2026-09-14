# %% [markdown]
# # 04 · Nested Leave-One-Campaign-Out Fusion
#
# A campaign is the unit of evaluation. Scaling, meta-model fitting,
# calibration, model selection, and threshold tuning happen without the outer
# held-out campaign.

# %%
from __future__ import annotations

import collections
import json
import sys
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
from aegis import io_utils as iou
from aegis import metrics as amx

settings = acfg.load_config()
acfg.set_seed(settings.seed)
DEVICE = acfg.resolve_device(settings.device)
FCFG = settings.fusion_model
FBETA = float(FCFG.get("fbeta", 1.5))

# %% [markdown]
# ## Build campaign-level branch scores

# %%
graph_scores = iou.load_frame(
    settings.paths.processed / "graph_scores_campaigns.parquet"
)
campaign_posts = iou.load_frame(
    settings.paths.processed / "campaign_bank_posts.parquet"
)
required = {"user_id", "campaign_id", "graph_score", "label"}
if missing := required - set(graph_scores.columns):
    raise RuntimeError(f"Graph campaign scores missing {sorted(missing)}; run notebook 03.")

text_score_path = settings.paths.processed / "text_scores_campaign_posts.parquet"
if text_score_path.exists():
    post_scores = iou.load_frame(text_score_path)
else:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = settings.paths.text_model
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    text_model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    text_model.to(DEVICE).eval()
    batch_size = int(FCFG.get("text_inference_batch_size", 64))
    scores = []
    texts = campaign_posts["text"].fillna("").astype(str).tolist()
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start:start + batch_size],
            truncation=True,
            max_length=int(settings.text_model.get("max_length", 256)),
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
        with torch.no_grad():
            probability = torch.softmax(text_model(**encoded).logits, dim=-1)[:, 1]
        scores.extend(probability.cpu().numpy().tolist())
    post_scores = campaign_posts.loc[
        :, ["post_id", "user_id", "campaign_id", "scenario"]
    ].copy()
    post_scores["text_score"] = np.asarray(scores, dtype=np.float32)
    iou.save_frame(post_scores, text_score_path)

text_accounts = (
    post_scores.groupby(["campaign_id", "user_id"], as_index=False)
    .agg(
        text_score=("text_score", "max"),
        text_mean_score=("text_score", "mean"),
        text_p95_score=("text_score", lambda values: float(np.quantile(values, 0.95))),
    )
)
frame = graph_scores.merge(
    text_accounts,
    on=["campaign_id", "user_id"],
    how="left",
    validate="one_to_one",
)
for column in ("text_score", "text_mean_score", "text_p95_score"):
    frame[column] = frame[column].fillna(0.0)
FEATURES = ["graph_score", "text_score", "text_mean_score", "text_p95_score"]
groups = frame["campaign_id"].astype(str).to_numpy()
y = frame["label"].astype(int).to_numpy()
X = frame[FEATURES].to_numpy(dtype=np.float64)
campaign_ids = sorted(np.unique(groups))
if len(campaign_ids) < 4:
    raise RuntimeError("Nested leave-one-campaign-out requires at least four campaigns.")
print(f"{len(frame):,} accounts across {len(campaign_ids)} campaigns")

# %% [markdown]
# ## Nested model selection and calibration

# %%
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logo = LeaveOneGroupOut()


def raw_candidate(kind: str, X_train, y_train, X_test):
    if kind == "graph":
        return X_test[:, 0], None
    if kind == "text":
        return X_test[:, 1], None
    if kind == "weighted":
        return 0.55 * X_test[:, 0] + 0.45 * X_test[:, 1], None
    estimator = Pipeline([
        ("scale", StandardScaler()),
        (
            "stack",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                C=float(FCFG.get("stacking_c", 1.0)),
                random_state=settings.seed,
            ),
        ),
    ])
    estimator.fit(X_train, y_train)
    return estimator.predict_proba(X_test)[:, 1], estimator


def inner_oof(kind: str, X_train, y_train, group_train):
    predictions = np.zeros(len(y_train), dtype=float)
    for fit_index, validation_index in logo.split(X_train, y_train, group_train):
        predictions[validation_index], _ = raw_candidate(
            kind,
            X_train[fit_index],
            y_train[fit_index],
            X_train[validation_index],
        )
    return predictions


def macro_and_worst(y_true, scores, group_values, threshold):
    reports = {
        campaign_id: amx.evaluate(
            y_true[group_values == campaign_id],
            scores[group_values == campaign_id],
            threshold=threshold,
            beta=FBETA,
        )
        for campaign_id in sorted(np.unique(group_values))
    }
    return (
        float(np.mean([report.f1 for report in reports.values()])),
        float(min(report.recall for report in reports.values())),
        reports,
    )


CANDIDATES = ("graph", "text", "weighted", "stacking")
outer_predictions = np.zeros(len(frame), dtype=float)
fold_rows = []
selected_kinds = []

for train_index, test_index in logo.split(X, y, groups):
    held_out = str(groups[test_index][0])
    X_train, y_train, group_train = X[train_index], y[train_index], groups[train_index]
    candidate_state = {}
    for kind in CANDIDATES:
        oof = inner_oof(kind, X_train, y_train, group_train)
        threshold, _ = amx.tune_threshold(
            y_train, oof, objective="fbeta", beta=FBETA
        )
        macro_f1, worst_recall, _ = macro_and_worst(
            y_train, oof, group_train, threshold
        )
        candidate_state[kind] = {
            "threshold": float(threshold),
            "macro_f1": macro_f1,
            "worst_recall": worst_recall,
        }

    best_simple = max(
        ("graph", "text", "weighted"),
        key=lambda name: (
            candidate_state[name]["worst_recall"],
            candidate_state[name]["macro_f1"],
        ),
    )
    stacking_gain = (
        candidate_state["stacking"]["macro_f1"]
        - candidate_state[best_simple]["macro_f1"]
    )
    selected = (
        "stacking"
        if stacking_gain >= float(FCFG.get("min_stacking_macro_gain", 0.01))
        and candidate_state["stacking"]["worst_recall"]
        >= candidate_state[best_simple]["worst_recall"]
        else best_simple
    )
    threshold = candidate_state[selected]["threshold"]

    if selected == "stacking":
        # Small campaign banks cannot support isotonic calibration reliably.
        min_class = int(np.bincount(y_train, minlength=2).min())
        method = (
            "isotonic"
            if min_class >= int(FCFG.get("isotonic_min_class_samples", 500))
            else "sigmoid"
        )
        base = Pipeline([
            ("scale", StandardScaler()),
            (
                "stack",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    C=float(FCFG.get("stacking_c", 1.0)),
                    random_state=settings.seed,
                ),
            ),
        ])
        calibration_cv = list(logo.split(X_train, y_train, group_train))
        estimator = CalibratedClassifierCV(
            estimator=base, method=method, cv=calibration_cv
        )
        estimator.fit(X_train, y_train)
        held_scores = estimator.predict_proba(X[test_index])[:, 1]
    else:
        held_scores, estimator = raw_candidate(
            selected, X_train, y_train, X[test_index]
        )
        method = "none"

    outer_predictions[test_index] = held_scores
    report = amx.evaluate(
        y[test_index], held_scores, threshold=threshold, beta=FBETA
    )
    selected_kinds.append(selected)
    fold_rows.append({
        "campaign_id": held_out,
        "selected_model": selected,
        "calibration": method,
        "threshold": float(threshold),
        **report.to_dict(),
        "candidate_selection": candidate_state,
    })
    print(f"held out {held_out}: {selected}/{method} -> {report}")

# %% [markdown]
# ## Honest aggregate and baseline comparison

# %%
fold_reports = {
    row["campaign_id"]: amx.evaluate(
        y[groups == row["campaign_id"]],
        outer_predictions[groups == row["campaign_id"]],
        threshold=row["threshold"],
        beta=FBETA,
    )
    for row in fold_rows
}
outer_binary = np.zeros(len(y), dtype=int)
for row in fold_rows:
    mask = groups == row["campaign_id"]
    outer_binary[mask] = (
        outer_predictions[mask] >= float(row["threshold"])
    ).astype(int)

micro_f1 = float(f1_score(y, outer_binary, zero_division=0))
macro_f1 = float(np.mean([report.f1 for report in fold_reports.values()]))
worst_campaign = min(fold_reports, key=lambda key: fold_reports[key].recall)
print(f"LOCO micro F1={micro_f1:.4f}")
print(f"LOCO macro F1={macro_f1:.4f}")
print(
    f"worst campaign recall={fold_reports[worst_campaign].recall:.4f} "
    f"({worst_campaign})"
)

baseline_rows = {}
for kind in ("graph", "text", "weighted"):
    scores, _ = raw_candidate(kind, X, y, X)
    threshold, _ = amx.tune_threshold(y, scores, objective="fbeta", beta=FBETA)
    macro, worst, reports = macro_and_worst(y, scores, groups, threshold)
    baseline_rows[kind] = {
        "threshold": float(threshold),
        "macro_f1": macro,
        "worst_recall": worst,
        "per_campaign": {
            campaign_id: report.to_dict()
            for campaign_id, report in reports.items()
        },
    }
print(pd.DataFrame(baseline_rows).T[["macro_f1", "worst_recall"]].to_string())

# %% [markdown]
# ## Refit deployment model after evaluation

# %%
kind_counts = collections.Counter(selected_kinds)
final_kind = kind_counts.most_common(1)[0][0]
all_oof = inner_oof(final_kind, X, y, groups)
final_threshold, _ = amx.tune_threshold(
    y, all_oof, objective="fbeta", beta=FBETA
)

if final_kind == "stacking":
    min_class = int(np.bincount(y, minlength=2).min())
    final_calibration = (
        "isotonic"
        if min_class >= int(FCFG.get("isotonic_min_class_samples", 500))
        else "sigmoid"
    )
    final_base = Pipeline([
        ("scale", StandardScaler()),
        (
            "stack",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                C=float(FCFG.get("stacking_c", 1.0)),
                random_state=settings.seed,
            ),
        ),
    ])
    final_estimator = CalibratedClassifierCV(
        estimator=final_base,
        method=final_calibration,
        cv=list(logo.split(X, y, groups)),
    ).fit(X, y)
else:
    final_calibration = "none"
    _, final_estimator = raw_candidate(final_kind, X, y, X)

artifact = {
    "kind": final_kind,
    "features": FEATURES,
    "threshold": float(final_threshold),
    "calibration": final_calibration,
    "estimator": final_estimator,
    "weights": [0.55, 0.45] if final_kind == "weighted" else None,
}
settings.paths.fusion_model.mkdir(parents=True, exist_ok=True)
joblib.dump(artifact, settings.paths.fusion_model / "fusion_model.joblib")

score_output = frame.loc[:, ["campaign_id", "user_id", "label", *FEATURES]].copy()
score_output["fusion_ooc_score"] = outer_predictions
iou.save_frame(
    score_output,
    settings.paths.processed / "fusion_scores_campaigns.parquet",
)

metrics = {
    "protocol": "nested_leave_one_campaign_out",
    "campaign_count": len(campaign_ids),
    "feature_columns": FEATURES,
    "micro_f1": micro_f1,
    "macro_f1": macro_f1,
    "worst_campaign": worst_campaign,
    "worst_campaign_metrics": fold_reports[worst_campaign].to_dict(),
    "folds": fold_rows,
    "baselines": baseline_rows,
    "final_model": {
        "kind": final_kind,
        "calibration": final_calibration,
        "threshold": float(final_threshold),
        "fit_scope": "all campaigns after evaluation",
    },
    "selection_counts": dict(kind_counts),
    "note": (
        "Cresci is an external graph diagnostic only and does not participate "
        "in fusion model selection."
    ),
}
iou.save_json(metrics, settings.paths.fusion_model / "fusion_metrics.json")
print(f"deployment artifact -> {settings.paths.fusion_model}")
