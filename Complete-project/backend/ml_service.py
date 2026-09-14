"""Lazy, thread-safe inference over the Phase 1 model artifacts."""

from __future__ import annotations

import json
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml" / "src"))

from aegis.config import load_config
from aegis.dataset_loaders import GraphBundle
from aegis.graph_features import build_features, build_transfer_features, to_pyg_data
from aegis.text_utils import (
    injection_lexical_score,
    injection_match_spans,
    injection_matches,
    llm_tell_matches,
    split_posts,
    split_sentence_spans,
)

from .analysis import (
    classify_subject,
    coordination_metrics,
    coordination_participants,
    evidence_score,
    graph_communities,
    node_evidence_score,
    suspicious_edges,
    triage_band,
)
from .models import NetworkAnalyzeRequest


class ArtifactError(RuntimeError):
    """A required Phase 1 artifact is absent or incompatible."""


def _build_swarm_gnn(checkpoint: dict[str, Any]):
    """Recreate notebook 03's architecture from its self-describing checkpoint."""
    import torch
    from torch.nn import functional
    from torch_geometric.nn import GATConv, SAGEConv

    class SwarmGNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = int(checkpoint["hidden"])
            architecture = str(checkpoint["architecture"])
            layers = int(checkpoint["num_layers"])
            dropout = float(checkpoint["dropout"])
            heads = int(checkpoint.get("heads", 4))
            self.dropout = dropout
            self.convs = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList()
            for layer in range(layers):
                source_width = (
                    int(checkpoint["in_channels"]) if layer == 0 else hidden
                )
                if architecture == "gat":
                    convolution = GATConv(
                        source_width,
                        hidden,
                        heads=heads,
                        concat=False,
                        dropout=dropout,
                    )
                else:
                    convolution = SAGEConv(source_width, hidden)
                self.convs.append(convolution)
                self.norms.append(torch.nn.BatchNorm1d(hidden))
            self.head = torch.nn.Linear(hidden, 2)

        def forward(self, x, edge_index):
            for convolution, normalisation in zip(self.convs, self.norms):
                x = convolution(x, edge_index)
                x = normalisation(x)
                x = functional.relu(x)
                x = functional.dropout(
                    x, p=self.dropout, training=self.training
                )
            return self.head(x)

    model = SwarmGNN()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


# Minimum out-of-distribution ROC-AUC before the graph model's own scores are
# allowed to raise the reported risk. Below this it is no better than a coin
# flip on traffic it has not seen.
GRAPH_TRANSFER_FLOOR = 0.70
DISPLAY_FEATURES = (
    "synchrony_score",
    "synchrony_partner_count",
    "reciprocity",
    "clustering_coefficient",
    "in_degree",
    "out_degree",
    "burstiness",
    "content_duplication_ratio",
    "cross_account_dup_ratio",
    "hashtag_jaccard_mean",
    "followers_to_following",
)


def payload_alerts(text: str) -> list[dict[str, Any]]:
    """
    Sentence-level injection and jailbreak flags.

    These come from the transparent lexical detector rather than the
    transformer: the classifier returns one probability for the whole input,
    which cannot tell an analyst *which sentence* carried the payload.
    """
    alerts: list[dict[str, Any]] = []
    for sentence_span in split_sentence_spans(text):
        sentence = str(sentence_span["text"])
        matched = injection_matches(sentence)
        if matched:
            sentence_start = int(sentence_span["start"])
            local_spans = injection_match_spans(sentence)
            alerts.append(
                {
                    "sentence": sentence,
                    "sentence_start": sentence_start,
                    "sentence_end": int(sentence_span["end"]),
                    "matched_phrases": matched,
                    "trigger_spans": [
                        {
                            **span,
                            "start": int(span["start"]) + sentence_start,
                            "end": int(span["end"]) + sentence_start,
                        }
                        for span in local_spans
                    ],
                    "severity": float(injection_lexical_score(sentence)),
                }
            )
    return alerts


class MLService:
    def __init__(self) -> None:
        self.settings = load_config(quiet=True)
        self._text_lock = threading.Lock()
        self._network_lock = threading.Lock()
        self._tokenizer = None
        self._text_model = None
        self._text_meta: dict[str, Any] | None = None
        self._network_model = None
        self._network_scaler = None
        self._network_meta: dict[str, Any] | None = None
        self._network_gnn = None
        self._network_gnn_error: str | None = None
        self._graph_transfer_auc: float | None = None

    def _load_text(self) -> None:
        if self._text_meta is not None:
            return
        model_dir = self.settings.paths.text_model
        meta_path = model_dir / "text_metrics.json"
        if not meta_path.exists():
            raise ArtifactError("text model missing; execute Phase 1 notebook 02")
        self._text_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            self._text_model = AutoModelForSequenceClassification.from_pretrained(
                str(model_dir)
            )
            self._text_model.eval()
        except Exception:  # noqa: BLE001 - any artifact/import failure uses declared fallback
            # The lexical path keeps development inference available if an
            # operator deploys without torch, while metadata makes the fallback
            # explicit in every response.
            self._tokenizer = None
            self._text_model = None

    def analyze_text(self, text: str) -> dict[str, Any]:
        with self._text_lock:
            self._load_text()
            assert self._text_meta is not None
            threshold = float(self._text_meta.get("threshold", 0.5))
            artifact_mode = str(self._text_meta.get("artifact_mode", "unknown"))

            if self._text_model is not None and self._tokenizer is not None:
                import torch

                encoded = self._tokenizer(
                    [text],
                    truncation=True,
                    max_length=int(self._text_meta.get("max_length", 256)),
                    padding=True,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    logits = self._text_model(**encoded).logits
                    score = float(torch.softmax(logits, dim=-1)[0, 1])
                model_name = str(self._text_meta.get("checkpoint", "transformer"))
                fallback = False
            else:
                score = float(injection_lexical_score(text))
                model_name = "lexical_injection_fallback"
                artifact_mode = "lexical_fallback"
                fallback = True

            warning = None
            reliable = artifact_mode == "finetuned_deberta_v3" and not fallback
            if artifact_mode != "finetuned_deberta_v3":
                warning = (
                    "Development artifact only: this checkpoint is smoke-trained "
                    "and its probability is close to chance, so the lexical "
                    "detector carries the reported risk. Run notebook 02 with "
                    "AEGIS_SMOKE_TEST=0 on a GPU for production DeBERTa-v3 weights."
                )
            if fallback:
                warning = (
                    "Transformer artifact could not be loaded; lexical fallback used."
                )

            alerts = payload_alerts(text)
            lexical = float(injection_lexical_score(text))
            return {
                "score": float(np.clip(score, 0.0, 1.0)),
                "label": "adversarial" if score >= threshold else "human_benign",
                "threshold": threshold,
                "model": model_name,
                "artifact_mode": artifact_mode,
                "artifact_reliable": reliable,
                "ai_generation_probability": float(np.clip(score, 0.0, 1.0)),
                # When the transformer is not production-grade its probability
                # is not evidence; the transparent lexical detector is what the
                # headline risk should rest on.
                "risk_score": float(np.clip(score if reliable else lexical, 0.0, 1.0)),
                "lexical_injection_score": lexical,
                "payload_alerts": alerts,
                "llm_style_phrases": llm_tell_matches(text),
                "warning": warning,
            }

    def analyze_thread(self, posts: list[str]) -> dict[str, Any]:
        """
        Score each post in a pasted thread.

        A thread is reported per post rather than as one blob because a single
        injected instruction inside an otherwise benign conversation is exactly
        what an analyst is looking for, and averaging would hide it.
        """
        if not posts:
            raise ValueError("thread contains no posts")

        results = []
        alerts: list[dict[str, Any]] = []
        scores: list[float] = []
        risks: list[float] = []
        model_name = "unknown"
        artifact_mode = "unknown"
        threshold = 0.5
        warning = None
        reliable = False

        for index, post in enumerate(posts):
            single = self.analyze_text(post)
            model_name = single["model"]
            artifact_mode = single["artifact_mode"]
            threshold = single["threshold"]
            warning = single["warning"]
            reliable = single["artifact_reliable"]
            scores.append(single["score"])
            risks.append(single["risk_score"])
            for alert in single["payload_alerts"]:
                alerts.append({**alert, "post_index": index})
            results.append(
                {
                    "post_index": index,
                    "text": post,
                    "score": single["score"],
                    "label": single["label"],
                    "payload_alerts": single["payload_alerts"],
                }
            )

        array = np.asarray(scores, dtype=float)
        risk_array = np.asarray(risks, dtype=float)
        # The thread verdict uses the strongest post, not the mean: one
        # adversarial payload compromises the thread regardless of the rest.
        peak = float(array.max())
        return {
            "model": model_name,
            "artifact_mode": artifact_mode,
            "artifact_reliable": reliable,
            "threshold": threshold,
            "posts_analyzed": len(results),
            "aggregate_score": float(array.mean()),
            "peak_score": peak,
            "risk_score": float(risk_array.max()),
            "label": "adversarial" if peak >= threshold else "human_benign",
            "flagged_posts": int((array >= threshold).sum()),
            "posts": results,
            "payload_alerts": alerts,
            "warning": warning,
        }

    def _load_network(self) -> None:
        if self._network_meta is not None:
            return
        model_dir = self.settings.paths.graph_model
        required = {
            "metadata": model_dir / "graph_metrics.json",
            "model": model_dir / "rf_features.joblib",
            "scaler": model_dir / "feature_scaler.joblib",
        }
        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            raise ArtifactError(
                f"graph artifacts missing ({', '.join(missing)}); "
                "execute Phase 1 notebook 03"
            )
        self._network_meta = json.loads(
            required["metadata"].read_text(encoding="utf-8")
        )
        self._network_model = joblib.load(required["model"])
        self._network_scaler = joblib.load(required["scaler"])
        # Notebook 03 records how the model performed on a campaign it was not
        # trained on. That transfer number, not the in-distribution one, is what
        # predicts behaviour on live submissions.
        transfer = self._network_meta.get("campaign_transfer") or {}
        graphsage = transfer.get("graphsage") or {}
        if "roc_auc" in graphsage:
            self._graph_transfer_auc = float(graphsage["roc_auc"])
        checkpoint_path = model_dir / "swarm_gnn.pt"
        if checkpoint_path.exists():
            try:
                import torch

                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                self._network_gnn = _build_swarm_gnn(checkpoint)
            except Exception as exc:  # noqa: BLE001 - RF is the declared fallback
                self._network_gnn_error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _bundle(payload: NetworkAnalyzeRequest) -> GraphBundle:
        nodes = pd.DataFrame([node.model_dump() for node in payload.nodes])
        nodes["label"] = 0  # inference has no ground-truth label
        nodes["split"] = "test"
        edges = pd.DataFrame(
            [edge.model_dump() for edge in payload.edges],
            columns=["source", "target", "relation"],
        )
        posts = pd.DataFrame(
            [post.model_dump() for post in payload.posts],
            columns=[
                "post_id",
                "user_id",
                "text",
                "created_at",
                "hashtags",
                "mentions",
            ],
        )
        if not posts.empty:
            posts["created_at"] = pd.to_datetime(
                posts["created_at"], errors="coerce", utc=True
            )
        return GraphBundle(
            name="api_network",
            nodes=nodes,
            edges=edges,
            posts=posts,
            provenance="API_INPUT",
            era="live",
        )

    def analyze_network(
        self,
        payload: NetworkAnalyzeRequest,
        *,
        target: str | None = None,
        text_score: float = 0.0,
        payload_alert_count: int = 0,
    ) -> dict[str, Any]:
        with self._network_lock:
            self._load_network()
            assert self._network_meta is not None
            bundle = self._bundle(payload)
            features = (
                build_transfer_features(bundle, self.settings)
                if self._network_meta.get("feature_transform") == "transfer_v1"
                else build_features(bundle, self.settings)
            )
            feature_names = list(self._network_meta["feature_columns"])
            matrix = features.features.loc[:, feature_names].to_numpy(
                dtype=np.float32
            )
            scaled = self._network_scaler.transform(matrix)
            if self._network_gnn is not None:
                import torch

                graph_data = to_pyg_data(
                    features, scaled=scaled, undirected=True
                )
                with torch.no_grad():
                    scores = torch.softmax(
                        self._network_gnn(
                            graph_data.x, graph_data.edge_index
                        ),
                        dim=-1,
                    )[:, 1].numpy()
                model_name = "graphsage_coordination_gnn"
            else:
                scores = self._network_model.predict_proba(scaled)[:, 1]
                model_name = "random_forest_coordination_features"
            threshold = float(self._network_meta.get("threshold") or 0.5)

            edges = [edge.model_dump() for edge in payload.edges]
            node_metadata = {
                str(node.user_id): node.model_dump()
                for node in payload.nodes
            }
            community_labels = {
                user_id: (
                    metadata.get("community_id")
                    or (
                        str(metadata["cluster_id"])
                        if metadata.get("cluster_id") is not None
                        else None
                    )
                )
                for user_id, metadata in node_metadata.items()
            }
            label_to_cluster = {
                label: index
                for index, label in enumerate(
                    sorted(
                        {
                            str(label)
                            for label in community_labels.values()
                            if label is not None
                        }
                    ),
                    start=1,
                )
            }
            cluster_ids = graph_communities(
                node_metadata,
                edges,
                supplied={
                    user_id: (
                        label_to_cluster.get(str(community_labels[user_id]))
                        if community_labels[user_id] is not None
                        else None
                    )
                    for user_id in node_metadata
                },
            )
            cluster_sizes = Counter(cluster_ids.values())
            rows = []
            for idx, user_id in enumerate(features.features["user_id"]):
                # Selecting from a mixed-type DataFrame row gives an object
                # Series even though these columns are numeric. Coerce before
                # ranking explanations; pandas rejects nlargest(object).
                raw = pd.to_numeric(
                    features.features.iloc[idx][feature_names],
                    errors="coerce",
                ).fillna(0.0)
                top = raw.abs().nlargest(5).index
                user_key = str(user_id)
                metadata = node_metadata.get(user_key, {})
                cluster_id = int(cluster_ids.get(user_key, 0))
                analyst_features = {
                    name: float(raw.get(name, 0.0))
                    for name in DISPLAY_FEATURES
                }
                rows.append(
                    {
                        "user_id": user_key,
                        "screen_name": metadata.get("screen_name"),
                        "score": float(np.clip(scores[idx], 0.0, 1.0)),
                        "evidence_score": node_evidence_score(raw),
                        "label": (
                            "adversarial"
                            if scores[idx] >= threshold
                            else "human_benign"
                        ),
                        "top_features": {
                            name: float(raw[name]) for name in top
                        },
                        "features": analyst_features,
                        "is_target": user_key == str(target),
                        "followers_count": float(
                            metadata.get("followers_count") or 0.0
                        ),
                        "following_count": float(
                            metadata.get("following_count") or 0.0
                        ),
                        "statuses_count": float(
                            metadata.get("statuses_count") or 0.0
                        ),
                        "account_age_days": float(
                            metadata.get("account_age_days") or 0.0
                        ),
                        "verified": bool(metadata.get("verified", False)),
                        "description": str(metadata.get("description") or ""),
                        "source_dataset": metadata.get("source_dataset"),
                        "source_group": metadata.get("source_group"),
                        "ground_truth": (
                            metadata.get("ground_truth")
                            or metadata.get("label_name")
                        ),
                        "provenance": metadata.get("provenance"),
                        "crawl_era": metadata.get("crawl_era"),
                        "cluster_id": cluster_id,
                        "community_label": community_labels.get(user_key),
                        "cluster_size": int(cluster_sizes.get(cluster_id, 1)),
                    }
                )

            aggregate = float(np.mean(scores)) if len(scores) else 0.0
            warning = None
            if not payload.edges or not payload.posts:
                warning = (
                    "Network input lacks edges or post timelines; structural, "
                    "temporal, and content coordination evidence is incomplete."
                )
            if self._network_gnn_error:
                warning = (
                    f"GraphSAGE artifact unavailable ({self._network_gnn_error}); "
                    "Random Forest fallback used."
                )

            metrics = coordination_metrics(features.features)
            flagged = {row["user_id"] for row in rows if row["label"] == "adversarial"}
            coordinated = coordination_participants(features.features)
            for row in rows:
                row["coordinated"] = row["user_id"] in coordinated
            clusters = []
            for cluster_id in sorted(cluster_sizes):
                members = [
                    row for row in rows if row["cluster_id"] == cluster_id
                ]
                truth_counts = Counter(
                    str(row["ground_truth"])
                    for row in members
                    if row.get("ground_truth")
                )
                clusters.append(
                    {
                        "cluster_id": int(cluster_id),
                        "size": len(members),
                        "coordinated_accounts": sum(
                            bool(row["coordinated"]) for row in members
                        ),
                        "mean_evidence_score": float(
                            np.mean(
                                [row["evidence_score"] for row in members]
                            )
                        ),
                        "ground_truth_counts": dict(truth_counts),
                    }
                )
            evidence = evidence_score(
                metrics,
                coordinated_accounts=len(coordinated),
                total_nodes=len(rows),
            )
            # The reported risk takes the strongest signal rather than a mean:
            # a text payload and measured coordination are each sufficient
            # grounds for review, so averaging would bury either one. The graph
            # model only joins that comparison when its transfer score says it
            # generalises beyond its training campaign; below that bar its
            # output is reported but not acted on.
            model_trusted = (
                self._graph_transfer_auc is not None
                and self._graph_transfer_auc >= GRAPH_TRANSFER_FLOOR
            )
            verdict = classify_subject(
                metrics=metrics,
                coordinated_accounts=len(coordinated),
                flagged_nodes=len(flagged),
                total_nodes=len(rows),
                network_score=aggregate,
                text_score=text_score,
                payload_alerts=payload_alert_count,
                model_trusted=model_trusted,
            )
            signals = [text_score, evidence]
            if model_trusted:
                signals.append(aggregate)
            elif self._graph_transfer_auc is not None:
                warning = (
                    "Graph model transfer ROC-AUC is "
                    f"{self._graph_transfer_auc:.2f} on an unseen campaign, so "
                    "its node scores are shown for reference only and the "
                    "verdict rests on the measured coordination features. "
                    "Retrain notebook 03 with AEGIS_SMOKE_TEST=0 to restore it."
                )
            headline = float(max(signals))
            return {
                "model": model_name,
                "threshold": threshold,
                "aggregate_score": headline,
                "score_components": {
                    "graph_model": aggregate,
                    "text_branch": float(text_score),
                    "coordination_evidence": evidence,
                },
                "model_trusted": model_trusted,
                "flagged_nodes": len(flagged),
                "total_nodes": len(rows),
                "nodes": rows,
                "coordinated_accounts": len(coordinated),
                "cluster_count": len(clusters),
                "clusters": clusters,
                "classification": verdict["classification"],
                "band": triage_band(headline),
                "reasons": verdict["reasons"],
                "coordination": metrics,
                "edges": edges,
                # Highlight the ring itself: links where both endpoints carry
                # coordination evidence or were flagged by the graph model.
                "suspicious_edges": suspicious_edges(edges, coordinated | flagged),
                "warning": warning,
            }

    def analyze_account(self, handle: str, *, peers: int = 5) -> dict[str, Any]:
        """
        Run the full hybrid path for one handle against simulated activity.

        The simulation is explicit in the response; see ``simulation.py`` for
        why live retrieval is not attempted.
        """
        from .simulation import simulate_account

        account = simulate_account(handle, peers=peers)
        thread = self.analyze_thread([str(post["text"]) for post in account.posts])
        # Re-attribute each alert to the account that posted it so the analyst
        # can act on a specific node in the graph.
        for alert in thread["payload_alerts"]:
            index = int(alert.get("post_index") or 0)
            if 0 <= index < len(account.posts):
                alert["user_id"] = str(account.posts[index]["user_id"])

        network = self.analyze_network(
            NetworkAnalyzeRequest.model_validate(account.as_network_payload()),
            target=account.handle,
            text_score=thread["risk_score"],
            payload_alert_count=len(thread["payload_alerts"]),
        )

        threat_score = float(network["aggregate_score"])
        return {
            "handle": account.handle,
            "data_source": "simulated",
            "simulation_note": account.note,
            "classification": network["classification"],
            "band": triage_band(threat_score),
            "threat_score": threat_score,
            "reasons": network["reasons"],
            "text": thread,
            "network": network,
            "warning": thread["warning"] or network["warning"],
        }


service = MLService()

__all__ = ["ArtifactError", "MLService", "payload_alerts", "service", "split_posts"]
