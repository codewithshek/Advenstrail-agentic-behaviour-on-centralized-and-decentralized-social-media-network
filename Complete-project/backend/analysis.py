"""
Analyst-facing interpretation of the raw model outputs.

The models return probabilities; an analyst needs a verdict, the evidence behind
it, and the specific accounts and sentences to look at. Everything in this
module is derived from values the Phase 1 models and features actually produce —
no score is invented here, and every threshold used for the verdict is declared
in one place so it can be audited and tuned.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

# Verdict thresholds. These separate *coordination* (multiple accounts acting
# together) from *individual* automation, which is the distinction between a
# swarm and a lone spambot. They are deliberately conservative: coordination
# requires corroboration from two independent feature families (timing and
# content/structure) rather than any single metric.
SYNCHRONY_FLOOR = 0.35
DUPLICATION_FLOOR = 0.30
RECIPROCITY_FLOOR = 0.50
SWARM_MIN_FLAGGED = 2
SPAMBOT_BURSTINESS = 0.30
# ``followers_to_following`` is log1p(followers) - log1p(following), so an
# amplifier account that follows far more than it is followed sits well below
# zero. -1.5 is roughly a 4.5x following-to-followers imbalance.
SPAMBOT_FOLLOW_LOG_RATIO = -1.5

SUBJECT_LABELS = ("human", "simple_spambot", "coordinated_ai_agent_swarm")


def _mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(np.nan_to_num(values.mean(), nan=0.0))


def _max(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(np.nan_to_num(values.max(), nan=0.0))


def coordination_metrics(features: pd.DataFrame) -> dict[str, float]:
    """
    Summarise the structural coordination features for the whole submission.

    ``temporal_synchrony`` and ``reciprocity_rate`` are the two headline numbers
    in the dashboard; the rest give the analyst the supporting detail.
    """
    return {
        "temporal_synchrony": _mean(features, "synchrony_score"),
        "peak_synchrony": _max(features, "synchrony_score"),
        "synchronised_partners": _max(features, "synchrony_partner_count"),
        "reciprocity_rate": _mean(features, "reciprocity"),
        "peak_reciprocity": _max(features, "reciprocity"),
        "cross_account_duplication": _mean(features, "cross_account_dup_ratio"),
        "peak_cross_account_duplication": _max(
            features, "cross_account_dup_ratio"
        ),
        "self_duplication": _mean(features, "content_duplication_ratio"),
        "burstiness": _mean(features, "burstiness"),
        "circadian_flatness": _mean(features, "circadian_flatness"),
        "hashtag_overlap": _mean(features, "hashtag_jaccard_mean"),
        "follower_following_log_ratio": _mean(features, "followers_to_following"),
    }


def coordination_participants(features: pd.DataFrame) -> set[str]:
    """
    Accounts whose own features carry coordination evidence.

    This is deliberately measured from the features rather than taken from the
    model's node scores. The checked-in graph artifact is smoke-trained and sits
    near chance on small live submissions, so basing the verdict on its flags
    would make the report unstable. Synchrony, duplication, and reciprocity are
    deterministic measurements of the submitted activity and hold up on their
    own.
    """
    if features.empty or "user_id" not in features.columns:
        return set()

    def column(name: str) -> pd.Series:
        if name not in features.columns:
            return pd.Series(0.0, index=features.index)
        return pd.to_numeric(features[name], errors="coerce").fillna(0.0)

    timing = column("synchrony_score") >= SYNCHRONY_FLOOR
    corroborated = (column("cross_account_dup_ratio") >= DUPLICATION_FLOOR) | (
        column("reciprocity") >= RECIPROCITY_FLOOR
    )
    selected = features.loc[timing & corroborated, "user_id"]
    return {str(user_id) for user_id in selected}


def node_evidence_score(row: pd.Series | dict[str, Any]) -> float:
    """Transparent per-account risk from measured coordination features."""

    def value(name: str) -> float:
        raw = row.get(name, 0.0)
        try:
            return float(np.nan_to_num(float(raw), nan=0.0))
        except (TypeError, ValueError):
            return 0.0

    partners = min(max(value("synchrony_partner_count") / 10.0, 0.0), 1.0)
    contributions = (
        (0.40, min(max(value("synchrony_score"), 0.0), 1.0)),
        (0.25, min(max(value("cross_account_dup_ratio"), 0.0), 1.0)),
        (0.20, min(max(value("reciprocity"), 0.0), 1.0)),
        (0.10, min(max(value("hashtag_jaccard_mean"), 0.0), 1.0)),
        (0.05, partners),
    )
    return float(min(1.0, sum(weight * score for weight, score in contributions)))


def graph_communities(
    user_ids: Iterable[str],
    edges: Iterable[dict[str, Any]],
    supplied: dict[str, int | None] | None = None,
) -> dict[str, int]:
    """
    Return deterministic community ids, retaining trusted source ids if supplied.

    Louvain is used only for display grouping; it does not affect threat labels.
    """
    ids = sorted({str(user_id) for user_id in user_ids})
    supplied = supplied or {}
    if any(supplied.get(user_id) is not None for user_id in ids):
        return {
            user_id: int(supplied.get(user_id) or 0)
            for user_id in ids
        }

    graph = nx.Graph()
    graph.add_nodes_from(ids)
    graph.add_edges_from(
        (str(edge.get("source", "")), str(edge.get("target", "")))
        for edge in edges
        if str(edge.get("source", "")) in graph
        and str(edge.get("target", "")) in graph
    )
    communities = nx.community.louvain_communities(
        graph, seed=42, resolution=1.2
    )
    ordered = sorted(
        communities,
        key=lambda members: (-len(members), min(map(str, members))),
    )
    return {
        str(user_id): cluster_id
        for cluster_id, members in enumerate(ordered, start=1)
        for user_id in members
    }


def suspicious_edges(
    edges: Iterable[dict[str, Any]],
    flagged: set[str],
) -> list[dict[str, Any]]:
    """
    Edges where *both* endpoints are flagged.

    A single flagged account interacting with the wider platform is normal; a
    closed loop of flagged accounts amplifying each other is the re-share ring
    that manipulates trending ranking, so only mutual links are marked.
    """
    marked: list[dict[str, Any]] = []
    for edge in edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source in flagged and target in flagged:
            marked.append(
                {
                    "source": source,
                    "target": target,
                    "relation": str(edge.get("relation", "interacts")),
                }
            )
    return marked


def classify_subject(
    *,
    metrics: dict[str, float],
    coordinated_accounts: int,
    flagged_nodes: int,
    total_nodes: int,
    network_score: float,
    text_score: float,
    payload_alerts: int,
    model_trusted: bool = True,
) -> dict[str, Any]:
    """
    Decide between human, lone spambot, and coordinated swarm.

    The ordering matters. Coordination is checked first because a swarm is also
    individually automated: if the timing and content evidence shows accounts
    moving together, "spambot" would understate the finding.
    """
    reasons: list[str] = []

    peak_duplication = metrics.get(
        "peak_cross_account_duplication",
        metrics["cross_account_duplication"],
    )
    peak_reciprocity = metrics.get(
        "peak_reciprocity",
        metrics["reciprocity_rate"],
    )
    duplicated = peak_duplication >= DUPLICATION_FLOOR
    reciprocal = peak_reciprocity >= RECIPROCITY_FLOOR

    # Coordination requires timing evidence across at least two accounts plus a
    # corroborating content or structural signal.
    coordinated = coordinated_accounts >= SWARM_MIN_FLAGGED and (
        duplicated or reciprocal
    )

    if coordinated:
        label = "coordinated_ai_agent_swarm"
        reasons.append(
            f"{coordinated_accounts} of {total_nodes} accounts post in synchrony "
            f"(peak {metrics['peak_synchrony']:.2f}) with corroborating evidence"
        )
        if duplicated:
            reasons.append(
                f"peak cross-account content duplication {peak_duplication:.2f} "
                "indicates a shared script"
            )
        if reciprocal:
            reasons.append(
                f"peak reciprocity {peak_reciprocity:.2f} indicates mutual "
                "amplification rather than organic conversation"
            )
    elif (
        metrics["burstiness"] >= SPAMBOT_BURSTINESS
        or metrics["follower_following_log_ratio"] <= SPAMBOT_FOLLOW_LOG_RATIO
    ):
        label = "simple_spambot"
        reasons.append(
            f"automation-like individual behaviour (burstiness "
            f"{metrics['burstiness']:.2f}, follower/following log-ratio "
            f"{metrics['follower_following_log_ratio']:.2f}) without "
            "multi-account coordination evidence"
        )
    else:
        label = "human"
        reasons.append(
            "no coordination evidence: synchrony, duplication, and reciprocity "
            "all below their decision floors"
        )

    if payload_alerts:
        reasons.append(
            f"{payload_alerts} sentence(s) contain prompt-injection or jailbreak "
            "phrasing"
        )
    if flagged_nodes:
        reasons.append(
            f"graph model flags {flagged_nodes} of {total_nodes} accounts"
            + ("" if model_trusted else " (reference only; see warning)")
        )
    if text_score >= 0.5:
        reasons.append(
            f"text branch scores {text_score:.2f} for AI-generated or "
            "adversarial content"
        )
    if model_trusted and network_score >= 0.5:
        reasons.append(f"network branch scores {network_score:.2f}")

    return {"classification": label, "reasons": reasons}


def evidence_score(
    metrics: dict[str, float],
    *,
    coordinated_accounts: int,
    total_nodes: int,
) -> float:
    """
    A transparent score built from the coordination measurements themselves.

    The graph artifact shipped with the repository is smoke-trained and returns
    values near 0.5 on small live submissions, so using it alone would report
    "50% risk" for both an obvious re-share ring and ordinary traffic. This
    score is computed directly from the features that carry the evidence, so the
    number the analyst sees moves with the evidence in the report. Weights are
    declared here rather than learned; when notebook 03 is retrained on GPU the
    model score becomes the stronger signal and dominates the combination.
    """
    coverage = coordinated_accounts / max(total_nodes, 1)
    contributions = (
        (0.35, min(max(metrics["peak_synchrony"], 0.0), 1.0)),
        (
            0.30,
            min(
                max(
                    metrics.get(
                        "peak_cross_account_duplication",
                        metrics["cross_account_duplication"],
                    ),
                    0.0,
                ),
                1.0,
            ),
        ),
        (
            0.15,
            min(
                max(
                    metrics.get("peak_reciprocity", metrics["reciprocity_rate"]),
                    0.0,
                ),
                1.0,
            ),
        ),
        (0.20, min(max(coverage, 0.0), 1.0)),
    )
    return float(
        min(1.0, sum(weight * value for weight, value in contributions))
    )


def triage_band(score: float) -> str:
    """Four-band triage used by the dashboard gauge."""
    if score >= 0.85:
        return "critical"
    if score >= 0.60:
        return "high"
    if score >= 0.35:
        return "elevated"
    return "low"


__all__ = [
    "SUBJECT_LABELS",
    "classify_subject",
    "coordination_metrics",
    "coordination_participants",
    "evidence_score",
    "graph_communities",
    "node_evidence_score",
    "suspicious_edges",
    "triage_band",
]
