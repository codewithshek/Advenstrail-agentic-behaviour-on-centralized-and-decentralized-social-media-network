"""
aegis.graph_features
====================

The "coordination physics" layer: turns an interaction graph plus a posting
timeline into a per-account feature matrix that separates *automation acting
alone* from *automation acting together*.

The premise
-----------
A single well-built bot is hard to catch — a 2026 LLM agent writes fluent,
varied, contextually appropriate text, so the text branch degrades badly on it
(this is the headline finding of TwiBot-24). What automation cannot easily hide
is **joint behaviour**: to have influence a swarm must act in concert, and
acting in concert leaves statistical fingerprints in *time* and in *structure*
that a lone human account does not produce.

So every feature here answers one of three questions:

**Timing** — does this account post when others post, more often than chance?
    ``synchrony_score``, ``synchrony_partner_count``, ``burstiness``,
    ``memory_coefficient``, ``posting_entropy``, ``circadian_flatness``
**Structure** — does it sit inside a mutually-boosting clique?
    ``reciprocity``, ``clustering_coefficient``, ``in_degree``, ``out_degree``,
    ``degree_ratio``, ``followers_to_following``
**Content** — does it recycle text, its own or its neighbours'?
    ``content_duplication_ratio``, ``cross_account_dup_ratio``,
    ``hashtag_jaccard_mean``

Two engineering commitments
---------------------------
1. **Null models, not raw counts.** Raw co-posting counts are dominated by how
   *much* an account posts and by global events — during a breaking-news minute
   half the platform is "synchronised". Every timing feature is therefore scored
   against what independence would predict, so a prolific account is not
   flagged for being prolific and a viral moment does not implicate everyone in
   it. This is the difference between a feature and an artefact.
2. **Near-linear cost.** TwiBot-22 has ~1M users. Nothing here is
   O(accounts²): co-posting is found by bucketing time and only pairing
   accounts *inside* a bucket, degree/reciprocity are groupby/merge operations,
   and duplicate detection reuses the MinHash+LSH from ``text_utils``. Buckets
   that are pathologically large (a platform-wide event) are subsampled with a
   warning rather than silently blowing up.

Dependencies are optional: NetworkX and PyTorch Geometric are used when present
and cleanly substituted when not, so this module imports and runs in a bare
environment.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .config import Settings, get_logger

log = get_logger("aegis.graph")

# Order matters: it is the column order of the feature matrix and therefore the
# input order of the GNN. Keep it in sync with graph_model.features in the YAML.
FEATURE_COLUMNS: Tuple[str, ...] = (
    "synchrony_score",
    "synchrony_partner_count",
    "reciprocity",
    "clustering_coefficient",
    "in_degree",
    "out_degree",
    "degree_ratio",
    "burstiness",
    "memory_coefficient",
    "posting_entropy",
    "circadian_flatness",
    "content_duplication_ratio",
    "cross_account_dup_ratio",
    "hashtag_jaccard_mean",
    "account_age_days",
    "followers_to_following",
)

# Stable, topology-transfer feature contract. Profile shortcuts deliberately
# stay out: account age and follower ratios are crawl/platform artefacts that
# can make an in-domain benchmark look excellent while failing on a new graph.
TRANSFER_FEATURE_COLUMNS: Tuple[str, ...] = (
    "synchrony_score",
    "synchrony_partner_count",
    "reciprocity",
    "clustering_coefficient",
    "in_degree",
    "out_degree",
    "degree_ratio",
    "burstiness",
    "memory_coefficient",
    "posting_entropy",
    "circadian_flatness",
    "content_duplication_ratio",
    "cross_account_dup_ratio",
    "hashtag_jaccard_mean",
)

# Above this many distinct accounts in one time bucket we stop treating the
# bucket as evidence of coordination and subsample it. A minute in which 40k
# accounts post is a news event, not a swarm.
MAX_BUCKET_ACCOUNTS = 400

# Guard for the exact clustering-coefficient computation.
CLUSTERING_EXACT_MAX_NODES = 60_000


# --------------------------------------------------------------------------- #
# Timing features
# --------------------------------------------------------------------------- #
def _inter_event_stats(timestamps: np.ndarray) -> Tuple[float, float]:
    """
    Burstiness and memory coefficient of an inter-event-time sequence.

    * **Burstiness** B = (σ − μ)/(σ + μ)  — Goh & Barabási (2008).
      B → +1 is bursty (long silences punctuated by rapid-fire activity),
      B → 0 is Poisson-random, B → −1 is perfectly regular. Scheduled
      automation lands near −1; a human lands in the middle; an agent reacting
      to a cue lands high.
    * **Memory** M = lag-1 autocorrelation of the inter-event times. Near 0 for
      humans, strongly positive for accounts on a fixed cadence.

    Both are scale-free, which is why they are used instead of raw post rate:
    they describe the *shape* of the timeline, not its volume.
    """
    if timestamps.size < 3:
        return 0.0, 0.0
    gaps = np.diff(np.sort(timestamps))
    gaps = gaps[gaps >= 0]
    if gaps.size < 2:
        return 0.0, 0.0

    mean = float(gaps.mean())
    std = float(gaps.std())
    burstiness = (std - mean) / (std + mean) if (std + mean) > 0 else 0.0

    if gaps.size < 3:
        return burstiness, 0.0
    first, second = gaps[:-1], gaps[1:]
    m1, m2 = first.mean(), second.mean()
    s1, s2 = first.std(), second.std()
    memory = float(((first - m1) * (second - m2)).mean() / (s1 * s2)) if s1 > 0 and s2 > 0 else 0.0
    return float(np.clip(burstiness, -1, 1)), float(np.clip(memory, -1, 1))


def _hour_histogram_features(hours: np.ndarray) -> Tuple[float, float]:
    """
    Entropy and circadian flatness of the hour-of-day posting histogram.

    A human sleeps. That produces a low-entropy, high-amplitude daily rhythm.
    An always-on agent produces a flat histogram — high entropy, low amplitude.
    ``circadian_flatness`` is measured from the magnitude of the 24-hour Fourier
    component, which is a cleaner read on "has a daily rhythm" than variance is,
    because it is insensitive to how many distinct hours were used.
    """
    if hours.size == 0:
        return 0.0, 0.0
    counts = np.bincount(hours.astype(int) % 24, minlength=24).astype(float)
    total = counts.sum()
    if total <= 0:
        return 0.0, 0.0
    probability = counts / total

    nonzero = probability[probability > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum() / np.log2(24))

    angles = 2.0 * np.pi * np.arange(24) / 24.0
    amplitude = float(
        np.hypot((probability * np.cos(angles)).sum(), (probability * np.sin(angles)).sum())
    )
    # amplitude is 0 for a flat histogram and up to ~1/… for a single spike;
    # normalise against the theoretical max for a delta distribution.
    flatness = float(np.clip(1.0 - amplitude / (1.0 / 1.0), 0.0, 1.0))
    return entropy, flatness


def compute_temporal_features(posts: pd.DataFrame) -> pd.DataFrame:
    """Per-account burstiness, memory, entropy, circadian flatness, volume."""
    if posts.empty:
        return pd.DataFrame(
            columns=[
                "user_id", "burstiness", "memory_coefficient",
                "posting_entropy", "circadian_flatness", "n_posts",
            ]
        )

    frame = posts.loc[:, ["user_id", "created_at"]].dropna().copy()
    frame["epoch"] = frame["created_at"].astype("int64") // 10**9
    frame["hour"] = frame["created_at"].dt.hour

    rows: List[Dict[str, Any]] = []
    for user_id, group in frame.groupby("user_id", sort=False):
        burstiness, memory = _inter_event_stats(group["epoch"].to_numpy())
        entropy, flatness = _hour_histogram_features(group["hour"].to_numpy())
        rows.append(
            {
                "user_id": user_id,
                "burstiness": burstiness,
                "memory_coefficient": memory,
                "posting_entropy": entropy,
                "circadian_flatness": flatness,
                "n_posts": len(group),
            }
        )
    return pd.DataFrame(rows)


def compute_synchrony(
    posts: pd.DataFrame,
    *,
    window_seconds: int = 60,
    min_events: int = 3,
    lift_threshold: float = 2.0,
    max_bucket_accounts: int = MAX_BUCKET_ACCOUNTS,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Score how much each account co-posts with others *beyond chance*.

    Method
    ------
    1. Floor every timestamp into a bucket of ``window_seconds``. Two accounts
       posting in the same bucket is one co-occurrence. (Also checks the
       neighbouring bucket, so a pair straddling a boundary is not missed.)
    2. Count observed co-occurrences per unordered pair, only pairing accounts
       *within* a bucket — so cost scales with Σ(bucket size²), not with
       accounts².
    3. Compare to the expectation under independence. If account *u* is active
       in ``a_u`` of ``W`` buckets, two independent accounts should share
       ``a_u · a_v / W`` buckets. The **lift** is observed / expected.
    4. Convert lift to a Poisson-style significance so that "3 co-posts when
       0.1 were expected" outranks "30 co-posts when 25 were expected", and
       squash to [0, 1].

    Returns per account: ``synchrony_score`` (max significance over partners,
    squashed), ``synchrony_partner_count`` (partners clearing both thresholds),
    and ``synchrony_mean_lift``.
    """
    empty = pd.DataFrame(
        columns=["user_id", "synchrony_score", "synchrony_partner_count", "synchrony_mean_lift"]
    )
    if posts.empty or "created_at" not in posts.columns:
        return empty

    frame = posts.loc[:, ["user_id", "created_at"]].dropna().copy()
    if frame.empty:
        return empty

    epoch = frame["created_at"].astype("int64") // 10**9
    frame["bucket"] = (epoch // int(max(1, window_seconds))).astype("int64")
    # One account posting 20 times in a minute is a separate signal (burstiness);
    # here each (account, bucket) counts once so volume cannot fake synchrony.
    presence = frame.drop_duplicates(["user_id", "bucket"])

    activity = presence.groupby("user_id").size()
    n_windows = int(presence["bucket"].nunique())
    if n_windows < 2:
        return empty

    rng = np.random.default_rng(seed)
    observed: Dict[Tuple[str, str], int] = defaultdict(int)
    oversized = 0

    # Bucket -> members, plus each account's own bucket set for the ±1 sweep.
    members: Dict[int, List[str]] = {
        bucket: group["user_id"].tolist()
        for bucket, group in presence.groupby("bucket", sort=True)
    }

    for bucket, users in members.items():
        # Straddle guard: include the next bucket's members so a pair separated
        # by one second across a boundary still registers.
        neighbours = members.get(bucket + 1, [])
        pool = list(dict.fromkeys(users + neighbours))
        if len(pool) < 2:
            continue
        if len(pool) > max_bucket_accounts:
            oversized += 1
            pool = list(rng.choice(pool, size=max_bucket_accounts, replace=False))
        pool.sort()
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                observed[(pool[i], pool[j])] += 1

    if oversized:
        log.warning(
            "synchrony: %d time bucket(s) exceeded %d accounts and were "
            "subsampled. Those are platform-wide events, not swarms — but the "
            "subsampling does make the score for accounts inside them noisier.",
            oversized, max_bucket_accounts,
        )

    best: Dict[str, float] = defaultdict(float)
    partners: Dict[str, int] = defaultdict(int)
    lift_sum: Dict[str, float] = defaultdict(float)
    lift_n: Dict[str, int] = defaultdict(int)

    for (u, v), count in observed.items():
        if count < min_events:
            continue
        expected = float(activity.get(u, 0)) * float(activity.get(v, 0)) / float(n_windows)
        expected = max(expected, 1e-9)
        lift = count / expected
        # Poisson-ish significance: excess over expectation in units of its own
        # standard deviation. Keeps small-but-shocking coincidences ahead of
        # large-but-unsurprising ones.
        significance = (count - expected) / math.sqrt(expected)
        score = 1.0 - math.exp(-max(significance, 0.0) / 6.0)
        for account in (u, v):
            best[account] = max(best[account], score)
            lift_sum[account] += lift
            lift_n[account] += 1
            if lift >= lift_threshold:
                partners[account] += 1

    accounts = sorted(set(activity.index))
    return pd.DataFrame(
        {
            "user_id": accounts,
            "synchrony_score": [best.get(a, 0.0) for a in accounts],
            "synchrony_partner_count": [partners.get(a, 0) for a in accounts],
            "synchrony_mean_lift": [
                lift_sum.get(a, 0.0) / lift_n[a] if lift_n.get(a) else 0.0 for a in accounts
            ],
        }
    )


# --------------------------------------------------------------------------- #
# Structural features
# --------------------------------------------------------------------------- #
def compute_structural_features(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    exact_clustering_max_nodes: int = CLUSTERING_EXACT_MAX_NODES,
) -> pd.DataFrame:
    """
    Degree, reciprocity and clustering, computed with pandas rather than
    NetworkX so it stays fast on million-node graphs (and works without it).

    ``reciprocity`` is the *local* variety: of the accounts this one follows,
    what fraction follow back. A mutually-boosting clique pushes this toward 1;
    a broadcast account or a scraper sits near 0. It is the single most
    discriminative structural feature for the Cresci-style retweet rings.
    """
    ids = nodes["user_id"].astype(str)
    out = pd.DataFrame({"user_id": ids})

    if edges.empty:
        for column in (
            "in_degree", "out_degree", "degree_ratio",
            "reciprocity", "clustering_coefficient",
        ):
            out[column] = 0.0
        log.warning(
            "structural features: edge list is empty, so every structural "
            "feature is zero. The graph branch is effectively disabled — check "
            "that the edge file loaded and its endpoints match the label set."
        )
        return out

    src = edges["source"].astype(str)
    dst = edges["target"].astype(str)

    out_degree = src.value_counts()
    in_degree = dst.value_counts()
    out["out_degree"] = out["user_id"].map(out_degree).fillna(0.0).astype(float)
    out["in_degree"] = out["user_id"].map(in_degree).fillna(0.0).astype(float)
    out["degree_ratio"] = out["out_degree"] / (out["in_degree"] + 1.0)

    # Reciprocity via a self-merge on the reversed edge list.
    directed = pd.DataFrame({"source": src, "target": dst}).drop_duplicates()
    reverse = directed.rename(columns={"source": "target", "target": "source"})
    mutual = directed.merge(reverse, on=["source", "target"], how="inner")
    mutual_count = mutual["source"].value_counts()
    out_unique = directed["source"].value_counts()
    out["reciprocity"] = (
        out["user_id"].map(mutual_count).fillna(0.0)
        / out["user_id"].map(out_unique).fillna(0.0).replace(0.0, np.nan)
    ).fillna(0.0)

    # Undirected clustering coefficient.
    out["clustering_coefficient"] = _clustering(
        directed, list(out["user_id"]), exact_max_nodes=exact_clustering_max_nodes
    )
    return out


def _clustering(
    directed: pd.DataFrame, order: Sequence[str], *, exact_max_nodes: int
) -> List[float]:
    """
    Local clustering coefficient on the undirected projection.

    Uses NetworkX when the graph is small enough to do it exactly, otherwise an
    adjacency-set triangle count with degree-capped neighbourhoods. The cap
    matters: one celebrity node with 500k neighbours would otherwise dominate
    the entire runtime for a feature that saturates long before that.
    """
    adjacency: Dict[str, Set[str]] = defaultdict(set)
    for u, v in zip(directed["source"], directed["target"]):
        if u == v:
            continue
        adjacency[u].add(v)
        adjacency[v].add(u)

    if len(adjacency) <= exact_max_nodes:
        try:
            import networkx as nx

            graph = nx.Graph()
            graph.add_nodes_from(order)
            graph.add_edges_from(
                (u, v) for u, v in zip(directed["source"], directed["target"]) if u != v
            )
            values = nx.clustering(graph)
            return [float(values.get(node, 0.0)) for node in order]
        except ImportError:
            log.debug("networkx absent — using the built-in triangle count")

    NEIGHBOUR_CAP = 2_000
    results: List[float] = []
    for node in order:
        neighbours = adjacency.get(node)
        if not neighbours or len(neighbours) < 2:
            results.append(0.0)
            continue
        if len(neighbours) > NEIGHBOUR_CAP:
            neighbours = set(sorted(neighbours)[:NEIGHBOUR_CAP])
        degree = len(neighbours)
        links = 0
        for other in neighbours:
            links += len(neighbours & adjacency.get(other, set()))
        results.append(links / (degree * (degree - 1)))
    return results


def compute_profile_features(nodes: pd.DataFrame) -> pd.DataFrame:
    """
    Cheap profile ratios.

    ``followers_to_following`` is log-compressed: the raw ratio is heavy-tailed
    enough to dominate an unnormalised feature vector, and the informative part
    of it is the order of magnitude, not the value.
    """
    out = pd.DataFrame({"user_id": nodes["user_id"].astype(str)})
    followers = pd.to_numeric(nodes.get("followers_count"), errors="coerce").fillna(0.0)
    following = pd.to_numeric(nodes.get("following_count"), errors="coerce").fillna(0.0)
    out["followers_to_following"] = np.log1p(followers) - np.log1p(following)
    out["account_age_days"] = (
        pd.to_numeric(nodes.get("account_age_days"), errors="coerce").fillna(0.0).clip(lower=0)
    )
    return out


# --------------------------------------------------------------------------- #
# Content features
# --------------------------------------------------------------------------- #
def compute_content_features(
    posts: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    threshold: float = 0.75,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Text-reuse features — the bridge between the two branches.

    ``content_duplication_ratio``
        Fraction of an account's posts that near-duplicate another of its own
        posts. Catches template-driven automation.
    ``cross_account_dup_ratio``
        Fraction that near-duplicate a *different* account's post. Direct
        evidence of a shared script, and unlike synchrony it survives an
        adversary who deliberately jitters their posting times.
    ``hashtag_jaccard_mean``
        Mean hashtag-set Jaccard overlap with graph neighbours. Coordinated
        accounts converge on a tag vocabulary.

    Duplicate detection is MinHash + LSH banding from ``text_utils`` — near
    linear, so this stays viable on millions of posts.

    On the ``threshold``, and a real limitation
    ------------------------------------------
    Measured on the synthetic campaign (agents share talking points but wrap
    them in per-persona voice), bot-vs-human Cohen's *d* for
    ``cross_account_dup_ratio`` runs::

        threshold   0.85    0.75    0.65    0.55    0.45    0.35
        cohens_d   -0.45   +0.59   +1.77   +2.36   +1.07   +0.54

    At 0.85 the feature is *inverted*: strict duplicate detection does not fire
    on paraphrased narrative reuse, so the residual organic collisions dominate.
    Separation peaks near 0.55 — but a "duplicate" detector at 0.55 is no longer
    detecting duplicates, and on real corpora it would match any two posts on
    the same topic. The default of 0.75 is a deliberate compromise: strict
    enough that a positive still means "substantially the same text", loose
    enough to catch light rewording.

    The honest consequence: **a swarm that paraphrases per persona will score
    near zero here.** That is a true negative for this feature, not a bug — the
    paraphrase case is covered by ``hashtag_jaccard_mean`` (Cohen's *d* ≈ 16 on
    the same data) and by the text branch, which is precisely why the project
    fuses the two rather than relying on either alone. Copy-paste swarms
    (Cresci-style retweet rings) are what this feature is for, and it is very
    strong on those.
    """
    columns = [
        "user_id", "content_duplication_ratio",
        "cross_account_dup_ratio", "hashtag_jaccard_mean",
    ]
    if posts.empty:
        return pd.DataFrame(columns=columns)

    from .text_utils import minhash_signature, normalise_text

    frame = posts.loc[:, ["user_id", "text"]].copy()
    frame["user_id"] = frame["user_id"].astype(str)
    frame["norm"] = frame["text"].fillna("").map(lambda t: normalise_text(t, lower=True))
    frame = frame[frame["norm"].str.len() > 0]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    totals = frame.groupby("user_id").size().to_dict()

    # distinct text -> {user_id: how many times that user posted it}
    owners: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for user_id, norm in zip(frame["user_id"], frame["norm"]):
        owners[norm][user_id] += 1
    distinct_texts = list(owners)

    # ---- group near-duplicate texts into clusters ---------------------- #
    # Union-find over distinct strings. Exact duplicates are already merged by
    # the dict above; this adds the near-duplicate edges found by LSH.
    parent = list(range(len(distinct_texts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    if 1 < len(distinct_texts) <= 200_000:
        bands, rows_per_band = 16, 4
        buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = defaultdict(list)
        signatures = [np.asarray(minhash_signature(text)) for text in distinct_texts]
        for index, signature in enumerate(signatures):
            for band in range(bands):
                key = tuple(signature[band * rows_per_band : (band + 1) * rows_per_band])
                buckets[(band, key)].append(index)

        checked: Set[Tuple[int, int]] = set()
        for candidates in buckets.values():
            if len(candidates) < 2 or len(candidates) > 200:
                continue
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    pair = (candidates[i], candidates[j])
                    if pair in checked:
                        continue
                    checked.add(pair)
                    if float((signatures[pair[0]] == signatures[pair[1]]).mean()) >= threshold:
                        union(pair[0], pair[1])
    elif len(distinct_texts) > 200_000:
        log.warning(
            "content features: %d distinct posts — skipping the near-duplicate "
            "pass and using exact duplicates only. Coordinated accounts that "
            "paraphrase will be under-scored.", len(distinct_texts)
        )

    # cluster root -> {user_id: post count contributed}
    clusters: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for index, text in enumerate(distinct_texts):
        root = find(index)
        for user_id, count in owners[text].items():
            clusters[root][user_id] += count

    # ---- bounded ratios ------------------------------------------------- #
    # Counted per POST, not per matching pair. An earlier version incremented a
    # counter for every LSH pair, which is unbounded and saturated the ratio at
    # 1.0 for every account with any reuse at all — making the feature useless.
    own_dup: Dict[str, int] = defaultdict(int)
    cross_dup: Dict[str, int] = defaultdict(int)
    for members in clusters.values():
        for user_id, count in members.items():
            if len(members) > 1:
                # Somebody else posted near-identical text: all of this
                # account's posts in the cluster are cross-account duplicates.
                cross_dup[user_id] += count
            elif count > 1:
                # Only this account in the cluster: the repeats are self-reuse.
                own_dup[user_id] += count - 1

    accounts = sorted(totals)
    out = pd.DataFrame(
        {
            "user_id": accounts,
            "content_duplication_ratio": [
                min(1.0, own_dup.get(a, 0) / totals[a]) for a in accounts
            ],
            "cross_account_dup_ratio": [
                min(1.0, cross_dup.get(a, 0) / totals[a]) for a in accounts
            ],
        }
    )

    # ---- hashtag overlap with graph neighbours -------------------------- #
    tag_sets: Dict[str, Set[str]] = defaultdict(set)
    if "hashtags" in posts.columns:
        for user_id, tags in zip(posts["user_id"].astype(str), posts["hashtags"]):
            if isinstance(tags, (list, tuple, set)):
                tag_sets[user_id].update(str(t).lower().lstrip("#") for t in tags)

    neighbours: Dict[str, Set[str]] = defaultdict(set)
    if not edges.empty:
        for u, v in zip(edges["source"].astype(str), edges["target"].astype(str)):
            neighbours[u].add(v)
            neighbours[v].add(u)

    NEIGHBOUR_SAMPLE = 200
    overlaps: List[float] = []
    for account in accounts:
        mine = tag_sets.get(account)
        peers = neighbours.get(account)
        if not mine or not peers:
            overlaps.append(0.0)
            continue
        peer_list = sorted(peers)[:NEIGHBOUR_SAMPLE]
        scores = []
        for peer in peer_list:
            theirs = tag_sets.get(peer)
            if not theirs:
                continue
            union = mine | theirs
            scores.append(len(mine & theirs) / len(union) if union else 0.0)
        overlaps.append(float(np.mean(scores)) if scores else 0.0)
    out["hashtag_jaccard_mean"] = overlaps
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
@dataclass
class FeatureBundle:
    """The feature matrix plus everything needed to train and audit a GNN."""

    features: pd.DataFrame          # user_id + FEATURE_COLUMNS + label + split
    feature_columns: Tuple[str, ...] = FEATURE_COLUMNS
    edges: pd.DataFrame = field(default_factory=pd.DataFrame)
    note: str = ""

    @property
    def X(self) -> np.ndarray:
        return self.features.loc[:, list(self.feature_columns)].to_numpy(dtype=np.float32)

    @property
    def y(self) -> np.ndarray:
        return self.features["label"].to_numpy(dtype=np.int64)

    def mask(self, split: str) -> np.ndarray:
        return (self.features["split"] == split).to_numpy()

    def separation_report(self) -> pd.DataFrame:
        """
        Per-feature bot-vs-human separation — the sanity check that decides
        whether the features are real or decorative.

        Reports each class mean plus Cohen's *d*. If ``synchrony_score`` and
        ``reciprocity`` do not show a meaningful effect here, no amount of GNN
        capacity will rescue the graph branch, and that is worth discovering
        before a 200-epoch training run rather than after.
        """
        rows = []
        bots = self.features["label"] == 1
        for column in self.feature_columns:
            values = self.features[column].astype(float)
            a, b = values[bots], values[~bots]
            if len(a) < 2 or len(b) < 2:
                continue
            pooled = math.sqrt(((a.std() ** 2) + (b.std() ** 2)) / 2.0)
            cohens_d = (a.mean() - b.mean()) / pooled if pooled > 1e-12 else 0.0
            rows.append(
                {
                    "feature": column,
                    "bot_mean": round(float(a.mean()), 4),
                    "human_mean": round(float(b.mean()), 4),
                    "cohens_d": round(float(cohens_d), 3),
                    "abs_d": abs(round(float(cohens_d), 3)),
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values("abs_d", ascending=False)
            .drop(columns="abs_d")
            .reset_index(drop=True)
        )


def build_features(
    bundle,
    settings: Settings,
    *,
    window_seconds: Optional[int] = None,
    min_events: Optional[int] = None,
) -> FeatureBundle:
    """
    Compute the full feature matrix for a :class:`~aegis.dataset_loaders.GraphBundle`.

    Missing inputs degrade gracefully and loudly: a corpus with no post
    timeline gets zeroed timing features and a warning, rather than an
    exception — because several of the legacy graph corpora ship without
    tweet text, and the structural half of the analysis is still valid there.
    """
    graph_config = settings.section("graph_model", default={}) or {}
    window = int(window_seconds or graph_config.get("synchrony_window_seconds", 60))
    min_ev = int(min_events or graph_config.get("synchrony_min_events", 3))

    nodes, edges, posts = bundle.nodes, bundle.edges, bundle.posts
    log.info(
        "features for %s: %d nodes, %d edges, %d posts (window=%ds)",
        bundle.name, len(nodes), len(edges), len(posts), window,
    )

    out = pd.DataFrame({"user_id": nodes["user_id"].astype(str)})
    out["label"] = nodes["label"].to_numpy()
    out["split"] = nodes["split"].to_numpy()

    pieces = [
        compute_structural_features(nodes, edges),
        compute_profile_features(nodes),
    ]
    if posts.empty:
        log.warning(
            "[%s] no post timeline — every timing and content feature will be "
            "zero. The structural features still carry signal, but synchrony "
            "is the strongest single predictor, so expect a lower ceiling.",
            bundle.name,
        )
    else:
        pieces += [
            compute_temporal_features(posts),
            compute_synchrony(
                posts, window_seconds=window, min_events=min_ev, seed=settings.seed
            ),
            compute_content_features(
                posts,
                edges,
                threshold=float(graph_config.get("content_dup_threshold", 0.75)),
                seed=settings.seed,
            ),
        ]

    for piece in pieces:
        if piece is None or piece.empty:
            continue
        out = out.merge(piece, on="user_id", how="left")

    for column in FEATURE_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).astype(float)

    ordered = ["user_id", "label", "split", *FEATURE_COLUMNS]
    extras = [c for c in out.columns if c not in ordered]
    out = out.loc[:, ordered + extras]

    return FeatureBundle(
        features=out,
        edges=edges,
        note=f"{bundle.name} ({bundle.provenance}) window={window}s min_events={min_ev}",
    )


def build_transfer_features(bundle, settings: Settings) -> FeatureBundle:
    """
    Build platform-agnostic structural/behavioural features for unseen graphs.

    Count-like features are log-scaled and normalised by graph size. The
    returned column order is an inference contract shared by notebook 03 and
    the backend.
    """
    base = build_features(bundle, settings)
    frame = base.features.copy()
    n_nodes = max(2, len(frame))
    denominator = math.log1p(n_nodes - 1)
    for column in ("in_degree", "out_degree", "synchrony_partner_count"):
        frame[column] = np.log1p(
            np.clip(pd.to_numeric(frame[column], errors="coerce").fillna(0.0), 0.0, None)
        ) / denominator
    frame["degree_ratio"] = np.tanh(
        np.log1p(
            np.clip(
                pd.to_numeric(frame["degree_ratio"], errors="coerce").fillna(0.0),
                0.0,
                None,
            )
        )
    )
    for column in ("campaign_id", "source_dataset", "scenario"):
        if column in bundle.nodes.columns:
            lookup = bundle.nodes.set_index("user_id")[column]
            frame[column] = frame["user_id"].map(lookup)
    return FeatureBundle(
        features=frame,
        feature_columns=TRANSFER_FEATURE_COLUMNS,
        edges=base.edges,
        note=f"{base.note}; transform=transfer_v1",
    )


def dropout_undirected_edges(edge_index, p: float, training: bool = True):
    """Drop a directed edge and its reverse together during GNN training."""
    if not training or p <= 0.0 or edge_index.numel() == 0:
        return edge_index
    if not 0.0 <= p < 1.0:
        raise ValueError("edge-dropout probability must be in [0, 1)")

    import torch

    source, target = edge_index[0], edge_index[1]
    low, high = torch.minimum(source, target), torch.maximum(source, target)
    pairs = torch.unique(torch.stack([low, high], dim=0), dim=1)
    keep = torch.rand(pairs.size(1), device=edge_index.device) >= p
    pairs = pairs[:, keep]
    if pairs.numel() == 0:
        return torch.empty((2, 0), dtype=edge_index.dtype, device=edge_index.device)
    non_self = pairs[0] != pairs[1]
    forward = pairs
    reverse = torch.stack([pairs[1, non_self], pairs[0, non_self]], dim=0)
    return torch.cat([forward, reverse], dim=1)


def scale_features(
    bundle: FeatureBundle, *, train_split: str = "train"
) -> Tuple[np.ndarray, Any]:
    """
    Standardise features using **train-split statistics only**.

    Fitting the scaler on the full matrix leaks test distribution into
    training. It is a small leak and it reliably inflates the reported number,
    which is exactly the kind of mistake that survives peer review.
    """
    X = bundle.X
    train_mask = bundle.mask(train_split)
    if train_mask.sum() < 2:
        log.warning("scaler: train split has <2 rows — falling back to full-matrix stats")
        train_mask = np.ones(len(X), dtype=bool)

    try:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(X[train_mask])
        return scaler.transform(X).astype(np.float32), scaler
    except ImportError:
        mean = X[train_mask].mean(axis=0)
        std = X[train_mask].std(axis=0)
        std[std < 1e-8] = 1.0
        log.debug("sklearn absent — using a numpy standardiser")
        return ((X - mean) / std).astype(np.float32), {"mean": mean, "std": std}


def to_pyg_data(
    bundle: FeatureBundle,
    *,
    scaled: Optional[np.ndarray] = None,
    undirected: bool = True,
):
    """
    Convert to a PyTorch Geometric ``Data`` object with train/val/test masks.

    The edge index is symmetrised by default. Message passing over a directed
    follow graph starves exactly the accounts we care about — a fresh agent
    with 3 followers receives almost no messages and the GNN never learns
    anything about it. Symmetrising costs a little semantic precision and buys
    a great deal of signal propagation; the direction information is preserved
    in ``in_degree``/``out_degree``/``reciprocity`` regardless.
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "to_pyg_data needs torch + torch-geometric. Install with:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  pip install torch-geometric"
        ) from exc

    frame = bundle.features
    index = {user_id: i for i, user_id in enumerate(frame["user_id"])}

    X = bundle.X if scaled is None else scaled
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(bundle.y, dtype=torch.long)

    if bundle.edges.empty:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        src, dst = [], []
        for u, v in zip(bundle.edges["source"].astype(str), bundle.edges["target"].astype(str)):
            i, j = index.get(u), index.get(v)
            if i is None or j is None:
                continue
            src.append(i)
            dst.append(j)
            if undirected:
                src.append(j)
                dst.append(i)
        edge_index = torch.tensor([src, dst], dtype=torch.long)

    data = Data(x=x, y=y, edge_index=edge_index)
    for split in ("train", "val", "test"):
        data[f"{split}_mask"] = torch.tensor(bundle.mask(split), dtype=torch.bool)
    data.user_id = frame["user_id"].tolist()
    return data


__all__ = [
    "FEATURE_COLUMNS", "TRANSFER_FEATURE_COLUMNS", "FeatureBundle",
    "compute_temporal_features", "compute_synchrony",
    "compute_structural_features", "compute_profile_features",
    "compute_content_features",
    "build_features", "build_transfer_features", "dropout_undirected_edges",
    "scale_features", "to_pyg_data",
]
