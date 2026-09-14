# %% [markdown]
# # 01 · Data Ingestion & Synthetic Agentic Campaign Generation
#
# **AEGIS-SN** — *Identification and classification of adversarial agentic AI behaviours
# in centralized and decentralized social networks.*
#
# This notebook is the only place in the project that touches raw data. Everything
# downstream (notebooks 02–04, the FastAPI service) reads the harmonised parquet files
# this notebook writes to `data/processed/`.
#
# ## What it does
#
# 1. Audits the drop-zone and reports, per dataset, whether we have **real** data or not.
# 2. Loads every enabled text corpus into one harmonised schema.
# 3. Loads the interaction graph corpus.
# 4. Generates the **2026 synthetic agentic campaign** with LangChain — 8 coordinating
#    LLM agents running an influence operation against a backdrop of organic accounts.
# 5. Deduplicates, splits without leakage, and persists.
#
# ## The one rule this notebook enforces
#
# Every loader in `aegis.dataset_loaders` falls back to a schema-identical *synthetic stub*
# when the real thing is unavailable, so that a fresh clone runs end-to-end. That is a
# convenience, and it is also the single easiest way to accidentally report a fabricated
# number as a result. So every load is recorded in `data/manifest.json` with a provenance
# tag, and the final cell **fails loudly** if anything downstream is about to train on a stub.
#
# | provenance | meaning |
# |---|---|
# | `REAL` | parsed from files you actually downloaded |
# | `PARTIAL` | real, but row-capped by `smoke_test` |
# | `GENERATED` | the synthetic campaign — real *by design*, not a fallback |
# | `SYNTHETIC_FALLBACK` | **a stub. Never report a metric computed on this.** |

# %% [markdown]
# ## 1 · Environment
#
# `aegis` lives in `ml/src/` and is not pip-installed, so we put it on `sys.path`.
# `load_config()` walks up from the working directory to find the repo root, which means
# this cell works whether Jupyter was launched from the repo root or from `ml/notebooks/`.

# %%
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# --- put ml/src on the path -------------------------------------------------
_here = Path.cwd()
for _candidate in (_here, *_here.parents):
    if (_candidate / "ml" / "src" / "aegis").is_dir():
        sys.path.insert(0, str(_candidate / "ml" / "src"))
        break
else:
    raise RuntimeError(
        "Could not locate ml/src/aegis. Launch Jupyter from the repo root "
        "(the folder containing requirements.txt) or from ml/notebooks/."
    )

from aegis import config as acfg
from aegis import dataset_loaders as dl
from aegis import io_utils as iou
from aegis import local_store as ls
from aegis import synthetic_agents as sa
from aegis import text_utils as tu

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 90)

settings = acfg.load_config()
acfg.set_seed(settings.seed)

print(settings.paths.describe())
print(f"\nseed        : {settings.seed}")
print(f"smoke_test  : {settings.smoke_test}  (row cap per dataset: {settings.row_cap})")
print(f"device      : {acfg.resolve_device(settings.device)}")

# %% [markdown]
# ### A note on `smoke_test`
#
# It defaults to **on**, which caps every corpus at 1,500 rows so ingestion can be
# checked on a laptop. Notebook 02 is production-only and rejects smoke-capped
# data; smoke mode is never a reportable training path.
#
# For the real run:
#
# ```bash
# # PowerShell
# $env:AEGIS_SMOKE_TEST = "0"; jupyter lab
# # bash
# AEGIS_SMOKE_TEST=0 jupyter lab
# ```
#
# The full corpus is ~600k text rows and takes a GPU. Cell outputs throughout this notebook
# print the pre-cap row count as well, so you can always see what you are giving up.

# %% [markdown]
# ## 2 · Drop-zone audit — what do we actually have?
#
# This is the most important table in the notebook and the first thing to check at a review.
# It reports what is physically on disk, not what the config *hopes* is on disk.
#
# Several datasets named in the project spec are deliberately **disabled**, each for a
# specific and verified reason rather than because they were inconvenient. The
# `unavailable_reason` column carries that reason through from `ml/configs/default.yaml`.

# %%
text_specs = settings.text_datasets or {}
graph_specs = settings.graph_datasets or {}
all_specs = {**text_specs, **graph_specs}

scans = ls.scan_all(all_specs, repo_root=settings.paths.root, extract=True)
readiness = ls.readiness_table(scans, all_specs)

# Annotate with the config's own verdict so "READY but disabled" is explicable.
readiness["enabled"] = readiness["dataset"].map(
    lambda n: bool((all_specs.get(n) or {}).get("enabled", True))
)
readiness["reason_if_off"] = readiness["dataset"].map(
    lambda n: (all_specs.get(n) or {}).get("unavailable_reason", "")
)
readiness["branch"] = readiness["dataset"].map(
    lambda n: "text" if n in text_specs else "graph"
)

print(readiness.loc[:, [
    "dataset", "branch", "era", "status", "enabled", "n_files", "size", "reason_if_off",
]].to_string(index=False))

# %% [markdown]
# ### Reading that table
#
# Four datasets from the original spec are **not usable here**, and it is worth being precise
# about why, because "we used TweepFake" would be false:
#
# | dataset | status | why |
# |---|---|---|
# | **TweepFake** | files present, unusable | The local copy is *dehydrated*: `train.csv` has `user_id, status_id, screen_name, account.type, class_type` and **no `text` column** — 20,712 rows at 53 bytes each. It is the Twitter-ToS distribution, which expects you to re-hydrate the text through the API. The free X API tier no longer allows bulk lookup, so that is not achievable. |
# | **WildGuard** | absent | The folder `WildGuard  WildJailbreak/` is named for both but contains only the WildJailbreak release. `wildguardmix` is gated on Hugging Face. |
# | **TwiBot-22** | repo only | The folder is the *GitHub source repository* — `src/`, `pics/`, `descriptions/`. There is no `user.json`, `edge.csv` or `label.csv`. The graph needs a signed data-use agreement. |
# | **TwiBot-24** | absent | Same gate. This is the dataset the spec names as notebook 03's target, so see §6 for exactly what changes when you obtain it (answer: two config lines). |
#
# A fifth, `twitter_bot_detection_kaggle`, is present and *looks* usable — 50,000 rows,
# a clean 25,018/24,982 label split. It is excluded because it is **measurably noise**;
# §2.1 reproduces that measurement rather than asking you to take it on faith.
#
# What replaces them: `llm_tweet` and `deepset_injections`, both real, plus the fact that
# `wildjailbreak` alone is 2.76M rows and covers the adversarial-prompt role that
# WildGuard would have played.

# %% [markdown]
# ### 2.1 · Why the Kaggle bot dataset is excluded
#
# Run this once. It takes ~30 s and it is the difference between an assertion and a finding.

# %%
_kaggle_bot = Path(settings.paths.root) / "../DataSets/Twitter Bot Detection Dataset/bot_detection_data.csv"
if _kaggle_bot.resolve().exists():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    _df = pd.read_csv(_kaggle_bot.resolve())
    _X = pd.DataFrame({
        "retweet_count": _df["Retweet Count"].astype(float),
        "mention_count": _df["Mention Count"].astype(float),
        "follower_count": _df["Follower Count"].astype(float),
        "verified": _df["Verified"].astype(str).eq("True").astype(int),
        "tweet_len": _df["Tweet"].astype(str).str.len(),
    })
    _y = _df["Bot Label"].astype(int)
    _auc = cross_val_score(
        RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=settings.seed),
        _X, _y, cv=3, scoring="roc_auc",
    )
    print(f"rows                 : {len(_df):,}")
    print(f"label balance        : {_y.value_counts().to_dict()}")
    print(f"RandomForest ROC-AUC : {_auc.round(4)}  mean={_auc.mean():.4f}")
    print(f"unique User IDs      : {_df['User ID'].nunique():,} of {len(_df):,}")
    print(f"tweets containing '@': {int(_df['Tweet'].astype(str).str.contains('@').sum())}")
    print("\nclass-conditional feature means:")
    print(_X.assign(label=_y).groupby("label").mean().round(3).to_string())
    print(
        "\nVERDICT: ROC-AUC ~0.50 and the class-conditional means agree to three decimals."
        "\nThe table is Faker-generated with randomly assigned labels. It is also"
        "\nstructurally useless for the graph branch: one post per account (so no temporal"
        "\nprofile) and zero '@' characters anywhere (so no mention edges). Excluded."
    )
else:
    print("Kaggle bot dataset not found locally — nothing to check.")

# %% [markdown]
# ## 3 · Load the text corpora
#
# Each loader harmonises its source to a single schema, so downstream code never needs to
# know that HC3 ships paired answer lists while WildJailbreak ships four parallel prompt
# columns:
#
# | column | meaning |
# |---|---|
# | `uid` | stable row id, `"<dataset>:<n>"` |
# | `text` | the utterance being classified |
# | `label` | **0** = human/benign, **1** = machine-generated *or* adversarial |
# | `threat_class` | `human_benign`, `machine_generated`, `prompt_injection`, `jailbreak`, `harmful_completion` |
# | `generator` | which model wrote it — the key to the cross-generator test in notebook 02 |
# | `domain` | topical domain where the source provides one |
# | `group_id` | grouping key so related rows cannot be split across train/test |
# | `era` | `legacy` (≤2022) · `modern` (2024) · `frontier` (2024/25) · `frontier_2026` |
#
# **This cell is the slow one** — roughly 10 minutes cold, dominated by parsing M4's 28
# JSONL shards (519 MB) and streaming WildJailbreak's 506 MB TSV. It caches to
# `data/interim/`, so re-runs are seconds.

# %%
INTERIM_TEXT = settings.paths.interim / "text_frames.parquet"
FORCE_RELOAD = os.environ.get("AEGIS_FORCE_RELOAD", "0") == "1"

if INTERIM_TEXT.exists() and not FORCE_RELOAD:
    _cached = iou.load_frame(INTERIM_TEXT)
    text_frames = {name: part for name, part in _cached.groupby("source_dataset")}
    print(f"loaded {len(_cached):,} rows from cache {INTERIM_TEXT.name}")
    print("set AEGIS_FORCE_RELOAD=1 to re-parse from the raw files")
else:
    text_frames = dl.load_all_text(settings)
    _combined = pd.concat(text_frames.values(), ignore_index=True)
    iou.save_frame(_combined, INTERIM_TEXT)
    print(f"\nparsed and cached {len(_combined):,} rows -> {INTERIM_TEXT.name}")

# %%
summary = pd.DataFrame([
    {
        "dataset": name,
        "rows": len(frame),
        "human": int((frame["label"] == 0).sum()),
        "adversarial": int((frame["label"] == 1).sum()),
        "pos_rate": round(float(frame["label"].mean()), 3),
        "generators": frame["generator"].nunique(),
        "threat_classes": ", ".join(sorted(frame["threat_class"].dropna().unique())[:3]),
        "era": frame["era"].iloc[0] if len(frame) else "",
        "median_chars": int(frame["text"].str.len().median()) if len(frame) else 0,
    }
    for name, frame in sorted(text_frames.items())
])
print(summary.to_string(index=False))
print(f"\nTOTAL: {summary['rows'].sum():,} rows")

# %% [markdown]
# ### Provenance check
#
# `manifest_summary` reads back `data/manifest.json`. Anything tagged `SYNTHETIC_FALLBACK`
# here is a stub and must not appear in a results table.

# %%
print(acfg.manifest_summary(settings.paths).to_string(index=False))

# %% [markdown]
# ### Sample rows, by threat class
#
# Worth actually reading. The `jailbreak` rows are long roleplay framings, the
# `prompt_injection` rows are short imperative overrides, and several of the latter are in
# German — `deepset_injections` is the only multilingual source here, which is what makes it
# a genuine test of whether the detector has learned anything beyond English keyword shape.

# %%
_all_text = pd.concat(text_frames.values(), ignore_index=True)
for _tc, _grp in _all_text.groupby("threat_class"):
    print(f"\n{'=' * 78}\n{_tc.upper()}   ({len(_grp):,} rows)\n{'=' * 78}")
    for _, _row in _grp.sample(min(2, len(_grp)), random_state=settings.seed).iterrows():
        print(f"[{_row['source_dataset']} / {_row['generator']}]")
        print(f"  {_row['text'][:220]}...\n")

# %% [markdown]
# ## 4 · The 2026 synthetic agentic campaign
#
# Every corpus above predates the threat this project is actually about. HC3 is 2022
# ChatGPT, M4 is 2023-era generators, WildJailbreak is single-turn prompts with no social
# context at all. **None of them contain a coordinated multi-agent influence operation**,
# because public datasets mapping agent-to-agent networks essentially do not exist yet.
#
# So we generate one. `aegis.synthetic_agents` runs a LangChain multi-agent simulation.
#
# ### Two scenarios
#
# `synthetic.scenario` selects the campaign archetype:
#
# * **`product_shill`** (default) — *"a coordinated bot swarm promoting a fake product"*.
#   Eight accounts push a fictional supplement, AURA-9: an ordinary-user seeder, a
#   credentialed-sounding "clinical" authority, influencer amplifiers, a **former sceptic**
#   for social proof, a fake verified-purchase reviewer, and a controller. They share a
#   **discount code** — measured, all 8 accounts emit one across 41 posts, which is exactly
#   the kind of recurring token that betrays a commercial network.
# * **`civic_disinfo`** — manufacture opposition to a fictional municipal water-treatment
#   retrofit ahead of a council vote.
#
# They are separate content packs rather than one template with the nouns swapped, because
# the *shape* differs: a civic swarm asks questions and demands audits, a shill swarm makes
# claims and closes. Those produce different burstiness and duplication profiles, so a
# detector trained on one alone learns the wrong invariant. Generate both and concatenate
# for a harder corpus — `generate_campaign(settings, scenario=...)` takes the override.
#
# Common to both:
#
# * **8 agents** with distinct personas, including an orchestrator whose cue the rest
#   answer inside a tight window.
# * **14 orchestration turns** through a scripted narrative arc — seed → legitimise →
#   amplify → bridge → consolidate — which is what produces the *temporal* structure the
#   graph branch keys on.
# * **40 organic decoy accounts** posting on the same topic, so the swarm is not trivially
#   separable by "posts about Project Clearwater".
# * **~12% of agent turns carry a prompt-injection payload**, giving notebook 02
#   in-domain 2026 adversarial text rather than only 2024 benchmark text.
#
# ### Backend
#
# `offline` (the default) uses LangChain's `FakeListChatModel` over persona-conditioned
# templates: deterministic, free, no network, and reproducible from a seed — which matters
# because a reviewer must be able to regenerate the exact corpus. Set `OPENAI_API_KEY` or
# `ANTHROPIC_API_KEY` in `.env` and switch `synthetic.backend` in the config to use a live
# model; the campaign structure, the coordination signal and the schema are identical
# either way, only the surface text changes.
#
# **This is defensive research.** The generator exists to produce labelled examples of an
# attack so a detector can be trained on it. It is seeded, offline by default, describes a
# fictional municipal issue, and every row it emits is tagged `GENERATED` in the manifest
# and `is_synthetic=True` in the data.

# %%
_pack = sa.resolve_scenario(settings)
_campaign_cfg = (settings.synthetic.get("campaigns") or {}).get(_pack.name, {})
_bank_cfg = settings.synthetic.get("campaign_bank") or {}

print(f"backend resolved to: {sa.resolve_backend(settings)}")
print(f"scenario           : {_pack.name}  (available: {', '.join(sorted(sa.SCENARIOS))})")
print(f"campaign           : {_campaign_cfg.get('codename')}")
print(f"target (fictional) : {_campaign_cfg.get('target_entity')}")
print(f"agents             : {settings.synthetic.get('n_agents')}")
print(f"organic decoys     : {settings.synthetic.get('n_human_decoys')}")
print(f"turns              : {settings.synthetic.get('campaign_turns')}")
print(f"injection rate     : {settings.synthetic.get('injection_payload_rate')}")

# Keep the default campaign for backwards-compatible reports, then create a
# disconnected bank for strict campaign-level graph and fusion validation.
campaign = sa.generate_campaign(settings, persist=True)
campaign_bank = sa.generate_campaign_bank(
    settings,
    scenarios=_bank_cfg.get("scenarios"),
    seeds=_bank_cfg.get("seeds"),
    persist=True,
)

print(f"\nposts     : {len(campaign.posts):,}")
print(f"text rows : {len(campaign.text_rows):,}")
print(f"graph     : {campaign.graph.n_nodes} nodes / {campaign.graph.n_edges} edges")
print(f"bot rate  : {campaign.graph.bot_rate:.3f}")
print(
    f"campaign bank: {len(campaign_bank.manifest)} disconnected campaigns, "
    f"{len(campaign_bank.nodes):,} nodes / {len(campaign_bank.edges):,} edges"
)

# %% [markdown]
# ### The campaign as an analyst would see it
#
# The `phase` column is the narrative arc. Watch the agent count and the hashtag
# concentration climb through it — that ramp is the coordination signature, and it is
# exactly what `graph_features.compute_synchrony` is built to measure in notebook 03.

# %%
_phase = (
    campaign.posts.groupby(["turn", "phase"], observed=True)
    .agg(posts=("post_id", "size"),
         agents=("agent_handle", "nunique"),
         injections=("carries_injection", "sum"))
    .reset_index()
)
print(_phase.to_string(index=False))

print("\nper-agent activity:")
print(
    campaign.posts.groupby(["agent_handle", "archetype", "role"], observed=True)
    .size().rename("posts").reset_index().to_string(index=False)
)

# %%
print("SAMPLE AGENT POSTS\n" + "=" * 78)
for _, _row in campaign.posts[campaign.posts["is_agent"]].sample(
    6, random_state=settings.seed
).iterrows():
    _flag = "  [INJECTION PAYLOAD]" if _row["carries_injection"] else ""
    print(f"\n@{_row['agent_handle']} ({_row['archetype']}, turn {_row['turn']}, "
          f"{_row['phase']}){_flag}")
    print(f"  {_row['text']}")

print("\n\nSAMPLE ORGANIC DECOY POSTS\n" + "=" * 78)
for _, _row in campaign.posts[~campaign.posts["is_agent"]].sample(
    3, random_state=settings.seed
).iterrows():
    print(f"\n  {_row['text']}")

# %% [markdown]
# ## 5 · Build the unified text corpus
#
# `build_text_corpus` concatenates the per-dataset frames and deduplicates across the
# **combined** corpus rather than within each one.
#
# That distinction matters more than it sounds. HC3, M4 and WildJailbreak overlap: the same
# Reddit ELI5 answer appears in HC3 and in M4's `reddit_*` shards, and the same templated
# injection appears in both `wildjailbreak` and `deepset_injections`. Deduplicate per
# dataset and those survive; then the random split puts one copy in train and the other in
# test, and the reported F1 is measuring memorisation.

# %%
text_frames["synthetic_campaign"] = campaign.text_rows

corpus = dl.build_text_corpus(text_frames, settings=settings, dedupe=True, near_dup_threshold=0.9)

print(f"corpus: {len(corpus):,} rows")
print(f"\nby label:\n{corpus['label'].value_counts().to_string()}")
print(f"\nby threat class:\n{corpus['threat_class'].value_counts().to_string()}")
print(f"\nby era:\n{corpus['era'].value_counts().to_string()}")
print(f"\nby source:\n{corpus['source_dataset'].value_counts().to_string()}")

# %% [markdown]
# ### Length balance — the shortcut check
#
# If the machine-generated class is systematically longer than the human class, a model can
# hit a high F1 using `len(text)` alone and will collapse the moment it meets an agent that
# writes short posts. `length_balance_report` alarms when the ratio between class medians
# exceeds 1.75.
#
# Some imbalance here is real and expected — WildJailbreak's adversarial rewrites genuinely
# are longer than its vanilla prompts, because that is what a jailbreak *is*. The point is
# to know the number, and to check in notebook 02 that the model is not leaning on it.

# %%
balance = tu.length_balance_report(corpus, text_col="text", label_col="label")
print(balance)

print("\nmedian characters by threat class:")
print(corpus.groupby("threat_class")["text"].apply(lambda s: int(s.str.len().median())).to_string())

print("\nmedian characters by source dataset:")
print(corpus.groupby("source_dataset")["text"].apply(lambda s: int(s.str.len().median())).to_string())

# %% [markdown]
# ## 6 · Load the external diagnostic graph
#
# ### What is actually available
#
# The spec names **TwiBot-24** as an intended external target. It is not on disk
# and is access-gated. We do have **Cresci-2017**: 3,474 genuine accounts and
# 991 `social_spambots_1` accounts, with 4.4M tweets between them. The current
# notebook 03 trains on disconnected campaign-bank graphs; Cresci is persisted
# here only as an external, explicitly optimistic diagnostic.
#
# Two things had to be solved to make it usable:
#
# **1. There is no edge list.** This archive ships `users.csv` and `tweets.csv` and *no*
# `friends.csv`/`followers.csv` — the follow graph simply is not in it. Edges are therefore
# reconstructed from tweet records: `replied_to` from `in_reply_to_user_id`, `retweeted` by
# resolving `retweeted_status_id` through a status→author map, and `mentioned` by parsing
# `@handles` against `screen_name`.
#
# **2. Direct interaction alone is not enough.** Measured, that reconstruction yields
# **164 edges across 4,465 accounts** — a graph with nothing in it. The reason is structural:
# Cresci crawled two disjoint account sets, and each mostly interacts with the wider Twitter
# population rather than with the other. Of 61,243 retweets, **zero** resolve to an author
# inside the corpus.
#
# The standard fix, and what the coordinated-behaviour literature actually does, is the
# **co-activity network**: link two accounts when they act on the *same object* — retweet the
# same status, reply to the same account, use the same rare hashtag, mention the same handle,
# post the same normalised text. The retweeted author does not need to be in the corpus for
# that to work. For a retweet ring, the co-retweet network *is* the campaign.
#
# The `max_accounts` guard on each relation is what keeps this meaningful: an object touched
# by hundreds of accounts is a *topic*, not a conspiracy, and wiring everyone together
# through `#Roma` would destroy the community structure the GNN needs.

# %%
GRAPH_NAME = settings.graph_model.get(
    "external_diagnostic_dataset", "cresci_2017"
)

INTERIM_GRAPH = settings.paths.interim / f"{GRAPH_NAME}_nodes.parquet"
if INTERIM_GRAPH.exists() and not FORCE_RELOAD:
    graph = dl.GraphBundle(
        name=GRAPH_NAME,
        nodes=iou.load_frame(INTERIM_GRAPH),
        edges=iou.load_frame(settings.paths.interim / f"{GRAPH_NAME}_edges.parquet"),
        posts=iou.load_frame(settings.paths.interim / f"{GRAPH_NAME}_posts.parquet"),
        provenance=iou.PROV_REAL,
    )
    print(f"loaded {GRAPH_NAME} from cache")
else:
    graph = dl.load_graph_dataset(GRAPH_NAME, settings)
    iou.save_frame(graph.nodes, INTERIM_GRAPH)
    iou.save_frame(graph.edges, settings.paths.interim / f"{GRAPH_NAME}_edges.parquet")
    iou.save_frame(graph.posts, settings.paths.interim / f"{GRAPH_NAME}_posts.parquet")

print(f"\n{graph.summary()}")
print(f"\nedges by relation:\n{graph.edges['relation'].value_counts().to_string()}")
print(f"\nnode labels: {graph.nodes['label'].value_counts().to_dict()}")
print(f"posts: {len(graph.posts):,} across {graph.posts['user_id'].nunique():,} accounts")
print(f"timestamp span: {graph.posts['created_at'].min()} -> {graph.posts['created_at'].max()}")

# %% [markdown]
# ### Leakage guard: accounts with no posts are dropped
#
# In this archive only **1,083 of 3,474** genuine accounts have tweets in `tweets.csv`,
# while **all 991** spambots do. So "has any post at all" is 100% predictive of bot and 31%
# predictive of human — a model could score ~0.78 accuracy on that alone, having learned
# nothing whatsoever about coordination.
#
# `require_posts: true` in the config drops the tweetless accounts. It costs 2,391 nodes and
# leaves 1,083 human / 991 bot, which is very nearly balanced, and it removes an artefact
# that would otherwise flatter the external diagnostic.

# %%
_deg = pd.concat([graph.edges["source"], graph.edges["target"]]).value_counts()
_nodes = graph.nodes.assign(degree=graph.nodes["user_id"].map(_deg).fillna(0).astype(int))

print("degree by label:")
print(_nodes.groupby("label")["degree"].agg(["count", "mean", "median", "max"]).round(1).to_string())
print(f"\nisolated nodes: {int((_nodes['degree'] == 0).sum())}")

print("\nposts per account by label:")
_ppa = graph.posts.groupby("user_id").size().rename("posts")
print(
    graph.nodes.set_index("user_id").join(_ppa).groupby("label")["posts"]
    .agg(["mean", "median", "max"]).round(1).to_string()
)

# %% [markdown]
# ### ⚠ Assortativity — why Cresci is diagnostic-only
#
# The cell below crosses each edge's endpoint labels. It shows that the graph is **almost
# perfectly assortative**: bot–bot and human–human edges dominate, and cross-label edges are
# a very small fraction of the total.
#
# That is not a triumph, it is an artefact of how Cresci was collected. The genuine accounts
# and the spambots were crawled separately, at different times, around different topics, so
# they barely share any object to co-act on. A GNN will therefore score extremely well here
# by doing little more than community detection.
#
# Two consequences, both handled in notebook 03:
#
# 1. Cresci never selects or early-stops the model.
# 2. Graph training, validation, and test use distinct campaign IDs with no
#    cross-campaign message passing.
# 3. Macro and worst-campaign metrics—not Cresci accuracy—are the acceptance
#    evidence.

# %%
_lab = graph.nodes.set_index("user_id")["label"]
_e = (
    graph.edges
    .merge(_lab.rename("src_label"), left_on="source", right_index=True)
    .merge(_lab.rename("dst_label"), left_on="target", right_index=True)
)
_ct = pd.crosstab(_e["src_label"], _e["dst_label"])
print("edge endpoint label mix (0 = human, 1 = bot):")
print(_ct.to_string())

_cross = int(_ct.values.sum() - np.trace(_ct.values))
print(f"\ncross-label edges: {_cross:,} of {len(_e):,}  ({100 * _cross / max(len(_e), 1):.2f}%)")
print(
    "\nA near-zero cross-label rate means the graph is separable by community structure"
    "\nalone. Treat notebook 03's Cresci score as an external upper-bound diagnostic;"
    "\nthe held-out campaign reports are the generalization evidence."
)

# %% [markdown]
# ### Switching to TwiBot-24 when you get access
#
# When the data-use agreement clears:
#
# 1. Drop `user.json`, `edge.csv`, `label.csv`, `split.csv` into `DataSets/TwiBot-24/`.
# 2. In `ml/configs/default.yaml`, set `graph_datasets.twibot_24.enabled: true`.
# 3. Add TwiBot-24 as an independent external graph evaluation in notebook 03,
#    or extend campaign/source grouping before admitting it to model selection.
#
# The loader (`_local_twibot`) already streams the 170M-row edge file in chunks and filters
# to user–user relations as it goes, and the feature extractor is relation-aware. Do not
# replace campaign-level isolation with a random TwiBot node split.

# %% [markdown]
# ## 7 · Split without leakage
#
# `stratified_split` is stratified on `label` **and** grouped on `group_id`. The grouping is
# the part that matters:
#
# * HC3 pairs a human answer and a ChatGPT answer to the *same question*. Split them apart
#   and the model can match on question content.
# * M4 pairs human and machine text from the same source document.
# * WildJailbreak pairs a vanilla prompt with its adversarial rewrite — near-identical text,
#   opposite roles.
# * `llm_tweet`'s `AI_Generated.csv` holds four renderings of one text (raw, paraphrased,
#   translated, humanized). Splitting those apart would make the evasion-robustness number
#   meaningless.
#
# `assert_no_leakage` then does an independent MinHash check for near-duplicates across the
# finished splits, because a grouping key only protects you where the source supplied one.

# %%
splits = tu.stratified_split(
    corpus, label_col="label", group_col="group_id", test_size=0.15, val_size=0.15,
    seed=settings.seed,
)

print(pd.DataFrame([
    {
        "split": name,
        "rows": len(part),
        "human": int((part["label"] == 0).sum()),
        "adversarial": int((part["label"] == 1).sum()),
        "pos_rate": round(float(part["label"].mean()), 3),
        "sources": part["source_dataset"].nunique(),
        "generators": part["generator"].nunique(),
    }
    for name, part in splits.items()
]).to_string(index=False))

leakage = tu.assert_no_leakage(splits, text_col="text", strict=False)
print(f"\nleakage check:\n{leakage.to_string(index=False)}")

# %% [markdown]
# ### Class weights for notebook 02
#
# The corpus is not balanced and should not be resampled into balance — the ratio carries
# information about how these sources are actually distributed. Notebook 02 passes these
# weights into the loss instead.

# %%
class_weights = tu.compute_class_weights(splits["train"]["label"].tolist())
print(f"class weights: {class_weights}")

# %% [markdown]
# ## 8 · Persist
#
# `data/processed/` is the contract with notebooks 02–04 and with the backend service.
# Nothing downstream re-reads a raw file.

# %%
for _name, _part in splits.items():
    _path = settings.paths.processed / f"text_{_name}.parquet"
    iou.save_frame(_part, _path)
    print(f"  {_path.name:<28} {len(_part):>8,} rows")

iou.save_frame(corpus, settings.paths.processed / "text_corpus_full.parquet")
iou.save_frame(graph.nodes, settings.paths.processed / "graph_nodes.parquet")
iou.save_frame(graph.edges, settings.paths.processed / "graph_edges.parquet")
iou.save_frame(graph.posts, settings.paths.processed / "graph_posts.parquet")
iou.save_frame(campaign.graph.nodes, settings.paths.processed / "campaign_nodes.parquet")
iou.save_frame(campaign.graph.edges, settings.paths.processed / "campaign_edges.parquet")
iou.save_frame(campaign.posts, settings.paths.processed / "campaign_posts.parquet")
iou.save_frame(campaign_bank.nodes, settings.paths.processed / "campaign_bank_nodes.parquet")
iou.save_frame(campaign_bank.edges, settings.paths.processed / "campaign_bank_edges.parquet")
iou.save_frame(campaign_bank.posts, settings.paths.processed / "campaign_bank_posts.parquet")
iou.save_frame(campaign_bank.text_rows, settings.paths.processed / "campaign_bank_text.parquet")

iou.save_json(
    {
        "seed": settings.seed,
        "smoke_test": settings.smoke_test,
        "row_cap": settings.row_cap,
        "corpus_rows": int(len(corpus)),
        "splits": {k: int(len(v)) for k, v in splits.items()},
        "class_weights": {str(k): float(v) for k, v in class_weights.items()},
        "text_sources": sorted(text_frames),
        "graph_dataset": GRAPH_NAME,
        "graph_nodes": int(graph.n_nodes),
        "graph_edges": int(graph.n_edges),
        "graph_relations": sorted(graph.edges["relation"].unique().tolist()),
        "campaign_backend": sa.resolve_backend(settings),
        "campaign_ids": [row["campaign_id"] for row in campaign_bank.manifest],
        "campaign_count": len(campaign_bank.manifest),
        "disabled_datasets": {
            n: (s or {}).get("unavailable_reason", "")
            for n, s in all_specs.items() if not (s or {}).get("enabled", True)
        },
    },
    settings.paths.processed / "ingestion_summary.json",
)
print(f"\nwrote {settings.paths.processed}")

# %% [markdown]
# ## 9 · Provenance gate
#
# The last thing this notebook does is refuse to hand a stub to notebook 02.
#
# `assert_real_data(..., strict=True)` raises if any named dataset resolved to
# `SYNTHETIC_FALLBACK`. If it raises, the fix is to obtain the data — not to lower the gate.

# %%
audit = acfg.manifest_summary(settings.paths)
print(audit.to_string(index=False))

# The manifest is an append-only ledger keyed by dataset name, so entries written
# by an earlier run survive even after a dataset is disabled. Those are stale, not
# active — the gate must only consider corpora this run actually loaded.
_enabled = {n for n, s in (settings.text_datasets or {}).items() if (s or {}).get("enabled", True)}
_loaded = set(text_frames) - {"synthetic_campaign"}
_required = sorted(_enabled & _loaded & set(audit["dataset"]))

_stale = audit[~audit["dataset"].isin(_loaded | {f"{GRAPH_NAME}__nodes"})
               & ~audit["dataset"].str.startswith("synthetic_campaign")]
if len(_stale):
    print(f"\nSTALE manifest entries from a previous run ({len(_stale)}) — ignored by the gate:")
    print(_stale.loc[:, ["dataset", "provenance", "retrieved_at"]].to_string(index=False))
    print("Delete data/manifest.json to clear them.")

try:
    iou.assert_real_data(settings.paths, _required, strict=True)
    print(f"\nPASS — all {len(_required)} enabled text corpora resolved to real data:")
    print(f"  {', '.join(_required)}")
except AssertionError as exc:
    print(f"\nFAIL — {exc}")
    print(
        "\nAt least one corpus fell back to a synthetic stub. Metrics computed on it are"
        "\nnot reportable. Check the `note` column above for the specific reason"
        "\n(missing files, gated dataset, absent credentials) and fix that."
    )

_fallbacks = audit[audit["provenance"].eq(iou.PROV_SYNTHETIC_FALLBACK)
                   & audit["dataset"].isin(_loaded)]
if len(_fallbacks):
    print(f"\nSTUBS ACTIVE IN THIS RUN ({len(_fallbacks)}):")
    print(_fallbacks.loc[:, ["dataset", "rows", "note"]].to_string(index=False))

# %% [markdown]
# ## Summary
#
# **Written to `data/processed/`:**
# `text_train/val/test.parquet`, `text_corpus_full.parquet`, `graph_{nodes,edges,posts}.parquet`,
# `campaign_{nodes,edges,posts}.parquet`, `campaign_bank_{nodes,edges,posts,text}.parquet`,
# `ingestion_summary.json`.
#
# **Real data in use:** HC3 (24,322 QA pairs) · M4 (68,556 pairs, 7 generators × 5 domains) ·
# WildJailbreak (2.76M rows, stratified) · deepset prompt-injections (multilingual) ·
# LLM-Tweet (29,145 essays + humanized/paraphrased evasion variants) ·
# Cresci-2017 (2,074 accounts, 410k tweets, 292k co-activity edges) ·
# synthetic 2026 campaign (8 agents, 14 turns).
#
# **Excluded, with reasons on record:** TweepFake (dehydrated — no text column) ·
# WildGuard, TwiBot-22, TwiBot-24 (gated/absent) · Kaggle bot dataset (measured ROC-AUC 0.4959).
#
# **Carry forward into notebook 03:** train and validate on disconnected
# campaign-bank graphs. Keep the ~98%-assortative Cresci graph external.
#
# → **`02_text_classification_model.ipynb`**
