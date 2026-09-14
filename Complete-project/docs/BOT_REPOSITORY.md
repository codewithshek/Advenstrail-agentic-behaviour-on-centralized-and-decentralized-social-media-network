# OSoMe Bot Repository — what is actually in these archives

Source: <https://botometer.osome.iu.edu/bot-repository/datasets.html>
Downloaded to `DataSets/BotRepository/<name>/`, one folder per set, each holding
the archive plus the site's `info.json` (name, description, filename,
publications, licence).

Every archive below was opened and read before anything was wired into the
pipeline. This file records what was found, not what the landing page claims.
Config blocks live under `graph_datasets:` in `ml/configs/default.yaml`.
The small profile-only sets use the `botrepo_` prefix and
`_local_bot_repository`. Three larger behavioral releases use dedicated
loaders: `caverlee_2011`, `cresci_2015`, and `cresci_2017_full`.

The local folders for `astroturf`, `varol-2017`, the TwiBot-20 sample and the
Bot Repository TwiBot-22 label-only sample were removed after this audit because
they cannot produce supervised features. Their disabled config entries and the
measurements below remain as provenance, preventing a future download from
being mistaken for usable training data.

---

## Findings for the small `botrepo_*` archives

**1. In these small archives, `<name>_tweets.json` contains no tweet text.**
Ten sets ship one. Every element of every one has exactly two keys —
`created_at` and `user` — so it is a hydrated Twitter user object plus the
timestamp of the probe that fetched it. Measured: **0 of 62,595 elements carry
a `text` or `full_text` key**, and no account appears more than once
(max 1 "tweet" per user in all ten). Twitter's terms of service are why —
OSoMe may redistribute profiles, not tweet bodies.

The consequence for this project is unavoidable: these corpora support the
**profile / tabular features only**. With no post timeline, `synchrony_score`,
`synchrony_partner_count`, `burstiness`, `memory_coefficient`,
`posting_entropy`, `circadian_flatness`, `content_duplication_ratio`,
`cross_account_dup_ratio` and `hashtag_jaccard_mean` are all structurally zero.
With no edge list, `in_degree`, `out_degree`, `degree_ratio`, `reciprocity` and
`clustering_coefficient` are zero too. Exactly two of the sixteen feature
columns are non-zero on this data: `account_age_days` and
`followers_to_following`. `aegis.graph_features.build_features` already handles
this and warns about it on both counts; a no-edge, no-post bundle is valid, it
just has a far lower ceiling than `cresci_2017`.

**2. Six of the eleven usable sets are single-class by design.**
`botwiki`, `celebrity`, `political-bots`, `pronbots`, `vendor-purchased` and
`verified-2019` contain only bots or only humans. That is intentional — OSoMe's
documented usage is to mix a bot set with a human set. They are enabled because
`load_bot_repository_corpus()` is what consumes them; loading one on its own
emits a `SINGLE-CLASS corpus` warning.

**3. Joining labels to user objects is a leakage guard, not housekeeping.**
Hydration coverage is not independent of the label. On `cresci-stock-2018` only
**38.4 % of bots** survived the re-crawl against **82.6 % of humans**
(11,406 of 18,508 bots have no user object, versus 1,305 of 7,479 humans).
Retaining those rows with zeroed profile fields would let any model take most of
its accuracy from "empty profile ⇒ bot", which is a property of the 2019 crawl
rather than of automation. The parser inner-joins and logs the drop, on the same
reasoning as `require_posts` on `cresci_2017`.

---

## Summary table

Counts are as loaded through `load_graph_dataset`, i.e. after the label/user
object inner join and after `_finalise_graph`. "Labelled ids" is the raw count
in `<name>.tsv`.

| dataset | labelled ids | accounts loaded | human / bot | user objects? | tweet text? | edges? | status |
|---|---:|---:|---|---|---|---|---|
| `botrepo_botometer_feedback_2019` | 528 | 518 | 380 / 138 | yes (98.1 %) | no | no | **ENABLED** |
| `botrepo_botwiki_2019` | 704 | 698 | 0 / 698 | yes (99.1 %) | no | no | **ENABLED** (single-class) |
| `botrepo_celebrity_2019` | 5,970 | 5,918 | 5,918 / 0 | yes (99.1 %) | no | no | **ENABLED** (single-class) |
| `botrepo_cresci_rtbust_2019` | 759 | 693 | 340 / 353 | yes (91.3 %) | no | no | **ENABLED** |
| `botrepo_cresci_stock_2018` | 25,987 | 13,276 | 6,174 / 7,102 | yes (51.1 %) | no | no | **ENABLED** |
| `botrepo_gilani_2017` | 2,633 | 2,484 | 1,394 / 1,090 | yes (94.3 %) | no | no | **ENABLED** |
| `botrepo_midterm_2018` | 50,538 | 50,538 | 8,092 / 42,446 | yes (100 %) | no | no | **ENABLED** |
| `botrepo_political_bots_2019` | 62 | 62 | 0 / 62 | yes (100 %) | no | no | **ENABLED** (single-class) |
| `botrepo_pronbots_2019` | 21,964 | 17,882 | 0 / 17,882 | yes (81.4 %) | no | no | **ENABLED** (single-class) |
| `botrepo_vendor_purchased_2019` | 1,088 | 1,087 | 0 / 1,087 | yes (99.9 %) | no | no | **ENABLED** (single-class) |
| `botrepo_verified_2019` | 2,000 | 1,987 | 1,987 / 0 | yes (99.4 %) | no | no | **ENABLED** (single-class) |
| `botrepo_astroturf` | 585 | — | 0 / 585 | **no** | no | no | REJECTED — ids + label only |
| `botrepo_varol_2017` | 2,573 | — | 1,747 / 826 | **no** | no | no | REJECTED — ids + label only |
| `botrepo_twibot_22` | 1,000,000 | — | 860,057 / 139,943 | **no** | no | no | REJECTED — label file only |
| `botrepo_twibot_20` | 100 | — | **no labels** | yes | **yes** | yes (30 usable) | REJECTED — unlabelled sample |
| `caverlee_2011` | 41,411 after 44 conflicts removed | 39,912 after `require_posts` | 19,227 / 20,685 | yes | **yes: 1,686,109 loaded** | **310,873 inferred** | **ENABLED — behavioral** |
| `cresci_2015` | 5,301 | 5,301 | 1,950 / 3,351 | yes | **yes: 279,943 loaded** | **382,724 observed + inferred** | **ENABLED — behavioral** |
| `cresci_2017_full` | 14,368 | 10,197 after `require_posts` | 1,083 / 9,114 | yes | **yes: 714,856 loaded** | **399,249 inferred** | **ENABLED — behavioral** |
| `botrepo_caverlee_2011` | — | — | — | — | — | — | DISABLED — superseded by `caverlee_2011` |

Union after de-duplication: **94,890 accounts, 70,821 bot / 24,069 human
(bot rate 0.746)**, 253 duplicate rows dropped.

---

## Per-dataset notes

### `botrepo_botometer_feedback_2019` — ENABLED
`botometer-feedback-2019.tar.gz` → `botometer-feedback-2019.tsv` (9.7 KB,
headerless, `id \t human|bot`) + `botometer-feedback-2019_tweets.json` (892 KB,
JSON array of `{created_at, user}`).
529 rows / 528 unique ids (one id appears twice, with **conflicting** labels —
the parser keeps the first). 385 human / 143 bot raw; 380 / 138 after the join.
Botometer users' own feedback, hand-labelled by K.C. Yang — the closest thing
here to an in-the-wild mixed sample.

> Yang, Kai-Cheng, Onur Varol, Clayton A. Davis, Emilio Ferrara, Alessandro
> Flammini, and Filippo Menczer. "Arming the public with artificial intelligence
> to counter social bots." *Human Behavior and Emerging Technologies* 1, no. 1
> (2019): 48-61. CC BY-NC-ND 4.0.

### `botrepo_botwiki_2019` — ENABLED (single-class: bots)
Same two-file shape. 704 ids, all `bot`; 698 join.
Self-identified bots from botwiki.org — art bots, weather bots, generative
poetry. Openly declared, benign automation, and therefore the most interesting
positive class in the family: a detector that cannot separate a botwiki bot from
a pronbot is measuring "is automated", not "is adversarial".

> Yang, Kai-Cheng, Onur Varol, Pik-Mai Hui, and Filippo Menczer. "Scalable and
> generalizable social bot detection through data selection." *AAAI* 34, no. 01
> (2020): 1096-1103. CC BY-NC-ND 4.0.

### `botrepo_celebrity_2019` — ENABLED (single-class: humans)
5,970 ids, all `human`. The user-object file holds **20,983 profiles**, of which
only 5,918 appear in the label file; the other 15,065 are unlabelled crawl
residue and are dropped by the inner join.
Sampling bias worth stating: these are high-follower broadcast accounts, so they
anchor the human class at the opposite end of the followers/following ratio from
the fake-follower bots — which is part of why the i.i.d. AUC below is optimistic.

> Yang et al. (2019), *HBET* 1(1): 48-61. CC BY-NC-ND 4.0.

### `botrepo_cresci_rtbust_2019` — ENABLED
759 ids, 368 human / 391 bot — the most balanced set in the family; 693 join
(340 / 353). Manually annotated Italian retweet-botnet accounts. Note the irony:
the RTbust method is temporal, and the temporal data is exactly what this
redistribution strips.

> Mazza, Michele, Stefano Cresci, Marco Avvenuti, Walter Quattrociocchi, and
> Maurizio Tesconi. "RTbust: Exploiting temporal patterns for botnet detection on
> Twitter." *WebSci* 2019, pp. 183-192. doi:10.5281/zenodo.2653137

### `botrepo_cresci_stock_2018` — ENABLED
25,987 ids, 7,479 human / 18,508 bot. Only 13,276 (51.1 %) join, and the loss is
label-dependent (see finding 3). Post-join it is 6,174 human / 7,102 bot — nearly
balanced, but the surviving bots are the ones still alive in 2019, so treat this
as a sample of *durable* stock spam rather than of stock spam.
This is the cashtag-piggybacking corpus that the disabled `cresci_2019` block
still wants, but only its account layer, so it does not close that gap.

> Cresci, Stefano, Fabrizio Lillo, Daniele Regoli, Serena Tardelli, and Maurizio
> Tesconi. "$FAKE: Evidence of Spam and Bot Activity in Stock Microblogs on
> Twitter." *ICWSM* 2018; and "Cashtag piggybacking: Uncovering spam and bot
> activity in stock microblogs on Twitter." *ACM TWEB* 13, no. 2 (2019): 11.
> doi:10.5281/zenodo.2686862

### `botrepo_gilani_2017` — ENABLED
2,652 rows / 2,633 unique ids (19 duplicates), 1,111 bot / 1,522 human;
2,484 join (1,090 / 1,394). Annotated across four follower-count bands, which
makes it the least follower-biased general sample here.

> Gilani, Zafar, Reza Farahbakhsh, Gareth Tyson, Liang Wang, and Jon Crowcroft.
> "Of bots and humans (on Twitter)." *ASONAM* 2017, pp. 349-354.

### `botrepo_midterm_2018` — ENABLED
The one set with a different shape: `midterm-2018.tsv` +
`midterm-2018_processed_user_objects.json` (27 MB), a flat, pre-flattened user
object using `user_id` / `user_created_at` / `probe_timestamp` instead of a
nested `user.*`. The shared alias table in `default.yaml` resolves both shapes.
50,538 ids, 42,446 bot / 8,092 human, **100 % coverage** — because these
profiles were saved at collection time rather than re-crawled later.
It is 53 % of the combined corpus by volume, so any number computed on the union
should also be reported with this corpus held out.

> Yang, Varol, Hui, and Menczer, *AAAI* 34(01): 1096-1103 (2020). CC BY-NC-ND 4.0.

### `botrepo_political_bots_2019` — ENABLED (single-class: bots)
62 ids, all `bot`, all 62 join. A single operator's political bot fleet
(`@rzazula`, since suspended). Statistically negligible; kept because it is one
coherent campaign and therefore a useful named slice when auditing which bot
types a model misses.

> Yang et al. (2019), *HBET* 1(1): 48-61. CC BY-NC-ND 4.0.

### `botrepo_pronbots_2019` — ENABLED (single-class: bots)
21,964 ids, all `bot`; 17,882 (81.4 %) join. Scam/spam bots harvested by Andy
Patel (github.com/r0zetta/pronbot2). Mass-registered, near-identical profiles —
the easiest positive class in the family and the one most likely to flatter a
headline number.

> Yang et al. (2019), *HBET* 1(1): 48-61. CC BY-NC-ND 4.0.

### `botrepo_vendor_purchased_2019` — ENABLED (single-class: bots)
1,088 ids, all `bot`; 1,087 join. Fake followers bought from several vendors —
ground truth in the strongest sense available, because the label comes from
having paid for them. Classic signature: default profile image, near-zero
statuses, high following count.

> Yang et al. (2019), *HBET* 1(1): 48-61. CC BY-NC-ND 4.0.

### `botrepo_verified_2019` — ENABLED (single-class: humans)
2,000 ids, all `human`; 1,987 join. Carries the celebrity-2019 caveat plus one
more: `verified` is itself a node feature, so on the combined corpus it is
partly a label proxy for these rows.

> Yang, Varol, Hui, and Menczer, *AAAI* 34(01): 1096-1103 (2020). CC BY-NC-ND 4.0.

### `botrepo_astroturf` — REJECTED
`astroturf.tar.gz` contains **exactly one member**: `astroturf.tsv`, 17 KB,
585 rows of `<user_id> \t political_Bot`. All one class, no user-object file, no
tweet file. Every profile field would be null and every graph field zero; it is
usable only after Twitter API hydration, which is not available. This is the
TweepFake trap and it is refused for the same reason.

> Sayyadiharikandeh, Mohsen, Onur Varol, Kai-Cheng Yang, Alessandro Flammini,
> and Filippo Menczer. "Detection of Novel Social Bots by Ensembles of
> Specialized Classifiers." *CIKM* 2020. CC BY-NC-ND 4.0.

### `botrepo_varol_2017` — REJECTED
`varol-2017.dat.gz` (15 KB) decompresses to 2,573 lines of
`<user_id> \t <0|1>`, 1,747 label-0 and 826 label-1. Ids and labels only; the
2016 crawl the paper used was never redistributed.

> Varol, Onur, Emilio Ferrara, Clayton A. Davis, Filippo Menczer, and Alessandro
> Flammini. "Online Human-Bot Interactions: Detection, Estimation, and
> Characterization." *ICWSM* 2017. CC BY-NC-ND 4.0.

### `botrepo_twibot_22` — REJECTED
`twibot-22.csv.gz` (7.5 MB) is a genuine CSV with header `id,label` and
**1,000,000 rows** (860,057 human / 139,943 bot), ids prefixed `u`. That is
TwiBot-22's label file and nothing else: no `user.json`, no `tweet_*.json`, no
`edge.csv`. The ids match nothing else on disk, so it is a million labels for
accounts we have no features for. The full release still requires the data-use
agreement at <https://twibot22.github.io/> — see the separate `twibot_22` block
in `default.yaml`, which points at the GitHub source repo.

> Feng, Shangbin, Zhaoxuan Tan, Herun Wan, Ningnan Wang, Zilong Chen, et al.
> "TwiBot-22: Towards Graph-Based Twitter Bot Detection." arXiv:2206.04564 (2022).

### `botrepo_twibot_20` — REJECTED (and the one worth chasing)
`twibot20.json` (3.2 MB) is a 100-record **sample** and is by far the most
feature-complete file in the whole download: each record has `ID`, a full
`profile` object, a `tweet` list (**17,372 tweet texts**, median 200 per
account) and a `neighbor` object with follower/following id lists on 62 of the
100 records.

It has **no `label` key on any record** (verified: 100/100 missing), so it
cannot be used for supervised training. The neighbour lists do not rescue it
either — only **30 of the 1,098** referenced ids are inside the sample, so the
induced graph is 30 edges over 100 nodes.

This is the highest-value follow-up in the download. The full TwiBot-20 release
has labels *and* tweet text *and* neighbour lists, which would give the graph
branch its second real interaction graph. Contact the authors.

> Feng, Shangbin, Herun Wan, Ningnan Wang, Jundong Li, and Minnan Luo.
> "TwiBot-20: A Comprehensive Twitter Bot Detection Benchmark." *CIKM* 2021.

### `caverlee_2011` — ENABLED (behavioral)
The completed archive contains 22,223 content-polluter and 19,276 legitimate
profiles plus 5,613,166 real tweet rows. After dropping 44 ids with conflicting
labels and applying `require_posts`, the loader returns 39,912 accounts,
1,686,109 capped posts, and 310,873 co-hashtag/co-mention/exact-text edges.

The tempting `*_followings.txt` files are not edge lists. Their second field is
the longitudinal `SeriesOfNumberOfFollowings`: the first value matches the
profile following count, values sit in count-like ranges, and almost none are
ids of labelled accounts. The loader therefore ignores them.

Measured warning: co-activity label assortativity is **0.4595**, raw numeric
`user_id` alone has **0.7998 ROC-AUC**, and an i.i.d. Random Forest reaches
**0.9725 ROC-AUC**. That score is an upper bound affected by collection-period
and id-allocation bias, not a publication claim.

> Lee, Kyumin, Brian David Eoff, and James Caverlee. "Seven Months with the
> Devils: A Long-Term Study of Content Polluters on Twitter." *ICWSM* 2011.

### `cresci_2015` — ENABLED (behavioral + observed graph)
Five groups are loaded directly from the nested archive: E13 and TFP are human;
FSF, INT and TWT are purchased fake followers. It returns 5,301 accounts,
279,943 capped posts, and 382,724 edges. Of those, 14,216 are surviving
`followers`/`following` rows shipped by the release; the remainder are
interaction and co-activity edges inferred from tweets.

Most shipped ego-network edges point outside the labelled corpus and are
discarded. The surviving observed graph is highly group-dependent:
assortativity is **0.8953**. Numeric ids alone reach **0.7886 ROC-AUC** and the
i.i.d. Random Forest reaches **0.9987**, so evaluation must be vendor/group
held-out rather than a random node split.

### `cresci_2017_full` — ENABLED (behavioral, seven bot types)
The complete release has nine labelled account groups. `require_posts` removes
tweetless accounts, including all of traditional-spambots 2/3/4, leaving 10,197
accounts across genuine users, fake followers, social spambots 1/2/3 and
traditional spambots 1. The loader streams all 6,637,616 tweet rows, keeps
714,856 under a 100-post/account cap, and builds 399,249 behavioral edges.

The retained `group` column enables per-bot-type reporting. Measured warning:
label assortativity is **0.6904**, numeric ids alone reach **0.7228 ROC-AUC**,
and an i.i.d. Random Forest reaches **0.9984**. Use bot-type holdouts and
cross-corpus transfer; do not present that random split as generalization.

---

## The combined corpus

```python
from aegis.config import load_config
from aegis.dataset_loaders import load_bot_repository_corpus

bundle = load_bot_repository_corpus(load_config())
bundle.nodes.groupby("source_dataset")["label"].agg(["size", "mean"])
```

`load_bot_repository_corpus` walks `BOT_REPOSITORY_GRAPHS`, skips anything
disabled in the config, **refuses any bundle whose provenance is not `REAL`**
(a synthetic stub must never be folded into something presented as a real
corpus), concatenates the node tables, drops duplicate `user_id`s first-writer-
wins, and keeps a `source_dataset` column on the result.

| | |
|---|---|
| accounts | **94,890** |
| label balance | 70,821 bot / 24,069 human (bot rate **0.746**) |
| edges / posts | 0 / 0 |
| duplicate rows dropped | 253 |

Per-source breakdown:

| source | accounts | bots | humans | bot rate |
|---|---:|---:|---:|---:|
| `botrepo_midterm_2018` | 50,521 | 42,446 | 8,075 | 0.840 |
| `botrepo_pronbots_2019` | 17,882 | 17,882 | 0 | 1.000 |
| `botrepo_cresci_stock_2018` | 13,275 | 7,102 | 6,173 | 0.535 |
| `botrepo_celebrity_2019` | 5,916 | 0 | 5,916 | 0.000 |
| `botrepo_gilani_2017` | 2,326 | 1,073 | 1,253 | 0.461 |
| `botrepo_verified_2019` | 1,932 | 0 | 1,932 | 0.000 |
| `botrepo_vendor_purchased_2019` | 1,070 | 1,070 | 0 | 1.000 |
| `botrepo_botwiki_2019` | 695 | 695 | 0 | 1.000 |
| `botrepo_cresci_rtbust_2019` | 693 | 353 | 340 | 0.509 |
| `botrepo_botometer_feedback_2019` | 518 | 138 | 380 | 0.266 |
| `botrepo_political_bots_2019` | 62 | 62 | 0 | 1.000 |

### Overlap and label conflicts

251 user_ids appear in more than one corpus. **213 agree on the label; 38 do
not**, and the disagreements are not random:

| conflicting pair | ids |
|---|---:|
| `celebrity_2019` (human) vs `gilani_2017` (bot) | 17 |
| `celebrity_2019` (human) vs `vendor_purchased_2019` (bot) | 15 |
| `gilani_2017` vs `verified_2019` | 4 |
| `gilani_2017` vs `midterm_2018` | 1 |
| `botometer_feedback_2019` vs `midterm_2018` | 1 |

The `celebrity` / `vendor-purchased` pairs are the interesting ones: fifteen
accounts that one OSoMe set treats as authentic celebrities and another treats
as purchased fake followers. First-writer-wins in `BOT_REPOSITORY_GRAPHS` order
resolves them to `celebrity_2019` (human). 38 rows out of 94,890 does not move
any metric, but it is a real ceiling on annotation quality and should be
mentioned rather than smoothed over.

---

## Measured signal: a RandomForest baseline

`RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
class_weight="balanced")` on the feature matrix produced by
`build_features(load_bot_repository_corpus(...))` — in which only
`account_age_days` and `followers_to_following` are non-zero.

**Stratified 5-fold, i.i.d. split: ROC-AUC = 0.9416.**

That number should not be quoted on its own. `source_dataset` is close to
perfectly predictive of `label` here (five of eleven corpora are 100 % bot, two
are 100 % human), and an i.i.d. split puts accounts from the same crawl on both
sides of it. Leave-one-corpus-out on the corpora that have both classes:

| held-out corpus | n | ROC-AUC |
|---|---:|---:|
| `botrepo_botometer_feedback_2019` | 518 | 0.643 |
| `botrepo_cresci_stock_2018` | 13,275 | 0.614 |
| `botrepo_gilani_2017` | 2,326 | 0.490 |
| `botrepo_cresci_rtbust_2019` | 693 | 0.467 |
| `botrepo_midterm_2018` | 50,521 | 0.419 |

So: two profile features carry **some** transferable signal on the two spam-like
corpora (0.61-0.64) and **none at all** on the manually-annotated general samples
(0.42-0.49, i.e. at or below chance — `midterm_2018` at 0.419 is inverted, which
is what you get when the training corpora's bots are low-follower fresh accounts
and midterm's are not).

The honest reading: this is **breadth, not depth**. The value of these eleven
corpora is 94,890 labelled accounts spanning eight distinct automation types
against three distinct human populations, which is a far better generalisation
test set than `cresci_2017`'s single 2014 campaign. It is not a substitute for
`cresci_2017` as a *training* corpus, because the features that make the graph
branch work — synchrony, reciprocity, content reuse — are all zero here. Any
result reported on this corpus must say which split was used.

---

## Reproducing the inspection

The archives extract in place on first `scan_dataset`, guarded by a
`.aegis_extracted` sentinel, so the second run costs nothing. To re-derive every
number above:

```python
from aegis.config import load_config
from aegis.dataset_loaders import BOT_REPOSITORY_GRAPHS, load_graph_dataset

settings = load_config()
for name in BOT_REPOSITORY_GRAPHS:
    print(load_graph_dataset(name, settings).summary())
```
