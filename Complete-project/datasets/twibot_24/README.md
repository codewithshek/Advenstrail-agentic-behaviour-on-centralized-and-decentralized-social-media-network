# TwiBot-24 — LLM-powered bots, 14 relation types (2024, frontier)  ★ PRIMARY

**Branch:** GRAPH · **Put files in:** `datasets/twibot_24/`

## Where to get it

```
Request form -> https://twibot24.github.io/
**ACCESS REQUEST REQUIRED**, same process as TwiBot-22.
```

**Scriptable?** No — gated behind a signed agreement.

## Expected layout

```
datasets/twibot_24/
├── user.json
├── tweet.json          (may be sharded: tweet_0.json, …)
├── edge.csv
├── label.csv
└── split.csv
```

## Columns the loader looks for

```
edge.csv  : source_id, relation, target_id
            relation in {followers, following, post, mentioned, retweeted,
                         quoted, replied_to, membership, pinned, discuss, own}
label.csv : id, label ('human' | 'bot')
split.csv : id, split
```

## Notes

**This is the target dataset for notebook 03.** Its bot population
includes LLM-driven accounts whose *text* is often indistinguishable from
human — which is exactly why text-only detection collapses on it and the graph
branch becomes load-bearing. The 14 relation types are what justify a
relational GNN rather than a plain GCN. If you can only get one graph dataset,
get this one.

## Reference

> Feng, Wu, Liu, Tan, Wang, Zheng, et al. *TwiBot-24: Revisiting the State of Twitter Bot Detection.* 2024.

---
*The loader checks this folder FIRST, before any network source. Anything you
drop here wins. Filenames are matched case-insensitively by glob, `.zip` and
`.tar.gz` archives are extracted in place, and nested folders are searched
recursively — so the raw download usually works untouched.*
