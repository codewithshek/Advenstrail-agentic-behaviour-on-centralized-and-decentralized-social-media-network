# TwiBot-22 — large heterogeneous bot benchmark (2022, legacy)

**Branch:** GRAPH · **Put files in:** `datasets/twibot_22/`

## Where to get it

```
Request form -> https://twibot22.github.io/
**ACCESS REQUEST REQUIRED.** You sign a data-use agreement and receive a
download link. Expect ~60 GB uncompressed for the full release.
```

**Scriptable?** No — gated behind a signed agreement.

## Expected layout

```
datasets/twibot_22/
├── user.json           (1M users)
├── tweet_0.json … tweet_8.json
├── edge.csv            (170M edges: source_id, relation, target_id)
├── label.csv           (id, label: 'human' | 'bot')
├── split.csv           (id, split: train | val | test)
├── hashtag.json        (optional)
└── list.json           (optional)
```

## Columns the loader looks for

```
edge.csv   : source_id, relation, target_id
             relation in {follow, friend, post, ...}
label.csv  : id, label
split.csv  : id, split
```

## Notes

**Disk-space warning.** The loader streams `edge.csv` in chunks and
never loads all tweets at once, but you still need the space. If you only want
the pipeline to run, TwiBot-24 alone is enough — this one is for the
2022-vs-2024 difficulty comparison in the write-up.

## Reference

> Feng, Tan, Wan, Wang, Zheng, Zhang, et al. *TwiBot-22: Towards Graph-Based Twitter Bot Detection.* NeurIPS 2022 Datasets & Benchmarks.

---
*The loader checks this folder FIRST, before any network source. Anything you
drop here wins. Filenames are matched case-insensitively by glob, `.zip` and
`.tar.gz` archives are extracted in place, and nested folders are searched
recursively — so the raw download usually works untouched.*
