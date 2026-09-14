# Cresci-2019 — cashtag piggybacking retweet rings (2019, legacy)

**Branch:** GRAPH · **Put files in:** `datasets/cresci_2019/`

## Where to get it

```
Zenodo -> https://zenodo.org/records/2653525
Also listed on the Botometer bot repository page.
```

**Scriptable?** No — manual download.

## Expected layout

```
datasets/cresci_2019/
├── users.csv      (or *.json)
├── tweets.csv
└── edges.csv / retweets.csv
```

## Columns the loader looks for

```
Same harmonised shape as cresci_2017. The loader is tolerant about
exact filenames and will glob for user/tweet/edge-ish names.
```

## Notes

Retweet-ONLY accounts, which makes this the purest available example of
the reciprocity + temporal-synchrony signature that notebook 03 engineers
features for. Small, fast, and a very clean sanity check that the synchrony
feature is implemented correctly: if `synchrony_score` does not separate these
rings, the feature is broken.

## Reference

> Cresci, Lillo, Regoli, Tardelli, Tesconi. *Cashtag Piggybacking: Uncovering Spam and Bot Activity in Stock Microblogs on Twitter.* ACM TWEB 13(2), 2019.

---
*The loader checks this folder FIRST, before any network source. Anything you
drop here wins. Filenames are matched case-insensitively by glob, `.zip` and
`.tar.gz` archives are extracted in place, and nested folders are searched
recursively — so the raw download usually works untouched.*
