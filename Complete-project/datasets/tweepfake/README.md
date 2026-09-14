# TweepFake — Deepfake Tweet Detection (2021, legacy)

**Branch:** TEXT · **Put files in:** `datasets/tweepfake/`

## Where to get it

```
Kaggle  ->  https://www.kaggle.com/datasets/mtesconi/twitter-deep-fake-text
Zenodo  ->  https://zenodo.org/records/4592516  (mirror; look for `tweepfake` release assets)
```

**Scriptable?** `kaggle datasets download -d mtesconi/twitter-deep-fake-text` (needs ~/.kaggle/kaggle.json)

## Expected layout

```
datasets/tweepfake/
├── train.csv
├── validation.csv
└── test.csv
```

## Columns the loader looks for

```
text            the tweet body
account.type    'human' | 'bot'          <- becomes label (bot => 1)
class_type      'human' | 'gpt2' | 'rnn' | 'othr' | ...   <- kept as `generator`
screen_name     author handle (optional)
```

## Notes

A .zip is fine — the loader extracts it in place. Column names differ
slightly across mirrors ('account.type' vs 'account_type' vs 'label'); the
loader sniffs for all known variants, so do not rename anything.

## Reference

> Fagni, Falchi, Gambini, Martella, Tesconi. *TweepFake: About detecting deepfake tweets.* PLoS ONE 16(5), 2021.

---
*The loader checks this folder FIRST, before any network source. Anything you
drop here wins. Filenames are matched case-insensitively by glob, `.zip` and
`.tar.gz` archives are extracted in place, and nested folders are searched
recursively — so the raw download usually works untouched.*
