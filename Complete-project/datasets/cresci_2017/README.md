# Cresci-2017 — social spambots & fake followers (2017, legacy)

**Branch:** GRAPH · **Put files in:** `datasets/cresci_2017/`

## Where to get it

```
Botometer / OSoMe -> https://botometer.osome.iu.edu/bot-repository/datasets.html
Zenodo mirror     -> https://zenodo.org/records/7013252
Kaggle mirror     -> https://www.kaggle.com/datasets/hsankesara/twitter-bot-detection-dataset
```

**Scriptable?** Partly: the Kaggle mirror is scriptable; the OSoMe original needs a manual click-through.

## Expected layout

```
datasets/cresci_2017/           <- keep the original folder names!
├── genuine_accounts.csv/
│   ├── users.csv
│   ├── tweets.csv
│   ├── friends.csv
│   └── followers.csv
├── social_spambots_1.csv/  (… _2, _3)
├── traditional_spambots_1.csv/  (… _2, _3, _4)
└── fake_followers.csv/
```

## Columns the loader looks for

```
users.csv     : id, screen_name, followers_count, friends_count,
                statuses_count, created_at, verified, description, ...
tweets.csv    : id, user_id, text, created_at, retweet_count, ...
friends.csv   : source_id, target_id      -> 'following' edges
followers.csv : source_id, target_id      -> 'followers' edges
```

## Notes

Label comes from the FOLDER NAME (genuine_* => 0, *spambots*/fake_* => 1),
so do not flatten the directory structure. This dataset introduced the
group-level view of automation that the whole graph branch rests on — it is the
1.0 baseline against which TwiBot-24's LLM bots look genuinely hard.

## Reference

> Cresci, Di Pietro, Petrocchi, Spognardi, Tesconi. *The Paradigm-Shift of Social Spambots: Evidence, Theories, and Tools for the Arms Race.* WWW '17 Companion.

---
*The loader checks this folder FIRST, before any network source. Anything you
drop here wins. Filenames are matched case-insensitively by glob, `.zip` and
`.tar.gz` archives are extracted in place, and nested folders are searched
recursively — so the raw download usually works untouched.*
