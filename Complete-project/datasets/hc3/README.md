# HC3 — Human-ChatGPT Comparison Corpus (Dec 2022, legacy)

**Branch:** TEXT · **Put files in:** `datasets/hc3/`

## Where to get it

```
Hugging Face -> https://huggingface.co/datasets/Hello-SimpleAI/HC3
Public (not gated). English config recommended: `all`.
```

**Scriptable?** `datasets.load_dataset('Hello-SimpleAI/HC3', 'all')` — works with no token.

## Expected layout

```
datasets/hc3/
├── all.jsonl                 <- easiest: one file, everything
│   (or the per-domain files, any subset works:)
├── reddit_eli5.jsonl
├── open_qa.jsonl
├── wiki_csai.jsonl
├── medicine.jsonl
└── finance.jsonl
```

## Columns the loader looks for

```
id                row id
question          the prompt both sides answered
human_answers     LIST of strings   -> exploded into label 0
chatgpt_answers   LIST of strings   -> exploded into label 1
source            domain tag
```

## Notes

The two answer columns are LISTS. The loader explodes them and keeps
`question` as `group_id`, so the human and ChatGPT answers to the SAME question
can never land in different splits. That grouping is worth ~3-5 F1 points of
honesty — without it the model memorises the topic.

## Reference

> Guo, Zhang, Wang, Jiang, Nie, Ding, Yue, Wu. *How Close is ChatGPT to Human Experts?* arXiv:2301.07597, 2023.

---
*The loader checks this folder FIRST, before any network source. Anything you
drop here wins. Filenames are matched case-insensitively by glob, `.zip` and
`.tar.gz` archives are extracted in place, and nested folders are searched
recursively — so the raw download usually works untouched.*
