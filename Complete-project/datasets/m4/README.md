# M4 — Multi-generator, Multi-domain, Multi-lingual MGT (2024, modern)

**Branch:** TEXT · **Put files in:** `datasets/m4/`

## Where to get it

```
GitHub   -> https://github.com/mbzuai-nlp/M4                (canonical release)
HF       -> https://huggingface.co/datasets/MBZUAI-nlp/M4
SemEval  -> https://github.com/mbzuai-nlp/SemEval2024-task8   (Subtask A/B data)
Successor benchmark that also works here:
           https://huggingface.co/datasets/yaful/MAGE
```

**Scriptable?** `datasets.load_dataset('MBZUAI-nlp/M4')`, else the GitHub jsonl release.

## Expected layout

```
datasets/m4/            <- drop ANY subset; more generators = better
├── wikipedia_chatgpt.jsonl
├── wikihow_davinci.jsonl
├── reddit_cohere.jsonl
├── arxiv_bloomz.jsonl
├── peerread_dolly.jsonl
└── ... (nested folders are fine, the loader globs recursively)
```

## Columns the loader looks for

```
human_text     -> label 0
machine_text   -> label 1
model          generator name  -> `generator`  (davinci|chatGPT|cohere|dolly|bloomz|gpt4|llama)
source         domain          -> `domain`
id / pair_id   used as `group_id` so a human/machine PAIR stays in one split
```

## Notes

**This is the most important text dataset in the project.** The
`generator` column is what lets notebook 02 report leave-one-generator-out
accuracy. A 2026 agent will use a generator absent from training, so the
cross-generator number is the only one that means anything. Drop at least three
different generators or that experiment cannot run.

## Reference

> Wang, Mansurov, Ivanov, Su, Shelmanov, Tsvigun, Whitehouse, et al. *M4: Multi-generator, Multi-domain, and Multi-lingual Black-Box Machine-Generated Text Detection.* EACL 2024.

---
*The loader checks this folder FIRST, before any network source. Anything you
drop here wins. Filenames are matched case-insensitively by glob, `.zip` and
`.tar.gz` archives are extracted in place, and nested folders are searched
recursively — so the raw download usually works untouched.*
