# WildGuard — prompt/response safety + adversarial jailbreaks (2024, frontier)

**Branch:** TEXT · **Put files in:** `datasets/wildguard/`

## Where to get it

```
Hugging Face -> https://huggingface.co/datasets/allenai/wildguardmix
**GATED.** You must be logged in and click 'Agree and access' on the dataset
page once; after that a read token works.
```

**Scriptable?** `datasets.load_dataset('allenai/wildguardmix', 'wildguardtrain', token=HF_TOKEN)`

## Expected layout

```
datasets/wildguard/
├── wildguardtrain.jsonl      (or .parquet / .csv — any of these work)
└── wildguardtest.jsonl
```

## Columns the loader looks for

```
prompt                  user turn
response                model turn (may be null)
prompt_harm_label       'harmful' | 'unharmful'
response_harm_label     'harmful' | 'unharmful' | null
response_refusal_label  'refusal' | 'compliance' | null
adversarial             bool  <- TRUE means a jailbreak attempt
subcategory             fine-grained harm type
```

## Notes

This is the corpus that teaches the detector to recognise an agent
being *weaponised* — the malicious-command half of the threat model. Label
mapping used by the loader: label 1 if (prompt_harm_label == harmful) OR
(adversarial == True); the `unharmful & adversarial` rows are retained as HARD
NEGATIVES because they are what stops the model flagging every odd-looking
prompt.

## Reference

> Han, Rao, Ziems, Kumar, Ammanabrolu, et al. *WildGuard: Open One-stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs.* NeurIPS 2024 Datasets & Benchmarks.

---
*The loader checks this folder FIRST, before any network source. Anything you
drop here wins. Filenames are matched case-insensitively by glob, `.zip` and
`.tar.gz` archives are extracted in place, and nested folders are searched
recursively — so the raw download usually works untouched.*
