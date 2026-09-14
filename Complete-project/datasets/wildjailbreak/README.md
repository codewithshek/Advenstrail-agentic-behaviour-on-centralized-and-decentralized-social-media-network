# WildJailbreak — vanilla/adversarial jailbreak pairs (2024, frontier)

**Branch:** TEXT · **Put files in:** `datasets/wildjailbreak/`

## Where to get it

```
Hugging Face -> https://huggingface.co/datasets/allenai/wildjailbreak
**GATED.** Same one-time 'Agree and access' click as WildGuard.
Files are TSV in the repo (`train/train.tsv`, `eval/eval.tsv`).
```

**Scriptable?** `datasets.load_dataset('allenai/wildjailbreak', 'train', token=HF_TOKEN, delimiter='\t')`

## Expected layout

```
datasets/wildjailbreak/
├── train.tsv
└── eval.tsv
```

## Columns the loader looks for

```
vanilla       plain-language version of the request
adversarial   the jailbroken rewrite (the interesting column)
completion    reference safe completion
data_type     vanilla_harmful | vanilla_benign
              adversarial_harmful | adversarial_benign
```

## Notes

The `*_benign` splits are the whole point. 262k rows of which a large
fraction are benign-but-weirdly-phrased. Training only on harmful jailbreaks
produces a detector with a catastrophic false-positive rate on eccentric human
writing, which is the single fastest way to lose analyst trust. The loader
keeps benign rows at label 0 on purpose.

## Reference

> Jiang, Rao, Han, Ettinger, Brahman, Kumar, et al. *WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models.* NeurIPS 2024.

---
*The loader checks this folder FIRST, before any network source. Anything you
drop here wins. Filenames are matched case-insensitively by glob, `.zip` and
`.tar.gz` archives are extracted in place, and nested folders are searched
recursively — so the raw download usually works untouched.*
