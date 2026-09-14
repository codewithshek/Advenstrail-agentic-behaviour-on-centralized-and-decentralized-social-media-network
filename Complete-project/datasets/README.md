# `datasets/` — dataset acquisition reference

The active configuration does **not** read corpus data from these placeholder
folders. `ml/configs/default.yaml` points each `manual_dir` at the shared
workspace-level `../DataSets/...` tree, which contains the real local corpora.
The folders here retain download/licensing instructions and `.gitkeep` markers
only.

For each configured `manual_dir`, the loader resolves sources in this order:

```
1. configured manual_dir (currently ../DataSets/...)   (highest priority)
2. Hugging Face hub     ← automatic, needs HF_TOKEN for gated repos
3. Kaggle API           ← automatic, needs ~/.kaggle/kaggle.json
4. synthetic fallback   ← schema-identical stub so the pipeline still runs
```

Local wins because it is explicit, free, offline, and instant on re-runs. If a
folder below has usable files, the loader will never touch the network for that
dataset. If a folder is empty and the network route is blocked (gated repo, no
credentials, access request pending), the loader emits a **clearly-labelled
synthetic stub** and records `provenance: SYNTHETIC_FALLBACK` in
`data/manifest.json`. This keeps ingestion testable, but the production text
notebook refuses smoke/fallback input; `aegis.io_utils.assert_real_data()` also
prevents publication of a metric computed on a stub.

## Readiness check

```powershell
.\.venv\Scripts\python.exe scripts\dataset_status.py
```

## What goes where

Each subfolder has its own `README.md` with the download URL, expected filenames,
and columns. Place acquired files in the matching `manual_dir` from
`ml/configs/default.yaml`, not beside these reference READMEs. Summary:

| Folder | Dataset | Era | Branch | How to get it |
|---|---|---|---|---|
| `tweepfake/` | TweepFake | 2021 legacy | text | Disabled until legally hydrated text is available |
| `hc3/` | Hello-SimpleAI/HC3 | 2022 legacy | text | Hugging Face, **public** |
| `m4/` | M4 multi-generator ★ | 2024 modern | text | Hugging Face / GitHub |
| `wildguard/` | WildGuardMix | 2024 frontier | text | Hugging Face, **gated** — one-time click-through |
| `wildjailbreak/` | WildJailbreak | 2024 frontier | text | Hugging Face, **gated** — one-time click-through |
| `cresci_2017/` | Cresci-2017 | 2017 legacy | graph | Botometer repo / Zenodo / Kaggle |
| `cresci_2019/` | Cresci-2019 | 2019 legacy | graph | Zenodo |
| `twibot_22/` | TwiBot-22 | 2022 legacy | graph | **Access request** — twibot22.github.io |
| `twibot_24/` | TwiBot-24 ★ | 2024 frontier | graph | **Access request** — twibot24.github.io |

★ = highest value. If you only fetch two datasets, fetch **M4** (text) and
**TwiBot-24** (graph): those two carry the 2024/2026 threat model that the whole
project is arguing about. Everything else is the historical baseline that makes
the comparison meaningful.

## You do not have to be tidy

The resolver is deliberately forgiving, because raw downloads never match a
spec exactly:

- **Archives are fine.** `.zip`, `.tar.gz`, `.tgz`, `.gz` are extracted in place
  on first use (and skipped on later runs).
- **Nested folders are fine.** Discovery is recursive, so
  `datasets/m4/M4-main/data/wikipedia_chatgpt.jsonl` is found.
- **Case does not matter.** `Train.CSV` matches `train.csv`.
- **Format is sniffed, not assumed.** `.csv`, `.tsv`, `.json`, `.jsonl`,
  `.parquet`, `.txt` are all read; delimiter is detected for `.csv`/`.tsv`/`.txt`.
- **Column names are sniffed too.** Mirrors rename things
  (`account.type` / `account_type` / `label`; `friends_count` / `following_count`).
  The loader matches a set of known aliases per field. **Do not rename columns
  to "help"** — you are more likely to break the match than fix it.
- **Partial is fine.** Three M4 generator files are enough to train; one Cresci
  subfolder is enough to smoke-test. The loader uses what it finds and records
  the row count.

## What NOT to do

- Do not commit these files. `.gitignore` excludes `datasets/**` except the
  READMEs and `.gitkeep` files. The corpora are up to ~60 GB and several are
  covered by data-use agreements that forbid redistribution.
- Do not redistribute TwiBot-22/24 or the gated AI2 sets. Point collaborators at
  the request form; do not hand them your copy.
- Do not mix datasets across folders. Label derivation for Cresci depends on the
  original folder names, and provenance tracking depends on one corpus per
  directory.

## Disk budget

| Dataset | Compressed | Notes |
|---|---|---|
| TweepFake | ~10 MB | trivial |
| HC3 | ~80 MB | trivial |
| M4 | ~1–3 GB | depends how many generator files you take |
| WildGuardMix | ~150 MB | |
| WildJailbreak | ~500 MB | 262k rows |
| Cresci-2017 | ~1.5 GB | tweets.csv dominates |
| Cresci-2019 | ~300 MB | |
| TwiBot-22 | ~60 GB | **only if you want the 2022-vs-2024 comparison** |
| TwiBot-24 | ~15 GB | the one that matters |

Smoke-test mode (`AEGIS_SMOKE_TEST=1`, the default) caps every corpus at 1,500
rows after loading so ingestion can be checked on partial downloads. Notebook
02 deliberately requires `AEGIS_SMOKE_TEST=0`; smoke output is never a
production text artifact.
