# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (AEGIS-SN)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 02 · Text Branch — Fine-tuning DeBERTa-v3
#
# **AEGIS-SN** — the NLP half of the hybrid detector.
#
# Binary target: **0 = human / benign**, **1 = machine-generated *or* adversarial**.
#
# ## Why those two things share one label
#
# It looks like a category error to put "an LLM wrote this" and "this is a jailbreak" in the
# same positive class. The justification is the deployed question, which is *"does an analyst
# need to look at this account?"* — and that is binary. An agentic account is a threat both
# when it fabricates consensus (machine-generated text) and when it carries a payload aimed at
# other agents (prompt injection). A model with one head answering the actual decision beats
# two models whose outputs someone then has to reconcile.
#
# The finer taxonomy is not thrown away: `threat_class` rides along, §7 reports per-class
# performance, and the dashboard uses it to explain *why* something fired.
#
# ## What this notebook establishes, in order
#
# 1. **Baselines first.** TF-IDF and a length-only classifier. Until you know what a bag of
#    words gets, a transformer's F1 is a number without a scale.
# 2. **Fine-tune** `microsoft/deberta-v3-base` with class-weighted loss.
# 3. **Tune the decision threshold** on validation — not 0.5.
# 4. **Cross-generator holdout** — the number that actually matters, §8.
#
# ## Prerequisite
#
# Run `01_data_ingestion_and_synthetic_gen.ipynb` first; this notebook reads only from
# `data/processed/`.

# %% [markdown]
# ## 1 · Environment
#
# **On Python version:** DeBERTa-v3 uses a SentencePiece tokenizer, and `sentencepiece`
# wheels are reliable on **Python 3.10/3.11** and patchy on 3.13+. If the tokenizer import
# fails, that is almost always the cause — build the venv on 3.11.

# %%
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_here = Path.cwd()
for _candidate in (_here, *_here.parents):
    if (_candidate / "ml" / "src" / "aegis").is_dir():
        sys.path.insert(0, str(_candidate / "ml" / "src"))
        break
else:
    raise RuntimeError("Could not locate ml/src/aegis — launch Jupyter from the repo root.")

from aegis import config as acfg
from aegis import io_utils as iou
from aegis import metrics as amx
from aegis import viz

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200)

settings = acfg.load_config()
acfg.set_seed(settings.seed)

DEVICE = acfg.resolve_device(settings.device)
CFG = settings.text_model
PRODUCTION_CHECKPOINT = CFG.get("base_checkpoint", "microsoft/deberta-v3-base")
CHECKPOINT = PRODUCTION_CHECKPOINT
MAX_LEN = int(CFG.get("max_length", 256))
FBETA = float(settings.fusion_model.get("fbeta", 1.5))

if settings.smoke_test:
    raise RuntimeError(
        "Notebook 02 is production-only. Set AEGIS_SMOKE_TEST=0 and rerun "
        "notebook 01 so the processed splits contain the full corpus."
    )
if DEVICE != "cuda":
    warnings.warn(
        "Full DeBERTa-v3 training without CUDA can take days. The notebook is "
        "correctly configured, but run it on a CUDA GPU for practical training."
    )

print(f"device      : {DEVICE}")
print(f"checkpoint  : {CHECKPOINT}")
print(f"max_length  : {MAX_LEN}")
print("training    : full-data production path (smoke artifacts disabled)")

# %% [markdown]
# ## 2 · Load the splits and re-check provenance
#
# Notebook 01 already gated on this. It is checked again here because these two notebooks
# get run days apart, and a metric computed on a stub that nobody re-verified is exactly the
# thing that ends up in a report.

# %%
splits = {}
for _name in ("train", "validation", "test"):
    _path = settings.paths.processed / f"text_{_name}.parquet"
    if not _path.exists():
        raise FileNotFoundError(f"{_path} missing — run notebook 01 first.")
    splits[_name] = iou.load_frame(_path)

HOLDOUT = [str(g).lower() for g in CFG.get("holdout_generators", [])]

def _is_holdout(value: object) -> bool:
    generator = str(value or "").lower()
    return any(name in generator for name in HOLDOUT)

# Establish the unseen-generator protocol before any fitting or threshold
# tuning. Held-out generators never enter the primary train or validation set.
_all_rows = pd.concat(splits.values(), ignore_index=True)
_holdout_machine = _all_rows[
    _all_rows["generator"].map(_is_holdout) & (_all_rows["label"] == 1)
].copy()
_human_pool = splits["test"][splits["test"]["label"] == 0]
_n_human = min(len(_human_pool), len(_holdout_machine))
_holdout_human = _human_pool.sample(n=_n_human, random_state=settings.seed)
holdout_df = pd.concat([_holdout_machine, _holdout_human], ignore_index=True).sample(
    frac=1.0, random_state=settings.seed
).reset_index(drop=True)

for _name in splits:
    splits[_name] = splits[_name][
        ~splits[_name]["generator"].map(_is_holdout)
    ].reset_index(drop=True)
train_df, val_df, test_df = splits["train"], splits["validation"], splits["test"]

if not HOLDOUT or _holdout_machine.empty:
    raise RuntimeError(
        f"Configured holdout generators {HOLDOUT} matched no machine rows; "
        "fix generator labels before training."
    )

print(pd.DataFrame([
    {
        "split": k,
        "rows": len(v),
        "pos_rate": round(float(v["label"].mean()), 3),
        "sources": v["source_dataset"].nunique(),
        "generators": v["generator"].nunique(),
        "median_chars": int(v["text"].str.len().median()),
    }
    for k, v in splits.items()
]).to_string(index=False))
print(
    f"strict generator holdout: {HOLDOUT} -> {len(holdout_df):,} rows "
    f"({len(_holdout_machine):,} machine + {_n_human:,} human controls)"
)

_audit = acfg.manifest_summary(settings.paths)
_in_corpus = set(pd.concat(splits.values())["source_dataset"].unique())
_stubs = _audit[
    _audit["provenance"].eq(iou.PROV_SYNTHETIC_FALLBACK) & _audit["dataset"].isin(_in_corpus)
]
if len(_stubs):
    raise RuntimeError(
        "Production training refused: synthetic fallback corpora remain in the "
        f"processed splits: {_stubs['dataset'].tolist()}"
    )
print(f"\nprovenance OK — {len(_in_corpus)} sources, no synthetic fallbacks")

# %%
print("composition of the training split:")
print(
    train_df.groupby(["source_dataset", "threat_class"]).size()
    .rename("rows").reset_index().to_string(index=False)
)

# %% [markdown]
# ## 3 · Baselines
#
# Two of them, and the second is the more informative:
#
# * **TF-IDF + logistic regression** — character and word n-grams. Strong on this task,
#   because pre-2024 machine text has real lexical tells.
# * **Length only** — a single feature. This is the shortcut detector. If it scores well
#   above chance, then part of any transformer's score is also just length, and §7's
#   length-stratified breakdown is where that gets checked.

# %%
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

baselines: dict[str, amx.ClassificationReport] = {}

_t0 = time.time()
_tfidf = make_pipeline(
    TfidfVectorizer(
        max_features=200_000, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
        strip_accents="unicode", lowercase=True,
    ),
    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=settings.seed),
)
_tfidf.fit(train_df["text"], train_df["label"])
_p = _tfidf.predict_proba(test_df["text"])[:, 1]
baselines["tfidf_logreg"] = amx.evaluate(test_df["label"], _p, beta=FBETA)
print(f"tf-idf + logreg   ({time.time() - _t0:.0f}s):  {baselines['tfidf_logreg']}")

_len_train = train_df["text"].str.len().to_numpy().reshape(-1, 1)
_len_test = test_df["text"].str.len().to_numpy().reshape(-1, 1)
_lenclf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=settings.seed),
)
_lenclf.fit(_len_train, train_df["label"])
baselines["length_only"] = amx.evaluate(
    test_df["label"], _lenclf.predict_proba(_len_test)[:, 1], beta=FBETA
)
print(f"length only       :  {baselines['length_only']}")

_dummy = DummyClassifier(strategy="stratified", random_state=settings.seed)
_dummy.fit(train_df[["text"]], train_df["label"])
baselines["random"] = amx.evaluate(
    test_df["label"], _dummy.predict_proba(test_df[["text"]])[:, 1], beta=FBETA
)
print(f"stratified random :  {baselines['random']}")

print("\n" + amx.compare_reports(baselines).to_string())

# %% [markdown]
# **Interpreting the length baseline.** Anything meaningfully above ~0.55 ROC-AUC means the
# corpus has a length confound. Some of it is genuine — a jailbreak prompt really is longer
# than the vanilla request it rewrites, because the elaborate framing *is* the attack. The
# risk is that the model learns only that, and then misses a short adversarial post. §7
# breaks performance down by text length to check.

# %% [markdown]
# ## 4 · Tokenisation
#
# `max_length=256` tokens. Most social posts fit well inside that; the long tail is M4's
# arXiv abstracts and WildJailbreak's roleplay framings, which get truncated. Truncation is
# the right trade: 512 doubles attention cost for a minority of rows, and for adversarial
# prompts the tell is usually in the opening instruction anyway.

# %%
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
print(f"tokenizer: {type(tokenizer).__name__}  vocab={tokenizer.vocab_size:,}")


class TextDataset(Dataset):
    """Tokenise lazily so the full corpus is never held as tensors."""

    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int):
        self.texts = frame["text"].astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoded = self.tokenizer(
            self.texts[idx], truncation=True, max_length=self.max_length, padding=False,
        )
        encoded["labels"] = self.labels[idx]
        return encoded


train_ds = TextDataset(train_df, tokenizer, MAX_LEN)
val_ds = TextDataset(val_df, tokenizer, MAX_LEN)
test_ds = TextDataset(test_df, tokenizer, MAX_LEN)

_lens = [len(tokenizer(t, truncation=False)["input_ids"]) for t in train_df["text"].head(2000)]
print(f"token lengths (2k sample): median={int(np.median(_lens))} "
      f"p95={int(np.percentile(_lens, 95))} max={max(_lens)}")
print(f"truncated at {MAX_LEN}: {100 * np.mean(np.array(_lens) > MAX_LEN):.1f}% of rows")

# %% [markdown]
# ## 5 · Model and class-weighted loss
#
# The corpus is imbalanced and deliberately not resampled — the ratio reflects how these
# sources actually are, and SMOTE-ing text is meaningless. Instead the loss is weighted by
# inverse class frequency, which leaves the data honest and moves the cost.

# %%
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

_classes = np.array(sorted(train_df["label"].astype(int).unique()))
_computed = compute_class_weight(
    class_weight="balanced",
    classes=_classes,
    y=train_df["label"].astype(int).to_numpy(),
)
class_weights = {int(label): float(weight) for label, weight in zip(_classes, _computed)}
weight_tensor = torch.tensor(
    [class_weights[index] for index in range(int(CFG.get("num_labels", 2)))],
    dtype=torch.float,
)
print(f"class weights (training split only): {class_weights}")

model = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT,
    num_labels=int(CFG.get("num_labels", 2)),
    id2label={0: "human_benign", 1: "adversarial"},
    label2id={"human_benign": 0, "adversarial": 1},
)
print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")


class WeightedTrainer(Trainer):
    """Trainer with train-only weighted CE and an explicit linear scheduler."""

    def __init__(self, *args, class_weight: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weight = class_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits.view(-1, model.config.num_labels),
            labels.view(-1),
            weight=self.class_weight.to(outputs.logits.device),
        )
        return (loss, outputs) if return_outputs else loss

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None:
            optimizer = optimizer or self.optimizer
            self.lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        return self.lr_scheduler


def compute_metrics(eval_pred) -> dict:
    """Metrics reported per epoch. Threshold remains fixed during early stopping."""
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    report = amx.evaluate(labels, probs, threshold=0.5, beta=FBETA)
    return {
        "f1": report.f1, "precision": report.precision, "recall": report.recall,
        "roc_auc": report.roc_auc, "pr_auc": report.pr_auc,
    }


# %% [markdown]
# ### Training configuration
#
# Under `smoke_test` this runs 1 epoch on ~1,500 rows to prove the wiring. The real run
# needs `AEGIS_SMOKE_TEST=0` and a GPU; on CPU, full training is measured in days, not hours.

# %%
EPOCHS = max(4, int(CFG.get("epochs", 4)))
FP16 = (DEVICE == "cuda") if CFG.get("fp16", "auto") == "auto" else bool(CFG.get("fp16"))
OUT_DIR = settings.paths.text_model

args = TrainingArguments(
    output_dir=str(OUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=int(CFG.get("batch_size", 16)),
    per_device_eval_batch_size=int(CFG.get("eval_batch_size", 64)),
    gradient_accumulation_steps=int(CFG.get("gradient_accumulation_steps", 2)),
    learning_rate=float(CFG.get("learning_rate", 2e-5)),
    weight_decay=float(CFG.get("weight_decay", 0.01)),
    warmup_ratio=float(CFG.get("warmup_ratio", 0.06)),
    lr_scheduler_type="linear",
    optim="adamw_torch",
    max_grad_norm=float(CFG.get("max_grad_norm", 1.0)),
    fp16=FP16,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model=str(CFG.get("metric_for_best_model", "eval_f1")),
    greater_is_better=True,
    save_total_limit=2,
    logging_steps=50,
    seed=settings.seed,
    data_seed=settings.seed,
    report_to=[],
    dataloader_pin_memory=(DEVICE == "cuda"),
)

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=int(CFG.get("early_stopping_patience", 2))
    )],
    class_weight=weight_tensor,
)

print(f"epochs={EPOCHS} batch={args.per_device_train_batch_size} "
      f"lr={args.learning_rate} scheduler=linear fp16={FP16}")
print(f"optimisation steps: ~{len(train_ds) * EPOCHS // (args.per_device_train_batch_size * args.gradient_accumulation_steps):,}")

# %%
_t0 = time.time()
train_result = trainer.train()
print(f"\ntrained in {(time.time() - _t0) / 60:.1f} min")
print(f"final training loss: {train_result.training_loss:.4f}")

trainer.save_model(str(OUT_DIR))
tokenizer.save_pretrained(str(OUT_DIR))
print(f"saved -> {OUT_DIR}")

# %%
history = pd.DataFrame([h for h in trainer.state.log_history if "eval_f1" in h])
if len(history):
    print(history.loc[:, [c for c in history.columns if c.startswith("eval_") or c == "epoch"]]
          .to_string(index=False))
    viz.plot_training_curve(
        trainer.state.log_history,
        save_as=settings.paths.figures / "02_training_curve.png",
    )


# %% [markdown]
# ## 6 · Threshold selection
#
# The default 0.5 is arbitrary. It is only optimal when the classes are balanced *and* the
# two error types cost the same, and neither holds here.
#
# We tune on **validation** and apply to test, using **F-beta with β=1.5**
# (`fusion_model.fbeta`). β>1 weights recall above precision, because a missed swarm does
# more damage than a false alarm — but only mildly above, because false positives are what
# destroy an analyst's trust in the tool, and a detector nobody believes has an effective
# recall of zero.
#
# Tuning on test and reporting that same number would be selecting on the test set. The
# threshold is chosen on validation and then frozen.

# %%
def predict_proba(dataset) -> np.ndarray:
    logits = trainer.predict(dataset).predictions
    return torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()


val_scores = predict_proba(val_ds)
test_scores = predict_proba(test_ds)

best_threshold, val_report = amx.tune_threshold(
    val_df["label"].to_numpy(), val_scores, objective="fbeta", beta=FBETA
)
print(f"tuned threshold : {best_threshold:.3f}   (default would be 0.500)")
print(f"validation      : {val_report}")

_at_default = amx.evaluate(val_df["label"], val_scores, threshold=0.5, beta=FBETA)
print(f"val @ 0.500     : {_at_default}")
print(f"\nF{FBETA} gain from tuning: {val_report.fbeta - _at_default.fbeta:+.4f}")

viz.plot_threshold_sweep(
    val_df["label"], val_scores, beta=FBETA, chosen=best_threshold,
    save_as=settings.paths.figures / "02_threshold_sweep.png",
)

# %% [markdown]
# ## 7 · Test-set results
#
# Bootstrap confidence intervals are reported alongside the point estimates. On a
# smoke-test split of ~200 test rows a bare F1 has a CI wide enough to swallow most of the
# claims one might want to make with it; showing the interval keeps that visible.

# %%
test_report = amx.evaluate(test_df["label"], test_scores, threshold=best_threshold, beta=FBETA)
print("DeBERTa-v3, test set")
print(f"  {test_report}")
print(f"\n{amx.confusion_frame(test_report).to_string()}")

for _metric in ("f1", "precision", "recall", "roc_auc"):
    _pt, _lo, _hi = amx.bootstrap_ci(
        test_df["label"].to_numpy(), test_scores, metric=_metric,
        threshold=best_threshold, seed=settings.seed,
    )
    print(f"  {_metric:<10} {_pt:.4f}   95% CI [{_lo:.4f}, {_hi:.4f}]")

# %%
all_models = dict(baselines)
all_models["deberta_v3"] = test_report
print(amx.compare_reports(all_models).to_string())

viz.plot_roc_pr(test_df["label"], test_scores,
                save_as=settings.paths.figures / "02_roc_pr.png")
viz.plot_confusion(test_report,
                   save_as=settings.paths.figures / "02_confusion.png")
viz.plot_score_distributions(test_df["label"], test_scores, threshold=best_threshold,
                             save_as=settings.paths.figures / "02_score_dist.png")

# %% [markdown]
# ### Breakdown by threat class, source and era
#
# The aggregate F1 hides the interesting failures. Three cuts:
#
# * **threat class** — is `prompt_injection` (short, imperative, sometimes German) detected
#   as well as `machine_generated` (long, fluent, English)?
# * **source dataset** — a big gap between HC3 and `llm_tweet` means the model has learned
#   *ChatGPT-2022 register*, not machine text.
# * **era** — accuracy should degrade from `legacy` to `frontier_2026`. If it does not, be
#   suspicious rather than pleased.

# %%
test_eval = test_df.copy()
test_eval["score"] = test_scores
test_eval["pred"] = (test_scores >= best_threshold).astype(int)
test_eval["correct"] = test_eval["pred"] == test_eval["label"]

for _dim in ("threat_class", "source_dataset", "era"):
    print(f"\n{'=' * 78}\nby {_dim}\n{'=' * 78}")
    print(amx.per_group_report(
        test_eval["label"], test_eval["score"], test_eval[_dim],
        threshold=best_threshold, beta=FBETA,
    ).to_string(index=False))

# %% [markdown]
# ### The length confound, revisited
#
# §3's length-only baseline told us how much signal length carries. This checks whether the
# transformer is leaning on it: accuracy is broken out by text-length quartile. Roughly flat
# is what we want. A large drop in the shortest quartile means short adversarial posts —
# precisely what a social-media agent produces — are the blind spot.

# %%
test_eval["length_bucket"] = pd.qcut(
    test_eval["text"].str.len(), q=4,
    labels=["Q1 shortest", "Q2", "Q3", "Q4 longest"], duplicates="drop",
)
print(
    test_eval.groupby("length_bucket", observed=True)
    .agg(rows=("label", "size"), pos_rate=("label", "mean"),
         accuracy=("correct", "mean"), mean_score=("score", "mean"))
    .round(3).to_string()
)

# %% [markdown]
# ## 8 · Cross-generator generalisation — the number that matters
#
# Everything above is in-distribution: the test split contains the same generators as the
# training split. **A 2026 adversarial agent will not use a generator that was in your
# training set.** So an in-distribution F1 is a vanity metric, and this section is the real
# evaluation.
#
# `text_model.holdout_generators` in the config names generators removed from training
# **entirely**. We retrain without them and score only on them. The drop between §7 and here
# is an estimate of what happens on first contact with a new model.
#
# This retrains from scratch, so it doubles the notebook's runtime. Skip it with
# `AEGIS_SKIP_HOLDOUT=1` while iterating — but it belongs in the report.

# %%
# The primary model was trained without these generators (cell 4), so this is
# a true untouched-generator evaluation rather than a second ad-hoc retrain.
holdout_ds = TextDataset(holdout_df, tokenizer, MAX_LEN)
holdout_scores = predict_proba(holdout_ds)
holdout_report = amx.evaluate(
    holdout_df["label"].to_numpy(),
    holdout_scores,
    threshold=best_threshold,
    beta=FBETA,
)

print(f"held-out generators : {HOLDOUT}")
print(f"in-distribution     : {test_report}")
print(f"unseen generators   : {holdout_report}")
print(f"F1 drop             : {test_report.f1 - holdout_report.f1:+.4f}")
print(f"recall drop         : {test_report.recall - holdout_report.recall:+.4f}")
print(
    "\nThis untouched-generator report—not the in-domain score—is the primary "
    "generalisation metric for the text branch."
)

# %% [markdown]
# ### Per-generator recall
#
# One row per generator, worst first. Two cuts to look for: `*/humanized` and
# `*/paraphrased` from `llm_tweet` (explicit evasion attempts — recall should be visibly
# lower) and `adversarial_rewrite` from WildJailbreak.

# %%
_machine = test_eval[test_eval["label"] == 1]
_per_gen = (
    _machine.groupby("generator")
    .agg(rows=("label", "size"), recall=("correct", "mean"), mean_score=("score", "mean"))
    .sort_values("recall")
    .round(3)
)
_per_gen["low_support"] = _per_gen["rows"] < 20
print(_per_gen.to_string())

_evasion = _per_gen[_per_gen.index.str.contains("humaniz|paraphras", case=False, regex=True)]
if len(_evasion):
    print(f"\nevasion variants — mean recall {_evasion['recall'].mean():.3f} "
          f"vs {_per_gen['recall'].mean():.3f} overall")

viz.plot_per_group_bars(
    _per_gen.reset_index().rename(columns={"generator": "group", "rows": "support"}),
    metric="recall", save_as=settings.paths.figures / "02_per_generator_recall.png",
)

# %% [markdown]
# ## 9 · Error analysis
#
# The highest-confidence mistakes. These are worth reading rather than skimming — they are
# where the corpus's own labelling assumptions show up.

# %%
_errors = test_eval[~test_eval["correct"]].copy()
_errors["confidence"] = np.abs(_errors["score"] - best_threshold)

print(f"{len(_errors)} errors of {len(test_eval)} ({100 * len(_errors) / len(test_eval):.1f}%)\n")

for _title, _subset in (
    ("FALSE POSITIVES — human text flagged adversarial", _errors[_errors["label"] == 0]),
    ("FALSE NEGATIVES — adversarial text missed", _errors[_errors["label"] == 1]),
):
    print(f"\n{'=' * 78}\n{_title}  ({len(_subset)})\n{'=' * 78}")
    for _, _row in _subset.nlargest(3, "confidence").iterrows():
        print(f"\n[{_row['source_dataset']} / {_row['threat_class']} / {_row['generator']}] "
              f"score={_row['score']:.3f}")
        print(f"  {_row['text'][:260]}...")

# %% [markdown]
# ## 10 · Persist for the fusion notebook
#
# Notebook 04 needs the model's score for every row, plus the frozen threshold. Writing the
# scores rather than re-running inference keeps notebook 04 fast and makes the fusion
# reproducible without a GPU.

# %%
for _name, _frame, _scores in (
    ("validation", val_df, val_scores),
    ("test", test_df, test_scores),
    ("generator_holdout", holdout_df, holdout_scores),
):
    _out = _frame.loc[:, ["uid", "label", "threat_class", "generator",
                          "source_dataset", "era", "group_id"]].copy()
    _out["text_score"] = _scores
    _out["text_pred"] = (_scores >= best_threshold).astype(int)
    iou.save_frame(_out, settings.paths.processed / f"text_scores_{_name}.parquet")
    print(f"  text_scores_{_name}.parquet  {len(_out):,} rows")

iou.save_json(
    {
        "checkpoint": CHECKPOINT,
        "production_checkpoint": PRODUCTION_CHECKPOINT,
        "artifact_mode": "finetuned_deberta_v3",
        "max_length": MAX_LEN,
        "epochs": EPOCHS,
        "smoke_test": False,
        "optimizer": "adamw_torch",
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "scheduler": "linear_with_warmup",
        "warmup_ratio": float(args.warmup_ratio),
        "threshold": float(best_threshold),
        "fbeta": FBETA,
        "class_weights": {str(k): float(v) for k, v in class_weights.items()},
        "test": test_report.to_dict(),
        "validation": val_report.to_dict(),
        "baselines": {k: v.to_dict() for k, v in baselines.items()},
        "cross_generator_holdout": {
            "generators": HOLDOUT,
            "report": holdout_report.to_dict(),
        },
        "per_generator_recall": _per_gen["recall"].to_dict(),
    },
    settings.paths.text_model / "text_metrics.json",
)
print(f"\nmodel + metrics -> {settings.paths.text_model}")

# %% [markdown]
# ## Summary
#
# **Artefacts:** fine-tuned DeBERTa-v3 in `models/text_model/`, per-row scores in
# `data/processed/text_scores_{val,test}.parquet`, metrics in `text_metrics.json`,
# figures in `reports/figures/02_*.png`.
#
# **Read the results in this order:** the tuned threshold (not 0.5) → the gap between
# DeBERTa and the TF-IDF baseline (how much the transformer is actually worth) → the
# length-only baseline (how much is confound) → **§8's cross-generator drop**, which is the
# number to quote.
#
# **The limitation to state plainly:** this branch classifies *text*. An agent that posts
# ordinary, unremarkable sentences — which a 2026 agent trivially can — defeats it. That is
# not a fixable weakness of this model; it is why the project has a second branch.
#
# → **`03_graph_coordination_model.ipynb`**
