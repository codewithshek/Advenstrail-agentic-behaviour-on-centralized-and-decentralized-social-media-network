# Full-Data Generalization and GPU Training Runbook

This runbook produces reportable artifacts for the current notebook code.
Execute from the `Complete-project` repository root.

## 1. Hardware and software

Recommended:

- Python 3.11;
- NVIDIA GPU with CUDA;
- at least 16 GB system RAM;
- sufficient disk for raw/interim/processed corpora and checkpoints;
- dataset credentials/licenses completed before execution.

Notebook 02 is intentionally production-only. It refuses smoke mode and warns
when CUDA is unavailable.

## 2. Create the environment

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel setuptools
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install torch-scatter torch-sparse `
  -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
Copy-Item .env.example .env
```

### Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
cp .env.example .env
```

Select a PyTorch/CUDA build compatible with the installed driver if CUDA 12.1
is not appropriate.

## 3. Preflight

### Verify CUDA

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('cuda build', torch.version.cuda); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Stop if `cuda available` is false for the production text run.

### Verify datasets

```powershell
.\.venv\Scripts\python.exe scripts\dataset_status.py
.\.venv\Scripts\python.exe scripts\audit_graph_corpora.py
```

Review [DATASETS_AND_PROVENANCE.md](DATASETS_AND_PROVENANCE.md). In particular:

- TweepFake must not be claimed while only dehydrated metadata exists;
- TwiBot-24 must not be claimed while the governed release is absent;
- enabled text sources must resolve to real records;
- synthetic fallback is forbidden in reportable training.

### Verify source and tests

```powershell
.\.venv\Scripts\python.exe -m pytest backend\tests ml\tests -q
.\.venv\Scripts\python.exe -m compileall -q backend ml\src ml\notebooks\_src scripts
.\.venv\Scripts\python.exe scripts\build_notebooks.py
```

## 4. Configure the run

PowerShell:

```powershell
$env:AEGIS_SMOKE_TEST = "0"
$env:AEGIS_DEVICE = "cuda"
$env:AEGIS_SEED = "42"
$env:AEGIS_SYNTH_BACKEND = "offline"
```

Bash:

```bash
export AEGIS_SMOKE_TEST=0
export AEGIS_DEVICE=cuda
export AEGIS_SEED=42
export AEGIS_SYNTH_BACKEND=offline
```

Use `offline` for deterministic campaign content. OpenAI/Anthropic generation
is optional and introduces provider/network variability.

## 5. Execute notebooks in order

The dependency graph is:

```text
01 ingestion/campaign bank
    ↓
02 text model and untouched-generator scores
    ↓
03 GraphSAGE and campaign graph scores
    ↓
04 nested LOCO fusion
```

### PowerShell

```powershell
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=-1 `
  ml\notebooks\01_data_ingestion_and_synthetic_gen.ipynb

.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=-1 `
  ml\notebooks\02_text_classification_model.ipynb

.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=-1 `
  ml\notebooks\03_graph_coordination_model.ipynb

.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=-1 `
  ml\notebooks\04_hybrid_fusion_model.ipynb
```

### Bash

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  ml/notebooks/01_data_ingestion_and_synthetic_gen.ipynb
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  ml/notebooks/02_text_classification_model.ipynb
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  ml/notebooks/03_graph_coordination_model.ipynb
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  ml/notebooks/04_hybrid_fusion_model.ipynb
```

Do not start notebook 03 if notebook 01 did not produce campaign-bank parquet
files. Do not start notebook 04 if notebooks 02 and 03 did not produce text and
graph scores.

## 6. Expected artifacts

### Notebook 01

```text
data/manifest.json
data/processed/text_corpus_full.parquet
data/processed/text_train.parquet
data/processed/text_validation.parquet
data/processed/text_test.parquet
data/processed/graph_nodes.parquet
data/processed/graph_edges.parquet
data/processed/graph_posts.parquet
data/processed/campaign_bank_nodes.parquet
data/processed/campaign_bank_edges.parquet
data/processed/campaign_bank_posts.parquet
data/processed/campaign_bank_text.parquet
data/processed/ingestion_summary.json
data/synthetic/campaign_bank_manifest.json
```

### Notebook 02

```text
models/text_model/config.json
models/text_model/tokenizer*
models/text_model/model.safetensors or pytorch_model.bin
models/text_model/text_metrics.json
data/processed/text_scores_validation.parquet
data/processed/text_scores_test.parquet
data/processed/text_scores_generator_holdout.parquet
```

### Notebook 03

```text
models/graph_model/swarm_gnn.pt
models/graph_model/feature_scaler.joblib
models/graph_model/rf_features.joblib
models/graph_model/graph_metrics.json
data/processed/graph_scores_campaigns.parquet
```

### Notebook 04

```text
models/fusion_model/fusion_model.joblib
models/fusion_model/fusion_metrics.json
data/processed/text_scores_campaign_posts.parquet
data/processed/fusion_scores_campaigns.parquet
```

## 7. Mandatory provenance checks

Inspect:

```powershell
Get-Content data\processed\ingestion_summary.json
Get-Content models\text_model\text_metrics.json
Get-Content models\graph_model\graph_metrics.json
Get-Content models\fusion_model\fusion_metrics.json
```

Require:

- `smoke_test` is false;
- text artifact mode is `finetuned_deberta_v3`;
- text checkpoint is the configured production checkpoint;
- no active source has `SYNTHETIC_FALLBACK`;
- graph protocol is `strict_inductive_campaign_holdout`;
- graph feature transform is `transfer_v1`;
- fusion protocol is `nested_leave_one_campaign_out`;
- every expected campaign ID appears exactly once as an outer fusion holdout.

## 8. Acceptance gates

### Data

1. No fallback rows enter training.
2. Dataset versions and provenance are recorded.
3. Group/campaign splits are disjoint.
4. Class distributions and row counts are plausible.

### Text

1. Full DeBERTa-v3 artifact loaded.
2. Class weights derive from the training split only.
3. Threshold derives from validation only.
4. Unseen-generator precision, recall, F1, ROC-AUC, and PR-AUC are present.
5. `90%+` accuracy is reported only if the untouched evaluation actually
   reaches it.

### Graph

1. Input profile shortcuts are absent.
2. Scaler derives from training campaigns only.
3. No validation/test edge participates in training.
4. Worst held-out campaign recall is at least `0.70`.
5. Macro held-out campaign F1 is at least `0.75`.
6. Cresci external performance cannot override a failed campaign gate.

### Fusion

1. Outer predictions cover every campaign once.
2. Threshold/model/calibration choices exclude the outer campaign.
3. Macro and worst-campaign metrics are reported.
4. Stacking is selected only when it beats simple candidates under the
   configured generalization rule.

These values are project gates, not promises. A failed gate is a valid
experimental result and must not be hidden by an in-domain accuracy number.

## 9. Archive the run

Executed notebooks contain environment-dependent output. Archive them
separately from clean source notebooks:

```powershell
Copy-Item ml\notebooks\01_data_ingestion_and_synthetic_gen.ipynb `
  artifacts\executed\01_data_ingestion_and_synthetic_gen.executed.ipynb
Copy-Item ml\notebooks\02_text_classification_model.ipynb `
  artifacts\executed\02_text_classification_model.executed.ipynb
Copy-Item ml\notebooks\03_graph_coordination_model.ipynb `
  artifacts\executed\03_graph_coordination_model.executed.ipynb
Copy-Item ml\notebooks\04_hybrid_fusion_model.ipynb `
  artifacts\executed\04_hybrid_fusion_model.executed.ipynb
```

Also record:

```powershell
python --version
pip freeze | Out-File artifacts\executed\requirements-lock.txt
nvidia-smi | Out-File artifacts\executed\gpu.txt
```

Where licensing permits, record dataset and artifact checksums.

## 10. Post-training API smoke test

Start the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app `
  --host 127.0.0.1 --port 8000
```

Then:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health

$body = @{
  handle = "validation_demo"
  peers = 5
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/analyze_account `
  -ContentType application/json `
  -Body $body
```

Check response fields:

- text `artifact_reliable` is true;
- graph `model_trusted` is consistent with transfer AUC;
- model names are not fallback names;
- feature-column mismatch does not occur;
- warnings are absent or understood.

## 11. Failure recovery

### Notebook 01: fallback provenance

Acquire/enable the real dataset or remove it from the configured training set.
Do not change the provenance gate.

### Notebook 02: no holdout examples

Inspect `generator` values in processed data and update
`holdout_generators` to valid, scientifically justified groups. Do not turn the
holdout into a random row split.

### CUDA out of memory

Reduce `batch_size`, increase `gradient_accumulation_steps`, or reduce
`max_length`. Record changes with the run.

### Graph feature mismatch

Delete/rebuild graph artifacts together. Never combine a scaler from one
feature contract with a checkpoint from another.

### Fusion calibration fails

Verify every campaign contains both classes. Small support should use sigmoid,
not isotonic calibration.

### High in-domain accuracy but poor worst-campaign recall

Treat the graph model as failed. Add independent campaigns, inspect feature
shortcuts, and retune regularization. Do not lower the transfer trust gate to
make the API use it.

## 12. Publication checklist

- [ ] Run configuration and seed archived.
- [ ] Dataset provenance and licenses documented.
- [ ] No fallback/smoke artifact represented as production.
- [ ] Baselines included.
- [ ] Text unseen-generator report included.
- [ ] Graph per-campaign and worst-campaign report included.
- [ ] Fusion nested LOCO report included.
- [ ] Calibration and threshold method disclosed.
- [ ] Confidence intervals or multi-seed variation included.
- [ ] Limitations and simulation status stated.
- [ ] No private raw account content exposed.
