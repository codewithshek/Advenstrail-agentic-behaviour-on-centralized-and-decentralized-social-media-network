# Datasets, Provenance, and Data Governance

This document records what each configured dataset contributes, whether it is
currently usable, and how AEGIS-SN prevents missing data from becoming a false
research claim.

The authoritative machine-readable configuration is
[`ml/configs/default.yaml`](../ml/configs/default.yaml).

## Provenance policy

Every loaded/generated frame is registered in `data/manifest.json`.

| Provenance | Definition | Reportable? |
|---|---|---:|
| `REAL` | Parsed from an acquired source release. | Yes, subject to license and split validity. |
| `PARTIAL` | Real source, row-capped for smoke/development execution. | Development only. |
| `GENERATED` | Deliberate defensive simulation. | Only as simulated performance. |
| `SYNTHETIC_FALLBACK` | Schema stub substituting for unavailable data. | No. |

Notebook 01 allows fallback so ingestion code can be tested from a fresh clone.
Its final provenance gate prevents fallback rows from being passed off as
reportable training data. Notebook 02 repeats the gate.

## Resolution order

```text
datasets/<name>/ → Hugging Face → Kaggle → synthetic fallback
```

Disabled datasets do not enter resolution and are not silently replaced.

## Text corpora

### TweepFake

- **Purpose:** legacy human versus GPT-2/RNN/LSTM/Markov/CharRNN social posts.
- **Configured status:** disabled.
- **Reason:** the local Twitter-ToS-compliant copy is dehydrated and contains
  status/user metadata without a usable `text` column.
- **Replacement role:** HC3 plus early-generator M4 slices.
- **Enable only if:** a legally hydrated release with text is available.

Do not write that this project trained on TweepFake unless the manifest for the
specific run records real TweepFake text.

### HC3

- **Purpose:** paired human and ChatGPT answers across domains.
- **Configured status:** enabled.
- **Strength:** balanced paired contrasts.
- **Limitation:** captures the distinctive 2022 ChatGPT style and is not a
  frontier-agent benchmark.
- **Grouping:** related answer pairs share a leakage group.

### M4

- **Purpose:** multi-generator, multi-domain, multilingual machine-generated
  text detection.
- **Configured status:** enabled.
- **Key metadata:** generator/model and domain.
- **Importance:** supports generator-level holdout rather than shuffled
  example-level generalization claims.

Configured/observed generator families include Davinci, ChatGPT, Cohere,
Dolly, BLOOMZ, GPT-4, LLaMA, and filename-derived variants.

### WildJailbreak

- **Purpose:** vanilla/adversarial and benign/harmful prompts/completions.
- **Configured status:** enabled.
- **Scale control:** bounded, stratified reading to preserve hard negatives.
- **Importance:** prevents the classifier from treating every unusual prompt as
  malicious.

### WildGuard

- **Purpose:** harmfulness, refusal, and adversarial prompt/response labels.
- **Configured status:** disabled.
- **Reason:** gated release not available in the configured local drop zone.
- **Enable only after:** accepting the source license and obtaining access.

### Deepset prompt injections

- **Purpose:** direct prompt-injection classification.
- **Configured status:** enabled.
- **Strength:** includes multilingual examples.
- **Limitation:** small corpus; never use it as the only injection source.

### LLM-Tweet / Human-vs-LLM

- **Purpose:** modern generated text and evasion variants.
- **Configured status:** enabled.
- **Variants:** generated, paraphrased, translated, and humanized.
- **Grouping:** variants of the same base row share one group.
- **Importance:** partially replaces unavailable hydrated TweepFake text with a
  more contemporary source.

### Synthetic campaign text

- **Purpose:** align the text branch with generated graph campaigns.
- **Configured status:** generated.
- **Labels:** agent posts are positive; decoy posts are human/benign;
  injection-carrying posts retain an injection threat class.
- **Constraint:** report only as simulator performance.

## Graph corpora

### Cresci-2017

- **Purpose:** real genuine-account and social-spambot diagnostic.
- **Configured status:** enabled.
- **Edges:** reconstructed from recorded reply, retweet, and mention
  interactions because the available release lacks follow/friend files.
- **Use in current graph notebook:** external diagnostic after campaign-based
  model selection.
- **Caveat:** class/topology assortativity can make random node splits
  unrealistically easy.

### Cresci-2019

- **Purpose:** cashtag-piggybacking and retweet-ring behaviour.
- **Configured status:** disabled.
- **Reason:** not downloaded.

### TwiBot-22

- **Purpose:** large heterogeneous graph benchmark.
- **Configured status:** disabled.
- **Reason:** the local directory is the source repository, not the governed
  dataset release. Required node/edge/label files are absent.
- **Acquisition:** follow the project's data-use agreement process.

### TwiBot-24

- **Purpose:** target frontier benchmark with LLM-powered accounts and many
  relation types.
- **Configured status:** disabled.
- **Reason:** governed release not present.
- **Importance:** the strongest intended external test for the project.

The loader and platform-neutral relation contract support TwiBot-style files,
but support is not equivalent to having used the dataset.

### Additional configured graph sources

`default.yaml` contains other real/manual Bot Repository and social graph
sources for comparative auditing. Their enabled status, discovery patterns,
schema caveats, and read limits are documented inline in that file and should
be checked with `scripts/dataset_status.py` before use.

## Generated campaign bank

The campaign bank provides independent units for model development:

| Scenario | Goal | Seeds |
|---|---|---|
| `product_shill` | Coordinated commercial narrative amplification. | 1337, 2027, 3119 |
| `civic_disinfo` | Coordinated civic/infrastructure disinformation. | 1337, 2027, 3119 |

Each campaign contains:

- role-specific agent personas;
- organic decoy accounts;
- campaign phases;
- synchronized narrative events;
- directed interactions and reciprocity;
- cross-account content reuse;
- optional prompt-injection payloads;
- immutable `campaign_id`, scenario, and seed.

Campaign generation backends:

```text
offline    deterministic templates, no credential
openai     LangChain OpenAI integration
anthropic  LangChain Anthropic integration
```

Select using `AEGIS_SYNTH_BACKEND`. Live generation reduces byte-for-byte
reproducibility and may transmit prompts to an external provider.

## Data splits and leakage controls

### Text

- source-specific pairs/variants/templates receive a `group_id`;
- group-aware splitting prevents near-duplicate variants crossing partitions;
- configured generator families are isolated before any model/baseline fit;
- class weights use training labels only;
- decision threshold uses validation only.

### Graph

- user/post IDs are campaign-namespaced;
- edges are restricted to endpoints in the same campaign;
- campaigns—not individual nodes—receive train/validation/test roles;
- validation/test edges never enter training message passing;
- scaling uses training campaigns only.

### Fusion

- `campaign_id` is the `LeaveOneGroupOut` unit;
- inner out-of-group predictions select model and threshold;
- one outer campaign is scored only after choices are fixed.

## Drop-zone layout

Place manually acquired corpora under `datasets/` according to each local
README:

```text
datasets/
├── hc3/
├── m4/
├── tweepfake/
├── wildguard/
├── wildjailbreak/
├── cresci_2017/
├── cresci_2019/
├── twibot_22/
└── twibot_24/
```

Several configured paths also point to the existing sibling `../DataSets/`
collection. Run the readiness audit rather than assuming the small documented
drop-zone folders contain the corpora.

## Audit commands

```powershell
.\.venv\Scripts\python.exe scripts\dataset_status.py
.\.venv\Scripts\python.exe scripts\audit_graph_corpora.py
```

After notebook 01:

```powershell
Get-Content data\manifest.json
Get-Content data\processed\ingestion_summary.json
```

Verify:

- every expected source appears;
- provenance matches the intended claim;
- smoke mode is false for reportable runs;
- no fallback source enters training;
- row counts and class balance are plausible;
- campaign IDs are present and distinct.

## Licensing and redistribution

Dataset licenses and platform terms override project convenience.

- Do not redistribute governed TwiBot releases.
- Do not commit hydrated social content without permission.
- Do not publish platform credentials or API tokens.
- Cite every source dataset and comply with attribution requirements.
- Preserve source-level provenance through derived artifacts.
- Share aggregate metrics rather than raw account content when possible.

Raw corpora, generated data caches, model weights, and local DBs are excluded
by `.gitignore`.

## Research reporting checklist

For every result, state:

1. dataset names and exact versions;
2. provenance state;
3. enabled/disabled substitutions;
4. preprocessing and read limits;
5. split unit (`group_id` or `campaign_id`);
6. class distribution;
7. hyperparameters and seed;
8. threshold-selection data;
9. micro, macro, and worst-group metrics;
10. whether the corpus is real, partial, generated, or fallback;
11. confidence intervals or multi-seed variation where available;
12. known licensing and representativeness limits.

## Adding a dataset

1. Add a documented config block to `ml/configs/default.yaml`.
2. Define discovery patterns and column aliases.
3. Implement/reuse a loader in `dataset_loaders.py`.
4. Normalize into the shared contract.
5. Register provenance.
6. Add minimum-row and schema validation.
7. Add grouping fields that prevent leakage.
8. Add focused tests.
9. Audit class balance, duplicates, and graph connectivity.
10. Update this document and the root README.
