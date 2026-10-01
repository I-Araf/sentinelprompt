# SentinelPrompt

Prompt injection detection robustness — CSE 4891 Data Mining project.

The project collects existing English prompt-injection datasets and adds a new
hand-designed **Bangla–English (Bn-En) code-mixed** benchmark, so that detectors
can be evaluated on low-resource and code-mixed inputs rather than English only.

## Project structure

```
sentinelprompt/
├── data/
│   ├── raw/                        # untouched source dumps
│   │   ├── deepset/                # deepset/prompt-injections (HF)
│   │   ├── lakera/                 # Lakera/gandalf_ignore_instructions (HF)
│   │   └── promptbench/            # PromptBench adv_prompts/*.md
│   ├── interim/                    # normalised CSVs, one per source
│   │   ├── deepset_prompts.csv     #    662 rows
│   │   ├── lakera_prompts.csv      #  1,000 rows
│   │   └── promptbench_prompts.csv # 11,024 rows
│   └── bnen/                       # the new Bangla–English benchmark
│       ├── SentinelPrompt-BnEn_template.csv   #  15 rows (schema example)
│       ├── SentinelPrompt-BnEn_draft_v1.csv   # 450 rows
│       └── SentinelPrompt-BnEn_draft_v2.csv   # 900 rows  <- main dataset
│   └── processed/                  # unified data, splits, adversarial sets, features
├── src/                            # all logic lives here; notebooks import it
│   ├── download_hf.py              # D1 + D2 download
│   ├── parse_promptbench.py        # D3 parse
│   ├── integrate.py                # schema unification + cleaning
│   ├── dedup_group.py              # MinHash + LSH near-duplicates -> group_id
│   ├── split.py                    # grouped, leakage-free split
│   ├── guards.py                   # protocol invariants (asserts)
│   ├── adversarial.py              # E2 paraphrase + E3 obfuscation sets
│   ├── eda.py                      # exploratory analysis
│   ├── features_tfidf.py           # TF-IDF features
│   ├── baselines.py                # LogReg, Naive Bayes, Linear SVM
│   ├── train_transformer.py        # DistilBERT, RoBERTa, ProtectAI DeBERTa
│   ├── evaluate.py                 # one metric/figure implementation for all models
│   ├── robustness.py               # RDR, ASR, CLD, McNemar, bootstrap CI
│   └── leakage_ablation.py         # E6
├── notebooks/                      # executed, with outputs saved
│   ├── 00_overview.ipynb           # every model side by side
│   ├── 01_EDA.ipynb
│   ├── 03_baselines.ipynb
│   ├── 04_transformers.ipynb
│   └── 05_robustness_analysis.ipynb
├── results/
│   ├── figures/                    # PNG, 300 dpi
│   └── tables/                     # CSV (predictions/ holds per-row outputs)
├── models/                         # weights; git-ignored, re-created by the notebooks
├── annotation/                     # D4 verification kit
├── requirements.txt
└── .gitignore
```

## Data sources

| Code | Source | Where | Rows (interim) |
|------|--------|-------|----------------|
| D1 | `deepset/prompt-injections` | Hugging Face | 662 |
| D2 | `Lakera/gandalf_ignore_instructions` | Hugging Face | 1,000 |
| D3 | PromptBench adversarial prompts | `adv_prompts/*.md` | 11,024 |
| D4 | SentinelPrompt-BnEn (this project) | hand-authored | 900 |

D1/D2 interim schema: `id, text, label, source, language, orig_split`.
D3 adds PromptBench metadata (`pb_model`, `pb_shot`, `pb_task`, `pb_attack`,
`prompt_kind`, `pb_sem_lang`, `original_prompt`, and the accuracy columns).

## The Bn-En dataset (`SentinelPrompt-BnEn_draft_v2.csv`)

900 rows × 13 columns, built as **300 groups of 3 parallel variants** — the same
underlying prompt written in English, in Bangla, and in Bangla–English code-mix.
That design makes it possible to measure robustness as a within-group difference
rather than across unrelated samples.

Columns: `id, group_id, text, label, source, language, script, attack_type,
subtype, variant_type, author, verified_by, notes`

Current balance:

- **label** — 450 injection (`1`) / 450 benign (`0`)
- **language** — 300 `en` / 300 `bn` / 300 `bn-en`
- **script** — 477 `latin` / 300 `bengali` / 123 `mixed`
- **attack_type** — 450 `none`, plus 75 each of `direct_override`, `role_play`,
  `encoding`, `context_switch`, `payload_splitting`, `indirect`
- **subtype** — includes 150 `hard_negative` benign prompts (benign text that
  superficially resembles an attack), and encoding variants
  (`base64`, `hex`, `rot13`, `reverse`, `url`)
- **author** — all `llm_draft`; `verified_by` is still empty, so human
  verification is the outstanding step for this draft

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Built with Python 3.12.

## Reproducing the data

Run from the project root (both scripts use paths relative to it):

```bash
python src/download_hf.py        # writes data/raw/{deepset,lakera}/ and data/interim/
python src/parse_promptbench.py  # writes data/interim/promptbench_prompts.csv
```

`src/download_hf.py` downloads D1 and D2 from Hugging Face and normalises them
to the shared schema. `src/parse_promptbench.py` expects the PromptBench
`adv_prompts` markdown files to already be in `data/raw/promptbench/`; it parses
clean prompts, attacked prompts, and their accuracy deltas, and decodes
`b"..."` byte literals back to text.

## Running the pipeline

From the project root, in this order:

```bash
python src/integrate.py          # D1-D4 -> data/processed/unified_v1.csv
python src/dedup_group.py        # near-duplicate groups -> unified_v2_grouped.csv
python src/split.py              # grouped split -> train/val/test_*.csv
python src/adversarial.py        # E2 paraphrase + E3 obfuscation test sets
```

Then run the notebooks: `01_EDA`, `03_baselines`, `04_transformers`,
`05_robustness_analysis`, and finally `00_overview`. Each one imports its logic from
`src/`, so a notebook and its script always give the same numbers.
`04_transformers` downloads three pre-trained checkpoints (~1.5 GB) on first run.

## Test conditions

| ID | Test set | Rows |
|---|---|---|
| E1 | Test-Clean (held-out deepset) | 100 |
| E2 | Test-Paraphrase (back-translation + WordNet, similarity-gated) | 80 |
| E3 | Test-Obfuscated (7 techniques, 10/20/30% strength) | 1,900 |
| E4 | Bangla–English code-mix (D4) | 900 |
| E5 | Lakera, cross-dataset (all injection) | 1,000 |
| FP | PromptBench, false-positive test (all safe) | 4,150 |
| E5x | E5 + FP pooled, for ROC/PR-AUC | 5,150 |

Every model is evaluated by `src/evaluate.py` on every condition. If accuracy exceeds
97%, the evaluation prints a warning and runs a leakage check against the training set.

## Status

- [x] D1–D3 downloaded and normalised
- [x] Bn-En benchmark drafted (v2, 900 rows)
- [x] Integration, near-duplicate grouping, grouped split
- [x] Adversarial test sets E2 and E3
- [x] Classical baselines and transformers on E1–E5
- [x] Robustness metrics (RDR, ASR, CLD) and E6 leakage ablation
- [ ] Human verification of the Bn-En drafts (`verified_by` is empty)
- [ ] Attack-type annotation of the D1/D2 injections
