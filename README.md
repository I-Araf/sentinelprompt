# SentinelPrompt

Prompt injection detection robustness — CSE 4891 Data Mining project.

The project collects existing English prompt-injection datasets and adds a new
**Bangla–English (Bn-En) code-mixed** benchmark, drafted with an LLM and partly
verified by hand, so that detectors can be evaluated on low-resource and
code-mixed inputs rather than English only.

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
│       ├── SentinelPrompt-BnEn_draft_v2.csv   # 900 rows, LLM draft (read-only)
│       └── SentinelPrompt-BnEn_v3.csv         # 900 rows, labels after human verification
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
│   ├── adversarial_training.py     # M8: augmented training set
│   ├── error_analysis.py           # error breakdowns, pairwise McNemar
│   ├── explain.py                  # SHAP, LIME, attention, shortcut audit
│   ├── explain_classical.py        # LogReg / NB coefficient inspection
│   ├── leakage_ablation.py         # E6
│   ├── annotation_agreement.py     # Cohen's kappa between annotators
│   └── d4_verify.py                # D4 v3 from the human annotation, E4 re-scored
├── notebooks/                      # executed, with outputs saved
│   ├── 00_overview.ipynb           # every model side by side
│   ├── 01_EDA.ipynb
│   ├── 03_baselines.ipynb
│   ├── 04_transformers.ipynb
│   ├── 05_robustness_analysis.ipynb
│   ├── 06_adversarial_training.ipynb  # M8
│   ├── 07_error_analysis.ipynb
│   └── 08_interpretability.ipynb
├── results/
│   ├── figures/                    # PNG, 300 dpi
│   └── tables/                     # CSV (predictions/ holds per-row outputs)
├── models/                         # weights; git-ignored, re-created by the notebooks
├── annotation/                     # D4 verification kit
│   ├── GUIDELINE.md                # labelling rules (role-play rule in §5)
│   ├── make_sample.py              # draws the 150-row sample
│   ├── pilot/                      # 50-row pilot, filled by all three annotators
│   ├── sample/                     # 150-row sample, filled by all three annotators
│   └── round1/                     # 600-row worksheets, blank (not used)
├── requirements.txt
└── .gitignore
```

## Data sources

| Code | Source | Where | Rows (interim) |
|------|--------|-------|----------------|
| D1 | `deepset/prompt-injections` | Hugging Face | 662 |
| D2 | `Lakera/gandalf_ignore_instructions` | Hugging Face | 1,000 |
| D3 | PromptBench adversarial prompts | `adv_prompts/*.md` | 11,024 |
| D4 | SentinelPrompt-BnEn (this project) | LLM draft, partly human-verified | 900 |

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
- **author** — all `llm_draft`

## Human verification of D4 (`SentinelPrompt-BnEn_v3.csv`)

The three team members annotated a 50-row pilot and a 150-row stratified sample
of the draft (`annotation/pilot/`, `annotation/sample/`). Each annotated row takes
the majority label of the three; a tie keeps the draft label. `verified_by` names
only the annotators in the majority.

- 199 rows human-verified, 1 tie, 700 rows still unverified draft
- one prompt (3 language versions) moves from injection to safe under the
  role-play rule; balance is now 447 injection / 453 safe
- label kappa on the sample: A–B 1.000, A–C 0.653, B–C 0.653
  (`results/tables/annotation_sample_agreement.csv`; caveats in
  `annotation/sample/README.md`)

Models are not retrained: D4 is test-only, so E4 is re-scored from the saved
predictions (`results/tables/e4_relabel_sensitivity.csv`). Draft and v3 labels
differ by at most 0.004 macro-F1. `data/processed/test_codemix.csv` and the
other metric tables keep the draft labels; quote E4 from the v3 rows.

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
`05_robustness_analysis`, `06_adversarial_training`, `07_error_analysis`, `08_interpretability`, and finally `00_overview`. Each one imports its logic from
`src/`, so a notebook and its script always give the same numbers.
`04_transformers` downloads three pre-trained checkpoints (~1.5 GB) on first run.

After the notebooks have written the prediction files:

```bash
python src/annotation_agreement.py annotation/pilot pilot    # pilot kappa
python src/annotation_agreement.py annotation/sample sample  # sample kappa
python src/d4_verify.py          # D4 v3 + E4 re-scored with the verified labels
```

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
- [x] M8 adversarial training (E8)
- [x] Error analysis and pairwise significance (Bonferroni, odds ratio)
- [x] Interpretability (SHAP, LIME, attention) and shortcut audit
- [x] Coefficient inspection for Logistic Regression / Naive Bayes (classical track)
- [ ] Human reading of the error inspection sheet (`results/tables/error_inspection_sheet.csv`)
- [x] Human verification of a 200-row sample of the Bn-En draft (v3; 700 rows unverified)
- [ ] Attack-type annotation of the D1/D2 injections
