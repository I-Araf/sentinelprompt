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
│   │   ├── deepset_prompts.csv     #    690 rows
│   │   ├── lakera_prompts.csv      #  1,104 rows
│   │   └── promptbench_prompts.csv # 11,024 rows
│   └── bnen/                       # the new Bangla–English benchmark
│       ├── SentinelPrompt-BnEn_template.csv   #  15 rows (schema example)
│       ├── SentinelPrompt-BnEn_draft_v1.csv   # 450 rows
│       └── SentinelPrompt-BnEn_draft_v2.csv   # 900 rows  <- main dataset
├── src/
│   ├── download_hf.py              # D1 + D2: HF download -> raw/ + interim/
│   └── parse_promptbench.py        # D3: parse adv_prompts .md -> interim/
├── requirements.txt
└── .gitignore
```

## Data sources

| Code | Source | Where | Rows (interim) |
|------|--------|-------|----------------|
| D1 | `deepset/prompt-injections` | Hugging Face | 690 |
| D2 | `Lakera/gandalf_ignore_instructions` | Hugging Face | 1,104 |
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

## Status

- [x] D1–D3 downloaded and normalised to interim CSVs
- [x] Bn-En benchmark drafted (v2, 900 rows)
- [ ] Human verification of the Bn-En drafts (`verified_by` is empty)
- [ ] Merge all sources into a single train/test split
- [ ] Baseline detectors + robustness evaluation across the three language variants
