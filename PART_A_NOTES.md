# Part A — Data Foundation (Sourov)

Your part in one line: **"Where the data came from, what state it was in, and what cleaning it revealed."**

## Files and order

| Order | File | What it does |
|---|---|---|
| 1 | `src/download_hf.py` | Downloads D1 (deepset) and D2 (Lakera) from Hugging Face |
| 2 | `src/parse_promptbench.py` | Reads the D3 (PromptBench) `.md` files into a CSV |
| 3 | `src/integrate.py` | Brings all four sources into one 9-column schema and cleans them |
| 4 | `src/dedup_group.py` | Finds near-duplicates and assigns `group_id` |
| 5 | `src/eda.py` | EDA — the six steps of §3.4, each as its own function |
| — | `notebooks/01_EDA.ipynb` | Runs the steps of `eda.py` one by one, with explanations and outputs |

Open the notebook and every output is already there. To run it again, from the project root:
```bash
python src/integrate.py && python src/dedup_group.py && python src/eda.py
```
Figures go to `results/figures/eda_*.png` (300 dpi), tables to `results/tables/eda_*.csv`.

## What is new this round

- `eda.py` used to run everything in one go. It is now **six separate functions** (`structural_profile`,
  `class_balance`, `length_distribution`, `lexical`, `duplicate_audit`, `label_quality`) — one for each of
  the six steps in the methodology. The notebook runs each step separately, with an explanation in between.
- Outputs moved from `reports/eda/` to `results/`, where the whole project keeps its results. The old
  `reports/eda` folder has therefore been removed.
- All numbers are the same as before — only where they are written and how the code is split have changed.

## Numbers to remember for the viva

**1. Data size.** 13,586 rows, down to **6,712** after removing exact duplicates. That means 6,874 rows
(**51%**) were exact copies, mostly in PromptBench (11,024 → 4,150), because the same prompt appears again
and again across different model/shot files.

**2. Balance.** 4,999 Safe vs 1,713 Injection — **74.5% Safe**. Always answering "Safe" would already give
74.5% accuracy — which is why macro-F1, not accuracy, is the main metric (the accuracy paradox).

**3. Every source is different.**

| Source | Safe | Injection |
|---|---|---|
| deepset | 399 | 263 |
| Lakera | 0 | 1,000 |
| PromptBench | 4,150 | 0 |
| Bangla–English (D4) | 450 | 450 |

Lakera is attacks only and PromptBench is harmless only — that is why they are not used for training but
serve as separate tests (E5 and FP).

**4. Length.** 95% of prompts are within 32 words, 99% within 47 words. So `max_length = 128` tokens is
enough for the transformers — a longer limit only makes them slower, because the cost of attention grows
with the square of the length.

**5. Near-duplicates.** MinHash + LSH found **2,398 near-duplicate pairs** (Jaccard ≥ 0.80).
Excluding D4, **1,502 rows (25.8%) sit in 327 groups** of more than one row. The largest group has 20 rows.

> The duplicate table in the EDA notebook shows 627 groups / 35.8% — that number includes D4. Every D4
> prompt deliberately appears three times, once per language (300 groups × 3), so including them raises
> the count. If asked, explain this difference.

**6. Words that pull towards injection (chi-square).** `instructions`, `ignore`, `you`, `previous`, `all`,
`password`, `previous instructions` — the language of a classic override attack.

## Three decisions you will be asked to justify

**a) Why minimal cleaning?** Stopwords, punctuation, emoji, spelling — none of it was removed. If we
"correct" `1gn0r3` into `ignore`, we erase the attack ourselves. It is like sweeping a crime scene clean —
the evidence is destroyed. Only NFKC normalisation and collapsing extra whitespace were applied, and the
original form is kept in the `text_raw` column.

**b) Why MinHash and not cosine?** Both were run, but groups were assigned **with MinHash only**. Cosine
was also merging different questions — `"Is the settlement building unfair?"` and
`"...building in Spain unfair?"` (Jaccard only 0.64). Combining both would have put 36.9% of rows into
groups, with the largest group at 32 rows. **A group cannot be split across train and test**, so
over-merging destroys the diversity of the data.

**c) The language-detection bug.** deepset contains English and some German. The first attempt used a
hand-written list of German words that included `was` and `die` — which are also very common English
words — so hundreds of English prompts were labelled German. Then `langdetect` made mistakes on short
phrases ("Agricultural policy sustainability Europe" → Romanian!). The final fix: since deepset contains
only English and German, a row becomes `other` **only when German is detected**. The remaining error is ~1%.

## How LSH works (you need to be able to explain this)

Comparing every pair among 5,812 rows would take 16.8 million comparisons. LSH first builds a
**shortlist** (3,427 candidate pairs), and only those pairs get their real Jaccard computed. It is like
narrowing down the neighbourhood before searching a whole city. The shortlist is approximate, so the
check is mandatory — of 3,427 candidates, 2,398 survived.

**Why 5-character pieces (shingles) and not words?** Splitting by characters also catches spelling
mistakes and homoglyph attacks — for example `grammar` vs `grɑmmar` (the `ɑ` in the middle is a
different character), Jaccard 0.87.

## Where your work goes in the paper

- The **Data section** rests entirely on your work.
- The **51% duplication and the 2,398 pairs** are the evidence for the G2 gap.
- Later we measured whether these duplicates actually inflated accuracy (E6, Iham's part). The answer:
  **no** — the duplicates are almost all harmless PromptBench prompts, which the model gets 100% right
  anyway. The examiner may ask "what is the problem with duplicates?" — the answer: **there can be one, but
  not always; we measured it instead of assuming it.** The grouped split is still kept as a safeguard.
