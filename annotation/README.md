# Annotation round — how to run it

Who checks what, the files, and the order of work. For the rules and definitions →
[GUIDELINE.md](GUIDELINE.md)

> **What was actually done:** the pilot (step 1) and a 150-row sample filled by all
> three annotators. The 600-row round-1 worksheets were not used and are blank. The
> results and their caveats are in [sample/README.md](sample/README.md).

## Files

```
annotation/
├── GUIDELINE.md              ← read this first
├── README.md                 ← this file
├── pilot/                    ← step 1: 50 rows, the same rows for all three
│   ├── pilot_sourav.csv
│   ├── pilot_saidul.csv
│   └── pilot_iham.csv
└── round1/                   ← step 3: 600 rows each
    ├── worksheet_sourav.csv
    ├── worksheet_saidul.csv
    └── worksheet_iham.csv
```

## Design of the allocation

The 300 groups are divided into 3 blocks, and each block is checked by **two people
separately**:

| Block | Groups | Checked by |
|---|---|---|
| 1 | 100 groups (300 rows) | Sourav + Saidul |
| 2 | 100 groups (300 rows) | Saidul + Iham |
| 3 | 100 groups (300 rows) | Iham + Sourav |

Result: each person has **600 rows**, every row is checked **exactly twice**, for
**1,800** judgements in total. With three different pairs there are **three separate
κ values**, so one person's habits cannot pull the whole result.

The three language versions of a group are kept in the same block (so that coverage
stays balanced), but inside a worksheet the rows are **shuffled**, so that they do
not fall next to each other.

## Order of work (methodology §3.6.3)

**Step 1 — Pilot (50 rows, started together)**
All three fill **the same 50 rows** of the `pilot/` files independently. The aim is
not to do all the data but to find out **whether the guideline has gaps**.

**Step 2 — Compare and fix the guideline**
Put the three pilot files side by side and discuss the disagreements. Where
disagreement repeats, the guideline is unclear → add an example to GUIDELINE.md to
make it clear. **This discussion happens only for the pilot, not for round 1.**

**Step 3 — Round 1 (600 rows each)**
Fill the `round1/` files with the revised guideline. This time **no discussion**.

**Step 4 — Compute κ and settle disagreements**
κ is computed for each pair (one for `ann_label`, another for `ann_attack_type`).
For rows where the two do not agree, **the third member** gives the final decision.

**Step 5 — Final file**
The agreed answers plus the adjudicator's decisions are combined into
`SentinelPrompt-BnEn_v3_verified.csv`, in which both `author` and `verified_by` are
filled.

> The original `data/bnen/` files are **never changed**. v3 is added as a separate
> file, so that the whole path from draft to final is kept.

## Rules for filling the files

- Excel / Google Sheets / LibreOffice — any of them works. The files are saved as
  **UTF-8 (with BOM)**, so the non-English text displays correctly.
- Fill only the columns that start with `ann_`. **Do not touch the other columns** —
  if `id` or `row_no` changes, the rows cannot be matched later.
- **Save as CSV** (not `.xlsx`).
- Keep the same file name when you finish.

## How to interpret κ (§3.6.3)

| κ | Meaning |
|---|---|
| < 0.20 | slight |
| 0.21–0.40 | fair |
| 0.41–0.60 | moderate |
| **0.61–0.80** | **substantial ← our target** |
| > 0.80 | almost perfect |

A low κ is not a failure. It means the guideline needs to be clearer, or the
categories really do overlap. That is also a finding worth writing in the paper.
