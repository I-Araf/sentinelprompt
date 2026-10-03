# D4 human-annotation sample (150 rows)

A stratified sample of SentinelPrompt-BnEn v2 checked by hand, after the pilot and
the guideline clarifications in `../GUIDELINE.md` §5.

- 50 prompt groups × 3 language versions (en, bn, bn-en) = 150 rows
- 75 attack / 75 safe by draft label; 36 of the safe rows are hard negatives
- pilot groups and guideline example groups left out
- every annotator received the same rows in the same order
- regenerate the blank sheets with `python annotation/make_sample.py`

## Sheets

| File | Annotator | Used for agreement |
|---|---|---|
| `sample_iham.csv` | Iham Araf (A) | yes |
| `sample_sourav.csv` | Sourav Biswas (C) | yes |
| `excluded/sample_saidul.csv` | Md. Saidul Islam Chowdhury (B) | **no** |

The sheets are stored exactly as received, including empty trailing columns and
rows left by Excel. Nothing in them has been corrected.

`excluded/sample_saidul.csv` is kept for the record only. It was not completed
independently of the other two sheets, so it is left out of every agreement figure
and of the final labels. Annotator B's independent annotation is the pilot
(`../pilot/pilot_saidul.csv`).

## Agreement (A vs C, n = 150)

| Field | Raw agreement | Cohen's κ |
|---|---|---|
| `ann_label` | 0.827 | **0.653** |
| `ann_attack_type` | 0.487 | 0.317 |

Reproduce with `python src/annotation_agreement.py annotation/sample sample`
(tables in `results/tables/annotation_sample_*.csv`).

## Caveats to report

- Annotator A had seen the draft labels before this round (pilot comparison and
  earlier checks), so A was not blind to the draft. A matches the draft on all 150
  rows.
- Only two independent annotators, so a single pairwise κ.
- Attack-type agreement is low; telling the techniques apart is harder than the
  attack/safe decision.
