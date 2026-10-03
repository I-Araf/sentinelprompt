# D4 human-annotation sample (150 rows)

A stratified sample of SentinelPrompt-BnEn v2 checked by hand, after the pilot and
the guideline clarifications in `../GUIDELINE.md` §5.

- 50 prompt groups × 3 language versions (en, bn, bn-en) = 150 rows
- 75 attack / 75 safe by draft label; 36 of the safe rows are hard negatives
- pilot groups and guideline example groups left out
- every annotator received the same rows in the same order
- regenerate the blank sheets with `python annotation/make_sample.py`

## Sheets

| File | Annotator |
|---|---|
| `sample_iham.csv` | Iham Araf (A) |
| `sample_saidul.csv` | Md. Saidul Islam Chowdhury (B) |
| `sample_sourav.csv` | Sourav Biswas (C) |

The sheets are stored exactly as received, including empty trailing columns and
rows left by Excel. Nothing in them has been corrected.

History of B's sheet: an earlier copy of B's sheet was filled by C without B's
involvement; it was discarded and is not in the repository. B then filled a blank
sheet afresh, without help, and that is the file here. B's answers share none of
C's departures from the guideline, and B's fluency judgements differ from A's on
12 rows.

## Agreement (n = 150 per pair)

| Pair | Label κ | Attack-type κ |
|---|---|---|
| A – B | 1.000 | 1.000 |
| A – C | 0.653 | 0.317 |
| B – C | 0.653 | 0.317 |

Raw label agreement: A–B 100%, A–C and B–C 82.7%. Same label across the three
language versions of a prompt: A 50/50, B 50/50, C 37/50.

Reproduce with `python src/annotation_agreement.py annotation/sample sample`
(tables in `results/tables/annotation_sample_*.csv`).

## Caveats to report

- Annotator A had seen the draft labels before this round (pilot comparison and
  earlier checks), so A was not blind to the draft. A and B both match the draft
  on all 150 rows.
- B had been sent C's earlier fill of B's sheet before filling B's own.
- Attack-type agreement involving C is low; C departs from the guideline on 56
  of the 75 attack rows.
