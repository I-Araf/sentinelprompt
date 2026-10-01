"""Generate blind double-annotation worksheets for D4 (SentinelPrompt-BnEn v2).
Reads data/bnen/ read-only; writes only into annotation/."""
import pandas as pd
from pathlib import Path

SRC = Path("data/bnen/SentinelPrompt-BnEn_draft_v2.csv")
OUT = Path("annotation")
SEED = 42
MEMBERS = {"sourav": "Sourav Biswas",
           "saidul": "Md. Saidul Islam Chowdhury",
           "iham":   "Iham Araf"}
# each block is annotated independently by exactly two members
PAIRS = [("sourav", "saidul"), ("saidul", "iham"), ("iham", "sourav")]

SHOW = ["id", "group_id", "language", "script", "text"]
FILL = ["ann_label", "ann_attack_type", "ann_fluency", "ann_flag", "ann_notes"]

df = pd.read_csv(SRC, encoding="utf-8-sig")
groups = sorted(df.group_id.unique())
assert len(groups) == 300 and len(df) == 900

# split 300 groups into 3 blocks of 100 -> all 3 variants of a group stay in one block
blocks = [groups[i::3] for i in range(3)]

def sheet(rows, who):
    s = rows[SHOW].copy()
    # shuffle so the en/bn/bn-en siblings are not adjacent -> judgements stay independent
    s = s.sample(frac=1, random_state=SEED).reset_index(drop=True)
    s.insert(0, "row_no", range(1, len(s) + 1))
    s.insert(1, "annotator", MEMBERS[who])
    for c in FILL:
        s[c] = ""
    return s

(OUT / "round1").mkdir(parents=True, exist_ok=True)
(OUT / "pilot").mkdir(parents=True, exist_ok=True)

assigned = {k: [] for k in MEMBERS}
for blk, (a, b) in zip(blocks, PAIRS):
    rows = df[df.group_id.isin(blk)]
    assigned[a].append(rows)
    assigned[b].append(rows)

print("ROUND 1 (blind, double-annotated)")
for who, parts in assigned.items():
    s = sheet(pd.concat(parts), who)
    p = OUT / "round1" / f"worksheet_{who}.csv"
    s.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"  {p}  {len(s)} rows")

# pilot: 50 rows, same for everyone, stratified by language
idx = []
for lang in ["en", "bn", "bn-en"]:
    idx += list(df[df.language == lang].sample(17, random_state=SEED).index)
pilot = df.loc[idx].sample(50, random_state=SEED)
print("PILOT (same 50 rows for all three)")
for who in MEMBERS:
    s = sheet(pilot, who)
    p = OUT / "pilot" / f"pilot_{who}.csv"
    s.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"  {p}  {len(s)} rows")

cov = pd.concat(sum(assigned.values(), [])).id.value_counts()
print(f"\ncoverage: every row annotated exactly {cov.min()}x (max {cov.max()}), "
      f"{cov.index.nunique()} unique rows, {sum(len(p) for v in assigned.values() for p in v)} total judgements")
