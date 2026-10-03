"""Generate the 150-row human-annotation sample for D4 (SentinelPrompt-BnEn v2).

50 groups x 3 language versions, balanced by draft label (25 attack groups,
4 per attack type plus 1, and 25 safe groups, 12 of them hard negatives).
Pilot groups and the groups used as worked examples in GUIDELINE.md are left
out. Every annotator gets the same rows in the same shuffled order.
Reads data/bnen/ read-only; writes only into the folder given (default
annotation/sample/blank).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("data/bnen/SentinelPrompt-BnEn_draft_v2.csv")
PILOT = Path("annotation/pilot/pilot_iham.csv")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("annotation/sample/blank")
MEMBERS = {"sourav": "Sourav Biswas",
           "saidul": "Md. Saidul Islam Chowdhury",
           "iham": "Iham Araf"}
# groups quoted as examples in GUIDELINE.md, so an annotator has already seen a ruling
EXAMPLES = {"BNEN_G046", "BNEN_G026", "BNEN_G001", "BNEN_G153", "BNEN_G152", "BNEN_G154",
            "BNEN_G051", "BNEN_G076", "BNEN_G101", "BNEN_G126", "BNEN_G010", "BNEN_G045"}
FILL = ["ann_label", "ann_attack_type", "ann_fluency", "ann_flag", "ann_notes"]

rd = lambda p: pd.read_csv(p, encoding="utf-8-sig", dtype=str, keep_default_na=False)
draft = rd(SRC)
pilot_groups = set(rd(PILOT).group_id)

groups = draft.drop_duplicates("group_id")[["group_id", "label", "attack_type", "subtype"]]
groups = groups[~groups.group_id.isin(pilot_groups | EXAMPLES)]

rng = np.random.RandomState(20261003)
pick = []
for _, sub in groups[groups.label == "1"].groupby("attack_type"):
    pick += list(sub.sample(4, random_state=rng).group_id)
rest = groups[(groups.label == "1") & (~groups.group_id.isin(pick))]
pick += list(rest.sample(1, random_state=rng).group_id)
safe = groups[groups.label == "0"]
pick += list(safe[safe.subtype == "hard_negative"].sample(12, random_state=rng).group_id)
pick += list(safe[safe.subtype != "hard_negative"].sample(13, random_state=rng).group_id)

rows = draft[draft.group_id.isin(pick)].sample(frac=1, random_state=42).reset_index(drop=True)
OUT.mkdir(parents=True, exist_ok=True)
for key, name in MEMBERS.items():
    s = rows[["id", "group_id", "language", "script", "text"]].copy()
    s.insert(0, "row_no", range(1, len(s) + 1))
    s.insert(1, "annotator", name)
    for c in FILL:
        s[c] = ""
    s.to_csv(OUT / f"sample_{key}.csv", index=False, encoding="utf-8-sig")
print(f"{len(rows)} rows, {rows.group_id.nunique()} groups, "
      f"draft labels {rows.label.value_counts().to_dict()} -> {OUT}")
