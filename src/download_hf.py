#!/usr/bin/env python3
"""D1 (deepset) + D2 (Lakera) download -> raw + interim CSV."""
from pathlib import Path

import pandas as pd
from datasets import load_dataset

SOURCES = {
    "deepset": {"hf": "deepset/prompt-injections", "prefix": "D1"},
    "lakera": {"hf": "Lakera/gandalf_ignore_instructions", "prefix": "D2"},
}
COLS = ["id", "text", "label", "source", "language", "orig_split"]


def pick_text_col(df):
    if "text" in df.columns:
        return "text"
    for c in df.columns:
        if df[c].dtype == object:
            return c
    raise ValueError(f"No text column in {list(df.columns)}")


for name, cfg in SOURCES.items():
    print(f"\n===== {name} ({cfg['hf']}) =====")
    ds = load_dataset(cfg["hf"])

    raw_dir = Path("data/raw") / name
    raw_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    for split, d in ds.items():
        df = d.to_pandas()
        df.to_csv(raw_dir / f"{split}.csv", index=False)
        print(f"  {split:12s}: {len(df):5d} rows, columns = {list(df.columns)}")
        df["orig_split"] = split
        parts.append(df)

    df = pd.concat(parts, ignore_index=True)
    text_col = pick_text_col(df)

    out = pd.DataFrame({"text": df[text_col].astype(str)})
    if name == "lakera":
        out["label"] = 1          # Lakera-র সব row injection
        out["language"] = "en"    # dataset card অনুযায়ী English
    else:
        out["label"] = df["label"].astype(int)
        out["language"] = ""      # English + কিছু German, পরে detect করা হবে
    out["source"] = name
    out["orig_split"] = df["orig_split"]
    out["id"] = [f"{cfg['prefix']}_{i:05d}" for i in range(1, len(out) + 1)]
    out = out[COLS]

    out_path = Path("data/interim") / f"{name}_prompts.csv"
    out.to_csv(out_path, index=False)

    print(f"  Total rows   : {len(out)}")
    print(f"  Unique texts : {out['text'].nunique()}")
    print(f"  Label counts : {out['label'].value_counts().to_dict()}")
    print(f"  Saved -> {out_path}")
