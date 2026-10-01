#!/usr/bin/env python3
"""Schema unification + cleaning (methodology 3.3.3, 3.5.1-3.5.2, 3.5.5a).

D1-D4 -> one table in the unified 9-column schema.
Reads data/interim/ and data/bnen/ (read-only); writes data/processed/.
Near-duplicate clustering (3.5.5b) is a separate later step.
"""
import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0          # deterministic detection

INTERIM = Path("data/interim")
BNEN = Path("data/bnen/SentinelPrompt-BnEn_draft_v2.csv")
OUT_DIR = Path("data/processed")
OUT = OUT_DIR / "unified_v1.csv"

SCHEMA = ["id", "text", "label", "source", "language",
          "attack_type", "variant_type", "group_id", "split"]
EXTRA = ["text_raw", "has_zero_width", "pii_redacted", "orig_split"]

ZERO_WIDTH = ["​", "‌", "‍", "﻿", "⁠"]
RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
RE_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{8,}\d)(?!\w)")
# deepset is English + some German; flag German so it lands in `other`
def detect_lang(t):
    """deepset is documented as English + some German (3.3.1), nothing else.
    So we only move a row out of `en` on a positive German detection. Short
    keyword-like prompts ("Free trade agreement Europa-USA") get confidently
    misdetected as nl/ro/da by langdetect, and that rule discards those.
    """
    t = str(t).strip()
    if len(t) < 10:
        return "en"
    try:
        top = detect_langs(t)[0]
    except LangDetectException:
        return "en"
    return "other" if (top.lang == "de" and top.prob >= 0.50) else "en"


def clean(s):
    """NFKC normalize + collapse whitespace. Deliberately minimal (3.5.4)."""
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def redact(s):
    s, n1 = RE_EMAIL.subn("[EMAIL]", s)
    s, n2 = RE_PHONE.subn("[PHONE]", s)
    return s, n1 + n2


def load():
    frames = []

    # --- D1 deepset ---
    d = pd.read_csv(INTERIM / "deepset_prompts.csv")
    d["language"] = d.text.map(detect_lang)
    d["attack_type"] = d.label.map(lambda y: "none" if y == 0 else "unannotated")
    d["variant_type"] = "original"
    frames.append(d)

    # --- D2 lakera ---
    d = pd.read_csv(INTERIM / "lakera_prompts.csv")
    d["language"] = "en"
    d["attack_type"] = "unannotated"          # all label=1
    d["variant_type"] = "original"
    frames.append(d)

    # --- D3 promptbench ---
    d = pd.read_csv(INTERIM / "promptbench_prompts.csv")
    d["language"] = d.pb_sem_lang.fillna("").map(
        lambda v: "en" if v in ("", "english") else "other")
    d["attack_type"] = "none"                 # all label=0
    d["variant_type"] = "original"
    d["orig_split"] = ""
    frames.append(d)

    # --- D4 bnen (held out; keeps its own group_id) ---
    d = pd.read_csv(BNEN, encoding="utf-8-sig")
    d["variant_type"] = "codemix"
    d["orig_split"] = ""
    frames.append(d)

    return frames


def main():
    frames = load()
    df = pd.concat([f.reindex(columns=sorted(set(SCHEMA + EXTRA + ["group_id"])),
                              fill_value=None) if False else f
                    for f in frames], ignore_index=True)

    df["text_raw"] = df.text.astype(str)
    df["text"] = df.text_raw.map(clean)
    df["has_zero_width"] = df.text_raw.map(
        lambda s: any(z in s for z in ZERO_WIDTH))
    red = df.text.map(redact)
    df["text"] = red.map(lambda r: r[0])
    df["pii_redacted"] = red.map(lambda r: r[1])

    before = len(df)
    df = df[df.text.str.strip() != ""]
    empty = before - len(df)

    # --- exact duplicate handling (3.5.5a) ---
    df["_h"] = df.text.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    conflict = (df.groupby("_h").label.nunique() > 1)
    bad = set(conflict[conflict].index)
    df = df[~df._h.isin(bad)]                  # label conflict -> drop both
    dup = df.duplicated("_h").sum()
    df = df.drop_duplicates("_h", keep="first")

    # --- provisional group_id (near-dup clustering merges these later) ---
    df["group_id"] = df.apply(
        lambda r: r.group_id if r.source == "bnen" else f"G_{r._h[:12]}", axis=1)
    df["split"] = ""                           # assigned in 3.8

    df = df[SCHEMA + EXTRA]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")

    print("===== INTEGRATION REPORT =====")
    print(f"rows in  : {before}")
    print(f"  empty text dropped        : {empty}")
    print(f"  label-conflict texts drop : {len(bad)} distinct texts")
    print(f"  exact duplicates dropped  : {dup}")
    print(f"rows out : {len(df)}")
    print(f"\nby source:\n{df.source.value_counts().to_string()}")
    print(f"\nby label:\n{df.label.value_counts().sort_index().to_string()}")
    print(f"\nby language:\n{df.language.value_counts().to_string()}")
    print(f"\nattack_type:\n{df.attack_type.value_counts().to_string()}")
    print(f"\nvariant_type:\n{df.variant_type.value_counts().to_string()}")
    print(f"\ngroups: {df.group_id.nunique()}")
    print(f"zero-width present : {int(df.has_zero_width.sum())}")
    print(f"PII redactions     : {int(df.pii_redacted.sum())}")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
