#!/usr/bin/env python3
"""M8 -- adversarially trained DistilBERT (methodology 3.10.9, E8).

Same model and hyperparameters as M5, but the training set is augmented with
paraphrased and obfuscated variants of its own rows. The question is whether
that lowers the robustness drop (RDR) seen in E2/E3, and what it does to the
conditions it was never shown (E4 Bangla, E5 Lakera, FP PromptBench).

Rules that keep the comparison honest:
  - variants come only from TRAINING groups (guards.assert_augmentation_from_train_only);
    validation stays originals-only, so threshold tuning is unchanged.
  - generators are the same as E2/E3 (adversarial.py), applied to train rows.
  - Base64 (O5) is deliberately held out of augmentation, so E3 contains one
    technique the model never trained on -- a check that robustness transfers
    rather than being memorised per technique.

Writes data/processed/train_augmented.csv and train_augmented_gate_log.csv.
"""
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adversarial as adv      # noqa: E402
import evaluate as ev          # noqa: E402
import guards                  # noqa: E402
from integrate import clean    # noqa: E402

DATA = Path("data/processed")
OUT = DATA / "train_augmented.csv"
SEED = 42
O_PER_ROW = 2                  # obfuscated variants per training row
HELD_OUT_TECHNIQUE = "O5_base64"


def obfuscation_variants(train):
    """Two random techniques per row, each at a random strength (10/20/30%)."""
    rng = random.Random(SEED)
    names = [n for n in adv.TECHNIQUES if n != HELD_OUT_TECHNIQUE]
    rows = []
    for _, r in train.iterrows():
        for name in rng.sample(names, O_PER_ROW):
            rate = rng.choice(adv.RATES)
            raw = adv.TECHNIQUES[name](r.text, rate, rng)
            rows.append({"id": f"{r.id}__aug_{name}_{int(rate * 100)}", "parent_id": r.id,
                         "text": clean(raw), "label": r.label, "group_id": r.group_id,
                         "source": r.source, "language": r.language,
                         "attack_type": r.attack_type, "variant_type": "obfuscation",
                         "technique": name, "perturbation_rate": rate})
    return pd.DataFrame(rows)


def paraphrase_variants(train):
    """Back-translation and WordNet with the same similarity gate as E2."""
    kept, log = adv.make_paraphrase(train)
    kept = kept.assign(id=kept.id.str.replace("__P", "__aug_P", regex=False))
    return kept.drop(columns=["text_raw", "similarity", "split"], errors="ignore"), log


def build(verbose=True):
    tr, va = ev.load_split("train"), ev.load_split("val")
    o = obfuscation_variants(tr)
    p, log = paraphrase_variants(tr)
    base = tr.assign(parent_id=tr.id, technique="", perturbation_rate=None, method="")
    aug = pd.concat([base, o, p], ignore_index=True)
    aug["split"] = "train_augmented"

    held = {n: ev.load_split(n) for n in ["val", "test_clean", "test_codemix",
                                           "test_crossdataset", "test_falsepos"]}
    guards.assert_augmentation_from_train_only(aug, *held.values(), names=list(held))
    guards.assert_no_lakera_in_train(aug, context="M8")
    guards.assert_no_bnen_in_train(aug, context="M8")
    assert HELD_OUT_TECHNIQUE not in set(aug.technique), "Base64 must stay out of training"

    aug.to_csv(OUT, index=False, encoding="utf-8")
    log.to_csv(DATA / "train_augmented_gate_log.csv", index=False, encoding="utf-8")
    if verbose:
        print(f"original train {len(tr)} -> augmented {len(aug)} rows "
              f"(+{len(o)} obfuscated, +{len(p)} paraphrased)")
        print("augmentation guards passed")
    return aug, log, va


def load():
    return pd.read_csv(OUT), ev.load_split("val")


if __name__ == "__main__":
    build()
