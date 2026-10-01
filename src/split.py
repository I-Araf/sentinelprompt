#!/usr/bin/env python3
"""Leakage-free partitioning + the leakage ablation (methodology 3.8).

Grouped, paraphrase-aware, stratified split. Also runs the deliberate S-Random
vs S-Grouped comparison (3.8.4) whose difference is Leakage Inflation (LI) --
one of the paper's headline numbers.

Held-out sources never enter train/val:
  - bnen (D4)   -> Test-CodeMix (E4), by design (3.3.1)
  - lakera (D2) -> Test-CrossDataset (E5), project decision 2026-10-02
  - promptbench -> false-positive-only test (all label=0)

Reads data/processed/unified_v2_grouped.csv; writes data/processed/splits_v1.csv.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guards  # noqa: E402

IN = Path("data/processed/unified_v2_grouped.csv")
OUT = Path("data/processed/splits_v1.csv")

SEED = 42
VAL_FRAC = 0.15 / 0.85      # 15% of the whole, taken out of the 85% remainder
TEST_FRAC = 0.15
HELD_OUT = {"bnen": "test_codemix", "lakera": "test_crossdataset",
            "promptbench": "test_falsepos"}


def grouped_split(df):
    """Train/val/test on the modelling pool, never splitting a group (3.8.2)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=SEED)
    rest_i, test_i = next(gss.split(df, df.label, groups=df.group_id))
    rest, test = df.iloc[rest_i], df.iloc[test_i]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=SEED)
    tr_i, val_i = next(gss2.split(rest, rest.label, groups=rest.group_id))
    return rest.iloc[tr_i], rest.iloc[val_i], test


def random_split(df):
    """The naive split prior work used -- rows shuffled with no group awareness."""
    shuf = df.sample(frac=1, random_state=SEED)
    n_test = int(len(shuf) * TEST_FRAC)
    return shuf.iloc[n_test:], shuf.iloc[:n_test]


def leaked_groups(train, test):
    return set(train.group_id) & set(test.group_id)


# one file per split, so notebooks and other scripts never re-derive a split
SPLIT_COLS = ["id", "text", "label", "group_id", "source", "language",
              "attack_type", "variant_type", "split"]


def save_split_files(out):
    for name, part in out.groupby("split"):
        part[SPLIT_COLS].to_csv(OUT.parent / f"{name}.csv", index=False,
                                encoding="utf-8")
    print("per-split files:", ", ".join(f"{n}.csv" for n in sorted(out.split.unique())))


def main():
    df = pd.read_csv(IN)

    held = df[df.source.isin(HELD_OUT)].copy()
    held["split"] = held.source.map(HELD_OUT)
    pool = df[~df.source.isin(HELD_OUT)].copy()      # deepset only, for now

    print("===== POOL =====")
    print(f"modelling pool : {len(pool)} rows "
          f"({pool.label.value_counts().sort_index().to_dict()}), "
          f"{pool.group_id.nunique()} groups")
    for src, name in HELD_OUT.items():
        n = int((held.source == src).sum())
        print(f"held out {src:12s} -> {name:18s} {n} rows")

    # ---- S-Grouped (ours) ----
    tr, val, te = grouped_split(pool)
    tr, val, te = tr.copy(), val.copy(), te.copy()
    tr["split"], val["split"], te["split"] = "train", "val", "test_clean"

    guards.assert_no_group_overlap(tr, val, te, names=["train", "val", "test_clean"])
    guards.assert_no_lakera_in_train(tr)
    guards.assert_no_bnen_in_train(tr)
    guards.assert_no_bnen_in_train(val, context="E4/val")
    guards.assert_originals_only(tr)
    guards.assert_originals_only(val, context="val")
    print("\nall split invariants passed (guards.py)")

    out = pd.concat([tr, val, te, held], ignore_index=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    save_split_files(out)

    print("\n===== S-GROUPED (ours) =====")
    for name, part in [("train", tr), ("val", val), ("test_clean", te)]:
        pos = part.label.mean() if len(part) else 0
        print(f"  {name:11s} {len(part):5d} rows  {part.group_id.nunique():5d} groups"
              f"  injection {pos*100:5.1f}%")
    print(f"  leaked groups train<->test : {len(leaked_groups(tr, te))}")

    # ---- S-Random (prior-work style), for the ablation ----
    rtr, rte = random_split(pool)
    leak = leaked_groups(rtr, rte)
    rows_affected = int(rte.group_id.isin(leak).sum())
    print("\n===== S-RANDOM (prior-work style, for E6) =====")
    print(f"  train {len(rtr)} / test {len(rte)}")
    print(f"  leaked groups train<->test : {len(leak)}")
    print(f"  test rows whose group is also in train : {rows_affected}"
          f"  ({rows_affected/len(rte)*100:.1f}% of test)")
    print("\n  -> S-Random lets the model see near-duplicates of its own test set.")
    print("     Accuracy(S-Random) - Accuracy(S-Grouped) = Leakage Inflation (LI),")
    print("     measured once the baselines run (E6).")

    # ---- diagnostic: how much leakage is even measurable in this pool? ----
    # Holding Lakera out for E5 shrank the pool to deepset alone, and deepset has
    # little internal duplication -- so E6's ablation has almost nothing to show.
    # Reported here (not saved) so the design tension is visible, not buried.
    ext = df[df.source.isin(["deepset", "lakera"])]
    ertr, erte = random_split(ext)
    eleak = leaked_groups(ertr, erte)
    erows = int(erte.group_id.isin(eleak).sum())
    print("\n===== E6 DIAGNOSTIC: measurable leakage by pool =====")
    print(f"  deepset only (current pool) : {len(pool):5d} rows, "
          f"{len(leak)} leaked groups, {rows_affected} affected test rows")
    print(f"  deepset + lakera           : {len(ext):5d} rows, "
          f"{len(eleak)} leaked groups, {erows} affected test rows")
    print("  -> E6 on the current pool is close to vacuous. See report.")

    # ---- CV folds for 3.8.5 ----
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    sizes = [len(t) for _, t in sgk.split(pool, pool.label, groups=pool.group_id)]
    print(f"\nStratifiedGroupKFold k=5 fold sizes: {sizes}")
    print(f"\nsaved -> {OUT}")
    print(out.split.value_counts().to_string())


if __name__ == "__main__":
    main()
