#!/usr/bin/env python3
"""Hard invariants for the experimental protocol (methodology 3.3.1, 3.7.4, 3.8).

These are deliberately `assert`-style: a violated invariant silently invalidates
a research claim, so it must stop the run rather than produce a plausible number.
Every split-producing or evaluation step calls into here.
"""


def assert_no_lakera_in_train(train_df, context="E5"):
    """D2 (Lakera) is the cross-dataset test set, so it must never be trained on.

    Project decision (2026-10-02): E5 Test-CrossDataset uses Lakera, because
    PromptBench is entirely label=0 and can only measure false positives.
    That makes Lakera a held-out source for any run that reports E5.
    """
    n = int((train_df["source"] == "lakera").sum())
    assert n == 0, (
        f"[{context}] {n} Lakera rows found in training data. Lakera is the "
        f"cross-dataset test source (E5) and must stay fully held out, "
        f"otherwise the E5 number is leakage, not generalization.")


def assert_no_bnen_in_train(df, context="E4"):
    """D4 is test-only (3.3.1). Training on it would destroy RQ3."""
    n = int((df["source"] == "bnen").sum())
    assert n == 0, (
        f"[{context}] {n} Bn-En rows found in training/validation data. D4 is "
        f"held out by design; training on it invalidates RQ3.")


def assert_no_group_overlap(*splits, names=None):
    """No group_id may appear in two splits (3.8.2) — the core leakage guard."""
    names = names or [f"split{i}" for i in range(len(splits))]
    sets = [set(s["group_id"]) for s in splits]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            shared = sets[i] & sets[j]
            assert not shared, (
                f"{len(shared)} group_id(s) shared between {names[i]} and "
                f"{names[j]} (e.g. {sorted(shared)[:3]}). A group holds a prompt "
                f"and all its near-duplicates/variants, so this is leakage.")


def assert_originals_only(df, context="train"):
    """Train and validation hold originals only; variants are test-side (3.8.3)."""
    bad = sorted(set(df["variant_type"]) - {"original"})
    assert not bad, (
        f"[{context}] non-original variant_type present: {bad}. Paraphrase/"
        f"obfuscation/codemix rows belong to the test conditions only.")
