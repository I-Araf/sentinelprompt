#!/usr/bin/env python3
"""E6 leakage ablation (methodology 3.8.4), with the controls it needs.

The naive E6 number, LI = Acc(S-Random) - Acc(S-Grouped), turned out to be ~0.
Before reporting that, we check it is not hiding an effect:

  1. LI on three pools of increasing duplication, across seeds.
  2. Inside one random split: leaked vs non-leaked test rows.
  3. The same, restricted to one source and label -- the control. Step 2
     alone shows a large, significant gap that is purely compositional:
     leaked rows are almost all easy PromptBench prompts.
  4. A learning curve, to rule out a ceiling effect from a large train set.

Result on this corpus: H4 is not supported. Duplication exists (51%) but does
not inflate accuracy, because the duplicates sit in a class the model separates
perfectly from ~100 examples. Re-run on the paraphrase set (P) once it exists:
paraphrased *attacks* straddling the split are where memorisation can bite.

Reads data/processed/unified_v2_grouped.csv; writes reports/leakage/.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features_tfidf import build_vectorizer  # noqa: E402

warnings.filterwarnings("ignore")
IN = Path("data/processed/unified_v2_grouped.csv")
OUT = Path("reports/leakage")
SEEDS = range(10)
TEST_FRAC = 0.15


def model(min_df=2):
    return Pipeline([("t", build_vectorizer(min_df=min_df)),
                     ("c", LogisticRegression(max_iter=2000, class_weight="balanced",
                                              random_state=42))])


def random_split(pool, seed):
    shuf = pool.sample(frac=1, random_state=seed)
    n = int(len(shuf) * TEST_FRAC)
    return shuf.iloc[n:], shuf.iloc[:n]


def pool_li(pool):
    rows = []
    for s in SEEDS:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=s)
        a, b = next(gss.split(pool, pool.label, groups=pool.group_id))
        splits = {"grouped": (pool.iloc[a], pool.iloc[b]),
                  "random": random_split(pool, s)}
        r = {"seed": s}
        for name, (tr, te) in splits.items():
            pred = model().fit(tr.text, tr.label).predict(te.text)
            r[f"acc_{name}"] = accuracy_score(te.label, pred)
            r[f"f1_{name}"] = f1_score(te.label, pred, average="macro")
        rtr, rte = splits["random"]
        r["leaked_test_rows"] = int(rte.group_id.isin(set(rtr.group_id)).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def leaked_vs_clean(pool, source=None, train_frac=1.0, seeds=SEEDS, min_df=2):
    """Accuracy on leaked vs non-leaked test rows inside a random split."""
    out = []
    for s in seeds:
        tr, te = random_split(pool, s)
        tr = tr.sample(frac=train_frac, random_state=s)
        if tr.label.nunique() < 2:
            continue
        te = te.copy()
        te["ok"] = model(min_df).fit(tr.text, tr.label).predict(te.text) == te.label.to_numpy()
        te["leaked"] = te.group_id.isin(set(tr.group_id))
        if source:
            te = te[te.source == source]
        if te.leaked.all() or not te.leaked.any():
            continue
        out.append({"seed": s, "train_n": len(tr),
                    "acc_leaked": te[te.leaked].ok.mean(),
                    "acc_clean": te[~te.leaked].ok.mean(),
                    "overall": te.ok.mean()})
    return pd.DataFrame(out)


def gap_line(r):
    d = r.acc_leaked - r.acc_clean
    if d.std() == 0:
        return f"gap {d.mean():+.4f} (no variance; test undefined)"
    t, p = stats.ttest_rel(r.acc_leaked, r.acc_clean)
    return f"gap {d.mean():+.4f} ± {d.std():.4f}, paired t={t:.2f}, p={p:.4g}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say("===== 1. LEAKAGE INFLATION BY POOL (10 seeds) =====")
    pools = {"deepset": df[df.source == "deepset"],
             "deepset+lakera": df[df.source.isin(["deepset", "lakera"])],
             "all_non_bnen": df[df.source != "bnen"]}
    summary = []
    for name, pool in pools.items():
        r = pool_li(pool)
        li = r.acc_random - r.acc_grouped
        summary.append({"pool": name, "rows": len(pool),
                        "leaked_test_rows": r.leaked_test_rows.mean(),
                        "acc_random": r.acc_random.mean(), "acc_grouped": r.acc_grouped.mean(),
                        "LI_acc_mean": li.mean(), "LI_acc_std": li.std()})
        say(f"  {name:15s} rows {len(pool):5d}  leaked test rows {r.leaked_test_rows.mean():6.1f}"
            f"  LI = {li.mean():+.4f} ± {li.std():.4f}")
    pd.DataFrame(summary).to_csv(OUT / "li_by_pool.csv", index=False)

    full = pools["all_non_bnen"]
    say("\n===== 2. LEAKED vs NON-LEAKED, same random split (uncontrolled) =====")
    say("  " + gap_line(leaked_vs_clean(full)))

    say("\n===== 3. CONTROL: PromptBench rows only (one source, one label) =====")
    say("  " + gap_line(leaked_vs_clean(full, source="promptbench")))
    say("  -> step 2's gap is compositional, not memorisation.")

    say("\n===== 4. LEARNING CURVE (control, 8 seeds) =====")
    curve = []
    for frac in [0.02, 0.05, 0.10, 0.25, 0.50, 1.00]:
        r = leaked_vs_clean(full, source="promptbench", train_frac=frac,
                            seeds=range(8), min_df=1)
        if len(r):
            curve.append({"train_n": int(r.train_n.mean()),
                          "acc_leaked": r.acc_leaked.mean(),
                          "acc_clean": r.acc_clean.mean(),
                          "gap": (r.acc_leaked - r.acc_clean).mean()})
            say(f"  train {curve[-1]['train_n']:5d}  leaked {curve[-1]['acc_leaked']:.4f}"
                f"  clean {curve[-1]['acc_clean']:.4f}  gap {curve[-1]['gap']:+.4f}")
    pd.DataFrame(curve).to_csv(OUT / "learning_curve.csv", index=False)

    say("\nConclusion: H4 not supported on this corpus. Re-run on the P set.")
    (OUT / "leakage_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved -> {OUT}/")


if __name__ == "__main__":
    main()
