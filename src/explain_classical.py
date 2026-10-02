#!/usr/bin/env python3
"""Interpretability of the classical baselines (methodology 3.15.3).

  Logistic Regression  the 30 largest positive and negative weights: which
                       n-grams push a prompt towards Injection, which towards Safe
  Linear SVM           the same reading of its weight vector
  Naive Bayes          log P(term | Injection) - log P(term | Safe), the
                       log-probability ratio named in the methodology

Feature names carry their block: `word__ignore previous` is a word n-gram,
`char__ ign` a character n-gram (char_wb pads words with spaces).
Reads the tuned pipelines saved by baselines.py in models/classical/.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = Path("models/classical")
FIG = Path("results/figures")
TAB = Path("results/tables")
DPI = 300
TOP = 30


def weights(name):
    """(feature names, one weight per feature) for a saved pipeline."""
    pipe = joblib.load(MODELS / f"{name}.joblib")
    names = pipe.named_steps["tfidf"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    if name == "naive_bayes":
        w = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    else:
        w = clf.coef_.ravel()
    return names, np.asarray(w, dtype=float)


def top_features(name, k=TOP):
    names, w = weights(name)
    order = np.argsort(w)
    rows = [{"model": name, "direction": "towards_injection", "rank": i + 1,
             "block": names[j].split("__")[0], "feature": names[j].split("__", 1)[1],
             "weight": w[j]} for i, j in enumerate(order[::-1][:k])]
    rows += [{"model": name, "direction": "towards_safe", "rank": i + 1,
              "block": names[j].split("__")[0], "feature": names[j].split("__", 1)[1],
              "weight": w[j]} for i, j in enumerate(order[:k])]
    return pd.DataFrame(rows)


def block_share(name):
    """How much of the total absolute weight sits in word vs char n-grams."""
    names, w = weights(name)
    blocks = np.array([n.split("__")[0] for n in names])
    tot = np.abs(w).sum()
    return {b: float(np.abs(w[blocks == b]).sum() / tot) for b in ["word", "char"]}


def plot(df, name, k=15):
    d = df[df.model == name]
    up = d[d.direction == "towards_injection"].head(k)[::-1]
    down = d[d.direction == "towards_safe"].head(k)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    lab = lambda r: f"{r.feature!r} ({r.block})"
    axes[0].barh([lab(r) for r in up.itertuples()], up.weight, color="#c0504d")
    axes[0].set_title("towards Injection")
    axes[1].barh([lab(r) for r in down.itertuples()][::-1], down.weight[::-1], color="#4878a8")
    axes[1].set_title("towards Safe")
    xl = "log P(t|Inj) - log P(t|Safe)" if name == "naive_bayes" else "weight"
    for ax in axes:
        ax.set_xlabel(xl)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(f"{name}: strongest features", fontsize=12)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"baselines_coefficients_{name}.png", dpi=DPI, bbox_inches="tight")
    return fig


def run(models=("logreg", "linear_svm", "naive_bayes")):
    tab = pd.concat([top_features(m) for m in models], ignore_index=True)
    TAB.mkdir(parents=True, exist_ok=True)
    tab.to_csv(TAB / "baselines_top_coefficients.csv", index=False)
    share = pd.DataFrame([{"model": m, **block_share(m)} for m in models])
    share.to_csv(TAB / "baselines_coefficient_block_share.csv", index=False)
    return tab, share


if __name__ == "__main__":
    t, s = run()
    print(s.to_string(index=False))
    print(t.groupby(["model", "direction"]).head(8).to_string(index=False))
