#!/usr/bin/env python3
"""Exploratory Data Analysis (methodology 3.4, steps 1-6).

Produces the 3.4.7 deliverables: class distribution, length histograms,
top n-grams per class, chi-square terms, duplicate statistics and a data
quality report. Each step is a function so the EDA notebook can run and
discuss them one at a time; main() runs them all.

Reads data/processed/unified_v2_grouped.csv.
Figures -> results/figures/eda_*.png (300 dpi); tables -> results/tables/eda_*.csv.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

IN = Path("data/processed/unified_v2_grouped.csv")
FIG = Path("results/figures")
TAB = Path("results/tables")
DPI = 300
TOP_N = 30


def load():
    df = pd.read_csv(IN)
    df["char_len"] = df.text.str.len()
    df["word_len"] = df.text.str.split().str.len()
    return df


def _save_fig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"eda_{name}.png", dpi=DPI, bbox_inches="tight")


def _save_tab(df, name, index=False):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"eda_{name}.csv", index=index)


def structural_profile(df):
    """3.4.1: size, nulls, empty text."""
    nulls = df.isnull().sum()
    out = {"rows": len(df), "columns": df.shape[1], "groups": df.group_id.nunique(),
           "empty_text": int((df.text.str.strip() == "").sum())}
    _save_tab(pd.DataFrame([out]), "structural_profile")
    return out, nulls[nulls > 0]


def class_balance(df):
    """3.4.2: class ratio overall and per source, plus the bar chart."""
    counts = df.label.value_counts().sort_index()
    by_source = pd.crosstab(df.source, df.label).rename(columns={0: "safe", 1: "injection"})
    _save_tab(by_source, "class_by_source", index=True)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.bar(["Safe (0)", "Injection (1)"], counts.values, color=["#4878a8", "#c0504d"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n{v / len(df) * 100:.1f}%", ha="center", va="bottom")
    ax.set_title("Class distribution (all sources)")
    ax.set_ylabel("rows")
    ax.set_ylim(0, counts.max() * 1.22)
    fig.tight_layout()
    _save_fig(fig, "class_distribution")
    return counts, by_source, fig


def length_distribution(df):
    """3.4.3: length statistics and per-class histograms."""
    stats = (df.groupby("label")[["char_len", "word_len"]]
               .describe(percentiles=[.5, .95, .99]).round(1))
    _save_tab(stats, "length_stats", index=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, name in [(axes[0], "char_len", "characters"), (axes[1], "word_len", "words")]:
        for lab, colour, tag in [(0, "#4878a8", "Safe"), (1, "#c0504d", "Injection")]:
            s = df[df.label == lab][col]
            ax.hist(s, bins=50, alpha=0.6, label=tag, color=colour,
                    range=(0, s.quantile(0.99) if len(s) else 1))
        ax.set_xlabel(name)
        ax.set_ylabel("rows")
        ax.set_title(f"Length in {name} (to 99th pct)")
        ax.legend()
    fig.tight_layout()
    _save_fig(fig, "length_histogram")
    pct = {q: float(df.word_len.quantile(q)) for q in (0.95, 0.99)}
    return stats, pct, fig


def lexical(df):
    """3.4.4: top n-grams per class and chi-square discriminative terms."""
    cv = CountVectorizer(ngram_range=(1, 3), min_df=3, max_features=60000)
    X = cv.fit_transform(df.text)
    vocab = cv.get_feature_names_out()
    rows = []
    for lab in (0, 1):
        freq = X[(df.label == lab).to_numpy()].sum(axis=0).A1
        for i in freq.argsort()[::-1][:TOP_N]:
            rows.append({"class": "Safe" if lab == 0 else "Injection",
                         "ngram": vocab[i], "count": int(freq[i])})
    top = pd.DataFrame(rows)
    _save_tab(top, "top_ngrams")
    sc, _ = chi2(X, df.label)
    sc = pd.Series(sc, index=vocab).dropna().sort_values(ascending=False)
    pos_rate = pd.Series((X[(df.label == 1).to_numpy()].sum(axis=0).A1 + 1) /
                         (X.sum(axis=0).A1 + 2), index=vocab)
    chi = pd.DataFrame({"term": sc.head(40).index, "chi2": sc.head(40).values,
                        "leans": ["Injection" if pos_rate[t] > 0.5 else "Safe"
                                  for t in sc.head(40).index]})
    _save_tab(chi, "chi2_terms")
    return top, chi


def duplicate_audit(df):
    """3.4.5: how much of the corpus sits in multi-row near-duplicate groups."""
    sizes = df.group_id.value_counts()
    multi = sizes[sizes > 1]
    out = {"groups_with_more_than_one_row": len(multi),
           "rows_in_those_groups": int(multi.sum()),
           "share_of_rows": round(multi.sum() / len(df), 4),
           "largest_group": int(sizes.max())}
    _save_tab(pd.DataFrame([out]), "duplicate_audit")
    return out


def label_quality(df):
    """3.4.6: what is still unannotated or unverified."""
    out = {"injections_without_attack_type": int((df.attack_type == "unannotated").sum()),
           "bnen_rows": int((df.source == "bnen").sum()),
           "bnen_rows_verified": 0}
    _save_tab(pd.DataFrame([out]), "label_quality")
    return out


def main():
    df = load()
    print(structural_profile(df)[0])
    print(class_balance(df)[0].to_dict())
    print(length_distribution(df)[1])
    print(lexical(df)[1].head(10).to_string(index=False))
    print(duplicate_audit(df))
    print(label_quality(df))
    print(f"figures -> {FIG}/eda_*.png ; tables -> {TAB}/eda_*.csv")


if __name__ == "__main__":
    main()
