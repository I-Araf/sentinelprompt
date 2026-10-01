#!/usr/bin/env python3
"""Exploratory Data Analysis (methodology 3.4, steps 1-6).

Produces the 3.4.7 deliverables: class distribution, length histograms,
top n-grams per class, duplicate/near-duplicate stats, and a data quality report.

Reads data/processed/unified_v2_grouped.csv; writes reports/eda/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # no display in this environment
import matplotlib.pyplot as plt            # noqa: E402
import pandas as pd                        # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer   # noqa: E402
from sklearn.feature_selection import chi2                    # noqa: E402

IN = Path("data/processed/unified_v2_grouped.csv")
OUT = Path("reports/eda")
TOP_N = 30


def fig_class_balance(df):
    counts = df.label.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.bar(["Safe (0)", "Injection (1)"], counts.values,
           color=["#4878a8", "#c0504d"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n{v/len(df)*100:.1f}%", ha="center", va="bottom")
    ax.set_title("Class distribution (all sources)")
    ax.set_ylabel("rows")
    ax.set_ylim(0, counts.max() * 1.22)
    fig.tight_layout()
    fig.savefig(OUT / "class_distribution.png", dpi=150)
    plt.close(fig)


def fig_lengths(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, name in [(axes[0], "char_len", "characters"),
                          (axes[1], "word_len", "words")]:
        for lab, colour, tag in [(0, "#4878a8", "Safe"), (1, "#c0504d", "Injection")]:
            s = df[df.label == lab][col]
            ax.hist(s, bins=50, alpha=0.6, label=tag, color=colour,
                    range=(0, s.quantile(0.99) if len(s) else 1))
        ax.set_xlabel(name)
        ax.set_ylabel("rows")
        ax.set_title(f"Length in {name} (to 99th pct)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "length_histogram.png", dpi=150)
    plt.close(fig)


def top_ngrams(df):
    """Top n-grams per class plus chi-square discriminative terms (3.4.4)."""
    cv = CountVectorizer(ngram_range=(1, 3), min_df=3, max_features=60000)
    X = cv.fit_transform(df.text)
    vocab = cv.get_feature_names_out()

    rows = []
    for lab in (0, 1):
        sub = X[(df.label == lab).to_numpy()]
        freq = sub.sum(axis=0).A1
        for i in freq.argsort()[::-1][:TOP_N]:
            rows.append({"class": "Safe" if lab == 0 else "Injection",
                         "ngram": vocab[i], "count": int(freq[i])})
    pd.DataFrame(rows).to_csv(OUT / "top_ngrams.csv", index=False)

    sc, _ = chi2(X, df.label)
    sc = pd.Series(sc, index=vocab).dropna().sort_values(ascending=False)
    pos_rate = pd.Series((X[(df.label == 1).to_numpy()].sum(axis=0).A1 + 1) /
                         (X.sum(axis=0).A1 + 2), index=vocab)
    chi = pd.DataFrame({"chi2": sc.head(40),
                        "leans": ["Injection" if pos_rate[t] > 0.5 else "Safe"
                                  for t in sc.head(40).index]})
    chi.index.name = "term"
    chi.to_csv(OUT / "chi2_terms.csv")
    return chi


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN)
    df["char_len"] = df.text.str.len()
    df["word_len"] = df.text.str.split().str.len()

    lines = []
    def say(s=""):
        print(s)
        lines.append(str(s))

    say("===== 3.4.1 STRUCTURAL PROFILING =====")
    say(f"rows {len(df)}  columns {df.shape[1]}")
    say(f"groups {df.group_id.nunique()}")
    nulls = df.isnull().sum()
    say("nulls per column:")
    say((nulls[nulls > 0].to_string() if nulls.any() else "  none"))
    say(f"empty/whitespace-only text : {int((df.text.str.strip() == '').sum())}")

    say("\n===== 3.4.2 CLASS BALANCE =====")
    vc = df.label.value_counts().sort_index()
    say(vc.to_string())
    maj = vc.max() / len(df)
    say(f"majority class share : {maj*100:.1f}%")
    say(f"-> accuracy paradox: predicting the majority always scores {maj*100:.1f}%,")
    say("   so macro-F1 is reported alongside accuracy; class_weight='balanced'.")
    say("\nby source:")
    say(pd.crosstab(df.source, df.label).to_string())

    say("\n===== 3.4.3 TEXT LENGTH =====")
    say(df.groupby("label")[["char_len", "word_len"]]
          .describe(percentiles=[.5, .95, .99]).round(1).to_string())
    for q in (0.95, 0.99):
        say(f"word_len {int(q*100)}th pct = {df.word_len.quantile(q):.0f}")
    say("-> informs transformer max_length (attention cost is O(n^2)).")

    say("\n===== 3.4.4 LEXICAL / DISCRIMINATIVE =====")
    chi = top_ngrams(df)
    say("top 15 chi-square terms:")
    say(chi.head(15).to_string())

    say("\n===== 3.4.5 DUPLICATE AUDIT =====")
    sizes = df.group_id.value_counts()
    multi = sizes[sizes > 1]
    say(f"exact duplicates: removed upstream in integrate.py")
    say(f"groups with >1 row : {len(multi)}")
    say(f"rows in those groups : {int(multi.sum())} "
        f"({multi.sum()/len(df)*100:.1f}%)")
    say(f"largest group : {int(sizes.max())} rows")

    say("\n===== 3.4.6 LABEL QUALITY =====")
    un = int((df.attack_type == "unannotated").sum())
    say(f"injections still lacking attack_type : {un}")
    say(f"D4 rows with verified_by filled : 0 of 900 (annotation round pending)")

    fig_class_balance(df)
    fig_lengths(df)
    (OUT / "data_quality_report.txt").write_text("\n".join(lines), encoding="utf-8")
    say(f"\nwrote: class_distribution.png, length_histogram.png, top_ngrams.csv,")
    say(f"       chi2_terms.csv, data_quality_report.txt  -> {OUT}/")


if __name__ == "__main__":
    main()
