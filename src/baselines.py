#!/usr/bin/env python3
"""Classical baselines + evaluation conditions (methodology 3.10, 3.12, E1-E7).

Models: Logistic Regression, Multinomial Naive Bayes, Linear SVM.
All use class_weight='balanced' where supported (3.4.2) -- the corpus is ~74% Safe.

Reads data/processed/splits_v1.csv; writes reports/baselines/.
"""
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guards                      # noqa: E402
from features_tfidf import build_vectorizer   # noqa: E402

IN = Path("data/processed/splits_v1.csv")
OUT = Path("reports/baselines")
SEED = 42

MODELS = {
    "logreg": lambda: LogisticRegression(max_iter=2000, class_weight="balanced",
                                         random_state=SEED),
    "multinomial_nb": lambda: MultinomialNB(),          # no class_weight support
    "linear_svm": lambda: LinearSVC(class_weight="balanced", random_state=SEED),
}

# E1-E5 + the false-positive-only condition
CONDITIONS = {
    "E1_test_clean": "test_clean",
    "E4_test_codemix": "test_codemix",
    "E5_test_crossdataset": "test_crossdataset",
    "FP_promptbench": "test_falsepos",
}


def score(y_true, y_pred):
    """Two held-out sets are single-class, where macro-F1 is undefined."""
    out = {"n": len(y_true), "accuracy": accuracy_score(y_true, y_pred)}
    if y_true.nunique() < 2:
        only = int(y_true.iloc[0])
        if only == 1:
            out["recall_injection"] = float((y_pred == 1).mean())
            out["note"] = "all-injection set: recall only"
        else:
            out["false_positive_rate"] = float((y_pred == 1).mean())
            out["note"] = "all-safe set: FPR only"
    else:
        out["macro_f1"] = f1_score(y_true, y_pred, average="macro")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN)
    tr = df[df.split == "train"]
    va = df[df.split == "val"]

    # the invariants that make E4/E5 meaningful at all
    guards.assert_no_lakera_in_train(tr)
    guards.assert_no_bnen_in_train(tr)
    guards.assert_originals_only(tr)
    print(f"train {len(tr)} | val {len(va)} | guards passed\n")

    rows, lines = [], []
    for name, make in MODELS.items():
        pipe = Pipeline([("tfidf", build_vectorizer()), ("clf", make())])
        t0 = time.perf_counter()
        pipe.fit(tr.text, tr.label)
        fit_s = time.perf_counter() - t0

        r = {"model": name, "fit_seconds": round(fit_s, 2)}
        r.update({f"val_{k}": v for k, v in
                  score(va.label, pipe.predict(va.text)).items()})

        for cond, split in CONDITIONS.items():
            part = df[df.split == split]
            t0 = time.perf_counter()
            pred = pipe.predict(part.text)
            ms = (time.perf_counter() - t0) / len(part) * 1000   # E7 latency
            s = score(part.label, pred)
            for k, v in s.items():
                r[f"{cond}_{k}"] = v
            r[f"{cond}_ms_per_row"] = round(ms, 4)

        rows.append(r)
        print(f"--- {name} (fit {fit_s:.2f}s) ---")
        for cond, split in CONDITIONS.items():
            part = df[df.split == split]
            s = score(part.label, pipe.predict(part.text))
            key = "macro_f1" if "macro_f1" in s else (
                "recall_injection" if "recall_injection" in s else "false_positive_rate")
            print(f"  {cond:22s} n={s['n']:5d}  acc={s['accuracy']:.3f}  "
                  f"{key}={s[key]:.3f}")
        lines.append(f"===== {name} =====")
        lines.append(classification_report(
            df[df.split == 'test_clean'].label,
            pipe.predict(df[df.split == 'test_clean'].text),
            target_names=["Safe", "Injection"], zero_division=0))

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "baseline_results.csv", index=False)
    (OUT / "classification_reports.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved -> {OUT}/baseline_results.csv, classification_reports.txt")


if __name__ == "__main__":
    main()
