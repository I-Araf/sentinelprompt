#!/usr/bin/env python3
"""Classical baselines M1-M3 (methodology 3.10.1-3.10.3, 3.11.2).

  M1 TF-IDF + Logistic Regression   C {0.01,0.1,1,10,100} x penalty {l1,l2}
                                    x class_weight {None,'balanced'}
                                    (penalty is set through l1_ratio: 0 = L2,
                                    1 = L1 -- sklearn >= 1.8 deprecates `penalty`)
  M2 TF-IDF + Multinomial NB        alpha {0.01,0.1,0.5,1.0}
  M3 TF-IDF + Linear SVM            C {0.1,1,10}, class_weight='balanced'

Tuning: GridSearchCV, StratifiedGroupKFold(k=5), scoring = macro-F1. Groups are
kept inside tuning too -- otherwise leakage enters through model selection.
The TF-IDF step sits inside the pipeline, so each fold fits its own vocabulary.
The decision threshold is then tuned on validation (3.12.3); test is used once.

Outputs: models/classical/*.joblib, data/processed/features/*.npz|.npy,
results/figures/, results/tables/.
"""
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev                                        # noqa: E402
import guards                                                # noqa: E402
from features_tfidf import build_vectorizer, save_matrices   # noqa: E402

MODELS = Path("models/classical")
SEED = 42

SPECS = {
    "logreg": (LogisticRegression(solver="liblinear", max_iter=5000, random_state=SEED),
               {"clf__C": [0.01, 0.1, 1, 10, 100], "clf__l1_ratio": [0, 1],
                "clf__class_weight": [None, "balanced"]}),
    "naive_bayes": (MultinomialNB(), {"clf__alpha": [0.01, 0.1, 0.5, 1.0]}),
    "linear_svm": (LinearSVC(class_weight="balanced", max_iter=20000, random_state=SEED),
                   {"clf__C": [0.1, 1, 10]}),
}


def load_data():
    tr, va = ev.load_split("train"), ev.load_split("val")
    guards.assert_no_lakera_in_train(tr)
    guards.assert_no_bnen_in_train(tr)
    guards.assert_originals_only(tr)
    guards.assert_no_group_overlap(tr, va, ev.load_split("test_clean"),
                                   names=["train", "val", "test_clean"])
    return tr, va


def save_features(tr, va, conditions):
    """Standalone vectorizer fitted on train only, matrices for every split."""
    vec = build_vectorizer().fit(tr.text)
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, MODELS / "tfidf_vectorizer.joblib")
    return save_matrices(vec, {"train": tr, "val": va, **conditions})


def score_fn(pipe):
    """Continuous score, higher = more likely injection."""
    if hasattr(pipe, "predict_proba"):
        return lambda texts: pipe.predict_proba(texts)[:, 1]
    return lambda texts: pipe.decision_function(texts)


def fit(name, tr, verbose=True):
    clf, grid = SPECS[name]
    pipe = Pipeline([("tfidf", build_vectorizer()), ("clf", clf)])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)
    t0 = time.perf_counter()
    gs.fit(tr.text, tr.label, groups=tr.group_id)
    secs = time.perf_counter() - t0
    if verbose:
        print(f"{name}: best CV macro-F1 {gs.best_score_:.4f} with {gs.best_params_}"
              f"  ({len(gs.cv_results_['params'])} combinations x 5 folds, {secs:.1f}s)")
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(gs.best_estimator_, MODELS / f"{name}.joblib")
    return gs, secs


def run_one(name, tr, va, conditions, verbose=True):
    gs, secs = fit(name, tr, verbose)
    pipe = gs.best_estimator_
    sf = score_fn(pipe)
    thr, val_f1 = ev.tune_threshold(va.label.to_numpy(), sf(va.text.tolist()))
    if verbose:
        print(f"{name}: threshold tuned on validation = {thr:.4f} (val macro-F1 {val_f1:.4f})")
    run = ev.evaluate_model(name, sf, thr, conditions, tr, verbose=verbose)
    t0 = time.perf_counter()
    pipe.predict(conditions["E1_clean"].text)
    run["meta"] = {"model": name, "best_params": str(gs.best_params_),
                   "cv_macro_f1": gs.best_score_, "val_macro_f1": val_f1,
                   "threshold": thr, "tuning_seconds": round(secs, 1),
                   "ms_per_row": round((time.perf_counter() - t0) /
                                       len(conditions["E1_clean"]) * 1000, 4),
                   "size_mb": round((MODELS / f"{name}.joblib").stat().st_size / 1e6, 2)}
    return run


def save_all(runs):
    ev.save_table(ev.tidy(pd.concat([r["metrics"] for r in runs])), "metrics_baselines")
    ev.save_table(pd.concat([r["reports"] for r in runs]), "classification_reports_baselines")
    ev.save_table(pd.DataFrame([r["meta"] for r in runs]), "baselines_tuning_and_efficiency")
    checks = [r["checks"] for r in runs if len(r["checks"])]
    if checks:
        ev.save_table(pd.concat(checks), "suspicious_accuracy_checks_baselines")


def main():
    tr, va = load_data()
    conditions = ev.load_conditions()
    print("feature matrices:", save_features(tr, va, conditions))
    runs = [run_one(n, tr, va, conditions) for n in SPECS]
    save_all(runs)
    print(ev.tidy(pd.concat([r["metrics"] for r in runs]))[
        ["model", "condition", "accuracy", "macro_f1", "roc_auc", "pr_auc"]].round(4).to_string())


if __name__ == "__main__":
    main()
