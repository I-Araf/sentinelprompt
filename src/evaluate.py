#!/usr/bin/env python3
"""One evaluation implementation for every model (methodology 3.12, 3.13.1).

Every model -- classical or transformer -- goes through the same functions here,
so a difference in numbers is a difference in models, never in metric code.

Positive class = Injection (1), negative = Safe (0), as in 3.12.1.
Figures -> results/figures/ (PNG, 300 dpi); tables -> results/tables/ (CSV).
"""
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

DATA = Path("data/processed")
FIG = Path("results/figures")
TAB = Path("results/tables")
PRED = TAB / "predictions"
DPI = 300
SUSPICIOUS_ACC = 0.97

# condition id -> split file(s). E5x pools Lakera (all injection) with
# PromptBench (all safe): each alone is single-class, so ROC/PR-AUC is only
# defined once they are pooled.
CONDITIONS = {
    "E1_clean": ["test_clean"],
    "E2_paraphrase": ["test_paraphrase"],
    "E3_obfuscated": ["test_obfuscated"],
    "E4_codemix": ["test_codemix"],
    "E5_crossdataset": ["test_crossdataset"],
    "FP_promptbench": ["test_falsepos"],
    "E5x_pooled": ["test_crossdataset", "test_falsepos"],
}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_split(name):
    return pd.read_csv(DATA / f"{name}.csv")


def load_conditions(verbose=True):
    """All test conditions that exist on disk; missing ones are reported."""
    out = {}
    for cond, files in CONDITIONS.items():
        paths = [DATA / f"{f}.csv" for f in files]
        if not all(p.exists() for p in paths):
            if verbose:
                print(f"  skip {cond}: {', '.join(p.name for p in paths if not p.exists())} not found")
            continue
        out[cond] = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def tune_threshold(y_val, scores_val):
    """Pick the threshold that maximises validation macro-F1 (3.12.3).
    The test set is never used for this (3.11.5)."""
    cands = np.unique(np.quantile(scores_val, np.linspace(0, 1, 201)))
    best_t, best_f = 0.5, -1.0
    for t in cands:
        f = f1_score(y_val, (scores_val >= t).astype(int), average="macro",
                     zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f


def compute_metrics(y, pred, scores):
    """Full metric set. Single-class sets get NaN where a metric is undefined
    instead of a number that looks valid but is not."""
    y, pred = np.asarray(y), np.asarray(pred)
    two_class = len(np.unique(y)) == 2
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = {
        "n": len(y), "n_injection": int(y.sum()),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0) if y.sum() else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1_injection": f1_score(y, pred, zero_division=0) if y.sum() else np.nan,
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0) if two_class else np.nan,
        "mcc": matthews_corrcoef(y, pred) if two_class else np.nan,
        "roc_auc": roc_auc_score(y, scores) if two_class else np.nan,
        "pr_auc": average_precision_score(y, scores) if two_class else np.nan,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    if not two_class:
        m["note"] = ("all-injection set: accuracy = recall" if y.sum()
                     else "all-safe set: accuracy = specificity")
    return m


def report_rows(model, cond, y, pred):
    rep = classification_report(y, pred, labels=[0, 1], target_names=["Safe", "Injection"],
                                output_dict=True, zero_division=0)
    rows = []
    for cls in ["Safe", "Injection", "macro avg", "weighted avg"]:
        r = rep[cls]
        rows.append({"model": model, "condition": cond, "class": cls,
                     "precision": r["precision"], "recall": r["recall"],
                     "f1": r["f1-score"], "support": int(r["support"])})
    return rows


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=DPI, bbox_inches="tight")


def draw_confusion(ax, y, pred, title):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    ax.imshow(cm, cmap="Blues")
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f"{v}\n{v / total:.1%}" if total else "0", ha="center",
                    va="center", color="white" if v > cm.max() / 2 else "black", fontsize=10)
    ax.set_xticks([0, 1], ["Safe", "Injection"])
    ax.set_yticks([0, 1], ["Safe", "Injection"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)


def draw_roc(ax, y, scores, title):
    if len(np.unique(y)) < 2:
        ax.text(0.5, 0.5, "ROC undefined:\nsingle-class set", ha="center", va="center")
        ax.set_axis_off()
        ax.set_title(title, fontsize=10)
        return
    fpr, tpr, _ = roc_curve(y, scores)
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y, scores):.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title, fontsize=10)


def draw_pr(ax, y, scores, title):
    if len(np.unique(y)) < 2:
        ax.text(0.5, 0.5, "PR curve undefined:\nsingle-class set", ha="center", va="center")
        ax.set_axis_off()
        ax.set_title(title, fontsize=10)
        return
    p, r, _ = precision_recall_curve(y, scores)
    ax.plot(r, p, lw=2, label=f"PR-AUC = {average_precision_score(y, scores):.3f}")
    ax.axhline(np.mean(y), ls="--", color="grey", lw=1, label="chance (base rate)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(title, fontsize=10)


DRAW = {"confusion": draw_confusion, "roc": draw_roc, "pr": draw_pr}


def save_single_figures(model, cond, y, pred, scores):
    """One PNG per model x condition x figure type (the archival copies)."""
    for kind, fn in DRAW.items():
        if kind != "confusion" and len(np.unique(y)) < 2:
            continue                       # undefined; noted in the panel instead
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        arg = pred if kind == "confusion" else scores
        fn(ax, y, arg, f"{model} — {cond}")
        _save(fig, f"{model}_{cond}_{kind}")
        plt.close(fig)


def panel(run, kind):
    """All conditions of one model side by side -- the notebook view."""
    conds = list(run["conditions"])
    cols = min(4, len(conds))
    rows = int(np.ceil(len(conds) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, cond in zip(axes, conds):
        c = run["conditions"][cond]
        DRAW[kind](ax, c["y"], c["pred"] if kind == "confusion" else c["scores"], cond)
    for ax in axes[len(conds):]:
        ax.set_axis_off()
    fig.suptitle(f"{run['model']} — {kind.upper() if kind != 'confusion' else 'confusion matrix'}",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, f"{run['model']}_panel_{kind}")
    return fig


def comparison_chart(metrics, metric="macro_f1", name=None):
    """Grouped bars: conditions on x, one bar per model."""
    d = metrics.pivot_table(index="condition", columns="model", values=metric)
    d = d.reindex([c for c in CONDITIONS if c in d.index])
    d = d.dropna(how="all")
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(d)), 4.4))
    n = len(d.columns)
    w = 0.8 / n
    x = np.arange(len(d))
    for i, m in enumerate(d.columns):
        vals = d[m].to_numpy()
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=m)
    ax.set_xticks(x, d.index, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.90, ls=":", color="grey", lw=1)
    ax.legend(fontsize=8, ncol=min(n, 4), loc="lower left")
    ax.set_title(f"{metric} by model and condition (dotted line: 0.90 target)")
    fig.tight_layout()
    _save(fig, name or f"comparison_{metric}")
    return fig


# --------------------------------------------------------------------------
# the >97% accuracy guard (requirement: warn + run a leakage check)
# --------------------------------------------------------------------------
def _shingles(s, k=5):
    s = str(s).lower()
    return {s[i:i + k] for i in range(max(1, len(s) - k + 1))}


def leakage_check(train, test, jaccard=0.80):
    """Does any test row also live in training -- by group, text, or near-copy?"""
    norm = lambda s: hashlib.sha256(" ".join(str(s).lower().split()).encode()).hexdigest()
    tr_hash = set(train.text.map(norm))
    lsh = MinHashLSH(threshold=jaccard, num_perm=64)
    for i, t in enumerate(train.text):
        m = MinHash(num_perm=64)
        for g in _shingles(t):
            m.update(g.encode("utf8"))
        lsh.insert(str(i), m)
    tr_sh = [_shingles(t) for t in train.text]
    near = 0
    for t in test.text:
        sh = _shingles(t)
        m = MinHash(num_perm=64)
        for g in sh:
            m.update(g.encode("utf8"))
        if any(len(sh & tr_sh[int(k)]) / max(1, len(sh | tr_sh[int(k)])) >= jaccard
               for k in lsh.query(m)):
            near += 1
    return {"shared_groups": len(set(train.group_id) & set(test.group_id)),
            "exact_text_overlap": int(test.text.map(norm).isin(tr_hash).sum()),
            "near_duplicates_in_train": near,
            "test_rows": len(test),
            "test_is_single_class": test.label.nunique() < 2}


def suspicious_accuracy_check(model, cond, acc, train, test, verbose=True):
    if acc <= SUSPICIOUS_ACC:
        return None
    chk = leakage_check(train, test)
    if verbose:
        print("\n" + "!" * 72)
        print(f"WARNING: {model} on {cond}: accuracy {acc:.4f} > {SUSPICIOUS_ACC:.2f}.")
        print("Unusually high accuracy is a classic symptom of train/test leakage.")
        print("Leakage check:")
        for k, v in chk.items():
            print(f"   {k:26s} {v}")
        leaked = chk["shared_groups"] or chk["exact_text_overlap"] or chk["near_duplicates_in_train"]
        if leaked:
            print("-> LEAKAGE FOUND. This number should not be reported as is.")
        elif chk["test_is_single_class"]:
            print("-> No leakage. The set holds a single class, so accuracy equals recall"
                  " (or specificity) and says nothing about the other class.")
        else:
            print("-> No leakage found. The high score is not explained by overlap.")
        print("!" * 72 + "\n")
    chk.update({"model": model, "condition": cond, "accuracy": acc})
    return chk


# --------------------------------------------------------------------------
# one entry point per model
# --------------------------------------------------------------------------
def evaluate_model(model, score_fn, threshold, conditions, train,
                   save_figures=True, save_predictions=True, tag=None, verbose=True):
    """Run one fitted model over every condition.

    score_fn(texts) -> continuous score where higher means "more likely injection"
    (probability, or SVM decision value). threshold is tuned on validation.
    """
    tag = tag or model
    run = {"model": model, "threshold": threshold, "conditions": {}}
    metrics, reports, checks, preds = [], [], [], []
    # score every distinct text once: E5x reuses E5 and FP rows, and transformer
    # inference is the expensive part
    texts = pd.unique(pd.concat([d.text for d in conditions.values()], ignore_index=True))
    lookup = dict(zip(texts, np.asarray(score_fn(list(texts)), dtype=float)))
    for cond, df in conditions.items():
        scores = df.text.map(lookup).to_numpy(dtype=float)
        pred = (scores >= threshold).astype(int)
        y = df.label.to_numpy()
        m = compute_metrics(y, pred, scores)
        m.update({"model": model, "condition": cond, "threshold": threshold})
        metrics.append(m)
        reports += report_rows(model, cond, y, pred)
        run["conditions"][cond] = {"y": y, "pred": pred, "scores": scores}
        if save_figures:
            save_single_figures(model, cond, y, pred, scores)
        chk = suspicious_accuracy_check(model, cond, m["accuracy"], train, df, verbose)
        if chk:
            checks.append(chk)
        if save_predictions:
            keep = [c for c in ["id", "parent_id", "group_id", "language", "technique",
                                "perturbation_rate", "method"] if c in df.columns]
            p = df[keep].copy()
            p["condition"], p["label"], p["score"], p["pred"] = cond, y, scores, pred
            preds.append(p)
    if save_predictions and preds:
        PRED.mkdir(parents=True, exist_ok=True)
        pd.concat(preds, ignore_index=True).to_csv(PRED / f"{tag}.csv", index=False)
    run["metrics"] = pd.DataFrame(metrics)
    run["reports"] = pd.DataFrame(reports)
    run["checks"] = pd.DataFrame(checks)
    return run


METRIC_COLS = ["model", "condition", "n", "n_injection", "threshold", "accuracy",
               "precision", "recall", "specificity", "f1_injection", "macro_f1", "mcc",
               "roc_auc", "pr_auc", "tp", "fp", "tn", "fn", "note"]


def tidy(metrics):
    m = metrics.copy()
    for c in METRIC_COLS:
        if c not in m.columns:
            m[c] = np.nan
    return m[METRIC_COLS]


def save_table(df, name):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False)
    return TAB / f"{name}.csv"
