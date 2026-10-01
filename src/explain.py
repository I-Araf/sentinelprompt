#!/usr/bin/env python3
"""Interpretability and the spurious-correlation audit (methodology 3.15.3, 3.15.4).

  SHAP   the primary explanation tool (3.15.3). Computed in log-odds space:
         fine-tuned transformers sit near P = 1 on many inputs, where probability
         differences collapse to ~0 and every token looks irrelevant. Log-odds
         keeps those contributions visible. The SHAP base value is the score when every
         token is replaced by the tokenizer's mask token. That input is out of
         distribution, so it is NOT the model's prior on empty text -- for that, see
         neutral_prior(), which scores real content-free strings.
  LIME   a second, independent local explanation, to cross-check SHAP.
  Attention  last-layer [CLS] attention, shown only as a supporting picture:
         attention is not explanation (Jain & Wallace, 2019), as 3.15.3 notes.
  Audit  3.15.4: does a model take a shortcut? Length-label correlation, a
         source-probe classifier (dataset artefacts), the score a model gives to
         content-free input, and a paired script test on D4 (same prompt in
         English vs Bangla, so only the script differs).

Covers the transformer models (Part C). Coefficient inspection for Logistic
Regression and log-probability ratios for Naive Bayes belong with the classical
track (Part B) and are not done here.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import train_transformer as tt

FIG = Path("results/figures")
DPI = 300
MODELS = ["roberta", "distilbert", "distilbert_adv", "protectai_deberta"]
EPS = 1e-6
NEUTRAL = ["", "ok", "hello", "thank you", "What time is it?", "আচ্ছা", "ধন্যবাদ", "kemon acho"]


# ------------------------------------------------------------------ models
def load_model(name, seed=42):
    """(model, tokenizer, positive_index) for one of MODELS."""
    if name == "protectai_deberta":
        return tt.load_external()
    m, k = tt.load(name, seed)
    return m, k, 1


def prob_fn(bundle):
    m, k, pos = bundle
    return lambda texts: tt.predict_scores(m, k, [str(t) for t in texts], positive_index=pos)


def logodds_fn(bundle):
    p = prob_fn(bundle)

    def f(texts):
        q = np.clip(p(texts), EPS, 1 - EPS)      # one forward pass, not two
        return np.log(q / (1 - q))
    return f


# ------------------------------------------------------------------ SHAP
def shap_explain(bundle, texts, max_evals=300):
    import shap
    f = logodds_fn(bundle)
    ex = shap.Explainer(f, shap.maskers.Text(bundle[1]))
    sv = ex(list(texts), max_evals=max_evals, silent=True)
    probs = prob_fn(bundle)(texts)
    out = []
    for i, t in enumerate(texts):
        out.append({"text": t, "tokens": [s.strip() for s in sv.data[i]],
                    "values": np.asarray(sv.values[i], dtype=float),
                    "base_logodds": float(sv.base_values[i]),
                    "masked_baseline": float(1 / (1 + np.exp(-sv.base_values[i]))),
                    "score": float(probs[i])})
    return out


def top_tokens(expl, k=5):
    """Strongest pushes towards injection (+) and towards safe (-)."""
    rows = []
    for e in expl:
        order = np.argsort(e["values"])
        pos = [(e["tokens"][i], round(e["values"][i], 3)) for i in order[::-1][:k] if e["values"][i] > 0]
        neg = [(e["tokens"][i], round(e["values"][i], 3)) for i in order[:k] if e["values"][i] < 0]
        rows.append({"text": e["text"], "score": round(e["score"], 4),
                     "masked_baseline": round(e["masked_baseline"], 4),
                     "towards_injection": pos, "towards_safe": neg})
    return pd.DataFrame(rows)


def global_tokens(bundle, texts, max_evals=200, top=20):
    """Mean SHAP per (lower-cased) token over a sample -- the model's triggers."""
    expl = shap_explain(bundle, texts, max_evals)
    rec = [(t.lower(), v) for e in expl for t, v in zip(e["tokens"], e["values"]) if t and t.strip()]
    d = pd.DataFrame(rec, columns=["token", "shap"])
    g = d.groupby("token").shap.agg(["mean", "count"]).query("count >= 3")
    return (g.sort_values("mean", ascending=False).head(top),
            g.sort_values("mean").head(top),
            float(np.mean([e["masked_baseline"] for e in expl])))


def plot_tokens(e, title, name=None, ax=None):
    """Tokens coloured by SHAP value: red pushes to injection, blue to safe."""
    toks, vals = e["tokens"], e["values"]
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(12, 0.6 + 0.35 * (len(toks) // 14 + 1)))
    ax.set_axis_off()
    lim = max(1e-9, np.abs(vals).max())
    x, y, step = 0.0, 1.0, 1.0 / 14
    for t, v in zip(toks, vals):
        if not t:
            continue
        a = min(1.0, abs(v) / lim)
        colour = (1, 1 - a, 1 - a) if v > 0 else (1 - a, 1 - a, 1)
        ax.text(x, y, t, fontsize=9, ha="left", va="top", family="DejaVu Sans",
                bbox=dict(boxstyle="round,pad=0.15", fc=colour, ec="none"),
                transform=ax.transAxes)
        x += max(step, 0.011 * (len(t) + 1))
        if x > 0.95:
            x, y = 0.0, y - 0.32
    ax.set_title(f"{title}   P(injection)={e['score']:.3f}, all-masked baseline={e['masked_baseline']:.3f}",
                 fontsize=9, loc="left")
    if own and name:
        FIG.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(FIG / f"{name}.png", dpi=DPI, bbox_inches="tight")
    return ax


# ------------------------------------------------------------------ LIME
def lime_explain(bundle, text, num_features=8, num_samples=500, seed=42):
    from lime.lime_text import LimeTextExplainer
    p = prob_fn(bundle)
    proba = lambda texts: np.column_stack([1 - p(texts), p(texts)])
    exp = LimeTextExplainer(class_names=["safe", "injection"], random_state=seed).explain_instance(
        text, proba, num_features=num_features, num_samples=num_samples)
    return exp.as_list()


# ------------------------------------------------------------------ attention
def cls_attention(bundle, text):
    """Last-layer attention from the first token, averaged over heads."""
    import torch
    m, k, _ = bundle
    enc = k(text, return_tensors="pt", truncation=True, max_length=128).to(next(m.parameters()).device)
    # the default SDPA kernel never materialises attention weights; eager does,
    # with identical logits. Switch only for this call, then restore.
    prev = m.config._attn_implementation
    m.set_attn_implementation("eager")
    try:
        with torch.no_grad():
            out = m(**enc, output_attentions=True)
    finally:
        m.set_attn_implementation(prev)
    att = out.attentions[-1][0].mean(0)[0].float().cpu().numpy()
    toks = k.convert_ids_to_tokens(enc["input_ids"][0])
    return pd.DataFrame({"token": toks, "attention": att})


# ------------------------------------------------------------------ 3.15.4 audit
def length_label_correlation(df):
    rows = []
    for name, g in [("all sources", df)] + list(df.groupby("source")):
        if g.label.nunique() == 2:
            r, p = pointbiserialr(g.label, g.text.str.split().str.len())
            rows.append({"subset": name, "n": len(g), "r_length_label": r, "p_value": p})
        else:
            rows.append({"subset": name, "n": len(g), "r_length_label": np.nan, "p_value": np.nan,
                         "note": "single class"})
    return pd.DataFrame(rows)


def source_probe(df, seed=42):
    """Can the SOURCE be predicted from the text? If easily, each dataset has a
    recognisable style a detector could learn instead of the attack itself."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(df.text)
    pred = cross_val_predict(LogisticRegression(max_iter=3000, class_weight="balanced"), X, df.source,
                             cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    return balanced_accuracy_score(df.source, pred), 1 / df.source.nunique()


def neutral_prior(bundles):
    """Score each model gives to content-free or trivially harmless input."""
    return pd.DataFrame({name: prob_fn(b)(NEUTRAL) for name, b in bundles.items()},
                        index=[repr(t) for t in NEUTRAL])


def script_shortcut(pred_file, d4):
    """D4 holds each prompt in en / bn / bn-en with the same group_id. For the
    harmless prompts, compare the model's score on the English version with the
    Bangla versions of the SAME prompt -- the content is fixed, only script changes."""
    p = pd.read_csv(pred_file)
    p = p[p.condition == "E4_codemix"].merge(d4[["id", "group_id"]], on="id", suffixes=("", "_d4"))
    p = p[p.label == 0]
    w = p.pivot_table(index="group_id_d4", columns="language", values="score")
    rows = []
    for lang in ["bn", "bn-en"]:
        pair = w[["en", lang]].dropna()
        stat = wilcoxon(pair[lang], pair["en"]) if len(pair) and (pair[lang] != pair["en"]).any() else None
        rows.append({"comparison": f"{lang} vs en", "pairs": len(pair),
                     "mean_score_en": pair["en"].mean(), f"mean_score_other": pair[lang].mean(),
                     "mean_shift": (pair[lang] - pair["en"]).mean(),
                     "p_value": stat.pvalue if stat else np.nan})
    return pd.DataFrame(rows)
