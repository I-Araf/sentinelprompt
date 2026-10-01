#!/usr/bin/env python3
"""Error analysis and model-vs-model significance (methodology 3.15.1, 3.15.2, 3.14).

Read-only: works entirely from the per-row prediction files written by
evaluate.evaluate_model plus the split CSVs; nothing here retrains or rescores.

  3.15.1  recall by attack type (D4 is the only set with attack-type labels),
          error by prompt length, by language, by perturbation strength
  3.15.2  an inspection sheet of up to 20 false negatives and 20 false positives
          per model. The `auto_tag` column is a keyword heuristic to speed up
          reading; `human_category` is left empty for the annotators, because
          the methodology asks for these to be read and classified by people.
  3.14    pairwise McNemar between models on the same rows, Bonferroni-corrected,
          with an odds-ratio effect size

Per-row analysis uses seed 42 for the transformers, the same run the figures use.
"""
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

DATA = Path("data/processed")
PRED = Path("results/tables/predictions")
TWO_CLASS = ["E1_clean", "E2_paraphrase", "E3_obfuscated", "E4_codemix", "E5x_pooled"]
SPLIT_FILE = {"E1_clean": "test_clean", "E2_paraphrase": "test_paraphrase",
              "E3_obfuscated": "test_obfuscated", "E4_codemix": "test_codemix",
              "E5_crossdataset": "test_crossdataset", "FP_promptbench": "test_falsepos"}
SEED = 42
N_INSPECT = 20


def load_predictions():
    """model -> per-row predictions, joined with text and metadata."""
    meta = pd.concat([pd.read_csv(DATA / f"{f}.csv").assign(_cond=c)
                      for c, f in SPLIT_FILE.items() if (DATA / f"{f}.csv").exists()],
                     ignore_index=True)
    meta = meta[["id", "_cond", "text", "attack_type", "source"]].drop_duplicates(["id", "_cond"])
    out = {}
    for p in sorted(PRED.glob("*.csv")):
        name = p.stem
        if re.search(r"_seed\d+$", name):
            if not name.endswith(f"_seed{SEED}"):
                continue
            name = re.sub(r"_seed\d+$", "", name)
        d = pd.read_csv(p)
        d = d[d.condition != "E5x_pooled"].merge(
            meta, left_on=["id", "condition"], right_on=["id", "_cond"], how="left").drop(columns="_cond")
        pooled = d[d.condition.isin(["E5_crossdataset", "FP_promptbench"])].assign(condition="E5x_pooled")
        d = pd.concat([d, pooled], ignore_index=True)
        d["error"] = (d.pred != d.label).astype(int)
        d["words"] = d.text.astype(str).str.split().str.len()
        out[name] = d
    return out


def attack_type_recall(preds):
    """Recall per attack type on D4, overall and per language (3.15.1)."""
    rows = []
    for m, d in preds.items():
        e4 = d[(d.condition == "E4_codemix") & (d.label == 1)]
        for (at, lang), g in e4.groupby(["attack_type", "language"]):
            rows.append({"model": m, "attack_type": at, "language": lang,
                         "recall": 1 - g.error.mean(), "n": len(g)})
        for at, g in e4.groupby("attack_type"):
            rows.append({"model": m, "attack_type": at, "language": "all",
                         "recall": 1 - g.error.mean(), "n": len(g)})
    return pd.DataFrame(rows)


def length_errors(preds, conditions=("E1_clean", "E2_paraphrase", "E3_obfuscated", "E4_codemix")):
    """Error rate by prompt-length bucket (quartiles of the pooled two-class sets)."""
    pool = pd.concat(preds.values())
    pool = pool[pool.condition.isin(conditions)]
    edges = np.unique(np.quantile(pool.words, [0, .25, .5, .75, 1]))
    labels = [f"{int(a)}-{int(b)} words" for a, b in zip(edges[:-1], edges[1:])]
    rows = []
    for m, d in preds.items():
        d = d[d.condition.isin(conditions)].copy()
        d["bucket"] = pd.cut(d.words, edges, labels=labels, include_lowest=True)
        for b, g in d.groupby("bucket", observed=True):
            rows.append({"model": m, "length": str(b), "error_rate": g.error.mean(),
                         "fn_rate": g[g.label == 1].error.mean(),
                         "fp_rate": g[g.label == 0].error.mean(), "n": len(g)})
    return pd.DataFrame(rows), labels


def language_errors(preds):
    """FN and FP rate per language on D4 (3.15.1)."""
    rows = []
    for m, d in preds.items():
        for lang, g in d[d.condition == "E4_codemix"].groupby("language"):
            rows.append({"model": m, "language": lang,
                         "fn_rate": g[g.label == 1].error.mean(),
                         "fp_rate": g[g.label == 0].error.mean(), "n": len(g)})
    return pd.DataFrame(rows)


def perturbation_errors(preds):
    rows = []
    for m, d in preds.items():
        e3 = d[(d.condition == "E3_obfuscated") & (d.technique != "O5_base64")]
        for r, g in e3.groupby("perturbation_rate"):
            rows.append({"model": m, "perturbation_rate": r, "error_rate": g.error.mean(),
                         "fn_rate": g[g.label == 1].error.mean(),
                         "fp_rate": g[g.label == 0].error.mean()})
    return pd.DataFrame(rows)


def hardest_rows(preds, conditions=("E1_clean", "E2_paraphrase", "E4_codemix")):
    """Rows every model gets wrong -- the shared blind spots."""
    keys = ["id", "condition"]
    frames = [d[d.condition.isin(conditions)][keys + ["error"]].rename(columns={"error": m})
              for m, d in preds.items()]
    j = frames[0]
    for f in frames[1:]:
        j = j.merge(f, on=keys)
    models = list(preds)
    j["models_wrong"] = j[models].sum(axis=1)
    first = next(iter(preds.values()))
    j = j.merge(first[keys + ["text", "label", "language", "attack_type"]], on=keys)
    return j[j.models_wrong == len(models)].sort_values("condition"), models


# --------------------------------------------------------------- inspection
ENCODING = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}|(?:[0-9a-f]{2}\s?){10,}|%[0-9A-Fa-f]{2}")
IMPERATIVE = re.compile(r"^\W*(examine|analy[sz]e|determine|classify|evaluate|identify|decide|"
                        r"translate|summari[sz]e|answer|write|read|given|assess|review|tell|"
                        r"please|explain|list|describe|categori[sz]e|indicate)\b", re.I)
BENGALI = re.compile(r"[ঀ-৿]")


def auto_tag(row):
    """A first-pass heuristic, not the methodology's human categorisation."""
    t = str(row.text)
    tags = []
    if row.get("variant_type_hint") == "obfuscation" or isinstance(row.get("technique"), str) and row.technique:
        tags.append("obfuscated")
    if ENCODING.search(t):
        tags.append("encoding-like")
    if BENGALI.search(t):
        tags.append("bengali-script")
    elif row.language in ("bn", "bn-en"):
        tags.append("romanised-bangla")
    if row.label == 0 and IMPERATIVE.search(t):
        tags.append("instruction-style-benign")
    if len(t.split()) <= 4:
        tags.append("very-short")
    return ", ".join(tags) or "none"


def inspection_sheet(preds, conditions=("E1_clean", "E2_paraphrase", "E3_obfuscated", "E4_codemix"),
                     seed=SEED):
    rows = []
    for m, d in preds.items():
        d = d[d.condition.isin(conditions)]
        for kind, mask in [("FN", (d.label == 1) & (d.pred == 0)),
                           ("FP", (d.label == 0) & (d.pred == 1))]:
            s = d[mask]
            s = s.sample(min(N_INSPECT, len(s)), random_state=seed)
            for _, r in s.iterrows():
                rows.append({"model": m, "error_type": kind, "condition": r.condition, "id": r.id,
                             "language": r.language, "attack_type": r.attack_type,
                             "technique": r.get("technique", ""), "score": round(r.score, 4),
                             "text": r.text, "auto_tag": auto_tag(r), "human_category": ""})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- 3.14
def pairwise_mcnemar(preds, conditions=TWO_CLASS, alpha=0.05):
    """Exact McNemar between every model pair on the same rows; Bonferroni over
    all comparisons; odds ratio b/c (Haldane +0.5) as the effect size."""
    rows = []
    for cond in conditions:
        per = {m: d[d.condition == cond].set_index("id").sort_index() for m, d in preds.items()}
        for a, b in itertools.combinations(per, 2):
            ia, ib = per[a], per[b]
            common = ia.index.intersection(ib.index)
            if not len(common):
                continue
            ok_a = (ia.loc[common, "pred"] == ia.loc[common, "label"]).to_numpy()
            ok_b = (ib.loc[common, "pred"] == ib.loc[common, "label"]).to_numpy()
            nb_, nc_ = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
            p = binomtest(nb_, nb_ + nc_, 0.5).pvalue if nb_ + nc_ else 1.0
            rows.append({"condition": cond, "model_a": a, "model_b": b, "n": len(common),
                         "a_right_b_wrong": nb_, "a_wrong_b_right": nc_, "p_value": p,
                         "odds_ratio": (nb_ + 0.5) / (nc_ + 0.5),
                         "acc_a": ok_a.mean(), "acc_b": ok_b.mean()})
    out = pd.DataFrame(rows)
    out["alpha_bonferroni"] = alpha / len(out)
    out["significant"] = out.p_value < out.alpha_bonferroni
    return out
