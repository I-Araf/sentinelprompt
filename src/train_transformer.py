#!/usr/bin/env python3
"""Transformer fine-tuning (M5 DistilBERT, M6 RoBERTa) and the zero-shot
external reference (M7 ProtectAI DeBERTa-v3), methodology 3.10.5-3.10.8, 3.11.3.

Hyperparameters follow 3.11.3: lr 2e-5, AdamW, weight decay 0.01, batch 16,
up to 5 epochs with early stopping on validation macro-F1 (patience 2), linear
warmup over the first 10% of steps, max length 128, grad-clip 1.0, dropout 0.1
(the models' default), seeds 42 / 1337 / 2024.

Two deliberate deviations, both forced by hardware rather than chosen:
  - fp16 mixed precision is a CUDA feature; on Apple MPS / CPU we train in fp32.
  - the learning-rate search {1e-5..5e-5} is not run; 2e-5 is the stated default.

Class imbalance is handled with a class-weighted cross-entropy (3.4.2).
Weights go to models/ (git-ignored); history and timings to results/tables/.
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev   # noqa: E402
import guards           # noqa: E402

MODELS = Path("models")
SEEDS = [42, 1337, 2024]
CONFIG = {"lr": 2e-5, "weight_decay": 0.01, "batch_size": 16, "max_epochs": 5,
          "patience": 2, "warmup_frac": 0.10, "max_length": 128, "grad_clip": 1.0}
CHECKPOINTS = {"distilbert": "distilbert/distilbert-base-uncased", "roberta": "FacebookAI/roberta-base"}
EXTERNAL = "protectai/deberta-v3-base-prompt-injection-v2"


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _batches(tok, texts, labels, bs, shuffle, seed=0):
    idx = list(range(len(texts)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        enc = tok([texts[k] for k in j], truncation=True, max_length=CONFIG["max_length"],
                  padding=True, return_tensors="pt")
        if labels is not None:
            enc["labels"] = torch.tensor([labels[k] for k in j])
        yield enc


@torch.no_grad()
def predict_scores(model, tok, texts, bs=64, positive_index=1):
    """P(injection) for each text."""
    dev = next(model.parameters()).device
    model.eval()
    out = []
    for enc in _batches(tok, texts, None, bs, shuffle=False):
        enc = {k: v.to(dev) for k, v in enc.items()}
        probs = torch.softmax(model(**enc).logits.float(), dim=-1)[:, positive_index]
        out.append(probs.cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def model_dir(short, seed):
    return MODELS / short / f"seed{seed}"


def train(short, seed, train_df, val_df, force=False, verbose=True):
    """Fine-tune one checkpoint with one seed. Reuses saved weights unless force."""
    out = model_dir(short, seed)
    hist_path = out / "history.json"
    if out.exists() and hist_path.exists() and not force:
        if verbose:
            print(f"[{short} seed {seed}] reusing saved weights in {out}/")
        return load(short, seed), json.loads(hist_path.read_text())

    set_seed(seed)
    dev = device()
    ckpt = CHECKPOINTS[short]
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2).to(dev)

    xtr, ytr = train_df.text.tolist(), train_df.label.astype(int).tolist()
    xva, yva = val_df.text.tolist(), val_df.label.astype(int).to_numpy()
    counts = np.bincount(ytr, minlength=2)
    weights = torch.tensor(len(ytr) / (2 * counts), dtype=torch.float, device=dev)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    steps_per_epoch = int(np.ceil(len(xtr) / CONFIG["batch_size"]))
    total = steps_per_epoch * CONFIG["max_epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                            weight_decay=CONFIG["weight_decay"])
    sched = get_linear_schedule_with_warmup(opt, int(CONFIG["warmup_frac"] * total), total)

    history, best_f1, bad, best_state = [], -1.0, 0, None
    t0 = time.perf_counter()
    for epoch in range(1, CONFIG["max_epochs"] + 1):
        model.train()
        losses = []
        for enc in _batches(tok, xtr, ytr, CONFIG["batch_size"], True, seed * 100 + epoch):
            enc = {k: v.to(dev) for k, v in enc.items()}
            labels = enc.pop("labels")
            loss = loss_fn(model(**enc).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(loss.item())
        val_scores = predict_scores(model, tok, xva)
        vf1 = f1_score(yva, (val_scores >= 0.5).astype(int), average="macro")
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_macro_f1": float(vf1)})
        if verbose:
            print(f"[{short} seed {seed}] epoch {epoch}  train loss {np.mean(losses):.4f}"
                  f"  val macro-F1 {vf1:.4f}")
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= CONFIG["patience"]:
                if verbose:
                    print(f"[{short} seed {seed}] early stop (no val gain for {bad} epochs)")
                break
    train_seconds = time.perf_counter() - t0

    model.load_state_dict(best_state)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    meta = {"checkpoint": ckpt, "seed": seed, "config": CONFIG, "history": history,
            "best_val_macro_f1": best_f1, "train_seconds": round(train_seconds, 1),
            "params": int(sum(p.numel() for p in model.parameters())),
            "size_mb": round(sum(f.stat().st_size for f in out.glob("*")) / 1e6, 1),
            "device": str(dev)}
    hist_path.write_text(json.dumps(meta, indent=2))
    return (model, tok), meta


def load(short, seed):
    out = model_dir(short, seed)
    tok = AutoTokenizer.from_pretrained(out)
    model = AutoModelForSequenceClassification.from_pretrained(out).to(device())
    return model, tok


def load_external():
    """M7: used zero-shot, never trained on our data."""
    tok = AutoTokenizer.from_pretrained(EXTERNAL)
    model = AutoModelForSequenceClassification.from_pretrained(EXTERNAL).to(device())
    labels = {v.upper(): int(k) for k, v in model.config.id2label.items()}
    return model, tok, labels.get("INJECTION", 1)


def latency_ms(model, tok, texts, positive_index=1, repeats=100):
    """Batch-1 latency, median and p95 over `repeats` prompts (3.12.4)."""
    times = []
    for t in (texts * (repeats // max(1, len(texts)) + 1))[:repeats]:
        s = time.perf_counter()
        predict_scores(model, tok, [t], bs=1, positive_index=positive_index)
        times.append((time.perf_counter() - s) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))


# --------------------------------------------------------------------------
# orchestration used by notebooks/04_transformers.ipynb
# --------------------------------------------------------------------------
KEY = ["accuracy", "precision", "recall", "specificity", "f1_injection", "macro_f1",
       "mcc", "roc_auc", "pr_auc"]


def load_data():
    tr, va = ev.load_split("train"), ev.load_split("val")
    guards.assert_no_lakera_in_train(tr)
    guards.assert_no_bnen_in_train(tr)
    guards.assert_originals_only(tr)
    return tr, va


def _free(model):
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_finetuned(short, tr, va, conditions, seeds=SEEDS, verbose=True):
    """Train + evaluate one architecture over all seeds."""
    per_seed, runs, metas = [], {}, []
    for s in seeds:
        (model, tok), meta = train(short, s, tr, va, verbose=verbose)
        sf = lambda texts, m=model, k=tok: predict_scores(m, k, texts)
        thr, vf1 = ev.tune_threshold(va.label.to_numpy(), sf(va.text.tolist()))
        run = ev.evaluate_model(short, sf, thr, conditions, tr, save_figures=(s == 42),
                                tag=f"{short}_seed{s}", verbose=verbose)
        m = run["metrics"].copy()
        m["seed"] = s
        per_seed.append(m)
        med, p95 = latency_ms(model, tok, conditions["E1_clean"].text.tolist()[:20])
        metas.append({"model": short, "seed": s, "threshold": thr, "val_macro_f1": vf1,
                      "epochs_run": len(meta["history"]), "train_seconds": meta["train_seconds"],
                      "params_million": round(meta["params"] / 1e6, 1),
                      "size_mb": meta["size_mb"], "latency_ms_median": round(med, 2),
                      "latency_ms_p95": round(p95, 2), "device": meta["device"]})
        runs[s] = run
        _free(model)
    per_seed = pd.concat(per_seed, ignore_index=True)
    agg = per_seed.groupby("condition", sort=False)[KEY].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return {"per_seed": per_seed[["seed", "condition", "threshold"] + KEY],
            "mean_std": agg[[f"{k}_{s}" for k in ["accuracy", "macro_f1", "recall", "roc_auc"]
                             for s in ["mean", "std"]]],
            "run42": runs[42], "runs": runs, "meta": metas}


def run_external(conditions, tr, verbose=True):
    """M7 zero-shot: default threshold 0.5, nothing tuned on our data."""
    model, tok, pos = load_external()
    sf = lambda texts: predict_scores(model, tok, texts, positive_index=pos)
    run = ev.evaluate_model("protectai_deberta", sf, 0.5, conditions, tr,
                            tag="protectai_deberta", verbose=verbose)
    med, p95 = latency_ms(model, tok, conditions["E1_clean"].text.tolist()[:20], positive_index=pos)
    meta = {"model": "protectai_deberta", "seed": None, "threshold": 0.5,
            "params_million": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
            "latency_ms_median": round(med, 2), "latency_ms_p95": round(p95, 2),
            "device": str(device())}
    _free(model)
    return {"run42": run, "meta": [meta]}


def save_all(summary):
    """Per-seed table, seed-mean table (used by the overview), reports, efficiency."""
    rows, means, reps, checks, metas = [], [], [], [], []
    for name, s in summary.items():
        metas += s["meta"]
        runs = s.get("runs", {None: s["run42"]})
        for seed, run in runs.items():
            m = run["metrics"].copy()
            m["seed"] = seed
            rows.append(m)
            if len(run["checks"]):
                checks.append(run["checks"].assign(seed=seed))
        reps.append(s["run42"]["reports"])
        allm = pd.concat([r["metrics"] for r in runs.values()], ignore_index=True)
        mean = allm.groupby(["model", "condition"], sort=False).mean(numeric_only=True).reset_index()
        std = allm.groupby(["model", "condition"], sort=False)[["macro_f1", "accuracy"]].std().reset_index()
        mean = mean.merge(std.rename(columns={"macro_f1": "macro_f1_std",
                                              "accuracy": "accuracy_std"}), on=["model", "condition"])
        mean["note"] = s["run42"]["metrics"]["note"].values if "note" in s["run42"]["metrics"] else np.nan
        means.append(mean)
    ev.save_table(pd.concat(rows, ignore_index=True), "metrics_transformers")
    summ = pd.concat(means, ignore_index=True)
    out = ev.tidy(summ)
    out["macro_f1_std"], out["accuracy_std"] = summ["macro_f1_std"], summ["accuracy_std"]
    out["seeds"] = [3 if m in CHECKPOINTS else 1 for m in out.model]
    ev.save_table(out, "metrics_transformers_summary")
    ev.save_table(pd.concat(reps, ignore_index=True), "classification_reports_transformers")
    eff = pd.DataFrame(metas)
    ev.save_table(eff, "transformers_efficiency_per_seed")
    ev.save_table(eff.groupby("model", sort=False).mean(numeric_only=True).reset_index()
                  .drop(columns=["seed"], errors="ignore"), "transformers_efficiency")
    if checks:
        ev.save_table(pd.concat(checks, ignore_index=True), "suspicious_accuracy_checks_transformers")
    return out
