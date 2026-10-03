#!/usr/bin/env python3
"""Build SentinelPrompt-BnEn v3 from the v2 draft and the human annotation, and
re-score E4 against it (methodology 3.6.3, 3.7.3).

Human evidence used, and nothing else:
  - pilot (annotation/pilot/, 50 rows, three annotators): majority label of the
    annotators who answered; a tie keeps the draft label;
  - sample (annotation/sample/, 150 rows, two independent annotators): where both
    give the same label it is verified; where they differ the draft label is kept
    and the row is marked as disagreeing. The sheet in annotation/sample/excluded
    is not read.
Every other row stays as the LLM draft, unverified.

The v2 draft is read-only. Models are not retrained: D4 is test-only, so E4 is
re-scored from the saved predictions with the v3 labels.

Writes:
  data/bnen/SentinelPrompt-BnEn_v3.csv
  results/tables/d4_verification_summary.csv
  results/tables/e4_relabel_sensitivity.csv
"""
from pathlib import Path

import pandas as pd

import evaluate as ev

DRAFT = Path("data/bnen/SentinelPrompt-BnEn_draft_v2.csv")
V3 = Path("data/bnen/SentinelPrompt-BnEn_v3.csv")
PILOT = Path("annotation/pilot")
SAMPLE = Path("annotation/sample")
TAB = Path("results/tables")
SPELLING = {"0.0": "0", "1.0": "1", "encoded": "encoding",
            "playload_splitting": "payload_splitting",
            "platload_splitting": "payload_splitting"}


def read_sheet(path):
    d = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    d.columns = [c.strip() for c in d.columns]
    d = d[[c for c in d.columns if c and not c.startswith("Unnamed")]]
    d = d[d["id"].str.strip() != ""].copy()
    for c in ["ann_label", "ann_attack_type"]:
        d[c] = d[c].str.strip().str.lower().replace(SPELLING)
    return d


def annotator_name(d):
    return d["annotator"].iloc[0].strip()


def pilot_votes():
    """id -> (majority label, majority type, names) for the pilot rows."""
    sheets = [read_sheet(p) for p in sorted(PILOT.glob("pilot_*.csv"))]
    out = {}
    for rid in sheets[0]["id"]:
        votes = [(s.set_index("id").loc[rid], annotator_name(s)) for s in sheets]
        votes = [(r, n) for r, n in votes if r["ann_label"] in ("0", "1")]
        labels = pd.Series([r["ann_label"] for r, _ in votes]).value_counts()
        if len(labels) > 1 and labels.iloc[0] == labels.iloc[1]:
            out[rid] = (None, None, "")
            continue
        lab = labels.index[0]
        agree = [(r, n) for r, n in votes if r["ann_label"] == lab]
        types = pd.Series([r["ann_attack_type"] for r, _ in agree]).value_counts()
        typ = types.index[0] if len(types) == 1 or types.iloc[0] > types.iloc[1] else None
        out[rid] = (lab, typ, "; ".join(n for _, n in agree))
    return out


def sample_votes():
    """id -> (label or None if the two disagree, type or None, names)."""
    sheets = [read_sheet(p) for p in sorted(SAMPLE.glob("sample_*.csv"))]
    assert len(sheets) == 2, "expected exactly two independent sample sheets"
    a, b = (s.set_index("id") for s in sheets)
    names = f"{annotator_name(sheets[0])}; {annotator_name(sheets[1])}"
    out = {}
    for rid in a.index:
        la, lb = a.at[rid, "ann_label"], b.at[rid, "ann_label"]
        ta, tb = a.at[rid, "ann_attack_type"], b.at[rid, "ann_attack_type"]
        if la == lb:
            out[rid] = (la, ta if ta == tb else None, names)
        else:
            out[rid] = (None, None, "")
    return out


def build_v3():
    d = pd.read_csv(DRAFT, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    d["label_draft"], d["attack_type_draft"] = d["label"], d["attack_type"]
    d["verification"] = "unverified"
    pil, sam = pilot_votes(), sample_votes()
    for i, r in d.iterrows():
        if r["id"] in pil:
            lab, typ, who = pil[r["id"]]
            status = "pilot_majority" if lab else "pilot_tie"
        elif r["id"] in sam:
            lab, typ, who = sam[r["id"]]
            status = "sample_agree" if lab else "sample_disagree"
        else:
            continue
        d.at[i, "verification"] = status
        if not lab:
            continue
        d.at[i, "verified_by"] = who
        if lab != r["label"]:
            d.at[i, "label"] = lab
            d.at[i, "attack_type"] = "none" if lab == "0" else (typ or r["attack_type"])
            d.at[i, "notes"] = (r["notes"] + " | " if r["notes"] else "") + \
                f"label changed {r['label']}->{lab} by human annotation ({status})"
        elif lab == "1" and typ and typ != r["attack_type"]:
            d.at[i, "attack_type"] = typ
            d.at[i, "notes"] = (r["notes"] + " | " if r["notes"] else "") + \
                f"attack_type changed {r['attack_type']}->{typ} ({status})"
    d.to_csv(V3, index=False, encoding="utf-8-sig")
    return d


def summarise(d):
    rows = []
    for status, g in d.groupby("verification"):
        rows.append({"verification": status, "rows": len(g),
                     "label_changed": int((g.label != g.label_draft).sum()),
                     "attack_type_changed": int((g.attack_type != g.attack_type_draft).sum())})
    s = pd.DataFrame(rows)
    s.loc[len(s)] = {"verification": "total", "rows": len(d),
                     "label_changed": int((d.label != d.label_draft).sum()),
                     "attack_type_changed": int((d.attack_type != d.attack_type_draft).sum())}
    return s


def rescore(d):
    """E4 metrics per prediction file: draft labels, v3 labels, human-verified rows."""
    lab = d.set_index("id")
    verified = set(d.id[d.verification.isin(["pilot_majority", "sample_agree"])])
    rows = []
    for f in sorted(ev.PRED.glob("*.csv")):
        p = pd.read_csv(f, dtype={"id": str})
        p = p[p.condition == "E4_codemix"]
        if p.empty:
            continue
        y_draft = lab.loc[p.id, "label_draft"].astype(int).values
        y_v3 = lab.loc[p.id, "label"].astype(int).values
        keep = p.id.isin(verified).values
        for name, y, m in [("draft_v2_all", y_draft, slice(None)),
                           ("v3_all", y_v3, slice(None)),
                           ("v3_human_verified", y_v3, keep)]:
            met = ev.compute_metrics(y[m], p.pred.values[m], p.score.values[m])
            rows.append({"model_run": f.stem, "labels": name, "n": int(len(y[m])), **met})
    return pd.DataFrame(rows)


def main():
    d = build_v3()
    s = summarise(d)
    TAB.mkdir(parents=True, exist_ok=True)
    s.to_csv(TAB / "d4_verification_summary.csv", index=False)
    r = rescore(d)
    r.to_csv(TAB / "e4_relabel_sensitivity.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.to_string(index=False))
    print(f"\nlabel balance v3: {d.label.value_counts().to_dict()}")
    cols = ["model_run", "labels", "n", "accuracy", "macro_f1", "recall", "specificity"]
    print(r[cols].round(3).to_string(index=False))
    print(f"\nsaved -> {V3}, {TAB}/d4_verification_summary.csv, {TAB}/e4_relabel_sensitivity.csv")


if __name__ == "__main__":
    main()
