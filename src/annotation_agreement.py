#!/usr/bin/env python3
"""Inter-annotator agreement for the D4 verification rounds (methodology 3.6.3, 3.7.3).

Usage, from the project root:
    python src/annotation_agreement.py <folder with filled worksheets> [tag]
    e.g.  python src/annotation_agreement.py annotation/pilot pilot

The folder holds one filled CSV per annotator (pilot_<name>.csv or
worksheet_<name>.csv). Nothing in those files is changed. Two kinds of
normalisation are applied in memory only, and both are listed in the output:
  - surrounding whitespace and letter case are ignored;
  - a fixed list of spelling slips (SPELLING below) is mapped to the intended
    category. A slip is a typing error, not a judgement, so correcting it does
    not alter what the annotator decided. A blank cell stays blank and that row
    is left out of that annotator's comparisons.

Reports, per pair of annotators: raw agreement and Cohen's kappa for the label
and for the attack type. Also, per annotator, how often the three language
versions of one prompt received the same label (the versions mean the same
thing, so this is a consistency check that needs no reference answer).

Agreement with the LLM draft is printed separately and is NOT an inter-annotator
figure: the draft is the thing being verified, not a second annotator.
"""
import itertools
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

TAB = Path("results/tables")
DRAFT = Path("data/bnen/SentinelPrompt-BnEn_draft_v2.csv")
ANN = ["ann_label", "ann_attack_type", "ann_fluency", "ann_flag"]
VALID = {
    "ann_label": {"0", "1"},
    "ann_attack_type": {"none", "direct_override", "role_play", "encoding",
                        "context_switch", "payload_splitting", "indirect"},
    "ann_fluency": {"ok", "awkward", "wrong"},
    "ann_flag": {"", "unclear", "pii", "duplicate", "broken_encoding"},
}
SPELLING = {"akward": "awkward", "awkard": "awkward",
            "playload_splitting": "payload_splitting",
            "platload_splitting": "payload_splitting",
            "encoded": "encoding", "0.0": "0", "1.0": "1"}


def load(path):
    d = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    d.columns = [c.strip() for c in d.columns]
    fixed = 0
    for c in ANN:
        raw = d[c].str.strip().str.lower()
        fixed += int(raw.isin(list(SPELLING)).sum())
        d[c] = raw.replace(SPELLING)
    return d, fixed


def validate(name, d, fixed):
    print(f"\n[{name}] {len(d)} rows, {fixed} spelling slip(s) normalised")
    for c in ["ann_label", "ann_attack_type", "ann_fluency"]:
        blank = d[d[c] == ""].row_no.tolist()
        bad = d[(d[c] != "") & (~d[c].isin(VALID[c]))]
        if blank:
            print(f"   {c}: blank in row(s) {blank}")
        if len(bad):
            print(f"   {c}: unknown value(s) {sorted(set(bad[c]))} in row(s) {bad.row_no.tolist()}")
    odd = d[((d.ann_label == "0") & (~d.ann_attack_type.isin(["none", ""]))) |
            ((d.ann_label == "1") & (d.ann_attack_type == "none"))]
    if len(odd):
        print(f"   label and attack type disagree in row(s) {odd.row_no.tolist()}")


def agreement(a, b):
    """Raw agreement and kappa on the rows both sides filled in."""
    both = (a != "") & (b != "")
    a, b = a[both], b[both]
    if not len(a):
        return 0, float("nan"), float("nan")
    k = cohen_kappa_score(a, b) if (a.nunique() > 1 or b.nunique() > 1) else float("nan")
    return len(a), float((a == b).mean()), float(k)


def consistency(d):
    """Share of prompts whose language versions all got the same label.

    A prompt with a blank label in any version is left out of the count: two
    matching versions out of three do not show the third would have matched.
    """
    complete = d.groupby("group_id").ann_label.transform(lambda s: (s != "").all())
    per = d[complete].groupby("group_id").ann_label.nunique()
    return int((per == 1).sum()), int(len(per))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    folder = Path(sys.argv[1]).expanduser()
    tag = sys.argv[2] if len(sys.argv) > 2 else "pilot"
    files = sorted(folder.glob("*.csv"))
    if len(files) < 2:
        sys.exit(f"need at least two filled worksheets in {folder}, found {len(files)}")

    sheets = {}
    for f in files:
        name = f.stem.split("_", 1)[-1]
        d, fixed = load(f)
        validate(name, d, fixed)
        sheets[name] = d.set_index("id")

    rows = []
    for x, y in itertools.combinations(sheets, 2):
        ids = sheets[x].index.intersection(sheets[y].index)
        for field in ["ann_label", "ann_attack_type"]:
            n, agree, k = agreement(sheets[x].loc[ids, field], sheets[y].loc[ids, field])
            rows.append({"annotator_a": x, "annotator_b": y, "field": field,
                         "rows_compared": n, "raw_agreement": agree, "cohen_kappa": k})
    pairs = pd.DataFrame(rows)

    cons = pd.DataFrame([{"annotator": n, "prompts_with_one_label": c[0], "prompts": c[1]}
                         for n, c in ((n, consistency(d.reset_index())) for n, d in sheets.items())])

    # every row where the annotators do not all agree on label or type
    first = next(iter(sheets.values()))
    wide = first[["group_id", "language", "text"]].copy()
    for n, d in sheets.items():
        wide[f"label_{n}"] = d.ann_label.reindex(wide.index)
        wide[f"type_{n}"] = d.ann_attack_type.reindex(wide.index)
    lab = wide[[c for c in wide.columns if c.startswith("label_")]]
    typ = wide[[c for c in wide.columns if c.startswith("type_")]]
    wide["label_differs"] = lab.nunique(axis=1) > 1
    wide["type_differs"] = typ.nunique(axis=1) > 1
    disagreements = wide[wide.label_differs | wide.type_differs].reset_index()

    draft_rows = []
    if DRAFT.exists():
        draft = pd.read_csv(DRAFT, encoding="utf-8-sig", dtype=str,
                            keep_default_na=False).set_index("id")
        for n, d in sheets.items():
            ids = d.index.intersection(draft.index)
            for field, ref in [("ann_label", "label"), ("ann_attack_type", "attack_type")]:
                cnt, agree, k = agreement(d.loc[ids, field], draft.loc[ids, ref].str.strip())
                draft_rows.append({"annotator": n, "field": field, "rows_compared": cnt,
                                   "raw_agreement": agree, "cohen_kappa": k})
    vs_draft = pd.DataFrame(draft_rows)

    TAB.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(TAB / f"annotation_{tag}_agreement.csv", index=False)
    cons.to_csv(TAB / f"annotation_{tag}_consistency.csv", index=False)
    disagreements.to_csv(TAB / f"annotation_{tag}_disagreements.csv", index=False)
    if len(vs_draft):
        vs_draft.to_csv(TAB / f"annotation_{tag}_vs_llm_draft.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n===== INTER-ANNOTATOR AGREEMENT =====")
    print(pairs.round(3).to_string(index=False))
    print("\n===== SAME LABEL ACROSS LANGUAGE VERSIONS OF ONE PROMPT =====")
    print(cons.to_string(index=False))
    print(f"\nrows where annotators differ: {len(disagreements)} "
          f"(label {int(disagreements.label_differs.sum())}, "
          f"type only {int((disagreements.type_differs & ~disagreements.label_differs).sum())})")
    if len(vs_draft):
        print("\n===== EACH ANNOTATOR vs THE LLM DRAFT (not inter-annotator agreement) =====")
        print(vs_draft.round(3).to_string(index=False))
    print(f"\nsaved -> {TAB}/annotation_{tag}_*.csv")


if __name__ == "__main__":
    main()
