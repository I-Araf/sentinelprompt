#!/usr/bin/env python3
"""Near-duplicate clustering -> group_id (methodology 3.5.5b-d, 3.4.5).

MinHash + LSH over 5-gram character shingles, Jaccard >= 0.80 (primary),
cross-checked against TF-IDF cosine >= 0.85 (3.5.5c). Each connected component
of the near-duplicate graph becomes one group_id, which is what 3.8 uses to
keep duplicates out of both sides of the split.

Reads data/processed/unified_v1.csv; writes data/processed/unified_v2_grouped.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer

IN = Path("data/processed/unified_v1.csv")
OUT = Path("data/processed/unified_v2_grouped.csv")

SHINGLE = 5          # character n-gram size (3.5.5b)
NUM_PERM = 128
JACCARD = 0.80       # primary threshold (3.5.5b)
COSINE = 0.85        # cross-check threshold (3.5.5c)
RANDOM_TEST_FRAC = 0.20   # for the leakage estimate in 3.4.5


def shingles(s, k=SHINGLE):
    s = str(s).lower()
    if len(s) < k:
        return {s}
    return {s[i:i + k] for i in range(len(s) - k + 1)}


class Union:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def join(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def minhash_pairs(texts):
    """Candidate pairs from LSH, then verified on exact Jaccard."""
    sh = [shingles(t) for t in texts]
    mh = []
    for s in sh:
        m = MinHash(num_perm=NUM_PERM)
        for g in s:
            m.update(g.encode("utf8"))
        mh.append(m)

    lsh = MinHashLSH(threshold=JACCARD, num_perm=NUM_PERM)
    for i, m in enumerate(mh):
        lsh.insert(str(i), m)

    pairs, cand = set(), 0
    for i, m in enumerate(mh):
        for key in lsh.query(m):
            j = int(key)
            if j <= i:
                continue
            cand += 1
            inter = len(sh[i] & sh[j])
            union = len(sh[i] | sh[j])
            if union and inter / union >= JACCARD:      # verify, don't trust LSH
                pairs.add((i, j))
    return pairs, cand


def cosine_pairs(texts):
    """Independent check (3.5.5c). Chunked so we never build a dense 6.7k^2."""
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                        min_df=2, sublinear_tf=True).fit_transform(texts)
    n = X.shape[0]
    pairs = set()
    step = 512
    for a in range(0, n, step):
        block = (X[a:a + step] @ X.T).toarray()
        rows, cols = np.where(block >= COSINE)
        for r, c in zip(rows, cols):
            i = a + r
            if c > i:
                pairs.add((i, int(c)))
    return pairs


def main():
    df = pd.read_csv(IN)
    # D4 keeps its own hand-built group_id (300 groups x 3 language variants);
    # it is fully held out, so its grouping cannot cause train/test leakage.
    bnen = df[df.source == "bnen"].copy()
    work = df[df.source != "bnen"].reset_index(drop=True).copy()
    texts = work.text.tolist()

    print(f"clustering {len(work)} non-D4 rows "
          f"(D4's {len(bnen)} rows keep their existing groups)")

    mh_pairs, cand = minhash_pairs(texts)
    print(f"  LSH candidates           : {cand}")
    print(f"  verified Jaccard >= {JACCARD} : {len(mh_pairs)} pairs")

    cos_pairs = cosine_pairs(texts)
    print(f"  TF-IDF cosine >= {COSINE}   : {len(cos_pairs)} pairs")
    both = mh_pairs & cos_pairs
    print(f"  agreement (both methods) : {len(both)} pairs"
          f"  [MinHash-only {len(mh_pairs - cos_pairs)},"
          f" cosine-only {len(cos_pairs - mh_pairs)}]")

    # MinHash/Jaccard is the primary method (3.5.5b); cosine is the cross-check
    # (3.5.5c), NOT a second source of truth. Inspecting the disagreements showed
    # cosine >= 0.85 over char n-grams admits semantically different prompts
    # ("Is the settlement building unfair?" vs "... building in Spain unfair?",
    # Jaccard 0.64), so taking the union over-merged: 36.9% of rows landed in a
    # multi-row group and the largest group swelled to 32 rows. Over-merging is
    # not free -- a group is indivisible at split time, so it destroys real
    # diversity. We therefore group on MinHash and report cosine as validation.
    uf = Union(len(work))
    for i, j in mh_pairs:
        uf.join(i, j)
    roots = [uf.find(i) for i in range(len(work))]
    work["group_id"] = [f"G_{r:05d}" for r in roots]

    out = pd.concat([work, bnen], ignore_index=True)
    out.to_csv(OUT, index=False, encoding="utf-8")

    sizes = work.group_id.value_counts()
    multi = sizes[sizes > 1]
    in_multi = int(multi.sum())
    pairs_all = mh_pairs

    print("\n===== GROUPING REPORT (non-D4) =====")
    print(f"rows                     : {len(work)}")
    print(f"groups                   : {work.group_id.nunique()}")
    print(f"groups with >1 row       : {len(multi)}")
    print(f"rows inside such groups  : {in_multi} ({in_multi/len(work)*100:.1f}%)")
    print(f"largest group            : {int(sizes.max())} rows")
    print(f"size distribution        : {sizes.value_counts().sort_index().head(8).to_dict()}")

    # cross-source duplicates = leakage between datasets
    cross = sum(1 for i, j in pairs_all if work.source[i] != work.source[j])
    print(f"cross-source pairs       : {cross}")
    if cross:
        sp = {}
        for i, j in pairs_all:
            if work.source[i] != work.source[j]:
                k = tuple(sorted((work.source[i], work.source[j])))
                sp[k] = sp.get(k, 0) + 1
        print(f"  by source pair         : {sp}")

    # 3.4.5 / 3.8.4: how much leakage a naive random split would have caused
    p = 2 * (1 - RANDOM_TEST_FRAC) * RANDOM_TEST_FRAC
    print(f"\nexpected leakage under a random {int((1-RANDOM_TEST_FRAC)*100)}/"
          f"{int(RANDOM_TEST_FRAC*100)} split:")
    print(f"  {len(pairs_all)} near-duplicate pairs x {p:.2f} "
          f"= ~{len(pairs_all)*p:.0f} pairs straddling train/test")
    dis = OUT.parent / "nearvdup_disagreements.csv"
    rows = [{"method": "cosine_only" if pr in (cos_pairs - mh_pairs) else "minhash_only",
             "text_a": work.text[i], "text_b": work.text[j],
             "source_a": work.source[i], "source_b": work.source[j]}
            for pr in ((cos_pairs - mh_pairs) | (mh_pairs - cos_pairs))
            for i, j in [pr]]
    pd.DataFrame(rows).to_csv(dis, index=False, encoding="utf-8")
    print(f"\ndisagreeing pairs saved for manual review -> {dis}")
    print(f"total rows out: {len(out)}  ->  {OUT}")


if __name__ == "__main__":
    main()
