#!/usr/bin/env python3
"""Track A feature representation -- sparse lexical (methodology 3.9.1).

Configuration is the one in 3.9.1 step A4:
  word n-grams (1, 3)  -- multi-word signatures such as "ignore previous instructions"
  char n-grams (3, 5)  -- survive the obfuscation in 3.7.2 (leetspeak, spacing,
                          homoglyphs) that destroys word tokens
  min_df 2, max_df 0.95, max_features 50,000, sublinear TF, L2 norm.
Word and char blocks are joined (FeatureUnion) into one hybrid representation.
Matrices are kept sparse (CSR, step A5) and saved as .npz.
"""
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

FEATURES = Path("data/processed/features")


def build_vectorizer(word_ngrams=(1, 3), char_ngrams=(3, 5), min_df=2,
                     max_df=0.95, max_features=50000):
    """Lowercasing is right for the classical track only; transformers must
    match their own pretrained tokenizer instead (3.5.3)."""
    common = dict(min_df=min_df, max_df=max_df, max_features=max_features,
                  sublinear_tf=True, lowercase=True, norm="l2")
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=word_ngrams, **common)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=char_ngrams, **common)),
    ])


def save_matrices(vectorizer, splits):
    """splits: {name: DataFrame}. Writes X_<name>.npz and y_<name>.npy."""
    FEATURES.mkdir(parents=True, exist_ok=True)
    shapes = {}
    for name, df in splits.items():
        X = sparse.csr_matrix(vectorizer.transform(df.text))
        sparse.save_npz(FEATURES / f"X_{name}.npz", X)
        np.save(FEATURES / f"y_{name}.npy", df.label.to_numpy())
        shapes[name] = X.shape
    return shapes
