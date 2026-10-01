#!/usr/bin/env python3
"""Track A feature representation -- sparse lexical (methodology 3.9.1).

Word n-grams catch phrasing ("ignore all previous"); char n-grams survive the
obfuscation in 3.7.2 (leetspeak, spacing, homoglyphs) that destroys word tokens.
Both are combined, so Track A is not defenceless against O-set perturbations.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_vectorizer(word_ngrams=(1, 2), char_ngrams=(3, 5), min_df=2):
    """TF-IDF with sublinear TF and L2 norm, lowercased (3.9.1 step A3/A4).

    Lowercasing is correct for the classical track only; the transformer track
    must match its own pretrained tokenizer instead (3.5.3).
    """
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=word_ngrams,
                                 min_df=min_df, sublinear_tf=True,
                                 lowercase=True, norm="l2")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=char_ngrams,
                                 min_df=min_df, sublinear_tf=True,
                                 lowercase=True, norm="l2")),
    ])
