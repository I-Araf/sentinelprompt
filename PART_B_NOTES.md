# Part B — Classical Track (Shuvo)

Your part in one line: **"How far old-style models can go, how fast they are, and where they break."**

## Files

| File | What it does |
|---|---|
| `src/features_tfidf.py` | TF-IDF features — word (1,3)-grams + char_wb (3,5)-grams, as in methodology §3.9.1 |
| `src/baselines.py` | Tuning, threshold, evaluation and saving for the three models |
| `src/explain_classical.py` | Which words each model relies on (coefficients and log-probability ratios) |
| `notebooks/03_baselines.ipynb` | The whole workflow step by step, with explanations and outputs |

Running it: open the notebook and every output is already there. To run it again, from the project root:
`python src/baselines.py` (`src/split.py` and `src/adversarial.py` must have run first).

What gets saved:
- models → `models/classical/*.joblib` (not in git; re-created by running the script)
- TF-IDF matrices → `data/processed/features/X_*.npz`, labels → `y_*.npy`
- figures → `results/figures/` (300 dpi), tables → `results/tables/`

## The three models at a glance

| | What it is | How it was tuned | What it picked |
|---|---|---|---|
| **LogReg** | Adds up a weight per word and turns the sum into a probability with a sigmoid | C × penalty × class_weight = 20 combinations | C=100, L2, class_weight=None |
| **Naive Bayes** | Bayes' theorem, assuming words are independent of each other | alpha = 4 values | alpha=0.1 |
| **Linear SVM** | Finds the widest possible gap between the two classes | C = 3 values | C=10 |

**How tuning works:** GridSearchCV, 5-fold **grouped** cross-validation, judged by macro-F1.
Why "grouped": so that one near-duplicate group never ends up on both sides even during tuning.
Otherwise leakage would creep in through model selection itself (methodology §3.11.2).

**Threshold:** 0.5 was not assumed. The threshold that gives the highest macro-F1 on the validation
set was chosen and then applied to the test sets (§3.12.3). The test sets are used only once, for the
final report.

## Results (macro-F1)

| Condition | LogReg | NB | SVM |
|---|---|---|---|
| **E1** Test-Clean | **0.911** | 0.866 | 0.889 |
| **E2** Paraphrase | 0.850 | 0.887 | 0.850 |
| **E3** Obfuscated | 0.862 | 0.847 | 0.855 |
| **E4** Bangla–English | 0.642 | **0.717** | 0.612 |

E5 and FP have no macro-F1 (`NaN`) because each contains only one class — covered separately below.

## 🔴 The real viva question — E5 and FP must be read together

| | LogReg | NB | SVM |
|---|---|---|---|
| E5 Lakera — recall (attacks caught) | 0.958 | **0.992** | 0.934 |
| FP PromptBench — false-alarm rate | 0.79 | **0.96** | 0.70 |

E5 looks excellent — NB catches 99% of attacks. But look right next to it: **NB also calls 96% of
PromptBench's harmless task prompts an "attack".**

The reason: the model learned **"English imperative tone = attack"**. Lakera's attacks are written as
commands, so it catches them. PromptBench's task prompts are also written as commands
(*"Examine the sentence and decide if its grammar is 'Acceptable'"*), so it flags them by mistake.
**The model has not learned the difference between "an instruction to the model" and "an instruction
that hijacks the model".**

That is why, when NB's E5 accuracy reaches 99.2%, the notebook shows a **warning** (it does so whenever
accuracy exceeds 97%) and runs a leakage check. Result: no leakage — the set contains only injections,
so accuracy is really just recall. The high number reflects how the set is built, not skill.

**E5x** (E5 + FP pooled) puts both into one number: ROC-AUC LogReg 0.810, NB 0.866, SVM 0.791.

## The second result — it breaks down on Bangla

From E1 to E4, macro-F1 drops: LogReg 0.911 → 0.642, SVM 0.889 → 0.612.
The worst part is **recall**: LogReg catches only **38%** of Bangla attacks, SVM **34%**.
A model trained on English misses most Bangla attacks — this is RQ3.

Interestingly, the simplest model, **Naive Bayes, does best here** (0.717).

## Which words the models actually rely on (§3.15.3)

`src/explain_classical.py` and the "What the classical models rely on" section of the notebook.
A linear model can be read completely — every feature has one weight. For LogReg and SVM that weight
is read directly; for Naive Bayes the methodology's measure is used:
log P(word | Injection) − log P(word | Safe).

**LogReg pulls most strongly towards "safe":** `germany`, `deutschland`, `europa`, `welche`, `can you`.
**Towards "attack":** `you`, `fuck`, `hate trump`, `hate`, `before`.

This may be the best viva question of all: **the model learned topics, not attack techniques.** A large
share of deepset's harmless prompts are questions about German/European politics, so the rule "talking
about Germany = safe" works on the training data — but it has nothing to do with prompt injection. This is
called a **shortcut**, or a spurious correlation. SVM shows the same picture.

**Naive Bayes is far more meaningful by comparison:** `forget`, `everything`, `vergiss` (German for
"forget"), `alle`, `ignore`, `instructions` — the actual language of an override attack. That may be why
NB does best on Bangla (E4).

In all three models, about **two-thirds of the total weight sits in character n-grams** (67–69%) — this
is why they hold up under obfuscation.

## Questions you may be asked

**a) Why character n-grams?** With word n-grams alone, `1gn0r3` or `i g n o r e` would break the whole
word. Character n-grams capture small pieces, so they survive. This is why all three models stay close to
0.85 on E3.

**b) Why did LogReg pick `class_weight=None` when the data is imbalanced?** The grid search tried both,
and None gave the higher CV macro-F1. The imbalance is handled by the threshold tuned on validation (0.707).

**c) Why does NB have no class_weight?** scikit-learn's MultinomialNB does not support it. The tuned
threshold (0.851) does that job.

**d) SVM's threshold is 0.283 — can a probability be that low?** SVM does not output a probability; it
outputs the distance from the boundary (the decision value). 0.283 is a cut-off on that distance, not a
probability.

**e) Why only 462 training rows?** Lakera and PromptBench are both held-out test sets, so only deepset is
used for training. This is a limitation and must be stated in the paper.

**f) How fast is it (E7)?** All three models decide in ~0.08 ms per prompt, and each model file is
~1 MB. The comparison with the transformers is in `00_overview`.
