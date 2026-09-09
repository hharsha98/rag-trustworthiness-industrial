# SEAHORSE validation -- METRICS.md Part II.4, conciseness

**Do these numbers show conciseness measuring redundancy, or just summary length?** Read the length-confound section (Section 2) before the headline AUC -- `conciseness` returns exactly 1.0 for any summary with fewer than 2 claims, so short summaries score perfectly *by construction*, and SEAHORSE summaries are short.

**Why this benchmark.** `conciseness` uses only the embedding model (`sentence-transformers/msmarco-distilbert-base-v4`, trained on MS MARCO query/passage pairs) -- it never calls the NLI model. The en-US slice of SEAHORSE used here draws from `wiki_lingua_english_en`, `xlsum_english` and `xsum` (verified from `gem_id` prefixes at load time), none of which is MS MARCO retrieval text, so this benchmark sits outside the embedder's training distribution.

## Data

- 101652 total SEAHORSE rows; 25053 with `worker_lang == 'en-US'`.
- Pivoted to **4355** distinct summaries (key: gem_id + model + summary).
- **4143** have Q2 (repetition) answered, **4140** have Q6 (concise-representation) answered, **4139** have BOTH (the paired subset used in Section 3).
- Analysis population (Section 1/2): **4143** of 4143 available Q2-answered summaries (all available -- no sampling needed).

## Section 1 -- primary: does C predict Q2 ("free of repeated information")?

Orientation: Q2 = "Yes" means the summary is NOT redundant, so a redundancy metric should predict "Yes" with a HIGH C. No sign-flipping here -- unlike the length baselines below, this direction is the metric's own claim, not chosen post hoc.

| n | Yes | No | rate(Yes) | ROC-AUC (95% CI) |
|---:|---:|---:|---:|---|
| 4143 | 3708 | 435 | 0.895 | 0.647 (0.618, 0.675) |

## Section 2 -- the length confound

**68.5%** of the analysis population has fewer than 2 claims, i.e. `conciseness` returns exactly 1.0 for them by construction, independent of content.

Two pure length baselines, each given its best-fitting sign (`best_sign`, chosen from a single point-AUC via the `roc_auc(-s,y) == 1-roc_auc(s,y)` identity, so only one direction is ever reported per baseline):

| Baseline | sign | ROC-AUC (95% CI) |
|---|---:|---|
| n_claims | -1 | 0.609 (0.584, 0.636) |
| n_words | +1 | 0.504 (0.474, 0.534) |
| **C (conciseness)** | (n/a) | 0.647 (0.618, 0.675) |

Paired permutation test, C vs the stronger baseline (**n_claims**): diff = **+0.0372**, **p = 0.1346** (n=10000 permutations).

**C is UNDEFINED-BY-CONSTRUCTION on 68.5% of this population (<2 claims => C == 1.0 exactly), and on that majority it is a constant carrying no information. Across the full population it therefore cannot beat the n_claims baseline (p=0.1346). But restricted to the 1307 summaries where C is actually computed (>=2 claims), C reaches 0.820 against 0.566 for the best length baseline -- a margin of +0.254 at p=0.0001. So the metric is not a dressed-up sentence count; it is a valid redundancy signal that is simply inapplicable to short answers. The design defect this exposes is returning 1.0 (a PERFECT score, which propagates into T_geom) where the honest answer is 'undefined'.**

**Restricted to the 1307 summaries with >= 2 claims** -- the only ones where C is actually computed rather than returned as the constant 1.0. The length baselines are rerun here too: comparing C against them on the full population is not meaningful, because there C is a constant across 68.5% of the rows and is therefore nearly the same variable as the claim count.

Class balance: 1092 Yes / 215 No (rate 0.836).

| Statistic | ROC-AUC (95% CI) |
|---|---|
| **C (conciseness)** | 0.820 (0.783, 0.853) |
| n_claims baseline (sign -1) | 0.566 (0.525, 0.606) |
| n_words baseline (sign +1) | 0.538 (0.495, 0.581) |

Paired permutation test, C vs the better baseline (n_claims): diff = **+0.2538**, **p = 0.0001** (10000 permutations). C significantly better: **True**.

**Design implication.** Where the metric is defined it is a genuine redundancy signal, not a proxy for length. The problem is what it does where it is NOT defined: `conciseness` returns **1.0 -- a perfect score** -- for any answer with fewer than 2 claims, and that value propagates into `T_geom` as though redundancy had been measured and found absent. The honest return there is 'undefined', not 'perfect'. This is structurally the same error as the rejected (1+cos)/2 relevance mapping's floor, which this repository documents as a defect.

## Section 3 -- discriminant validity: C vs Q2 (repetition) vs Q6 (concise-representation)

On the **4139** summaries with both Q2 and Q6 answered. Q2 and Q6 are independently-collected judgments about different properties, not complements of each other on this subset, so scoring C against both is legitimate (unlike experiment 10's Experiment B, where the two targets WERE exact complements and reporting both would have printed one fact twice).

| Target | n Yes | n No | ROC-AUC (95% CI) |
|---|---:|---:|---|
| Q2 (repetition) | 3704 | 435 | 0.647 (0.618, 0.675) |
| Q6 (concise-representation) | 1566 | 2573 | 0.514 (0.499, 0.528) |

Paired permutation test (swap-labels construction; n=10000), AUC(C,Q2) - AUC(C,Q6): diff = **+0.1331**, **p = 0.0001**.

**METRICS.md's "penalises padding, not length" framing is SUPPORTED on the discriminant test**: C tracks Q2 (repetition) more strongly than Q6 (concise-representation), and the gap is paired-permutation-significant.

## Bottom line

1. Primary AUC(C, Q2) = 0.647 (0.618, 0.675).
2. Length confound: C is UNDEFINED-BY-CONSTRUCTION on 68.5% of this population (<2 claims => C == 1.0 exactly), and on that majority it is a constant carrying no information. Across the full population it therefore cannot beat the n_claims baseline (p=0.1346). But restricted to the 1307 summaries where C is actually computed (>=2 claims), C reaches 0.820 against 0.566 for the best length baseline -- a margin of +0.254 at p=0.0001. So the metric is not a dressed-up sentence count; it is a valid redundancy signal that is simply inapplicable to short answers. The design defect this exposes is returning 1.0 (a PERFECT score, which propagates into T_geom) where the honest answer is 'undefined'.
3. Discriminant validity verdict: **SUPPORTED**.

## Artefacts

- `seahorse_validation.json` -- full numeric results
- `seahorse_validation.png` -- ROC curves (C vs length baselines), C distribution by Q2 answer, and AUC(C,Q2) vs AUC(C,Q6) with CI bars
