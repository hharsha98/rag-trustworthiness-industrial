# HAGRID calibration -- sequel to experiment 11, second independent benchmark for `Config.support_threshold`

**Contamination, stated plainly.** The NLI backbone (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) is fine-tuned on MNLI, FEVER and ANLI. FEVER is built from Wikipedia, and HAGRID's `quotes` are Wikipedia passages (via MIRACL). So, unlike experiment 11's AttrEval-GenSearch (live search-engine output, chosen precisely because it sits outside that training distribution), **this benchmark is not cleanly out-of-distribution for the scoring model.** Consequence: an absolute ROC-AUC/PR-AUC measured here may be optimistic relative to a genuinely held-out benchmark. This is a lesser problem for *threshold calibration* specifically -- a relative question about where the decision boundary sits, which is less sensitive to a uniform bias in P_entailment than an absolute capability claim is -- but it still weakens the strength of any recommendation drawn from this dataset alone, and is not waved away below.

**Task mapping.** For every labelled sentence with >=1 resolvable citation marker: `claim = sentence.text`, cited passages = `quotes[]` whose `idx` the sentence's markers reference, statistic = `max` over cited passages of `P_entailment(quote, claim)`, ground truth = `sentence.attributable`. This generalizes experiment 11's single-citation mapping to HAGRID's multi-citation sentences; Analysis 6 below isolates the single-citation subset, where it collapses to exactly experiment 11's per-citation case.

## 1. Class balance and citation-parsing counts

2388 total sentences across all rows/answers. 2150 carry an `attributable` label (238 do not and are dropped). Of the labelled sentences: 73 have no parsable citation marker, 1 have marker(s) but none resolve to a real quote in that row (dangling citations), leaving **2076 usable (claim, cited-passages, label) items** -- n=1590 attributable=1 / 486 attributable=0 (0.766 positive rate).

| | count |
|---|---:|
| Total sentences | 2388 |
| Labelled (`attributable` present) | 2150 |
| Unlabelled (dropped) | 238 |
| Labelled, zero parsable citation markers (dropped) | 73 |
| Labelled, markers present but all dangling (dropped) | 1 |
| **Usable items** | **2076** |

## 2. Threshold-free discrimination (statistic vs. binary label)

| Metric | Point (95% CI, 10,000-sample bootstrap) |
|---|---|
| ROC-AUC | 0.780 (0.754, 0.805) |
| PR-AUC | 0.893 (0.877, 0.910) |

## 3. At the shipped threshold (tau = 0.5)

| | Predicted supported | Predicted not supported |
|---|---:|---:|
| **Actually attributable** | TP=1301 | FN=289 |
| **Actually not attributable** | FP=182 | TN=304 |

| Precision | Recall | F1 | Accuracy |
|---:|---:|---:|---:|
| 0.877 | 0.818 | 0.847 | 0.773 |

## 4. Threshold sweep

F1 as a function of tau over a 101-point grid on [0, 1].

| | tau | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Shipped default | 0.50 | 0.847 | 0.877 | 0.818 |
| HAGRID empirical best | 0.03 | 0.897 | 0.841 | 0.960 |

## 5. Cross-dataset agreement -- the headline

| Dataset | Own optimal tau | F1 at own optimum | F1 at 0.5 |
|---|---:|---:|---:|
| HAGRID (this experiment, n=2076) | 0.03 | 0.897 | 0.847 |
| AttrEval-GenSearch (experiment 11, n=242) | 0.21 | 0.715 | 0.691 |

**Transfer, both directions:**

| | tau used | F1 |
|---|---:|---:|
| HAGRID at AttrEval's optimal tau (0.21) | 0.21 | 0.875 (vs. 0.897 at its own optimum) |
| AttrEval at HAGRID's optimal tau (0.03) | 0.03 | 0.632 (vs. 0.715 at its own optimum) |

Gap between the two datasets' optimal tau: **0.18** (HAGRID 0.03 vs. AttrEval 0.21).

**Flatness of the F1 curve near the optimum (HAGRID, tau in [0.2, 0.5]):** F1 ranges from 0.847 to 0.875 across that span (0.029 total movement). This is a fairly flat stretch -- a shipped default anywhere in this range costs little F1 relative to the empirical optimum, which weakens the case for retuning to a precise value even where the datasets agree on direction.

**The F1 optima diverge (0.03 vs 0.21) -- but that is an artefact of the objective, not a disagreement about the metric.** F1 ignores true negatives, so the F1-optimal threshold moves with class prevalence, and these two benchmarks have opposite skew: AttrEval is 33.5% positive, HAGRID 76.6%. On a mostly-positive set, liberal prediction is rewarded -- which is why HAGRID's 'optimum' at tau=0.03 (F1 0.897) sits barely above the degenerate accept-everything classifier at tau=0 (F1 0.867, whose precision 0.766 is exactly the base rate). Concluding 'the datasets disagree, therefore do not retune' would mistake that artefact for a property of the score.

**Under a prevalence-independent criterion the two benchmarks substantially agree.** Youden's J (TPR - FPR, which weights sensitivity and specificity equally regardless of base rate) puts the optimum at tau = 0.20 on AttrEval (J = 0.567) and tau = 0.29 on HAGRID (J = 0.456) -- both well below the shipped 0.5. Consistently with that, tau ~= 0.21 improves F1 on BOTH datasets relative to 0.5 (AttrEval 0.691 -> 0.715; HAGRID 0.847 -> 0.875). So the score's discriminative boundary really does sit nearer 0.2-0.3 than 0.5.

**Recommendation: KEEP `Config.support_threshold = 0.5`, and document this finding rather than act on it.** The reason is not that the evidence is absent -- it is that F1 and J are both symmetric objectives, and this threshold does not sit in a symmetric problem. Attribution feeds a *trustworthiness* score. A false "supported" verdict credits a citation that does not hold up and inflates reported trust; a false "unsupported" verdict deflates it. For a metric whose entire purpose is to avoid overstating how well-grounded an answer is, the conservative error is the second one, and 0.5 buys precision (0.877 here vs 0.869 at tau=0.21) at recall's expense deliberately. This is the same principle experiments/09 applied to `retrieval_gate`, where the objective was chosen from the error asymmetry rather than from F1 -- there it argued for higher sensitivity, here it argues for higher precision, because the downstream consequences differ. The F1 gains available from moving are also small (+0.02 to +0.03) and come entirely from recall.

A deployment that would rather catch more true citations than avoid crediting weak ones should set `support_threshold` to about 0.21; that is now a documented, evidence-backed option rather than an untested guess. The default stays conservative.

## 6. Single-citation vs. multi-citation sentences

The `max`-over-cited-quotes aggregation is a no-op for single-citation sentences (it reduces to plain `P_entailment(quote, claim)`, exactly experiment 11's mapping) and does real work only for multi-citation sentences. Reported separately so the aggregation's effect is visible rather than averaged away.

| Subset | n | n positive | ROC-AUC (95% CI) | F1 @ tau=0.5 |
|---|---:|---:|---|---:|
| Single-citation (n_cited=1) | 1630 | 1234 | 0.837 (0.811, 0.860) | 0.856 |
| Multi-citation (n_cited>1) | 446 | 356 | 0.532 (0.463, 0.599) | 0.816 |

## Honesty notes

- Contamination (see top of this report): HAGRID's Wikipedia/MIRACL text overlaps the FEVER training data of the NLI backbone in source and genre, unlike experiment 11's AttrEval-GenSearch. Absolute AUC/PR-AUC numbers above should be read with that in mind; the cross-dataset agreement question in Section 5 is comparatively more robust to it, but not immune.
- 238 sentences have no `attributable` label and are excluded entirely (not counted as either class).
- 74 labelled sentences carry no usable citation (no marker, or marker(s) resolving to nothing in that row's quotes) and are excluded -- these are real gaps in the source data, not an artifact of the parser.
- If the two benchmarks disagree (Section 5), that disagreement is the finding, not a reason to pick the more convenient number.

## Artefacts

- `hagrid_calibration.json` -- full numeric results
- `hagrid_calibration.png` -- ROC curve, F1-vs-tau sweep (HAGRID overlaid with AttrEval, experiment 11), and score distribution by label
