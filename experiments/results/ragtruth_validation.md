# RAGTruth validation -- METRICS.md Part III, independent benchmark

**2,700 independently, human-annotated RAG outputs from RAGTruth** (Wu et al., 2024): Yelp reviews (Data2txt), CNN/DailyMail (Summary) and MARCO passages (QA).

**Why this benchmark.** The default NLI backbone (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) is fine-tuned on MNLI, FEVER and ANLI. Evaluating a faithfulness metric against a benchmark built from any of those corpora would be circular -- the model would already have seen that exact style of premise/hypothesis pair as a training signal. RAGTruth is built from Yelp/CNN-DailyMail/MARCO text, none of which are MNLI, FEVER or ANLI, so it sits outside the NLI model's training distribution. That is the entire reason a result measured here is informative rather than circular.

Sampled **900 of 2700** items (seed=0), stratified evenly across `task_type`.

## Observed class balance

| Scope | n | hallucinated | rate |
|---|---:|---:|---:|
| Sampled (pooled) | 900 | 295 | 0.328 |
| Sampled -- Summary | 300 | 64 | 0.213 |
| Sampled -- Data2txt | 300 | 191 | 0.637 |
| Sampled -- QA | 300 | 40 | 0.133 |
| Full RAGTruth test split (pooled) | 2700 | 943 | 0.349 |
| Full -- Summary | 900 | 204 | 0.227 |
| Full -- Data2txt | 900 | 579 | 0.643 |
| Full -- QA | 900 | 160 | 0.178 |

## Experiment A -- how well does faithfulness detect real hallucinations?

Decision statistic is *negated* faithfulness (lower faithfulness -> more likely hallucinated), via `faithfulness.faithfulness(split_claims(output), chunk_passages([context]), nli).score`.

| Scope | n | ROC-AUC (95% CI) | PR-AUC (95% CI) |
|---|---:|---|---|
| Pooled | 900 | 0.683 (0.648, 0.718) | 0.425 (0.396, 0.465) |
| Summary | 300 | 0.630 (0.557, 0.700) | 0.272 (0.233, 0.346) |
| Data2txt | 300 | 0.560 (0.492, 0.627) | 0.663 (0.616, 0.725) |
| QA | 300 | 0.603 (0.513, 0.692) | 0.169 (0.140, 0.237) |

## Experiment B -- is the conflict/unsupported split real, or decorative?

METRICS.md reports faithfulness `F` (mean max entailment) and contradiction rate `kappa` (mean max contradiction) separately, arguing refuted and unsupported claims are different failure modes. RAGTruth's two label types map directly: `evident_conflict` ~ refuted, `baseless_info` ~ unsupported. Tested here on the **244** sampled items with exactly one label type present (so the two signals are not confounded): 110 conflict-only, 134 baseless-only.

**One target, not two.** On this subset `has_baseless` is by construction `not has_conflict`, and ROC-AUC against a complemented label is exactly `1 - AUC`. Scoring both targets therefore produces four numbers containing two independent facts, and turns "kappa wins on conflict" and "(1-F) wins on baseless" into the same statement written twice. An earlier version of this experiment did exactly that and reported the two as independent corroboration; they were not. Only the `has_conflict` target is reported below.

**What this subset cannot show.** Every item here is hallucinated, so neither statistic is being asked to separate hallucinated from faithful output -- only conflict-type from baseless-type hallucination. Absolute detection ability is Experiment A's question, not this one's.

| Statistic | Target | ROC-AUC (95% CI) | Clears chance? |
|---|---|---|---|
| kappa (contradiction rate) | has_conflict | 0.600 (0.528, 0.671) | yes |
| 1 - F (negated faithfulness) | has_conflict | 0.516 (0.442, 0.589) | no -- CI includes 0.5 |

Paired permutation test (n=10000), kappa vs (1-F) on `has_conflict`: gap = **+0.0844**, **p = 0.0771**.

**The design claim is PARTIALLY SUPPORTED, and the qualification matters.** kappa does carry conflict-specific signal -- its CI clears chance, while (1-F)'s does not, so the two statistics are not interchangeable. But kappa being *better* than (1-F) at this does not reach significance (p = 0.0771 against a 0.05 threshold), so the strong form of the claim -- that kappa is demonstrably the right instrument for contradictions -- is not established by this evidence. Reporting kappa separately is defensible; claiming it is proven superior is not.

## Artefacts

- `ragtruth_validation.json` -- full numeric results
- `ragtruth_validation.png` -- ROC curves (pooled + per task_type) and Experiment B's two AUCs with CI bars
