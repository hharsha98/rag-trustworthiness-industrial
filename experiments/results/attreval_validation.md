# AttrEval-GenSearch validation -- METRICS.md Part II.2 (attribution), independent benchmark

Ground truth here is independent, third-party human annotation of live generative-search-engine output (New Bing), not built by construction: **AttrEval-GenSearch** (Yue et al., 2023 / `osunlp/AttrScore`), 242 (statement, cited-passage) pairs judged Attributable / Extrapolatory / Contradictory.

**Why this benchmark.** The default NLI backbone (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) is fine-tuned on MNLI, FEVER and ANLI. AttrEval-GenSearch is built from live search-engine answers annotated in 2023 across everyday-web domains (e.g. "Pet and Animal", "Economics and Finance") -- not curated NLI benchmark text -- so it sits outside that training distribution in source and genre. This is not a formal guarantee of zero overlap: MNLI/FEVER/ANLI are themselves drawn from varied web/Wikipedia text, so some topical overlap is possible in the loose sense that any two broad English corpora can overlap. What this benchmark adds is independent, third-party citation-support judgment on real generative-search output, which experiment 10's RAGTruth check does not cover (that one judges hallucination in RAG *summaries/QA/data2txt*, not explicit citation-to-claim support).

**Task-metric correspondence.** Each row maps onto `attribution(claims=[answer], citations={0: 0}, passages=[reference], nli, tau)` exactly: one claim, one passage, one citation. With a single claim and citation, `AttributionResult.precision` is 1.0 iff `P_entailment(reference, answer) >= tau` and 0.0 otherwise -- confirmed by direct assertion on 5 real rows before scoring (`verify_correspondence`), not assumed.

n = **242** (small -- confidence intervals below are correspondingly wide; treat point estimates cautiously). Class balance: **81 Attributable (supported=1)** vs. **161 not-supported (supported=0)** (128 Extrapolatory + 33 Contradictory) -- a 0.335 positive rate.

## 1. Threshold-free discrimination (P_entailment vs. binary label)

| Metric | Point (95% CI, 10,000-sample bootstrap) |
|---|---|
| ROC-AUC | 0.805 (0.737, 0.868) |
| PR-AUC | 0.741 (0.653, 0.829) |

## 2. At the shipped threshold (tau = 0.5) -- the deployment-relevant number

This is what `Config().support_threshold` actually does to citation judgments today; the AUC above says the score *could* discriminate, this says whether the *default* cutoff realizes that.

| | Predicted supported | Predicted not supported |
|---|---:|---:|
| **Actually Attributable** | TP=48 | FN=33 |
| **Actually not supported** | FP=10 | TN=151 |

| Precision | Recall | F1 | Accuracy |
|---:|---:|---:|---:|
| 0.828 | 0.593 | 0.691 | 0.822 |

## 3. Threshold sweep

F1 as a function of tau over a 101-point grid on [0, 1].

| | tau | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Shipped default | 0.50 | 0.691 | 0.828 | 0.593 |
| Empirical best | 0.21 | 0.715 | 0.771 | 0.667 |

**The optimal tau (0.21) is far from the shipped default (0.5)**, a gap of 0.29. On this benchmark, `Config.support_threshold` would need to move from 0.5 to approximately **0.21** to reach the F1 achievable here (0.715 vs. 0.691 at 0.5). This is an actionable finding about the default, not a footnote -- though note n=242 (and the class split within it) makes a single-dataset optimum a fragile target for a global default; treat it as evidence to weigh against other benchmarks (e.g. experiment 10's RAGTruth-derived thresholds for faithfulness), not a mandate to retune blind.

## 4. Three-way separation: does kappa distinguish Contradictory from Extrapolatory?

`attribution()` collapses Extrapolatory and Contradictory into the same "not supported" outcome (Analyses 1-4 above). METRICS.md separately reports faithfulness's contradiction rate (`kappa`) on the theory that refuted and unsupported are different failure modes; AttrEval-GenSearch's own three-way label lets that claim be tested independently, on attribution's own NLI backbone, using P_contradiction as kappa.

**One comparison, reported once.** Scoring kappa against `is_Contradictory` on the Contradictory/Extrapolatory subset (Attributable excluded) and also against `is_Extrapolatory` on the same subset would be the same fact twice: `AUC(kappa, is_Extrapolatory) == 1 - AUC(kappa, is_Contradictory)` identically on a two-class subset. Only `is_Contradictory` is reported.

| Statistic | Target | n (Contradictory / Extrapolatory) | ROC-AUC (95% CI) | Clears chance? |
|---|---|---|---|---|
| kappa (P_contradiction) | is_Contradictory | 33 / 128 | 0.746 (0.639, 0.841) | yes |


**kappa does separate Contradictory from Extrapolatory** on this benchmark: its CI clears chance (0.5), supporting METRICS.md's claim that refuted and unsupported are distinguishable failure modes (AUC=0.746).

Label distribution: Attributable=81, Extrapolatory=128, Contradictory=33 (n=242).

## Honesty notes

- n=242 is small; every CI above should be read at its full width, not just its point estimate.
- Class imbalance: 81/242 Attributable vs. 161/242 not-supported (128 Extrapolatory + 33 Contradictory).
- If the metric performs poorly on this benchmark, that is the headline finding, not a footnote: a citation-support metric that fails on third-party, human-annotated citation judgments is exactly the kind of gap this repository exists to surface.

## Artefacts

- `attreval_validation.json` -- full numeric results
- `attreval_validation.png` -- ROC curve (P_entailment), F1-vs-tau sweep, and P_entailment/P_contradiction distributions by label
