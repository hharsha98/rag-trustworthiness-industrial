# BEIR/SciFact retrieval ablation

This benchmark is [BEIR](https://github.com/beir-cellar/beir)'s SciFact task: a third-party corpus of 5183 biomedical-claim-verification abstracts, third-party queries, and third-party binary relevance judgments (`BeIR/scifact-qrels`), none of which were authored by this repository. It measures the same six configurations as `experiments/07_retrieval_ablation.py` ({dense, sparse, hybrid} x {no rerank, rerank}), replacing that experiment's 10-query, 32-passage, single-annotator in-house judgment set. The reranking-arm confound documented in experiment 07 -- 20 candidates approaching the size of a tiny corpus, so every first-stage retriever handed the cross-encoder nearly the same pool -- does not apply here: 20 candidates out of 5183 documents is 0.3859% of the corpus, so the first stage is genuinely selective.

Corpus: 5183 documents. Queries evaluated: 300 of 300 judged. Bootstrap: 10000 resamples, 95% CI, seed=0.

## k = 3

| configuration | nDCG@k | Recall@k | MRR@k | nDCG vs dense (95% CI) |
|---|---|---|---|---|
| dense | 0.484 [0.431, 0.535] | 0.524 | 0.479 | -- (baseline) |
| dense+rerank | 0.609 [0.558, 0.659] | 0.647 | 0.609 | +0.125 [+0.084, +0.168] **beats dense** |
| sparse | 0.626 [0.575, 0.676] | 0.681 | 0.617 | +0.142 [+0.092, +0.193] **beats dense** |
| sparse+rerank | 0.640 [0.590, 0.689] | 0.689 | 0.636 | +0.156 [+0.108, +0.205] **beats dense** |
| hybrid | 0.577 [0.525, 0.628] | 0.627 | 0.569 | +0.093 [+0.060, +0.129] **beats dense** |
| hybrid+rerank | 0.643 [0.593, 0.692] | 0.695 | 0.638 | +0.160 [+0.112, +0.207] **beats dense** |

## k = 5

| configuration | nDCG@k | Recall@k | MRR@k | nDCG vs dense (95% CI) |
|---|---|---|---|---|
| dense | 0.510 [0.458, 0.560] | 0.588 | 0.494 | -- (baseline) |
| dense+rerank | 0.625 [0.575, 0.673] | 0.685 | 0.617 | +0.115 [+0.079, +0.153] **beats dense** |
| sparse | 0.642 [0.593, 0.690] | 0.719 | 0.625 | +0.132 [+0.083, +0.181] **beats dense** |
| sparse+rerank | 0.658 [0.609, 0.706] | 0.731 | 0.645 | +0.148 [+0.102, +0.194] **beats dense** |
| hybrid | 0.608 [0.559, 0.656] | 0.701 | 0.587 | +0.098 [+0.068, +0.130] **beats dense** |
| hybrid+rerank | 0.661 [0.612, 0.707] | 0.736 | 0.647 | +0.151 [+0.107, +0.195] **beats dense** |

## k = 10

| configuration | nDCG@k | Recall@k | MRR@k | nDCG vs dense (95% CI) |
|---|---|---|---|---|
| dense | 0.529 [0.480, 0.578] | 0.646 | 0.501 | -- (baseline) |
| dense+rerank | 0.633 [0.583, 0.681] | 0.705 | 0.620 | +0.103 [+0.071, +0.137] **beats dense** |
| sparse | 0.664 [0.618, 0.710] | 0.782 | 0.634 | +0.135 [+0.089, +0.181] **beats dense** |
| sparse+rerank | 0.680 [0.634, 0.724] | 0.795 | 0.653 | +0.150 [+0.107, +0.193] **beats dense** |
| hybrid | 0.644 [0.599, 0.687] | 0.810 | 0.601 | +0.115 [+0.087, +0.143] **beats dense** |
| hybrid+rerank | 0.687 [0.642, 0.731] | 0.814 | 0.657 | +0.158 [+0.118, +0.198] **beats dense** |

## Conclusion

**Every one of the 5 non-baseline configurations beats dense retrieval at every k tested (3, 5, 10)**, each improvement surviving the 95% paired bootstrap CI. The strongest is `hybrid+rerank` at k=3 (+0.160 nDCG). 

**This overturns experiment 07's null result.** That experiment could not distinguish these configurations because its benchmark was too small and too easy -- 10 queries over 32 self-authored passages, with MRR saturated at 1.000. The techniques were not ineffective; the measurement was not sensitive enough to see them. On a third-party benchmark 160x larger, the differences are unambiguous.

Dense retrieval underperforming BM25 on SciFact is itself a known result: the BEIR paper reports the same ordering for general-purpose bi-encoders on this dataset, whose specialised biomedical vocabulary favours lexical matching. That these numbers reproduce a published finding is evidence the pipeline is measuring retrieval rather than a bug in itself.

## MRR saturation

MRR is **not** saturated here (unlike experiment 07, where it was 1.000 for every configuration at every k). See the per-k tables above for the actual values and spread across configurations -- this is one of the two limitations experiment 07 flagged, and this benchmark resolves it.

## How this differs from experiment 07

- **Corpus size:** 5183 documents vs. 32. **Reranking candidate ratio:** 20/5183 = 0.3859% here vs. ~62% there -- the first stage is genuinely selective, so the three `+rerank` rows above are not expected to collapse to identical numbers the way experiment 07's did.
- **Queries:** 300 (of 300 judged) vs. 10.
- **Judgments:** third-party, binary (BeIR/scifact-qrels) vs. this repository's own single-annotator, graded {0,1,2} judgments.
- **Provenance:** corpus, queries, and judgments here are all from BEIR, independent of this repository -- unlike experiment 07's self-authored corpus and judgment set.

Corpus embeddings for 'sentence-transformers/msmarco-distilbert-base-v4' are cached at `/Users/harsha/projects/rag-trustworthiness-industrial/data/benchmarks/scifact_corpus_embeddings__sentence-transformers__msmarco-distilbert-base-v4.npy` so a second run does not re-embed the corpus.

