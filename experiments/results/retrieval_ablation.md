# Retrieval ablation

The corpus is original text written for this repository, so its section boundaries are known exactly and each question was written against a specific section. Judgments are therefore made at section level and inherited by every chunk drawn from that section (a chunk's `page` is its 1-based section index). This is a small, single-annotator set on a corpus the annotator wrote: it is sufficient to rank retrieval configurations against one another, and it is NOT evidence of absolute retrieval quality on unseen corpora.

Corpus: 32 passages. Queries: 10. Bootstrap: 10000 resamples, 95% CI, seed=0.

## k = 3

| configuration | nDCG@k | Recall@k | MRR@k | nDCG vs dense (95% CI) |
|---|---|---|---|---|
| dense | 0.832 [0.738, 0.927] | 0.883 | 1.000 | -- (baseline) |
| dense+rerank | 0.749 [0.643, 0.859] | 0.883 | 1.000 | -0.083 [-0.172, +0.000] (inside noise) |
| sparse | 0.783 [0.687, 0.881] | 0.833 | 1.000 | -0.049 [-0.106, -0.006] (worse) |
| sparse+rerank | 0.749 [0.643, 0.859] | 0.883 | 1.000 | -0.083 [-0.172, +0.000] (inside noise) |
| hybrid | 0.794 [0.676, 0.909] | 0.833 | 1.000 | -0.038 [-0.091, +0.000] (inside noise) |
| hybrid+rerank | 0.749 [0.643, 0.859] | 0.883 | 1.000 | -0.083 [-0.172, +0.000] (inside noise) |

## k = 5

| configuration | nDCG@k | Recall@k | MRR@k | nDCG vs dense (95% CI) |
|---|---|---|---|---|
| dense | 0.769 [0.664, 0.877] | 0.933 | 1.000 | -- (baseline) |
| dense+rerank | 0.781 [0.688, 0.879] | 0.883 | 1.000 | +0.012 [-0.060, +0.079] (inside noise) |
| sparse | 0.767 [0.644, 0.891] | 0.950 | 1.000 | -0.001 [-0.079, +0.082] (inside noise) |
| sparse+rerank | 0.781 [0.688, 0.879] | 0.883 | 1.000 | +0.012 [-0.060, +0.079] (inside noise) |
| hybrid | 0.769 [0.656, 0.882] | 0.967 | 1.000 | -0.000 [-0.068, +0.072] (inside noise) |
| hybrid+rerank | 0.781 [0.688, 0.879] | 0.883 | 1.000 | +0.012 [-0.060, +0.079] (inside noise) |

## k = 10

| configuration | nDCG@k | Recall@k | MRR@k | nDCG vs dense (95% CI) |
|---|---|---|---|---|
| dense | 0.817 [0.725, 0.906] | 0.967 | 1.000 | -- (baseline) |
| dense+rerank | 0.823 [0.723, 0.918] | 0.933 | 1.000 | +0.006 [-0.030, +0.046] (inside noise) |
| sparse | 0.801 [0.706, 0.896] | 1.000 | 1.000 | -0.016 [-0.095, +0.056] (inside noise) |
| sparse+rerank | 0.818 [0.714, 0.916] | 0.933 | 1.000 | +0.001 [-0.038, +0.044] (inside noise) |
| hybrid | 0.836 [0.754, 0.917] | 0.967 | 1.000 | +0.018 [-0.013, +0.053] (inside noise) |
| hybrid+rerank | 0.818 [0.712, 0.916] | 0.933 | 1.000 | +0.001 [-0.039, +0.044] (inside noise) |

## Conclusion

No configuration beat the dense baseline outside the 95% confidence interval, at any k. Every observed difference is consistent with query-sampling noise on this 10-query set. This is a valid, publishable null result on this corpus/judgment set -- it is not evidence that BM25/hybrid/reranking never help, only that this measurement could not distinguish them from dense retrieval here.

## What this benchmark cannot show

**MRR is saturated and therefore uninformative here.** It is 1.000 for every configuration at every k, because the top-ranked passage is already a directly-relevant one for all 10 queries. That is a real measurement, not a bug -- and it means MRR has no headroom on this corpus and cannot separate any two configurations. Read the identical 1.000 column as 'this benchmark is too easy to measure ranking quality', not as 'all configurations rank equally well'.

**The reranking arm is confounded by candidate-pool saturation.** Reranking scores the top 20 first-stage candidates, but the corpus holds only 32 passages -- so the first stage passes through roughly 62% of everything, and all three first-stage retrievers hand the cross-encoder nearly the same pool. That is why the three `+rerank` rows report near-identical nDCG: the cross-encoder is reordering the same set each time. **This experiment therefore cannot compare dense, sparse and hybrid retrieval when reranking is enabled.** Doing so needs a corpus large enough that the first stage is genuinely selective -- a rule of thumb is candidates well under a tenth of the corpus.

Neither limitation affects the headline result, which concerns the non-reranked configurations, and both argue the same way: this measurement is a floor on what these techniques could do, not a ceiling.

