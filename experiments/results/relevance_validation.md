# Relevance validation -- METRICS.md Part II.3, context_relevance

**Read the floor section (Section 4) as a calibration claim, not a ranking claim.** ROC-AUC is invariant to the monotone remap that separates the clamped metric from its historical affine predecessor, so an AUC comparison between them would be meaningless by construction -- Section 4 proves that numerically instead of computing it as if it were informative.

**Embedder:** `sentence-transformers/msmarco-distilbert-base-v4`, trained on MS MARCO query/passage pairs. BEIR/SciFact (biomedical claim verification) and BEIR/NFCorpus (biomedical/nutrition) are neither of them MS MARCO retrieval text, so both sit outside the embedder's training distribution.

**Circularity guard (Trap 1).** Evaluation pairs are built from BEIR's own qrels, never from this repository's dense retriever's output (which shares the embedder under test). Positives: qrel score > 0. Negatives, two tiers: *random* (uniform sample from the corpus) and *hard* (top-ranked non-relevant documents by BM25 -- lexical, so independent of the embedder). Standard BEIR caveat: qrels are sparse, so an unjudged document is not certified irrelevant -- both negative tiers, especially *random*, may contain false negatives.

n_random_neg_per_query = 5, n_hard_neg_per_query = 5, seed = 0.

NFCorpus was included as a second, independent dataset.

## Dataset: scifact

Corpus: 5183 documents. Judged queries used: 300.

### Section 1 -- ROC-AUC / PR-AUC of `max(0, cos)` vs judged-relevant, by tier

| tier | n | n_pos | n_neg | ROC-AUC (95% CI) | PR-AUC (95% CI) |
|---|---:|---:|---:|---|---|
| random | 1839 | 339 | 1500 | 0.956 (0.942, 0.969) | 0.906 (0.882, 0.927) |
| hard | 1839 | 339 | 1500 | 0.745 (0.712, 0.775) | 0.472 (0.425, 0.523) |

### Section 2 -- BM25 lexical baseline vs cosine (v2), by tier (paired permutation test)

| tier | AUC(cosine) | AUC(BM25) | diff | p-value | verdict |
|---|---:|---:|---:|---:|---|
| random | 0.956 | 0.972 | -0.016 | 0.4454 | not significantly different |
| hard | 0.745 | 0.611 | +0.134 | 0.0001 | cosine significantly better |

### Section 3 -- the floor (Trap 2): calibration, not discrimination

On the **3000** known-irrelevant passages in this dataset (both negative tiers pooled), the score distribution under the current mapping vs an affine alternative that was considered and rejected:

| mapping | mean | median | p5 | p95 | min |
|---|---:|---:|---:|---:|---:|
| clamp `max(0,t)` (current) | 0.250 | 0.240 | 0.006 | 0.520 | 0.000 |
| affine `(1+t)/2` (alternative mapping, rejected) | 0.624 | 0.620 | 0.503 | 0.760 | 0.416 |

**AUC-identity proof**: AUC(clamp) = 0.850417896, AUC(affine) = 0.850417896, |diff| = 0.00e+00 (matches to 1e-9). This confirms the two mappings are, as expected, indistinguishable by ROC-AUC on this data -- the calibration table above, not this identity, is what shows the floor fix's actual effect.

## Dataset: nfcorpus

Corpus: 3633 documents. Judged queries used: 323.

### Section 1 -- ROC-AUC / PR-AUC of `max(0, cos)` vs judged-relevant, by tier

| tier | n | n_pos | n_neg | ROC-AUC (95% CI) | PR-AUC (95% CI) |
|---|---:|---:|---:|---|---|
| random | 13949 | 12334 | 1615 | 0.613 (0.600, 0.626) | 0.927 (0.924, 0.930) |
| hard | 13627 | 12334 | 1293 | 0.248 (0.235, 0.262) | 0.836 (0.831, 0.840) |

### Section 2 -- BM25 lexical baseline vs cosine (v2), by tier (paired permutation test)

| tier | AUC(cosine) | AUC(BM25) | diff | p-value | verdict |
|---|---:|---:|---:|---:|---|
| random | 0.613 | 0.550 | +0.063 | 0.0001 | cosine significantly better |
| hard | 0.248 | 0.050 | +0.198 | 0.0001 | cosine significantly better |

### Section 3 -- the floor (Trap 2): calibration, not discrimination

On the **2908** known-irrelevant passages in this dataset (both negative tiers pooled), the score distribution under the current mapping vs an affine alternative that was considered and rejected:

| mapping | mean | median | p5 | p95 | min |
|---|---:|---:|---:|---:|---:|
| clamp `max(0,t)` (current) | 0.150 | 0.110 | 0.000 | 0.449 | 0.000 |
| affine `(1+t)/2` (alternative mapping, rejected) | 0.570 | 0.555 | 0.464 | 0.725 | 0.355 |

**AUC-identity proof**: AUC(clamp) = 0.450453090, AUC(affine) = 0.450169182, |diff| = 2.84e-04 (does NOT match to 1e-9). Not an exact match here, honestly reported rather than rounded away: 21.26% of raw cosines on this dataset are negative (vs a much smaller fraction on the other dataset), and when two or more distinct negative cosines are clipped to the same 0, they become tied under the clamp mapping where they were NOT tied under the strictly-monotone affine mapping -- exactly the boundary case the module docstring flags ('over the range that actually occurs'). The residual is tiny (5e-4) and does not change the conclusion that AUC cannot see the floor -- it is the mechanism by which the 'always exactly equal' claim can, at the margin, fail to hold.

### Section 4 -- how much of this depends on counting MARGINAL relevance as positive?

Sections 1-3 binarise the qrels at `score > 0`, the standard BEIR rule. On a graded, densely-judged set that rule does a lot of work: this dataset has grade 1: 11758, grade 2: 576. Restricting the positive class to the top grade (2) asks the sharper question -- does the metric rank a lexically-matched hard negative above a document a human called *definitely* relevant?

| tier | AUC, all positives (`score > 0`) | AUC, grade 2 only (95% CI) |
|---|---:|---|
| random | 0.613 (n_pos=12334) | 0.884 (0.867, 0.901) (n_pos=576) |
| hard | 0.248 (n_pos=12334) | 0.560 (0.532, 0.588) (n_pos=576) |

**This materially changes the reading of the hard tier.** At `score > 0` the AUC is 0.248; restricted to grade 2 it is 0.560 (0.560 (0.532, 0.588)). A sub-chance number under the binarised rule therefore does NOT support the claim that the metric ranks hard negatives above genuinely relevant documents; it supports the much weaker claim that it ranks them above *marginally* relevant ones, which is a statement about how this benchmark defines relevance at least as much as about the metric. Both numbers are reported because neither alone is the whole answer.

## Limitations

- **Sparse qrels (standard BEIR caveat).** An unjudged document is not certified irrelevant. Both negative tiers -- especially *random*, which was never seen by a human annotator -- may contain false negatives, which would understate the true AUC.

- **Hard-tier BM25 negatives are lexical, not semantic, negatives.** A document can rank highly under BM25 (shares vocabulary with the query) while still being genuinely off-topic, which is exactly the discriminating case this tier is meant to probe -- but it also means a hard negative could occasionally be a real, unjudged positive that happens to share vocabulary with the query.

- **Section 2's hard-tier row is not a fair general test of BM25.** The hard negatives were SELECTED as the query's top-ranked BM25 documents (excluding positives) -- so by construction they score very highly under BM25, often as high as or higher than the true positives. That mechanically depresses AUC(BM25) on the hard tier specifically (biasing that one comparison IN FAVOUR of cosine), independent of BM25's real retrieval quality. The random-tier row does not have this bias and is the fairer BM25-vs-cosine comparison of the two.

## Artefacts

- `relevance_validation.json` -- full numeric results
- `relevance_validation.png` -- ROC curves (cosine vs BM25) by tier, score distributions by relevance, and the clamp-vs-affine floor comparison
