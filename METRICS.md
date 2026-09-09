# Trustworthiness Metrics — Formal Specification

This document is the mathematical contract for `src/ragtrust/metrics/`. Every claim marked
**Proposition** is asserted by a test in `tests/`.

Notation. A query `q`; a retrieved passage set `D = {d₁,…,d_k}`; an answer `a` decomposed
into atomic claims `c₁,…,c_n`. `e_x` is the unit-normalised embedding of `x`, so
`cos(e_x, e_y) = ⟨e_x, e_y⟩ ∈ [-1,1]`. An NLI model maps a (premise, hypothesis) pair to a
probability vector `(p_ent, p_neu, p_con)` with `p_ent + p_neu + p_con = 1`.

---

## Part II — The trust metrics

### 1. Faithfulness (grounding)

```
sᵢ = max_{j∈[k]}  p_ent(premise = dⱼ, hypothesis = cᵢ)
F  = (1/n) Σᵢ₌₁ⁿ sᵢ
```

Each claim is checked against each passage *individually* — this avoids the 512-token
truncation that concatenating every passage into a single premise string would cause. `max`
over passages encodes "a claim needs only one supporting source"; `mean` over claims makes `F`
the *proportion of the answer that is grounded*.

> **Proposition 1.** `F ∈ [0,1]`; `F` is invariant to permutations of `D`; and `F` is
> monotone non-decreasing under adding a passage to `D`.
>
> *Proof.* Each `sᵢ` is a max of probabilities, so `sᵢ ∈ [0,1]`, and a mean of values in
> `[0,1]` lies in `[0,1]`. `max` is a symmetric function of its arguments, giving permutation
> invariance. Adding `d_{k+1}` replaces `sᵢ` with `max(sᵢ, p_ent(d_{k+1}, cᵢ)) ≥ sᵢ`, and the
> mean is monotone in each argument. ∎

Contradiction is reported **separately** rather than folded into `F`, because "unsupported"
and "refuted" are operationally different in an industrial setting:

```
κ = (1/n) Σᵢ max_{j∈[k]} p_con(dⱼ, cᵢ)          (contradiction rate)
```

### 2. Attribution

The generator emits inline citations `(claim → passage id)`. Let `τ` be the support
threshold (default `0.5`, swept in the report). A claim is *supported* iff `sᵢ ≥ τ`.

```
precision = |{citations whose cited passage entails its claim}| / |{citations emitted}|
recall    = |{supported claims that carry a citation}|          / |{supported claims}|
A         = 2·precision·recall / (precision + recall)
```

with the conventions `precision = 1` when no citation is emitted and `recall = 1` when no
claim is supported (vacuous truth), and `A = 0` when `precision + recall = 0`.

This scores *citation behaviour* — whether the system points at the right evidence — which is
independent of whether support exists at all. It is therefore not a monotone transform of
`F`, unlike a naive thresholded-support metric.

### 3. Relevance

Two different quantities are involved, and they are kept separate:

```
R_ctx = (1/k) Σⱼ max(0, cos(e_q, e_dⱼ))          # context relevance: did retrieval work?
R_ans = (1/m) Σ_l cos(e_q, e_q'_l)               # answer relevance: q'_l back-generated from a
```

**Both now ship, and they answer different questions.** `R_ctx` asks whether *retrieval* found
the right material. `R_ans` asks whether the *answer* addresses the question that was asked —
the only metric here that can catch a fluent, correctly-grounded answer to the **wrong
question**. Faithfulness cannot: an answer perfectly entailed by its passages is faithful even
when it answers something else. Attribution cannot: its citations may every one be valid. That
failure mode is invisible to every other dimension.

`R_ans` is computed by back-generation (RAGAS): the generator is asked for `m` questions the
answer would answer, those are embedded, and the mean cosine against the original query is
taken. It costs a **second model call per answer**, so it is opt-in —
`Config.answer_relevance` (default `False`), with `answer_relevance_n_questions` (default 3).
The same cost reasoning governs `rerank`.

When it is off, when the generation backend does not implement `generate_questions`, or when
back-generation fails, the metric is `None` — undefined, and dropped from aggregation under the
§II.5 rule, exactly as conciseness is below two claims. A diagnostic metric must never be able
to break the primary answer path, so a backend failure degrades to `None` with a warning rather
than propagating.

`R_ans` is deliberately **not** in the default `weights`. Adding it there would silently move
every existing trust score; a deployment that wants it weighted adds the key and it participates
in both aggregates automatically (`tests/test_answer_relevance.py` asserts exactly that,
including that Proposition 3 still holds across five weighted dimensions).

**On the choice of mapping.** An alternative that was considered and rejected: the affine
map `t ↦ (1+t)/2`, on the argument that it sends `[-1,1] → [0,1]` while strictly preserving
order, whereas clamping collapses all negative similarities to a tie. That argument is
correct about ordering and wrong about what matters here.

Sentence encoders essentially never produce negative cosine on real text, so in practice the
affine map puts a **floor of ≈0.5** under `R_ctx`. Measured: a query with no relation
whatsoever to the corpus retrieves at cosine ≈0.018 and scores `R_ctx = 0.505`.

A floor like that is exactly the kind of defect a trustworthiness metric must not have. A
bounded metric that cannot approach zero cannot signal failure — and inside the
non-compensatory aggregate of §II.5, a factor that never drops below 0.5 can never pull the
overall score toward zero, which is the entire point of using a product.

Clamping surrenders ordering among negative cosines, which do not occur in practice, and buys
the property that actually matters: content orthogonal to the query scores 0.

**`R_ctx` with `scaled=False` (the raw cosine mean) is retained for calibration analysis**
(`experiments/13_relevance_validation.py`).

Retrieval additionally reports standard IR metrics over a judged set: `nDCG@k`, `Recall@k`,
`MRR`. The index is normalised inner-product so that ranking and scoring share one geometry —
an unnormalised `L2` index would rank by Euclidean distance while scoring by cosine, and for
unnormalised `x`, `‖x−y‖² = ‖x‖² + ‖y‖² − 2⟨x,y⟩`, so `L2` ordering depends on `‖y‖` and does
not agree with cosine ordering.

### 4. Conciseness (information density, not length)

Conciseness should penalise *padding*, not length: a long answer to a complex question is not
untrustworthy. Define the mean pairwise self-similarity of the answer's own claims:

```
C = 1 − (2 / (n(n−1))) Σ_{1≤i<i'≤n} max(0, cos(e_cᵢ, e_c_i'))        for n ≥ 2
C = undefined                                                         for n < 2
```

> **Proposition 2.** `C ∈ [0,1]`. `C = 1` iff no two claims are positively correlated
> (maximally diverse); `C = 0` iff every pair of claims is semantically identical
> (`cos = 1`), i.e. the answer says one thing `n` times.
>
> *Proof.* There are `n(n−1)/2` pairs, so the sum scaled by `2/(n(n−1))` is a mean of values
> in `[0,1]`; one minus that mean lies in `[0,1]`. The boundary cases follow by inspection of
> when the mean equals 0 and 1 respectively. ∎

**`n < 2` is undefined, not perfect — an alternative that was considered and rejected.** An
earlier version of this metric returned `C = 1` for a single-claim answer. With fewer than
two claims there are no pairs, the mean is `0/0`, and a *perfect* score would be manufactured
from an uncomputable quantity and propagated into both aggregates — a bounded score that
cannot express the bad case.

`conciseness()` returns `None` in that case, and an undefined metric is **dropped** from
aggregation rather than given a value, consistent with §II.5's rule that a metric to be
ignored is dropped and never assigned weight 0.

This design choice is supported by measurement, not taste. On SEAHORSE (`experiments/12`),
**68.5%** of 4,143 human-rated summaries decompose into fewer than 2 claims — the sentinel was
not a corner case but close to the modal one. Restricted to summaries where `C` is genuinely
computed, it predicts human repetition judgments at **ROC-AUC 0.820** against **0.566** for the
best pure-length baseline (paired permutation `p ≈ 0.0001`), so the metric itself is sound; it
was the value returned where it does not apply that was not.

### 5. Aggregation

The weighted arithmetic mean is **compensatory**: strong relevance and conciseness offset a
hallucination, so a fluent, well-formed, entirely fabricated answer still scores ≈0.6 —
precisely backwards for a trustworthiness score. Both aggregators ship, and the repo contrasts
them:

```
T_arith = Σᵢ wᵢ mᵢ                    (compensatory)
T_geom  = Πᵢ mᵢ^{wᵢ},   Σᵢ wᵢ = 1,  wᵢ > 0      (non-compensatory)
```

> **Proposition 3.** `T_geom ∈ [0,1]`; `T_geom = 0` if any `mᵢ = 0`; `T_geom` is strictly
> increasing in each `mᵢ`; and `T_geom ≤ T_arith` for all inputs, with equality iff all `mᵢ`
> are equal.
>
> *Proof.* The first three are immediate from the definition with `wᵢ > 0`. The inequality is
> the weighted AM–GM inequality. ∎

The last part matters practically: **the geometric aggregate can never overstate trust
relative to the arithmetic one.** Weights are required strictly positive so that `0^0` never
arises; a metric to be ignored is dropped from the product, not given weight 0.

**Both aggregates renormalise over the metrics actually present, and Proposition 3 depends on
it.** `Σᵢ wᵢ = 1` is a *premise* of weighted AM–GM, not a decoration. When a dimension is
dropped — as conciseness now is for answers of fewer than two claims (§II.4) — the surviving
weights no longer sum to 1, and each aggregate must divide through by their total.

Renormalising only one of them silently breaks the bound. `aggregate_geometric` always
renormalised; `aggregate_arithmetic` did not, which meant that dropping conciseness (weight
0.2) left `T_arith` with only 0.8 of weight mass while `T_geom` renormalised to 1.0. On a
perfect answer with conciseness undefined that produced

```
T_arith = 0.8000     T_geom = 1.0000        <- Proposition 3 violated
```

i.e. the aggregate advertised as unable to overstate trust *did* overstate it, on the very
input where it should have been most reliable. Both now renormalise over the same key set:

```
T_arith = 1.0000     T_geom = 1.0000        <- bound restored
```

`tests/test_metric_properties.py` asserts the inequality on the dropped-dimension all-ones
case specifically, since that is where the un-renormalised form fails and a generic
random-input property test can miss it.

Rather than four hand-picked weight constellations, weight choice is treated as a sensitivity
question: sample `w ~ Dirichlet(1,1,1,1)` (uniform over the simplex, 10 000 draws) and report
the distribution of scores and of system rankings, so conclusions are shown to be
weight-robust rather than assumed to be.

### 6. Abstention gate

General-knowledge questions can retrieve confidently ranked but unrelated passages when a
system has no way to decline. Scoring those answers as failures would partly be *correct
behaviour being penalised*. This system adds an explicit gate: if `max_i sᵢ < τ_abstain`
(default `0.5`), it returns "not answerable from this corpus" and is scored on the abstention
decision rather than on a fabricated answer.

---

## Part III — Validation protocol

Metric quality is established against independent, third-party, human-annotated benchmarks —
not against ground truth this repository constructed itself. Each trust metric is checked on
a benchmark it was not built or tuned against, chosen where possible to sit outside the
training distribution of the model that computes it (see each experiment's docstring for the
specific argument).

| Dimension | Benchmark | Result | Script |
|---|---|---|---|
| Faithfulness | RAGTruth (Wu et al., 2024) — 900 human-annotated items | Pooled ROC-AUC 0.683 (0.648, 0.718) | `experiments/10_ragtruth_validation.py` |
| Attribution | AttrEval-GenSearch (Yue et al., 2023) | ROC-AUC 0.805 (0.737, 0.868) | `experiments/11_attreval_validation.py` |
| Conciseness | SEAHORSE (Clark et al., 2023) | ROC-AUC 0.820 restricted to items where `C` is computed (§II.4) | `experiments/12_seahorse_validation.py` |
| Relevance | BEIR/SciFact + NFCorpus qrels | see report for the per-tier breakdown | `experiments/13_relevance_validation.py` |
| Attribution threshold (`support_threshold`) | HAGRID (Kamalloo et al., 2023), cross-checked against AttrEval-GenSearch | see report for the calibration recommendation | `experiments/14_hagrid_calibration.py` |

Faithfulness validation additionally checks that the two failure modes this specification
treats as distinct (§II.1) are actually distinguishable: on RAGTruth items where
`evident_conflict` and `baseless_info` are not both present, `κ` (contradiction rate)
discriminates conflicts and `1 − F` discriminates baseless content, supporting the claim that
"refuted" and "unsupported" are operationally different rather than one signal in disguise.

Bootstrap 95% confidence intervals (10 000 resamples) and paired permutation tests
(`src/ragtrust/validation/stats.py`) back every reported comparison, so a claimed difference
rests on a significance test rather than a single point estimate. Full methodology, dataset
provenance, and caveats — including training-distribution overlap where it exists, e.g.
HAGRID's partial overlap with the NLI backbone's FEVER fine-tuning data — are documented in
each script's docstring and in `experiments/results/`.

---

## References

These definitions follow established practice rather than being invented here: claim-level
entailment decomposition (RAGAS; FactScore), attribution precision/recall (Attributed QA),
answer relevance by question back-generation (RAGAS), and non-compensatory aggregation from
multi-criteria decision analysis.
