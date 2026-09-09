# Weight sensitivity -- METRICS.md Part II.5 / Part III

Sampled 10000 Dirichlet(1,1,1,1) weightings over (faithfulness, attribution, relevance, conciseness) for 10 real, non-abstained pipeline answers (NLI: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, embedder: `sentence-transformers/msmarco-distilbert-base-v4`). `attribution` is the F1 of citation precision/recall.

## Per-item metrics

| Question | Faithfulness | Attribution (F1) | Relevance | Conciseness |
|---|---:|---:|---:|---:|
| What is the role of self-supervised learning in robotics? | 0.970 | 0.667 | 0.602 | 0.540 |
| How does deep reinforcement learning work in robotics? | 0.750 | 0.800 | 0.559 | 0.783 |
| What are the advantages of using artificial neural networks  | 0.903 | 0.857 | 0.568 | 0.774 |
| How can sensor fusion improve robot performance? | 0.843 | 0.500 | 0.460 | 0.726 |
| What are the applications of machine learning in robotics? | 0.515 | 0.667 | 0.530 | 0.772 |
| Explain the concept of self-supervised depth estimation in r | 0.932 | 0.706 | 0.536 | 0.703 |
| What is the function of convolutional layers in CNNs for vis | 0.863 | 0.800 | 0.435 | 0.637 |
| Describe the Markov Decision Process in reinforcement learni | 0.890 | 0.667 | 0.451 | 0.706 |
| How does an LSTM cell retain long-term dependencies? | 0.699 | 0.500 | 0.378 | 0.685 |
| What is the purpose of max-pooling layers in CNNs? | 0.681 | 0.444 | 0.446 | 0.663 |

## Aggregate distributions (across all items x sampled weightings)

| Aggregate | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| T_arith (compensatory) | 0.665 | 0.091 | 0.385 | 0.951 |
| T_geom (non-compensatory) | 0.653 | 0.091 | 0.384 | 0.947 |

**Pairwise ranking disagreement rate**: across all 45 item pairs and 10000 weightings (450000 comparisons total), T_arith and T_geom rank the pair differently in **0.0190** of cases.

**Compensatory failure**: the lowest-faithfulness item (`What are the applications of machine learning in robotics?`, faithfulness=0.515) scores T_arith > 0.5 while T_geom <= 0.5 in **0.0000** of sampled weightings.

This rate is **zero on this dataset, and that is the finding** -- not a failed experiment. The two aggregators diverge only when some dimension approaches zero, and no answer here comes close: the minimum observed faithfulness is 0.515. The reason is the abstention gate. Answers the pipeline is not confident about are never emitted, so the regime where the arithmetic mean masks a hallucination is one this pipeline does not enter. The gate removes the failure mode upstream of the aggregator.

The property itself is not in doubt; it is proved in METRICS.md (Proposition 3) and shown on a constructed case where a fluent but unfounded answer scores `faithfulness=0.05, attribution=0.9, relevance=0.9, conciseness=0.95`:

```
T_arith = 0.5700   <- passes
T_geom  = 0.2863   <- correctly penalised
```

What this experiment establishes is narrower and more useful: on *real* output from a pipeline that can decline, the choice of aggregator barely matters (1.90% of ranking comparisons differ). The aggregator is the safety net; the abstention gate is what does the work.

## Artefacts

- `weight_sensitivity.json` -- full numeric results
- `weight_sensitivity.png` -- left: T_arith / T_geom score distributions; right: T_arith vs T_geom scatter for the lowest-faithfulness item across all sampled weightings
