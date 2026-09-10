from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    """Runtime configuration for the pipeline.

    Defaults work out of the box on a slide-style technical corpus; every one is
    overridable per deployment.
    """

    # --- models -----------------------------------------------------------------
    embed_model: str = "sentence-transformers/msmarco-distilbert-base-v4"
    nli_model: str = "roberta-large-mnli"

    # --- retrieval --------------------------------------------------------------
    k: int = 5

    # Retrieval backend: "dense" (cosine over embeddings), "sparse" (BM25,
    # retrieval/sparse.py), or "hybrid" (Reciprocal Rank Fusion of both,
    # retrieval/hybrid.py).
    #
    # The default is "hybrid" because it was measured, not assumed. On BEIR/SciFact --
    # 5183 third-party documents, 300 third-party queries, third-party judgments --
    # every alternative beat plain dense retrieval at every k, each improvement
    # surviving a 95% paired bootstrap CI (experiments/08_beir_ablation.py):
    #
    #     nDCG@10   dense 0.529 | hybrid 0.644 (+0.115) | hybrid+rerank 0.687 (+0.158)
    #
    # An earlier in-house benchmark found no difference, and this defaulted to "dense"
    # on that basis. That benchmark was 10 queries over 32 self-authored passages and
    # was not sensitive enough to detect the effect; experiments/07 documents why.
    retrieval_mode: str = "hybrid"

    # Optional cross-encoder reranking on top of whichever `retrieval_mode` produced
    # the first-stage candidates. It measurably helps -- a further +0.043 nDCG@10 over
    # hybrid alone on BEIR/SciFact, and the best configuration overall -- but it stays
    # OFF by default for two reasons that are about cost, not quality:
    #
    #   1. It runs a second transformer over every candidate, so query latency roughly
    #      doubles and a second model must be downloaded.
    #   2. Cross-encoder scores are unbounded logits, not cosine similarities, so
    #      `retrieval_gate` below is NOT meaningful against them. Enabling reranking
    #      without addressing that silently breaks the pre-generation abstention gate.
    #      Read retrieval/rerank.py's module docstring first.
    rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 20

    # --- chunking ---------------------------------------------------------------
    # Passages are overlapping windows of source lines, so a heading stays attached
    # to the prose beneath it. Splitting per line instead yields ~9-word fragments --
    # mostly headings -- which an NLI model cannot entail and a generator cannot
    # answer from.
    chunk_window: int = 8
    chunk_stride: int = 4
    chunk_min_chars: int = 30

    # --- abstention -------------------------------------------------------------
    # Two independent gates, in increasing cost order.
    #
    # retrieval_gate fires BEFORE generation: if nothing retrieves above it, the
    # corpus does not cover the question and there is no point paying for a
    # generation call. Measured separation on the bundled corpus is ~0.72 in-corpus
    # against ~0.08 out-of-corpus, so 0.25 sits in open space between the two rather
    # than being fitted to either.
    #
    # abstain_threshold fires AFTER generation, on measured grounding: the corpus
    # looked relevant, but nothing actually supports what the model produced.
    # Calibrated, not guessed: experiments/09_gate_calibration.py scores 300 answerable
    # BEIR/SciFact queries against the SciFact corpus, and 900 unanswerable ones drawn
    # from three other BEIR sets (Quora as easy negatives, FiQA as medium, and NFCorpus
    # -- also biomedical -- as hard). Separation is strong: ROC-AUC 0.962 pooled, and
    # still 0.907 against the same-domain hard tier.
    #
    # 0.30 is the largest threshold retaining >=99% of answerable queries. The objective
    # is deliberately NOT accuracy or F1: the two errors are asymmetric. A false
    # abstention refuses a question the corpus can answer and nothing downstream
    # recovers it, whereas a false acceptance costs one generation call and is then
    # caught by the grounding gate below. So this gate is tuned for sensitivity and
    # leans on that second gate, rather than balancing the two error types.
    #
    # The previous 0.25 was fitted on a 32-passage toy corpus. It turned out to sit in a
    # reasonable place (99.3% retention) but admitted far more: 56%/47%/86% of the easy/
    # medium/hard negatives, against 27%/16%/73% at 0.30 for the same retention.
    #
    # Raise it to 0.37 for ~97.5% retention or 0.39 for ~95% if false acceptances are
    # more costly in your deployment than refusals.
    retrieval_gate: float = 0.30
    abstain_threshold: float = 0.5
    support_threshold: float = 0.5

    # --- aggregation ------------------------------------------------------------
    # Weights for the overall trust score. Strictly positive: the geometric
    # aggregate raises each metric to its weight, and 0**0 is undefined. A metric to
    # be ignored is dropped, not given weight 0.
    weights: dict = field(default_factory=lambda: {
        "faithfulness": 0.4,
        "attribution": 0.2,
        "relevance": 0.2,
        "conciseness": 0.2,
    })

    seed: int = 0

    # --- diagnostics --------------------------------------------------------------
    # Answer relevance (METRICS.md Part II.3): separate from context relevance above
    # -- it is the only metric that catches a fluent, well-grounded answer to the
    # WRONG question. Retrieval can be relevant and grounding can be perfect while
    # the answer addresses something else entirely; nothing else here detects that.
    #
    # Computed by back-generation (the RAGAS approach): ask the generator for
    # `answer_relevance_n_questions` questions the answer would answer
    # (Generator.generate_questions, an OPTIONAL method -- see generation/base.py),
    # embed them, and take the mean cosine against the original query's embedding
    # (metrics/relevance.py::answer_relevance).
    #
    # OFF by default for the same kind of reason `rerank` is off: cost, not doubt.
    # Turning it on issues a SECOND model call per answer (the back-generation
    # call), so it roughly doubles generation cost/latency. That is a cost decision
    # for your deployment to make, not a claim that the metric does not work.
    # If the configured generator backend has no `generate_questions` (most do
    # not need one), or a back-generation call fails, the pipeline degrades
    # `metrics["answer_relevance"]` to `None` rather than erroring -- it never adds
    # itself to the default `weights` below, so enabling it does not change
    # existing trust scores unless you also add "answer_relevance" to `weights`.
    answer_relevance: bool = False
    answer_relevance_n_questions: int = 3

    # Contextual Retrieval (Anthropic's method, ingest/contextualize.py): before
    # embedding/BM25-indexing, ask an LLM for a one-sentence blurb situating each
    # chunk in its document, and prepend it to the chunk for retrieval only (the
    # original chunk is kept separately as `passage_source_text` and is what NLI,
    # generation, and citations always see -- see pipeline.py::answer()).
    #
    # OFF by default for the same kind of reason `rerank` and `answer_relevance`
    # are off: cost, not doubt. This issues ONE LLM call per chunk at INDEX time
    # (not per query), so indexing a large corpus gets proportionally slower and
    # more expensive the first time it is built. That is a deliberate cost
    # decision for your deployment to make. If the configured generator has no
    # `complete` method, or a call fails, affected chunks silently fall back to
    # uncontextualised text rather than failing the index build.
    contextual: bool = False
    contextual_model: str = "llama3.2:3b"

    # --- agentic retrieval --------------------------------------------------------
    # agentic.py::answer_iterative's retrieval loop: how many retrieve-score-
    # reformulate rounds it may spend before giving up. Each round beyond the
    # first costs one reformulation call plus (unless a gate fires) one
    # generation call, so this is a cost cap, not a quality target -- the loop
    # already stops early the moment measured trust clears abstain_threshold
    # (see agentic.py's module docstring on why trust, not an LLM
    # self-assessment, is the stopping criterion). 3 is enough headroom for one
    # or two reformulations to find a better angle on the query without
    # letting a persistently low-trust query burn an unbounded number of calls.
    agentic_max_rounds: int = 3

    # routing.py::route's margin above retrieval_gate: below retrieval_gate is
    # ROUTE_ABSTAIN, at or above retrieval_gate + this margin is ROUTE_SINGLE
    # (confident enough that iterating is not worth its extra generation
    # calls), and the band in between is ROUTE_ITERATIVE. 0.15 was chosen to
    # be roughly half the ~0.72 in-corpus vs ~0.08 out-of-corpus separation
    # retrieval_gate itself is calibrated against (see retrieval_gate's
    # docstring above) -- wide enough to catch queries that are genuinely
    # borderline, not so wide that ROUTE_SINGLE only ever fires on the
    # easiest queries.
    route_iterate_margin: float = 0.15

    def __post_init__(self):
        if self.chunk_window < 1 or self.chunk_stride < 1:
            raise ValueError("chunk_window and chunk_stride must both be >= 1")
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.retrieval_mode not in ("dense", "sparse", "hybrid"):
            raise ValueError(
                f"retrieval_mode must be one of 'dense', 'sparse', 'hybrid'; got {self.retrieval_mode!r}")
        if any(w <= 0 for w in self.weights.values()):
            raise ValueError("all aggregation weights must be > 0")
        if self.answer_relevance_n_questions < 1:
            raise ValueError("answer_relevance_n_questions must be >= 1")
        if self.agentic_max_rounds < 1:
            raise ValueError("agentic_max_rounds must be >= 1")
