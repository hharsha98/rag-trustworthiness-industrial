"""Retrieval-augmented generation with trustworthiness scoring built into the answer path.

The design intent is that a caller cannot obtain an answer without also obtaining the
evidence for it and a measurement of how well that evidence supports it. `answer()` returns
one object carrying the answer, the passages it came from, per-claim grounding, the four
trust metrics, both aggregates, and whether the system declined.

Two abstention gates run in increasing cost order (see `Config`): a retrieval gate before
generation, and a grounding gate after it.
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

from .config import Config
from .generation.base import GenerationError
from .ingest.contextualize import contextualize_chunks
from .ingest.loader import chunk_passages, load_corpus
from .metrics.aggregate import aggregate_arithmetic, aggregate_geometric
from .metrics.attribution import attribution
from .metrics.claims import split_claims
from .metrics.conciseness import conciseness
from .metrics.faithfulness import FaithfulnessResult, faithfulness
from .metrics.relevance import answer_relevance, context_relevance, max_context_similarity
from .retrieval.index import Retriever, import_faiss
from .trace import StageTiming

CORPUS_SUFFIXES = (".pdf", ".md", ".txt")


@dataclass
class AnswerResult:
    """Everything needed to judge an answer, not just the answer."""

    answer: str
    passages: list
    claims: list
    citations: dict
    metrics: dict
    abstained: bool
    faithfulness: FaithfulnessResult
    abstain_reason: str = None
    trust: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    # Per-stage wall-clock timings from `answer`/`answer_with` (retrieve, gate,
    # generate, decompose, entail, score -- whichever ran before an abstention
    # gate cut the rest short, or all six on a full answer). This is what makes
    # dashboard/index.html's `.steps` row -- which already lists and animates
    # exactly these six stage names -- reflect what the pipeline actually spent
    # time on, rather than a fixed-duration animation with no data behind it.
    stage_timings: list = field(default_factory=list)

    @property
    def is_trustworthy(self) -> bool:
        """Conservative single-bit verdict: answered, and the non-compensatory
        aggregate cleared the grounding threshold."""
        return (not self.abstained) and self.trust.get("geometric", 0.0) >= 0.5

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "is_trustworthy": self.is_trustworthy,
            "trust": self.trust,
            "metrics": self.metrics,
            "claims": self.claims,
            "citations": {str(k): v for k, v in self.citations.items()},
            "per_claim_support": list(self.faithfulness.per_claim),
            "stage_timings": [{"stage": s.stage, "ms": round(s.ms, 3)} for s in self.stage_timings],
            "passages": [
                {"rank": i + 1,
                 "id": getattr(p, "id", None),
                 "score": round(float(getattr(p, "score", 0.0)), 4),
                 "source": self.sources.get(getattr(p, "id", None), {}),
                 "text": getattr(p, "text", "")}
                for i, p in enumerate(self.passages)
            ],
        }


class RAGTrustPipeline:
    def __init__(self, config: Config = None, generator=None, nli=None, embedder=None):
        self.config = config or Config()
        self._generator = generator
        self._nli = nli
        self._embedder = embedder
        self._retriever = None
        self.passages_text: list = []
        # The untouched chunk text, parallel to `passages_text` (same index).
        # When Config.contextual is off these are identical; when it's on,
        # `passages_text[i]` carries an LLM-written blurb for retrieval only
        # and `passage_source_text[i]` is what everything else must see --
        # see the invariant comment in `answer()` and in ingest/contextualize.py.
        self.passage_source_text: list = []
        # passage index -> {"source": str, "page": int}, for citing back to origin
        self.passage_meta: dict = {}

    # ------------------------------------------------------------------ models

    @property
    def nli(self):
        if self._nli is None:
            from .metrics.nli import NLIScorer

            self._nli = NLIScorer(self.config.nli_model)
        return self._nli

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.config.embed_model)
        return self._embedder

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = self._build_retriever()
        return self._retriever

    def _build_retriever(self):
        """Construct the retriever according to `Config.retrieval_mode`/`rerank`.

        Default config (`retrieval_mode="dense"`, `rerank=False`) reproduces the
        original single-line `Retriever(self.embedder, normalize=True)` exactly.
        """
        mode = self.config.retrieval_mode
        if mode == "dense":
            base = Retriever(self.embedder, normalize=True)
        elif mode == "sparse":
            from .retrieval.sparse import BM25Retriever

            base = BM25Retriever()
        elif mode == "hybrid":
            from .retrieval.hybrid import HybridRetriever
            from .retrieval.sparse import BM25Retriever

            base = HybridRetriever(Retriever(self.embedder, normalize=True), BM25Retriever())
        else:  # pragma: no cover -- Config.__post_init__ already validates this
            raise ValueError(f"Unknown retrieval_mode: {mode!r}")

        if self.config.rerank:
            from .retrieval.rerank import CrossEncoderReranker

            return CrossEncoderReranker(
                base, model_name=self.config.rerank_model,
                candidates=self.config.rerank_candidates,
            )
        return base

    # ----------------------------------------------------------------- indexing

    def _contextualize_if_enabled(self, chunks: list, document_text: str) -> tuple:
        """Apply Contextual Retrieval to `chunks` when `Config.contextual` is on
        and a generator is available; otherwise a no-op.

        Returns `(texts, source_texts)`. When contextualisation does not run,
        `source_texts` is `None` so `_install` falls back to `texts == source_texts`
        -- the same behaviour as before this feature existed, byte-for-byte,
        which is the point: `Config.contextual` defaults to False specifically
        so existing callers see no change (see Config.contextual's docstring).
        """
        if not (self.config.contextual and self._generator is not None):
            return [c["text"] for c in chunks], None
        contextualized = contextualize_chunks(
            chunks, document_text, self._generator,
            model_tag=self.config.contextual_model,
        )
        return (
            [c["text"] for c in contextualized],
            [c["source_text"] for c in contextualized],
        )

    def index_corpus(self, path: str) -> "RAGTrustPipeline":
        """Index one corpus file (.pdf, .md or .txt) using windowed chunking."""
        pages = load_corpus(str(path))
        chunks = chunk_passages(
            pages,
            window=self.config.chunk_window,
            stride=self.config.chunk_stride,
            min_chars=self.config.chunk_min_chars,
        )
        name = Path(path).name
        meta = [{"source": name, "page": c["page"]} for c in chunks]
        texts, source_texts = self._contextualize_if_enabled(chunks, "\n".join(pages))
        return self._install(texts, meta, source_texts)

    def index_dir(self, directory: str, pattern: str = "*") -> "RAGTrustPipeline":
        """Index every supported document under `directory`, keeping provenance."""
        root = Path(directory)
        files = sorted(p for p in root.rglob(pattern)
                       if p.is_file() and p.suffix.lower() in CORPUS_SUFFIXES)
        if not files:
            raise FileNotFoundError(
                f"No {'/'.join(CORPUS_SUFFIXES)} files found under {root}")
        texts: list = []
        source_texts: list = []
        meta: list = []
        for f in files:
            pages = load_corpus(str(f))
            chunks = chunk_passages(
                pages,
                window=self.config.chunk_window,
                stride=self.config.chunk_stride,
                min_chars=self.config.chunk_min_chars,
            )
            file_texts, file_source_texts = self._contextualize_if_enabled(chunks, "\n".join(pages))
            texts += file_texts
            source_texts += file_source_texts if file_source_texts is not None else file_texts
            meta += [{"source": str(f.relative_to(root)), "page": c["page"]} for c in chunks]
        return self._install(texts, meta, source_texts)

    def index_pdf(self, path: str) -> "RAGTrustPipeline":
        """Backwards-compatible alias for `index_corpus`.

        This previously segmented by sentence, which on a slide-style corpus produced
        heading-sized fragments and made retrieval near-useless. It now chunks.
        """
        return self.index_corpus(path)

    def index_texts(self, texts: list, meta: list = None) -> "RAGTrustPipeline":
        """Index passages the caller has already prepared. No further chunking."""
        return self._install(list(texts), meta)

    def _install(self, texts: list, meta: list = None, source_texts: list = None) -> "RAGTrustPipeline":
        if not texts:
            raise ValueError("Refusing to build an empty index.")
        self.passages_text = texts
        # No `source_texts` (the common, non-contextual case) means retrieval text
        # IS the source text -- there is nothing an LLM added to strip back out.
        self.passage_source_text = list(source_texts) if source_texts else list(texts)
        self.passage_meta = {i: (meta[i] if meta else {}) for i in range(len(texts))}
        self.retriever.build(self.passages_text)
        return self

    def source_text(self, passage_id: int) -> str:
        """The untouched chunk for `passage_id`, never the contextualised blurb.

        Falls back to `passages_text[passage_id]` only when `passage_id` is out
        of range for `passage_source_text` -- e.g. an index persisted before
        this feature existed and loaded via `load()`'s backward-compat path.
        """
        if 0 <= passage_id < len(self.passage_source_text):
            return self.passage_source_text[passage_id]
        return self.passages_text[passage_id]

    # -------------------------------------------------------------- persistence

    @staticmethod
    def _dense_component(retriever):
        """Return the dense `Retriever` inside a possibly-wrapped retriever, or None.

        Retrievers compose: a reranker wraps a base retriever, and a hybrid wraps a
        dense and a sparse one. Only the dense component owns a FAISS index worth
        persisting -- BM25 is term statistics and rebuilds from the passages in
        milliseconds. Walking the wrappers keeps persistence working for every
        `retrieval_mode`, rather than only for the unwrapped dense case.
        """
        seen = set()
        node = retriever
        while node is not None and id(node) not in seen:
            seen.add(id(node))
            if getattr(node, "index", None) is not None:
                return node
            node = getattr(node, "base_retriever", None) or getattr(node, "dense", None)
        return None

    def save(self, directory: str) -> "RAGTrustPipeline":
        """Persist the index so a restart does not re-embed the whole corpus."""
        faiss = import_faiss()

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        (out / "passages.json").write_text(json.dumps({
            "embed_model": self.config.embed_model,
            "retrieval_mode": self.config.retrieval_mode,
            "passages": self.passages_text,
            "meta": {str(k): v for k, v in self.passage_meta.items()},
            "source_passages": self.passage_source_text,
        }))
        dense = self._dense_component(self.retriever)
        if dense is None:
            raise ValueError(
                "This retriever exposes no dense component, so there is no embedding "
                "index to persist. Sparse-only retrieval rebuilds from the passages; "
                "call index_texts() instead of load()."
            )
        faiss.write_index(dense.index, str(out / "index.faiss"))
        return self

    def load(self, directory: str) -> "RAGTrustPipeline":
        faiss = import_faiss()

        src = Path(directory)
        payload = json.loads((src / "passages.json").read_text())
        stored = payload.get("embed_model")
        if stored and stored != self.config.embed_model:
            raise ValueError(
                f"Index was built with embed_model {stored!r} but this Config uses "
                f"{self.config.embed_model!r}. Vectors from different encoders are not "
                f"comparable -- rebuild the index, or match the model.")
        self.passages_text = payload["passages"]
        self.passage_meta = {int(k): v for k, v in payload.get("meta", {}).items()}
        # `source_passages` postdates this feature -- an index written before it
        # existed has no such key, and its `passages` were never contextualised
        # anyway, so falling back to them there keeps a pre-existing index file
        # loadable without a migration step.
        self.passage_source_text = payload.get("source_passages") or list(self.passages_text)

        retriever = self.retriever
        dense = self._dense_component(retriever)
        if dense is None:
            # No embedding index to restore (sparse-only): rebuild from the passages,
            # which for BM25 is just term counting.
            retriever.build(self.passages_text)
            return self

        # Restore the dense index from disk rather than re-embedding, then rebuild only
        # the cheap components. A sparse sibling has to be built here because its term
        # statistics were never persisted.
        dense.passages = list(self.passages_text)
        dense.index = faiss.read_index(str(src / "index.faiss"))
        sparse = getattr(retriever, "sparse", None) or getattr(
            getattr(retriever, "base_retriever", None), "sparse", None)
        if sparse is not None:
            sparse.build(self.passages_text)
        return self

    # -------------------------------------------------------------------- query

    def answer(self, query: str) -> AnswerResult:
        if self._generator is None:
            raise ValueError("A generator must be provided to produce an answer.")
        if not self.passages_text:
            raise ValueError("Nothing indexed. Call index_corpus/index_dir/load first.")

        # Timed separately from answer_with's own stages, not folded into a shared
        # loop there, because answer_with is also called directly by
        # agentic.py::answer_iterative with a pool the CALLER already assembled
        # from several retrieval rounds -- there is no single retriever.search()
        # call to time on that path, so "retrieve" only appears for this
        # single-shot entrypoint that owns it.
        retrieve_start = time.perf_counter()
        retrieved = self.retriever.search(query, self.config.k)
        retrieve_ms = (time.perf_counter() - retrieve_start) * 1000.0

        result = self.answer_with(query, retrieved)
        result.stage_timings.insert(0, StageTiming(stage="retrieve", ms=retrieve_ms))
        return result

    def answer_with(self, query: str, retrieved: list) -> AnswerResult:
        """Score and answer `query` against a passage pool the CALLER assembled,
        rather than retrieving internally. `answer()` is now a one-line wrapper
        that retrieves and delegates here.

        This split exists for agentic.py::answer_iterative, which unions passages
        across several retrieval rounds and needs to hand that pool to the same
        gating/generation/aggregation logic `answer()` uses for a single
        retrieval -- there was previously no seam for a caller to score a
        passage set it assembled itself.
        """
        if not retrieved:
            raise ValueError("Nothing retrieved to answer from.")

        # *** THE ENFORCEMENT POINT OF THE CONTEXTUAL-RETRIEVAL INVARIANT ***
        # `retrieved` carries passages tagged with whatever text was indexed --
        # under Config.contextual=True that is an LLM-WRITTEN blurb prepended to
        # the chunk (ingest/contextualize.py), because the blurb is what makes
        # retrieval better. That blurb is not a fact: the model can invent an
        # entity, a date, a relationship that isn't there. This line throws it
        # away and swaps in the untouched source chunk for every passage BEFORE
        # anything below reads `.text` -- the cosine gate, `generator.generate`,
        # NLI premises in `faithfulness`/`attribution`, citations, and
        # `to_dict()`. If a generated blurb ever reached the NLI model as an
        # entailment premise, a claim could be scored "faithful" because it is
        # entailed by text the model invented rather than by the corpus --
        # exactly the defect this repository exists to detect. Moving this line,
        # or reading retrieval-time text anywhere past it, silently reintroduces
        # that defect. When Config.contextual is off this is a no-op:
        # `source_text(id)` returns the same string `retriever.search` already
        # put in `p.text`.
        #
        # This line lives at the top of `answer_with`, not `answer`, so the
        # invariant holds for EVERY caller of `answer_with` -- including
        # agentic.py's iterative loop, which assembles `retrieved` itself from
        # several rounds of `retriever.search` and never goes through `answer()`
        # at all. Putting the swap only in `answer()` would let a raw,
        # blurb-carrying passage pool reach NLI/generation/citations for any
        # caller that bypasses `answer()`.
        retrieved = [replace(p, text=self.source_text(p.id)) for p in retrieved]
        passage_texts = [p.text for p in retrieved]
        sources = {getattr(p, "id", None): self.passage_meta.get(getattr(p, "id", None), {})
                   for p in retrieved}

        # Gate 1 -- before generation. Nothing in the corpus is close enough to be worth
        # spending a generation call on.
        #
        # The gate is computed from cosine similarity directly, NOT from `Passage.score`.
        # Retriever scores are not on a common scale: dense returns cosine in [-1,1], BM25
        # returns unbounded term-weight sums, RRF returns reciprocal-rank sums bounded by
        # about 2/(rrf_k+1) ~ 0.03, and a cross-encoder returns unbounded logits. A single
        # threshold compared against `score` therefore means something different in every
        # retrieval mode -- and with RRF it rejects everything, because no RRF score can
        # reach 0.25. Asking "is anything here semantically about the query?" is a cosine
        # question, so the gate asks it in cosine space regardless of how ranking happened.
        # Stage timings collected as the method proceeds, not restructured around --
        # each stage below wraps an already-existing step in a perf_counter pair and
        # appends one StageTiming; an abstention gate firing partway through simply
        # means the list handed to `_declined` stops at whichever stage ran last.
        stage_timings = []

        gate_start = time.perf_counter()
        top = max_context_similarity(query, passage_texts, self.embedder)
        stage_timings.append(StageTiming(stage="gate", ms=(time.perf_counter() - gate_start) * 1000.0))
        if top < self.config.retrieval_gate:
            return self._declined(
                retrieved, sources,
                f"No passage retrieved above the relevance gate "
                f"(best similarity {top:.3f} < {self.config.retrieval_gate}).",
                {"relevance": context_relevance(query, passage_texts, self.embedder)},
                stage_timings=stage_timings,
            )

        generate_start = time.perf_counter()
        generated = self._generator.generate(query, retrieved)
        stage_timings.append(StageTiming(stage="generate", ms=(time.perf_counter() - generate_start) * 1000.0))

        decompose_start = time.perf_counter()
        claims = split_claims(generated.text)
        stage_timings.append(StageTiming(stage="decompose", ms=(time.perf_counter() - decompose_start) * 1000.0))

        entail_start = time.perf_counter()
        f_result = faithfulness(claims, passage_texts, self.nli)
        stage_timings.append(StageTiming(stage="entail", ms=(time.perf_counter() - entail_start) * 1000.0))

        # Gate 2 -- after generation. The corpus looked relevant, but nothing actually
        # supports what the model produced.
        max_support = max(f_result.per_claim) if f_result.per_claim else 0.0
        if max_support < self.config.abstain_threshold:
            return self._declined(
                retrieved, sources,
                f"No claim in the generated answer is supported by the retrieved passages "
                f"(best support {max_support:.3f} < {self.config.abstain_threshold}).",
                {"faithfulness": f_result.score,
                 "contradiction_rate": f_result.contradiction_rate},
                faithfulness_result=f_result,
                stage_timings=stage_timings,
            )

        score_start = time.perf_counter()
        attr = attribution(claims, generated.citations, passage_texts, self.nli,
                           tau=self.config.support_threshold)

        # Answer relevance (METRICS.md Part II.3) is a diagnostic add-on, opt-in via
        # Config.answer_relevance because it issues a second generation call
        # (back-generation) per answer. It must NEVER be able to turn a good answer
        # into a failed one: a backend without `generate_questions`, a disabled
        # flag, a raised GenerationError, or an empty back-generation result all
        # degrade to `None` rather than raising or aborting `answer()`.
        ans_relevance_score = None
        if self.config.answer_relevance:
            generate_questions = getattr(self._generator, "generate_questions", None)
            if callable(generate_questions):
                try:
                    generated_questions = generate_questions(
                        generated.text, self.config.answer_relevance_n_questions)
                except Exception as exc:
                    warnings.warn(
                        f"answer_relevance back-generation failed ({exc.__class__.__name__}: "
                        f"{exc}); degrading metrics['answer_relevance'] to None."
                    )
                    generated_questions = None
                if generated_questions:
                    ans_relevance_score = answer_relevance(query, generated_questions, self.embedder)

        metrics = {
            "faithfulness": f_result.score,
            "contradiction_rate": f_result.contradiction_rate,
            "attribution": attr.f1,
            "attribution_precision": attr.precision,
            "attribution_recall": attr.recall,
            "relevance": context_relevance(query, passage_texts, self.embedder, scaled=True),
            "answer_relevance": ans_relevance_score,
            "conciseness": conciseness(claims, self.embedder),
        }
        # `metrics` keeps conciseness=None visible (e.g. < 2 claims -- undefined,
        # see metrics/conciseness.py) so callers can see it was undefined; only
        # aggregation drops None values, per aggregate.py's "dropped, not
        # weight 0" rule.
        scored = {k: metrics[k] for k in self.config.weights
                  if k in metrics and metrics[k] is not None}
        trust = {
            "arithmetic": aggregate_arithmetic(scored, self.config.weights),
            "geometric": aggregate_geometric(scored, self.config.weights),
            "weights": dict(self.config.weights),
        }
        stage_timings.append(StageTiming(stage="score", ms=(time.perf_counter() - score_start) * 1000.0))

        return AnswerResult(
            answer=generated.text, passages=retrieved, claims=claims,
            citations=generated.citations, metrics=metrics, abstained=False,
            faithfulness=f_result, trust=trust, sources=sources,
            stage_timings=stage_timings,
        )

    def _declined(self, retrieved, sources, reason, metrics, faithfulness_result=None,
                  stage_timings=None):
        empty = faithfulness_result or FaithfulnessResult(
            score=0.0, per_claim=[], contradiction_rate=0.0, support_index=[])
        return AnswerResult(
            answer="Not answerable from this corpus.",
            passages=retrieved, claims=[], citations={}, metrics=metrics,
            abstained=True, faithfulness=empty, abstain_reason=reason,
            trust={"arithmetic": 0.0, "geometric": 0.0, "weights": dict(self.config.weights)},
            sources=sources,
            # `stage_timings or []` rather than a mutable-default parameter (the
            # classic Python footgun: a `[]` default would be the SAME list object
            # shared across every call site that omits the argument). Every real
            # caller in this file passes its own list; `None` only shows up if
            # `_declined` is ever called directly (e.g. from a test) without one.
            stage_timings=stage_timings or [],
        )
