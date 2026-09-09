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
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .generation.base import GenerationError
from .ingest.loader import chunk_passages, load_corpus
from .metrics.aggregate import aggregate_arithmetic, aggregate_geometric
from .metrics.attribution import attribution
from .metrics.claims import split_claims
from .metrics.conciseness import conciseness
from .metrics.faithfulness import FaithfulnessResult, faithfulness
from .metrics.relevance import answer_relevance, context_relevance, max_context_similarity
from .retrieval.index import Retriever

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

    def index_corpus(self, path: str) -> "RAGTrustPipeline":
        """Index one corpus file (.pdf, .md or .txt) using windowed chunking."""
        chunks = chunk_passages(
            load_corpus(str(path)),
            window=self.config.chunk_window,
            stride=self.config.chunk_stride,
            min_chars=self.config.chunk_min_chars,
        )
        name = Path(path).name
        meta = [{"source": name, "page": c["page"]} for c in chunks]
        return self._install([c["text"] for c in chunks], meta)

    def index_dir(self, directory: str, pattern: str = "*") -> "RAGTrustPipeline":
        """Index every supported document under `directory`, keeping provenance."""
        root = Path(directory)
        files = sorted(p for p in root.rglob(pattern)
                       if p.is_file() and p.suffix.lower() in CORPUS_SUFFIXES)
        if not files:
            raise FileNotFoundError(
                f"No {'/'.join(CORPUS_SUFFIXES)} files found under {root}")
        texts: list = []
        meta: list = []
        for f in files:
            chunks = chunk_passages(
                load_corpus(str(f)),
                window=self.config.chunk_window,
                stride=self.config.chunk_stride,
                min_chars=self.config.chunk_min_chars,
            )
            texts += [c["text"] for c in chunks]
            meta += [{"source": str(f.relative_to(root)), "page": c["page"]} for c in chunks]
        return self._install(texts, meta)

    def index_pdf(self, path: str) -> "RAGTrustPipeline":
        """Backwards-compatible alias for `index_corpus`.

        This previously segmented by sentence, which on a slide-style corpus produced
        heading-sized fragments and made retrieval near-useless. It now chunks.
        """
        return self.index_corpus(path)

    def index_texts(self, texts: list, meta: list = None) -> "RAGTrustPipeline":
        """Index passages the caller has already prepared. No further chunking."""
        return self._install(list(texts), meta)

    def _install(self, texts: list, meta: list = None) -> "RAGTrustPipeline":
        if not texts:
            raise ValueError("Refusing to build an empty index.")
        self.passages_text = texts
        self.passage_meta = {i: (meta[i] if meta else {}) for i in range(len(texts))}
        self.retriever.build(self.passages_text)
        return self

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
        import faiss

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        (out / "passages.json").write_text(json.dumps({
            "embed_model": self.config.embed_model,
            "retrieval_mode": self.config.retrieval_mode,
            "passages": self.passages_text,
            "meta": {str(k): v for k, v in self.passage_meta.items()},
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
        import faiss

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

        retrieved = self.retriever.search(query, self.config.k)
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
        top = max_context_similarity(query, passage_texts, self.embedder)
        if top < self.config.retrieval_gate:
            return self._declined(
                retrieved, sources,
                f"No passage retrieved above the relevance gate "
                f"(best similarity {top:.3f} < {self.config.retrieval_gate}).",
                {"relevance": context_relevance(query, passage_texts, self.embedder)},
            )

        generated = self._generator.generate(query, retrieved)
        claims = split_claims(generated.text)
        f_result = faithfulness(claims, passage_texts, self.nli)

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
            )

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

        return AnswerResult(
            answer=generated.text, passages=retrieved, claims=claims,
            citations=generated.citations, metrics=metrics, abstained=False,
            faithfulness=f_result, trust=trust, sources=sources,
        )

    def _declined(self, retrieved, sources, reason, metrics, faithfulness_result=None):
        empty = faithfulness_result or FaithfulnessResult(
            score=0.0, per_claim=[], contradiction_rate=0.0, support_index=[])
        return AnswerResult(
            answer="Not answerable from this corpus.",
            passages=retrieved, claims=[], citations={}, metrics=metrics,
            abstained=True, faithfulness=empty, abstain_reason=reason,
            trust={"arithmetic": 0.0, "geometric": 0.0, "weights": dict(self.config.weights)},
            sources=sources,
        )
