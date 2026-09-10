"""On-disk store for uploaded corpora, each indexed independently of the
bundled demo corpus so a client can query its own document without redeploying
the service.

One directory per corpus under `root`:

    <root>/<corpus_id>/source.<ext>      raw upload, kept so a corpus can be re-indexed later
    <root>/<corpus_id>/passages.json     written by RAGTrustPipeline.save()
    <root>/<corpus_id>/index.faiss       likewise
    <root>/<corpus_id>/meta.json         CorpusRecord fields
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .ingest.validate import UploadRejected, safe_display_name, validate_upload
from .pipeline import RAGTrustPipeline

# uuid4().hex is always exactly 32 lowercase hex characters, so a well-formed
# corpus id can never contain a path separator, "..", or anything else that
# would let it escape `root` when joined into a path.
CORPUS_ID_RE = re.compile(r"\A[0-9a-f]{32}\Z")


@dataclass(frozen=True)
class CorpusRecord:
    corpus_id: str
    filename: str
    bytes: int
    passages: int
    embed_model: str
    created_at: str  # ISO-8601 UTC, trailing "Z"

    def to_dict(self) -> dict:
        return asdict(self)


class CorpusStore:
    def __init__(self, root: str, *, ttl_hours: float = None, max_corpora: int = None,
                 cache_size: int = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = (
            ttl_hours if ttl_hours is not None
            else float(os.environ.get("RAGTRUST_CORPUS_TTL_HOURS", "24"))
        )
        self.max_corpora = (
            max_corpora if max_corpora is not None
            else int(os.environ.get("RAGTRUST_MAX_CORPORA", "20"))
        )
        self.cache_size = (
            cache_size if cache_size is not None
            else int(os.environ.get("RAGTRUST_CORPUS_CACHE", "4"))
        )
        # Embedding every chunk of an upload is the CPU-bound step of `create`,
        # and the target VPS has 2 vCPUs -- letting two uploads embed at once
        # would just make both slower rather than either faster. A semaphore
        # of 1 makes concurrent uploads queue instead of compete for the CPU.
        self._index_semaphore = threading.Semaphore(1)
        # uvicorn runs sync route handlers in a threadpool, so cache access is
        # genuinely concurrent -- this lock protects the LRU's read-modify-write
        # (move-to-end / evict-oldest) from interleaving across threads.
        self._cache_lock = threading.Lock()
        self._cache: "OrderedDict[str, RAGTrustPipeline]" = OrderedDict()

    # ------------------------------------------------------------------- paths

    def _resolve(self, corpus_id: str) -> Path:
        # Validated BEFORE it touches a path. A client-supplied corpus_id that
        # fails this shape check is rejected here rather than sanitised, so a
        # value like "../../etc" or "x/y" can never reach Path(root) / corpus_id
        # in the first place -- path traversal is structurally impossible, not
        # merely guarded against.
        if not CORPUS_ID_RE.match(corpus_id or ""):
            raise KeyError(corpus_id)
        return self.root / corpus_id

    # ------------------------------------------------------------------ create

    def create(self, filename: str, data: bytes, base_pipeline: RAGTrustPipeline) -> CorpusRecord:
        suffix = validate_upload(filename, data)
        display_name = safe_display_name(filename)
        self.sweep()

        # The id is a fresh uuid4, never derived from `filename` or any other
        # client-controlled input. That is what makes path traversal
        # structurally impossible rather than something to be sanitised away --
        # there is no string from the request that ever becomes part of a path.
        # `filename` is kept only in meta.json, for display.
        corpus_id = uuid4().hex
        corpus_dir = self.root / corpus_id
        corpus_dir.mkdir(parents=True, exist_ok=False)

        try:
            (corpus_dir / f"source{suffix}").write_bytes(data)

            with self._index_semaphore:
                pipeline = RAGTrustPipeline(
                    base_pipeline.config, embedder=base_pipeline.embedder,
                )
                try:
                    pipeline.index_corpus(str(corpus_dir / f"source{suffix}"))
                except ValueError as exc:
                    # `_install` raises this when chunking produced zero passages --
                    # most commonly a scanned PDF with no extractable text layer (an
                    # image of text, not text). Left uncaught, `create` would still
                    # succeed and leave behind an index with nothing in it, which
                    # then abstains on every question -- indistinguishable from a
                    # broken service rather than what it actually is, an unreadable
                    # upload.
                    raise UploadRejected(
                        "The upload produced no indexable content (likely a scanned "
                        "PDF with no extractable text layer, or an empty document)."
                    ) from exc
                # index_corpus() stamps every passage with Path(path).name as its
                # source -- which here is the internal storage name, "source.pdf",
                # identical for every upload. Left alone, a user who uploads
                # quarterly-report.pdf sees "source.pdf" beside each citation, and
                # two different uploads become indistinguishable in the dashboard.
                # Provenance is the thing this project is most specific about, so
                # the display name is restored before the index is persisted.
                for entry in pipeline.passage_meta.values():
                    entry["source"] = display_name
                pipeline.save(str(corpus_dir))
            record = CorpusRecord(
                corpus_id=corpus_id,
                filename=display_name,
                bytes=len(data),
                passages=len(pipeline.passages_text),
                embed_model=base_pipeline.config.embed_model,
                created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            )
            # Written INSIDE the try so a failure here is cleaned up like any
            # other. A corpus dir without a readable meta.json is invisible to
            # list() -- and therefore to sweep(), which prunes via list() -- so
            # leaving one behind would be an orphan that no retention policy can
            # ever reclaim and that does not count against max_corpora.
            (corpus_dir / "meta.json").write_text(json.dumps(record.to_dict()))
        except Exception:
            shutil.rmtree(corpus_dir, ignore_errors=True)
            raise

        return record

    # -------------------------------------------------------------------- read

    def list(self) -> list:
        records = []
        if not self.root.is_dir():
            return records
        for entry in self.root.iterdir():
            if not entry.is_dir():
                continue
            try:
                data = json.loads((entry / "meta.json").read_text())
                record = CorpusRecord(**data)
            except (OSError, ValueError, TypeError):
                # Missing/corrupt meta.json (e.g. a partially-written dir from a
                # crash mid-create) -- skip it rather than let one bad corpus
                # take down the whole listing.
                continue
            # `corpus_id` here comes from a FILE's contents, not from the directory
            # name, and sweep()/_delete_quietly() feed it straight to rmtree. Every
            # meta.json this code writes holds a uuid4 hex matching its own directory,
            # so this can only diverge if something outside this class wrote one --
            # but "a value that reaches rmtree must be a validated id" should hold at
            # every point it is true anywhere, not only in _resolve(). Anchoring it to
            # the directory name keeps the traversal guarantee whole.
            if record.corpus_id != entry.name or not CORPUS_ID_RE.match(entry.name):
                continue
            records.append(record)
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records

    def get(self, corpus_id: str) -> CorpusRecord:
        corpus_dir = self._resolve(corpus_id)
        try:
            data = json.loads((corpus_dir / "meta.json").read_text())
        except (OSError, ValueError):
            raise KeyError(corpus_id)
        return CorpusRecord(**data)

    # ------------------------------------------------------------------ delete

    def delete(self, corpus_id: str) -> None:
        corpus_dir = self._resolve(corpus_id)
        if not corpus_dir.is_dir():
            raise KeyError(corpus_id)
        with self._cache_lock:
            self._cache.pop(corpus_id, None)
        shutil.rmtree(corpus_dir)

    # ------------------------------------------------------------------- sweep

    def sweep(self) -> int:
        """Delete corpora older than the TTL, then oldest-first while the
        count still exceeds max_corpora. Returns the number removed."""
        records = self.list()
        removed = 0

        if self.ttl_hours > 0:
            cutoff = datetime.now(timezone.utc).timestamp() - self.ttl_hours * 3600
            survivors = []
            for r in records:
                try:
                    age = datetime.strptime(r.created_at, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
                        tzinfo=timezone.utc
                    ).timestamp()
                except ValueError:
                    # An unparseable timestamp must not take out the sweep -- sweep()
                    # runs on the upload path, so raising here would turn one damaged
                    # meta.json into "no one can upload anything, ever". Treat it as
                    # not-yet-expired and let the max_corpora cap below reclaim it,
                    # which is the outcome that loses no recoverable data.
                    survivors.append(r)
                    continue
                if age < cutoff:
                    self._delete_quietly(r.corpus_id)
                    removed += 1
                else:
                    survivors.append(r)
            records = survivors

        # `records` is newest-first (see list()); drop from the tail (oldest)
        # while still over the cap.
        while len(records) > self.max_corpora:
            oldest = records.pop()
            self._delete_quietly(oldest.corpus_id)
            removed += 1

        return removed

    def _delete_quietly(self, corpus_id: str) -> None:
        with self._cache_lock:
            self._cache.pop(corpus_id, None)
        shutil.rmtree(self.root / corpus_id, ignore_errors=True)

    # -------------------------------------------------------------- pipelines

    def pipeline_for(self, corpus_id: str, base_pipeline: RAGTrustPipeline) -> RAGTrustPipeline:
        """A pipeline that answers against this corpus instead of the base one.

        Contrast with `service.py::_pipeline_for_k`, which deliberately SHARES
        the retriever with the base pipeline because only `k` differs there.
        Here the whole corpus differs, so the retriever, passages and metadata
        must NOT be shared -- only the expensive model objects (_generator,
        _nli, _embedder) are aliased from `base_pipeline`, so switching corpora
        never reloads a model, but each corpus keeps its own index.
        """
        with self._cache_lock:
            cached = self._cache.get(corpus_id)
            if cached is not None:
                self._cache.move_to_end(corpus_id)
                return cached

        corpus_dir = self._resolve(corpus_id)
        if not corpus_dir.is_dir():
            raise KeyError(corpus_id)

        pipeline = RAGTrustPipeline(
            base_pipeline.config,
            generator=base_pipeline._generator,
            nli=base_pipeline._nli,
            embedder=base_pipeline._embedder,
        )
        pipeline.load(str(corpus_dir))

        with self._cache_lock:
            self._cache[corpus_id] = pipeline
            self._cache.move_to_end(corpus_id)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

        return pipeline
