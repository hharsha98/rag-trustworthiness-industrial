"""HTTP service for ragtrust.

Wraps `RAGTrustPipeline` behind FastAPI: the index is loaded once, at app
creation, and every request reuses the same retriever/nli/embedder/generator.
An abstention is a successful response (200, `abstained: true`) -- declining
to answer is correct behaviour, not a failure. Only an unreachable generator
backend or a bad request produce a non-200 status.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .config import Config
from .corpora import CorpusStore
from .generation.base import GenerationError
from .ingest.validate import UploadRejected, max_upload_bytes
from .pipeline import RAGTrustPipeline
from .ratelimit import SlidingWindowLimiter, client_key

logger = logging.getLogger(__name__)

# Repo root, resolved from this file's location rather than the process's
# current working directory -- `dashboard/` and `data/cached_answers.json` are
# both found relative to it, so `/api/examples` and the static mount work the
# same whether the service is launched via `ragtrust serve`, `uvicorn
# ragtrust.service:app` from the repo root (as the systemd unit in deploy/
# does), or pytest.
_REPO_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------------ models


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: Optional[int] = None
    # Absent (None) means "the bundled corpus loaded at create_app" -- every
    # existing client and test that never heard of corpus uploads keeps
    # answering against that corpus with no change in behaviour.
    corpus_id: Optional[str] = None

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty or whitespace-only")
        return value

    @field_validator("k")
    @classmethod
    def _k_positive(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 1:
            raise ValueError("k must be >= 1")
        return value


class AnswerResponse(BaseModel):
    answer: str
    abstained: bool
    abstain_reason: Optional[str] = None
    is_trustworthy: bool
    trust: dict
    metrics: dict
    claims: list
    citations: dict
    per_claim_support: list
    passages: list
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    passages: int
    embed_model: str
    nli_model: str
    corpora: int


# ------------------------------------------------------------------------- app


def _make_generator(name: str, model: str = None):
    """Build a generator by name, matching the CLI's `--generator` choices."""
    if name == "ollama":
        from .generation.ollama import DEFAULT_MODEL, OllamaGenerator

        return OllamaGenerator(model=model or DEFAULT_MODEL)
    if name == "hf_api":
        from .generation.hf_api import DEFAULT_MODEL, HFAPIGenerator

        return HFAPIGenerator(model=model or DEFAULT_MODEL)
    if name == "cached":
        from .generation.cached import CachedGenerator

        return CachedGenerator(model or "data/cached_answers.json")
    if name == "openai_compat":
        from .generation.openai_compat import DEFAULT_MODEL, OpenAICompatGenerator

        # base_url/api_key are read from RAGTRUST_LLM_BASE_URL/RAGTRUST_LLM_API_KEY
        # inside the constructor when left as None -- deployment (deploy/ragtrust.service)
        # configures those via its EnvironmentFile rather than a CLI/service argument.
        return OpenAICompatGenerator(model=model or DEFAULT_MODEL)
    raise ValueError(f"Unknown generator: {name!r}")


def create_app(index_dir: str, config: Config = None, generator: Any = None) -> FastAPI:
    """Build the FastAPI app, loading the index once so every request reuses
    the same retriever, embedder, NLI model and generator."""
    index_path = Path(index_dir)
    if not index_path.is_dir():
        raise FileNotFoundError(f"Index directory not found: {index_dir}")

    cfg = config or Config()
    pipeline = RAGTrustPipeline(cfg, generator=generator)
    pipeline.load(str(index_path))  # raises ValueError on embed-model mismatch

    app = FastAPI(title="ragtrust", description=(
        "Trustworthiness-scored RAG: answers carry the passages they came "
        "from, per-claim grounding, and both trust aggregates."
    ))
    app.state.pipeline = pipeline
    # Uploaded corpora persist across restarts under this directory (default
    # var/corpora, inside the repo) unless RAGTRUST_UPLOAD_DIR points somewhere
    # else -- deploy/ragtrust.service points it at /var/lib/ragtrust/corpora,
    # the one path the hardened systemd unit grants write access to.
    upload_root = os.environ.get("RAGTRUST_UPLOAD_DIR") or str(_REPO_ROOT / "var" / "corpora")
    app.state.corpus_store = CorpusStore(upload_root)

    # --- Upload abuse protection. The service is about to be exposed publicly
    # with no authentication, and POST /corpora both writes to disk and spends
    # CPU indexing -- see ratelimit.py's module docstring for the full picture.
    # Read once at app-creation time (not per-request) so a single process's
    # policy is stable for its lifetime; tests that need a different policy
    # build a fresh app rather than mutating environ mid-run.
    upload_rate_limit = int(os.environ.get("RAGTRUST_UPLOAD_RATE_LIMIT", "5"))
    upload_rate_window = float(os.environ.get("RAGTRUST_UPLOAD_RATE_WINDOW", "3600"))
    upload_global_limit = int(os.environ.get("RAGTRUST_UPLOAD_GLOBAL_LIMIT", "30"))
    app.state.upload_limiter = SlidingWindowLimiter(upload_rate_limit, upload_rate_window)
    # A separate limiter keyed by one constant, not per-client. A per-client
    # limit assumes source addresses are a scarce resource; under IPv6 a
    # single host routinely controls a /64 -- billions of addresses -- so
    # per-client limiting alone bounds nothing in the case that matters most.
    # This global cap is what actually bounds total disk and CPU spent on
    # uploads; the per-client limiter above is what keeps one ordinary
    # (non-adversarial, single-address) abuser from consuming the whole
    # global budget by itself.
    app.state.upload_global_limiter = SlidingWindowLimiter(upload_global_limit, upload_rate_window)
    _GLOBAL_UPLOAD_KEY = "__global__"
    # False by default: trusting X-Forwarded-For unconditionally would let any
    # direct caller forge its own client identity and bypass the per-client
    # limiter entirely (see client_key's docstring). Set RAGTRUST_TRUST_PROXY=true
    # only when Caddy (deploy/Caddyfile.snippet) is the sole path to this
    # process and is known to overwrite the header on every request.
    trust_proxy = os.environ.get("RAGTRUST_TRUST_PROXY", "").strip().lower() in ("1", "true", "yes")
    # Unset or empty means the feature is OFF and uploads stay public -- the
    # demo (this dashboard, open with no login) has to keep working for
    # someone who just loads the page with no token in hand. This default is
    # deliberate, not an oversight: an operator who wants the token enforced
    # opts in by setting it.
    upload_token = os.environ.get("RAGTRUST_UPLOAD_TOKEN", "")

    def _retry_after_message(limit: int, window_seconds: float, retry_after: int) -> str:
        """Render `detail` for a 429 in plain language, per-hour phrasing when
        the window is an hour (the default and the common case), generic
        otherwise. Rounds the wait to whatever unit reads most sensibly."""
        if window_seconds == 3600:
            window_desc = "per hour"
        elif window_seconds == 60:
            window_desc = "per minute"
        else:
            window_desc = f"per {int(window_seconds)}s"
        if retry_after >= 3600:
            wait_desc = f"{round(retry_after / 3600)} hour(s)"
        elif retry_after >= 60:
            wait_desc = f"{round(retry_after / 60)} minute(s)"
        else:
            wait_desc = f"{retry_after} second(s)"
        return f"Upload limit reached ({limit} {window_desc}). Try again in {wait_desc}."

    def _pipeline_for_request(corpus_id: Optional[str], k: Optional[int]) -> RAGTrustPipeline:
        """Select the corpus pipeline (the bundled base one when `corpus_id`
        is None), then apply the `k` override to whichever pipeline that was."""
        if corpus_id is None:
            base = app.state.pipeline
        else:
            base = app.state.corpus_store.pipeline_for(corpus_id, app.state.pipeline)

        # A pipeline reflecting a per-request `k` override, sharing every
        # heavy resource (retriever/index, embedder, nli, generator) with the
        # selected pipeline so an override never re-embeds or reloads anything.
        if k is None or k == base.config.k:
            return base
        scoped = RAGTrustPipeline(
            replace(base.config, k=k),
            generator=base._generator, nli=base._nli, embedder=base._embedder,
        )
        scoped._retriever = base._retriever
        scoped.passages_text = base.passages_text
        scoped.passage_meta = base.passage_meta
        # Without this, `scoped.passage_source_text` stays at its `__init__`
        # default ([]), so `scoped.source_text(id)` would fall through to
        # `scoped.passages_text[id]` -- the contextualised (blurb-prefixed)
        # text under Config.contextual=True -- for every k-override request.
        # That would hand the LLM-written blurb to NLI/generation/citations via
        # this code path alone, reintroducing the exact defect `answer()`'s
        # `replace(p, text=self.source_text(p.id))` line exists to prevent.
        scoped.passage_source_text = base.passage_source_text
        return scoped

    @app.get("/health", response_model=HealthResponse)
    def health() -> dict:
        p = app.state.pipeline
        return {
            "status": "ok",
            "passages": len(p.passages_text),
            "embed_model": p.config.embed_model,
            "nli_model": p.config.nli_model,
            "corpora": len(app.state.corpus_store.list()),
        }

    @app.get("/config")
    def get_config() -> dict:
        return asdict(app.state.pipeline.config)

    @app.post("/answer", response_model=AnswerResponse)
    def answer(req: AnswerRequest) -> dict:
        try:
            active = _pipeline_for_request(req.corpus_id, req.k)
        except KeyError:
            # Covers both an unknown-but-well-formed id and a malformed one
            # (CorpusStore._resolve raises KeyError for both) -- either way the
            # client asked for a corpus that is not there to answer from, which
            # is a 404, not a 500.
            raise HTTPException(status_code=404, detail=f"No such corpus: {req.corpus_id}")
        start = time.perf_counter()
        try:
            result = active.answer(req.question)
        except GenerationError as exc:
            raise HTTPException(status_code=503, detail=f"Generation backend unreachable: {exc}")
        except ValueError as exc:
            # `create_app`'s `generator` argument defaults to None, so an app can
            # be built that serves /health and /config but cannot answer; the
            # pipeline raises ValueError in that case. Uncaught, it surfaced as a
            # 500 with a traceback, which reads as a crash rather than a
            # configuration mistake. 503 with the reason is the honest status for
            # "this deployment is not able to answer right now".
            raise HTTPException(
                status_code=503,
                detail=f"Service is not configured to answer: {exc}",
            )
        latency_ms = (time.perf_counter() - start) * 1000.0

        payload = result.to_dict()
        payload["latency_ms"] = round(latency_ms, 2)
        return payload

    @app.get("/api/examples")
    def examples() -> list:
        """Preset questions for the dashboard's one-click examples: just the
        question text and its category from `data/cached_answers.json`, never
        the cached answer/passages (those stay internal to the `cached`
        generator backend). Returns [] if the file is absent or unreadable
        rather than erroring -- example presets are a UI nicety, not something
        that should be able to break the API."""
        path = _REPO_ROOT / "data" / "cached_answers.json"
        if not path.is_file():
            return []
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return []
        return [
            {"question": question, "category": entry.get("category") if isinstance(entry, dict) else None}
            for question, entry in data.items()
        ]

    # These routes must be registered before the StaticFiles mount below:
    # Starlette matches routes in registration order and that mount is a
    # catch-all at "/", so anything registered after it would never be reached.
    @app.post("/corpora", status_code=201)
    async def upload_corpus(request: Request, file: UploadFile = File(...)) -> dict:
        # Order matters: token check, then per-client limiter, then global
        # limiter, then the existing size/validation logic below (unchanged).
        # A request that presents a valid token is an authenticated operator,
        # not anonymous traffic, so it skips both limiters entirely -- rate
        # limits exist to bound abuse from callers who haven't proven
        # anything about themselves, not to throttle someone who has.
        if upload_token:
            provided = request.headers.get("X-Upload-Token", "")
            # hmac.compare_digest is constant-time: it always walks the full
            # length of both arguments, so how long the comparison takes does
            # not depend on how many leading bytes of `provided` happen to
            # match `upload_token`. A plain `==` short-circuits on the first
            # mismatch, which leaks the token one correct byte at a time to
            # an attacker who can measure response latency across many guesses.
            if not hmac.compare_digest(provided, upload_token):
                # Detail is a fixed string, never `provided` or `upload_token`
                # -- echoing either back (here or in a log line) would hand a
                # secret, or an attacker's best guess at one, to whoever can
                # read the response or the logs.
                raise HTTPException(status_code=401, detail="Missing or invalid upload token.")
        else:
            per_client_key = client_key(request, trust_proxy)
            allowed, retry_after = app.state.upload_limiter.check(per_client_key)
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail=_retry_after_message(upload_rate_limit, upload_rate_window, retry_after),
                    headers={"Retry-After": str(retry_after)},
                )
            # Checked second, only once the per-client limiter has already
            # let this request through -- see the comment on this limiter's
            # construction above for why it exists in addition to, not
            # instead of, the per-client one.
            allowed, retry_after = app.state.upload_global_limiter.check(_GLOBAL_UPLOAD_KEY)
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail=_retry_after_message(upload_global_limit, upload_rate_window, retry_after),
                    headers={"Retry-After": str(retry_after)},
                )

        limit = max_upload_bytes()
        chunks = []
        total = 0
        # Content-Length is client-supplied and cannot be trusted to match the
        # actual body -- a client can lie about it (or omit it, under chunked
        # transfer encoding) and send more bytes anyway. Enforcing the limit
        # against bytes actually read, and aborting mid-stream the moment they
        # cross it, means a hostile upload is capped at ~1 MiB over the limit
        # rather than fully buffered into memory first.
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds the {limit}-byte limit.",
                )
            chunks.append(chunk)
        data = b"".join(chunks)

        start = time.perf_counter()
        try:
            # `create` chunks and embeds the whole document -- seconds to minutes of
            # blocking CPU work. This route has to be `async def` to stream the body
            # above (`await file.read`), which means its body runs ON the event loop,
            # so calling `create` directly here would stall every other request in the
            # process -- /health, /answer, the dashboard's static files -- for the full
            # duration of the indexing. The other routes in this module are plain
            # `def`, which FastAPI already runs in a threadpool; this hands `create`
            # to that same threadpool so an upload is slow only for the uploader.
            record = await run_in_threadpool(
                app.state.corpus_store.create, file.filename, data, app.state.pipeline,
            )
        except UploadRejected as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        seconds = time.perf_counter() - start

        payload = record.to_dict()
        payload["seconds"] = round(seconds, 2)
        return payload

    @app.get("/corpora")
    def list_corpora() -> list:
        return [r.to_dict() for r in app.state.corpus_store.list()]

    @app.delete("/corpora/{corpus_id}", status_code=204)
    def delete_corpus(corpus_id: str) -> None:
        try:
            app.state.corpus_store.delete(corpus_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"No such corpus: {corpus_id}")

    @app.get("/corpora/limits")
    def upload_limits() -> dict:
        # Policy only -- never `upload_token` itself. This lets the dashboard
        # tell an anonymous visitor what the rules are (e.g. to render "5
        # uploads/hour" before they hit the limit) without exposing anything
        # that would let them bypass the token check.
        return {
            "per_client": upload_rate_limit,
            "global": upload_global_limit,
            "window_seconds": upload_rate_window,
            "token_required": bool(upload_token),
        }

    # Mounted LAST and at "/" so it never shadows the API routes above: Starlette
    # matches routes in registration order and returns on the first match, so
    # /health, /config, /answer, and /api/examples (all registered earlier) win
    # over this catch-all mount even though its prefix is "/". `html=True` makes
    # StaticFiles serve dashboard/index.html for "/" and other directory paths.
    # A missing dashboard/ must never take the API down with it -- log and move on.
    dashboard_dir = _REPO_ROOT / "dashboard"
    if dashboard_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")
    else:
        logger.warning(
            "dashboard/ directory not found at %s; serving API only (no frontend).",
            dashboard_dir,
        )

    return app


# --------------------------------------------------------------------- module app


class _LazyASGIApp:
    """Defers building the real app until the ASGI server actually calls it,
    so `import ragtrust.service` never fails just because RAGTRUST_INDEX is
    unset (e.g. under test collection or other tooling that merely imports
    this module). This is what makes `uvicorn ragtrust.service:app` work:
    uvicorn only requires an ASGI-callable object named `app`, not a
    fully-built FastAPI instance at import time."""

    def __init__(self):
        self._app: FastAPI = None

    def _ensure(self) -> FastAPI:
        if self._app is None:
            index_dir = os.environ.get("RAGTRUST_INDEX")
            if not index_dir:
                raise RuntimeError(
                    "RAGTRUST_INDEX environment variable must be set to run "
                    "`uvicorn ragtrust.service:app` directly (or use `ragtrust serve`)."
                )
            generator_name = os.environ.get("RAGTRUST_GENERATOR", "cached")
            model = os.environ.get("RAGTRUST_MODEL")
            generator = _make_generator(generator_name, model)
            self._app = create_app(index_dir, generator=generator)
        return self._app

    async def __call__(self, scope, receive, send):
        app = self._ensure()
        await app(scope, receive, send)


app = _LazyASGIApp()
