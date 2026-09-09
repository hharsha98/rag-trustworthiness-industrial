"""HTTP service for ragtrust.

Wraps `RAGTrustPipeline` behind FastAPI: the index is loaded once, at app
creation, and every request reuses the same retriever/nli/embedder/generator.
An abstention is a successful response (200, `abstained: true`) -- declining
to answer is correct behaviour, not a failure. Only an unreachable generator
backend or a bad request produce a non-200 status.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .config import Config
from .generation.base import GenerationError
from .pipeline import RAGTrustPipeline

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

    def _pipeline_for_k(k: Optional[int]) -> RAGTrustPipeline:
        """A pipeline reflecting a per-request `k` override, sharing every
        heavy resource (retriever/index, embedder, nli, generator) with the
        base pipeline so an override never re-embeds or reloads anything."""
        base = app.state.pipeline
        if k is None or k == base.config.k:
            return base
        scoped = RAGTrustPipeline(
            replace(base.config, k=k),
            generator=base._generator, nli=base._nli, embedder=base._embedder,
        )
        scoped._retriever = base._retriever
        scoped.passages_text = base.passages_text
        scoped.passage_meta = base.passage_meta
        return scoped

    @app.get("/health", response_model=HealthResponse)
    def health() -> dict:
        p = app.state.pipeline
        return {
            "status": "ok",
            "passages": len(p.passages_text),
            "embed_model": p.config.embed_model,
            "nli_model": p.config.nli_model,
        }

    @app.get("/config")
    def get_config() -> dict:
        return asdict(app.state.pipeline.config)

    @app.post("/answer", response_model=AnswerResponse)
    def answer(req: AnswerRequest) -> dict:
        active = _pipeline_for_k(req.k)
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
