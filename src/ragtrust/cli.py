"""Command-line interface for ragtrust.

Wraps the library in `pipeline.py` / `config.py` -- it builds a `Config`,
drives a `RAGTrustPipeline`, and formats the result. No trustworthiness
logic lives here; this is presentation and argument handling only.

Subcommands: index, ask, serve, eval. See each `_cmd_*` function's docstring
or `ragtrust <subcommand> --help`.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from .config import Config
from .generation.base import GenerationError
from .pipeline import RAGTrustPipeline

DEFAULT_CACHE_PATH = "data/cached_answers.json"


class CLIError(Exception):
    """Raised for user-facing failures that should print a clean message
    (not a traceback) and exit non-zero."""


# --------------------------------------------------------------------------- config


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    """Shared flags for every subcommand that builds a Config."""
    defaults = Config()
    parser.add_argument("--embed-model", default=defaults.embed_model,
                         help=f"Sentence embedding model (default: {defaults.embed_model})")
    parser.add_argument("--nli-model", default=defaults.nli_model,
                         help=f"NLI model for faithfulness/attribution (default: {defaults.nli_model})")
    parser.add_argument("--k", type=int, default=defaults.k,
                         help=f"Passages to retrieve per query (default: {defaults.k})")
    parser.add_argument("--retrieval-gate", type=float, default=defaults.retrieval_gate,
                         help=f"Pre-generation relevance gate (default: {defaults.retrieval_gate})")
    parser.add_argument("--abstain-threshold", type=float, default=defaults.abstain_threshold,
                         help=f"Post-generation grounding gate (default: {defaults.abstain_threshold})")


def _build_config(args: argparse.Namespace) -> Config:
    return Config(
        embed_model=args.embed_model,
        nli_model=args.nli_model,
        k=args.k,
        retrieval_gate=args.retrieval_gate,
        abstain_threshold=args.abstain_threshold,
    )


def _add_generator_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--generator", choices=["ollama", "hf_api", "cached"], default="ollama",
                         help="Generation backend (default: ollama)")
    parser.add_argument("--model", default=None,
                         help="Model name for ollama/hf_api (ignored for cached)")
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH,
                         help=f"Path to cached answers JSON, used by --generator cached "
                              f"(default: {DEFAULT_CACHE_PATH})")


def _build_generator(args: argparse.Namespace):
    if args.generator == "ollama":
        from .generation.ollama import DEFAULT_MODEL, OllamaGenerator

        return OllamaGenerator(model=args.model or DEFAULT_MODEL)
    if args.generator == "hf_api":
        from .generation.hf_api import DEFAULT_MODEL, HFAPIGenerator

        return HFAPIGenerator(model=args.model or DEFAULT_MODEL)
    if args.generator == "cached":
        from .generation.cached import CachedGenerator

        return CachedGenerator(args.cache_path)
    raise CLIError(f"Unknown generator: {args.generator!r}")


def _require_index_dir(index_dir: str) -> Path:
    path = Path(index_dir)
    if not path.is_dir():
        raise CLIError(f"Index directory not found: {index_dir}")
    if not (path / "passages.json").exists() or not (path / "index.faiss").exists():
        raise CLIError(
            f"{index_dir} does not look like a ragtrust index "
            f"(missing passages.json / index.faiss). Build one with `ragtrust index`."
        )
    return path


def _load_pipeline(args: argparse.Namespace, generator=None) -> RAGTrustPipeline:
    index_dir = _require_index_dir(args.index)
    cfg = _build_config(args)
    pipeline = RAGTrustPipeline(cfg, generator=generator)
    try:
        pipeline.load(str(index_dir))
    except ValueError as exc:
        # Covers the embed-model mismatch raised by RAGTrustPipeline.load().
        raise CLIError(str(exc)) from exc
    return pipeline


# ------------------------------------------------------------------------- formatting


def _print_human_result(result_dict: dict) -> None:
    print(f"Answer: {result_dict['answer']}")
    print()

    if result_dict["abstained"]:
        print(f"ABSTAINED: {result_dict['abstain_reason']}")
        print()

    if result_dict["metrics"]:
        print("Metrics:")
        width = max(len(k) for k in result_dict["metrics"])
        for key, value in result_dict["metrics"].items():
            # A metric may legitimately be None -- undefined, not zero.
            # `conciseness` is None below two claims (pairwise self-similarity
            # needs a pair) and `answer_relevance` is None whenever its opt-in
            # flag is off or the backend cannot back-generate questions. Both are
            # dropped from aggregation rather than scored as 0. Formatting either
            # with `:.3f` raises TypeError -- which crashed `ragtrust ask`
            # outright on the default configuration. Print the undefined marker
            # rather than inventing a number.
            rendered = "n/a" if value is None else f"{value:.3f}"
            print(f"  {key.ljust(width)}  {rendered}")
        print()

    trust = result_dict["trust"]
    print(f"Trust (arithmetic): {trust.get('arithmetic', 0.0):.3f}")
    print(f"Trust (geometric):  {trust.get('geometric', 0.0):.3f}")
    print(f"Is trustworthy:     {result_dict['is_trustworthy']}")


# -------------------------------------------------------------------------- commands


def _cmd_index(args: argparse.Namespace) -> int:
    """Build and persist an index from a corpus file or directory."""
    cfg = _build_config(args)
    pipeline = RAGTrustPipeline(cfg)
    corpus = Path(args.corpus)
    if not corpus.exists():
        raise CLIError(f"Corpus not found: {args.corpus}")

    try:
        if corpus.is_dir():
            pipeline.index_dir(str(corpus))
        else:
            pipeline.index_corpus(str(corpus))
    except (FileNotFoundError, ValueError) as exc:
        # FileNotFoundError: index_dir() found no supported files.
        # ValueError: _install() refuses to build an empty index.
        raise CLIError(str(exc)) from exc

    pipeline.save(args.out)

    n = len(pipeline.passages_text)
    mean_words = (sum(len(t.split()) for t in pipeline.passages_text) / n) if n else 0.0
    print(f"Indexed {n} passages (mean {mean_words:.1f} words/passage).")
    print(f"Wrote index to {args.out}")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    """Load an index, answer one question, and print the result."""
    generator = _build_generator(args)
    pipeline = _load_pipeline(args, generator=generator)

    try:
        result = pipeline.answer(args.question)
    except GenerationError as exc:
        raise CLIError(f"Generation backend unreachable: {exc}") from exc

    result_dict = result.to_dict()
    if args.json:
        print(json.dumps(result_dict, indent=2))
    else:
        _print_human_result(result_dict)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    """Run the HTTP service (Deliverable 2) via uvicorn."""
    _require_index_dir(args.index)
    import uvicorn

    from .service import create_app

    generator = _build_generator(args)
    cfg = _build_config(args)
    try:
        app = create_app(args.index, config=cfg, generator=generator)
    except ValueError as exc:
        raise CLIError(str(exc)) from exc

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    """Answer a batch of questions and report aggregate trustworthiness stats."""
    questions_path = Path(args.questions)
    if not questions_path.exists():
        raise CLIError(f"Questions file not found: {args.questions}")

    try:
        raw = json.loads(questions_path.read_text())
    except json.JSONDecodeError as exc:
        raise CLIError(f"Could not parse {args.questions} as JSON: {exc}") from exc

    if not isinstance(raw, list) or not raw:
        raise CLIError(
            f"{args.questions} must be a non-empty JSON array of question strings "
            f'or objects like {{"question": "..."}}'
        )

    questions = []
    for item in raw:
        if isinstance(item, str):
            questions.append(item)
        elif isinstance(item, dict) and "question" in item:
            questions.append(item["question"])
        else:
            raise CLIError(f"Unrecognised question entry: {item!r}")

    generator = _build_generator(args)
    pipeline = _load_pipeline(args, generator=generator)

    per_question = []
    errors = []
    for q in questions:
        try:
            result_dict = pipeline.answer(q).to_dict()
        except GenerationError as exc:
            errors.append({"question": q, "error": str(exc)})
            continue
        per_question.append({"question": q, **result_dict})

    answered = [r for r in per_question if not r["abstained"]]
    abstained = [r for r in per_question if r["abstained"]]

    def _stats(values: list) -> dict:
        # A metric (e.g. conciseness -- undefined for < 2 claims, see
        # metrics/conciseness.py) may be None for some or all answered items;
        # None means "not computed", not zero, so it is dropped here rather
        # than fed into statistics.mean/median, same as aggregation drops it.
        defined = [v for v in values if v is not None]
        if not defined:
            return {"mean": None, "median": None}
        return {"mean": statistics.mean(defined), "median": statistics.median(defined)}

    metric_names = sorted({k for r in answered for k in r["metrics"]})
    metric_stats = {name: _stats([r["metrics"][name] for r in answered]) for name in metric_names}
    trust_stats = {
        "arithmetic": _stats([r["trust"]["arithmetic"] for r in answered]),
        "geometric": _stats([r["trust"]["geometric"] for r in answered]),
    }

    summary = {
        "total": len(questions),
        "answered": len(answered),
        "abstained": len(abstained),
        "errored": len(errors),
        "metrics": metric_stats,
        "trust": trust_stats,
    }

    print(f"Questions: {summary['total']}  "
          f"Answered: {summary['answered']}  "
          f"Abstained: {summary['abstained']}  "
          f"Errored: {summary['errored']}")
    if answered:
        print()
        print("Metric means (answered only):")
        width = max(len(k) for k in metric_stats)
        for name, stat in metric_stats.items():
            mean_str = f"{stat['mean']:.3f}" if stat["mean"] is not None else "n/a"
            median_str = f"{stat['median']:.3f}" if stat["median"] is not None else "n/a"
            print(f"  {name.ljust(width)}  mean={mean_str}  median={median_str}")
        print()
        print(f"Trust arithmetic: mean={trust_stats['arithmetic']['mean']:.3f} "
              f"median={trust_stats['arithmetic']['median']:.3f}")
        print(f"Trust geometric:  mean={trust_stats['geometric']['mean']:.3f} "
              f"median={trust_stats['geometric']['median']:.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"summary": summary, "results": per_question, "errors": errors}, indent=2))
        print(f"\nWrote full results to {args.out}")

    return 0


# ---------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragtrust", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Build and persist an index from a corpus")
    p_index.add_argument("corpus", help="Path to a .pdf/.md/.txt file, or a directory of them")
    p_index.add_argument("--out", required=True, help="Directory to write the index to")
    _add_config_args(p_index)
    p_index.set_defaults(func=_cmd_index)

    p_ask = sub.add_parser("ask", help="Answer one question against a saved index")
    p_ask.add_argument("question")
    p_ask.add_argument("--index", required=True, help="Directory of a saved index")
    p_ask.add_argument("--json", action="store_true", help="Print res.to_dict() as JSON only")
    _add_generator_args(p_ask)
    _add_config_args(p_ask)
    p_ask.set_defaults(func=_cmd_ask)

    p_serve = sub.add_parser("serve", help="Run the HTTP service")
    p_serve.add_argument("--index", required=True, help="Directory of a saved index")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    _add_generator_args(p_serve)
    _add_config_args(p_serve)
    p_serve.set_defaults(func=_cmd_serve)

    p_eval = sub.add_parser("eval", help="Batch-answer questions and report trust statistics")
    p_eval.add_argument("questions", help="JSON array of question strings, or [{\"question\": ...}]")
    p_eval.add_argument("--index", required=True, help="Directory of a saved index")
    p_eval.add_argument("--out", default=None, help="Optional path to write full per-question results")
    _add_generator_args(p_eval)
    _add_config_args(p_eval)
    p_eval.set_defaults(func=_cmd_eval)

    return parser


def main(argv: list = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CLIError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except GenerationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
