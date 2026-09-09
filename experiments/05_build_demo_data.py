#!/usr/bin/env python3
"""Precompute everything the static demo page needs, into docs/demo_data.json.

The hosted demo is a static page (GitHub Pages), so it cannot run models. This script
does the model work once, offline, and records real output: for each cached question
in `data/cached_answers.json`, the retrieved passages, the generated answer, and this
system's own metrics.

Nothing here is illustrative or hand-written -- every number is produced by the same code
paths the library and tests use.

Usage:  python experiments/05_build_demo_data.py [--nli-model MODEL]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch + faiss on macOS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragtrust.generation.ollama import parse_citations  # noqa: E402
from ragtrust.metrics.attribution import attribution  # noqa: E402
from ragtrust.metrics.claims import split_claims  # noqa: E402
from ragtrust.metrics.conciseness import conciseness  # noqa: E402
from ragtrust.metrics.faithfulness import faithfulness  # noqa: E402
from ragtrust.metrics.nli import NLIScorer  # noqa: E402
from ragtrust.metrics.relevance import context_relevance  # noqa: E402

DEFAULT_NLI = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
OUT = ROOT / "docs" / "demo_data.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli-model", default=DEFAULT_NLI)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    answers = json.loads((ROOT / "data" / "cached_answers.json").read_text())

    print(f"loading models ({args.nli_model}) ...")
    nli = NLIScorer(model_name=args.nli_model)
    embedder = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v4")

    rows = []
    for idx, (q, entry) in enumerate(answers.items()):
        passages = [p["text"] for p in entry["passages"]]
        answer = entry["answer"]
        abstained = bool(entry.get("abstained", False))

        row = {
            "idx": idx,
            "question": q,
            "category": entry.get("category"),
            "abstained": abstained,
            "answer": answer,
            "top_score": entry.get("top_score"),
            "passages": [
                {"rank": p["id"], "page": p.get("page"), "score": round(p["score"], 4),
                 "text": p["text"]}
                for p in entry["passages"]
            ],
        }

        if not abstained:
            claims = split_claims(answer)
            citations = parse_citations(answer)
            f = faithfulness(claims, passages, nli)
            a = attribution(claims, citations, passages, nli, tau=0.5)
            conciseness_score = conciseness(claims, embedder)
            row["claims"] = claims
            row["citations"] = {str(k): v for k, v in citations.items()}
            row["metrics"] = {
                "faithfulness": round(f.score, 4),
                "contradiction_rate": round(f.contradiction_rate, 4),
                "per_claim": [round(x, 4) for x in f.per_claim],
                "support_index": f.support_index,
                "attribution_f1": round(a.f1, 4),
                "attribution_precision": round(a.precision, 4),
                "attribution_recall": round(a.recall, 4),
                "relevance": round(context_relevance(q, passages, embedder, scaled=True), 4),
                # conciseness is undefined (None) for < 2 claims -- see
                # metrics/conciseness.py -- so pass it through as None (JSON
                # null) rather than rounding.
                "conciseness": conciseness_score if conciseness_score is None else round(conciseness_score, 4),
            }
        rows.append(row)
        print(f"  [{idx:>2}] {str(entry.get('category')):<8} "
              f"{'ABSTAINED' if abstained else 'scored':<10} {q[:46]}")

    results = ROOT / "experiments" / "results"
    weight_sensitivity_path = results / "weight_sensitivity.json"
    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "nli_model": args.nli_model,
        "questions": rows,
    }
    if weight_sensitivity_path.exists():
        payload["weight_sensitivity"] = json.loads(weight_sensitivity_path.read_text())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))
    scored = sum(1 for r in rows if not r["abstained"])
    print(f"\nwrote {OUT.relative_to(ROOT)}: {len(rows)} questions "
          f"({scored} scored, {len(rows)-scored} abstained), {OUT.stat().st_size//1024} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
