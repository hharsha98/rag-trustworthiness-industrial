"""Natural language inference scoring.

`NLIScorer` wraps an HF sequence-classification NLI checkpoint (default
`roberta-large-mnli`) and always returns a probability dict keyed by the
canonical names {"entailment", "neutral", "contradiction"}, mapped
case-insensitively from the model's own `config.id2label` -- never by a
hardcoded index order, since different MNLI checkpoints permute the label
indices.

`FakeNLI` is a deterministic, dependency-free stub for unit tests: no model
download, same interface, derives an entailment probability from token
overlap (Jaccard) between premise and hypothesis.
"""
import re

import torch

_CANONICAL_LABELS = {"entailment", "neutral", "contradiction"}


class NLIScorer:
    def __init__(self, model_name: str = "roberta-large-mnli"):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._label_index = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()

        label_index = {}
        for idx, name in self._model.config.id2label.items():
            key = str(name).strip().lower()
            if key not in _CANONICAL_LABELS:
                raise ValueError(
                    f"Unexpected NLI label {name!r} (index {idx}) in model "
                    f"{self.model_name!r}; expected one of {_CANONICAL_LABELS}"
                )
            label_index[key] = int(idx)
        if set(label_index) != _CANONICAL_LABELS:
            raise ValueError(
                f"Model {self.model_name!r} does not expose all three NLI "
                f"labels; found {set(label_index)}"
            )
        self._label_index = label_index

    def probs(self, premise: str, hypothesis: str) -> dict:
        return self.batch_probs([(premise, hypothesis)])[0]

    def batch_probs(self, pairs: list) -> list:
        if not pairs:
            return []
        self._ensure_loaded()
        premises = [p for p, _ in pairs]
        hypotheses = [h for _, h in pairs]
        with torch.no_grad():
            inputs = self._tokenizer(
                premises, hypotheses, return_tensors="pt", padding=True, truncation=True
            )
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        out = []
        for row in probs:
            out.append(
                {
                    "entailment": float(row[self._label_index["entailment"]]),
                    "neutral": float(row[self._label_index["neutral"]]),
                    "contradiction": float(row[self._label_index["contradiction"]]),
                }
            )
        return out


class FakeNLI:
    """Deterministic stub matching the NLIScorer interface, for tests that
    must not download real model weights. Entailment is the Jaccard overlap
    of premise/hypothesis tokens; the remaining probability mass is split
    evenly between neutral and contradiction so the three always sum to 1."""

    def probs(self, premise: str, hypothesis: str) -> dict:
        return self.batch_probs([(premise, hypothesis)])[0]

    def batch_probs(self, pairs: list) -> list:
        return [self._score(p, h) for p, h in pairs]

    @staticmethod
    def _score(premise: str, hypothesis: str) -> dict:
        p_tokens = set(re.findall(r"\w+", (premise or "").lower()))
        h_tokens = set(re.findall(r"\w+", (hypothesis or "").lower()))
        union = p_tokens | h_tokens
        entailment = (len(p_tokens & h_tokens) / len(union)) if union else 0.0
        remainder = 1.0 - entailment
        return {
            "entailment": entailment,
            "neutral": remainder / 2.0,
            "contradiction": remainder / 2.0,
        }
