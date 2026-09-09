---
title: RAG Trustworthiness Industrial
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
---

# RAG Trustworthiness -- Industrial

A live demo of a retrieval-augmented generation system for local knowledge retrieval
in industrial environments, with trustworthiness metrics -- faithfulness, attribution,
relevance, and conciseness -- computed and displayed alongside every answer.

Tabs:
- **Ask** -- ask a question against the indexed robotics corpus and see the generated
  answer, retrieved passages, and the trust-metrics scorecard.
- **About** -- project background and links.

This Space runs on CPU. The generator backend defaults to cached/recorded answers so the
demo works without any API keys; set `HF_TOKEN` to enable live generation via the
HuggingFace Inference API. See the repository README for the full write-up and the
`docker compose` stack that runs the complete self-hosted (Ollama) version of this system.
