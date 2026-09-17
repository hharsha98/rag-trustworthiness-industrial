# dashboard/

One self-contained `index.html` — inline CSS and JS, no build step, no framework, no bundler.
`src/ragtrust/service.py` mounts this directory as `StaticFiles(html=True)` at `/`, so anything
dropped in here is served with no backend change. That is also why `demo-run.json` is reachable
at `/demo-run.json`: the page links to it so a sceptical visitor can diff the raw file against
what is on screen.

## The recorded runs

`demo-run.json` and `demo-abstain.json` are verbatim `POST /answer` responses, captured from the
live deployment and replayed on first arrival.

They exist because a real answer on this hardware takes about a minute — 56s for the recorded
one, up to 99s observed — and because `/answer` is rate limited to 200 calls/hour across all
clients. Firing a live call on page load would mean 200 page views exhausting the budget, so the
hiring manager who actually types a question gets a 429. The replay costs nothing and starts
immediately; the visitor's own question then runs live.

### Four rules

1. **Never edit a number.** The file is a transcript. If a captured run has an unflattering
   metric, that is the run. `tests/test_demo_fixture.py` checks shape, never values, precisely so
   there is no pressure to re-record until the numbers look good.
2. **Never capture against an uploaded corpus.** A response embeds `passages[].text` verbatim.
   The evaluation corpus used elsewhere in this project is third-party lecture material that
   `.gitignore` excludes because this repository has no redistribution rights to it. A fixture
   captured against it would publish the exact text that exclusion exists to prevent — and
   deleting it later does not un-publish it. Capture with `corpus_id` **omitted**, so the run goes
   against `data/demo_corpus.md`, which is original text written for this repo. There is a test
   for this; it is the highest-severity one in the file.
3. **Keep both outcomes.** The abstention is the other half of the product, and it is the only
   place the timeline's `not run` state is real — the gate returns before generate, decompose,
   entail and score ever happen.
4. **Label it as a recording.** The page says so persistently and links the raw JSON. That claim
   is only worth something while the provenance fields below stay accurate.

### Re-capturing

Against the live deployment (or a local `ragtrust serve`), with `corpus_id` omitted:

```bash
curl -s -X POST https://<host>/answer \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the role of self-supervised learning in robotics?"}' \
  -o /tmp/cap-run.json

curl -s -X POST https://<host>/answer \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the capital of France?"}' \
  -o /tmp/cap-abstain.json
```

Capture against the **deployed** service, not a local one using the cached generator: a cached
run records `generate` at a few milliseconds, which would badly understate what generation
actually costs and flatter the verification-vs-generation comparison the page draws.

Then wrap each response, adding only these keys and changing nothing inside `response`:

```json
{
  "_what": "...",
  "recorded_utc": "2026-01-01T00:00:00Z",
  "recorded_against": "demo_corpus.md",
  "retrieval_mode": "hybrid",
  "source_commit": "abc1234",
  "total_ms": 56470.0,
  "response": {}
}
```

`retrieval_mode` lives in the wrapper on purpose. The page uses *that* value to label passage
scores when replaying, not the live `/config` — otherwise changing the deployment's retrieval
mode would silently relabel a recorded RRF score as a cosine similarity, which is the one thing
this page must never do (they are not the same quantity on the same scale; RRF sits near 0.03 at
rank 1).

Finally, `pytest tests/test_demo_fixture.py`.
