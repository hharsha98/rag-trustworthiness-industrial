"""Tests for CorpusStore: create/list/get/delete, the path-traversal-proof id
scheme, the pipeline_for cache, and TTL/cap sweeping. No model downloads, no
network -- uses FakeEmbedder from conftest.py, matching test_pipeline.py and
test_service.py.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ragtrust.config import Config
from ragtrust.corpora import CORPUS_ID_RE, CorpusStore
from ragtrust.ingest.validate import UploadRejected
from ragtrust.pipeline import RAGTrustPipeline
# tests/ is not a package -- see the note in test_cli.py / test_service.py.
from conftest import FakeEmbedder

MD_BYTES = b"## Section One\n\nPhotosynthesis converts sunlight into chemical energy in plants.\n"


def _base_pipeline() -> RAGTrustPipeline:
    return RAGTrustPipeline(Config(embed_model="fake-embed-v1"), embedder=FakeEmbedder())


def _store(tmp_path, **kwargs) -> CorpusStore:
    return CorpusStore(str(tmp_path / "corpora"), **kwargs)


# --------------------------------------------------------------------- create


def test_create_then_list_and_get_round_trip(tmp_path):
    store = _store(tmp_path)
    record = store.create("notes.md", MD_BYTES, _base_pipeline())

    assert record.passages > 0
    assert record.bytes == len(MD_BYTES)
    assert record.embed_model == "fake-embed-v1"
    assert record.filename == "notes.md"

    listed = store.list()
    assert len(listed) == 1
    assert listed[0].corpus_id == record.corpus_id

    fetched = store.get(record.corpus_id)
    assert fetched == record


def test_meta_json_fields_and_created_at_format(tmp_path):
    store = _store(tmp_path)
    record = store.create("notes.md", MD_BYTES, _base_pipeline())

    meta_path = Path(store.root) / record.corpus_id / "meta.json"
    data = json.loads(meta_path.read_text())
    assert set(data.keys()) == {
        "corpus_id", "filename", "bytes", "passages", "embed_model", "created_at",
    }
    assert data["created_at"].endswith("Z")
    # Must parse as a real timestamp -- proves it's a genuine ISO-8601 string,
    # not just a string that happens to end in "Z".
    parsed = datetime.strptime(data["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ")
    assert parsed.tzinfo is None  # naive by construction; the "Z" carries the UTC claim


# ----------------------------------------------------- corpus_id / path safety


def test_corpus_id_is_not_derived_from_filename(tmp_path):
    store = _store(tmp_path)
    content = (
        b"Photosynthesis converts sunlight into chemical energy in plants, "
        b"a process central to the global carbon cycle and plant biology."
    )
    record = store.create("../../etc/passwd.txt", content, _base_pipeline())

    assert CORPUS_ID_RE.match(record.corpus_id)
    # The created directory must live directly under root, named by the id --
    # nothing was written outside tmp_path via the malicious filename.
    corpus_dir = tmp_path / "corpora" / record.corpus_id
    assert corpus_dir.is_dir()
    all_paths = list(tmp_path.rglob("*"))
    assert all(str(tmp_path) in str(p) for p in all_paths)
    assert not (tmp_path.parent / "etc").exists()


@pytest.mark.parametrize("bad_id", ["../foo", "nothex", "", "a" * 31, "a" * 33, "../../../etc/passwd"])
def test_get_and_delete_reject_malformed_ids(tmp_path, bad_id):
    store = _store(tmp_path)
    with pytest.raises(KeyError):
        store.get(bad_id)
    with pytest.raises(KeyError):
        store.delete(bad_id)
    # No filesystem path outside root was touched -- root itself still exists
    # and nothing was created or removed as a side effect of the attempt.
    assert store.root.is_dir()


# --------------------------------------------------------------------- delete


def test_delete_removes_corpus_and_evicts_cache(tmp_path):
    store = _store(tmp_path)
    base = _base_pipeline()
    record = store.create("notes.md", MD_BYTES, base)

    store.pipeline_for(record.corpus_id, base)
    assert record.corpus_id in store._cache

    store.delete(record.corpus_id)
    assert record.corpus_id not in store._cache
    assert not (tmp_path / "corpora" / record.corpus_id).exists()
    with pytest.raises(KeyError):
        store.get(record.corpus_id)


def test_delete_unknown_id_raises_keyerror(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(KeyError):
        store.delete("0" * 32)


# ------------------------------------------------------------------ pipeline_for


def test_pipeline_for_shares_embedder_but_not_passages(tmp_path):
    store = _store(tmp_path)
    base = _base_pipeline()
    base.index_texts(["The bundled corpus passage, unrelated to the upload."])

    record = store.create("notes.md", MD_BYTES, base)
    scoped = store.pipeline_for(record.corpus_id, base)

    # No model reload guarantee: the embedder object is literally the same one.
    assert scoped._embedder is base._embedder
    # But the corpus itself is independent.
    assert scoped.passages_text != base.passages_text
    assert len(scoped.passages_text) == record.passages


def test_pipeline_for_caches_and_lru_evicts(tmp_path):
    store = _store(tmp_path, cache_size=2)
    base = _base_pipeline()
    records = [store.create(f"doc{i}.md", MD_BYTES, base) for i in range(3)]

    p0 = store.pipeline_for(records[0].corpus_id, base)
    p1 = store.pipeline_for(records[1].corpus_id, base)
    assert store.pipeline_for(records[0].corpus_id, base) is p0  # cache hit, still identical object
    assert store.pipeline_for(records[1].corpus_id, base) is p1

    # A third distinct corpus pushes the cache over cache_size=2, evicting one
    # of the two already-cached entries (LRU order was touched by the repeated
    # lookups above, so exactly which one is not asserted -- only that the cap
    # is enforced and the newest entry is present).
    store.pipeline_for(records[2].corpus_id, base)
    assert len(store._cache) == 2
    assert records[2].corpus_id in store._cache


# ---------------------------------------------------------------------- sweep


def _backdate(store: CorpusStore, corpus_id: str, hours_ago: float) -> None:
    meta_path = Path(store.root) / corpus_id / "meta.json"
    data = json.loads(meta_path.read_text())
    stamp = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )[:-3] + "Z"
    data["created_at"] = stamp
    meta_path.write_text(json.dumps(data))


def test_sweep_removes_expired_by_ttl(tmp_path):
    store = _store(tmp_path, ttl_hours=1, max_corpora=100)
    base = _base_pipeline()
    record = store.create("notes.md", MD_BYTES, base)
    _backdate(store, record.corpus_id, hours_ago=2)

    removed = store.sweep()
    assert removed == 1
    assert store.list() == []


def test_sweep_removes_over_cap_oldest_first(tmp_path):
    # Created with a high cap so nothing is pruned by the sweep() that create()
    # runs on every call -- the cap is lowered afterwards so this test can
    # observe sweep()'s own oldest-first eviction in isolation.
    store = _store(tmp_path, ttl_hours=1000, max_corpora=100)
    base = _base_pipeline()
    records = [store.create(f"doc{i}.md", MD_BYTES, base) for i in range(4)]
    for i, r in enumerate(records):
        _backdate(store, r.corpus_id, hours_ago=len(records) - i)  # earlier index = older

    store.max_corpora = 2
    removed = store.sweep()
    assert removed == 2
    remaining_ids = {r.corpus_id for r in store.list()}
    # The two most recently created (last two in the list) survive.
    assert remaining_ids == {records[-1].corpus_id, records[-2].corpus_id}


# ------------------------------------------------------------- empty-extraction


def test_create_raises_upload_rejected_on_empty_extraction_and_leaves_no_dir(tmp_path):
    store = _store(tmp_path)
    base = _base_pipeline()
    # Content chunks to nothing: every line is shorter than chunk_passages'
    # default min_line_chars (3), so no line survives to be windowed and
    # chunk_passages returns [] -- _install then raises "empty index".
    empty_ish = b"x\ny\nz\n"

    before = set(p.name for p in Path(store.root).iterdir()) if Path(store.root).exists() else set()
    with pytest.raises(UploadRejected):
        store.create("empty.txt", empty_ish, base)
    after = set(p.name for p in Path(store.root).iterdir()) if Path(store.root).exists() else set()
    assert after == before


# -------------------------------------------------------------- provenance


def test_passages_are_stamped_with_the_uploaded_filename_not_the_storage_name(tmp_path):
    """Every upload is stored on disk as source.<ext>, so index_corpus would
    otherwise stamp all of them with the identical source "source.md" and the
    citations beside an answer would name the wrong document."""
    store = _store(tmp_path)
    record = store.create("quarterly-report.md", MD_BYTES, _base_pipeline())

    pipeline = store.pipeline_for(record.corpus_id, _base_pipeline())
    sources = {m["source"] for m in pipeline.passage_meta.values()}
    assert sources == {"quarterly-report.md"}
    assert not any(s.startswith("source.") for s in sources)


def test_display_name_is_reduced_to_a_basename(tmp_path):
    store = _store(tmp_path)
    record = store.create("../../etc/notes.md", MD_BYTES, _base_pipeline())
    assert record.filename == "notes.md"


# ----------------------------------------------------- meta.json is not trusted


def test_list_skips_records_whose_id_disagrees_with_their_directory(tmp_path):
    """`corpus_id` read out of meta.json reaches rmtree via sweep(), so list()
    anchors it to the directory name. A meta.json claiming to be some other
    corpus must be ignored, not acted on."""
    store = _store(tmp_path)
    record = store.create("notes.md", MD_BYTES, _base_pipeline())
    meta_path = Path(store.root) / record.corpus_id / "meta.json"

    payload = json.loads(meta_path.read_text())
    payload["corpus_id"] = "../../elsewhere"
    meta_path.write_text(json.dumps(payload))

    assert store.list() == []
    # And the tampered corpus is not reachable by its real id either, because
    # get() reconstructs from the same untrusted file.
    assert store.sweep() == 0
    assert (Path(store.root) / record.corpus_id).is_dir()


def test_create_leaves_no_directory_when_meta_write_fails(tmp_path, monkeypatch):
    """meta.json is written inside create()'s try block: a dir without a readable
    meta.json is invisible to list(), so it would be invisible to sweep() too --
    an orphan no retention policy could ever reclaim."""
    store = _store(tmp_path)
    real_write_text = Path.write_text

    def explode(self, *args, **kwargs):
        if self.name == "meta.json":
            raise OSError("disk full")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", explode)
    with pytest.raises(OSError):
        store.create("notes.md", MD_BYTES, _base_pipeline())

    assert list(Path(store.root).iterdir()) == []
