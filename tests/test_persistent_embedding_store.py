from __future__ import annotations

import numpy as np

from ranker.persistent_embedding_store import PersistentEmbeddingStore


def test_candidate_vectors_round_trip_as_float16(tmp_path):
    path = tmp_path / "candidate_embeddings.sqlite3"
    original = np.arange(768, dtype=np.float32).reshape(2, 384) / 100.0
    store = PersistentEmbeddingStore(path, "model-v1")
    store.put_many(["測る", "図る"], original)
    assert store.count() == 2

    restored = store.get_many(["図る", "測る", "ない"])
    assert set(restored) == {"測る", "図る"}
    assert restored["測る"].dtype == np.float32
    np.testing.assert_allclose(restored["測る"], original[0], rtol=2e-3, atol=2e-3)

    # Exercise the chunked lookup used by a large precomputed dictionary.
    bulk_words = [f"候補{i}" for i in range(401)]
    bulk_vectors = np.zeros((len(bulk_words), 384), dtype=np.float32)
    store.put_many(bulk_words, bulk_vectors)
    assert len(store.get_many(bulk_words)) == len(bulk_words)


def test_model_generation_does_not_reuse_old_vectors(tmp_path):
    path = tmp_path / "candidate_embeddings.sqlite3"
    store = PersistentEmbeddingStore(path, "model-v1")
    store.put_many(["測る"], np.ones((1, 384), dtype=np.float32))
    assert PersistentEmbeddingStore(path, "model-v2").get_many(["測る"]) == {}
