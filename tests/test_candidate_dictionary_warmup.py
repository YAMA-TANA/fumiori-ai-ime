from __future__ import annotations

import numpy as np

from ranker.onnx_dual_encoder_base import (
    OnnxDualEncoderIMEReranker,
    _model_fingerprint,
)
from ranker.persistent_embedding_store import PersistentEmbeddingStore


def test_warmup_persists_only_missing_vectors_without_filling_memory(tmp_path, monkeypatch):
    monkeypatch.setattr("ranker.onnx_dual_encoder_base.time.sleep", lambda _: None)
    store = PersistentEmbeddingStore(tmp_path / "candidates.sqlite3", "model-v1")
    store.put_many(["既存"], np.ones((1, 384), dtype=np.float32))

    class FakeRanker:
        candidate_store = store
        candidate_warmup_limit = 2
        candidate_cache = {}
        encoded = []

        def _iter_candidate_vocabulary(self):
            yield from ("既存", "新規", "上限外")

        def _wait_for_live_prefetch(self):
            pass

        def encode_texts(self, words, max_length=16):
            self.encoded.extend(words)
            return np.full((len(words), 384), 2.0, dtype=np.float32)

    ranker = FakeRanker()
    OnnxDualEncoderIMEReranker._warm_candidate_dictionary(ranker)

    assert ranker.encoded == ["新規"]
    assert ranker.candidate_cache == {}
    assert store.count() == 2
    assert store.missing_words(["既存", "新規", "上限外"]) == ["上限外"]


def test_model_fingerprint_survives_repackaging(tmp_path):
    source = tmp_path / "source.onnx"
    packaged = tmp_path / "packaged.onnx"
    source.write_bytes(b"same model weights")
    packaged.write_bytes(source.read_bytes())
    assert _model_fingerprint(source) == _model_fingerprint(packaged)
    packaged.write_bytes(b"different model weights")
    assert _model_fingerprint(source) != _model_fingerprint(packaged)
